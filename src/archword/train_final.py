"""Обучение финальной модели для демонстрации и экспорт файлов в models/.

    pip install pymorphy3 pymorphy3-dicts-ru
    python -m archword.train_final --raw data/raw --out-dir models

Нужны исходные данные (частотный словарь в data/raw, см. DATA.md): они в git не попадают, а в models/ кладутся
компактные таблицы (форма -> частота, лемма -> частота) и model.json с коэффициентами, поэтому приложению
исходные 130 МБ не нужны. Модель - Ridge по частотам, длине и символьным n-граммам, как лучшая линейная модель в
кросс-валидации (`archword.classic_cv`), но обученная на всей разметке. Оценка качества в meta - кросс-валидация
по тем же 5 частям на тех же признаках, а предсказания для приложения даёт модель, обученная на всех словах.

Таблицы частот хранят значения не ниже MIN_IPM (0,02 на млн), остальное считается как слово без частоты. Признаки
для обучения считаются по тем же сжатым таблицам, что и для приложения, чтобы обучение и применение не расходились.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .classic_cv import _model, cross_predict
from .crossval import DEFAULT_ROUNDS, assign_folds, load_own
from .data import load_freq, load_lemma_freq
from .lemma import LemmaFreq, load_lemma_freq_morph, save_totals
from .predict import FORM_FILE, LEMMA_FILE, MIN_IPM, MODEL_FILE, LinearModel, numeric_matrix, to_prob
from .transfer import score


def build_form_table(raw: str | Path, min_ipm: float = MIN_IPM) -> pd.DataFrame:
    """Таблица «форма -> частота формы, частота леммы по OpenCorpora» без значений ниже min_ipm."""
    table = pd.DataFrame({"ipm": pd.Series(load_freq(raw)), "oc_lemma": pd.Series(load_lemma_freq(raw))})
    return compact_form_table(table, min_ipm)


def compact_form_table(table: pd.DataFrame, min_ipm: float = MIN_IPM) -> pd.DataFrame:
    table = table.where(table >= min_ipm)  # ниже порога -> NaN -> при чтении станет FLOOR_IPM
    table = table.dropna(how="all")
    table.index.name = "w"
    return table.round(4)


def export_pipeline(pipe) -> dict:
    """Коэффициенты обученного Pipeline (StandardScaler + TfidfVectorizer -> Ridge) в обычные списки."""
    ct, ridge = pipe.named_steps["features"], pipe.named_steps["ridge"]
    scaler, tfidf = ct.named_transformers_["num"], ct.named_transformers_["chars"]
    if tfidf.analyzer != "char_wb" or not tfidf.sublinear_tf or tfidf.norm != "l2" or not tfidf.lowercase:
        raise ValueError("Экспорт рассчитан на TfidfVectorizer(analyzer='char_wb', sublinear_tf=True, norm='l2')")
    vocab = [g for g, _ in sorted(tfidf.vocabulary_.items(), key=lambda kv: kv[1])]
    return {
        "num_mean": scaler.mean_.tolist(), "num_scale": scaler.scale_.tolist(),
        "vocab": vocab, "idf": tfidf.idf_.tolist(), "ngram_range": list(tfidf.ngram_range),
        "coef": ridge.coef_.ravel().tolist(), "intercept": float(np.ravel(ridge.intercept_)[0]),
    }


def fit_calibration(pred, labels, hard: int = 2) -> dict:
    """Логистическая кривая «предсказание модели -> вероятность, что оценка не ниже hard» по предсказаниям вне обучения."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1e4).fit(np.asarray(pred, float).reshape(-1, 1), (np.asarray(labels) >= hard).astype(int))
    return {"a": float(clf.coef_[0, 0]), "b": float(clf.intercept_[0])}


def suggest_threshold(labels, prob, hard: int = 2) -> dict:
    """Порог вероятности, при котором лучше всего находятся слова с оценкой не ниже `hard` (максимум F1), и его точность и полнота."""
    y = np.asarray(labels) >= hard
    prob = np.asarray(prob, float)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in np.arange(0.05, 0.91, 0.01):
        flag = prob + 1e-9 >= t  # допуск на погрешность float на границе
        tp = int((flag & y).sum())
        f1 = 2 * tp / max(1, int(flag.sum()) + int(y.sum()))
        if f1 > best["f1"]:
            best = {"threshold": float(round(t, 2)), "f1": round(f1, 3), "precision": round(tp / max(1, int(flag.sum())), 3),
                    "recall": round(tp / max(1, int(y.sum())), 3)}
    return best


def fit_artifacts(own: pd.DataFrame, forms: pd.DataFrame, lemma: LemmaFreq, k: int = 5, n_boot: int = 500) -> dict:
    """Кросс-валидация для оценки качества и финальная модель на всех словах. Возвращает содержимое model.json."""
    words = own["key"].tolist()
    X = pd.concat([pd.Series(words, name="word"), pd.DataFrame(numeric_matrix(words, forms, lemma))], axis=1)
    X.columns = ["word", "f1", "f2", "f3", "f4"]
    oof = cross_predict(own, assign_folds(own, k), X, "ngram")
    y = own["label"].to_numpy()
    cv = score(y, oof, n_boot=n_boot)
    final = _model("ngram").fit(X.to_numpy(dtype=object), own["y"].to_numpy())
    spec = export_pipeline(final.best_estimator_)
    # самопроверка экспорта: numpy-модель должна повторять sklearn на обучающих словах
    exported = LinearModel(spec).predict(words, X[["f1", "f2", "f3", "f4"]].to_numpy(float))
    expected = final.best_estimator_.predict(X.to_numpy(dtype=object))
    if not np.allclose(exported, expected, atol=1e-6):
        raise RuntimeError("Экспортированная модель расходится с sklearn: проверьте export_pipeline")
    calibration = fit_calibration(oof, y)
    choice = suggest_threshold(y, to_prob(oof, calibration))
    meta = {
        "n_train": int(len(own)), "rounds": sorted(own["round"].unique().tolist()), "min_ipm": MIN_IPM,
        "alpha": float(final.best_params_["ridge__alpha"]), "cv": {k_: cv[k_] for k_ in ("spearman", "spearman_ci95", "auc_label_ge2", "auc_label_ge3")},
        "calibration": calibration, "suggested_threshold": choice["threshold"],
        "threshold_stats": {k_: choice[k_] for k_ in ("precision", "recall", "f1")},
        "note": "Оценка предсказывает ответ одного читателя (автора разметки), шкала 1-5.",
    }
    return {"model": spec, "meta": meta}


def save(out_dir: str | Path, artifacts: dict, forms: pd.DataFrame, lemma_totals: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / MODEL_FILE).write_text(json.dumps(artifacts, ensure_ascii=False), encoding="utf-8")
    forms.reset_index().to_csv(out / FORM_FILE, index=False, encoding="utf-8")
    save_totals(lemma_totals, out / LEMMA_FILE)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--own", action="append", default=None)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args(argv)
    lemma = load_lemma_freq_morph(a.raw)
    forms = build_form_table(a.raw)
    own = load_own(a.own or DEFAULT_ROUNDS)
    artifacts = fit_artifacts(own, forms, lemma, a.k)
    save(a.out_dir, artifacts, forms, lemma.totals)
    meta = artifacts["meta"]
    print(f"слов в обучении: {meta['n_train']}; кросс-валидация: Spearman {meta['cv']['spearman']} "
          f"AUC>=2 {meta['cv']['auc_label_ge2']} AUC>=3 {meta['cv']['auc_label_ge3']}")
    st = meta["threshold_stats"]
    print(f"предлагаемый порог вероятности: {meta['suggested_threshold']} (на разметке: точность {st['precision']}, полнота {st['recall']}); "
          f"файлы записаны в {a.out_dir}/")


if __name__ == "__main__":
    main()
