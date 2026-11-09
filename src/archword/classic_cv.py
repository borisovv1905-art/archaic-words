"""Сильный классический бейзлайн в тех же условиях, что и нейросеть: кросс-валидация по словам на своей разметке.

Нейросеть (archword.crossval) учится на своих оценках. Честно сравнивать её с «голой» частотой нельзя: частота
своей разметки не видит. Поэтому здесь обычная линейная модель (Ridge) учится на тех же частях и тех же оценках, но
на признаках: логарифм частоты формы, частоты леммы (колонка OpenCorpora и, если передан --lemma-morph, pymorphy3),
длина слова и, во втором варианте, символьные n-граммы слова (окончания, приставки).

    python -m archword.classic_cv --raw data/raw --out results/classic_cv.json [--lemma-morph]

Результат - таблица предсказаний в том же формате, что у crossval: её можно сравнивать через archword.compare --other.
Параметр регуляризации выбирается внутренней кросс-валидацией на обучающей части каждого fold, тест не трогается.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .crossval import DEFAULT_ROUNDS, assign_folds, load_own
from .data import load_freq, load_lemma_freq, log_freq
from .transfer import score

ALPHAS = (0.1, 1.0, 10.0, 100.0)


def numeric_features(words, tables: dict) -> pd.DataFrame:
    """Логарифмы частот по каждой таблице и длина слова."""
    cols = {f"log_{name}": log_freq(words, t) for name, t in tables.items()}
    cols["len"] = np.array([len(str(w)) for w in words], float)
    return pd.DataFrame(cols)


def _model(kind: str):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    parts = [("num", StandardScaler(), slice(1, None))]  # колонка 0 - слово, остальные числовые
    if kind == "ngram":
        parts.append(("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True), 0))
    pipe = Pipeline([("features", ColumnTransformer(parts)), ("ridge", Ridge())])
    return GridSearchCV(pipe, {"ridge__alpha": list(ALPHAS)}, cv=3, scoring="neg_mean_squared_error")


def cross_predict(own: pd.DataFrame, folds: np.ndarray, X: pd.DataFrame, kind: str) -> np.ndarray:
    """Предсказания вне обучения: каждая часть предсказывается моделью, обученной на остальных."""
    oof = np.full(len(own), np.nan)
    y = own["y"].to_numpy()
    for f in sorted(set(folds.tolist())):
        test = folds == f
        model = _model(kind).fit(X[~test].to_numpy(dtype=object), y[~test])
        oof[test] = model.predict(X[test].to_numpy(dtype=object))
    return oof


def run(raw, own_paths=DEFAULT_ROUNDS, k: int = 5, lemma_morph=None, n_boot: int = 2000) -> tuple[dict, pd.DataFrame]:
    own = load_own(own_paths)
    folds = assign_folds(own, k)
    tables = {"form": load_freq(raw), "lemma": load_lemma_freq(raw)}
    if lemma_morph is not None:
        tables["lemma_pymorphy"] = lemma_morph
    num = numeric_features(own["word"], tables)
    X = pd.concat([own[["word"]].reset_index(drop=True), num], axis=1)
    table = own[["word", "context", "label", "round"]].copy()
    table["fold"] = folds
    report = {"n": int(len(own)), "n_folds": k, "features": list(num.columns), "models": {}}
    y = own["label"].to_numpy()
    for kind, name in (("numeric", "pred_ridge_numeric"), ("ngram", "pred_ridge_ngram")):
        table[name] = cross_predict(own, folds, X, kind)
        report["models"][name] = score(y, table[name].to_numpy(), n_boot=n_boot)
    return report, table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--own", action="append", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lemma-morph", action="store_true", help="добавить признак - частоту леммы от pymorphy3")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results/classic_cv.json")
    a = ap.parse_args(argv)
    lemma_morph = None
    if a.lemma_morph:
        from .lemma import load_lemma_freq_morph  # нужен pymorphy3
        lemma_morph = load_lemma_freq_morph(a.raw)
    report, table = run(a.raw, a.own or DEFAULT_ROUNDS, a.k, lemma_morph, a.boot)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    table.to_csv(out.with_suffix(".predictions.csv"), index=False, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
