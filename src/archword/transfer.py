"""Оценка бейзлайнов на собственной разметке: перенос с публичных данных на прозу Пушкина.

    python -m archword.transfer --raw data/raw --labeled data/annotation/round1_labeled.csv \
        --out results/transfer_round1.json
    python -m archword.transfer --raw data/raw --labeled data/annotation/round1_labeled.csv \
        --labeled data/annotation/round2_labeled.csv --out results/transfer_rounds.json

Модели обучаются на публичной выборке (Библия, оценка 0..1), а проверяются на словах
«Повестей Белкина» с оценкой читателя 1..5. Шкалы разные, поэтому сравниваются
только ранги (Спирмен) и способность отделить «трудные» слова (AUC для порогов 2 и 3).
Доверительные интервалы получены бутстрэпом по словам.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .baseline import build_frame, make_model
from .data import load_complexity, load_freq, log_freq

THRESHOLDS = (2, 3)
# строки с такой пометкой в колонке note в оценку не входят: имена и опечатки/артефакты текста
SKIP_NOTES = {"имя", "ошибка", "опечатка"}


def load_labeled(path: str | Path) -> pd.DataFrame:
    """Читает размеченную таблицу (CSV или XLSX), проверяет шкалу, убирает имена."""
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path, encoding="utf-8-sig")
    missing = {"word", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"В таблице нет колонок: {sorted(missing)}")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    bad = df[~df["label"].isin([1, 2, 3, 4, 5])]
    if len(bad):
        ids = bad["id"].tolist() if "id" in bad else bad.index.tolist()
        raise ValueError(f"Оценка должна быть целым числом 1..5. Проверьте строки: {ids[:10]}")
    df["label"] = df["label"].astype(int)
    if "note" in df:
        note = df["note"].astype(str).str.strip().str.lower()
        df = df[~note.isin(SKIP_NOTES)]
    return df.reset_index(drop=True)


def _ci(values: list[float]) -> list[float]:
    return [round(float(x), 3) for x in np.percentile(values, [2.5, 97.5])] if values else []


def _bootstrap(y, p, fn, n_boot: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        try:
            v = fn(y[i], p[i])
        except ValueError:
            continue
        if np.isfinite(v):
            out.append(v)
    return out


def _spearman(y, p) -> float:
    if np.ptp(y) == 0 or np.ptp(p) == 0:
        raise ValueError("константа")
    return float(spearmanr(y, p)[0])


def score(y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    res = {"spearman": round(_spearman(y, p), 3)}
    res["spearman_ci95"] = _ci(_bootstrap(y, p, _spearman, n_boot, seed))
    for t in THRESHOLDS:
        pos = (y >= t).astype(int)
        key = f"auc_label_ge{t}"
        if 0 < pos.sum() < len(pos):
            res[key] = round(float(roc_auc_score(pos, p)), 3)
            res[key + "_ci95"] = _ci(_bootstrap(pos, p, lambda a, b: roc_auc_score(a, b), n_boot, seed))
            res[f"positives_ge{t}"] = int(pos.sum())
        else:
            res[key] = None
    return res


def evaluate(raw: str | Path, labeled: str | Path, n_boot: int = 2000, seed: int = 0) -> dict:
    """Оценка бейзлайнов на одной размеченной таблице."""
    return _evaluate_sheet(load_freq(raw), load_complexity(raw, "train"), load_labeled(labeled), n_boot, seed)


def evaluate_many(raw: str | Path, labeled: list[str | Path], n_boot: int = 2000, seed: int = 0) -> dict:
    """Оценка по каждой таблице отдельно и по их объединению.

    Таблицы собраны по-разному (раунд 1: случайная выборка по группам частоты, раунд 2: только редкие слова),
    поэтому главный результат даёт каждая таблица отдельно, а объединение нужно для сравнения с общим числом слов.
    """
    freq, train = load_freq(raw), load_complexity(raw, "train")
    sheets = {Path(p).stem: load_labeled(p) for p in labeled}
    report = {name: _evaluate_sheet(freq, train, df, n_boot, seed) for name, df in sheets.items()}
    if len(sheets) > 1:
        both = pd.concat(sheets.values(), ignore_index=True)
        report["combined"] = _evaluate_sheet(freq, train, both, n_boot, seed)
    return report


def _evaluate_sheet(freq, train, sheet: pd.DataFrame, n_boot: int, seed: int) -> dict:
    y = sheet["label"].to_numpy()
    report = {
        "n_words": int(len(sheet)),
        "label_counts": {int(k): int(v) for k, v in sheet["label"].value_counts().sort_index().items()},
        "models": {},
    }
    preds = {"only_-log_freq": -log_freq(sheet["word"], freq)}
    frame = build_frame(sheet.rename(columns={"word": "token"}), freq)
    for kind in ("freq_len", "freq_len_charngram"):
        model = make_model(kind).fit(build_frame(train, freq), train["complexity"].to_numpy())
        preds[kind] = model.predict(frame)
    for name, p in preds.items():
        report["models"][name] = score(y, np.asarray(p, float), n_boot, seed)
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--labeled", action="append", default=None,
                    help="размеченная таблица (CSV или XLSX); можно повторять, тогда будет и результат по объединению")
    ap.add_argument("--out", default="results/transfer_round1.json")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    labeled = args.labeled or ["data/annotation/round1_labeled.csv"]
    report = evaluate(args.raw, labeled[0], n_boot=args.boot) if len(labeled) == 1 \
        else evaluate_many(args.raw, labeled, n_boot=args.boot)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
