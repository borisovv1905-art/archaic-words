"""Ансамбль из двух моделей: среднее рангов их предсказаний вне обучения (без подбора весов, поэтому без утечки).

    python -m archword.ensemble --a results/crossval_bible_then_own.predictions.csv --a-col pred_bible_then_own \\
        --b results/classic_cv.predictions.csv --b-col pred_ridge_ngram --out results/ensemble_cv.predictions.csv

Ранги нужны потому, что у моделей разные шкалы (нейросеть выдаёт 0..1, линейная модель - тоже 0..1, но распределения разные).
Таблицы должны содержать одни и те же слова в одном порядке (обе строятся от одного разбиения archword.crossval).
Результат сравнивается через archword.compare.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import rankdata

KEYS = ["word", "context", "label", "round", "fold"]


def rank_average(a: pd.DataFrame, a_col: str, b: pd.DataFrame, b_col: str) -> pd.DataFrame:
    if len(a) != len(b) or not a[KEYS].equals(b[KEYS]):
        raise ValueError("Таблицы должны содержать одни и те же слова в одном порядке (общее разбиение на части).")
    out = a[KEYS].copy()
    out[a_col], out[b_col] = a[a_col].to_numpy(), b[b_col].to_numpy()
    out["pred_ensemble"] = (rankdata(out[a_col]) + rankdata(out[b_col])) / 2
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True); ap.add_argument("--a-col", required=True)
    ap.add_argument("--b", required=True); ap.add_argument("--b-col", required=True)
    ap.add_argument("--out", default="results/ensemble_cv.predictions.csv")
    x = ap.parse_args(argv)
    read = lambda p: pd.read_csv(p, encoding="utf-8-sig")
    table = rank_average(read(x.a), x.a_col, read(x.b), x.b_col)
    Path(x.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(x.out, index=False, encoding="utf-8")
    print(f"записано {len(table)} строк в {x.out}")


if __name__ == "__main__":
    main()
