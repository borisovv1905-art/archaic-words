"""Бейзлайны на публичной выборке: частота+длина и дополнительно символьные n-граммы.

    python -m archword.baseline --raw data/raw --out results/baseline_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_complexity, load_freq
from .features import NUMERIC, add_text_column, numeric_features
from .metrics import mean_baseline_report, regression_report


def build_frame(df, freq):
    return add_text_column(numeric_features(df["token"], freq), df["token"])


def make_model(kind: str) -> Pipeline:
    if kind == "freq_len":
        pre = ColumnTransformer([("num", StandardScaler(), NUMERIC)])
    elif kind == "freq_len_charngram":
        pre = ColumnTransformer(
            [
                ("num", StandardScaler(), NUMERIC),
                ("chr", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True), "word"),
            ]
        )
    else:
        raise ValueError(f"неизвестная модель: {kind}")
    return Pipeline([("pre", pre), ("reg", Ridge(alpha=1.0))])


def run(raw: str | Path) -> dict:
    freq = load_freq(raw)
    train, test = load_complexity(raw, "train"), load_complexity(raw, "test")
    overlap = len(set(train.token.str.lower()) & set(test.token.str.lower()))
    xtr, xte = build_frame(train, freq), build_frame(test, freq)
    ytr, yte = train["complexity"].to_numpy(), test["complexity"].to_numpy()

    report = {
        "data": {"train": len(train), "test": len(test), "token_overlap_train_test": overlap},
        "models": {"mean": mean_baseline_report(ytr, yte)},
    }
    rho = regression_report(yte, -xte["log_freq"].to_numpy())["spearman"]
    report["models"]["only_-log_freq (ранжирование без обучения)"] = {"spearman": rho}
    for kind in ("freq_len", "freq_len_charngram"):
        model = make_model(kind).fit(xtr, ytr)
        report["models"][kind] = regression_report(yte, model.predict(xte))
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/baseline_metrics.json")
    args = ap.parse_args(argv)
    report = run(args.raw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
