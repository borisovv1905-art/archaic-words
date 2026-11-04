"""Согласие разметчика с самим собой: повторная оценка тех же слов.

    python -m archword.reliability --old data/annotation/round1_labeled.csv \
        --old data/annotation/round2_labeled.csv --new data/annotation/round3_labeled.csv \
        --out results/reliability.json

Слова, которые есть и в старых таблицах, и в новой, считаются повторными (они были добавлены в таблицу
вслепую, см. `archword.annotate --retest-n`). Считаются точное совпадение оценок, совпадение с точностью до
единицы, взвешенная каппа, корреляция Спирмена и совпадение решения «слово трудное» для порогов 2 и 3.
Это потолок для любой модели: нельзя ждать, что модель согласуется с разметчиком лучше, чем он сам с собой.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from .text import normalize
from .transfer import load_labeled

THRESHOLDS = (2, 3)


def _stat(a: np.ndarray, b: np.ndarray, fn) -> float | None:
    try:
        v = float(fn(a, b))
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def _qwk(a, b):
    return cohen_kappa_score(a, b, weights="quadratic")


def _rho(a, b):
    return spearmanr(a, b)[0]


def _boot(a: np.ndarray, b: np.ndarray, fn, n_boot: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        v = _stat(a[i], b[i], fn)
        if v is not None:
            vals.append(v)
    return [round(float(x), 3) for x in np.percentile(vals, [2.5, 97.5])] if vals else []


REMEMBER_NOTE = "повтор"  # пометка в note: «помню, что уже оценивал это слово»


def pair(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Слова, размеченные дважды: колонки word, old, new, remembered (была пометка «повтор»)."""
    o = old.assign(key=old["word"].map(normalize)).drop_duplicates("key")[["key", "word", "label"]]
    flag = new["note"].astype(str).str.strip().str.lower().eq(REMEMBER_NOTE) if "note" in new else False
    n = new.assign(key=new["word"].map(normalize), remembered=flag).drop_duplicates("key")[["key", "label", "remembered"]]
    both = o.merge(n, on="key", suffixes=("_old", "_new"))
    both = both.rename(columns={"label_old": "old", "label_new": "new"})
    return both[["word", "old", "new", "remembered"]].reset_index(drop=True)


def compare(old: pd.DataFrame, new: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    pr = pair(old, new)
    if len(pr) < 2:
        raise ValueError("Не нашлось повторно размеченных слов: проверьте, что в новой таблице есть слова из старых.")
    a, b = pr["old"].to_numpy(int), pr["new"].to_numpy(int)
    res = {
        "n_words": int(len(pr)),
        "exact_agreement": round(float((a == b).mean()), 3),
        "within_one": round(float((np.abs(a - b) <= 1).mean()), 3),
        "mean_old": round(float(a.mean()), 2),
        "mean_new": round(float(b.mean()), 2),
        "qwk": None, "qwk_ci95": [], "spearman": None, "spearman_ci95": [],
    }
    q, r = _stat(a, b, _qwk), _stat(a, b, _rho)
    if q is not None:
        res["qwk"], res["qwk_ci95"] = round(q, 3), _boot(a, b, _qwk, n_boot, seed)
    if r is not None:
        res["spearman"], res["spearman_ci95"] = round(r, 3), _boot(a, b, _rho, n_boot, seed)
    marked = pr["remembered"].to_numpy(bool)
    res["marked_as_remembered"] = int(marked.sum())
    if marked.any() and (~marked).sum() >= 5:
        a2, b2 = a[~marked], b[~marked]
        res["without_remembered"] = {"n_words": int(len(a2)), "exact_agreement": round(float((a2 == b2).mean()), 3),
                                     "qwk": None if _stat(a2, b2, _qwk) is None else round(_stat(a2, b2, _qwk), 3)}
    for t in THRESHOLDS:
        x, y = (a >= t).astype(int), (b >= t).astype(int)
        res[f"binary_ge{t}"] = {
            "agreement": round(float((x == y).mean()), 3),
            "kappa": None if _stat(x, y, cohen_kappa_score) is None else round(_stat(x, y, cohen_kappa_score), 3),
            "hard_old": int(x.sum()), "hard_new": int(y.sum()),
        }
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", action="append", required=True, help="старые размеченные таблицы (можно несколько)")
    ap.add_argument("--new", required=True, help="новая таблица, в которой есть повторные слова")
    ap.add_argument("--out", default="results/reliability.json")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    old = pd.concat([load_labeled(p) for p in args.old], ignore_index=True)
    report = compare(old, load_labeled(args.new), n_boot=args.boot)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
