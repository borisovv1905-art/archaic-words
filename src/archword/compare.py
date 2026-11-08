"""Парное сравнение дообученной модели с частотными бейзлайнами на одних и тех же словах.

    python -m archword.compare --raw data/raw \
        --predictions results/finetune_best.predictions.csv \
        --pred-col pred_bible_then_own --out results/finetune_vs_baselines.json

Доверительные интервалы отдельных моделей перекрываются легче, чем интервал разности: слова
в тесте общие, и модели ошибаются на них согласованно. Поэтому разность считается на каждой
бутстрэп-выборке слов сразу для обеих моделей (парный бутстрэп). Также считается доля выборок,
где модель не лучше бейзлайна (это односторонняя оценка p, а не поправленная на множественные сравнения).

Таблица ошибок: для каждого слова ранг в предсказании модели и ранг в оценке читателя
(по убыванию трудности), разность рангов показывает, где модель больше всего расходится с читателем.
Если в таблице предсказаний есть колонка round (как в `archword.crossval`), метрики считаются ещё и по раундам.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

from .data import load_freq, load_lemma_freq, log_freq
from .transfer import THRESHOLDS


def _metric(name: str, y: np.ndarray, p: np.ndarray) -> float:
    if name == "spearman":
        if np.ptp(y) == 0 or np.ptp(p) == 0:
            raise ValueError("константа")
        return float(spearmanr(y, p)[0])
    t = int(name.rsplit("ge", 1)[1])
    pos = (y >= t).astype(int)
    if pos.sum() in (0, len(pos)):
        raise ValueError("один класс")
    return float(roc_auc_score(pos, p))


METRICS = ["spearman"] + [f"auc_ge{t}" for t in THRESHOLDS]


def paired_bootstrap(y, p_model, p_base, n_boot: int = 5000, seed: int = 0) -> dict:
    """Разность «модель минус бейзлайн» по каждой метрике с парным бутстрэпом по словам."""
    y, p_model, p_base = (np.asarray(a, float) for a in (y, p_model, p_base))
    rng = np.random.default_rng(seed)
    n = len(y)
    out = {}
    for name in METRICS:
        diffs = []
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            try:
                diffs.append(_metric(name, y[i], p_model[i]) - _metric(name, y[i], p_base[i]))
            except ValueError:
                continue
        d = np.array(diffs)
        out[name] = {
            "model": round(_metric(name, y, p_model), 3),
            "baseline": round(_metric(name, y, p_base), 3),
            "diff": round(_metric(name, y, p_model) - _metric(name, y, p_base), 3),
            "diff_ci95": [round(float(x), 3) for x in np.percentile(d, [2.5, 97.5])],
            "share_not_better": round(float((d <= 0).mean()), 4),
            "n_boot_ok": int(len(d)),
        }
    return out


def precision_at_r(y, p, t: int) -> float:
    """Точность среди R слов с самыми высокими баллами, где R - число слов с оценкой не ниже t (R-precision).

    При равных баллах (у частотного бейзлайна их много) берётся математическое ожидание при случайном порядке.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    pos = y >= t
    r = int(pos.sum())
    if r == 0:
        raise ValueError("нет трудных слов")
    cut = np.sort(p)[::-1][r - 1]
    above, tied = p > cut, p == cut
    hits = pos[above].sum() + pos[tied].mean() * (r - above.sum())
    return round(float(hits / r), 3)


def by_round(df: pd.DataFrame, cols: dict[str, np.ndarray]) -> dict:
    """Spearman и AUC для порога 2 по раундам разметки (если в таблице есть колонка round)."""
    out = {}
    for r, g in df.groupby("round"):
        i = g.index.to_numpy()
        row = {"n": int(len(g)), "share_hard": round(float((g["label"] >= 2).mean()), 3)}
        for name, v in cols.items():
            try:
                row[name] = {"spearman": round(_metric("spearman", g["label"].to_numpy(float), v[i]), 3),
                             "auc_ge2": round(_metric("auc_ge2", g["label"].to_numpy(float), v[i]), 3)}
            except ValueError:
                row[name] = None
        out[str(r)] = row
    return out


def error_table(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Ранги по убыванию трудности: 1 = самое трудное. rank_gap > 0: модель считает слово менее трудным, чем читатель."""
    t = df[["word", "label", pred_col]].copy()
    # средние ранги при равных оценках, чтобы 160 «единиц» не получили разные ранги по порядку строк
    t["rank_reader"] = rankdata(-t["label"].to_numpy(), method="average")
    t["rank_model"] = rankdata(-t[pred_col].to_numpy(), method="average")
    t["rank_gap"] = t["rank_model"] - t["rank_reader"]
    return t.sort_values("rank_gap", ascending=False).reset_index(drop=True)


def run(raw, predictions, pred_col: str, n_boot: int = 5000, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(predictions, encoding="utf-8-sig")
    if pred_col not in df or "label" not in df or "word" not in df:
        raise ValueError(f"В {predictions} нужны колонки word, label, {pred_col}")
    freq, lem = load_freq(raw), load_lemma_freq(raw)
    y = df["label"].to_numpy(float)
    p = df[pred_col].to_numpy(float)
    bases = {
        "-log_freq": -log_freq(df["word"], freq),
        "-log_lemma_freq": -log_freq(df["word"], lem),
    }
    report = {"n": int(len(df)), "pred_col": pred_col, "vs": {k: paired_bootstrap(y, p, v, n_boot, seed) for k, v in bases.items()}}
    allp = {"model": p, **bases}
    report["precision_at_r"] = {name: {f"ge{t}": precision_at_r(y, v, t) for t in THRESHOLDS if (y >= t).any()} for name, v in allp.items()}
    if "round" in df:
        report["by_round"] = by_round(df, allp)
    return report, error_table(df, pred_col)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--pred-col", required=True)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--out", default="results/finetune_vs_baselines.json")
    a = ap.parse_args(argv)
    report, table = run(a.raw, a.predictions, a.pred_col, a.boot)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    table.to_csv(out.with_suffix(".errors.csv"), index=False, encoding="utf-8")
    for base, res in report["vs"].items():
        print(base)
        for m, r in res.items():
            print(f"  {m:9s} модель {r['model']:.3f} | база {r['baseline']:.3f} | разность {r['diff']:+.3f} {r['diff_ci95']} | доля «не лучше» {r['share_not_better']}")
    print("точность среди R слов с наибольшим баллом (R - число трудных слов):", report["precision_at_r"])


if __name__ == "__main__":
    main()
