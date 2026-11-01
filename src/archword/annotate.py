"""Таблица для собственной разметки: слова повестей, которые нужно оценить.

    python -m archword.annotate --corpus data/corpus/belkin.json --raw data/raw \
        --n 200 --out data/annotation/belkin_sheet.csv

Второй раунд: добрать редкие слова, не повторяя уже размеченные:
    python -m archword.annotate --max-ipm 2 --n 300 --seed 1 \
        --exclude data/annotation/round1_labeled.csv --out data/annotation/round2_sheet.csv

Шкала оценки (та же, что у публичных датасетов, 1..5):
    1 — слово знакомо, объяснять нечего
    2 — понятно по контексту
    3 — слышал, но уверенно не объясню
    4 — смутно догадываюсь
    5 — не знаю, нужен словарь

Слова выбираются случайно внутри пяти групп по частоте в современном корпусе.
Нельзя размечать только «подозрительные» слова: тогда модель не увидит, что
такое «простое слово», и оценка получится завышенной.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from .data import load_freq, log_freq
from .text import iter_occurrences, normalize

COLUMNS = ["id", "word", "tale", "context", "ipm", "maybe_name", "label", "note"]
MIN_LEN = 3


def collect(tales: dict[str, str]) -> pd.DataFrame:
    """Одна строка на уникальную нормализованную форму: первое вхождение с контекстом."""
    seen: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for tale, body in tales.items():
        for tok, sentence, maybe_name in iter_occurrences(body):
            key = normalize(tok.text)
            counts[key] = counts.get(key, 0) + 1
            if key not in seen:
                seen[key] = {"word": key, "form": tok.text, "tale": tale, "context": sentence, "maybe_name": maybe_name}
            elif not maybe_name:
                seen[key]["maybe_name"] = False  # встретилось и без заглавной — значит не имя
    df = pd.DataFrame(seen.values())
    if df.empty:
        return df
    df["count"] = df["word"].map(counts)
    return df[df["word"].str.len() >= MIN_LEN].reset_index(drop=True)


def sample_sheet(
    tales: dict[str, str],
    freq: dict[str, float],
    n: int = 200,
    seed: int = 0,
    include_names: bool = False,
    max_ipm: float | None = None,
    exclude: set[str] | frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Выбирает слова для разметки.

    По умолчанию слова берутся случайно внутри пяти групп по частоте.
    Если задан max_ipm, берутся только слова не чаще этого порога (на миллион слов),
    без деления на группы: так добирают редкие слова, где почти все «трудные».
    Слова из exclude (уже размеченные) пропускаются.
    """
    df = collect(tales)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    if not include_names:
        df = df[~df["maybe_name"]]
    banned = {normalize(w) for w in exclude}
    df = df[~df["word"].isin(banned)]
    df = df.assign(lf=log_freq(df["word"], freq))
    df["ipm"] = df["word"].map(lambda w: freq.get(w, 0.0))
    if max_ipm is not None:
        df = df[df["ipm"] <= max_ipm]
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    # группы равного размера по частоте (дубликаты границ допустимы); с max_ipm одна группа
    q = 1 if max_ipm is not None else min(5, len(df))
    df["bin"] = 0 if q == 1 else pd.qcut(df["lf"].rank(method="first"), q=q, labels=False)
    rng = random.Random(seed)
    per_bin = max(1, n // df["bin"].nunique())
    picked = []
    for _, grp in df.groupby("bin"):
        idx = list(grp.index)
        rng.shuffle(idx)
        picked.extend(idx[:per_bin])
    out = df.loc[picked].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out["id"] = range(1, len(out) + 1)
    out["label"] = ""
    out["note"] = ""
    out["word"] = out["form"].str.lower()
    out["maybe_name"] = out["maybe_name"].astype(int)
    return out[COLUMNS]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus/belkin.json")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ipm", type=float, default=None,
                    help="брать только слова не чаще этого порога (на миллион слов), например 2")
    ap.add_argument("--exclude", action="append", default=[],
                    help="CSV с уже размеченными словами (колонка word); можно повторять")
    ap.add_argument("--out", default="data/annotation/belkin_sheet.csv")
    args = ap.parse_args(argv)
    tales = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    banned = set()
    for path in args.exclude:
        banned |= set(pd.read_csv(path, encoding="utf-8-sig")["word"].astype(str))
    sheet = sample_sheet(tales, load_freq(args.raw), n=args.n, seed=args.seed,
                         max_ipm=args.max_ipm, exclude=banned)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Записано {len(sheet)} слов в {out}. Заполните колонку label (1..5).")


if __name__ == "__main__":
    main()
