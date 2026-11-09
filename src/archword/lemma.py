"""Частота леммы с настоящей лемматизацией (pymorphy3) вместо колонки лемм из частотного словаря.

В словаре OpenCorpora у глагола две леммы («сетовать» как инфинитив и как глагол), и частота делится между ними,
поэтому у глаголов частота леммы занижена (см. README). Здесь каждая словоформа словаря получает одну лемму от pymorphy3
(первый, самый вероятный разбор), а частота леммы - это сумма частот всех форм с этой леммой.

    pip install pymorphy3 pymorphy3-dicts-ru
    python -m archword.lemma --raw data/raw          # один раз: считает и сохраняет таблицу (несколько минут)

Таблица кладётся в data/raw/auxiliary/lemma_freq_pymorphy3.csv (папка data/raw в git не попадает).
Лемматизатор передаётся функцией, поэтому тесты обходятся без pymorphy3.
Разбор без контекста: для омонимичных форм берётся самый вероятный разбор, это упрощение.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pandas as pd

from .data import FLOOR_IPM, FREQ_FILE, _need
from .text import normalize

CACHE_NAME = "lemma_freq_pymorphy3.csv"
Lemmatizer = Callable[[str], str]


def pymorphy_lemmatizer() -> Lemmatizer:
    try:
        import pymorphy3
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise SystemExit("Нужен pymorphy3: pip install pymorphy3 pymorphy3-dicts-ru") from exc
    morph = pymorphy3.MorphAnalyzer()

    @lru_cache(maxsize=None)
    def lemmatize(word: str) -> str:
        return normalize(morph.parse(word)[0].normal_form)

    return lemmatize


class LemmaFreq:
    """Похож на словарь «слово -> частота леммы на миллион»: поддерживает .get(слово, по_умолчанию).

    Слова, которых нет в словаре, тоже получают частоту леммы, если лемма в нём есть (это и отличает таблицу
    от словаря по формам: «подымал» не в словаре, а лемма «подымать» в словаре есть).
    """

    def __init__(self, totals: dict[str, float], lemmatize: Lemmatizer):
        self.totals, self.lemmatize = totals, lemmatize

    def get(self, word: str, default: float | None = None):
        w = normalize(word)
        return self.totals.get(self.lemmatize(w), default)

    def __len__(self) -> int:
        return len(self.totals)


def build_totals(raw: str | Path, lemmatize: Lemmatizer, progress: bool = False) -> dict[str, float]:
    """Сумма частот всех форм словаря по леммам, найденным лемматизатором."""
    path = _need(Path(raw) / FREQ_FILE, "частотный словарь")
    df = pd.read_csv(path, sep="\t", usecols=[0, 2], header=0, names=["w", "ipm"])
    df["w"] = df["w"].astype(str).str.replace("﻿", "", regex=False).map(normalize)
    df["ipm"] = pd.to_numeric(df["ipm"], errors="coerce")
    forms = df.dropna().groupby("w")["ipm"].max()  # как в load_freq: максимум по омонимам
    totals: dict[str, float] = {}
    for i, (w, ipm) in enumerate(forms.items()):
        lem = lemmatize(w)
        totals[lem] = totals.get(lem, 0.0) + float(ipm)
        if progress and i % 200_000 == 0:
            print(f"  разобрано {i:,} из {len(forms):,} форм", flush=True)
    return totals


def save_totals(totals: dict[str, float], path: str | Path) -> None:
    pd.Series(totals, name="ipm").rename_axis("lemma").reset_index().to_csv(path, index=False, encoding="utf-8")


def load_totals(path: str | Path) -> dict[str, float]:
    df = pd.read_csv(path, encoding="utf-8")
    return dict(zip(df["lemma"].astype(str), df["ipm"].astype(float)))


def load_lemma_freq_morph(raw: str | Path, lemmatize: Lemmatizer | None = None, progress: bool = False) -> LemmaFreq:
    """Таблица частот лемм: из кэша, если он есть, иначе считается и сохраняется."""
    cache = Path(raw) / "auxiliary" / CACHE_NAME
    lemmatize = lemmatize or pymorphy_lemmatizer()
    if cache.is_file():
        totals = load_totals(cache)
    else:
        totals = build_totals(raw, lemmatize, progress)
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_totals(totals, cache)
    return LemmaFreq(totals, lemmatize)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--words", nargs="*", default=["сетовать", "сетовал", "подымал", "ямщиком", "гробовщика"],
                    help="слова для быстрой проверки результата")
    a = ap.parse_args(argv)
    table = load_lemma_freq_morph(a.raw, progress=True)
    print(f"лемм в таблице: {len(table):,}")
    for w in a.words:
        print(f"  {w:14s} лемма {table.lemmatize(normalize(w)):14s} частота леммы {table.get(w, FLOOR_IPM):10.2f} на млн")


if __name__ == "__main__":
    main()
