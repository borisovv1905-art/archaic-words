"""Загрузка публичных датасетов и частотного словаря.

Файлы скачиваются вручную (см. DATA.md) и кладутся в data/raw/.
Ожидаемая структура:
    data/raw/lexical_complexity/word_complexity_ru_{train,test}.tsv
    data/raw/auxiliary/rus_freq_dictionary_1992-2019.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .text import normalize

BIBLE_DIR = "lexical_complexity"
FREQ_FILE = Path("auxiliary") / "rus_freq_dictionary_1992-2019.csv"
FLOOR_IPM = 0.01  # частота для слов, которых нет в словаре


class DataMissing(FileNotFoundError):
    pass


def _need(path: Path, what: str) -> Path:
    if not path.is_file():
        raise DataMissing(f"Не найден {what}: {path}. Как получить файл — в DATA.md.")
    return path


def load_complexity(raw: str | Path, split: str) -> pd.DataFrame:
    """Слова в контексте предложений (синодальная Библия) с оценкой сложности 0..1."""
    if split not in {"train", "test"}:
        raise ValueError("split должен быть 'train' или 'test'")
    path = _need(Path(raw) / BIBLE_DIR / f"word_complexity_ru_{split}.tsv", f"{split}-выборка")
    df = pd.read_csv(path, sep="\t")
    missing = {"sentence", "token", "complexity"} - set(df.columns)
    if missing:
        raise ValueError(f"В {path.name} нет колонок: {sorted(missing)}")
    return df.dropna(subset=["token", "complexity"]).reset_index(drop=True)


def load_freq(raw: str | Path) -> dict[str, float]:
    """Словарь «словоформа -> частота на миллион слов» (максимум по омонимам)."""
    path = _need(Path(raw) / FREQ_FILE, "частотный словарь")
    df = pd.read_csv(path, sep="\t", usecols=[0, 2], header=0, names=["w", "ipm"])
    df["w"] = df["w"].astype(str).str.replace("﻿", "", regex=False).map(normalize)
    df["ipm"] = pd.to_numeric(df["ipm"], errors="coerce")
    return df.dropna().groupby("w")["ipm"].max().to_dict()


def log_freq(words, freq: dict[str, float]) -> np.ndarray:
    """log10 частоты; для отсутствующих слов берётся FLOOR_IPM."""
    return np.array([np.log10(freq.get(normalize(w), FLOOR_IPM)) for w in words])
