"""Признаки слова: частота, длина, «нет в словаре», символьные n-граммы."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import log_freq
from .text import normalize

NUMERIC = ["log_freq", "oov", "length"]


def numeric_features(words, freq: dict[str, float]) -> pd.DataFrame:
    words = list(words)
    norm = [normalize(w) for w in words]
    return pd.DataFrame(
        {
            "log_freq": log_freq(words, freq),
            "oov": [int(w not in freq) for w in norm],
            "length": [len(w) for w in norm],
        }
    )


def add_text_column(df: pd.DataFrame, words) -> pd.DataFrame:
    out = df.copy()
    out["word"] = [normalize(w) for w in words]
    return out
