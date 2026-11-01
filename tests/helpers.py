"""Синтетические данные для тестов."""
import numpy as np
import pandas as pd


def code(i):  # уникальное «слово» из букв: номер в 26-ричной записи
    return "с" + chr(1072 + i // 26) + chr(1072 + i % 26) + "ъ"


def make_raw(tmp_path, n=60, seed=0):
    """Папка в формате data/raw: частотный словарь и выборки train/test."""
    rng = np.random.default_rng(seed)
    (tmp_path / "lexical_complexity").mkdir(parents=True)
    (tmp_path / "auxiliary").mkdir(parents=True)
    words = [code(i) for i in range(n)]
    freq_rows = [f"{w}\t{i + 1}\t{10 ** rng.uniform(-1, 3):.4f}\t1\tx\ty" for i, w in enumerate(words)]
    (tmp_path / "auxiliary" / "rus_freq_dictionary_1992-2019.csv").write_text(
        "СЛОВО\tРАНГ\tЧАСТОТА\tДОК\tЛЕММЫ\tPOS\n" + "\n".join(freq_rows), encoding="utf-8"
    )
    freq = {r.split("\t")[0]: float(r.split("\t")[2]) for r in freq_rows}
    cols = {"sentence": "а", "start": 0, "end": 1}
    for split, ws in (("train", words[:45]), ("test", words[45:])):
        df = pd.DataFrame({**cols, "token": ws, "complexity": [0.5 - 0.15 * np.log10(freq[w]) for w in ws]})
        df.to_csv(tmp_path / "lexical_complexity" / f"word_complexity_ru_{split}.tsv", sep="\t", index=False)
    return tmp_path, words, freq
