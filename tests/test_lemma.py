import numpy as np
import pandas as pd

from archword import lemma as lm
from archword.data import FLOOR_IPM, log_freq


def _fake_lemmatizer(word):
    # «лемма» - слово без окончания «ов/ал/а/ом»: хватает, чтобы проверить суммирование по формам
    for end in ("ал", "ов", "ом", "а"):
        if word.endswith(end) and len(word) > len(end) + 2:
            return word[: -len(end)]
    return word


def _raw(tmp_path, rows):
    aux = tmp_path / "raw" / "auxiliary"
    aux.mkdir(parents=True)
    head = "СЛОВО\tРАНГ\tЧАСТОТА\tДОК\tЛЕММЫ\tPOS\n"
    (aux / "rus_freq_dictionary_1992-2019.csv").write_text(
        head + "".join(f"{w}\t{i}\t{f}\t1\t1 {w} (NOUN)\tNOUN\n" for i, (w, f) in enumerate(rows, 1)), encoding="utf-8")
    return tmp_path / "raw"


def test_totals_sum_frequencies_of_forms_with_same_lemma(tmp_path):
    raw = _raw(tmp_path, [("гроб", 10.0), ("гроба", 5.0), ("гробов", 2.0), ("дом", 7.0)])
    totals = lm.build_totals(raw, _fake_lemmatizer)
    assert round(totals["гроб"], 6) == 17.0 and totals["дом"] == 7.0


def test_unseen_form_gets_frequency_of_its_lemma(tmp_path):
    raw = _raw(tmp_path, [("гроб", 10.0), ("гроба", 5.0)])
    table = lm.LemmaFreq(lm.build_totals(raw, _fake_lemmatizer), _fake_lemmatizer)
    assert table.get("гробом", FLOOR_IPM) == 15.0  # формы «гробом» в словаре нет, лемма есть
    assert table.get("совсемредкое", FLOOR_IPM) == FLOOR_IPM
    assert np.isclose(log_freq(["Гробом"], table)[0], np.log10(15.0))  # регистр и ё нормализуются


def test_cache_is_written_once_and_reused(tmp_path):
    raw = _raw(tmp_path, [("гроб", 10.0), ("гроба", 5.0)])
    calls = []

    def counting(word):
        calls.append(word)
        return _fake_lemmatizer(word)

    first = lm.load_lemma_freq_morph(raw, counting)
    n_first = len(calls)
    assert (raw / "auxiliary" / lm.CACHE_NAME).is_file() and n_first >= 2
    second = lm.load_lemma_freq_morph(raw, counting)
    assert len(calls) == n_first  # словарь заново не разбирался
    assert first.totals == second.totals


def test_save_and_load_totals_roundtrip(tmp_path):
    totals = {"гроб": 17.0, "дом": 7.5}
    lm.save_totals(totals, tmp_path / "t.csv")
    assert lm.load_totals(tmp_path / "t.csv") == totals
