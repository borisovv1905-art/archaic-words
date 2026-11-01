import numpy as np
import pandas as pd

from archword.data import FLOOR_IPM, DataMissing, load_complexity, load_freq, log_freq
from archword.features import numeric_features
from archword.metrics import agreement, mean_baseline_report, regression_report, to_bins


def test_regression_report_perfect_and_constant():
    r = regression_report([1, 2, 3], [1, 2, 3])
    assert r["spearman"] == 1.0 and r["rmse"] == 0.0
    c = regression_report([1, 2, 3], [2, 2, 2])
    assert c["spearman"] == 0.0


def test_mean_baseline_is_constant():
    r = mean_baseline_report([0.0, 1.0], [0.5, 0.5])
    assert r["rmse"] == 0.0 and r["spearman"] == 0.0


def test_to_bins_scale():
    assert to_bins([0.0, 0.2, 0.4, 0.5, 0.9]).tolist() == [1, 2, 3, 4, 5]


def test_agreement_identical_and_opposite():
    assert agreement([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])["qwk"] == 1.0
    assert agreement([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])["spearman"] == -1.0


def test_log_freq_and_oov():
    freq = {"дом": 100.0}
    lf = log_freq(["Дом", "шлафорка"], freq)
    assert np.isclose(lf[0], 2.0) and np.isclose(lf[1], np.log10(FLOOR_IPM))
    f = numeric_features(["Дом", "шлафорка"], freq)
    assert f["oov"].tolist() == [0, 1] and f["length"].tolist() == [3, 8]


def test_loaders_with_tmp_files(tmp_path):
    (tmp_path / "lexical_complexity").mkdir()
    (tmp_path / "auxiliary").mkdir()
    pd.DataFrame({"sentence": ["а"], "token": ["чаяние"], "start": [0], "end": [6], "complexity": [0.7]}).to_csv(
        tmp_path / "lexical_complexity" / "word_complexity_ru_train.tsv", sep="\t", index=False
    )
    (tmp_path / "auxiliary" / "rus_freq_dictionary_1992-2019.csv").write_text(
        "﻿СЛОВО\tРАНГ\tЧАСТОТА\tДОК\tЛЕММЫ\tPOS\nи\t1\t27994.2\t5\tx\ty\nДом\t2\t170.0\t4\tx\ty\nдом\t3\t10.0\t4\tx\ty\n",
        encoding="utf-8",
    )
    tr = load_complexity(tmp_path, "train")
    assert tr.loc[0, "token"] == "чаяние"
    freq = load_freq(tmp_path)
    assert freq["дом"] == 170.0 and "и" in freq  # максимум по регистрам, BOM убран


def test_missing_data_is_explained(tmp_path):
    try:
        load_freq(tmp_path)
    except DataMissing as e:
        assert "DATA.md" in str(e)
    else:
        raise AssertionError("ожидалась ошибка")
