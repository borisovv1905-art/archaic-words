import numpy as np
import pandas as pd

from archword import compare as cmp


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(1, 5, n).astype(float)
    good = y + rng.normal(0, 0.7, n)
    noise = rng.normal(0, 1, n)
    return y, good, noise


def test_paired_bootstrap_detects_clearly_better_model():
    y, good, noise = _data()
    res = cmp.paired_bootstrap(y, good, noise, n_boot=300)
    assert res["spearman"]["diff"] > 0.3
    assert res["spearman"]["diff_ci95"][0] > 0
    assert res["spearman"]["share_not_better"] < 0.01


def test_paired_bootstrap_identical_predictions_have_zero_difference():
    y, good, _ = _data()
    res = cmp.paired_bootstrap(y, good, good, n_boot=200)
    for name in cmp.METRICS:
        assert res[name]["diff"] == 0 and res[name]["diff_ci95"] == [0, 0]


def test_paired_bootstrap_is_reproducible():
    y, good, noise = _data()
    assert cmp.paired_bootstrap(y, good, noise, n_boot=100, seed=3) == cmp.paired_bootstrap(y, good, noise, n_boot=100, seed=3)


def test_error_table_ranks_use_average_for_ties_and_sign_of_gap():
    df = pd.DataFrame({"word": ["а", "б", "в", "г"], "label": [5, 1, 1, 1], "p": [0.1, 0.9, 0.2, 0.3]})
    t = cmp.error_table(df, "p").set_index("word")
    assert t.loc["а", "rank_reader"] == 1 and t.loc["б", "rank_reader"] == 3  # три «единицы» делят ранги 2,3,4
    # модель считает «а» лёгким (низкий балл), читатель - самым трудным: модель ставит его ниже, ранг больше
    assert t.loc["а", "rank_gap"] > 0 and t.loc["б", "rank_gap"] < 0
    assert t.index[0] == "а"


def test_precision_at_r_counts_hits_in_top_r():
    y = np.array([1, 1, 3, 3, 1, 1.0])
    assert cmp.precision_at_r(y, np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.0]), 3) == 1.0
    assert cmp.precision_at_r(y, np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.0]), 3) == 0.0


def test_precision_at_r_splits_ties_evenly():
    y = np.array([3, 1, 1, 1.0])
    # все баллы равны: из R=1 слов случайный выбор попадает в трудное слово с вероятностью 1/4
    assert cmp.precision_at_r(y, np.zeros(4), 3) == 0.25


def test_by_round_reports_each_round():
    df = pd.DataFrame({"label": [1, 3, 1, 4, 1, 1], "round": ["a", "a", "a", "b", "b", "b"]})
    out = cmp.by_round(df, {"model": df["label"].to_numpy(float)})
    assert set(out) == {"a", "b"} and out["a"]["n"] == 3
    assert out["a"]["model"]["auc_ge2"] == 1.0


def test_run_accepts_other_model_and_checks_alignment(tmp_path):
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw" / "auxiliary"
    raw.mkdir(parents=True)
    words = [f"слово{i}" for i in range(80)]
    (raw / "rus_freq_dictionary_1992-2019.csv").write_text(
        "Lemma\tPoS\tFreq(ipm)\tR\tLEMMAS\n" + "".join(f"{w}\tx\t{i + 1}.0\t1\t{i + 1} {w} (NOUN)\n" for i, w in enumerate(words)), encoding="utf-8")
    y = rng.integers(1, 5, 80)
    df = pd.DataFrame({"word": words, "context": ["фраза"] * 80, "label": y, "m": y + rng.normal(0, 1, 80), "o": rng.normal(0, 1, 80)})
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    df[["word", "context", "label", "m"]].to_csv(a, index=False)
    df[["word", "context", "label", "o"]].to_csv(b, index=False)
    report, _ = cmp.run(tmp_path / "raw", a, "m", n_boot=30, other=b, other_col="o")
    assert "other:o" in report["vs"] and report["vs"]["other:o"]["spearman"]["diff"] > 0
    df.iloc[::-1][["word", "context", "label", "o"]].to_csv(b, index=False)
    try:
        cmp.run(tmp_path / "raw", a, "m", n_boot=30, other=b, other_col="o")
    except ValueError:
        return
    raise AssertionError("ожидалась ошибка про разные строки")
