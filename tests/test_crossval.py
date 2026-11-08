import numpy as np
import pandas as pd

from archword import crossval as cv


def _own(n=60):
    rng = np.random.default_rng(1)
    words = [f"слово{i // 2}" if i % 7 == 0 else f"слово{i}" for i in range(n)]  # у части слов по две строки
    label = np.where(rng.random(n) < 0.25, rng.integers(2, 6, n), 1)
    return pd.DataFrame({"word": words, "context": [f"Он сказал {w}." for w in words], "label": label,
                         "y": (label - 1) / 4, "round": np.where(np.arange(n) < n // 2, "round1", "round2")}).assign(
        key=lambda d: d["word"].str.lower().str.replace("ё", "е"))


def test_assign_folds_keeps_each_word_in_one_fold():
    own = _own()
    folds = cv.assign_folds(own, 5)
    own["fold"] = folds
    assert own.groupby("key")["fold"].nunique().max() == 1
    assert set(folds) == {0, 1, 2, 3, 4}


def test_assign_folds_treats_yo_and_case_as_same_word():
    own = pd.DataFrame({"word": ["Ещё", "еще", "дом", "лес", "кот", "сад"], "label": [1, 1, 3, 1, 1, 2]})
    own["key"] = own["word"].map(lambda w: w.lower().replace("ё", "е"))
    folds = cv.assign_folds(own, 2)
    assert folds[0] == folds[1]


def test_assign_folds_spreads_hard_words_evenly_and_is_reproducible():
    own = _own(200)
    folds = cv.assign_folds(own, 5)
    hard_per_fold = [int(((own["label"] >= 2) & (folds == f)).sum()) for f in range(5)]
    assert max(hard_per_fold) - min(hard_per_fold) <= 3
    assert (folds == cv.assign_folds(own, 5)).all()


def test_assign_folds_needs_two_folds():
    try:
        cv.assign_folds(_own(), 1)
    except ValueError:
        return
    raise AssertionError("ожидалась ошибка")


def _fake(setting, bible, own, folds, cfg):
    # «модель», которая видит только обучающие части: предсказание = среднее по обучению + шум, зависящий от сида
    rng = np.random.default_rng(cfg.seed)
    oof = np.zeros(len(own))
    for f in set(folds.tolist()):
        test = folds == f
        assert not set(own.loc[test, "key"]) & set(own.loc[~test, "key"]), "утечка слова между частями"
        oof[test] = own.loc[test, "y"].to_numpy() + rng.normal(0, 0.1, test.sum())
    return oof


def test_run_cv_reports_pooled_metrics_and_per_seed_columns():
    own = _own(120)
    report, table = cv.run_cv(pd.DataFrame({"word": [], "context": [], "y": []}), own, "bible_then_own", 5, (0, 1), trainer=_fake, n_boot=50)
    assert report["n"] == 120 and report["seeds"] == ["0", "1"] and report["n_folds"] == 5
    # у большинства слов оценка 1 (много равных), поэтому Spearman ниже, чем AUC
    assert report["pooled"]["spearman"] > 0.6 and report["pooled"]["auc_label_ge2"] > 0.95
    assert {"pred_bible_then_own_s0", "pred_bible_then_own_s1", "pred_bible_then_own"} <= set(table.columns)
    assert set(report["per_round"]) == {"round1", "round2"}


def test_run_cv_rejects_incomplete_predictions():
    own = _own()
    bad = lambda *a: np.full(len(own), np.nan)
    try:
        cv.run_cv(pd.DataFrame(), own, "own", 3, (0,), trainer=bad)
    except ValueError:
        return
    raise AssertionError("ожидалась ошибка")


def test_merge_tables_combines_seeds_and_checks_alignment(tmp_path):
    own = _own(80)
    paths = []
    for s in (0, 1):
        _, table = cv.run_cv(pd.DataFrame(), own, "own", 4, (s,), trainer=_fake, n_boot=0)
        p = tmp_path / f"s{s}.csv"
        table.to_csv(p, index=False)
        paths.append(p)
    merged = cv.merge_tables(paths)
    assert cv.seed_columns(merged, "own") == ["pred_own_s0", "pred_own_s1"]
    assert cv.summarize(merged.assign(**{"pred_own": 0}), "own", 20)["seeds"] == ["0", "1"]
    bad = pd.read_csv(paths[1]).iloc[::-1]
    bad.to_csv(paths[1], index=False)
    try:
        cv.merge_tables(paths)
    except ValueError:
        return
    raise AssertionError("ожидалась ошибка про разные строки")
