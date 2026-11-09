import json

import numpy as np
import pandas as pd

from archword import predict as pr
from archword import train_final as tf
from archword.classic_cv import _model
from archword.lemma import LemmaFreq


def _lemmatizer(word):
    for end in ("ом", "ов", "а", "у"):
        if word.endswith(end) and len(word) > len(end) + 2:
            return word[: -len(end)]
    return word


def _lemma():
    return LemmaFreq({"гроб": 17.0, "дом": 40.0, "ямщик": 2.0, "перекладн": 0.3, "тележк": 3.0}, _lemmatizer)


def _forms():
    return pd.DataFrame({"ipm": [10.0, 40.0, 1.5, np.nan], "oc_lemma": [17.0, 40.0, 2.0, 0.3]},
                        index=pd.Index(["гроб", "дом", "ямщик", "перекладной"], name="w"))


def _own(n=80):
    rng = np.random.default_rng(0)
    pool = ["гроб", "дом", "ямщик", "перекладной", "тележка", "урядник", "десятник", "каморка", "ветошь", "кров"]
    words = [pool[i % len(pool)] + ("ий" if i >= len(pool) else "") + str(i // 20) for i in range(n)]
    hard = np.isin([w[:4] for w in words], ["ямщи", "пере", "урядн", "десят", "ветош"]) | (rng.random(n) < 0.15)
    label = np.where(hard, rng.integers(2, 6, n), 1)
    return pd.DataFrame({"word": words, "key": words, "context": ["ф"] * n, "label": label, "y": (label - 1) / 4,
                         "round": np.where(np.arange(n) < n // 2, "round1", "round2")})


def test_char_wb_ngrams_match_sklearn_analyzer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    analyzer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).build_analyzer()
    for w in ["ямщик", "а", "он", "кой-где", "перекладной", "ёж"]:
        assert pr.char_wb_ngrams(w) == analyzer(w)


def test_numpy_model_reproduces_sklearn_pipeline_exactly():
    own = _own()
    words = own["key"].tolist()
    X = pd.concat([pd.Series(words, name="word"), pd.DataFrame(pr.numeric_matrix(words, _forms(), _lemma()))], axis=1)
    X.columns = ["word", "f1", "f2", "f3", "f4"]
    fitted = _model("ngram").fit(X.to_numpy(dtype=object), own["y"].to_numpy())
    spec = tf.export_pipeline(fitted.best_estimator_)
    ours = pr.LinearModel(json.loads(json.dumps(spec))).predict(words, X[["f1", "f2", "f3", "f4"]].to_numpy(float))
    assert np.allclose(ours, fitted.best_estimator_.predict(X.to_numpy(dtype=object)), atol=1e-8)
    unseen = ["ямщиком", "незнакомец", "а"]
    num = pr.numeric_matrix(unseen, _forms(), _lemma())
    expected = fitted.best_estimator_.predict(pd.concat([pd.Series(unseen, name="word"), pd.DataFrame(num, columns=list("abcd"))], axis=1).to_numpy(dtype=object))
    assert np.allclose(pr.LinearModel(spec).predict(unseen, num), expected, atol=1e-8)


def test_numeric_matrix_uses_floor_for_missing_and_lemma_for_unseen_forms():
    m = pr.numeric_matrix(["гроб", "перекладной", "неведомое", "гробом"], _forms(), _lemma())
    assert np.isclose(m[0, 0], 1.0) and np.isclose(m[2, 0], np.log10(0.01))  # форма из таблицы и слово без частоты
    assert np.isclose(m[1, 0], np.log10(0.01)) and np.isclose(m[1, 1], np.log10(0.3))  # частота формы ниже порога
    assert np.isclose(m[3, 2], np.log10(17.0))  # формы «гробом» нет, но лемма «гроб» известна
    assert m[:, 3].tolist() == [4, 11, 9, 6]


def test_maybe_name_marks_capitalized_words_inside_sentence_only():
    t = "Он сказал: «Кто идет?» Вышел Юрко. — Куда, Иван?"
    pos = lambda w: t.index(w)  # noqa: E731
    assert not pr.maybe_name(t, pos("Он")) and not pr.maybe_name(t, pos("Кто")) and not pr.maybe_name(t, pos("Куда"))
    assert pr.maybe_name(t, pos("Юрко")) and pr.maybe_name(t, pos("Иван"))
    assert not pr.maybe_name(t, pos("идет"))


def _scorer(tmp_path):
    own = _own()
    arts = tf.fit_artifacts(own, _forms(), _lemma(), k=4, n_boot=20)
    tf.save(tmp_path, arts, _forms(), _lemma().totals)
    return pr.Scorer.load(tmp_path, _lemmatizer), arts


def test_scorer_roundtrip_and_ratings_stay_on_scale(tmp_path):
    scorer, arts = _scorer(tmp_path)
    r = scorer.score_words(["Ямщиком", "дом", "ямщиком"])
    assert list(r.index) == ["дом", "ямщиком"] and ((r["rating"] >= 1) & (r["rating"] <= 5)).all()
    assert r.at["ямщиком", "lemma"] == "ямщик"
    assert ((r["prob"] >= 0) & (r["prob"] <= 1)).all()
    assert 0.05 <= scorer.suggested_threshold <= 0.9 and arts["meta"]["n_train"] == 80
    assert set(arts["meta"]["calibration"]) == {"a", "b"} and arts["meta"]["calibration"]["a"] > 0


def test_annotate_gives_positions_that_point_into_original_text(tmp_path):
    scorer, _ = _scorer(tmp_path)
    text = "Приехал Юрко на перекладной.\nЯмщик спал."
    occ = scorer.annotate(text)
    assert [text[s:e] for s, e in zip(occ["start"], occ["end"])] == occ["token"].tolist()
    assert occ.loc[occ["token"] == "Юрко", "rating"].isna().all()  # имя пропущено
    assert scorer.annotate(text, skip_names=False)["rating"].notna().all()


def test_highlight_marks_only_words_above_threshold_and_escapes_html(tmp_path):
    occ = pd.DataFrame({"token": ["дом", "ямщик"], "start": [0, 4], "end": [3, 9], "word": ["дом", "ямщик"],
                        "lemma": ["дом", "ямщик"], "rating": [1.2, 4.0], "prob": [0.05, 0.8], "is_name": [False, False]})
    out = pr.highlight_html("дом ямщик <b>&", occ, 0.3)
    assert out.count("<mark") == 1 and ">ямщик</mark>" in out and "&lt;b&gt;&amp;" in out and "вероятность 80%" in out
    assert "<mark" not in pr.highlight_html("дом ямщик", occ, 0.9)


def test_summary_table_groups_repeated_words_and_sorts_by_rating():
    occ = pd.DataFrame({"token": ["Ямщик", "дом", "ямщик", "урядник"], "word": ["ямщик", "дом", "ямщик", "урядник"],
                        "lemma": ["ямщик", "дом", "ямщик", "урядник"], "rating": [3.0, 1.0, 3.0, 4.5], "prob": [0.4, 0.02, 0.4, 0.7],
                        "is_name": [False] * 4, "start": [0, 1, 2, 3], "end": [1, 2, 3, 4]})
    t = pr.summary_table(occ, 0.3)
    assert t["слово"].tolist() == ["урядник", "Ямщик"] and t["раз"].tolist() == [1, 2]


def test_suggest_threshold_finds_cutoff_between_classes_and_reports_quality():
    labels = np.array([1] * 20 + [4] * 10)
    prob = np.array([0.1] * 20 + [0.6] * 10)
    best = tf.suggest_threshold(labels, prob)
    assert 0.11 <= best["threshold"] <= 0.6 and best["precision"] == 1.0 and best["recall"] == 1.0


def test_calibration_maps_higher_predictions_to_higher_probabilities():
    pred = np.concatenate([np.full(40, 0.05), np.full(40, 0.5)])
    labels = np.concatenate([np.ones(36), np.full(4, 3), np.ones(10), np.full(30, 3)])
    cal = tf.fit_calibration(pred, labels)
    p = pr.to_prob([0.05, 0.5], cal)
    assert cal["a"] > 0 and p[0] < 0.2 < 0.6 < p[1]
    assert pr.to_prob([-0.3, 2.0], None).tolist() == [0.0, 1.0]


def test_compact_form_table_drops_rare_and_keeps_lemma_only_rows():
    t = pd.DataFrame({"ipm": [5.0, 0.001, 0.005], "oc_lemma": [5.0, 0.001, 3.0]}, index=["a", "b", "c"])
    c = tf.compact_form_table(t)
    assert list(c.index) == ["a", "c"] and np.isnan(c.at["c", "ipm"]) and c.at["c", "oc_lemma"] == 3.0
