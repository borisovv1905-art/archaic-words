import numpy as np
import pandas as pd

from archword import classic_cv as cc


def _own(n=80):
    rng = np.random.default_rng(0)
    words = [f"слово{i}" + ("ийский" if i % 5 == 0 else "") for i in range(n)]
    label = np.where(np.arange(n) % 5 == 0, rng.integers(3, 6, n), 1)
    return pd.DataFrame({"word": words, "y": (label - 1) / 4, "label": label})


def test_numeric_features_have_log_columns_and_length():
    f = cc.numeric_features(["дом", "кот"], {"form": {"дом": 100.0, "кот": 10.0}})
    assert list(f.columns) == ["log_form", "len"] and f["log_form"].tolist() == [2.0, 1.0] and f["len"].tolist() == [3, 3]


def test_cross_predict_returns_prediction_for_every_row_and_learns_signal():
    own = _own()
    folds = np.arange(len(own)) % 4
    X = pd.concat([own[["word"]], pd.DataFrame({"log_form": np.where(own["label"] > 1, -1.0, 1.0), "len": own["word"].str.len()})], axis=1)
    for kind in ("numeric", "ngram"):
        oof = cc.cross_predict(own, folds, X, kind)
        assert not np.isnan(oof).any()
        assert oof[own["label"] > 1].mean() > oof[own["label"] == 1].mean()
