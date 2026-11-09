import pandas as pd

from archword import ensemble as en


def _t(col, vals):
    return pd.DataFrame({"word": list("абвг"), "context": ["ф"] * 4, "label": [1, 1, 3, 4], "round": ["r"] * 4, "fold": [0, 0, 1, 1], col: vals})


def test_rank_average_combines_two_scales():
    out = en.rank_average(_t("a", [0.1, 0.2, 0.3, 0.4]), "a", _t("b", [100, 200, 400, 300]), "b")
    assert out["pred_ensemble"].tolist() == [1.0, 2.0, 3.5, 3.5]


def test_rank_average_rejects_misaligned_tables():
    b = _t("b", [1, 2, 3, 4]).iloc[::-1].reset_index(drop=True)
    try:
        en.rank_average(_t("a", [1, 2, 3, 4]), "a", b, "b")
    except ValueError:
        return
    raise AssertionError("ожидалась ошибка")
