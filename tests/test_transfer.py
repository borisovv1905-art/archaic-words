import numpy as np
import pandas as pd

from archword import annotate, transfer
from helpers import make_raw


def _labeled(tmp_path, words, freq, note_name=None):
    # чем реже слово, тем выше оценка: модели должны это уловить
    lf = np.log10([freq[w] for w in words])
    label = np.clip(np.round(3.5 - lf), 1, 5).astype(int)
    df = pd.DataFrame({"id": range(1, len(words) + 1), "word": words, "label": label, "note": ""})
    if note_name is not None:
        df.loc[note_name, "note"] = "имя"
    path = tmp_path / "labeled.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path, df


def test_load_labeled_drops_empty_and_names(tmp_path):
    raw, words, freq = make_raw(tmp_path / "raw")
    path, df = _labeled(tmp_path, words[:10], freq, note_name=2)
    df.loc[0, "label"] = np.nan
    df.to_csv(path, index=False, encoding="utf-8-sig")
    out = transfer.load_labeled(path)
    assert len(out) == 8  # пустая оценка и «имя» убраны
    assert out["label"].between(1, 5).all()


def test_load_labeled_rejects_bad_scale(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"id": [1, 2], "word": ["а", "б"], "label": [1, 7]}).to_csv(p, index=False)
    try:
        transfer.load_labeled(p)
    except ValueError as e:
        assert "1..5" in str(e) and "2" in str(e)
    else:
        raise AssertionError("ожидалась ошибка")


def test_evaluate_finds_signal(tmp_path):
    raw, words, freq = make_raw(tmp_path / "raw")
    path, _ = _labeled(tmp_path, words[:40], freq)
    rep = transfer.evaluate(raw, path, n_boot=200)
    m = rep["models"]["only_-log_freq"]
    assert rep["n_words"] == 40
    assert m["spearman"] > 0.9
    assert m["spearman_ci95"][0] <= m["spearman"] <= m["spearman_ci95"][1] + 1e-9
    assert m["auc_label_ge2"] > 0.9


def test_score_without_positives_gives_none():
    y = np.ones(20, int)
    y[0] = 2
    s = transfer.score(y, np.linspace(0, 1, 20), n_boot=50)
    assert s["auc_label_ge3"] is None


def test_annotate_max_ipm_and_exclude():
    tales = {"А": "Шлафорка лежала на стуле. Кибитка стояла у дома. Стол был большой. Землемер пришёл утром."}
    freq = {"стуле": 20.0, "дома": 60.0, "стол": 70.0, "большой": 120.0, "кибитка": 0.25, "землемер": 0.38,
            "лежала": 30.0, "стояла": 40.0, "пришел": 80.0, "утром": 50.0, "был": 900.0}
    rare = annotate.sample_sheet(tales, freq, n=10, max_ipm=2)
    assert set(rare["word"]) == {"шлафорка", "кибитка", "землемер"}  # слова не из словаря тоже считаются редкими
    again = annotate.sample_sheet(tales, freq, n=10, max_ipm=2, exclude={"Кибитка", "шлафорка"})
    assert set(again["word"]) == {"землемер"}


def test_load_labeled_skips_error_notes(tmp_path):
    p = tmp_path / "n.csv"
    pd.DataFrame({"id": [1, 2, 3], "word": ["а", "б", "в"], "label": [1, 3, 2],
                  "note": ["", "ошибка", "Имя"]}).to_csv(p, index=False)
    assert transfer.load_labeled(p)["word"].tolist() == ["а"]
