import numpy as np
import pandas as pd

from archword import annotate, reliability


def _labeled(n=60):
    rows = []
    for i in range(n):
        label = 1 if i % 3 else 2 + (i % 4)           # около трети слов с оценкой 2..5
        rows.append({"id": i + 1, "word": f"слово{i}", "tale": "А", "context": f"Это слово{i} в предложении.",
                     "ipm": 0.5, "maybe_name": 0, "label": label, "note": ""})
    return pd.DataFrame(rows)


def _sheet(n=20):
    return pd.DataFrame({"id": range(1, n + 1), "word": [f"новое{i}" for i in range(n)], "tale": "Б",
                         "context": [f"Новое{i} слово." for i in range(n)], "ipm": 0.2, "maybe_name": 0,
                         "label": "", "note": ""})


def test_add_retest_counts_and_hides_old_labels():
    lab, sheet = _labeled(), _sheet()
    out = annotate.add_retest(sheet, lab, {}, n=10, seed=1)
    assert len(out) == len(sheet) + 10
    assert out["id"].tolist() == list(range(1, len(out) + 1))
    assert (out["label"] == "").all() and (out["note"] == "").all()   # старых оценок в таблице нет
    old_words = set(lab["word"])
    repeated = out[out["word"].isin(old_words)]
    assert len(repeated) == 10
    old = lab.set_index("word").loc[repeated["word"], "label"]
    assert (old >= 2).sum() == 5 and (old == 1).sum() == 5            # половина трудных, половина лёгких
    assert not out["word"].duplicated().any()


def test_add_retest_is_shuffled_and_reproducible():
    lab, sheet = _labeled(), _sheet()
    a = annotate.add_retest(sheet, lab, {}, n=10, seed=2)
    b = annotate.add_retest(sheet, lab, {}, n=10, seed=2)
    assert a["word"].tolist() == b["word"].tolist()
    first_new = [i for i, w in enumerate(a["word"]) if w.startswith("новое")]
    assert first_new != list(range(len(first_new)))                   # новые слова не идут блоком


def test_add_retest_skips_names_errors_and_words_already_in_sheet():
    lab, sheet = _labeled(), _sheet()
    lab.loc[0, "note"] = "имя"
    lab.loc[1, "note"] = "ошибка"
    lab.loc[2, "word"] = "новое3"                                       # уже есть в новой таблице
    out = annotate.add_retest(sheet, lab, {}, n=len(lab), seed=0)
    assert "слово0" not in set(out["word"]) and "слово1" not in set(out["word"])
    assert list(out["word"]).count("новое3") == 1


def test_add_retest_takes_what_exists_when_pool_is_small():
    lab = _labeled(6)
    out = annotate.add_retest(_sheet(5), lab, {}, n=40, seed=0)
    assert len(out) == 5 + len(lab)


def test_sample_sheet_skips_censored_fragments():
    tales = {"А": "Они поехали жить в ***ское поместье. Кибитка стояла у крыльца."}
    out = annotate.sample_sheet(tales, {}, n=10)
    assert "ское" not in set(out["word"])
    assert "кибитка" in set(out["word"])


def test_reliability_perfect_and_disagreement():
    old = pd.DataFrame({"word": list("абвгдежзик"), "label": [1, 1, 2, 3, 4, 5, 1, 2, 3, 5]})
    same = reliability.compare(old, old.copy(), n_boot=50)
    assert same["exact_agreement"] == 1.0 and same["qwk"] == 1.0 and same["spearman"] == 1.0
    new = old.copy()
    new["label"] = [1, 2, 2, 3, 4, 4, 1, 3, 3, 5]                       # три слова сдвинулись на единицу
    r = reliability.compare(old, new, n_boot=50)
    assert r["exact_agreement"] == 0.7 and r["within_one"] == 1.0
    assert 0.8 < r["qwk"] < 1.0
    assert r["binary_ge3"]["agreement"] == 0.9
    assert r["n_words"] == 10


def test_reliability_uses_only_repeated_words_and_normalizes_yo():
    old = pd.DataFrame({"word": ["ёлка", "дом", "лес"], "label": [1, 3, 5]})
    new = pd.DataFrame({"word": ["елка", "дом", "новое"], "label": [1, 3, 2]})
    r = reliability.compare(old, new, n_boot=20)
    assert r["n_words"] == 2                                            # «ёлка» и «елка» - одно слово


def test_reliability_without_repeated_words_gives_clear_error():
    try:
        reliability.compare(pd.DataFrame({"word": ["а"], "label": [1]}), pd.DataFrame({"word": ["б"], "label": [1]}))
    except ValueError as e:
        assert "повторно" in str(e)
    else:
        raise AssertionError("ожидалась ошибка")


def test_cli_blind_sheet_with_retest(tmp_path):
    import json
    from helpers import make_raw
    raw, words, freq = make_raw(tmp_path / "raw")
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"А": " ".join(f"Тут есть {w} и дом." for w in words[:40])}, ensure_ascii=False),
                      encoding="utf-8")
    old = pd.DataFrame({"id": [1, 2, 3, 4], "word": words[40:44], "tale": "А", "context": "Контекст.", "ipm": 1.0,
                        "maybe_name": 0, "label": [1, 2, 1, 5], "note": ""})
    old_csv = tmp_path / "old.csv"
    old.to_csv(old_csv, index=False, encoding="utf-8-sig")
    out = tmp_path / "sheet.csv"
    annotate.main(["--corpus", str(corpus), "--raw", str(raw), "--n", "10", "--seed", "1",
                   "--retest-from", str(old_csv), "--retest-n", "4", "--blind", "--out", str(out)])
    sheet = pd.read_csv(out, encoding="utf-8-sig")
    assert list(sheet.columns) == ["id", "word", "tale", "context", "label", "note"]
    assert len(sheet) == 10 + 4
    assert set(words[40:44]) <= set(sheet["word"])
    assert sheet["label"].isna().all()


def test_reliability_reports_agreement_without_remembered_words():
    old = pd.DataFrame({"word": list("абвгдежзик"), "label": [1, 1, 2, 3, 4, 5, 1, 2, 3, 5]})
    new = old.copy()
    new["label"] = [1, 1, 2, 3, 4, 5, 1, 2, 5, 5]                        # одно слово расходится
    new["note"] = ["повтор", "", "", "", "", "", "", "", "Повтор", ""]
    r = reliability.compare(old, new, n_boot=20)
    assert r["marked_as_remembered"] == 2 and r["n_words"] == 10          # строки остаются в расчёте
    assert r["without_remembered"]["n_words"] == 8
    assert r["without_remembered"]["exact_agreement"] == 1.0              # расходившееся слово было «помнимым»
    assert r["exact_agreement"] == 0.9
