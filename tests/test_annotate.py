import pandas as pd

from archword.annotate import COLUMNS, collect, sample_sheet

TALES = {
    "Метель": "Шлафорка лежала на стуле. Пришёл Адриян поздно. Кибитка стояла у дома. "
    "Стол был большой. Дом был большой. Он был дома. Понтировать он не умел.",
    "Выстрел": "Кибитка уехала. Тогда Адриян молчал. Землемер пришёл утром. Дом стоял.",
}
FREQ = {"стол": 70.0, "дом": 170.0, "был": 900.0, "большой": 120.0, "он": 5000.0, "кибитка": 0.25,
        "землемер": 0.38, "стоял": 50.0, "молчал": 20.0, "пришёл": 80.0, "дома": 60.0}


def test_collect_unique_and_names():
    df = collect(TALES).set_index("word")
    assert df.loc["кибитка", "count"] == 2
    assert bool(df.loc["адриян", "maybe_name"]) is True
    assert bool(df.loc["шлафорка", "maybe_name"]) is False  # начало предложения


def test_collect_drops_short_words():
    words = set(collect({"x": "Он и я ушли."})["word"])
    assert words == {"ушли"}


def test_sample_sheet_shape_and_columns():
    sheet = sample_sheet(TALES, FREQ, n=10, seed=1)
    assert list(sheet.columns) == COLUMNS
    assert sheet["id"].tolist() == list(range(1, len(sheet) + 1))
    assert (sheet["label"] == "").all()
    assert "адриян" not in set(sheet["word"])  # имена по умолчанию исключены
    assert sheet["word"].is_unique


def test_sample_sheet_is_reproducible():
    a = sample_sheet(TALES, FREQ, n=6, seed=3)
    b = sample_sheet(TALES, FREQ, n=6, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_sample_sheet_covers_rare_and_common():
    sheet = sample_sheet(TALES, FREQ, n=20, seed=0)
    ipm = sheet["ipm"]
    assert ipm.min() == 0.0 or ipm.min() < 1  # есть редкие слова, в том числе вне словаря
    assert ipm.max() > 50  # и частотные


def test_empty_corpus():
    assert sample_sheet({}, FREQ).empty
