from archword.corpus import clean, decode_best, split_tales

SAMPLE = """ПОВЕСТИ БЕЛКИНА
Оглавление
Выстрел
Метель
Гробовщик

ВЫСТРЕЛ
Мы стояли в местечке ***. Жизнь армейского офицера известна.

МЕТЕЛЬ
Лошади тронулись. <b>Метель</b> усиливалась с каждой минутой.

Гробовщик
Последние пожитки гробовщика были взвалены на похоронные дроги.
"""
TITLES = ["Выстрел", "Метель", "Гробовщик"]


def test_decode_best_picks_koi8r():
    raw = "Метель усиливалась".encode("koi8-r")
    assert decode_best(raw) == "Метель усиливалась"


def test_decode_best_utf8():
    assert decode_best("Выстрел".encode("utf-8")) == "Выстрел"


def test_clean_strips_tags():
    assert clean("<b>Метель</b>\r\n\n\n\nтекст") == "Метель\n\nтекст"


def test_split_ignores_table_of_contents():
    tales = split_tales(clean(SAMPLE), TITLES)
    assert list(tales) == TITLES
    assert tales["Выстрел"].startswith("Мы стояли")
    assert "усиливалась" in tales["Метель"] and "<b>" not in tales["Метель"]
    assert tales["Гробовщик"].startswith("Последние пожитки")


def test_split_reports_missing_title():
    try:
        split_tales("ВЫСТРЕЛ\nтекст", ["Выстрел", "Метель"])
    except ValueError as e:
        assert "Метель" in str(e)
    else:
        raise AssertionError("ожидалась ошибка")
