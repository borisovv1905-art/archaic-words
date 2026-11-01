from archword.text import iter_occurrences, normalize, split_sentences, tokenize


def test_normalize_yo_and_case():
    assert normalize("Ёлка") == "елка"


def test_tokenize_positions_and_hyphen():
    text = "Барышня-крестьянка ушла, 5 раз."
    toks = tokenize(text)
    assert [t.text for t in toks] == ["Барышня-крестьянка", "ушла", "раз"]
    assert all(text[t.start : t.end] == t.text for t in toks)


def test_split_sentences():
    s = split_sentences("Он вышел. Она осталась! «Кто там?» — спросил Адриян.")
    assert len(s) == 3


def test_maybe_name_flag():
    occ = list(iter_occurrences("Метель началась. Потом пришёл Адриян и ушёл."))
    flags = {t.text: name for t, _, name in occ}
    assert flags["Метель"] is False  # начало предложения — не признак имени
    assert flags["Адриян"] is True
    assert flags["пришёл"] is False
