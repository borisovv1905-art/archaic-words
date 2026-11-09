"""Проверка готовой модели из models/ на настоящих файлах (пропускается, если models/ нет или нет pymorphy3)."""
from pathlib import Path

import pytest

MODELS = Path(__file__).resolve().parent.parent / "models"

pytestmark = pytest.mark.skipif(not (MODELS / "model.json").is_file(), reason="нет папки models/ (python -m archword.train_final)")


@pytest.fixture(scope="module")
def scorer():
    pytest.importorskip("pymorphy3")
    from archword.predict import Scorer
    return Scorer.load(MODELS)


def test_ratings_and_probabilities_are_in_range(scorer):
    r = scorer.score_words(["дом", "кибитка", "шлафорок", "ямщик", "понтировать"])
    assert ((r["rating"] >= 1) & (r["rating"] <= 5)).all()
    assert ((r["prob"] >= 0) & (r["prob"] <= 1)).all()


def test_rare_archaic_word_is_harder_than_common_word(scorer):
    r = scorer.score_words(["дом", "шлафорок"])
    assert r.at["шлафорок", "prob"] > r.at["дом", "prob"]


def test_annotate_covers_every_word_and_threshold_is_valid(scorer):
    occ = scorer.annotate("Ямщик сидел на облучке, а я дремал в кибитке.")
    assert len(occ) == 9 and occ["prob"].between(0, 1).all()
    assert 0.05 <= scorer.suggested_threshold <= 0.9
