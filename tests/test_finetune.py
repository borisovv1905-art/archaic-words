import numpy as np
import pandas as pd

from archword import finetune as ft


def _own(words, labels):
    return pd.DataFrame({"word": words, "context": [f"Он сказал {w} тихо." for w in words], "label": labels, "y": ft.label_to_unit(labels)})


def _bible(n=6):
    return pd.DataFrame({"word": [f"б{i}" for i in range(n)], "context": ["Бог сказал."] * n, "y": np.linspace(0, .8, n)})


def test_label_to_unit_maps_scale():
    assert ft.label_to_unit([1, 3, 5]).tolist() == [0.0, 0.5, 1.0]


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


def test_own_examples_needs_context():
    assert _raises(lambda: ft.own_examples(pd.DataFrame({"word": ["а"], "label": [1]})), ValueError)


def test_own_examples_drops_rows_without_context():
    sheet = pd.DataFrame({"word": ["а", "б"], "context": ["фраза", None], "label": [1, 5]})
    out = ft.own_examples(sheet)
    assert out["word"].tolist() == ["а"] and out["y"].tolist() == [0.0]


def test_drop_overlap_removes_test_words_with_yo_and_case():
    train = _own(["Ещё", "дом", "кот"], [1, 1, 2])
    test = _own(["еще", "лес"], [3, 3])
    assert ft.drop_overlap(train, test)["word"].tolist() == ["дом", "кот"]


def test_stages_for_each_setting():
    b, o = _bible(), _own(["а", "б"], [1, 5])
    cfg = ft.Config(epochs=2, own_epochs=5)
    assert [e for _, e in ft.stages_for("bible", b, o, cfg)] == [2]
    assert [e for _, e in ft.stages_for("own", b, o, cfg)] == [5]
    assert [e for _, e in ft.stages_for("bible_then_own", b, o, cfg)] == [2, 5]
    mixed = ft.stages_for("mixed", b, o, cfg)
    assert len(mixed) == 1 and len(mixed[0][0]) == len(b) + len(o)
    assert _raises(lambda: ft.stages_for("что-то", b, o, cfg), ValueError)


def test_run_uses_trainer_per_seed_and_no_leak():
    rng = np.random.default_rng(0)
    test = _own([f"т{i}" for i in range(40)], rng.integers(1, 6, 40))
    own = pd.concat([_own([f"с{i}" for i in range(20)], rng.integers(1, 6, 20)), _own(["т0", "т1"], [1, 1])], ignore_index=True)
    calls = []

    def fake(stages, evals, cfg):
        calls.append((len(stages), cfg.seed))
        for df, _ in stages:   # слов теста в обучении быть не должно
            assert not set(df["word"]) & {"т0", "т1"} or df is stages[0][0]
        # «модель»: предсказание = истинный y теста + шум, шум зависит от сида
        noise = np.random.default_rng(cfg.seed).normal(0, .1, len(evals["test"]))
        out = {"test": evals["test"]["y"].to_numpy() + noise}
        if "bible_test" in evals:
            out["bible_test"] = evals["bible_test"]["y"].to_numpy()
        return out

    rep, table = ft.run(_bible(), own, test, _bible(8), settings=("bible", "own"), seeds=(0, 1), trainer=fake, n_boot=10)
    assert rep["n_own_train"] == 20 and len(calls) == 4
    assert rep["settings"]["own"]["spearman"] > 0.8
    assert len(rep["settings"]["own"]["spearman_per_seed"]) == 2
    assert rep["settings"]["bible"]["public_test_spearman"] == 1.0
    assert {"pred_bible", "pred_own"} <= set(table.columns) and len(table) == 40
