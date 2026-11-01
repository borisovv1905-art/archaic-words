import json

import pandas as pd

from archword import annotate, baseline
from helpers import make_raw


def _raw(tmp_path):
    return make_raw(tmp_path)[0]


def test_baseline_run_structure_and_signal(tmp_path):
    rep = baseline.run(_raw(tmp_path))
    assert rep["data"]["token_overlap_train_test"] == 0
    assert set(rep["models"]) >= {"mean", "freq_len", "freq_len_charngram"}
    # сложность задана как убывающая функция частоты: модель должна это найти
    assert rep["models"]["freq_len"]["spearman"] > 0.9
    assert rep["models"]["freq_len"]["rmse"] < rep["models"]["mean"]["rmse"]


def test_annotate_cli_writes_sheet(tmp_path):
    raw = _raw(tmp_path / "raw")
    corpus = tmp_path / "belkin.json"
    corpus.write_text(json.dumps({"Метель": "Шлафорка лежала на стуле. Кибитка стояла у дома. Слованн лежал."},
                                 ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out" / "sheet.csv"
    annotate.main(["--corpus", str(corpus), "--raw", str(raw), "--n", "5", "--out", str(out)])
    sheet = pd.read_csv(out, encoding="utf-8-sig")
    assert list(sheet.columns) == annotate.COLUMNS
    assert len(sheet) >= 3 and sheet["label"].isna().all()
