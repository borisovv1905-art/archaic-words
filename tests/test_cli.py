import json

import numpy as np
import pandas as pd

from archword import annotate, baseline


def _raw(tmp_path, n=60, seed=0):
    rng = np.random.default_rng(seed)
    (tmp_path / "lexical_complexity").mkdir(parents=True)
    (tmp_path / "auxiliary").mkdir(parents=True)
    def code(i):  # уникальное «слово» из букв: номер в 26-ричной записи
        return "с" + chr(1072 + i // 26) + chr(1072 + i % 26) + "ъ"

    words = [code(i) for i in range(n)]
    freq_rows = [f"{w}\t{i + 1}\t{10 ** rng.uniform(-1, 3):.4f}\t1\tx\ty" for i, w in enumerate(words)]
    (tmp_path / "auxiliary" / "rus_freq_dictionary_1992-2019.csv").write_text(
        "СЛОВО\tРАНГ\tЧАСТОТА\tДОК\tЛЕММЫ\tPOS\n" + "\n".join(freq_rows), encoding="utf-8"
    )
    freq = {r.split("\t")[0]: float(r.split("\t")[2]) for r in freq_rows}
    cols = {"sentence": "а", "start": 0, "end": 1}
    for split, ws in (("train", words[:45]), ("test", words[45:])):
        df = pd.DataFrame({**cols, "token": ws, "complexity": [0.5 - 0.15 * np.log10(freq[w]) for w in ws]})
        df.to_csv(tmp_path / "lexical_complexity" / f"word_complexity_ru_{split}.tsv", sep="\t", index=False)
    return tmp_path


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
