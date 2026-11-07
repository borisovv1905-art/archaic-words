"""Дообучение небольшой русской модели (rubert-tiny2) и проверка на разметке «Повестей Белкина».

Запуск (нужны torch и transformers, лучше на GPU, например в Colab, см. notebooks/finetune_colab.ipynb):

    python -m archword.finetune --raw data/raw \\
        --train-own data/annotation/round1_labeled.csv --train-own data/annotation/round2_labeled.csv \\
        --test data/annotation/round3_new_labeled.csv --out results/finetune_round3.json

Модель получает пару «слово, предложение» и выдаёт число. Сравниваются четыре постановки:

    bible           обучение только на публичной выборке (Библия, 0..1)
    own             обучение только на своей разметке (раунды 1-2, оценка 1..5, приведена к 0..1)
    bible_then_own  сначала Библия, потом дообучение на своей разметке
    mixed           Библия и своя разметка вместе в одной выборке

Проверка идёт на отдельной таблице (раунд 3), слов которой нет ни в обучении, ни в подборе настроек.
Гиперпараметры заданы заранее и не подбираются по тесту: тест трогается один раз.
Метрики те же, что в `archword.transfer`: Spearman и AUC для порогов 2 и 3 с бутстрэп-интервалами.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .data import load_complexity
from .transfer import load_labeled, score

SETTINGS = ("bible", "own", "bible_then_own", "mixed")
DEFAULT_MODEL = "cointegrated/rubert-tiny2"


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    epochs: int = 4
    own_epochs: int = 8
    lr: float = 5e-5
    batch_size: int = 16
    max_len: int = 128
    seed: int = 0


def norm_word(w: str) -> str:
    return str(w).lower().replace("ё", "е").strip()


def label_to_unit(label) -> np.ndarray:
    """Оценка читателя 1..5 -> 0..1."""
    return (np.asarray(label, float) - 1.0) / 4.0


def public_examples(raw: str | Path, split: str) -> pd.DataFrame:
    df = load_complexity(raw, split)
    return pd.DataFrame({"word": df["token"].astype(str), "context": df["sentence"].astype(str),
                         "y": df["complexity"].astype(float).to_numpy()})


def own_examples(sheet: pd.DataFrame) -> pd.DataFrame:
    """Своя разметка: контекст берётся из колонки context, цель приводится к 0..1."""
    if "context" not in sheet:
        raise ValueError("В размеченной таблице нет колонки context: без предложения модели нечего читать.")
    df = sheet.dropna(subset=["context"]).copy()
    return pd.DataFrame({"word": df["word"].astype(str).to_numpy(), "context": df["context"].astype(str).to_numpy(),
                         "y": label_to_unit(df["label"]), "label": df["label"].astype(int).to_numpy()})


def drop_overlap(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Убирает из обучения слова, которые есть в тесте (без утечки по слову)."""
    bad = {norm_word(w) for w in test["word"]}
    return train[~train["word"].map(norm_word).isin(bad)].reset_index(drop=True)


def stages_for(setting: str, bible: pd.DataFrame, own: pd.DataFrame, cfg: Config | None = None) -> list[tuple[pd.DataFrame, int]]:
    """Этапы обучения: пары (выборка, число эпох). Модель учится на этапах подряд.

    Своя разметка небольшая, поэтому на ней эпох больше, чем на публичной выборке.
    """
    cfg = cfg or Config()
    if setting == "bible":
        return [(bible, cfg.epochs)]
    if setting == "own":
        return [(own, cfg.own_epochs)]
    if setting == "bible_then_own":
        return [(bible, cfg.epochs), (own, cfg.own_epochs)]
    if setting == "mixed":
        both = pd.concat([bible[["word", "context", "y"]], own[["word", "context", "y"]]], ignore_index=True)
        return [(both, cfg.epochs)]
    raise ValueError(f"неизвестная постановка: {setting}; доступны {SETTINGS}")


# ---------------------------------------------------------------- обучение (нужны torch и transformers)

def train_and_predict(stages: list[tuple[pd.DataFrame, int]], evals: dict[str, pd.DataFrame], cfg: Config) -> dict[str, np.ndarray]:
    """Обучает модель по этапам и возвращает предсказания для каждой оценочной выборки."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise SystemExit("Нужны torch и transformers: pip install torch transformers. "
                         "Проще запустить блокнот notebooks/finetune_colab.ipynb в Colab.") from exc

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model, num_labels=1).to(device)

    def batches(df: pd.DataFrame, shuffle: bool):
        idx = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
        for i in range(0, len(idx), cfg.batch_size):
            part = df.iloc[idx[i:i + cfg.batch_size]]
            enc = tok(part["word"].tolist(), part["context"].tolist(), truncation="only_second",
                      max_length=cfg.max_len, padding=True, return_tensors="pt").to(device)
            yield enc, part

    for stage, epochs in stages:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
        steps = epochs * int(np.ceil(len(stage) / cfg.batch_size))
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        model.train()
        for _ in range(epochs):
            for enc, part in batches(stage, True):
                target = torch.tensor(part["y"].to_numpy(), dtype=torch.float32, device=device)
                pred = model(**enc).logits.squeeze(-1)
                loss = torch.nn.functional.mse_loss(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()

    model.eval()
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, df in evals.items():
            preds = [model(**enc).logits.squeeze(-1).float().cpu().numpy() for enc, _ in batches(df, False)]
            out[name] = np.concatenate(preds)
    return out


# ---------------------------------------------------------------- прогон и отчёт

Trainer = Callable[[list[tuple[pd.DataFrame, int]], dict[str, pd.DataFrame], Config], dict[str, np.ndarray]]


def run(bible: pd.DataFrame, own_train: pd.DataFrame, test: pd.DataFrame, bible_test: pd.DataFrame | None = None,
        settings=SETTINGS, seeds=(0, 1, 2), cfg: Config | None = None, trainer: Trainer = train_and_predict,
        n_boot: int = 2000) -> tuple[dict, pd.DataFrame]:
    """Прогоняет постановки и сиды. Возвращает отчёт и таблицу предсказаний для разбора ошибок.

    Оценка на тесте считается по среднему предсказанию сидов (ансамбль), отдельно хранятся
    Spearman каждого сида: так видно разброс из-за случайности обучения.
    """
    cfg = cfg or Config()
    own_train = drop_overlap(own_train, test)
    y = test["label"].to_numpy()
    evals = {"test": test}
    if bible_test is not None:
        evals["bible_test"] = bible_test
    report: dict = {"config": asdict(cfg), "n_test": int(len(test)), "n_own_train": int(len(own_train)),
                    "n_bible_train": int(len(bible)), "settings": {}}
    table = test[["word", "context", "label"]].copy() if "label" in test else test.copy()
    for setting in settings:
        per_seed: dict[str, list[np.ndarray]] = {k: [] for k in evals}
        for seed in seeds:
            c = Config(**{**asdict(cfg), "seed": int(seed)})
            preds = trainer(stages_for(setting, bible, own_train, c), evals, c)
            for k in evals:
                per_seed[k].append(np.asarray(preds[k], float))
        mean_test = np.mean(per_seed["test"], axis=0)
        res = score(y, mean_test, n_boot=n_boot)
        res["spearman_per_seed"] = [round(float(score(y, p, n_boot=0)["spearman"]), 3) for p in per_seed["test"]]
        if bible_test is not None:
            yt = bible_test["y"].to_numpy()
            res["public_test_spearman"] = round(float(score(yt, np.mean(per_seed["bible_test"], axis=0), n_boot=0)["spearman"]), 3)
        report["settings"][setting] = res
        table[f"pred_{setting}"] = mean_test
    return report, table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--train-own", action="append", default=None, help="размеченные раунды для обучения (повторяемый)")
    ap.add_argument("--test", default="data/annotation/round3_new_labeled.csv")
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=SETTINGS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--own-epochs", type=int, default=Config.own_epochs)
    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results/finetune_round3.json")
    args = ap.parse_args(argv)

    own_paths = args.train_own or ["data/annotation/round1_labeled.csv", "data/annotation/round2_labeled.csv"]
    own = pd.concat([own_examples(load_labeled(p)) for p in own_paths], ignore_index=True)
    test_sheet = load_labeled(args.test)
    test = own_examples(test_sheet)
    cfg = Config(model=args.model, epochs=args.epochs, own_epochs=args.own_epochs, lr=args.lr)
    report, table = run(public_examples(args.raw, "train"), own, test, public_examples(args.raw, "test"),
                        settings=args.settings, seeds=tuple(range(args.seeds)), cfg=cfg, n_boot=args.boot)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    table.to_csv(out.with_suffix(".predictions.csv"), index=False, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
