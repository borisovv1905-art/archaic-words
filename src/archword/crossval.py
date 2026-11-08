"""Кросс-валидация по словам на всей собственной разметке (раунды 1-3).

Один тест на 196 словах даёт широкие интервалы. Здесь вся разметка (около 700 слов) делится на
k частей так, что одно слово (с учётом ё/е и регистра) целиком попадает в одну часть. Модель учится на
остальных частях и предсказывает отложенную: каждое слово получает предсказание от модели, которая его не видела.
Из этих предсказаний считаются общие метрики (в них втрое больше слов, чем в одном тесте).

    python -m archword.crossval --raw data/raw --setting bible_then_own --seeds 1 --first-seed 0 \\
        --out results/crossval_bible_then_own_s0.json
    python -m archword.crossval --merge results/crossval_*_s0.predictions.csv results/crossval_*_s1.predictions.csv \\
        --setting bible_then_own --out results/crossval_bible_then_own.json

Для bible_then_own этап Библии обучается один раз на сид, а дальше для каждой части дообучается своя копия:
так прогон занимает не в k раз дольше. Гиперпараметры те же, что в `archword.finetune`, и по метрикам не подбираются.
Сравнение с частотными бейзлайнами - в `archword.compare` (нужен частотный словарь, в Colab его нет).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .finetune import DEFAULT_MODEL, Config, norm_word, own_examples, public_examples
from .transfer import load_labeled, score

SETTINGS = ("bible_then_own", "own")
DEFAULT_ROUNDS = (
    "data/annotation/round1_labeled.csv",
    "data/annotation/round2_labeled.csv",
    "data/annotation/round3_new_labeled.csv",
)
FOLD_SEED = 42  # разбиение на части не зависит от сида обучения: у всех запусков одни и те же части


def load_own(paths) -> pd.DataFrame:
    """Вся своя разметка одной таблицей; в колонке round - раунд (по имени файла), в key - нормализованное слово."""
    parts = []
    for i, p in enumerate(paths, 1):
        df = own_examples(load_labeled(p))
        stem = Path(p).stem.split("_")[0]
        df["round"] = stem if stem.startswith("round") else f"file{i}"
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["key"] = out["word"].map(norm_word)
    return out


def assign_folds(own: pd.DataFrame, k: int = 5, seed: int = FOLD_SEED) -> np.ndarray:
    """Номер части для каждой строки. Слово целиком в одной части; трудные слова (оценка не ниже 2) делятся поровну."""
    if k < 2:
        raise ValueError("Частей должно быть не меньше двух.")
    top = own.groupby("key")["label"].max()
    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    n = 0
    for hard in (True, False):
        words = sorted(top[(top >= 2) == hard].index)
        rng.shuffle(words)
        for w in words:
            fold_of[w] = n % k
            n += 1
    return own["key"].map(fold_of).to_numpy()


# ---------------------------------------------------------------- обучение (нужны torch и transformers)

class _Learner:
    """Модель с возможностью запомнить и восстановить веса. Цикл обучения тот же, что в finetune.train_and_predict."""

    def __init__(self, cfg: Config):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise SystemExit("Нужны torch и transformers: pip install torch transformers. "
                             "Проще запустить блокнот notebooks/crossval_colab.ipynb в Colab.") from exc
        self.torch, self.cfg = torch, cfg
        self.get_sched = get_linear_schedule_with_warmup
        self.reseed(cfg.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(cfg.model)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model, num_labels=1).to(self.device)

    def reseed(self, seed: int) -> None:
        self.torch.manual_seed(seed)
        np.random.seed(seed)

    def _batches(self, df: pd.DataFrame, shuffle: bool):
        idx = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
        for i in range(0, len(idx), self.cfg.batch_size):
            part = df.iloc[idx[i:i + self.cfg.batch_size]]
            enc = self.tok(part["word"].tolist(), part["context"].tolist(), truncation="only_second",
                           max_length=self.cfg.max_len, padding=True, return_tensors="pt").to(self.device)
            yield enc, part

    def fit(self, stage: pd.DataFrame, epochs: int) -> None:
        torch = self.torch
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=0.01)
        steps = epochs * int(np.ceil(len(stage) / self.cfg.batch_size))
        sched = self.get_sched(opt, int(0.1 * steps), steps)
        self.model.train()
        for _ in range(epochs):
            for enc, part in self._batches(stage, True):
                target = torch.tensor(part["y"].to_numpy(), dtype=torch.float32, device=self.device)
                pred = self.model(**enc).logits.squeeze(-1)
                loss = torch.nn.functional.mse_loss(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        with self.torch.no_grad():
            preds = [self.model(**enc).logits.squeeze(-1).float().cpu().numpy() for enc, _ in self._batches(df, False)]
        return np.concatenate(preds)

    def snapshot(self) -> dict:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def restore(self, state: dict) -> None:
        self.model.load_state_dict(state)


def train_cv(setting: str, bible: pd.DataFrame, own: pd.DataFrame, folds: np.ndarray, cfg: Config) -> np.ndarray:
    """Предсказания «вне обучения» для всех строк own: каждая часть предсказывается моделью, которая её не видела."""
    learner = _Learner(cfg)
    if setting == "bible_then_own":
        learner.fit(bible, cfg.epochs)
    base = learner.snapshot()
    oof = np.full(len(own), np.nan)
    for f in sorted(set(folds.tolist())):
        learner.restore(base)
        learner.reseed(cfg.seed * 1000 + int(f))
        test = folds == f
        learner.fit(own[~test].reset_index(drop=True), cfg.own_epochs)
        oof[test] = learner.predict(own[test].reset_index(drop=True))
        print(f"сид {cfg.seed}, часть {f}: готово", flush=True)
    return oof


# ---------------------------------------------------------------- прогон и отчёт

CvTrainer = Callable[[str, pd.DataFrame, pd.DataFrame, np.ndarray, Config], np.ndarray]


def seed_columns(table: pd.DataFrame, setting: str) -> list[str]:
    return [c for c in table.columns if c.startswith(f"pred_{setting}_s")]


def summarize(table: pd.DataFrame, setting: str, n_boot: int = 2000) -> dict:
    """Общие метрики по предсказаниям вне обучения (среднее по сидам), по частям и по раундам."""
    cols = seed_columns(table, setting)
    if not cols:
        raise ValueError(f"В таблице нет колонок pred_{setting}_s*")
    y = table["label"].to_numpy()
    mean = table[cols].mean(axis=1).to_numpy()
    report = {"setting": setting, "n": int(len(table)), "n_folds": int(table["fold"].nunique()),
              "seeds": [c.rsplit("_s", 1)[1] for c in cols],
              "pooled": score(y, mean, n_boot=n_boot),
              "per_seed_spearman": [round(float(score(y, table[c].to_numpy(), n_boot=0)["spearman"]), 3) for c in cols],
              "per_fold": {}, "per_round": {}}
    for f, g in table.groupby("fold"):
        try:
            report["per_fold"][str(int(f))] = {"n": int(len(g)), "spearman": score(g["label"].to_numpy(), g[cols].mean(axis=1).to_numpy(), n_boot=0)["spearman"]}
        except ValueError:
            pass
    if "round" in table:
        for r, g in table.groupby("round"):
            try:
                s = score(g["label"].to_numpy(), g[cols].mean(axis=1).to_numpy(), n_boot=0)
                report["per_round"][str(r)] = {"n": int(len(g)), "share_hard": round(float((g["label"] >= 2).mean()), 3),
                                               "spearman": s["spearman"], "auc_ge2": s.get("auc_label_ge2")}
            except ValueError:
                pass
    return report


def run_cv(bible: pd.DataFrame, own: pd.DataFrame, setting: str = "bible_then_own", k: int = 5, seeds=(0,),
           cfg: Config | None = None, trainer: CvTrainer = train_cv, n_boot: int = 2000) -> tuple[dict, pd.DataFrame]:
    if setting not in SETTINGS:
        raise ValueError(f"неизвестная постановка: {setting}; доступны {SETTINGS}")
    cfg = cfg or Config()
    folds = assign_folds(own, k)
    table = own[["word", "context", "label", "round"]].copy()
    table["fold"] = folds
    for seed in seeds:
        c = Config(**{**asdict(cfg), "seed": int(seed)})
        oof = np.asarray(trainer(setting, bible, own, folds, c), float)
        if len(oof) != len(own) or np.isnan(oof).any():
            raise ValueError("Тренер вернул предсказания не для всех слов.")
        table[f"pred_{setting}_s{int(seed)}"] = oof
    table[f"pred_{setting}"] = table[seed_columns(table, setting)].mean(axis=1)
    report = summarize(table, setting, n_boot)
    report["config"] = asdict(cfg)
    report["n_bible_train"] = int(len(bible))
    return report, table


def merge_tables(paths) -> pd.DataFrame:
    """Склеивает таблицы отдельных запусков (по сидам). Строки и разбиение на части должны совпадать."""
    tables = [pd.read_csv(p, encoding="utf-8-sig") for p in paths]
    base = tables[0]
    keys = ["word", "context", "label", "round", "fold"]
    out = base[keys].copy()
    for p, t in zip(paths, tables):
        if len(t) != len(base) or not t[keys].equals(base[keys]):
            raise ValueError(f"{p}: слова или разбиение на части отличаются от первого файла.")
        for c in t.columns:
            if "_s" in c and c.startswith("pred_"):
                if c in out:
                    raise ValueError(f"Колонка {c} встречается в нескольких файлах (один и тот же сид).")
                out[c] = t[c]
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--own", action="append", default=None, help="размеченные таблицы (повторяемый ключ)")
    ap.add_argument("--setting", choices=SETTINGS, default="bible_then_own")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=1, help="сколько сидов обучить в этом запуске")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--own-epochs", type=int, default=Config.own_epochs)
    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true", help="быстрая проверка (около минуты): крошечные выборки, одна эпоха; цифры не имеют смысла")
    ap.add_argument("--merge", nargs="+", help="не обучать, а склеить таблицы предсказаний от разных запусков")
    ap.add_argument("--out", default="results/crossval_bible_then_own.json")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.merge:
        table = merge_tables(a.merge)
        table[f"pred_{a.setting}"] = table[seed_columns(table, a.setting)].mean(axis=1)
        report = summarize(table, a.setting, a.boot)
    else:
        own = load_own(a.own or DEFAULT_ROUNDS)
        cfg = Config(model=a.model, epochs=a.epochs, own_epochs=a.own_epochs, lr=a.lr)
        bible = public_examples(a.raw, "train") if a.setting == "bible_then_own" else pd.DataFrame(columns=["word", "context", "y"])
        k = a.k
        if a.smoke:
            bible, own = bible.head(64), own.sample(100, random_state=0).reset_index(drop=True)
            cfg, k = Config(model=a.model, epochs=1, own_epochs=1, lr=a.lr), 2
        report, table = run_cv(bible, own, a.setting, k, tuple(range(a.first_seed, a.first_seed + a.seeds)), cfg, n_boot=a.boot)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    table.to_csv(out.with_suffix(".predictions.csv"), index=False, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
