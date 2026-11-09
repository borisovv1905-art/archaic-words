"""Оценка трудности слов в произвольном тексте: ядро демонстрации.

Модель - линейная (Ridge) по признакам «логарифм частоты формы, частоты леммы (OpenCorpora и pymorphy3), длина слова»
плюс символьные n-граммы слова; обучается в `archword.train_final` на всей собственной разметке. Здесь модель
хранится не как pickle, а как JSON с коэффициентами и считается на numpy: так приложению не нужен scikit-learn,
версии библиотек не конфликтуют, а файл можно прочитать глазами.

    scorer = Scorer.load("models")
    occurrences = scorer.annotate(text)             # для каждого слова текста: оценка 1..5
    html = highlight_html(text, occurrences, 2.0)   # подсветка слов не ниже порога

Оценка 1..5 - это шкала разметчика: «1 - знаю, 5 - совсем незнакомо». Модель предсказывает, как оценил бы слово читатель,
чья разметка использована в обучении (один человек), а не объективную трудность.
"""
from __future__ import annotations

import html
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FLOOR_IPM
from .lemma import LemmaFreq, load_totals, pymorphy_lemmatizer
from .text import WORD_RE, normalize

MIN_IPM = 0.02  # частоты ниже этого порога в таблицах не хранятся (слово получает FLOOR_IPM): так файл втрое меньше
NUM_FEATURES = ("log_form", "log_lemma", "log_lemma_pymorphy", "len")
FORM_FILE, LEMMA_FILE, MODEL_FILE = "form_freq.csv.gz", "lemma_freq.csv.gz", "model.json"
_QUOTES_AND_DASHES = set('«»"“”„—–-(')


# ---------------------------------------------------------------- признаки

def log_ipm(values) -> np.ndarray:
    """log10 частоты; пропуски и нули заменяются на FLOOR_IPM, как для слов, которых нет в словаре."""
    v = np.asarray(values, float)
    return np.log10(np.where(np.isfinite(v) & (v > 0), v, FLOOR_IPM))


def numeric_matrix(words: list[str], form_table: pd.DataFrame, lemma: LemmaFreq) -> np.ndarray:
    """Числовые признаки слов (уже нормализованных), в порядке NUM_FEATURES."""
    found = form_table.reindex(words)
    return np.column_stack([
        log_ipm(found["ipm"].to_numpy(float)),
        log_ipm(found["oc_lemma"].to_numpy(float)),
        log_ipm([lemma.get(w, FLOOR_IPM) for w in words]),
        np.array([len(w) for w in words], float),
    ])


def char_wb_ngrams(word: str, n_min: int = 2, n_max: int = 4) -> list[str]:
    """Символьные n-граммы с границами слова, как у TfidfVectorizer(analyzer='char_wb') для одного слова."""
    out = []
    for token in word.split():
        w = " " + token + " "
        for n in range(n_min, n_max + 1):
            offset = 0
            out.append(w[offset:offset + n])
            while offset + n < len(w):
                offset += 1
                out.append(w[offset:offset + n])
            if offset == 0:  # слово короче n: считается один раз
                break
    return out


class LinearModel:
    """Линейная модель из model.json: стандартизованные числовые признаки + tf-idf символьных n-грамм."""

    def __init__(self, spec: dict):
        self.mean = np.asarray(spec["num_mean"], float)
        self.scale = np.asarray(spec["num_scale"], float)
        self.vocab = {g: i for i, g in enumerate(spec["vocab"])}
        self.idf = np.asarray(spec["idf"], float)
        self.coef = np.asarray(spec["coef"], float)
        self.intercept = float(spec["intercept"])
        self.ngram_range = tuple(spec["ngram_range"])
        self.n_num = len(self.mean)

    def predict(self, words: list[str], numeric: np.ndarray) -> np.ndarray:
        out = self.intercept + ((numeric - self.mean) / self.scale) @ self.coef[: self.n_num]
        for i, w in enumerate(words):
            counts = Counter(g for g in char_wb_ngrams(w.lower(), *self.ngram_range) if g in self.vocab)
            if not counts:
                continue
            idx = np.array([self.vocab[g] for g in counts])
            vals = (1.0 + np.log(np.array(list(counts.values()), float))) * self.idf[idx]
            out[i] += float(vals @ self.coef[self.n_num + idx] / np.linalg.norm(vals))
        return out


# ---------------------------------------------------------------- оценщик

def to_rating(pred) -> np.ndarray:
    """Предсказание в шкале 0..1 -> оценка 1..5 (за пределы шкалы не выходит)."""
    return 1.0 + 4.0 * np.clip(np.asarray(pred, float), 0.0, 1.0)


def to_prob(pred, calibration: dict | None) -> np.ndarray:
    """Предсказание модели -> вероятность того, что читатель поставит слову оценку не ниже 2 (слово стоит пояснить).

    Калибровка - логистическая кривая a*pred + b, подобранная в train_final на предсказаниях вне обучения.
    Без калибровки возвращается предсказание, обрезанное до 0..1.
    """
    pred = np.asarray(pred, float)
    if not calibration:
        return np.clip(pred, 0.0, 1.0)
    return 1.0 / (1.0 + np.exp(-(calibration["a"] * pred + calibration["b"])))


def maybe_name(text: str, start: int) -> bool:
    """Слово с заглавной буквы не в начале предложения: вероятное имя (имена модель оценивает плохо)."""
    if not text[start].isupper():
        return False
    i = start - 1
    while i >= 0 and (text[i].isspace() or text[i] in _QUOTES_AND_DASHES):
        i -= 1
    return i >= 0 and text[i] not in ".!?…:"


class Scorer:
    def __init__(self, model: LinearModel, form_table: pd.DataFrame, lemma: LemmaFreq, meta: dict | None = None):
        self.model, self.form_table, self.lemma, self.meta = model, form_table, lemma, meta or {}

    @classmethod
    def load(cls, models_dir: str | Path = "models", lemmatizer=None) -> "Scorer":
        d = Path(models_dir)
        spec = json.loads((d / MODEL_FILE).read_text(encoding="utf-8"))
        forms = pd.read_csv(d / FORM_FILE, encoding="utf-8").drop_duplicates("w").set_index("w")
        lemma = LemmaFreq(load_totals(d / LEMMA_FILE), lemmatizer or pymorphy_lemmatizer())
        return cls(LinearModel(spec["model"]), forms, lemma, spec.get("meta", {}))

    @property
    def suggested_threshold(self) -> float:
        """Порог вероятности, при котором на разметке лучше всего находились слова с оценкой не ниже 2 (максимум F1)."""
        return float(self.meta.get("suggested_threshold", 0.3))

    def score_words(self, words) -> pd.DataFrame:
        """Оценка 1..5 и вероятность «слово стоит пояснить» для каждого уникального слова (индекс - нормализованное слово)."""
        uniq = sorted({normalize(w) for w in words})
        if not uniq:
            return pd.DataFrame(columns=["lemma", "rating", "prob"])
        pred = self.model.predict(uniq, numeric_matrix(uniq, self.form_table, self.lemma))
        return pd.DataFrame({"lemma": [self.lemma.lemmatize(w) for w in uniq], "rating": to_rating(pred),
                             "prob": to_prob(pred, self.meta.get("calibration"))}, index=uniq)

    def annotate(self, text: str, skip_names: bool = True) -> pd.DataFrame:
        """Каждое слово текста: позиция, лемма, оценка 1..5 и вероятность (для вероятных имён NaN, если skip_names)."""
        tokens = [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text)]
        scores = self.score_words([t for t, _, _ in tokens])
        rows = []
        for token, start, end in tokens:
            word = normalize(token)
            name = maybe_name(text, start)
            skip = name and skip_names
            rating = np.nan if skip else float(scores.at[word, "rating"])
            prob = np.nan if skip else float(scores.at[word, "prob"])
            rows.append((token, start, end, word, scores.at[word, "lemma"], rating, prob, name))
        return pd.DataFrame(rows, columns=["token", "start", "end", "word", "lemma", "rating", "prob", "is_name"])


# ---------------------------------------------------------------- вывод

def highlight_html(text: str, occurrences: pd.DataFrame, threshold: float, column: str = "prob") -> str:
    """Текст с подсветкой слов, у которых значение в колонке column не ниже порога: чем выше, тем ярче фон (для prob 0..1)."""
    def esc(s: str) -> str:
        return html.escape(s).replace("\n", "<br>")

    parts, pos = [], 0
    for r in occurrences.itertuples():
        parts.append(esc(text[pos:r.start]))
        piece = esc(text[r.start:r.end])
        value = getattr(r, column)
        if not math.isnan(value) and value >= threshold:
            alpha = 0.25 + 0.65 * min(1.0, max(0.0, value))
            title = html.escape(f"{r.lemma}: вероятность {r.prob:.0%}, оценка {r.rating:.1f}", quote=True)
            piece = f'<mark title="{title}" style="background: rgba(255, 140, 0, {alpha:.2f}); padding: 0 2px; border-radius: 3px;">{piece}</mark>'
        parts.append(piece)
        pos = r.end
    parts.append(esc(text[pos:]))
    return "".join(parts)


def summary_table(occurrences: pd.DataFrame, threshold: float, column: str = "prob") -> pd.DataFrame:
    """Слова не ниже порога: по одной строке на слово, по убыванию вероятности."""
    g = occurrences.dropna(subset=[column])
    g = g[g[column] >= threshold]
    if g.empty:
        return pd.DataFrame(columns=["слово", "лемма", "вероятность", "оценка", "раз"])
    out = g.groupby("word").agg(слово=("token", "first"), лемма=("lemma", "first"), вероятность=("prob", "first"),
                                оценка=("rating", "first"), раз=("word", "size"))
    return out.sort_values(["вероятность", "раз"], ascending=False).reset_index(drop=True)
