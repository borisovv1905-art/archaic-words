"""Токенизация русского текста без внешних зависимостей."""
from __future__ import annotations

import re
from typing import Iterator, NamedTuple

WORD_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
_SENT_END = re.compile(r"(?<=[.!?…])[\"»”)]*\s+(?=[\"«“(—–-]?\s*[А-ЯЁA-Z])")


class Token(NamedTuple):
    text: str
    start: int
    end: int


def normalize(word: str) -> str:
    """Нижний регистр и ё -> е: так слова сопоставляются со словарями."""
    return word.lower().replace("ё", "е")


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def split_sentences(text: str) -> list[str]:
    """Грубое деление на предложения: достаточно для показа контекста слова."""
    parts = [p.strip() for p in _SENT_END.split(" ".join(text.split()))]
    return [p for p in parts if p]


def iter_occurrences(text: str) -> Iterator[tuple[Token, str, bool]]:
    """Для каждого слова: токен, предложение, в котором оно стоит, и признак
    «слово с заглавной буквы не в начале предложения» (вероятное имя).

    Ограничение: имя в начале предложения этим способом не отличить от обычного
    слова. Такие случаи разметчик помечает в колонке note."""
    for sentence in split_sentences(text):
        for i, tok in enumerate(tokenize(sentence)):
            first = i == 0
            maybe_name = tok.text[0].isupper() and not first
            yield tok, sentence, maybe_name
