"""Корпус «Повестей Белкина»: загрузка с lib.ru и деление на повести.

    python -m archword.corpus --out data/corpus/belkin.json

Тексты не хранятся в репозитории: скрипт скачивает их при запуске.
Сеть нужна только этому шагу; остальной код работает с готовым JSON.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LIBRU_URL = "http://lib.ru/LITRA/PUSHKIN/belkin.txt"
TALES = ["Выстрел", "Метель", "Гробовщик", "Станционный смотритель", "Барышня-крестьянка"]
_CYR = re.compile(r"[А-Яа-яЁё]")
_TAGS = re.compile(r"<[^>]+>")


def decode_best(raw: bytes, encodings=("utf-8", "koi8-r", "cp1251")) -> str:
    """Выбирает кодировку, при которой доля кириллицы в тексте наибольшая."""
    best, best_share = None, -1.0
    for enc in encodings:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        share = len(_CYR.findall(text)) / max(len(text), 1)
        if share > best_share:
            best, best_share = text, share
    if best is None:
        raise ValueError("не удалось определить кодировку текста")
    return best


def clean(text: str) -> str:
    text = _TAGS.sub("", text).replace("\r", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_tales(text: str, titles=TALES) -> dict[str, str]:
    """Делит текст по строкам-заголовкам.

    Заголовок может встретиться дважды (в оглавлении и в тексте), поэтому для
    каждой повести берётся самый длинный фрагмент до следующего заголовка.
    """
    lines = text.split("\n")
    wanted = {re.sub(r"\W+", " ", t.lower()).strip(): t for t in titles}
    marks = []  # (номер строки, повесть)
    for i, line in enumerate(lines):
        key = re.sub(r"\W+", " ", line.lower()).strip()
        if key in wanted:
            marks.append((i, wanted[key]))
    best: dict[str, str] = {}
    for n, (i, title) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(lines)
        body = "\n".join(lines[i + 1 : end]).strip()
        if len(body) > len(best.get(title, "")):
            best[title] = body
    missing = [t for t in titles if t not in best]
    if missing:
        raise ValueError(f"в тексте не найдены заголовки: {missing}")
    return {t: best[t] for t in titles}


def fetch(url: str = LIBRU_URL, timeout: int = 60) -> dict[str, str]:
    import requests

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return split_tales(clean(decode_best(resp.content)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/corpus/belkin.json")
    ap.add_argument("--url", default=LIBRU_URL)
    args = ap.parse_args(argv)
    tales = fetch(args.url)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tales, ensure_ascii=False, indent=1), encoding="utf-8")
    for title, body in tales.items():
        print(f"{title}: {len(body.split())} слов")


if __name__ == "__main__":
    main()
