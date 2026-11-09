"""Собирает страницу проекта для GitHub Pages: README + ключевые цифры из results/ + ссылка на демо.

    python site/build.py            # результат: site/dist/index.html
Адрес демо берётся из переменной окружения DEMO_URL (если не задана, кнопка не показывается).
"""
import json
import os
from pathlib import Path

import markdown
from jinja2 import Template

ROOT = Path(__file__).resolve().parent.parent
PAGE = Template("""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>archword: какие слова классики пояснить</title>
<style>
body{font:16px/1.6 system-ui,sans-serif;max-width:860px;margin:0 auto;padding:24px 16px;color:#1d1d1f;background:#fff}
@media (prefers-color-scheme:dark){body{background:#16181c;color:#e8e8ea}a{color:#8ab4f8}th,td{border-color:#3a3d44!important}}
h1,h2,h3{line-height:1.25} table{border-collapse:collapse;margin:1em 0;display:block;overflow-x:auto}
th,td{border:1px solid #d0d0d5;padding:6px 10px;text-align:left} code{background:rgba(127,127,127,.15);padding:1px 4px;border-radius:4px}
pre{background:rgba(127,127,127,.12);padding:12px;overflow-x:auto;border-radius:6px}
.hero{padding:16px 0 8px}.btn{display:inline-block;background:#e8700a;color:#fff!important;padding:10px 18px;border-radius:8px;text-decoration:none;font-weight:600}
.facts{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}.fact{flex:1 1 150px;border:1px solid rgba(127,127,127,.35);border-radius:8px;padding:10px 14px}
.fact b{display:block;font-size:1.5em}
</style></head><body>
<div class="hero"><p>{% if demo_url %}<a class="btn" href="{{ demo_url }}">Открыть демонстрацию</a> {% endif %}
<a href="https://github.com/borisovv1905-art/archaic-words">Код на GitHub</a></p>
{% if facts %}<div class="facts">{% for value, label in facts %}<div class="fact"><b>{{ value }}</b>{{ label }}</div>{% endfor %}</div>{% endif %}</div>
{{ body|safe }}
</body></html>""")


def key_facts() -> list[tuple[str, str]]:
    facts = []
    try:
        meta = json.loads((ROOT / "models" / "model.json").read_text(encoding="utf-8"))["meta"]
        cv = meta["cv"]
        facts += [(str(meta["n_train"]), "размеченных слов"), (f"{cv['spearman']:.2f}", "Spearman (кросс-валидация)"),
                  (f"{cv['auc_label_ge2']:.2f}", "AUC: найти слова с оценкой ≥ 2")]
    except (OSError, KeyError, ValueError):
        pass
    return facts


def build(out_dir: Path | None = None) -> Path:
    out_dir = out_dir or ROOT / "site" / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    body = markdown.markdown((ROOT / "README.md").read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "sane_lists"])
    html = PAGE.render(body=body, facts=key_facts(), demo_url=os.environ.get("DEMO_URL", "").strip())
    target = out_dir / "index.html"
    target.write_text(html, encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build())
