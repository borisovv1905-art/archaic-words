import importlib.util
import os
from pathlib import Path

import markdown  # noqa: F401  (в dev-зависимостях)
import jinja2  # noqa: F401


def _load():
    spec = importlib.util.spec_from_file_location("site_build", Path(__file__).resolve().parent.parent / "site" / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_site_builds_with_and_without_demo_button(tmp_path):
    mod = _load()
    old = os.environ.get("DEMO_URL")
    try:
        os.environ["DEMO_URL"] = "https://example.com/demo"
        html = mod.build(tmp_path).read_text(encoding="utf-8")
        assert "archword" in html and 'href="https://example.com/demo"' in html and "<table>" in html
        os.environ.pop("DEMO_URL")
        assert "Открыть демонстрацию" not in mod.build(tmp_path).read_text(encoding="utf-8")
    finally:
        if old is not None:
            os.environ["DEMO_URL"] = old
