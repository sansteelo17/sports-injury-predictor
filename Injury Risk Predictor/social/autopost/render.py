"""Render a card to a PNG by driving its own HTML.

Cards read their data from ``?data=<base64 JSON>`` and set
``html[data-rendered="1"]`` once painted. We build that URL against the local
card file, wait for the flag, and screenshot the ``#card`` element at 2x.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict

from . import config

CARD_FILES = {
    "matchday_board": "01_matchday_board.html",
}


def _payload_url(card_path: Path, payload: Dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    b64 = base64.b64encode(raw).decode("ascii")
    return f"file://{card_path}?data={b64}"


def render_card(post_type: str, payload: Dict, out_path: Path) -> Path:
    from playwright.sync_api import sync_playwright

    card_file = CARD_FILES.get(post_type)
    if not card_file:
        raise ValueError(f"No card wired for post_type={post_type}")
    card_path = (config.CARDS_DIR / card_file).resolve()
    if not card_path.exists():
        raise FileNotFoundError(card_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = _payload_url(card_path, payload)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=2)
        page.goto(url, wait_until="networkidle")
        # The card flips this once it has painted (or errored).
        page.wait_for_selector('html[data-rendered="1"]', timeout=15000)
        # Let fonts settle so the screenshot isn't a FOUT frame.
        page.wait_for_timeout(400)
        page.locator("#card").screenshot(path=str(out_path))
        browser.close()
    return out_path
