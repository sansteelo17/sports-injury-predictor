"""Config + env loading for the Yara auto-post pipeline.

Everything is env-driven so the same code runs locally (cron) or anywhere else.
Loads the repo .env for local convenience.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CARDS_DIR = ROOT / "social" / "cards"
OUT_DIR = ROOT / "social" / "out"


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()


def get(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


# Data API
API_BASE = get("YARASPEAKS_API_BASE", "https://api.yaraspeaks.com").rstrip("/")

# LLM (OpenAI-compatible). Reuses the project's OPENAI_* keys.
OPENAI_API_KEY = get("OPENAI_API_KEY")
OPENAI_BASE_URL = get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
# Editorial/copy quality matters; default to the club narrative model.
AUTOPOST_MODEL = get("AUTOPOST_MODEL", get("OPENAI_MODEL", "gpt-5.4-mini"))

# Email (Gmail SMTP). GMAIL_APP_PASSWORD is a Google App Password (not your
# normal password): myaccount.google.com -> Security -> App passwords.
GMAIL_USER = get("GMAIL_USER", "gakpovwovwo@gmail.com")
GMAIL_APP_PASSWORD = get("GMAIL_APP_PASSWORD")
DRAFT_EMAIL_TO = get("DRAFT_EMAIL_TO", GMAIL_USER)
DRAFT_EMAIL_FROM = get("DRAFT_EMAIL_FROM", GMAIL_USER)
