"""Derive the World Cup stage label for a date from the fixtures, so the
matchday label switches itself: Group Stage -> Round of 32 -> Round of 16 ->
Quarter-Final -> Semi-Final -> Final. No manual switching.
"""
from __future__ import annotations

import pandas as pd

from . import config

_FIXTURES = config.ROOT / "data" / "processed" / "world_cup_2026_fixtures.pkl"

_LABELS = {
    "GROUP_STAGE": "Group Stage",
    "LAST_32": "Round of 32",
    "LAST_16": "Round of 16",
    "QUARTER_FINALS": "Quarter-Final",
    "SEMI_FINALS": "Semi-Final",
    "THIRD_PLACE": "Third-Place Play-off",
    "FINAL": "Final",
}
# Most-advanced wins when a transition day carries two stages (e.g. the day the
# group stage ends and the Round of 32 begins).
_ORDER = ["GROUP_STAGE", "LAST_32", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS",
          "THIRD_PLACE", "FINAL"]


def label(stage: str) -> str:
    """Friendly label for a raw stage code (LAST_32 -> Round of 32)."""
    return _LABELS.get(str(stage), str(stage).replace("_", " ").title() or "World Cup")


def stage_label_for(date_str: str) -> str:
    """Friendly stage label for the fixtures on ``date_str`` (YYYY-MM-DD). If no
    games that day, use the next upcoming fixture's stage. Falls back to
    'World Cup'."""
    try:
        fx = pd.read_pickle(_FIXTURES).dropna(subset=["utc_date"]).copy()
        fx["utc_date"] = pd.to_datetime(fx["utc_date"], utc=True)
        fx["day"] = fx["utc_date"].dt.strftime("%Y-%m-%d")
        day = fx[fx["day"] == date_str]
        if not day.empty:
            present = [s for s in _ORDER if s in set(day["stage"])]
            stage = present[-1] if present else day.iloc[0]["stage"]
        else:
            now = pd.Timestamp(date_str, tz="UTC")
            up = fx[fx["utc_date"] >= now].sort_values("utc_date")
            if up.empty:
                return "World Cup"
            stage = up.iloc[0]["stage"]
        return _LABELS.get(str(stage), "World Cup")
    except Exception:
        return "World Cup"
