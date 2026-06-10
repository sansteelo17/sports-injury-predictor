"""Fetch the candidate pool for a competition from the lean board endpoint.

`/api/board-candidates` already returns the top risk-featured players with just
the fields a card needs (it skips the slow per-player image/shirt/FPL work), so
this is one cheap call even for the ~1200-row World Cup frame.
"""
from __future__ import annotations

from typing import Dict, List

import requests

from . import config


def fetch_candidates(competition: str, pool: int = 30) -> List[Dict]:
    url = f"{config.API_BASE}/api/board-candidates"
    resp = requests.get(url, params={"competition": competition, "limit": pool}, timeout=60)
    resp.raise_for_status()
    rows = resp.json() or []
    return [{
        "player_name": p.get("name"),
        "team": p.get("team"),
        "position": p.get("position"),
        "risk_score_pct": p.get("risk_score_pct"),
        "risk_level": p.get("risk_level"),
        "archetype": p.get("archetype"),
        "days_since_last_injury": p.get("days_since_last_injury"),
        "injury_news": p.get("injury_news"),
    } for p in rows]
