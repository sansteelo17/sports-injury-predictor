"""Battle card builder: pick the next marquee World Cup fixture, then the
highest-risk, likely-starting marquee player from each side.

Fixtures come from the local WC fixtures pickle (same source the posting
calendar uses); players + risk/caps come from the board-candidates API.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import pandas as pd

from . import config, fetch

_FIXTURES_PKL = config.ROOT / "data" / "processed" / "world_cup_2026_fixtures.pkl"

# Heuristic profile: a fixture with two of these is a marquee tie.
MARQUEE_NATIONS = {
    "Brazil", "Argentina", "France", "England", "Spain", "Germany", "Portugal",
    "Netherlands", "Belgium", "Croatia", "Uruguay", "Mexico", "United States",
    "Italy", "Colombia", "Morocco", "Japan", "Senegal",
}


def _upcoming() -> pd.DataFrame:
    df = pd.read_pickle(_FIXTURES_PKL).dropna(subset=["utc_date"]).copy()
    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True)
    now = dt.datetime.now(dt.timezone.utc)
    return df[df["utc_date"] >= now].sort_values("utc_date")


def pick_fixture() -> Optional[Dict]:
    up = _upcoming()
    if up.empty:
        return None
    up = up.copy()
    up["score"] = up.apply(
        lambda r: (r["home_team"] in MARQUEE_NATIONS) + (r["away_team"] in MARQUEE_NATIONS), axis=1)
    best = up.sort_values(["score", "utc_date"], ascending=[False, True]).iloc[0]
    return {
        "home": best["home_team"], "away": best["away_team"],
        "date": str(best["utc_date"])[:10], "utc": best["utc_date"].to_pydatetime(),
        "stage": str(best.get("stage") or "").replace("_", " ").title() or "World Cup",
    }


def _pick_player(cands: List[Dict], team: str) -> Optional[Dict]:
    team_c = [c for c in cands if c.get("team") == team]
    if not team_c:
        return None
    # cands are risk-ranked; prefer a capped (likely-starting) name, else the top.
    capped = [c for c in team_c if (c.get("caps") or 0) >= 10]
    return (capped or team_c)[0]


def _x_factor(p: Dict) -> str:
    ds = p.get("days_since_last_injury")
    if isinstance(ds, int) and ds < 30:
        return f"Only {ds} days since his last injury."
    g, m = p.get("club_goals"), p.get("club_minutes")
    if isinstance(g, int) and isinstance(m, int) and g == 0 and m > 500:
        return "No club goals this season, his sharpness is a real question."
    if isinstance(p.get("caps"), int) and p["caps"] >= 50:
        return f"{p['caps']} caps, a nailed-on starter carrying the load."
    return f"{p.get('risk_level', 'Real')} risk going into the tie."


def build() -> Optional[Dict]:
    fx = pick_fixture()
    if not fx:
        return None
    cands = fetch.fetch_candidates("world-cup-2026", pool=80, date=fx["date"])
    a = _pick_player(cands, fx["home"])
    b = _pick_player(cands, fx["away"])
    if not a or not b:
        return None
    for p in (a, b):
        p["x_factor"] = _x_factor(p)
    return {"fixture": fx, "left": a, "right": b}
