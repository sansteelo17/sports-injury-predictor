"""Lineup reaction: when a confirmed starting XI contradicts a Yara flag (a
high-risk player the manager starts anyway), surface the manager-vs-model moment.

Lineups come from API-Football (api-sports.io; your key is on the free plan).
They publish ~30 minutes before kickoff, so until a match's XI is out there is
nothing to react to and the format produces no post, which is correct.
"""
from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional

import requests

from . import config

_BASE = "https://v3.football.api-sports.io"
_WC_LEAGUE = 1     # FIFA World Cup
_SEASON = 2026


def _norm(s) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s or "").lower())
                if not unicodedata.combining(c))
    return s.strip()


def _hdr() -> Dict:
    return {"x-apisports-key": config.API_FOOTBALL_KEY}


def _fixtures_on(date: str) -> List[Dict]:
    try:
        r = requests.get(f"{_BASE}/fixtures",
                         params={"league": _WC_LEAGUE, "season": _SEASON, "date": date},
                         headers=_hdr(), timeout=20)
        return r.json().get("response", []) if r.ok else []
    except Exception:
        return []


def _lineup_for(fixture_id: int) -> Dict[str, List[str]]:
    try:
        r = requests.get(f"{_BASE}/fixtures/lineups", params={"fixture": fixture_id},
                         headers=_hdr(), timeout=20)
        out: Dict[str, List[str]] = {}
        for team_lu in (r.json().get("response", []) if r.ok else []):
            names = [p.get("player", {}).get("name") for p in team_lu.get("startXI", [])]
            out[_norm(team_lu.get("team", {}).get("name"))] = [n for n in names if n]
        return out
    except Exception:
        return {}


def published_lineups(date: str) -> Dict[str, List[str]]:
    """Teams with a published starting XI for the day -> {normalised team: [XI names]}."""
    xis: Dict[str, List[str]] = {}
    for fx in _fixtures_on(date):
        xis.update(_lineup_for(fx.get("fixture", {}).get("id")))
    return xis


def _name_in_xi(player_name: str, xi: List[str]) -> bool:
    pn = _norm(player_name)
    last = pn.split()[-1] if pn.split() else pn
    for nm in xi:
        n = _norm(nm)
        if pn == n or (len(last) >= 4 and last in n):
            return True
    return False


def started_flags(candidates: List[Dict], xis: Dict[str, List[str]],
                  min_risk: int = 60) -> List[Dict]:
    """High-risk players the manager named in the confirmed XI anyway."""
    hits = []
    for c in candidates:
        if (c.get("risk_score_pct") or 0) < min_risk:
            continue
        xi = xis.get(_norm(c.get("team")))
        if xi and _name_in_xi(c.get("player_name", ""), xi):
            hits.append(c)
    hits.sort(key=lambda c: c.get("risk_score_pct", 0), reverse=True)
    return hits
