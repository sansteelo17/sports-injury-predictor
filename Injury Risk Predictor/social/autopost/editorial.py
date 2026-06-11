"""Editorial brain: pick the most story-worthy players for a format and bind
them into one narrative. Returns the board payload (post_type, matchday,
league, top_5, narrative_spine, should_post)."""
from __future__ import annotations

import json
from typing import Dict, List

from . import llm

SYSTEM = """You are Yara's editorial brain. You write for FPL managers and analysts who have seen everything. You think in narratives, not bullet points.

You receive a flat list of players with injury-risk model output and a post_type telling you which format to build. Pick the most story-worthy players for that format and bind them into one coherent narrative.

HARD RULES
- No exclamation marks. No em dashes or en dashes; use a comma or full stop.
- No tabloid words: BREAKING, URGENT, HUGE, MASSIVE, SHOCK.
- Every player entry MUST include risk_score_pct.
- MARQUEE FIRST. The board has to land with a casual fan, so strongly prefer recognisable, marquee names (stars and well-known internationals) over obscure squad players, even when an unknown name has a slightly higher risk number. Use club_team to judge profile (a star at a big club outranks a fringe player). Never fill the board with names a fan would not recognise.
- League scalability: if league is not "Premier League", do not invent FPL price/ownership; lean on risk, minutes, recency, injury news.
- If the input players array is empty, return should_post=false with empty top_5. Never invent players. player_name must match an input exactly.

OUTPUT - return ONLY this JSON, no prose, no markdown fence:
{
  "post_type": "matchday_board",
  "matchday": "string (e.g. MD1, R16, GW1)",
  "league": "string",
  "top_5": [
    {
      "player_name": "string (must match an input exactly)",
      "team": "string",
      "position": "string",
      "risk_score_pct": 0-100 integer,
      "archetype": "fragile | managed | robust | monitor",
      "signal_one_liner": "<= 90 chars, no '!', no dash, the one reason this risk is live",
      "delta_pct": signed integer or null
    }
  ],
  "narrative_spine": "1-2 sentence thesis tying the players together",
  "should_post": true
}

SELF-CHECK: top_5 length is 5 (or should_post=false); no fabricated names; no '!', no dash anywhere."""


def _fallback(candidates: List[Dict], matchday: str, league: str) -> Dict:
    top = candidates[:5]
    for c in top:
        if not c.get("signal_one_liner"):
            news = (c.get("injury_news") or "").strip()
            c["signal_one_liner"] = (news[:88] if news else
                                     f"{c.get('risk_level','')} risk into this round").strip()
        c.setdefault("delta_pct", None)
    return {
        "post_type": "matchday_board",
        "matchday": matchday,
        "league": league,
        "top_5": top,
        "narrative_spine": "The five players carrying the most injury risk into this round.",
        "should_post": len(top) > 0,
    }


def select(candidates: List[Dict], matchday: str, league: str) -> Dict:
    if not candidates:
        return {"post_type": "matchday_board", "matchday": matchday, "league": league,
                "top_5": [], "should_post": False}
    user = json.dumps({
        "trigger_type": "cron",
        "post_type": "matchday_board",
        "matchday": matchday,
        "league": league,
        "players_json": candidates,
    }, ensure_ascii=False)
    try:
        out = llm.chat_json(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            max_tokens=1600,
        )
        if isinstance(out, dict) and out.get("top_5"):
            # Guard: keep only players that exist in the input (no fabrication).
            names = {str(c["player_name"]).lower() for c in candidates}
            out["top_5"] = [p for p in out["top_5"]
                            if str(p.get("player_name", "")).lower() in names][:5]
            out.setdefault("matchday", matchday)
            out.setdefault("league", league)
            out.setdefault("post_type", "matchday_board")
            out["should_post"] = bool(out["top_5"])
            if out["top_5"]:
                return out
    except Exception as e:
        print(f"[editorial] LLM failed, using fallback: {e}")
    return _fallback(candidates, matchday, league)
