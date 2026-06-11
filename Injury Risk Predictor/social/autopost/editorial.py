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
- MARQUEE FIRST (name-value tiebreak). The board sells the day, so among players who carry real risk, strongly prefer the recognisable, marquee names (stars, well-known internationals, players at big clubs) over obscure squad players. Use club_team to judge profile. A marquee name at 55% should make the board over a fringe name at 62%, the few points of risk are not worth a name nobody knows. Aim for a board where at least three names are instantly recognisable; never fill it with players a casual fan would not know.
- LIKELY TO PLAY. Do not feature players who will not start. Use caps (international caps) and club_minutes as the signal: a player with under 5 caps is fringe squad cover and almost certainly will not start, leave him out unless his risk is genuinely exceptional. A regular (30+ caps) who plays week in week out for his club is the kind of name worth a post. Featuring a benchwarmer reads as not knowing the squad.
- POSITION SPREAD. Do not stack the board with forwards. Strikers can take a slight priority because risk concentrates there, but a credible board spans the pitch: include defenders, midfielders, and a goalkeeper when they carry risk. Marquee across the team, not just the attack.
- USE SEASON FORM. club_goals, club_assists and club_minutes are real. Let them sharpen the read (a striker with almost no goals this season, or a player short of minutes, is part of the story), but never invent numbers not given.
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

SELF-CHECK: top_5 length matches the requested count (or should_post=false); no fabricated names; no '!', no dash anywhere."""


def _fallback(candidates: List[Dict], matchday: str, league: str,
              count: int = 5, post_type: str = "matchday_board") -> Dict:
    top = candidates[:count]
    for c in top:
        if not c.get("signal_one_liner"):
            news = (c.get("injury_news") or "").strip()
            c["signal_one_liner"] = (news[:88] if news else
                                     f"{c.get('risk_level','')} risk into this round").strip()
        c.setdefault("delta_pct", None)
    return {
        "post_type": post_type,
        "matchday": matchday,
        "league": league,
        "top_5": top,
        "narrative_spine": f"The {len(top)} players carrying the most injury risk.",
        "should_post": len(top) > 0,
    }


def select(candidates: List[Dict], matchday: str, league: str,
           count: int = 5, post_type: str = "matchday_board") -> Dict:
    if not candidates:
        return {"post_type": post_type, "matchday": matchday, "league": league,
                "top_5": [], "should_post": False}
    user = json.dumps({
        "trigger_type": "cron",
        "post_type": post_type,
        "matchday": matchday,
        "league": league,
        "instruction": f"Pick EXACTLY {count} players for this format, highest risk first.",
        "players_json": candidates,
    }, ensure_ascii=False)
    try:
        out = llm.chat_json(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            max_tokens=2200,
        )
        if isinstance(out, dict) and out.get("top_5"):
            # Guard: keep only players that exist in the input (no fabrication).
            names = {str(c["player_name"]).lower() for c in candidates}
            out["top_5"] = [p for p in out["top_5"]
                            if str(p.get("player_name", "")).lower() in names][:count]
            out.setdefault("matchday", matchday)
            out.setdefault("league", league)
            out["post_type"] = post_type
            out["should_post"] = bool(out["top_5"])
            if out["top_5"]:
                return out
    except Exception as e:
        print(f"[editorial] LLM failed, using fallback: {e}")
    return _fallback(candidates, matchday, league, count, post_type)
