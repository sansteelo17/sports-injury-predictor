"""Copywriter: the X post + Reddit long-form for a matchday board, in Yara's
real voice (reuses YARA_SYSTEM_PROMPT so social matches the site, no drift).
The board image already lists the players and numbers, so this is the caption,
the take, not a readout. Punctuation cleanup runs after because the model still
slips a dash or a bang in occasionally."""
from __future__ import annotations

import json
import re
from typing import Dict, Tuple

from src.inference.llm_client import YARA_SYSTEM_PROMPT

from . import llm

# Yara's voice bible + the rules specific to a board caption on X / Reddit.
SOCIAL_RULES = """

SOCIAL FORMAT (read this carefully):
You are posting the day's injury-risk board for {league}, {matchday}. The IMAGE
already shows every player and their risk number, so you are writing the take,
not a readout. Never say "matchday board", "top 5", "here are", or "swipe".

Write about the players given, in the order of who matters, and find the thread
that ties the day together (a fixture, a stage, a pattern of tired legs).

X POST (hard constraints):
- ONE post, 270 characters maximum. Count them. Brevity is the whole craft here.
- Open in your voice (an aphorism, a fixture image, or the standout), land the
  headline name with their number, then glance at one or two others. Do NOT list
  all of them. A number must appear.
- No markdown, no hashtags, no emojis, no em dashes, no quotation marks.

REDDIT POST:
- 3 short paragraphs, same voice, room to breathe. Carry the thread across the
  group and reconcile the numbers with what you would actually do. Bold names
  with **double asterisks**. No em dashes.

Return ONLY: {{"x_post": "...", "reddit_post": "..."}}"""


def _clean(text: str) -> str:
    text = re.sub(r"\s*[—–―]\s*", ", ", text)
    text = re.sub(r",\s*,", ",", text)
    text = text.replace("!", ".")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def write(payload: Dict) -> Tuple[str, str]:
    top5 = payload.get("top_5", [])
    spine = payload.get("narrative_spine", "")
    system = YARA_SYSTEM_PROMPT + SOCIAL_RULES.format(
        league=payload.get("league", ""), matchday=payload.get("matchday", ""))
    # Hand her the material: name, side, position, risk, and the one-line hook.
    players = [{
        "name": p.get("player_name"),
        "team": p.get("team"),
        "position": p.get("position"),
        "risk_pct": p.get("risk_score_pct"),
        "archetype": p.get("archetype"),
        "hook": p.get("signal_one_liner"),
    } for p in top5]
    user = ("Write the board's X post and Reddit post for these players, "
            "highest risk first.\n"
            f"THREAD: {spine}\n"
            f"PLAYERS: {json.dumps(players, ensure_ascii=False)}")
    try:
        out = llm.chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=1400,
        )
        x_post = _clean(str(out.get("x_post", "")).replace("**", ""))
        reddit = _clean(str(out.get("reddit_post", "")))
    except Exception as e:
        print(f"[copywriter] LLM failed, using fallback: {e}")
        lead = top5[0] if top5 else {}
        x_post = _clean(
            f"Bodies keep score. {lead.get('player_name','')} sits highest at "
            f"{lead.get('risk_score_pct','')}% going into {payload.get('matchday','')}. "
            "Full board: yaraspeaks.com")
        reddit = x_post
    if len(x_post) > 280:
        x_post = x_post[:277].rstrip() + "..."
    return x_post, reddit
