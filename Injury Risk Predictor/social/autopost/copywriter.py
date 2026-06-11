"""Copywriter: X post + Reddit long-form about exactly the players in the card,
in Yara's first-person voice. Followed by a deterministic punctuation cleanup
because the model breaks the no-dash/no-bang rules ~15% of the time."""
from __future__ import annotations

import json
import re
from typing import Dict, Tuple

from . import llm

SYSTEM = """You are writing the X post and Reddit long-form for an image that shows EXACTLY these players, in this order: {top_5_json}. Do not introduce anyone not in that list. Lead with the highest-risk player, name at least two others, mirror the narrative spine: "{spine}".

VOICE (Yara is the model, first person):
- First person, model-aware: "I had him at 82." Never "the model predicts."
- A number in every post: risk %, minutes, days, or odds.
- Sentence case. No exclamation marks. No emoji. No em or en dashes.
- Pushes back without sneering. Owns misses plainly. No "breaking / exclusive / must-see".

FORMAT
- X post <= 270 chars (single post, no thread).
- Reddit: 3 to 4 short paragraphs, bold names only with **double asterisks**.
- Output ONLY: {{"x_post": "...", "reddit_post": "..."}}"""


def _clean(text: str) -> str:
    text = re.sub(r"\s*[—–―]\s*", ", ", text)  # em/en/horizontal dash
    text = re.sub(r",\s*,", ",", text)
    text = text.replace("!", ".")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def write(payload: Dict) -> Tuple[str, str]:
    top5 = payload.get("top_5", [])
    spine = payload.get("narrative_spine", "")
    system = SYSTEM.format(top_5_json=json.dumps(top5, ensure_ascii=False), spine=spine)
    user = json.dumps({
        "league": payload.get("league"),
        "matchday": payload.get("matchday"),
        "players": top5,
        "narrative_spine": spine,
    }, ensure_ascii=False)
    try:
        out = llm.chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=1200,
        )
        # X has no markdown, so strip the bold the model adds; Reddit keeps it.
        x_post = _clean(str(out.get("x_post", "")).replace("**", ""))
        reddit = _clean(str(out.get("reddit_post", "")))
    except Exception as e:
        print(f"[copywriter] LLM failed, using fallback: {e}")
        lead = top5[0] if top5 else {}
        x_post = _clean(
            f"My matchday risk board. I have {lead.get('player_name','')} top at "
            f"{lead.get('risk_score_pct','')}%. Full read: yaraspeaks.com"
        )
        reddit = x_post
    if len(x_post) > 280:
        x_post = x_post[:277].rstrip() + "..."
    return x_post, reddit
