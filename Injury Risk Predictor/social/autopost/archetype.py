"""Archetype explainer builder: feature one archetype, explain it in Yara's
vocabulary, and show this week's marquee examples from the live board."""
from __future__ import annotations

import datetime as dt
from typing import Dict, Optional

from . import fetch

# The model's archetype vocabulary -> a plain-language read + its triggers.
ARCHETYPES = {
    "Currently Vulnerable": {
        "desc": "A short-term state, not a label. A player who is usually durable "
                "but is sitting in the top tier of recent load. They cool off in two "
                "to three weeks. They also break down far more often before that.",
        "triggers": ["A sharp spike in recent workload",
                     "Heavy minutes packed into a short window",
                     "Form holding while the body quietly runs hot"],
    },
    "Fragile": {
        "desc": "A body that breaks down often, and recently. High recurrence, short "
                "gaps between problems, the kind of profile where every hard sprint "
                "asks a question.",
        "triggers": ["A long list of past injuries",
                     "Short gaps between one problem and the next",
                     "A recent issue still close in the rear-view"],
    },
    "Injury Prone": {
        "desc": "The pattern, not the moment. A career-long history of missing games, "
                "the sort of record managers learn to manage around rather than hope away.",
        "triggers": ["A heavy career injury count",
                     "Real days lost, season after season",
                     "A history that repeats more than it resolves"],
    },
    "Recurring Issues": {
        "desc": "The same problem keeps coming back. Same area, again and again, "
                "until the body has a memory you cannot coach out of it.",
        "triggers": ["The same body area flagged repeatedly",
                     "A recurrence rate above the norm",
                     "Setbacks that echo old ones"],
    },
    "Moderate Risk": {
        "desc": "Neither bulletproof nor fragile. Enough history to watch, not enough "
                "to panic. The honest middle.",
        "triggers": ["A modest injury history", "No single alarming pattern",
                     "Risk that moves with the schedule"],
    },
    "Durable": {
        "desc": "Plays through almost everything. Low history, low recent load, the "
                "kind of availability a manager builds around.",
        "triggers": ["A short injury record", "Minutes managed well",
                     "A body that keeps cooperating"],
    },
    "Clean Record": {
        "desc": "Almost never misses. The body cooperates, the availability is a given, "
                "and that is its own kind of value.",
        "triggers": ["Few or no past injuries", "Long stretches without a setback",
                     "Availability you can plan around"],
    },
}


def build(competition: str = "world-cup-2026") -> Optional[Dict]:
    cands = fetch.fetch_candidates(competition, pool=80)

    def group(min_caps: int) -> Dict:
        out: Dict = {}
        for c in cands:
            if (c.get("caps") or 0) < min_caps:
                continue
            out.setdefault(c.get("archetype"), []).append(c)
        return out

    by_arch = group(10)
    eligible = [a for a in ARCHETYPES if len(by_arch.get(a, [])) >= 3]
    if not eligible:  # relax the likely-starter filter if needed
        by_arch = group(0)
        eligible = [a for a in ARCHETYPES if len(by_arch.get(a, [])) >= 3]
    if not eligible:
        return None

    # Rotate by day so the vocabulary varies across the tournament.
    idx = dt.datetime.now(dt.timezone.utc).timetuple().tm_yday % len(eligible)
    arch = eligible[idx]
    info = ARCHETYPES[arch]
    examples = sorted(by_arch[arch], key=lambda c: c.get("risk_score_pct", 0), reverse=True)[:3]
    return {"archetype": arch, "description": info["desc"],
            "triggers": info["triggers"], "examples": examples}
