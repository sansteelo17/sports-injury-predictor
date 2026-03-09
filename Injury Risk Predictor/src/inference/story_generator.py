"""
Generate personalized injury risk stories for players.

Uses player data and risk factors to create human-readable narratives
that explain WHY a player has their specific risk level.
"""

import math
import os
import re
from typing import Dict, List, Optional

from .context_rag import build_dynamic_rag_line, retrieve_player_context
from .llm_client import generate_grounded_narrative


def _pl(n: int, word: str) -> str:
    """Pluralize: _pl(1, 'goal') -> '1 goal', _pl(2, 'goal') -> '2 goals'."""
    if n == 1:
        return f"{n} {word}"
    if word.endswith("y") and word[-2:] not in ("ay", "ey", "oy", "uy"):
        return f"{n} {word[:-1]}ies"
    return f"{n} {word}s"


POPULAR_NAME_OVERRIDES = {
    "mohamed salah": "Salah",
    "bruno fernandes": "Bruno",
    "heung-min son": "Son",
    "son heung-min": "Son",
    "bukayo saka": "Saka",
    "erling haaland": "Haaland",
    "alexander isak": "Isak",
    "martin odegaard": "Odegaard",
    "martin ødegaard": "Odegaard",
}

DISPLAY_TEAM_NAME_OVERRIDES = {
    "Manchester United": "Man Utd",
    "Manchester City": "Man City",
    "Tottenham": "Spurs",
    "Tottenham Hotspur": "Spurs",
    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham",
    "Brighton Hove": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Leeds United": "Leeds",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Nott'ham Forest": "Nott'm Forest",
}


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        result = float(value)
        if result != result or result == float("inf") or result == float("-inf"):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        if value is None:
            return default
        f = float(value)
        if f != f or f == float("inf") or f == float("-inf"):
            return default
        return int(f)
    except (TypeError, ValueError):
        return default


def _as_sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if text.endswith((".", "!", "?")):
        return text
    return f"{text}."


def _count_phrase(count: int, singular: str, plural: Optional[str] = None) -> str:
    label = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {label}"


def _days_fan_label(days: int) -> str:
    """Fan-readable duration label for long injury-free runs."""
    d = _safe_int(days, 0)
    if d >= 1095:
        return f"{round(d / 365, 1):g}+ years"
    if d >= 365:
        return f"{round(d / 365, 1):g} years"
    return f"{d} days"


def _format_goal_assist_recent(goals: int, assists: int, samples: int) -> str:
    """Build natural attacking form phrase without forcing zero-values."""
    g = _safe_int(goals, 0)
    a = _safe_int(assists, 0)
    n = _safe_int(samples, 0)
    if n <= 0:
        return ""
    if g > 0 and a > 0:
        return f"{_pl(g, 'goal')} and {_pl(a, 'assist')} in the last {n}"
    if g > 0:
        return f"{_pl(g, 'goal')} in the last {n}"
    if a > 0:
        return f"{_pl(a, 'assist')} in the last {n}"
    return f"No goals or assists in the last {n}"


def _format_h2h_output(goals: int, assists: int, samples: int, opponent: str) -> str:
    """Build natural head-to-head output phrase without zero-value clutter."""
    g = _safe_int(goals, 0)
    a = _safe_int(assists, 0)
    s = _safe_int(samples, 0)
    opp = _display_team_name(opponent)
    if s <= 0:
        return ""
    if g > 0 and a > 0:
        return f"{_pl(g, 'goal')} and {_pl(a, 'assist')} in {s} meetings with {opp}"
    if g > 0:
        return f"{_pl(g, 'goal')} in {s} meetings with {opp}"
    if a > 0:
        return f"{_pl(a, 'assist')} in {s} meetings with {opp}"
    return f"No goals or assists in {s} meetings with {opp}"


def _natural_conceding_line(opponent: str, avg_conceded: float) -> str:
    """Convert conceded-rate number into fan-readable football phrasing."""
    opp = _display_team_name(opponent)
    avg = _safe_float(avg_conceded, 0.0)
    if avg <= 0:
        return ""
    if avg < 0.8:
        return f"{opp} have been tight at the back lately"
    if avg < 1.35:
        return f"{opp} have been conceding about a goal a game lately"
    if avg < 1.8:
        return f"{opp} have been conceding around one and a half goals a game lately"
    if avg < 2.0:
        return f"{opp} have been conceding about two goals a game lately"
    return f"{opp} have been leaking over two goals a game lately"


def _display_team_name(team_name: Optional[str]) -> str:
    value = (team_name or "").strip()
    if not value:
        return "this opponent"
    mapped = DISPLAY_TEAM_NAME_OVERRIDES.get(value, value)
    if mapped.islower():
        mapped = mapped.title()
    return mapped


def _position_group(position: str) -> str:
    pos = (position or "").strip().lower()
    if not pos:
        return "other"
    # Check goalkeeper FIRST — they are not defenders
    if any(token in pos for token in ["goalkeeper", "keeper", "gk"]):
        return "goalkeeper"
    if any(token in pos for token in ["centre-back", "center-back", "full-back", "fullback",
                                       "wing-back", "wingback", "defender", "back"]):
        return "defender"
    if any(token in pos for token in ["forward", "fwd", "striker", "winger", "centre-forward",
                                       "center-forward"]):
        return "attacker"
    if any(token in pos for token in ["midfielder", "midfield", "cm", "dm", "am", "playmaker"]):
        return "midfielder"
    return "other"


def _call_name(player_data: Dict) -> str:
    """Return football-popular short name for narrative voice."""
    for key in ("story_name", "popular_name", "display_name", "web_name"):
        value = (player_data.get(key) or "").strip()
        if value:
            return value

    full_name = (player_data.get("name") or "").strip()
    if not full_name:
        return "This player"

    lowered = full_name.lower()
    if lowered in POPULAR_NAME_OVERRIDES:
        return POPULAR_NAME_OVERRIDES[lowered]

    parts = full_name.split()
    if len(parts) == 1:
        return parts[0]
    particles = {"van", "von", "de", "da", "di", "del", "der", "dos", "le", "la"}
    if len(parts) >= 2 and parts[-2].lower() in particles:
        return f"{parts[-2]} {parts[-1]}"
    return parts[-1]


def _first_chunk_text(chunks: List[Dict], kind: str) -> Optional[str]:
    for chunk in chunks:
        if chunk.get("kind") == kind:
            text = (chunk.get("text") or "").strip()
            if text:
                return text
    return None


def _extract_player_importance(extra_context: Optional[Dict]) -> Dict:
    """Safely pull player-importance context from narrative payload."""
    if not extra_context:
        return {}
    importance = extra_context.get("player_importance") or {}
    return importance if isinstance(importance, dict) else {}


def _build_importance_sentence(importance: Dict, first_name: str, risk_pct: int) -> Optional[str]:
    """Return one concise importance-aware sentence for story/FPL copy."""
    if not importance:
        return None

    tier = str(importance.get("tier", "") or "").strip()
    ownership = _safe_float(importance.get("ownership_pct", 0.0), 0.0)
    price = _safe_float(importance.get("price", 0.0), 0.0)
    price_tier = str(importance.get("price_tier", "") or "").strip()
    role_importance = str(importance.get("role_importance", "") or "").strip()
    form_signal = str(importance.get("form_signal", "") or "").strip()
    h2h_signal = str(importance.get("h2h_signal", "") or "").strip()

    parts: List[str] = []

    # Ownership context — the core of FPL importance
    if tier in {"Core", "High"} and ownership > 0:
        if risk_pct >= 45:
            parts.append(
                f"{ownership:.1f}% of managers own {first_name}, "
                f"so that {risk_pct}% risk can swing rank either way"
            )
        else:
            parts.append(
                f"{ownership:.1f}% ownership means fading {first_name} carries rank risk"
            )
    elif tier == "Differential" and ownership > 0:
        parts.append(
            f"Just {ownership:.1f}% ownership for {first_name}. "
            "Returns here would be pure differential upside"
        )

    # Price context for premium players
    if price >= 10.0 and not parts:
        parts.append(f"At £{price:.1f}m, {first_name} locks a big chunk of budget")

    # Role context with specific data if available
    if not parts and role_importance:
        role = role_importance.strip().lower()
        if role == "talisman":
            parts.append(f"{first_name} is the talisman for this side")
        elif role == "primary attacker":
            parts.append(f"{first_name} is the primary attacking outlet")
        elif role == "creative hub":
            parts.append(f"{first_name} is central to chance creation")
        elif role == "defensive anchor":
            parts.append(f"{first_name} anchors the defensive setup")
        else:
            parts.append(f"{first_name} profiles as a {role} in this context")

    if not parts:
        return None

    return ". ".join(parts) + "."


# ── OptaJoe voice helpers ───────────────────────────────────────────────────

_KICKER_POOLS = {
    "high_risk": ["Fragile.", "Alarming.", "Exposed.", "Volatile.", "Dangerous."],
    "moderate_risk": ["Watchful.", "Wary.", "Loaded.", "Teetering.", "Cautious."],
    "low_risk": ["Bankable.", "Steady.", "Reliable.", "Durable.", "Nailed."],
    "form_hot": ["Scorching.", "Relentless.", "Inevitable.", "Clinical.", "Unstoppable."],
    "form_cold": ["Barren.", "Stalled.", "Drifting.", "Fading.", "Quiet."],
    "premium": ["Elite.", "Locked.", "Essential.", "Cornerstone.", "Untouchable."],
    "avoid": ["Pass.", "Fade.", "Skip.", "Dead.", "Empty."],
    "default": ["Noted.", "Telling.", "Significant.", "Sharp.", "Clear."],
}

ENABLE_NARRATIVE_KICKERS = (
    (os.getenv("NARRATIVE_ENABLE_KICKERS", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
)


def _stat_lead(number, unit: str = "") -> str:
    """Format a stat-first lead: '30 - ' or '0.52 - '."""
    if isinstance(number, float):
        formatted = f"{number:.2f}" if number < 10 else f"{number:.0f}"
    else:
        formatted = str(number)
    sep = "" if unit.startswith("%") else " "
    return f"{formatted}. " if not unit else f"{formatted}{sep}{unit}. "


def _pick_kicker(category: str = "default", salt: str = "") -> str:
    """Deterministic one-word kicker from pool (disabled by default)."""
    if not ENABLE_NARRATIVE_KICKERS:
        return ""
    pool = _KICKER_POOLS.get(category, _KICKER_POOLS["default"])
    idx = abs(hash(salt or "x")) % len(pool)
    return pool[idx]


def _build_fpl_signal_stack(
    recent_form: Dict,
    vs_opponent: Dict,
    opponent_defense: Dict,
    fixture_history: Dict,
    is_home: Optional[bool],
    role: str,
    first_name: str,
    opponent: str,
) -> List[str]:
    """Rank the 3 strongest data signals as short clauses for FPL copy."""
    signals: List[tuple] = []  # (strength, clause)

    # Recent form signal
    r_goals = _safe_int(recent_form.get("goals", 0))
    r_assists = _safe_int(recent_form.get("assists", 0))
    r_samples = _safe_int(recent_form.get("samples", 0))
    r_returns = _safe_int(recent_form.get("returns", 0))
    r_cs = _safe_int(recent_form.get("clean_sheets", 0))
    r_gi = r_goals + r_assists

    if role == "defender" and r_cs >= 2 and r_samples > 0:
        signals.append((r_cs * 3, f"{r_cs} clean sheets in {r_samples}"))
    if r_gi >= 3 and r_samples > 0:
        signals.append((r_gi * 2.5, f"{r_gi} goal involvements in the last {r_samples}"))
    elif r_returns >= 2 and r_samples > 0:
        signals.append((r_returns * 2, f"{r_returns} returns in {r_samples}"))
    elif r_gi > 0 and r_samples > 0:
        signals.append((r_gi * 1.5, f"{_pl(r_goals, 'goal')} and {_pl(r_assists, 'assist')} in {r_samples}"))

    # H2H signal
    vs_goals = _safe_int(vs_opponent.get("goals", 0))
    vs_assists = _safe_int(vs_opponent.get("assists", 0))
    vs_samples = _safe_int(vs_opponent.get("samples", 0))
    vs_gi = vs_goals + vs_assists
    if vs_gi >= 2 and vs_samples >= 2:
        signals.append((vs_gi * 2.0, f"{vs_gi} in {vs_samples} H2H meetings with {opponent}"))
    elif vs_gi >= 1 and vs_samples >= 1:
        signals.append((vs_gi * 1.2, f"{_pl(vs_goals, 'goal')} in {_pl(vs_samples, 'meeting')} vs {opponent}"))

    # Opponent defense signal
    opp_conceded = _safe_float(opponent_defense.get("avg_goals_conceded_last5", 0))
    if opp_conceded >= 1.5:
        signals.append((opp_conceded * 1.8, f"{opponent} leaking around one and a half to two goals a game"))
    elif opp_conceded >= 1.0:
        signals.append((opp_conceded * 1.0, f"{opponent} conceding about a goal a game"))
    elif 0 < opp_conceded < 0.8:
        signals.append((0.5, f"{opponent} tight at the back lately"))

    # Fixture history signal
    fh_samples = _safe_int(fixture_history.get("samples", 0))
    fh_wins = _safe_int(fixture_history.get("wins", 0))
    if fh_samples >= 3 and fh_wins >= 2:
        signals.append((fh_wins * 1.3, f"{fh_wins} wins in {fh_samples} vs {opponent}"))

    # Venue signal
    if is_home is True:
        signals.append((1.0, f"home fixture"))
    elif is_home is False:
        signals.append((0.3, f"away trip"))

    # Sort by strength descending, return top 3 clauses
    signals.sort(key=lambda x: x[0], reverse=True)
    return [clause for _, clause in signals[:3]]


def generate_player_story(player_data: Dict, extra_context: Optional[Dict] = None) -> str:
    """Generate a personalized, journalistic risk story for a player."""
    name = player_data.get("name", "This player")
    first_name = _call_name(player_data)
    prev_injuries = _safe_int(player_data.get("previous_injuries", 0))
    days_lost = _safe_float(player_data.get("total_days_lost", 0))
    days_since = _safe_int(player_data.get("days_since_last_injury", 365), 365)
    archetype = player_data.get("archetype", "Unknown")
    prob = _safe_float(player_data.get("ensemble_prob", 0.5), 0.5)
    age = _safe_int(player_data.get("age", 25), 25)
    acwr = _safe_float(player_data.get("acwr", 1.0), 1.0)
    fatigue = _safe_float(player_data.get("fatigue_index", 0.0), 0.0)
    role = _position_group(str(player_data.get("position", "") or ""))
    goals = _safe_int(player_data.get("goals", 0), 0)
    assists = _safe_int(player_data.get("assists", 0), 0)
    avg_days = days_lost / prev_injuries if prev_injuries > 0 else 0
    risk_pct = round(prob * 100)

    extra_context = extra_context or {}
    player_importance = _extract_player_importance(extra_context)
    matchup_context = extra_context.get("matchup_context") or {}
    recent_form = matchup_context.get("recent_form") or {}
    recent_samples = _safe_int(recent_form.get("samples", 0), 0)
    recent_goals = _safe_int(recent_form.get("goals", 0), 0)
    recent_assists = _safe_int(recent_form.get("assists", 0), 0)
    recent_gi = recent_goals + recent_assists
    recent_clean_sheets = _safe_int(recent_form.get("clean_sheets", 0), 0)
    vs_opp = matchup_context.get("vs_opponent") or {}
    vs_samples = _safe_int(vs_opp.get("samples", 0), 0)
    vs_goals = _safe_int(vs_opp.get("goals", 0), 0)
    vs_assists = _safe_int(vs_opp.get("assists", 0), 0)
    opponent_defense = matchup_context.get("opponent_defense") or {}
    opp_conceded = _safe_float(opponent_defense.get("avg_goals_conceded_last5", 0.0), 0.0)
    next_fixture = extra_context.get("next_fixture") or {}
    opponent = _display_team_name(matchup_context.get("opponent") or next_fixture.get("opponent") or "")
    is_home = matchup_context.get("is_home")
    if is_home is None:
        is_home = next_fixture.get("is_home")

    # Per-injury detail records (sorted most recent first)
    injury_records = extra_context.get("injury_records") or []

    # Additional context for richer narratives
    matches_last_7 = _safe_int(player_data.get("matches_last_7", 0), 0)
    matches_last_14 = _safe_int(player_data.get("matches_last_14", 0), 0)
    matches_last_30 = _safe_int(player_data.get("matches_last_30", 0), 0)
    fixture_history = extra_context.get("fixture_history") or {}
    fixture_samples = _safe_int(fixture_history.get("samples", 0), 0)
    fh_wins = _safe_int(fixture_history.get("wins", 0), 0)
    fh_losses = _safe_int(fixture_history.get("losses", 0), 0)
    vs_cs = _safe_int(vs_opp.get("clean_sheets", 0), 0)

    # Sparse profile: player has thin injury data, so lean on performance/fixture context
    is_sparse_profile = (
        prev_injuries <= 2 and days_since >= 60 and acwr < 1.5 and 21 <= age <= 31
    )

    sentences = []
    # Track which topics the LEAD already covered so later sections skip them
    covered = set()  # e.g. {"recency", "history", "workload", "severity", "fixture"}

    # LEAD — natural first-person Yara voice, full sentences
    if prob >= 0.60 and days_since < 60:
        sentences.append(
            f"I have {first_name} at {risk_pct}% risk. Only {days_since} days back from the "
            f"last injury, which is not enough time for the body to fully trust itself again."
        )
        covered.update({"recency"})
    elif prob >= 0.60 and prev_injuries >= 5:
        sentences.append(
            f"{prev_injuries} injuries costing {_safe_int(days_lost)} days. That is the record "
            f"driving {first_name} to {risk_pct}% risk, and it is hard to argue with the volume."
        )
        covered.update({"history"})
    elif prob >= 0.60 and acwr > 1.5:
        sentences.append(
            f"Workload ratio sitting at {acwr:.2f} for {first_name}, which is above the threshold "
            f"where injury risk starts climbing. I have this at {risk_pct}%."
        )
        covered.update({"workload"})
    elif prob >= 0.60 and avg_days >= 30:
        sentences.append(
            f"When {first_name} gets hurt, it tends to be serious. {avg_days:.0f} days average "
            f"across {_pl(prev_injuries, 'injury')}, which is why the risk reads {risk_pct}%."
        )
        covered.update({"history", "severity"})
    elif prob >= 0.60 and matches_last_30 >= 5:
        sentences.append(
            f"{matches_last_30} matches in 30 days for {first_name}. The schedule alone "
            f"is enough to push risk to {risk_pct}% before you factor in anything else."
        )
        covered.update({"workload"})
    elif prob >= 0.60:
        sentences.append(
            f"I have {first_name} at {risk_pct}% risk. {_pl(prev_injuries, 'injury').capitalize()} "
            f"on record totalling {_safe_int(days_lost)} days lost."
        )
        covered.update({"history"})
    elif prob >= 0.40 and days_since < 60:
        sentences.append(
            f"{first_name} is {days_since} days post-injury, which puts this right "
            f"in the window where recurrence risk is highest. I have it at {risk_pct}%."
        )
        covered.update({"recency"})
    elif prob >= 0.40 and prev_injuries >= 4:
        sentences.append(
            f"{prev_injuries} career injuries for {first_name}. None of them individually "
            f"alarming, but that accumulation is what pushes the risk to {risk_pct}%."
        )
        covered.update({"history"})
    elif prob >= 0.40 and avg_days >= 30:
        sentences.append(
            f"When {first_name} picks up a knock, it tends to be costly. "
            f"{avg_days:.0f} days per layoff on average, and that pattern is reflected in the {risk_pct}% reading."
        )
        covered.update({"history", "severity"})
    elif prob >= 0.40:
        sentences.append(
            f"I have {first_name} at {risk_pct}% risk. {_pl(prev_injuries, 'injury').capitalize()} "
            f"and {_safe_int(days_lost)} days lost on the books."
        )
        covered.update({"history"})
    elif prob >= 0.25 and prev_injuries >= 3:
        sentences.append(
            f"{risk_pct}% risk for {first_name}. Manageable, "
            f"but {_pl(prev_injuries, 'injury')} on file keep the baseline honest."
        )
        covered.update({"history"})
    elif prob < 0.20 and days_since >= 365:
        sentences.append(
            f"Over {_days_fan_label(days_since)} without injury for {first_name}. "
            f"I have this at just {risk_pct}%, which makes this one of the safer profiles in the squad."
        )
        covered.update({"recency"})
    elif prob < 0.20:
        sentences.append(
            ("Clean injury record" if prev_injuries == 0 else f"{_pl(prev_injuries, 'injury').capitalize()} logged")
            + f" for {first_name} at {risk_pct}% risk. Nothing here to worry about."
        )
    else:
        sentences.append(
            f"{risk_pct}% risk for {first_name} with {_pl(prev_injuries, 'injury')} "
            f"and {_safe_int(days_lost)} days lost. Nothing that jumps off the page."
        )
        covered.update({"history"})

    # HISTORY — with specific injury detail when available
    # Analyze injury records for skew, recurring body areas, worst injury
    worst_record = None
    recurring_area = None
    severity_skewed = False
    skew_other_avg = 0.0
    if injury_records:
        severities = [r["severity_days"] for r in injury_records if r.get("severity_days", 0) > 0]
        if severities and len(severities) >= 2:
            max_sev = max(severities)
            others = [s for s in severities if s != max_sev]
            if others:
                others_avg = sum(others) / len(others)
                overall_avg = sum(severities) / len(severities)
                avg_lift = overall_avg - others_avg
                # Trigger outlier copy only when one injury meaningfully distorts the profile.
                if max_sev >= 70 and max_sev >= 2.3 * others_avg and avg_lift >= 18:
                    severity_skewed = True
                    skew_other_avg = others_avg
        # Worst injury
        for r in injury_records:
            if r.get("severity_days", 0) > 0:
                if worst_record is None or r["severity_days"] > worst_record["severity_days"]:
                    worst_record = r
        # Most common body area (excluding "unknown")
        areas = [r["body_area"] for r in injury_records if r.get("body_area") and r["body_area"] != "unknown"]
        if areas:
            from collections import Counter
            area_counts = Counter(areas)
            top_area, top_count = area_counts.most_common(1)[0]
            if top_count >= 2:
                recurring_area = top_area

    # HISTORY detail — only when LEAD didn't cover it, or when we have new info (skew, recurring area)
    if "history" not in covered:
        if prev_injuries == 0:
            sentences.append(
                "No injuries on file, which keeps the baseline risk low."
            )
        elif severity_skewed and worst_record and prev_injuries <= 4:
            worst_area = worst_record.get("injury_raw") or worst_record.get("body_area", "injury")
            worst_days = worst_record["severity_days"]
            worst_date = worst_record.get("date", "")
            date_str = f" in {worst_date[:7]}" if worst_date and len(worst_date) >= 7 else ""
            others_avg = round(skew_other_avg) if skew_other_avg > 0 else round((days_lost - worst_days) / max(prev_injuries - 1, 1))
            sentences.append(
                f"One long {worst_area}{date_str} layoff cost {worst_days} days and skews the whole profile. "
                f"Strip that out and the other {_pl(prev_injuries - 1, 'injury')} averaged just {others_avg} days."
            )
        elif recurring_area and prev_injuries >= 3:
            sentences.append(
                f"The {recurring_area.lower()} keeps coming back. {_pl(prev_injuries, 'injury').capitalize()} totalling "
                f"{_safe_int(days_lost)} days, and that specific area is the clear weak point."
            )
        elif worst_record and avg_days >= 40:
            worst_area = worst_record.get("injury_raw") or worst_record.get("body_area", "injury")
            worst_date = worst_record.get("date", "")
            date_str = f" ({worst_date[:7]})" if worst_date and len(worst_date) >= 7 else ""
            sentences.append(
                f"The worst was a {worst_area}{date_str} that kept {first_name} out for "
                f"{worst_record['severity_days']} days. Average layoff sits at {avg_days:.0f} days "
                f"across {_pl(prev_injuries, 'injury')}."
            )
        elif prev_injuries <= 2 and days_since >= 365:
            sentences.append(
                f"Only {_pl(prev_injuries, 'injury')} on record and over {_days_fan_label(days_since)} "
                f"since the last one. Availability has been strong."
            )
        elif prev_injuries <= 2:
            if worst_record:
                worst_area = worst_record.get("injury_raw") or worst_record.get("body_area", "injury")
                sentences.append(
                    f"Only {_pl(prev_injuries, 'injury')} on file. The notable one was a {worst_area} "
                    f"that cost {worst_record['severity_days']} days."
                )
            else:
                sentences.append(
                    f"Light injury history with just {_pl(prev_injuries, 'injury')} on record."
                )
        elif prev_injuries >= 8 and avg_days >= 30:
            sentences.append(
                f"{prev_injuries} injuries averaging {avg_days:.0f} days each. "
                f"Both the frequency and the severity are concerning."
            )
        elif prev_injuries >= 5:
            sentences.append(
                f"{prev_injuries} injuries totalling {_safe_int(days_lost)} days lost. "
                f"That volume is hard to ignore."
            )
        elif avg_days <= 15 and prev_injuries > 0:
            sentences.append(
                f"Past injuries have been minor, averaging just {avg_days:.0f} days out each time."
            )
    elif recurring_area and prev_injuries >= 3:
        sentences.append(
            f"Worth noting the {recurring_area.lower()} keeps recurring as the weak point."
        )
    elif severity_skewed and worst_record:
        worst_area = worst_record.get("injury_raw") or worst_record.get("body_area", "injury")
        worst_days = worst_record["severity_days"]
        others_avg = round(skew_other_avg) if skew_other_avg > 0 else round((days_lost - worst_days) / max(prev_injuries - 1, 1))
        sentences.append(
            f"One {worst_area} layoff ({worst_days} days) skews the numbers. The rest averaged {others_avg} days."
        )

    # RECENCY & WORKLOAD
    if "recency" not in covered:
        if days_since < 30:
            sentences.append(
                f"Only {days_since} days since the last injury, which puts {first_name} "
                f"right in the highest-risk window for recurrence."
            )
            covered.add("recency")
        elif days_since < 60:
            sentences.append(
                f"{days_since} days post-injury. Past the acute danger zone but not fully out of it yet."
            )
            covered.add("recency")
    if "workload" not in covered:
        if acwr >= 1.8:
            sentences.append(
                f"Workload ratio is at {acwr:.2f}, which is a significant spike. "
                f"The body has been asked to do more than it has been conditioned for recently."
            )
            covered.add("workload")
        elif acwr >= 1.5 and fatigue >= 1.0:
            sentences.append(
                f"Workload ratio at {acwr:.2f} with a fatigue index of {fatigue:.1f}. "
                f"Load has been creeping up relative to the longer-term conditioning base."
            )
            covered.add("workload")

    # WORKLOAD context when not flagged above
    if "workload" not in covered:
        if matches_last_7 >= 3:
            sentences.append(
                f"{matches_last_7} matches in 7 days is genuine fixture congestion "
                f"that increases soft-tissue risk regardless of underlying fitness."
            )
        elif matches_last_30 >= 8:
            sentences.append(
                f"{matches_last_30} matches in 30 days. That kind of schedule tests "
                f"recovery capacity even for the most durable players."
            )
        elif is_sparse_profile and acwr > 0:
            if acwr <= 0.8:
                sentences.append(
                    f"Workload ratio is {acwr:.2f}. Lighter recent schedule keeps stress low "
                    f"but also means less match conditioning in the legs."
                )
            else:
                sentences.append(
                    f"Workload ratio is {acwr:.2f}. No spike, no fatigue flag. Load looks well managed."
                )

    # AGE
    if age >= 34:
        sentences.append(
            f"At {age}, recovery margins are thinner and load management becomes critical."
        )
    elif age >= 32:
        sentences.append(
            f"At {age}, fixture congestion takes a bigger toll than it would on a younger player."
        )
    elif age <= 20:
        sentences.append(
            f"At {age}, the body is still developing, which can be protective but also unpredictable."
        )

    # FIXTURE — layer all 3 data dimensions: recent form, opponent defense, player H2H
    if opponent:
        team = _display_team_name(str(player_data.get("team", "") or ""))
        venue_word = "at home" if is_home else "away" if is_home is not None else ""

        # 1. Player's recent form (role-aware) — full sentence
        if role in ("defender", "goalkeeper"):
            if recent_samples > 0 and recent_clean_sheets >= 2:
                sentences.append(
                    f"{first_name} has kept {recent_clean_sheets} clean sheets in the last {recent_samples}, so the defensive form is there."
                )
            elif recent_samples > 0 and recent_clean_sheets == 0:
                sentences.append(
                    f"No clean sheets in {first_name}'s last {recent_samples}, which makes this a harder sell defensively."
                )
        else:
            if recent_gi >= 3 and recent_samples > 0:
                sentences.append(
                    f"{_format_goal_assist_recent(recent_goals, recent_assists, recent_samples)} tells you {first_name} is match-sharp and in rhythm right now."
                )
            elif recent_gi >= 1 and recent_samples > 0:
                recent_output_line = _format_goal_assist_recent(recent_goals, recent_assists, recent_samples)
                if recent_output_line:
                    sentences.append(f"{recent_output_line} for {first_name}.")
            elif recent_samples >= 3 and recent_gi == 0:
                sentences.append(
                    f"No goal involvements in {recent_samples} appearances is a concern, the output has dried up."
                )

        # 2. Opponent defense — full sentence
        conceded_line = _natural_conceding_line(opponent, opp_conceded)
        if conceded_line:
            sentences.append(f"{conceded_line}.")

        # 3. Player's personal record vs this opponent — full sentence
        if role in ("defender", "goalkeeper"):
            if vs_samples >= 2 and vs_cs > 0:
                sentences.append(
                    f"{first_name} has kept {_pl(vs_cs, 'clean sheet')} in {_pl(vs_samples, 'meeting')} against {opponent}, which is a good sign for this fixture."
                )
            elif vs_samples >= 2 and vs_cs == 0:
                sentences.append(
                    f"No clean sheets in {vs_samples} meetings against {opponent} though, so the H2H record does not help."
                )
        else:
            if vs_samples >= 1 and vs_goals + vs_assists >= 1:
                sentences.append(
                    f"Against {opponent} specifically, {first_name} has {_format_h2h_output(vs_goals, vs_assists, vs_samples, opponent).replace(f' in {vs_samples} meetings with {opponent}', '')} in {_pl(vs_samples, 'meeting')}."
                )
            elif vs_samples >= 2 and vs_goals + vs_assists == 0:
                sentences.append(
                    f"No returns in {vs_samples} meetings against {opponent}, so the personal record does not inspire confidence here."
                )

        # 4. Team fixture history (only if no player H2H already mentioned)
        has_h2h = any("meeting" in s for s in sentences)
        if fixture_samples >= 3 and not has_h2h:
            sentences.append(
                f"{team} have gone {fh_wins}W-{fh_losses}L in {fixture_samples} recent meetings with {opponent}."
            )

        # 5. Venue context woven in naturally if not already implied
        fixture_already_mentioned = any(opponent in s for s in sentences)
        if venue_word and not fixture_already_mentioned:
            if is_home:
                sentences.append(f"Home fixture against {opponent} helps.")
            elif is_sparse_profile:
                sentences.append(f"Away to {opponent} adds a small element of uncertainty.")

        covered.add("fixture")

    # RECENT FORM fallback — only if fixture section didn't fire
    if "fixture" not in covered and is_sparse_profile and recent_samples > 0 and len(sentences) < 5:
        recent_output_line = _format_goal_assist_recent(recent_goals, recent_assists, recent_samples)
        if recent_gi >= 3:
            sentences.append(
                f"{recent_output_line}, so {first_name} is clearly in rhythm right now."
            )
        elif recent_gi == 0 and recent_samples >= 3:
            sentences.append(
                f"Zero goal involvements in {recent_samples} is a quiet spell worth watching."
            )
        elif recent_gi > 0:
            sentences.append(f"{recent_output_line}.")

    # Minimum sentence guarantee for sparse profiles
    if is_sparse_profile and len(sentences) < 3 and opponent:
        sentences.append(
            f"There is not much injury data to work with here, so the {opponent} fixture and workload patterns carry more weight than usual."
        )

    # ARCHETYPE — natural sentence that adds context, not a label
    if archetype == "Currently Vulnerable" and "recency" not in covered:
        sentences.append(
            f"The profile right now screams vulnerability, {first_name} is in that window where the body has not fully rebuilt trust."
        )
    elif archetype == "Fragile" and "history" not in covered:
        sentences.append(
            f"When {first_name} does get hurt, it tends to be serious. That pattern is worth respecting."
        )
    elif archetype == "Injury Prone":
        sentences.append(
            f"The frequency of injuries is the concern here, {first_name} picks up knocks more often than most."
        )
    elif archetype == "Recurring":
        sentences.append(
            f"There is a recurring pattern in {first_name}'s injury history, the same areas keep showing up."
        )
    elif archetype == "Durable" and not any("durable" in s.lower() or "clean" in s.lower() for s in sentences):
        sentences.append(
            f"{first_name} has been durable over a long stretch, and that kind of resilience is worth noting."
        )

    # OPEN QUESTION
    context_chunks = retrieve_player_context(
        player_data, extra_context=extra_context, top_k=12, include_open_question=True,
    )
    open_question = ""
    for chunk in context_chunks:
        if chunk.get("kind") == "open_question":
            q_text = chunk.get("text", "")
            # Skip if it just restates fixture info we already said
            if "fixture" in covered and opponent and opponent.lower() in q_text.lower():
                continue
            open_question = _as_sentence(q_text)
            break

    story_body = ". ".join(s.rstrip(".") for s in sentences[:5]) + "."
    # RAG fact line feeds the LLM as context but does NOT get prepended —
    # the LEAD and FIXTURE sections already cover the same data in natural sentences.
    story = story_body
    if open_question:
        story += " " + open_question
    story = _as_sentence(story)
    fallback_story = story or f"{first_name} profiles at moderate injury risk."

    return generate_grounded_narrative(
        task=(
            f"Write a stat-first risk narrative for {first_name}. "
            "Lead with the most decisive number. Short declarative sentences. "
            "Every sentence earns its place with a specific fact. "
            "Reference the player's recent form, their personal record against this opponent, "
            "and the opponent's defensive record if available in the context. "
            "Do NOT repeat any fact already stated. Keep it sharp and specific."
        ),
        player_name=name,
        context_chunks=context_chunks,
        fallback_text=fallback_story,
        require_open_question=True,
    )


def generate_risk_factors_list(player_data: Dict) -> List[Dict]:
    """
    Generate a list of key risk factors for a player.

    Returns:
        List of dicts with 'factor', 'impact' (positive/negative), 'description'
    """
    factors = []

    prev_injuries = player_data.get("previous_injuries", 0)
    days_lost = player_data.get("total_days_lost", 0)
    days_since = player_data.get("days_since_last_injury", 365)
    age = player_data.get("age", 25)
    avg_days = days_lost / prev_injuries if prev_injuries > 0 else 0

    # Recency
    if days_since < 30:
        factors.append({
            "factor": "Recent Return",
            "impact": "high_risk",
            "description": f"Only {days_since} days since last injury - re-injury risk is highest in this window"
        })
    elif days_since < 60:
        factors.append({
            "factor": "Recovery Window",
            "impact": "moderate_risk",
            "description": f"{days_since} days since injury - still in elevated risk period"
        })
    elif days_since > 365:
        factors.append({
            "factor": "Injury-Free Period",
            "impact": "protective",
            "description": f"Over a year since last injury - good sign of durability"
        })

    # Injury count
    if prev_injuries >= 10:
        factors.append({
            "factor": "Injury History",
            "impact": "high_risk",
            "description": f"{prev_injuries} previous injuries indicates elevated baseline risk"
        })
    elif prev_injuries == 0:
        factors.append({
            "factor": "Clean Record",
            "impact": "protective",
            "description": "No significant injuries on record"
        })

    # Severity
    if avg_days >= 40:
        factors.append({
            "factor": "Injury Severity",
            "impact": "high_risk",
            "description": f"Average of {avg_days:.0f} days per injury - tends toward serious injuries"
        })
    elif avg_days < 20 and prev_injuries > 0:
        factors.append({
            "factor": "Quick Recovery",
            "impact": "protective",
            "description": f"Average of {avg_days:.0f} days per injury - recovers quickly"
        })

    # Age
    if age >= 32:
        factors.append({
            "factor": "Age Factor",
            "impact": "moderate_risk",
            "description": f"At {age}, recovery times may be longer"
        })
    elif age <= 21:
        factors.append({
            "factor": "Young Player",
            "impact": "neutral",
            "description": f"At {age}, still developing physically"
        })

    # Workload
    acwr = _safe_float(player_data.get("acwr", 0.0), 0.0)
    if acwr >= 1.5:
        factors.append({
            "factor": "Workload Spike",
            "impact": "high_risk",
            "description": f"ACWR at {acwr:.2f}. Workload has ramped up faster than the body is conditioned for"
        })
    elif 0 < acwr <= 0.8:
        factors.append({
            "factor": "Light Workload",
            "impact": "protective",
            "description": f"ACWR at {acwr:.2f}. Recent load is lighter, reducing physical stress"
        })
    elif acwr > 0:
        factors.append({
            "factor": "Managed Workload",
            "impact": "neutral",
            "description": f"ACWR at {acwr:.2f}. Workload is in the safe zone"
        })

    # Match density
    matches_30 = _safe_int(player_data.get("matches_last_30", 0), 0)
    if matches_30 >= 6:
        factors.append({
            "factor": "Fixture Congestion",
            "impact": "moderate_risk",
            "description": f"{matches_30} matches in 30 days is a demanding schedule"
        })

    # Sort by impact (high_risk first, then moderate, then neutral, then protective)
    impact_order = {"high_risk": 0, "moderate_risk": 1, "neutral": 2, "protective": 3}
    factors.sort(key=lambda x: impact_order.get(x["impact"], 2))

    return factors


def get_recommendation_text(player_data: Dict) -> str:
    """
    Generate a single recommendation paragraph for a player.
    """
    archetype = player_data.get("archetype", "Unknown")
    days_since = player_data.get("days_since_last_injury", 365)
    prob = player_data.get("ensemble_prob", 0.5)

    if days_since < 30:
        return "Priority should be on careful workload management during this critical re-injury window. Consider reduced minutes and avoid back-to-back matches where possible."

    if archetype == "Fragile":
        return "Given the history of serious injuries, a proactive approach to load management is recommended. Quality recovery time between matches and avoiding fixture congestion periods would be beneficial."

    if archetype == "Injury Prone":
        return "Regular physiotherapy and strength conditioning work can help manage the elevated injury risk. Monitoring training loads closely and ensuring adequate rest is important."

    if archetype == "Currently Vulnerable":
        return "The recent return from injury means careful monitoring is essential. A gradual return to full match intensity with adequate recovery between fixtures is recommended."

    if prob >= 0.45:
        return "With elevated risk factors, rotation with squad depth and careful fixture management during busy periods would help mitigate injury probability."

    if prob < 0.30:
        return "Current risk indicators are favorable. Maintaining the current training and recovery regime should support continued availability."

    return "Standard monitoring and recovery protocols are appropriate. No specific interventions indicated at this time."


def get_fpl_insight(player_data: Dict, extra_context: Optional[Dict] = None) -> Optional[str]:
    """Generate one punchy, stat-led FPL manager tip with real fixture context."""
    first_name = _call_name(player_data)
    if first_name and first_name[0].islower():
        first_name = first_name[0].upper() + first_name[1:]
    injury_prob = _safe_float(player_data.get("ensemble_prob", 0.5), 0.5)
    days_since = _safe_int(player_data.get("days_since_last_injury", 365), 365)
    is_currently_injured = bool(player_data.get("is_currently_injured", False))
    minutes = _safe_int(player_data.get("minutes", 0), 0)
    position = str(player_data.get("position", "Unknown") or "Unknown")
    role = _position_group(position)
    goals_per_90 = _safe_float(player_data.get("goals_per_90", 0.0), 0.0)
    assists_per_90 = _safe_float(player_data.get("assists_per_90", 0.0), 0.0)
    output_per_90 = goals_per_90 + assists_per_90
    risk_pct = round(injury_prob * 100)

    extra_context = extra_context or {}
    player_importance = _extract_player_importance(extra_context)
    matchup_context = extra_context.get("matchup_context") or {}
    recent_form = matchup_context.get("recent_form") or {}
    recent_samples = _safe_int(recent_form.get("samples", 0), 0)
    recent_goals = _safe_int(recent_form.get("goals", 0), 0)
    recent_assists = _safe_int(recent_form.get("assists", 0), 0)
    recent_returns = _safe_int(recent_form.get("returns", 0), 0)
    recent_clean_sheets = _safe_int(recent_form.get("clean_sheets", 0), 0)
    recent_gi = recent_goals + recent_assists

    vs_opponent = matchup_context.get("vs_opponent") or {}
    vs_samples = _safe_int(vs_opponent.get("samples", 0), 0)
    vs_goals = _safe_int(vs_opponent.get("goals", 0), 0)
    vs_assists = _safe_int(vs_opponent.get("assists", 0), 0)
    vs_returns = _safe_int(vs_opponent.get("returns", 0), 0)
    vs_clean_sheets = _safe_int(vs_opponent.get("clean_sheets", 0), 0)

    opponent_defense = matchup_context.get("opponent_defense") or {}
    opp_conceded = _safe_float(opponent_defense.get("avg_goals_conceded_last5", 0.0), 0.0)
    fixture_history = extra_context.get("fixture_history") or {}

    next_fixture = extra_context.get("next_fixture") or {}
    opponent = _display_team_name(
        matchup_context.get("opponent") or next_fixture.get("opponent") or "this opponent"
    )
    is_home = matchup_context.get("is_home")
    if is_home is None:
        is_home = next_fixture.get("is_home")
    action = "Start"
    reason_bits: List[str] = []

    importance_tier = str(player_importance.get("tier", "") or "").strip()
    importance_ownership = _safe_float(player_importance.get("ownership_pct", 0.0), 0.0)

    # Fixture attractiveness: high-output player + weak opponent = still worth starting
    fixture_is_attractive = (
        (opp_conceded >= 1.2 and is_home is True)
        or (opp_conceded >= 1.5)
        or (vs_samples >= 2 and (vs_goals + vs_assists) >= 2)
    )
    high_output_player = output_per_90 >= 0.4 or (role in ("defender", "goalkeeper") and recent_clean_sheets >= 2)

    is_key_player = importance_tier in {"Core", "High"} or importance_ownership >= 15

    # ── Gather matchup signals (position-aware) ──
    team = _display_team_name(str(player_data.get("team", "") or ""))

    # Fixture history (team level)
    fixture_history = extra_context.get("fixture_history") or {}
    fh_samples = _safe_int(fixture_history.get("samples", 0), 0)
    fh_wins = _safe_int(fixture_history.get("wins", 0), 0)
    team_dominates = fh_samples >= 3 and fh_wins >= 2

    if role in ("defender", "goalkeeper"):
        # Defenders: form = clean sheets, H2H = clean sheets, fixture = team dominance
        has_form = recent_clean_sheets >= 2 and recent_samples > 0
        form_cold = recent_clean_sheets == 0 and recent_samples >= 3
        has_h2h = vs_samples >= 1 and vs_clean_sheets > 0
        has_good_fixture = team_dominates or (is_home is True and fh_samples >= 2)
        # "opponent concedes a lot" is irrelevant for a defender's clean sheet potential
        form_detail = f"{recent_clean_sheets} clean sheets in {recent_samples}" if has_form else ""
        h2h_detail = f"{vs_clean_sheets} clean sheet{'s' if vs_clean_sheets != 1 else ''} in {vs_samples} against {opponent}" if has_h2h else ""
    else:
        # Attackers/midfielders: form = goal involvements, H2H = returns, fixture = weak opponent
        has_form = recent_gi >= 2 and recent_samples > 0
        form_cold = recent_gi == 0 and recent_samples >= 3
        has_h2h = vs_samples >= 1 and (vs_goals + vs_assists) >= 1
        has_good_fixture = opp_conceded >= 1.0
        form_detail = f"{recent_gi} goal involvements in {recent_samples}" if has_form else ""
        h2h_detail = (
            f"{vs_goals + vs_assists} return{'s' if (vs_goals + vs_assists) != 1 else ''} in {vs_samples} against {opponent}"
            if has_h2h else ""
        )

    signals_for = sum([has_form, has_h2h, has_good_fixture, is_home is True])
    all_align = has_form and has_good_fixture and has_h2h

    # ── Decision ──
    if minutes < 180:
        action = "Bench"
        reason_bits.append(
            f"{first_name} does not have enough minutes yet to read anything into"
        )
    elif is_currently_injured or days_since < 14:
        action = "Bench"
        reason_bits.append(
            f"Only {days_since} days back, too soon to be starting in FPL"
        )
    elif all_align:
        action = "Start"
        if injury_prob >= 0.50:
            reason_bits.append(
                f"I can see why a {team} fan would want {first_name} in against {opponent} this week. "
                f"{form_detail}, {h2h_detail}, and the team have been dominant in this fixture. "
                f"The risk is real but everything else says go for it. Bench cover is a must"
            )
        else:
            reason_bits.append(
                f"{form_detail}, {h2h_detail}, and {team} own this fixture. "
                f"As good a week as any to have {first_name} in"
            )
    elif has_form and has_good_fixture:
        action = "Start"
        if role in ("defender", "goalkeeper"):
            reason_bits.append(
                f"{form_detail} and {team} tend to control games against {opponent}. "
                f"Clean sheet potential is there"
            )
        elif injury_prob >= 0.50:
            reason_bits.append(
                f"{first_name} is in form and {opponent} have been giving up chances. "
                f"Worth the gamble this week but make sure the bench is solid"
            )
        else:
            reason_bits.append(
                f"{form_detail} and {opponent} are not exactly shutting teams out. Good week to start"
            )
    elif has_form and has_h2h:
        action = "Start"
        reason_bits.append(
            f"{form_detail} and {h2h_detail}. That is enough to go with"
        )
    elif has_good_fixture and has_h2h:
        action = "Start"
        if role in ("defender", "goalkeeper"):
            reason_bits.append(
                f"{h2h_detail} and {team} tend to boss this fixture. Good clean sheet spot"
            )
        else:
            reason_bits.append(
                f"{opponent} are conceding and {h2h_detail}. The fixture does the talking"
            )
    elif has_form:
        if injury_prob >= 0.45:
            action = "Start"
            reason_bits.append(
                f"{form_detail} but the body and the fixture do not scream must-start"
            )
        else:
            action = "Start"
            reason_bits.append(f"{form_detail}. Fine to run with this week")
    elif has_good_fixture:
        if role in ("defender", "goalkeeper"):
            action = "Start" if injury_prob < 0.50 else "Bench"
            if action == "Start":
                reason_bits.append(
                    f"{team} have been solid in this fixture but {first_name}'s own form needs to back it up"
                )
            else:
                reason_bits.append(
                    f"The fixture history favours {team} but {risk_pct}% risk is hard to ignore for a defender"
                )
        else:
            action = "Start" if injury_prob < 0.50 else "Bench"
            if action == "Start":
                reason_bits.append(
                    f"{opponent} are conceding enough to make {first_name} interesting, but the form needs to back it up"
                )
            else:
                reason_bits.append(
                    f"{opponent} are beatable but {first_name}'s body is the bigger story right now"
                )
    elif form_cold:
        action = "Bench"
        if role in ("defender", "goalkeeper"):
            reason_bits.append(
                f"No clean sheets recently for {first_name}. Hard to justify the start"
            )
        else:
            reason_bits.append(
                f"{first_name} has gone quiet recently. Hard to justify the start on reputation alone"
            )
    elif injury_prob >= 0.55:
        action = "Bench"
        reason_bits.append(
            f"{risk_pct}% risk and no standout matchup data to override it"
        )
    elif injury_prob < 0.30:
        action = "Start"
        reason_bits.append(f"Low risk and no red flags. {first_name} is fine to start")
    else:
        action = "Start"
        reason_bits.append(f"Nothing jumps out either way. Fine to start if {first_name} is in your plans")

    # One reason only — the action label is metadata, not part of the sentence
    primary_reason = reason_bits[0] if reason_bits else ""
    fallback_text = _as_sentence(primary_reason) if primary_reason else ""

    context_chunks = retrieve_player_context(
        player_data,
        extra_context=extra_context,
        query="fpl manager tip injury availability fixture",
        top_k=6,
        include_open_question=False,
    )

    polished = generate_grounded_narrative(
        task=(
            f"Write a 1-2 sentence FPL decision for {first_name} in Yara's voice. "
            f"The decision is: {action}. Give the reason why, do not repeat "
            "facts from the risk analysis. Sound like a sharp football friend."
        ),
        player_name=player_data.get("name", "This player"),
        context_chunks=context_chunks,
        fallback_text=fallback_text,
        require_open_question=False,
    )

    return _as_sentence(re.sub(r"\s{2,}", " ", (polished or fallback_text)).strip())


def _prob_to_odds(prob: float) -> dict:
    """Convert a probability to American, decimal, and fractional odds."""
    prob = max(0.02, min(0.98, prob))
    if prob >= 0.5:
        american = int(-100 * prob / (1 - prob))
        american_str = str(american)
    else:
        american = int(100 * (1 - prob) / prob)
        american_str = f"+{american}"
    decimal_odds = round(1 / prob, 2)
    if decimal_odds >= 2:
        fractional = f"{int(round(decimal_odds - 1))}/1"
    else:
        denom = int(round(1 / (decimal_odds - 1)))
        fractional = f"1/{denom}"
    return {"american": american_str, "decimal": decimal_odds, "fractional": fractional}


def calculate_scoring_odds(player_data: Dict, extra_context: Optional[Dict] = None) -> Optional[Dict]:
    """
    Calculate odds for a player to score in the next match.

    Combines historical scoring rate with availability probability.
    """
    position = str(player_data.get("position", "") or "").lower()
    if any(token in position for token in ["def", "gk", "goalkeeper", "back"]):
        return None

    goals_per_90 = player_data.get("goals_per_90", 0)
    assists_per_90 = player_data.get("assists_per_90", 0)
    injury_prob = player_data.get("ensemble_prob", 0.5)
    minutes = player_data.get("minutes", 0)

    if minutes < 270:
        return None

    base_score_prob = min(goals_per_90, 1.5)
    availability = 1 - (injury_prob * 0.5)
    score_prob = max(0.02, min(0.85, base_score_prob * availability))
    involvement_prob = min((goals_per_90 + assists_per_90) * availability, 0.95)
    odds = _prob_to_odds(score_prob)

    player_call_name = _call_name(player_data)
    scoring_context = retrieve_player_context(
        player_data,
        extra_context=extra_context,
        query="scoring odds injury adjusted probability goals assists availability fixture market",
        top_k=10,
        include_open_question=False,
    )
    scoring_context_line = build_dynamic_rag_line(
        player_data,
        extra_context=extra_context,
        section="scoring",
        context_chunks=scoring_context,
    )
    fallback_analysis = (
        f"Yara estimates {player_call_name}'s chance to score at {round(score_prob * 100)}% after injury adjustment. "
        + (
            f"{scoring_context_line}"
            if scoring_context_line
            else (
                f"Baseline sits at {goals_per_90:.2f} goals/90 and {assists_per_90:.2f} assists/90 "
                f"with availability {availability:.2f}."
            )
        )
    )
    scoring_analysis = generate_grounded_narrative(
        task="Write a concise odds-to-score analysis for this player.",
        player_name=player_data.get("name", "This player"),
        context_chunks=scoring_context,
        fallback_text=fallback_analysis,
        require_open_question=False,
    )

    return {
        "score_probability": round(score_prob, 3),
        "involvement_probability": round(involvement_prob, 3),
        "goals_per_90": round(goals_per_90, 2),
        "assists_per_90": round(assists_per_90, 2),
        "american": odds["american"],
        "decimal": odds["decimal"],
        "fractional": odds["fractional"],
        "availability_factor": round(availability, 2),
        "analysis": scoring_analysis,
    }


def get_fpl_value_assessment(player_data: Dict, extra_context: Optional[Dict] = None) -> Optional[Dict]:
    """
    Generate FPL value assessment combining injury risk with attacking output.
    """
    name = player_data.get("name", "Player")
    first_name = _call_name(player_data)
    goals = _safe_int(player_data.get("goals", 0), 0)
    assists = _safe_int(player_data.get("assists", 0), 0)
    goals_per_90 = _safe_float(player_data.get("goals_per_90", 0), 0.0)
    assists_per_90 = _safe_float(player_data.get("assists_per_90", 0), 0.0)
    price = _safe_float(player_data.get("price", 0), 0.0)
    injury_prob = _safe_float(player_data.get("ensemble_prob", 0.5), 0.5)
    position = str(player_data.get("position", "Unknown") or "Unknown")
    archetype = player_data.get("archetype", "Unknown")
    minutes = _safe_int(player_data.get("minutes", 0), 0)
    role = _position_group(position)

    if minutes < 270 or price <= 0:
        return None

    extra_context = extra_context or {}
    matchup_context = extra_context.get("matchup_context") or {}
    recent_form = matchup_context.get("recent_form") or {}
    recent_samples = _safe_int(recent_form.get("samples", 0), 0)
    recent_goals = _safe_int(recent_form.get("goals", 0), 0)
    recent_assists = _safe_int(recent_form.get("assists", 0), 0)
    recent_returns = _safe_int(recent_form.get("returns", 0), 0)
    recent_clean_sheets = _safe_int(recent_form.get("clean_sheets", 0), 0)

    ga_per_90 = goals_per_90 + assists_per_90
    risk_factor = max(0.05, 1 - injury_prob)
    recent_output_rate = (
        (recent_goals + recent_assists) / max(1, recent_samples)
        if recent_samples > 0
        else 0.0
    )
    recent_clean_sheet_rate = (
        recent_clean_sheets / max(1, recent_samples)
        if recent_samples > 0
        else 0.0
    )

    if role == "defender":
        # Defenders get value mostly from clean-sheet potential, with some attacking upside.
        output_signal = (recent_clean_sheet_rate * 0.7) + (ga_per_90 * 0.3)
        tier_thresholds = (0.42, 0.30, 0.20, 0.12)  # premium, strong, decent, rotation
    else:
        # Mid/attacker value is driven by output rate and recent return cadence.
        output_signal = (ga_per_90 * 0.75) + ((recent_output_rate * 0.25) if recent_samples > 0 else 0.0)
        tier_thresholds = (0.60, 0.40, 0.25, 0.12)

    adjusted_value = output_signal * risk_factor
    premium_cut, strong_cut, decent_cut, rotation_cut = tier_thresholds

    if adjusted_value >= premium_cut:
        tier, emoji = "Premium", "gem"
    elif adjusted_value >= strong_cut:
        tier, emoji = "Strong", "badge-check"
    elif adjusted_value >= decent_cut:
        tier, emoji = "Decent", "thumbs-up"
    elif adjusted_value >= rotation_cut:
        tier, emoji = "Rotation", "rotate-cw"
    else:
        tier, emoji = "Avoid", "ban"

    # Injury history context for verdict
    prev_injuries = _safe_int(player_data.get("previous_injuries", player_data.get("player_injury_count", 0)), 0)
    days_since = _safe_int(player_data.get("days_since_last_injury", 365), 365)
    total_days_lost = _safe_float(player_data.get("total_days_lost", 0), 0.0)
    avg_days_per_injury = total_days_lost / prev_injuries if prev_injuries > 0 else 0
    injury_prone = prev_injuries >= 5 or (prev_injuries >= 3 and avg_days_per_injury >= 30)
    high_risk = injury_prob >= 0.40
    low_output = output_signal < 0.18
    durability_signal = days_since >= 240 and prev_injuries <= 2

    # Player importance context
    player_importance = _extract_player_importance(extra_context)
    importance_tier = str(player_importance.get("tier", "") or "").strip()
    importance_ownership = _safe_float(player_importance.get("ownership_pct", 0.0), 0.0)
    is_key_player = importance_tier in {"Core", "High"} or importance_ownership >= 15

    # Matchup context for value override
    next_fixture = extra_context.get("next_fixture") or {}
    opponent_defense = matchup_context.get("opponent_defense") or {}
    opp_conceded = _safe_float(
        opponent_defense.get("avg_goals_conceded_last5",
            matchup_context.get("opponent_goals_conceded_avg", 0)),
        0.0,
    )
    vs_opponent = matchup_context.get("vs_opponent") or {}
    vs_samples = _safe_int(vs_opponent.get("samples", 0), 0)
    vs_goals = _safe_int(vs_opponent.get("goals", 0), 0)
    vs_assists = _safe_int(vs_opponent.get("assists", 0), 0)
    is_home = matchup_context.get("is_home") or next_fixture.get("is_home")

    # Check if form + fixture + H2H all point the same way (position-aware)
    val_fixture_history = extra_context.get("fixture_history") or {}
    val_fh_samples = _safe_int(val_fixture_history.get("samples", 0), 0)
    val_fh_wins = _safe_int(val_fixture_history.get("wins", 0), 0)

    if role in ("defender", "goalkeeper"):
        # Defenders: clean sheet form, clean sheet H2H, team dominance
        val_has_form = recent_clean_sheets >= 2 and recent_samples > 0
        val_has_h2h = vs_samples >= 1 and _safe_int(vs_opponent.get("clean_sheets", 0), 0) > 0
        val_has_good_fixture = (val_fh_samples >= 3 and val_fh_wins >= 2) or (is_home is True and val_fh_samples >= 2)
    else:
        # Attackers: goal form, scoring H2H, weak opponent defense
        val_has_form = (recent_goals + recent_assists) >= 2 and recent_samples > 0
        val_has_h2h = vs_samples >= 1 and (vs_goals + vs_assists) >= 1
        val_has_good_fixture = opp_conceded >= 1.0
    matchup_all_align = val_has_form and val_has_good_fixture and val_has_h2h

    # Override: when matchup fully aligns, Avoid becomes Rotation
    if tier == "Avoid" and matchup_all_align and not low_output:
        tier, emoji = "Rotation", "rotate-cw"
    elif tier == "Avoid" and is_key_player and val_has_form and val_has_good_fixture and not low_output:
        tier, emoji = "Rotation", "rotate-cw"

    risk_pct = round(injury_prob * 100)

    if tier == "Avoid":
        if low_output and high_risk:
            verdict = (
                f"{ga_per_90:.2f} G+A per 90 at {risk_pct}% risk. "
                f"Not much coming back for {first_name} and the body is a concern too. "
                f"Would look elsewhere. {_pick_kicker('avoid', first_name)}"
            )
        elif low_output and durability_signal:
            verdict = (
                f"The body is fine for {first_name}, that is not the issue. "
                f"It is the {ga_per_90:.2f} G+A per 90 at £{price:.1f}m. "
                f"That money works harder somewhere else. {_pick_kicker('avoid', first_name)}"
            )
        elif low_output:
            verdict = (
                f"{ga_per_90:.2f} G+A per 90 at £{price:.1f}m. "
                f"{first_name} needs to do more to justify the spot. {_pick_kicker('avoid', first_name)}"
            )
        elif high_risk:
            verdict = (
                f"I see the talent but {risk_pct}% risk keeps eating into {first_name}'s value. "
                f"The injury drag is real. {_pick_kicker('avoid', first_name)}"
            )
        else:
            verdict = (
                f"Hard to make the numbers work for {first_name} at £{price:.1f}m right now. "
                f"That budget does more elsewhere. {_pick_kicker('avoid', first_name)}"
            )
    elif tier == "Rotation":
        if low_output:
            verdict = (
                f"{first_name} is getting minutes but not doing much with them. "
                f"{ga_per_90:.2f} G+A per 90. Bench fodder at best."
            )
        elif high_risk and matchup_all_align:
            opponent_name = _display_team_name(
                matchup_context.get("opponent") or next_fixture.get("opponent") or ""
            )
            if role in ("defender", "goalkeeper"):
                verdict = (
                    f"The {risk_pct}% risk would normally push {first_name} to Avoid, "
                    f"but the clean sheet form, the team's dominance in this fixture, "
                    f"and the personal record against {opponent_name} all say this week is different. "
                    f"Worth a punt with bench cover."
                )
            else:
                verdict = (
                    f"The {risk_pct}% risk would normally push {first_name} to Avoid, "
                    f"but the form, {opponent_name}'s defence, and the personal record against them all say this week is different. "
                    f"Worth a punt with bench cover."
                )
        elif high_risk:
            verdict = (
                f"The talent is there for {first_name} but {risk_pct}% risk makes this week-to-week. "
                f"Not someone to lock in, revisit each gameweek."
            )
        else:
            verdict = (
                f"Fixture-dependent for {first_name}. Fine as depth on a good week, "
                f"but not someone to set and forget."
            )
    elif tier == "Decent":
        if high_risk:
            verdict = (
                f"Decent value when fit, but that is the catch with {first_name}. "
                f"{risk_pct}% risk means you need a real backup plan. "
                f"{_pick_kicker('moderate_risk', first_name)}"
            )
        else:
            verdict = (
                f"{ga_per_90:.2f} G+A per 90 at £{price:.1f}m. "
                f"{first_name} gives a reliable floor without breaking the bank. "
                f"{_pick_kicker('default', first_name)}"
            )
    elif tier == "Strong":
        if recent_returns >= 3 and recent_samples > 0:
            verdict = (
                f"{recent_returns} returns in the last {recent_samples} for {first_name}. "
                f"The form is real and the body is holding up. "
                f"I like this a lot. {_pick_kicker('form_hot', first_name)}"
            )
        else:
            verdict = (
                f"Quietly one of the better value picks in the game right now. "
                f"{ga_per_90:.2f} G+A per 90 at £{price:.1f}m for {first_name}. "
                f"{_pick_kicker('premium', first_name)}"
            )
    else:  # Premium
        if goals_per_90 >= 0.5:
            verdict = (
                f"{goals_per_90:.2f} goals per 90 for {first_name} at {risk_pct}% risk. "
                f"The maths prints money. Armband candidate. {_pick_kicker('premium', first_name)}"
            )
        else:
            verdict = (
                f"Set and forget {first_name}. Output ceiling and floor both "
                f"look elite at £{price:.1f}m. {_pick_kicker('premium', first_name)}"
            )

    if archetype == "Currently Vulnerable" and days_since < 60:
        verdict = (
            f"{first_name} is back but it has only been {days_since} days. "
            f"I would give it another week before trusting this in FPL. "
            f"{_pick_kicker('moderate_risk', first_name)}"
        )
    elif injury_prone and days_since < 120 and tier in {"Decent", "Strong", "Premium"}:
        verdict = (
            f"Real upside here with {first_name}, but {prev_injuries} injuries "
            f"averaging {avg_days_per_injury:.0f} days each. High ceiling, high volatility. "
            f"Make sure the bench can cover. {_pick_kicker('moderate_risk', first_name)}"
        )

    position_insight = None
    if role == "attacker" and ga_per_90 >= 0.5 and injury_prob < 0.20:
        position_insight = (
            f"{_stat_lead(goals_per_90, 'goals/90')}{goals} goals this season at {risk_pct}% risk. "
            f"Captain-viable. {_pick_kicker('premium', first_name + 'pos')}"
        )
    elif role == "midfielder" and assists_per_90 >= 0.3 and injury_prob < 0.20:
        position_insight = (
            f"{_stat_lead(assists_per_90, 'assists/90')}{assists} assists and barely misses games. "
            f"Elite creative slot. {_pick_kicker('premium', first_name + 'pos')}"
        )
    elif role == "defender" and (recent_clean_sheet_rate >= 0.4 or ga_per_90 >= 0.2):
        position_insight = (
            f"{_stat_lead(ga_per_90, 'G+A/90')}Attacking upside from a defender is rare. "
            f"Clean sheets are the floor, {first_name}'s output is the ceiling. {_pick_kicker('premium', first_name + 'pos')}"
        )

    verdict = _as_sentence(verdict)

    return {
        "tier": tier,
        "tier_emoji": emoji,
        "verdict": verdict,
        "position_insight": position_insight,
        "adjusted_value": round(adjusted_value, 2),
        "goals_per_90": round(goals_per_90, 2),
        "assists_per_90": round(assists_per_90, 2),
        "price": price,
        "risk_factor": round(risk_factor, 2),
    }


def calculate_clean_sheet_odds(player_data: Dict) -> Optional[Dict]:
    """Calculate clean sheet odds for defenders and goalkeepers."""
    position = player_data.get("position", "")
    injury_prob = player_data.get("ensemble_prob", 0.5)

    pos_lower = position.lower()
    if not any(p in pos_lower for p in ["def", "gk", "goalkeeper", "back"]):
        return None

    goals_conceded_per_game = 1.2  # PL average
    base_cs_prob = math.exp(-goals_conceded_per_game)
    availability = 1 - (injury_prob * 0.5)
    cs_prob = max(0.05, min(0.7, base_cs_prob * availability))
    odds = _prob_to_odds(cs_prob)

    return {
        "clean_sheet_probability": round(cs_prob, 3),
        "goals_conceded_per_game": round(goals_conceded_per_game, 2),
        "american": odds["american"],
        "decimal": odds["decimal"],
        "availability_factor": round(availability, 2),
    }


def generate_yara_response(player_data: Dict, market_odds: Optional[Dict] = None) -> Optional[Dict]:
    """
    Generate Yara's opinionated comparison between model projections and live market odds.

    Personality: confident, data-backed, concise, occasionally sharp. Never generic.

    Args:
        player_data: Dict with player info including ensemble_prob, goals_per_90, etc.
        market_odds: Dict from OddsClient.get_anytime_scorer_odds() or None

    Returns:
        Dict with response_text, fpl_tip, market_probability, yara_probability,
        market_odds_decimal, bookmaker, or None if insufficient data
    """
    name = player_data.get("name", "This player")
    first_name = _call_name(player_data)
    goals_per_90 = player_data.get("goals_per_90", 0)
    assists_per_90 = player_data.get("assists_per_90", 0)
    injury_prob = player_data.get("ensemble_prob", 0.5)
    minutes = player_data.get("minutes", 0)
    position = player_data.get("position", "Unknown")
    archetype = player_data.get("archetype", "Unknown")
    days_since = player_data.get("days_since_last_injury", 365)

    if minutes < 270:
        return None

    # Yara's projection: scoring probability adjusted for availability
    availability = 1 - (injury_prob * 0.5)
    yara_prob = max(0.02, min(0.85, goals_per_90 * availability))

    # For defenders/GKs, use clean sheet angle instead
    pos_lower = position.lower()
    is_defensive = any(p in pos_lower for p in ["def", "gk", "goalkeeper", "back"])
    if is_defensive and goals_per_90 < 0.1:
        return None  # No scoring angle for pure defenders

    if market_odds:
        market_prob = market_odds.get("implied_probability", 0)
        market_decimal = market_odds.get("decimal_odds", 0)
        bookmaker = market_odds.get("bookmaker", "Unknown")
        opponent = market_odds.get("opponent", "their opponent")
        is_home = market_odds.get("is_home", False)

        edge = yara_prob - market_prob
        edge_pct = abs(edge) * 100

        # Generate response based on edge direction
        if edge > 0.08:
            response = _yara_value_found(first_name, yara_prob, market_prob, opponent,
                                          is_home, injury_prob, days_since, archetype, goals_per_90)
        elif edge < -0.08:
            response = _yara_market_generous(first_name, yara_prob, market_prob, opponent,
                                              injury_prob, archetype)
        else:
            response = _yara_aligned(first_name, yara_prob, market_prob, opponent, injury_prob)

        fpl_tip = _generate_fpl_tip(player_data, yara_prob, market_prob, opponent, is_home)

        return {
            "response_text": response,
            "fpl_tip": fpl_tip,
            "market_probability": round(market_prob, 3),
            "yara_probability": round(yara_prob, 3),
            "market_odds_decimal": market_decimal,
            "bookmaker": bookmaker,
        }
    else:
        # No market data, projection only
        response = _yara_no_market(first_name, yara_prob, injury_prob, goals_per_90,
                                    assists_per_90, archetype, days_since)
        fpl_tip = _generate_fpl_tip(player_data, yara_prob, None, None, None)

        return {
            "response_text": response,
            "fpl_tip": fpl_tip,
            "market_probability": None,
            "yara_probability": round(yara_prob, 3),
            "market_odds_decimal": None,
            "bookmaker": None,
        }


def _yara_value_found(name, yara, market, opponent, is_home, inj_prob, days_since, archetype, g90):
    """Yara sees value. Her projection exceeds market."""
    edge = round((yara - market) * 100, 1)

    if days_since < 60:
        context = "Recent return adds some noise, but the underlying output supports it."
    elif inj_prob < 0.25:
        context = "Fitness profile is clean too."
    elif g90 >= 0.5:
        context = f"Scoring at {g90:.2f} per 90 backs that up."
    elif archetype == "Durable":
        context = "The durability is being underpriced here."
    else:
        venue = "at home" if is_home else "away"
        context = f"Playing {venue} against {opponent} helps."

    return (
        f"I see a {edge}% gap on {name}. I have {yara*100:.0f}%, "
        f"the bookies have {market*100:.0f}%. {context} "
        f"{_pick_kicker('form_hot', name)}"
    )


def _yara_market_generous(name, yara, market, opponent, inj_prob, archetype):
    """Market is more generous than Yara's model."""
    gap = round((market - yara) * 100, 1)

    if inj_prob >= 0.45:
        reason = f"and the {inj_prob*100:.0f}% injury risk is part of why."
    elif archetype in ("Fragile", "Currently Vulnerable"):
        reason = f"and that {archetype.lower()} profile makes availability a question mark."
    else:
        reason = "and the underlying scoring rate does not back the market price."

    return (
        f"The market is being generous with {name}. They say {market*100:.0f}%, "
        f"I am at {yara*100:.0f}%. That is a {gap}% gap the other way, "
        f"{reason} {_pick_kicker('avoid', name)}"
    )


def _yara_aligned(name, yara, market, opponent, inj_prob):
    """Market and model agree."""
    if inj_prob < 0.3:
        note = "Low injury risk is one small tailwind that might not be fully priced in."
    elif inj_prob >= 0.45:
        note = "The only variable is the teamsheet. Watch for late news."
    else:
        note = "No real edge either way. I would look elsewhere for value."

    return (
        f"Market and model land in the same place on {name}, around {yara*100:.0f}%. "
        f"{note} {_pick_kicker('default', name)}"
    )


def _yara_no_market(name, yara, inj_prob, g90, a90, archetype, days_since):
    """No market odds available. Projection only."""
    if g90 >= 0.5 and inj_prob < 0.35:
        tone = (
            f"No market line to compare, but {name} is scoring at {g90:.2f} per 90 "
            f"and the body is holding up. I have the scoring probability at {yara*100:.0f}%. "
            f"{_pick_kicker('form_hot', name)}"
        )
    elif g90 >= 0.3 and inj_prob < 0.4:
        tone = (
            f"No bookmaker line for {name} this week. "
            f"I have {yara*100:.0f}% to score. Decent output, manageable risk. "
            f"{_pick_kicker('default', name)}"
        )
    elif inj_prob >= 0.45:
        tone = (
            f"No market odds, and honestly the injury picture is the bigger story. "
            f"{inj_prob*100:.0f}% risk with only a {yara*100:.0f}% chance to score. "
            f"{_pick_kicker('high_risk', name)}"
        )
    elif days_since < 60:
        tone = (
            f"No market data for {name}. Still only {days_since} days back from injury. "
            f"Projecting {yara*100:.0f}% but would want to see more first. "
            f"{_pick_kicker('moderate_risk', name)}"
        )
    else:
        involvement = (g90 + a90) * 100
        tone = (
            f"No market line to compare this week. "
            f"I have {name} at {involvement:.0f}% for any goal involvement. "
            f"{_pick_kicker('default', name)}"
        )

    return tone


def _generate_fpl_tip(player_data, yara_prob, market_prob, opponent, is_home):
    """Generate FPL-specific tip based on model + market analysis."""
    name = _call_name(player_data)
    injury_prob = player_data.get("ensemble_prob", 0.5)
    archetype = player_data.get("archetype", "Unknown")
    form = player_data.get("form", 0)
    price = player_data.get("price", 0)
    days_since = player_data.get("days_since_last_injury", 365)

    # High value + low risk = strong pick
    if yara_prob >= 0.4 and injury_prob < 0.3:
        if market_prob and yara_prob > market_prob:
            return f"Strong pick. {name}'s output exceeds market expectations and injury risk is low. Captain material if the fixture is right."
        return f"Reliable starter with consistent output. {name} is a set-and-forget option right now."

    # Good output but injury concerns
    if yara_prob >= 0.3 and injury_prob >= 0.4:
        return f"High ceiling, high floor. {name} delivers when fit but the injury flag means you need bench cover ready."

    # Recently returned
    if days_since < 30:
        return f"Just back from injury. {name} may be eased in so don't expect full 90s. Wait a week before committing."

    # Fixture-dependent
    if opponent and is_home is not None:
        venue = "home" if is_home else "away"
        if is_home and yara_prob >= 0.25:
            return f"Home fixture favors {name}. Worth a punt if you have the funds, but don't reach."
        elif not is_home and yara_prob < 0.25:
            return f"Tough away fixture. {name} is a hold, not a buy this week."

    # Low output
    if yara_prob < 0.15:
        return f"{name}'s scoring rate is too low for FPL relevance. Look elsewhere unless you need a bench filler."

    # Default
    if price and price > 8:
        return f"At {price}m, {name} needs consistent returns to justify the price tag. Monitor form before committing."
    return f"Solid mid-range option. {name} won't lose you your league but won't win it either."


def generate_lab_notes(player_data: Dict, extra_context: Optional[Dict] = None) -> Optional[Dict]:
    """
    Generate Yara's Lab Notes using RAG context and plain-language LLM narration.

    Args:
        player_data: Dict with player info including model features

    Returns:
        Dict with summary, key_drivers, technical, or None if insufficient data
    """
    name = player_data.get("name", "This player")
    first_name = _call_name(player_data)
    prob = _safe_float(player_data.get("ensemble_prob", 0.5), 0.5)
    archetype = str(player_data.get("archetype", "") or "").strip()

    # RAG context used to ground plain-language explanation.
    lab_context = retrieve_player_context(
        player_data,
        extra_context=extra_context,
        query=(
            "lab notes explainability workload recency injury history fixture "
            "market odds form availability risk drivers"
        ),
        top_k=12,
        include_open_question=False,
    )

    # Check key features and build driver list
    drivers = []

    # 1. ACWR (Acute:Chronic Workload Ratio)
    acwr = _safe_float(player_data.get("acwr", 0), 0.0)
    if acwr >= 1.5:
        drivers.append({
            "name": "Workload Ratio",
            "value": round(acwr, 2),
            "impact": "risk_increasing",
            "explanation": f"Workload ratio is {acwr:.2f}, elevated relative to recent conditioning."
        })
    elif 0 < acwr <= 0.8:
        drivers.append({
            "name": "Workload Ratio",
            "value": round(acwr, 2),
            "impact": "protective",
            "explanation": f"Workload ratio is {acwr:.2f}, so recent load is lighter and recovery pressure is lower."
        })

    # 2. Fatigue Index
    fatigue = _safe_float(player_data.get("fatigue_index", 0), 0.0)
    if fatigue >= 1.0:
        drivers.append({
            "name": "Fatigue Index",
            "value": round(fatigue, 2),
            "impact": "risk_increasing",
            "explanation": "Recent physical load is outpacing normal recovery rhythm."
        })

    # 3. Injury Count
    inj_count = _safe_int(player_data.get("player_injury_count", player_data.get("previous_injuries", 0)), 0)
    if inj_count >= 5:
        drivers.append({
            "name": "Injury History",
            "value": int(inj_count),
            "impact": "risk_increasing",
            "explanation": f"{inj_count} prior injuries create a meaningful repeat-risk pattern."
        })
    elif inj_count >= 3:
        drivers.append({
            "name": "Injury History",
            "value": int(inj_count),
            "impact": "risk_increasing",
            "explanation": f"{inj_count} prior injuries keep baseline risk above average."
        })
    elif inj_count == 0:
        drivers.append({
            "name": "Injury History",
            "value": 0,
            "impact": "protective",
            "explanation": "No recorded injuries in this dataset, which supports current availability."
        })

    # 4. Days Since Last Injury
    days_since = _safe_int(player_data.get("days_since_last_injury", 365), 365)
    if days_since <= 30:
        drivers.append({
            "name": "Recency",
            "value": int(days_since),
            "impact": "risk_increasing",
            "explanation": f"{days_since} days since last injury means the re-injury window is still open."
        })
    elif days_since <= 60:
        drivers.append({
            "name": "Recency",
            "value": int(days_since),
            "impact": "risk_increasing",
            "explanation": f"{days_since} days since injury; risk is easing but not fully settled."
        })
    elif days_since > 365:
        drivers.append({
            "name": "Recency",
            "value": int(days_since),
            "impact": "protective",
            "explanation": "Over a year injury-free is a strong durability signal."
        })

    # 5. Age
    age = _safe_int(player_data.get("age", 25), 25)
    if age >= 32:
        drivers.append({
            "name": "Age Factor",
            "value": int(age),
            "impact": "risk_increasing",
            "explanation": f"At {age}, recovery margins are usually tighter through heavy schedules."
        })
    elif age <= 21:
        drivers.append({
            "name": "Age Factor",
            "value": int(age),
            "impact": "neutral",
            "explanation": f"At {age}, physical development is still in progress."
        })

    # 6. Acute Load
    acute = _safe_float(player_data.get("acute_load", 0), 0.0)
    if acute >= 3:
        drivers.append({
            "name": "Match Congestion",
            "value": round(acute, 1),
            "impact": "risk_increasing",
            "explanation": "Recent fixture congestion increases fatigue accumulation risk."
        })

    # 7. Injury Prone Flag
    is_ip = _safe_int(player_data.get("is_injury_prone", 0), 0)
    prone_archetypes = {"Injury Prone", "Fragile", "Recurring", "Currently Vulnerable"}
    if (
        is_ip == 1
        and inj_count >= 4
        and days_since <= 240
        and archetype in prone_archetypes
    ):
        drivers.append({
            "name": "Injury Prone Profile",
            "value": "Yes",
            "impact": "risk_increasing",
            "explanation": f"Recurring injury frequency is still part of {first_name}'s short-term risk profile."
        })

    # 8. Spike Flag
    spike = _safe_int(player_data.get("spike_flag", 0), 0)
    if spike == 1:
        drivers.append({
            "name": "Workload Spike",
            "value": "Detected",
            "impact": "risk_increasing",
            "explanation": "A sharp workload spike is detected, which raises short-term setback risk."
        })

    # 9. Add one contextual RAG driver so notes feel player/fixture-specific.
    context_driver = build_dynamic_rag_line(
        player_data,
        extra_context=extra_context,
        section="lab",
        context_chunks=lab_context,
    )
    if context_driver:
        drivers.append({
            "name": "Fixture Context",
            "value": "Live",
            "impact": "neutral",
            "explanation": _as_sentence(context_driver).replace("tracked matches", "meetings"),
        })

    if not drivers:
        return None

    # Sort: risk-increasing first, then protective
    impact_order = {"risk_increasing": 0, "neutral": 1, "protective": 2}
    drivers.sort(key=lambda d: impact_order.get(d["impact"], 1))

    # Build plain-language fallback summary, then let LLM polish it.
    risk_drivers = [d for d in drivers if d["impact"] == "risk_increasing"]
    protective_drivers = [d for d in drivers if d["impact"] == "protective"]

    if risk_drivers:
        driver_names = [d["name"].lower() for d in risk_drivers[:3]]
        if len(driver_names) == 1:
            summary = f"This week, {first_name}'s main pressure point is {driver_names[0]}."
        else:
            summary = (
                f"This week, {first_name}'s key risk drivers are "
                f"{', '.join(driver_names[:-1])} and {driver_names[-1]}."
            )
        if protective_drivers:
            prot_names = [d["name"].lower() for d in protective_drivers[:2]]
            summary += f" What helps: {', '.join(prot_names)}."
    else:
        summary = f"{first_name}'s profile is steady right now, without strong risk spikes."

    llm_context = list(lab_context)
    for d in drivers[:4]:
        llm_context.append({
            "kind": "driver",
            "text": f"{d['name']}: {d['explanation']}",
            "tags": set(),
            "weight": 1.0,
        })
    summary = generate_grounded_narrative(
        task=(
            "Write Yara's Lab Notes summary for builders in plain football language. "
            "Keep it human-readable and avoid jargon. 2-3 short sentences."
        ),
        player_name=name,
        context_chunks=llm_context,
        fallback_text=summary,
        require_open_question=False,
    )

    # Build technical section
    lgb = player_data.get("lgb_prob", prob)
    xgb = player_data.get("xgb_prob", prob)
    cat = player_data.get("catboost_prob", prob)
    model_probs = [float(lgb), float(xgb), float(cat)]
    model_range = max(model_probs) - min(model_probs)
    agreement = max(0, round(1 - model_range / max(max(model_probs), 0.01), 3))

    # Feature highlights for technical section
    tech_features = []
    for feat_name, feat_key in [
        ("ACWR", "acwr"), ("Fatigue Index", "fatigue_index"),
        ("Acute Load", "acute_load"), ("Chronic Load", "chronic_load"),
        ("Monotony", "monotony"), ("Strain", "strain"),
        ("Workload Slope", "workload_slope"), ("Spike Flag", "spike_flag"),
    ]:
        val = player_data.get(feat_key)
        if val is not None and val != 0:
            tech_features.append({"name": feat_name, "value": round(float(val), 3)})

    technical = {
        "model_agreement": agreement,
        "methodology": "",
        "feature_highlights": tech_features[:8],
    }

    return {
        "summary": summary,
        "key_drivers": drivers[:4],
        "technical": technical,
    }
