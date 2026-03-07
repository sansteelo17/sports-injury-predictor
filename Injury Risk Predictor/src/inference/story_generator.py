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
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(float(value))
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
    if any(token in pos for token in ["goalkeeper", "keeper", "gk", "def", "back"]):
        return "defender"
    if any(token in pos for token in ["forward", "fwd", "striker", "winger", "wing"]):
        return "attacker"
    if any(token in pos for token in ["midfielder", "mid", "am", "cm", "dm", "playmaker"]):
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
    return f"{formatted} — " if not unit else f"{formatted} {unit} — "


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
        signals.append((opp_conceded * 1.8, f"{opponent} leaking {opp_conceded:.1f} goals/game"))
    elif opp_conceded >= 1.0:
        signals.append((opp_conceded * 1.0, f"{opponent} conceding {opp_conceded:.1f}/game"))
    elif 0 < opp_conceded < 0.8:
        signals.append((0.5, f"{opponent} tight at {opp_conceded:.1f} conceded/game"))

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
    matchup_context = extra_context.get("matchup_context") or {}
    recent_form = matchup_context.get("recent_form") or {}
    recent_samples = _safe_int(recent_form.get("samples", 0), 0)
    recent_goals = _safe_int(recent_form.get("goals", 0), 0)
    recent_assists = _safe_int(recent_form.get("assists", 0), 0)
    recent_gi = recent_goals + recent_assists
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
        prev_injuries <= 2 and days_since >= 60 and acwr < 1.3 and 21 <= age <= 31
    )

    sentences = []

    # LEAD: stat-first headline — OptaJoe voice
    if prob >= 0.60 and days_since < 60:
        sentences.append(
            f"{_stat_lead(days_since)}{first_name} has been back just {days_since} days "
            f"and already sits at {risk_pct}% risk. The last setback echoes through every metric"
        )
    elif prob >= 0.60 and prev_injuries >= 5:
        sentences.append(
            f"{_stat_lead(prev_injuries)}{first_name} has {prev_injuries} injuries on file, "
            f"costing {_safe_int(days_lost)} days total. That history drives the {risk_pct}% reading"
        )
    elif prob >= 0.60 and acwr > 1.3:
        sentences.append(
            f"ACWR is {acwr:.2f}, just above the 1.30 risk threshold. "
            f"{first_name} sits at {risk_pct}% risk with load rising faster than recovery."
        )
    elif prob >= 0.60 and avg_days >= 30:
        sentences.append(
            f"{_stat_lead(avg_days, 'days')}{first_name} averages {avg_days:.0f} days out per setback "
            f"across {_pl(prev_injuries, 'injury')}. Severity is the headline at {risk_pct}% risk"
        )
    elif prob >= 0.60 and matches_last_30 >= 5:
        sentences.append(
            f"{_stat_lead(matches_last_30)}{first_name} has played {matches_last_30} matches "
            f"in 30 days. The schedule alone pushes risk to {risk_pct}%"
        )
    elif prob >= 0.60:
        sentences.append(
            f"{_stat_lead(risk_pct, '%')}{first_name} reads {risk_pct}% risk — "
            f"{_pl(prev_injuries, 'injury')}, {_safe_int(days_lost)} days lost. Elevated across the board"
        )
    elif prob >= 0.40 and days_since < 60:
        sentences.append(
            f"{_stat_lead(days_since)}{first_name} is only {days_since} days post-injury "
            f"and profiles at {risk_pct}%. Still inside the danger window"
        )
    elif prob >= 0.40 and prev_injuries >= 4:
        sentences.append(
            f"{_stat_lead(prev_injuries)}{prev_injuries} career injuries anchor "
            f"{first_name} at {risk_pct}% risk. Not alarming alone, but the baseline is elevated"
        )
    elif prob >= 0.40 and avg_days >= 30:
        sentences.append(
            f"{_stat_lead(avg_days, 'days')}{first_name} averages {avg_days:.0f} days per layoff. "
            f"At {risk_pct}%, each setback tends to be costly"
        )
    elif prob >= 0.40:
        sentences.append(
            f"{_stat_lead(risk_pct, '%')}{first_name} reads {risk_pct}% risk with "
            f"{_pl(prev_injuries, 'injury')} and {_safe_int(days_lost)} days lost on record"
        )
    elif prob >= 0.25 and prev_injuries >= 3:
        sentences.append(
            f"{_stat_lead(risk_pct, '%')}{first_name} profiles at {risk_pct}%, manageable, "
            f"but {prev_injuries} injuries on file keep the baseline honest"
        )
    elif prob < 0.20 and days_since >= 365:
        sentences.append(
            f"{_stat_lead(days_since, 'days')}Over {days_since} days without injury. "
            f"{first_name} reads just {risk_pct}% — one of the safer picks in the pool"
        )
    elif prob < 0.20:
        sentences.append(
            f"{_stat_lead(risk_pct, '%')}{first_name} reads {risk_pct}% risk. "
            + ("Clean record" if prev_injuries == 0 else f"{_pl(prev_injuries, 'injury')} logged")
            + ", manageable workload"
        )
    else:
        sentences.append(
            f"{_stat_lead(risk_pct, '%')}{first_name} sits at {risk_pct}% with "
            f"{_pl(prev_injuries, 'injury')} and {_safe_int(days_lost)} days lost. Nothing alarming"
        )

    # HISTORY — with specific injury detail when available
    # Analyze injury records for skew, recurring body areas, worst injury
    worst_record = None
    recurring_area = None
    severity_skewed = False
    if injury_records:
        severities = [r["severity_days"] for r in injury_records if r.get("severity_days", 0) > 0]
        if severities and len(severities) >= 2:
            max_sev = max(severities)
            others = [s for s in severities if s != max_sev]
            if others and max_sev >= 45 and max_sev >= 2 * (sum(others) / len(others)):
                severity_skewed = True
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

    if prev_injuries == 0:
        sentences.append(
            "Zero injuries on file, which keeps the baseline risk low."
        )
    elif severity_skewed and worst_record and prev_injuries <= 4:
        worst_area = worst_record.get("injury_raw") or worst_record.get("body_area", "injury")
        worst_days = worst_record["severity_days"]
        worst_date = worst_record.get("date", "")
        date_str = f" in {worst_date[:7]}" if worst_date and len(worst_date) >= 7 else ""
        others_avg = round((days_lost - worst_days) / max(prev_injuries - 1, 1))
        sentences.append(
            f"One long {worst_area}{date_str} layoff ({worst_days} days) pulls the average up to {avg_days:.0f} days. "
            f"Away from that outlier, the other {_pl(prev_injuries - 1, 'injury')} averaged {others_avg} days."
        )
    elif recurring_area and prev_injuries >= 3:
        sentences.append(
            f"{recurring_area.capitalize()} keeps recurring — "
            f"{_pl(prev_injuries, 'injury')}, {_safe_int(days_lost)} days lost. Clear weak point"
        )
    elif worst_record and avg_days >= 40:
        worst_area = worst_record.get("injury_raw") or worst_record.get("body_area", "injury")
        worst_date = worst_record.get("date", "")
        date_str = f" ({worst_date[:7]})" if worst_date and len(worst_date) >= 7 else ""
        sentences.append(
            f"{_stat_lead(worst_record['severity_days'], 'days')}The worst was a {worst_area}{date_str}. "
            f"Average layoff: {avg_days:.0f} days across {_pl(prev_injuries, 'injury')}"
        )
    elif prev_injuries <= 2 and days_since >= 365:
        sentences.append(
            f"Just {_pl(prev_injuries, 'injury')} on record and {_days_fan_label(days_since)} injury-free. "
            "Recent availability has been strong."
        )
    elif prev_injuries <= 2:
        if worst_record:
            worst_area = worst_record.get("injury_raw") or worst_record.get("body_area", "injury")
            sentences.append(
                f"Only {_pl(prev_injuries, 'injury')} logged — the notable one a {worst_area}, "
                f"{worst_record['severity_days']} days out"
            )
        else:
            sentences.append(
                f"Only {_pl(prev_injuries, 'injury')} on record, so the risk read is less noisy than a heavy-injury profile."
            )
    elif prev_injuries >= 8 and avg_days >= 30:
        sentences.append(
            f"{_stat_lead(prev_injuries)}{prev_injuries} injuries at {avg_days:.0f} days each. "
            f"Frequency and severity both flagging"
        )
    elif prev_injuries >= 5:
        sentences.append(
            f"{_stat_lead(_safe_int(days_lost), 'days lost')}{prev_injuries} injuries on file. "
            f"Hard to overlook that volume"
        )
    elif avg_days <= 15 and prev_injuries > 0:
        sentences.append(
            f"Past injuries have been minor — {avg_days:.0f}-day average layoff"
        )

    # RECENCY & WORKLOAD
    lead_mentions_recency = any(
        ("days post-injury" in s.lower()) or ("days since the last injury" in s.lower())
        for s in sentences[:2]
    )
    if not lead_mentions_recency:
        if days_since < 30:
            sentences.append(
                f"{days_since} days since the last injury. Still in the highest-risk window for recurrence"
            )
        elif days_since < 60:
            sentences.append(
                f"{days_since} days post-injury — past the acute zone but not yet clear"
            )
    if acwr >= 1.5:
        sentences.append(
            f"Workload ratio at {acwr:.2f} — that spike is a strong short-term injury warning sign"
        )
    elif acwr >= 1.3 and fatigue >= 1.0:
        sentences.append(
            f"Workload ratio {acwr:.2f} with fatigue at {fatigue:.1f}. The body is doing more than it is conditioned for"
        )

    # WORKLOAD NARRATIVE — explain the "why" even when workload is normal
    if not any("workload" in s.lower() or "fatigue" in s.lower() or "acwr" in s.lower() for s in sentences):
        if matches_last_7 >= 2:
            sentences.append(
                f"{matches_last_7} matches in 7 days. Demanding schedule regardless of the model read"
            )
        elif matches_last_30 >= 6:
            sentences.append(
                f"{matches_last_30} matches in 30 days. Heavy fixture load testing recovery capacity"
            )
        elif is_sparse_profile and acwr > 0:
            if acwr <= 0.8:
                sentences.append(
                    f"Workload ratio {acwr:.2f} — lighter recent load keeps stress low but limits match conditioning"
                )
            else:
                sentences.append(
                    f"Workload ratio {acwr:.2f} — no spike, no fatigue flag. Load is well managed"
                )

    # AGE
    if age >= 34:
        sentences.append(
            f"{age} years old. Recovery margins thin, load management critical"
        )
    elif age >= 32:
        sentences.append(
            f"At {age}, fixture congestion hits harder than it would for a younger body"
        )
    elif age <= 20:
        sentences.append(
            f"At {age}, the body is still developing — protective but unpredictable"
        )

    # FIXTURE — context about the upcoming opponent (all positions)
    if opponent:
        team = _display_team_name(str(player_data.get("team", "") or ""))
        if role in ("defender", "goalkeeper"):
            if opp_conceded >= 1.2 and opp_conceded < 1.8:
                sentences.append(
                    f"{opponent} scoring {opp_conceded:.1f}/game — lighter defensive pressure for {first_name}"
                )
            elif opp_conceded > 0 and opp_conceded < 0.8:
                sentences.append(
                    f"{opponent} at {opp_conceded:.1f} conceded/game. Stern test at the back"
                )
            elif vs_samples >= 1 and vs_cs > 0:
                sentences.append(
                    f"{_pl(vs_cs, 'clean sheet')} in {_pl(vs_samples, 'meeting')} against {opponent}. Positive record"
                )
            elif fixture_samples >= 2 and is_sparse_profile:
                sentences.append(
                    f"{team}: {fh_wins}W {fh_losses}L in {fixture_samples} vs {opponent}"
                )
        else:
            if opp_conceded >= 1.2 and recent_gi >= 3 and recent_samples > 0:
                sentences.append(
                    f"{opponent} conceding {opp_conceded:.1f}/game. "
                    f"{first_name} has {_pl(recent_goals, 'goal')} and {_pl(recent_assists, 'assist')} "
                    f"in the last {recent_samples} — form meets fixture"
                )
            elif opp_conceded >= 1.2:
                sentences.append(
                    f"{opponent} are conceding {opp_conceded:.1f} goals per game lately, so this fixture gives {first_name} a clearer route to returns."
                )
            elif vs_samples >= 1 and vs_goals + vs_assists >= 1:
                sentences.append(
                    f"{_pl(vs_goals, 'goal')} and {_pl(vs_assists, 'assist')} in "
                    f"{_pl(vs_samples, 'meeting')} vs {opponent}"
                    + (". History favours a return" if vs_samples >= 3 and vs_goals + vs_assists >= 3
                       else "")
                )
            elif opp_conceded > 0 and opp_conceded < 0.8:
                sentences.append(
                    f"{opponent} conceding just {opp_conceded:.1f}/game. Tight defence, tough assignment"
                )
            elif fixture_samples >= 2 and is_sparse_profile:
                sentences.append(
                    f"{team}: {fh_wins}W {fh_losses}L in {fixture_samples} vs {opponent}"
                )
            elif opp_conceded >= 0.8 and opp_conceded < 1.2 and is_sparse_profile:
                sentences.append(
                    f"{opponent} at {opp_conceded:.1f} conceded/game — league average, could go either way"
                )

    # RECENT FORM — compensate for thin injury data with performance context
    if is_sparse_profile and recent_samples > 0 and len(sentences) < 5:
        if recent_gi >= 3:
            sentences.append(
                f"{_pl(recent_goals, 'goal')} and {_pl(recent_assists, 'assist')} "
                f"in the last {recent_samples}. Match-sharp and in rhythm"
            )
        elif recent_gi == 0 and recent_samples >= 3:
            sentences.append(
                f"Zero goal involvements in {recent_samples}. Quiet spell worth monitoring"
            )
        elif recent_gi > 0:
            sentences.append(
                f"{_pl(recent_goals, 'goal')} and {_pl(recent_assists, 'assist')} "
                f"in {recent_samples} — ticking along"
            )

    # HOME/AWAY — venue context when fixture block didn't already cover it
    fixture_already_mentioned = any(opponent in s for s in sentences) if opponent else False
    if opponent and is_home is not None and not fixture_already_mentioned:
        if is_home:
            sentences.append(f"Home fixture against {opponent}. Tailwind")
        elif is_sparse_profile:
            sentences.append(f"Away to {opponent}. Small adaptation factor")

    # Minimum sentence guarantee for sparse profiles
    if is_sparse_profile and len(sentences) < 3 and opponent:
        sentences.append(
            f"Thin injury data — the {opponent} fixture and workload carry more weight here"
        )

    # ARCHETYPE
    archetype_flavour = {
        "Currently Vulnerable": (
            "Tagged 'Currently Vulnerable' — re-injury risk stays elevated until a proper run of games lands"
        ),
        "Fragile": (
            f"Yara tags {first_name} as 'Fragile'. When injuries hit, they cost weeks, not days"
        ),
        "Injury Prone": (
            "Tagged 'Injury Prone' — frequency, not one bad spell. The body breaks down more often than average"
        ),
        "Recurring": (
            "Repeat injuries — manageable individually, nagging across a season"
        ),
        "Durable": (
            f"Tagged 'Durable'. {first_name}'s availability record is among the best in the pool"
        ),
    }
    if archetype in archetype_flavour:
        sentences.append(archetype_flavour[archetype])

    # OPEN QUESTION
    context_chunks = retrieve_player_context(
        player_data, extra_context=extra_context, top_k=12, include_open_question=True,
    )
    story_fact_line = build_dynamic_rag_line(
        player_data,
        extra_context=extra_context,
        section="story",
        context_chunks=context_chunks,
    )
    open_question = ""
    for chunk in context_chunks:
        if chunk.get("kind") == "open_question":
            open_question = _as_sentence(chunk.get("text", ""))
            break

    story_body = ". ".join(s.rstrip(".") for s in sentences[:5]) + "."
    if story_fact_line:
        normalized_fact = story_fact_line.strip().lower().rstrip(".")
        normalized_body = story_body.lower()
        if normalized_fact and normalized_fact not in normalized_body:
            story = f"{_as_sentence(story_fact_line)} {story_body}"
        else:
            story = story_body
    else:
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
            "Keep it sharp and specific."
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
    if acwr >= 1.3:
        factors.append({
            "factor": "Workload Spike",
            "impact": "high_risk",
            "description": f"ACWR at {acwr:.2f} — workload has ramped up faster than the body is conditioned for"
        })
    elif 0 < acwr <= 0.8:
        factors.append({
            "factor": "Light Workload",
            "impact": "protective",
            "description": f"ACWR at {acwr:.2f} — recent load is lighter, reducing physical stress"
        })
    elif acwr > 0:
        factors.append({
            "factor": "Managed Workload",
            "impact": "neutral",
            "description": f"ACWR at {acwr:.2f} — workload is in the safe zone"
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
    venue_label = "at home" if is_home else "away" if is_home is not None else ""

    # Stat-led opener in Opta-like style
    if role == "defender" and recent_samples > 0:
        lead = f"{recent_clean_sheets} — {first_name} has {recent_clean_sheets} clean sheets in the last {recent_samples}."
    elif recent_samples > 0:
        lead = f"{recent_gi} — {first_name} has {recent_goals} goals and {recent_assists} assists in the last {recent_samples}."
    elif vs_samples > 0 and role == "defender":
        lead = f"{vs_clean_sheets} — {first_name} has {vs_clean_sheets} clean sheets in {vs_samples} meetings with {opponent}."
    elif vs_samples > 0:
        lead = f"{vs_goals + vs_assists} — {first_name} has {vs_goals} goals and {vs_assists} assists in {vs_samples} meetings with {opponent}."
    else:
        lead = f"{risk_pct}% — Current injury-risk read for {first_name}."

    action = "Start if owned"
    reason_bits: List[str] = []

    if minutes < 180:
        action = "Bench"
        reason_bits.append(f"minutes are too thin ({minutes})")
    elif is_currently_injured or days_since < 21:
        action = "Bench"
        reason_bits.append(f"only {days_since} days since the last setback")
    elif injury_prob >= 0.45:
        action = "Start only with bench cover"
        reason_bits.append("availability risk is still elevated")
    elif role == "defender":
        if recent_clean_sheets >= 2 and injury_prob < 0.35:
            action = "Start"
            reason_bits.append("clean-sheet floor is live")
        elif recent_clean_sheets == 0 and recent_samples >= 4:
            action = "Bench-first"
            reason_bits.append("recent clean-sheet trend is cold")
    else:
        if output_per_90 >= 0.55 and recent_returns >= 2 and injury_prob < 0.35:
            action = "Start"
            reason_bits.append("output trend and availability both support it")
        elif recent_returns == 0 and recent_samples >= 4:
            action = "Bench"
            reason_bits.append("no recent returns")

    # Fixture and matchup context
    if role == "defender":
        if opp_conceded > 0:
            reason_bits.append(f"{opponent} average {opp_conceded:.2f} goals per game recently")
        if vs_samples >= 2:
            reason_bits.append(f"{vs_clean_sheets} clean sheets in {vs_samples} H2H meetings")
    else:
        if opp_conceded >= 1.2:
            reason_bits.append(f"{opponent} have been conceding {opp_conceded:.2f} per game")
        if vs_samples >= 2 and (vs_goals + vs_assists) > 0:
            reason_bits.append(f"{vs_goals + vs_assists} returns in {vs_samples} H2H meetings")
        elif vs_samples >= 2 and vs_returns == 0:
            reason_bits.append("no H2H returns yet in this fixture")

    # Keep tip aligned with value tier to avoid contradictions
    value_assessment = get_fpl_value_assessment(player_data, extra_context=extra_context)
    if value_assessment:
        tier = (value_assessment.get("tier") or "").strip().lower()
        if tier == "avoid":
            action = "Bench"
            reason_bits = ["upside has been too thin for the price and risk profile"]
        elif tier == "rotation" and action == "Start":
            action = "Start only with bench cover"

    # Inject one strong signal-stack clause for variation without bloat
    signals = _build_fpl_signal_stack(
        recent_form=recent_form,
        vs_opponent=vs_opponent,
        opponent_defense=opponent_defense,
        fixture_history=fixture_history,
        is_home=is_home,
        role=role,
        first_name=first_name,
        opponent=opponent,
    )
    if signals:
        reason_bits.append(signals[0])

    context_chunks = retrieve_player_context(
        player_data,
        extra_context=extra_context,
        query="fpl manager tip recent form head-to-head fixture opponent defense injury availability minutes",
        top_k=10,
        include_open_question=False,
    )
    fpl_context_line = build_dynamic_rag_line(
        player_data,
        extra_context=extra_context,
        section="fpl",
        context_chunks=context_chunks,
    )
    if fpl_context_line:
        reason_bits.append(fpl_context_line.rstrip("."))

    # De-duplicate and keep concise
    seen = set()
    concise_reasons: List[str] = []
    for bit in reason_bits:
        clean = re.sub(r"\s+", " ", (bit or "").strip())
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            concise_reasons.append(clean)
        if len(concise_reasons) >= 2:
            break

    fixture_slice = f"{venue_label} vs {opponent}".strip() if venue_label else f"vs {opponent}"
    reason_text = "; ".join(concise_reasons) if concise_reasons else "mixed signal profile this week"
    fallback_text = _as_sentence(
        re.sub(
            r"\s{2,}",
            " ",
            f"{lead} {action} for {first_name} {fixture_slice}: {reason_text}.",
        ).strip(),
    )

    # Optional LLM polish, grounded by section-planned RAG facts.
    polished = generate_grounded_narrative(
        task=(
            "Write one punchy FPL manager tip sentence. "
            "Lead with a decisive stat, then clear action for managers."
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
    low_output = output_signal < (0.18 if role == "defender" else 0.18)
    durability_signal = days_since >= 240 and prev_injuries <= 2

    # Use name hash to deterministically pick phrasing variants
    _variant = hash(name) % 3

    risk_pct = round(injury_prob * 100)

    if tier == "Avoid":
        if low_output and high_risk:
            verdict = (
                f"{_stat_lead(ga_per_90, 'G+A/90')}{first_name} at {risk_pct}% risk. "
                f"Output thin, availability heavy. {_pick_kicker('avoid', first_name)}"
            )
        elif low_output and durability_signal:
            verdict = (
                f"{_stat_lead(days_since, 'days injury-free')}Body is fine. "
                f"Output isn't — {ga_per_90:.2f} G+A/90 at £{price:.1f}m. {_pick_kicker('avoid', first_name)}"
            )
        elif low_output:
            verdict = (
                f"{_stat_lead(ga_per_90, 'G+A/90')}At £{price:.1f}m, {first_name} needs more. "
                f"{_pick_kicker('avoid', first_name)}"
            )
        elif high_risk:
            verdict = (
                f"{_stat_lead(risk_pct, '% risk')}Injury drag swallows the upside for {first_name}. "
                f"{_pick_kicker('avoid', first_name)}"
            )
        else:
            verdict = (
                f"{_stat_lead(adjusted_value)}Projected return too low for £{price:.1f}m. "
                f"Reinvest elsewhere. {_pick_kicker('avoid', first_name)}"
            )
    elif tier == "Rotation":
        if low_output:
            verdict = (
                f"{_stat_lead(ga_per_90, 'G+A/90')}Minutes are usable, upside isn't. "
                f"Bench fodder. {_pick_kicker('default', first_name)}"
            )
        elif high_risk:
            verdict = (
                f"{_stat_lead(risk_pct, '% risk')}Talent is there but the body needs more runway. "
                f"Bench and revisit. {_pick_kicker('moderate_risk', first_name)}"
            )
        else:
            verdict = (
                f"Fixture-dependent for {first_name}. Better as depth than a locked starter. "
                f"{_pick_kicker('default', first_name)}"
            )
    elif tier == "Decent":
        if high_risk:
            verdict = (
                f"{_stat_lead(risk_pct, '% risk')}Decent value when fit, "
                f"but availability is the wildcard. Need a backup plan. {_pick_kicker('moderate_risk', first_name)}"
            )
        else:
            verdict = (
                f"{_stat_lead(ga_per_90, 'G+A/90')}Balanced upside and risk for {first_name}. "
                f"Reliable floor at £{price:.1f}m. {_pick_kicker('default', first_name)}"
            )
    elif tier == "Strong":
        if recent_returns >= 3 and recent_samples > 0:
            verdict = (
                f"{_stat_lead(recent_returns)}{recent_returns} returns in {recent_samples} for {first_name}. "
                f"The form is real, the body is holding. {_pick_kicker('form_hot', first_name)}"
            )
        else:
            verdict = (
                f"{_stat_lead(ga_per_90, 'G+A/90')}Consistent output, manageable risk at £{price:.1f}m. "
                f"{first_name} is quietly one of the better value picks. {_pick_kicker('premium', first_name)}"
            )
    else:  # Premium
        if goals_per_90 >= 0.5:
            verdict = (
                f"{_stat_lead(goals_per_90, 'goals/90')}{first_name} at {risk_pct}% risk. "
                f"The maths prints money. Armband candidate. {_pick_kicker('premium', first_name)}"
            )
        else:
            verdict = (
                f"{_stat_lead(adjusted_value)}Set and forget {first_name}. "
                f"Output ceiling and floor both elite for £{price:.1f}m. {_pick_kicker('premium', first_name)}"
            )

    if archetype == "Currently Vulnerable" and days_since < 60:
        verdict = (
            f"{_stat_lead(days_since, 'days')}{first_name} is back but the return window caps FPL trust. "
            f"Give it another week. {_pick_kicker('moderate_risk', first_name)}"
        )
    elif injury_prone and days_since < 120 and tier in {"Decent", "Strong", "Premium"}:
        verdict = (
            f"{_stat_lead(prev_injuries, 'injuries')}{avg_days_per_injury:.0f}-day average layoff. "
            f"Real upside for {first_name}, high volatility. Need bench insurance. {_pick_kicker('moderate_risk', first_name)}"
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
        market_odds_decimal, bookmaker — or None if insufficient data
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
        # No market data — projection only
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
    """Yara sees value — her projection exceeds market."""
    edge = round((yara - market) * 100, 1)

    if days_since < 60:
        context = "Recent return adds uncertainty, but underlying output is there."
    elif inj_prob < 0.25:
        context = "Fitness profile is clean — no red flags."
    elif g90 >= 0.5:
        context = f"{g90:.2f} goals/90 backs it up."
    elif archetype == "Durable":
        context = "Durability is underpriced."
    else:
        venue = "at home" if is_home else "away"
        context = f"Playing {venue} against {opponent} favours the over."

    return (
        f"{_stat_lead(edge, '%')}That's the gap between my model and the bookies on {name}. "
        f"I have {yara*100:.0f}%, they have {market*100:.0f}%. {context} "
        f"{_pick_kicker('form_hot', name)}"
    )


def _yara_market_generous(name, yara, market, opponent, inj_prob, archetype):
    """Market is more generous than Yara's model."""
    gap = round((market - yara) * 100, 1)

    if inj_prob >= 0.45:
        reason = f"Injury risk at {inj_prob*100:.0f}% is non-trivial."
    elif archetype in ("Fragile", "Currently Vulnerable"):
        reason = f"The {archetype.lower()} tag means availability isn't guaranteed."
    else:
        reason = "Underlying scoring rate doesn't justify the market price."

    return (
        f"{_stat_lead(gap, '%')}Market overrates {name} by {gap}%. "
        f"They say {market*100:.0f}%, I'm at {yara*100:.0f}%. {reason} "
        f"{_pick_kicker('avoid', name)}"
    )


def _yara_aligned(name, yara, market, opponent, inj_prob):
    """Market and model agree."""
    if inj_prob < 0.3:
        note = "Low injury risk is one tailwind not fully priced in."
    elif inj_prob >= 0.45:
        note = "Watch the teamsheet — availability is the only variable."
    else:
        note = "No edge either way. Move on to better spots."

    return (
        f"~{yara*100:.0f}% — market and model agree on {name}. {note} "
        f"{_pick_kicker('default', name)}"
    )


def _yara_no_market(name, yara, inj_prob, g90, a90, archetype, days_since):
    """No market odds available — projection only."""
    if g90 >= 0.5 and inj_prob < 0.35:
        tone = (
            f"{_stat_lead(yara * 100, '%')}No market line, but {name} scores at {g90:.2f}/90 "
            f"and the body is holding. {_pick_kicker('form_hot', name)}"
        )
    elif g90 >= 0.3 and inj_prob < 0.4:
        tone = (
            f"{_stat_lead(yara * 100, '%')}No market comparison. "
            f"Decent output, manageable risk. {_pick_kicker('default', name)}"
        )
    elif inj_prob >= 0.45:
        tone = (
            f"{_stat_lead(inj_prob * 100, '% risk')}No market line. "
            f"Only {yara*100:.0f}% to score — injury is the headline. {_pick_kicker('high_risk', name)}"
        )
    elif days_since < 60:
        tone = (
            f"{_stat_lead(days_since, 'days back')}No market data. "
            f"Projecting {yara*100:.0f}% but still in the return window. {_pick_kicker('moderate_risk', name)}"
        )
    else:
        involvement = (g90 + a90) * 100
        tone = (
            f"{_stat_lead(yara * 100, '%')}No market line. "
            f"{involvement:.0f}% for any goal involvement. {_pick_kicker('default', name)}"
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
        return f"Just back from injury. {name} may be eased in — don't expect full 90s. Wait a week before committing."

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
        Dict with summary, key_drivers, technical — or None if insufficient data
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
    if acwr >= 1.25:
        drivers.append({
            "name": "Workload Ratio",
            "value": round(acwr, 2),
            "impact": "risk_increasing",
            "explanation": f"Workload ratio is {acwr:.2f}, above the safe range, so physical stress is running hot."
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
