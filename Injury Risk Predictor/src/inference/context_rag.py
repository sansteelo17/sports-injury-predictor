"""Lightweight retrieval helpers for context-rich player narratives.

This is an intentionally small, dependency-free RAG layer:
- Build context chunks from structured player/fixture/market data
- Retrieve the most relevant chunks with lexical scoring
- Return grounded facts that narrative generators can blend
"""

from __future__ import annotations

import random
import re
import time
from typing import Any, Dict, List, Optional, Set


DEFAULT_STORY_QUERY = (
    "injury risk history severity recency durability workload availability "
    "fixture opponent market odds scoring output"
)


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
    # Keep football-style short names in narrative copy.
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

_LAST_OPEN_QUESTION: Dict[str, str] = {}
_LAST_SECTION_RAG_LINE: Dict[str, str] = {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _count_phrase(count: int, singular: str, plural: Optional[str] = None) -> str:
    label = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {label}"


def _days_to_years_label(days: int) -> str:
    if days < 365:
        return f"{days} days"
    years = days / 365.0
    if years >= 3:
        return f"Over {years:.1f} years"
    if years >= 2:
        return f"About {years:.1f} years"
    return "Over 1 year"


def _parse_scoreline(score: str) -> Optional[tuple[int, int]]:
    text = (score or "").strip()
    match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _call_name(player_data: Dict[str, Any]) -> str:
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


def _display_team_name(team_name: str) -> str:
    return DISPLAY_TEAM_NAME_OVERRIDES.get(team_name, team_name)


def _position_group(position: Any) -> str:
    pos = str(position or "").strip().lower()
    if not pos:
        return "other"

    defender_tokens = [
        "goalkeeper",
        "keeper",
        "gk",
        "def",
        "back",
        "centre-back",
        "center-back",
        "full-back",
        "fullback",
        "wing-back",
        "wingback",
    ]
    attacker_tokens = [
        "forward",
        "fwd",
        "striker",
        "winger",
        "wing",
        "centre-forward",
        "center-forward",
        "inside forward",
        "attacker",
    ]
    midfielder_tokens = [
        "midfielder",
        "mid",
        "cm",
        "dm",
        "am",
        "playmaker",
    ]

    if any(token in pos for token in defender_tokens):
        return "defender"
    if any(token in pos for token in attacker_tokens):
        return "attacker"
    if any(token in pos for token in midfielder_tokens):
        return "midfielder"
    return "other"


def _statement(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    # Section RAG lines for odds/FPL should stay declarative.
    if value.endswith("?"):
        value = value[:-1].rstrip()
    if not value:
        return ""
    if value.endswith((".", "!")):
        return value
    return f"{value}."


def _choose_variation(candidates: List[str], salt: str, avoid: Optional[str] = None) -> Optional[str]:
    options = [c.strip() for c in candidates if (c or "").strip()]
    if not options:
        return None
    if avoid and len(options) > 1:
        filtered = [o for o in options if o != avoid]
        if filtered:
            options = filtered
    entropy = f"{salt}:{time.time_ns()}:{random.random()}"
    idx = abs(hash(entropy)) % len(options)
    return options[idx]


def _build_dynamic_open_question(
    call_name: str,
    team_name: Optional[str],
    player_position: str,
    matchup_opponent: Optional[str],
    recent_form: Dict[str, Any],
    vs_opponent: Dict[str, Any],
    fixture_history: Dict[str, Any],
    fixture_samples: int,
    all_time_samples: int,
    days_since: int,
    market_prob: float,
    opponent_defense_avg_conceded: float = 0.0,
    opponent_defense_samples: int = 0,
) -> str:
    opp = _display_team_name(matchup_opponent) if matchup_opponent else None
    team_display = _display_team_name(team_name) if team_name else "their side"
    role = _position_group(player_position)
    recent_samples = _safe_int(recent_form.get("samples", 0))
    recent_goals = _safe_int(recent_form.get("goals", 0))
    recent_assists = _safe_int(recent_form.get("assists", 0))
    recent_avg_points = _safe_float(recent_form.get("avg_points", 0.0), 0.0)
    recent_clean_sheets = _safe_int(recent_form.get("clean_sheets", 0))
    vs_samples = _safe_int(vs_opponent.get("samples", 0))
    vs_goals = _safe_int(vs_opponent.get("goals", 0))
    vs_assists = _safe_int(vs_opponent.get("assists", 0))
    vs_returns = _safe_int(vs_opponent.get("returns", 0))
    vs_clean_sheets = _safe_int(vs_opponent.get("clean_sheets", 0))
    market_pct = round(market_prob * 100) if market_prob > 0 else 0

    candidates: List[str] = []

    if role == "defender":
        recent_cs_phrase = (
            f"after {_count_phrase(recent_clean_sheets, 'clean sheet')} in the last {recent_samples}"
            if recent_samples > 0
            else "on current defensive form"
        )
        if opp and recent_samples > 0:
            if recent_clean_sheets > 0:
                candidates.extend([
                    f"After {_count_phrase(recent_clean_sheets, 'clean sheet')} in the last {recent_samples}, can {call_name} anchor another one against {opp}?",
                    f"{recent_clean_sheets} shutouts from the last {recent_samples}: is {call_name} set for another defensive return versus {opp}?",
                    f"After {_count_phrase(recent_clean_sheets, 'clean sheet')} recently, can {call_name} turn this {opp} matchup into another one?",
                ])
            else:
                candidates.extend([
                    f"No clean sheets in the last {recent_samples}: can {call_name} and the back line flip that against {opp}?",
                    f"Is this the fixture where {call_name} breaks the recent clean-sheet drought versus {opp}?",
                ])
        if opp and vs_samples >= 2:
            if recent_samples > 0:
                candidates.extend([
                    (
                        f"After {_count_phrase(recent_clean_sheets, 'clean sheet')} in the last {recent_samples}, and "
                        f"{_count_phrase(vs_clean_sheets, 'clean sheet')} in {vs_samples} meetings with {opp}, "
                        f"can {call_name} add another?"
                    ),
                    (
                        f"{_count_phrase(recent_clean_sheets, 'clean sheet')} recently plus "
                        f"{_count_phrase(vs_clean_sheets, 'shutout')} in {vs_samples} vs {opp}: "
                        f"does {call_name} lock this down again?"
                    ),
                ])
            else:
                candidates.extend([
                    f"{_count_phrase(vs_clean_sheets, 'clean sheet')} in {vs_samples} meetings with {opp} - can {call_name} add another?",
                    f"Does {call_name} keep this tight against {opp} after {_count_phrase(vs_clean_sheets, 'prior shutout')} in this matchup?",
                ])
        if opp and (fixture_samples >= 3 or all_time_samples >= 5):
            total_ga = _safe_int(
                fixture_history.get("goals_against" if fixture_samples >= 3 else "all_time_goals_against", 0)
            )
            sample_n = fixture_samples if fixture_samples >= 3 else all_time_samples
            ga_per_match = total_ga / max(1, sample_n)
            recent_cs_line = (
                f"with {_count_phrase(recent_clean_sheets, 'clean sheet')} in the last {recent_samples}"
                if recent_samples > 0
                else "with current defensive form"
            )
            candidates.extend([
                f"{opp} average {ga_per_match:.2f} goals in this matchup. But {recent_cs_line}, can {call_name} and {team_display} keep the door shut this time?",
                f"{recent_cs_phrase.capitalize()}, can {call_name} convert this matchup history into a clean-sheet outcome against {opp}?",
            ])
        if opp and market_prob > 0:
            candidates.extend([
                f"{recent_cs_phrase.capitalize()}, model clean-sheet probability is around {market_pct}% - does {call_name} turn that into real points against {opp}?",
                f"With {_count_phrase(recent_clean_sheets, 'clean sheet')} recently and odds near {market_pct}%, can {call_name} deliver the defensive haul versus {opp}?",
            ])
    elif role == "attacker":
        recent_attack_phrase = (
            f"{recent_goals} goals and {recent_assists} assists in the last {recent_samples}"
            if recent_samples > 0
            else "limited recent attacking sample"
        )
        opponent_defense_phrase = (
            f"{opp} are conceding {opponent_defense_avg_conceded:.2f} per game lately"
            if opp and opponent_defense_samples > 0 and opponent_defense_avg_conceded > 0
            else ""
        )
        h2h_attack_phrase = (
            f"{_count_phrase(vs_goals, 'goal')} and {_count_phrase(vs_assists, 'assist')} in {vs_samples} meetings vs {opp}"
            if opp and vs_samples >= 2
            else (f"H2H history vs {opp} is still thin" if opp else "H2H history is limited")
        )
        if opp and recent_samples > 0:
            if (recent_goals + recent_assists) > 0:
                candidates.extend([
                    (
                        f"{call_name} has {recent_goals} goals and {recent_assists} assists in the last {recent_samples}; "
                        f"does that form produce another return against {opp}?"
                    ),
                    (
                        f"With {recent_goals + recent_assists} goal involvements in the last {recent_samples}, "
                        f"can {call_name} hit again versus {opp}?"
                    ),
                ])
                if opponent_defense_phrase:
                    candidates.append(
                        f"{call_name} has {recent_goals + recent_assists} involvements in the last {recent_samples}, and {opponent_defense_phrase}; does {call_name} return again?"
                    )
            else:
                candidates.extend([
                    f"No goals or assists in the last {recent_samples} - does {call_name} break through against {opp} now?",
                    f"Can {call_name} end the recent output drought in this matchup with {opp}?",
                ])
        if opp and vs_samples >= 2:
            if (vs_goals + vs_assists) > 0:
                candidates.extend([
                    f"{_count_phrase(vs_goals, 'goal')} and {_count_phrase(vs_assists, 'assist')} in {vs_samples} matches vs {opp} - does {call_name} cash again?",
                    f"Can {call_name} lean on that H2H output profile against {opp} one more time?",
                ])
            else:
                candidates.extend([
                    f"No tracked H2H returns for {call_name} versus {opp} yet - is this the breakout fixture?",
                ])
        if opp and market_prob > 0:
            candidates.extend([
                f"With {recent_attack_phrase} and {h2h_attack_phrase}, can {call_name} beat a market line near {market_pct}% against {opp}?",
                f"{recent_attack_phrase.capitalize()} plus {h2h_attack_phrase}: does {call_name} outpace the implied {market_pct}% output chance versus {opp}?",
            ])
    elif role == "midfielder":
        if opp and recent_samples > 0:
            candidates.extend([
                f"Can {call_name} run this game from midfield and turn control into returns against {opp}?",
                (
                    f"{_count_phrase(recent_goals, 'goal')} and {_count_phrase(recent_assists, 'assist')} in the last {recent_samples}: "
                    f"does {call_name} turn midfield volume into returns against {opp}?"
                ),
                f"At {recent_avg_points:.1f} average FPL points recently, does {call_name} convert midfield volume into output against {opp}?",
            ])
            if opponent_defense_samples > 0 and opponent_defense_avg_conceded > 0:
                candidates.append(
                    f"With {opp} conceding {opponent_defense_avg_conceded:.2f} per game lately, does {call_name} find the final pass or shot that decides this fixture?"
                )
        if opp and vs_samples >= 2:
            candidates.extend([
                (
                    f"In {vs_samples} meetings with {opp}, {call_name} has "
                    f"{_count_phrase(vs_goals, 'goal')}, {_count_phrase(vs_assists, 'assist')}, "
                    f"and {_count_phrase(vs_returns, 'return match')} - does that profile hold?"
                ),
                f"Can {call_name} dictate the central zones against {opp} and turn it into points?",
            ])
        if opp and market_prob > 0:
            candidates.extend([
                f"With market expectation near {market_pct}%, does {call_name} outplay that line through goal involvement against {opp}?",
                f"Is {call_name} the midfielder who can beat a {market_pct}% line in this {opp} fixture?",
            ])
    else:
        if opp and vs_samples >= 2:
            candidates.extend([
                f"Does {call_name} bend this matchup script again against {opp}?",
                f"Can {call_name} turn prior output against {opp} into another return night?",
                f"Is this the game where {call_name} adds a new chapter to this {opp} matchup?",
            ])

        if opp and recent_samples > 0:
            candidates.extend([
                f"After {recent_goals} goals and {recent_assists} assists in the last {recent_samples}, does {call_name} break loose against {opp}?",
                f"Can {call_name} flip this {recent_avg_points:.1f}-point trend into real output versus {opp}?",
                f"Is this {opp} fixture where {call_name}'s process finally converts into points?",
            ])

        if opp and (fixture_samples >= 3 or all_time_samples >= 5):
            if fixture_samples >= 3:
                fh_wins = _safe_int(fixture_history.get("wins", 0))
                fh_draws = _safe_int(fixture_history.get("draws", 0))
                fh_losses = _safe_int(fixture_history.get("losses", 0))
            else:
                fh_wins = _safe_int(fixture_history.get("all_time_wins", 0))
                fh_draws = _safe_int(fixture_history.get("all_time_draws", 0))
                fh_losses = _safe_int(fixture_history.get("all_time_losses", 0))
            trend = f"{fh_wins}-{fh_draws}-{fh_losses}"
            candidates.extend([
                f"Does that {trend} trend versus {opp} still carry signal this week?",
                f"Can {call_name} tilt this tie again, or does {opp} reset the pattern?",
            ])

    if days_since < 60:
        if role == "defender":
            candidates.extend([
                f"Can {call_name} handle the recovery load and still deliver a clean-sheet performance in this spot?",
            ])
        elif role == "attacker":
            candidates.extend([
                f"Can {call_name} manage the load and still produce attacking output in this matchup?",
            ])
        elif role == "midfielder":
            candidates.extend([
                f"Does {call_name} have the legs to control midfield and return despite the tight recovery window?",
            ])
        else:
            candidates.extend([
                f"Can {call_name} push full intensity without a setback in this spot?",
                f"Will {call_name}'s body hold up under this week's intensity spike?",
            ])

    if market_prob > 0 and role == "other":
        candidates.extend([
            f"Is the market underrating {call_name}'s true injury-adjusted ceiling here?",
            f"Are bookies a beat behind {call_name}'s real game-state upside?",
            f"Does {call_name} outperform a market line sitting around {market_pct}%?",
        ])

    if not candidates:
        candidates = [
            f"What does this fixture really reveal about {call_name}'s risk-reward profile?",
            f"Is this the spot where {call_name}'s underlying numbers finally tell the truth?",
            f"Can {call_name} convert availability into meaningful returns this week?",
        ]

    key = f"{call_name.lower()}|{(opp or '').lower()}|{role}"
    last_question = _LAST_OPEN_QUESTION.get(key)
    selected = _choose_variation(
        candidates,
        salt=(
            f"{call_name}:{role}:{opp}:{recent_samples}:{vs_samples}:{fixture_samples}:"
            f"{all_time_samples}:{days_since}:{recent_clean_sheets}:{vs_clean_sheets}:{round(market_prob, 3)}"
        ),
        avoid=last_question,
    )
    final_question = selected or f"Can {call_name} turn this fixture into a statement performance?"
    _LAST_OPEN_QUESTION[key] = final_question
    return final_question


def _add_chunk(
    chunks: List[Dict[str, Any]],
    kind: str,
    text: str,
    tags: List[str],
    weight: float,
) -> None:
    if not text:
        return
    chunks.append(
        {
            "kind": kind,
            "text": text.strip(),
            "tags": set(tags),
            "weight": float(weight),
        }
    )


def build_player_context_chunks(
    player_data: Dict[str, Any],
    extra_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Create grounded context chunks from player, fixture, and market data."""
    extra_context = extra_context or {}
    chunks: List[Dict[str, Any]] = []

    name = player_data.get("name", "This player")
    call_name = _call_name(player_data)
    archetype = player_data.get("archetype", "Unknown")
    player_position = str(player_data.get("position", "") or "")
    position_group = _position_group(player_position)

    prob = _safe_float(player_data.get("ensemble_prob"), 0.5)
    prev_injuries = _safe_int(player_data.get("previous_injuries", player_data.get("player_injury_count", 0)))
    total_days_lost = _safe_int(player_data.get("total_days_lost", 0))
    days_since = _safe_int(player_data.get("days_since_last_injury", 365))
    worst_injury = _safe_float(player_data.get("player_worst_injury", 0))
    avg_days = (total_days_lost / prev_injuries) if prev_injuries > 0 else 0.0

    goals_per_90 = _safe_float(player_data.get("goals_per_90", 0.0))
    assists_per_90 = _safe_float(player_data.get("assists_per_90", 0.0))
    acwr = _safe_float(player_data.get("acwr", 0.0))
    fatigue = _safe_float(player_data.get("fatigue_index", 0.0))

    # Core risk and injury-history context
    risk_pct = round(prob * 100)
    if prob >= 0.45:
        risk_text = f"{call_name} sits at {risk_pct}% injury risk — firmly in the elevated bracket."
    elif prob >= 0.30:
        risk_text = f"{call_name} profiles at {risk_pct}% risk — moderate but worth watching."
    else:
        risk_text = f"{call_name} is at {risk_pct}% risk, one of the safer profiles in the pool."
    _add_chunk(chunks, "risk_headline", risk_text, ["risk", "probability", "headline"], 3.0)

    if prev_injuries == 0:
        _add_chunk(
            chunks, "history",
            f"No injury history on file for {call_name}, which keeps the baseline low but also means less data to work with.",
            ["history", "injury", "protective"], 2.7,
        )
    elif prev_injuries >= 8 and avg_days >= 30:
        _add_chunk(
            chunks, "history",
            f"{prev_injuries} career injuries averaging {avg_days:.0f} days out each time — the frequency and severity are both concerning.",
            ["history", "injury", "severity"], 2.8,
        )
    elif prev_injuries >= 5:
        _add_chunk(
            chunks, "history",
            f"A track record of {prev_injuries} injuries with {total_days_lost} total days lost is hard to overlook.",
            ["history", "injury", "severity"], 2.8,
        )
    elif avg_days >= 40:
        _add_chunk(
            chunks, "history",
            f"Only {prev_injuries} injuries on record, but they've been significant — averaging {avg_days:.0f} days out each time.",
            ["history", "injury", "severity"], 2.8,
        )
    else:
        _add_chunk(
            chunks, "history",
            f"{prev_injuries} recorded injuries with {total_days_lost} total days lost ({avg_days:.0f} days per injury on average).",
            ["history", "injury", "severity"], 2.8,
        )

    # Per-injury detail chunks (body area, dates, skew detection)
    injury_records = extra_context.get("injury_records") or []
    if injury_records:
        severities = [r["severity_days"] for r in injury_records if r.get("severity_days", 0) > 0]
        # Worst injury detail
        worst = max(injury_records, key=lambda r: r.get("severity_days", 0)) if injury_records else None
        if worst and worst.get("severity_days", 0) > 0:
            worst_area = worst.get("injury_raw") or worst.get("body_area", "injury")
            worst_date = worst.get("date", "")
            date_str = f" in {worst_date[:7]}" if worst_date and len(worst_date) >= 7 else ""
            _add_chunk(chunks, "worst_injury",
                       f"{call_name}'s worst injury was a {worst_area}{date_str}, "
                       f"out for {worst['severity_days']} days and missing {worst.get('games_missed', 0)} games.",
                       ["injury", "history", "severity", "detail"], 2.8)

        # Severity skew detection
        if len(severities) >= 2:
            max_sev = max(severities)
            others = [s for s in severities if s != max_sev]
            if others and max_sev >= 2 * (sum(others) / len(others)):
                others_avg = round(sum(others) / len(others))
                _add_chunk(chunks, "severity_skew",
                           f"The {avg_days:.0f}-day average severity is skewed: "
                           f"one injury lasted {max_sev} days while the others averaged {others_avg}.",
                           ["injury", "severity", "skew", "context"], 2.9)

        # Recurring body area
        areas = [r["body_area"] for r in injury_records if r.get("body_area") and r["body_area"] != "unknown"]
        if areas:
            from collections import Counter
            area_counts = Counter(areas)
            top_area, top_count = area_counts.most_common(1)[0]
            if top_count >= 2:
                _add_chunk(chunks, "recurring_area",
                           f"{call_name} has had {top_count} {top_area} injuries — a recurring weak point.",
                           ["injury", "body_area", "recurring", "pattern"], 2.7)

        # Most recent injury
        if injury_records[0].get("severity_days", 0) > 0:
            recent = injury_records[0]
            r_area = recent.get("injury_raw") or recent.get("body_area", "injury")
            r_date = recent.get("date", "")
            date_str = f" ({r_date})" if r_date else ""
            _add_chunk(chunks, "last_injury_detail",
                       f"{call_name}'s most recent injury was a {r_area}{date_str}, "
                       f"out for {recent['severity_days']} days.",
                       ["injury", "recent", "detail"], 2.6)

    if prev_injuries <= 2 and prev_injuries > 0:
        if days_since >= 365:
            _add_chunk(
                chunks, "sample_sensitivity",
                f"Just {prev_injuries} injuries logged with a {days_since}-day healthy run since — the small sample actually paints a positive picture.",
                ["sample", "history", "uncertainty"], 2.9,
            )
        else:
            _add_chunk(
                chunks, "sample_sensitivity",
                f"Only {prev_injuries} injuries on record, so one bad spell can skew the averages significantly.",
                ["sample", "history", "uncertainty"], 2.9,
            )

    if days_since < 30:
        _add_chunk(
            chunks, "recency",
            f"At just {days_since} days post-injury, {call_name} is still in the highest-risk window for a recurrence.",
            ["recency", "injury", "risk"], 2.8,
        )
    elif days_since < 60:
        _add_chunk(
            chunks, "recency",
            f"{call_name} is {days_since} days post-injury — past the acute danger zone but not yet fully in the clear.",
            ["recency", "injury", "risk"], 2.8,
        )
    elif days_since >= 365:
        _add_chunk(
            chunks, "recency",
            f"{_days_to_years_label(days_since)} injury-free is a strong availability signal — the body has shown it can handle sustained demands.",
            ["recency", "availability", "protective"], 2.8,
        )

    if avg_days >= 45 or worst_injury >= 60:
        _add_chunk(
            chunks, "severity_pattern",
            f"When {call_name} does get injured, it tends to be serious — the worst spell lasted {worst_injury:.0f} days.",
            ["severity", "history", "risk"], 2.6,
        )
    elif prev_injuries > 0 and avg_days <= 20:
        _add_chunk(
            chunks, "severity_pattern",
            "Past injuries have typically been minor and resolved quickly, which is reassuring.",
            ["severity", "recovery", "protective"], 2.5,
        )

    _add_chunk(
        chunks, "archetype",
        f"Yara profiles {call_name} as '{archetype}'.",
        ["archetype", "profile"], 1.8,
    )

    # Workload context
    if acwr >= 1.5:
        _add_chunk(
            chunks, "workload",
            "The workload ratio has spiked recently — one of the strongest predictors of soft-tissue injuries in the research.",
            ["workload", "acwr", "risk"], 2.5,
        )
    elif acwr >= 1.35:
        _add_chunk(
            chunks, "workload",
            f"Workload ratio is creeping up (ACWR {acwr:.2f}), suggesting the body is being asked to do more than it's conditioned for.",
            ["workload", "acwr", "risk"], 2.5,
        )
    elif acwr > 0:
        _add_chunk(
            chunks, "workload",
            f"Workload is well managed (ACWR {acwr:.2f}) — no red flags from the load data.",
            ["workload", "acwr", "protective"], 2.0,
        )

    if fatigue >= 1.0:
        _add_chunk(
            chunks, "fatigue",
            "Fatigue indicators are elevated, which compresses the recovery margins between matches.",
            ["fatigue", "workload", "risk"], 2.2,
        )

    # Output context for attacking players
    goals = _safe_int(player_data.get("goals", 0))
    assists = _safe_int(player_data.get("assists", 0))
    if goals >= 10:
        _add_chunk(
            chunks, "output",
            f"{call_name} has been prolific with {goals} goals this season.",
            ["output", "goals", "performance"], 2.1,
        )
    elif goals > 0 or assists > 0:
        _add_chunk(
            chunks, "output",
            f"{call_name} has {goals} goals and {assists} assists this season.",
            ["output", "goals", "assists", "performance"], 2.1,
        )
    elif goals_per_90 > 0 or assists_per_90 > 0:
        _add_chunk(
            chunks, "output",
            f"Attacking output sits at {goals_per_90:.2f} goals/90 and {assists_per_90:.2f} assists/90.",
            ["output", "goals", "assists", "performance"], 2.1,
        )

    # Fixture context
    next_fixture = extra_context.get("next_fixture") or {}
    opponent = next_fixture.get("opponent")
    is_home = next_fixture.get("is_home")
    if opponent:
        venue = "at home vs" if is_home else "away at"
        _add_chunk(
            chunks,
            "fixture",
            f"Next fixture is {venue} {opponent}.",
            ["fixture", "opponent", "match"],
            2.0,
        )

    fixture_history = extra_context.get("fixture_history") or {}
    fixture_samples = _safe_int(fixture_history.get("samples", 0))
    all_time_samples = _safe_int(fixture_history.get("all_time_samples", fixture_samples))
    if fixture_samples > 0:
        fh_team = _display_team_name(fixture_history.get("team") or player_data.get("team", "Team"))
        fh_opp = _display_team_name(fixture_history.get("opponent") or opponent or "opponent")
        fh_wins = _safe_int(fixture_history.get("wins", 0))
        fh_draws = _safe_int(fixture_history.get("draws", 0))
        fh_losses = _safe_int(fixture_history.get("losses", 0))
        fh_gf = _safe_int(fixture_history.get("goals_for", 0))
        fh_ga = _safe_int(fixture_history.get("goals_against", 0))
        fh_period = fixture_history.get("period_label", "recent history")

        if fixture_samples >= 2:
            _add_chunk(
                chunks,
                "fixture_history",
                (
                    f"{fh_team} vs {fh_opp} ({fh_period}): "
                    f"{fh_wins}-{fh_draws}-{fh_losses} in W-D-L, goals {fh_gf}-{fh_ga} "
                    f"across {fixture_samples} meetings."
                ),
                ["fixture", "history", "head_to_head", "opponent"],
                2.9,
            )
        else:
            _add_chunk(
                chunks,
                "fixture_history",
                (
                    f"Only {fixture_samples} recent meeting for {fh_team} vs {fh_opp}, "
                    "so there is not enough recent history to call a clear trend yet."
                ),
                ["fixture", "history", "sample", "uncertainty"],
                2.5,
            )

        if all_time_samples > fixture_samples and all_time_samples >= 3:
            at_wins = _safe_int(fixture_history.get("all_time_wins", 0))
            at_draws = _safe_int(fixture_history.get("all_time_draws", 0))
            at_losses = _safe_int(fixture_history.get("all_time_losses", 0))
            at_gf = _safe_int(fixture_history.get("all_time_goals_for", 0))
            at_ga = _safe_int(fixture_history.get("all_time_goals_against", 0))
            _add_chunk(
                chunks,
                "fixture_history_all_time",
                (
                    f"All-time PL matchup context: {fh_team} vs {fh_opp} stands at "
                    f"{at_wins}-{at_draws}-{at_losses} in W-D-L with goals {at_gf}-{at_ga} "
                    f"across {all_time_samples} meetings."
                ),
                ["fixture", "history", "head_to_head", "all_time"],
                2.7,
            )

        recent_meetings = fixture_history.get("recent_meetings") or []
        if recent_meetings:
            latest = recent_meetings[-1]
            latest_score = latest.get("score", "")
            latest_date = latest.get("date", "")
            if latest_score:
                latest_result = str(latest.get("result", "")).upper()
                if latest_result not in {"W", "L", "D"}:
                    parsed = _parse_scoreline(latest_score)
                    if parsed:
                        team_goals, opp_goals = parsed
                        if team_goals > opp_goals:
                            latest_result = "W"
                        elif team_goals < opp_goals:
                            latest_result = "L"
                        else:
                            latest_result = "D"
                if latest_result == "W":
                    latest_text = f"Latest meeting: {fh_team} beat {fh_opp} {latest_score} on {latest_date}."
                elif latest_result == "L":
                    latest_text = f"Latest meeting: {fh_team} lost {latest_score} to {fh_opp} on {latest_date}."
                elif latest_result == "D":
                    latest_text = f"Latest meeting: {fh_team} and {fh_opp} drew {latest_score} on {latest_date}."
                else:
                    latest_text = (
                        f"Latest meeting: {fh_team} {latest_score} {fh_opp} on {latest_date} "
                        "(team score listed first)."
                    )
                _add_chunk(
                    chunks,
                    "fixture_latest",
                    latest_text,
                    ["fixture", "history", "latest", "opponent"],
                    2.2,
                )

    # Player matchup context (recent form + history vs opponent)
    matchup = extra_context.get("matchup_context") or {}
    recent_form = matchup.get("recent_form") or {}
    if recent_form.get("samples", 0):
        rf_samples = _safe_int(recent_form.get("samples", 0))
        rf_goals = _safe_int(recent_form.get("goals", 0))
        rf_assists = _safe_int(recent_form.get("assists", 0))
        rf_returns = _safe_int(recent_form.get("returns", 0))
        rf_avg_points = _safe_float(recent_form.get("avg_points", 0.0), 0.0)
        rf_clean_sheets = _safe_int(recent_form.get("clean_sheets", 0))
        rf_gi = rf_goals + rf_assists
        if rf_gi == 0:
            form_text = (
                f"{call_name} has blanked in all {rf_samples} recent outings — "
                f"averaging {rf_avg_points:.1f} FPL points."
            )
        elif rf_gi >= 4:
            form_text = (
                f"{call_name} is in scorching form with {_count_phrase(rf_goals, 'goal')} "
                f"and {_count_phrase(rf_assists, 'assist')} in the last {rf_samples}."
            )
        else:
            form_text = (
                f"{_count_phrase(rf_goals, 'goal')} and {_count_phrase(rf_assists, 'assist')} "
                f"in the last {rf_samples} for {call_name}, averaging {rf_avg_points:.1f} FPL points."
            )
        _add_chunk(
            chunks,
            "recent_form",
            form_text,
            ["recent", "form", "output", "performance"],
            2.6,
        )
        if position_group == "defender":
            cs_text = (
                f"{rf_clean_sheets} clean sheets in the last {rf_samples} appearances."
                if rf_clean_sheets > 0
                else f"No clean sheets in the last {rf_samples} appearances."
            )
            _add_chunk(
                chunks,
                "defensive_form",
                cs_text,
                ["recent", "form", "defense", "clean_sheet"],
                2.7,
            )

    vs_opponent = matchup.get("vs_opponent") or {}
    matchup_opponent = matchup.get("opponent") or opponent
    matchup_opponent_display = _display_team_name(matchup_opponent) if matchup_opponent else matchup_opponent
    if matchup_opponent:
        vs_samp = _safe_int(vs_opponent.get("samples", 0))
        vs_g = _safe_int(vs_opponent.get("goals", 0))
        vs_a = _safe_int(vs_opponent.get("assists", 0))
        vs_gi = vs_g + vs_a
        if vs_samp >= 2:
            if position_group == "defender":
                vs_cs = _safe_int(vs_opponent.get("clean_sheets", 0))
                if vs_cs >= 2:
                    vs_text = f"{call_name} has a strong defensive record against {matchup_opponent_display} with {vs_cs} clean sheets in {vs_samp} meetings."
                else:
                    vs_text = f"{call_name} has kept {vs_cs} clean sheet{'s' if vs_cs != 1 else ''} in {vs_samp} meetings against {matchup_opponent_display}."
                _add_chunk(chunks, "vs_opponent", vs_text,
                    ["opponent", "history", "head_to_head", "defense", "clean_sheet"], 2.9)
            else:
                if vs_gi >= 3:
                    vs_text = f"{call_name} loves playing against {matchup_opponent_display} — {vs_g} goals and {vs_a} assists in {vs_samp} meetings."
                elif vs_gi == 0 and vs_samp >= 3:
                    vs_text = f"{call_name} has never returned against {matchup_opponent_display} in {vs_samp} career meetings."
                else:
                    vs_text = f"{call_name} has {vs_g} goals and {vs_a} assists in {vs_samp} meetings against {matchup_opponent_display}."
                _add_chunk(chunks, "vs_opponent", vs_text,
                    ["opponent", "history", "head_to_head", "form"], 2.9)
        elif vs_samp == 1:
            vs_gi_1 = vs_g + vs_a
            if position_group == "defender":
                vs_cs_1 = _safe_int(vs_opponent.get("clean_sheets", 0))
                if vs_cs_1 > 0:
                    vs_text = (f"{call_name} kept a clean sheet in the only tracked meeting "
                               f"against {matchup_opponent_display} — small sample, but a positive sign.")
                else:
                    vs_text = (f"Only one meeting against {matchup_opponent_display} on record "
                               f"and no clean sheet — this fixture is still an unknown.")
            else:
                if vs_gi_1 > 0:
                    vs_text = (f"{call_name} returned {_count_phrase(vs_g, 'goal')} and "
                               f"{_count_phrase(vs_a, 'assist')} in the only tracked meeting "
                               f"against {matchup_opponent_display} — limited data, but a positive signal.")
                else:
                    vs_text = (f"One meeting against {matchup_opponent_display} on record with "
                               f"no goal involvement — not enough to establish a pattern either way.")
            _add_chunk(chunks, "vs_opponent", vs_text,
                       ["opponent", "history", "head_to_head", "sample"], 2.6)

    opponent_defense = matchup.get("opponent_defense") or {}
    if opponent_defense.get("samples", 0):
        opp_avg = _safe_float(opponent_defense.get("avg_goals_conceded_last5", 0.0))
        opp_samp = _safe_int(opponent_defense.get("samples", 0))
        opp_display = matchup_opponent_display or "Opponent"
        if opp_avg >= 1.4:
            opp_text = f"{opp_display} have been leaking {opp_avg:.1f} goals a game in their last {opp_samp} — a defence under pressure."
        elif opp_avg < 0.8:
            opp_text = f"{opp_display} have been tight at the back, conceding just {opp_avg:.1f} per game recently."
        else:
            opp_text = f"{opp_display} have conceded {opp_avg:.1f} goals per game in their last {opp_samp}."
        _add_chunk(chunks, "opponent_defense", opp_text,
            ["opponent", "defense", "conceded", "fixture"], 2.7)

    # Match density chunk
    matches_7 = _safe_int(player_data.get("matches_last_7", 0))
    matches_30 = _safe_int(player_data.get("matches_last_30", 0))
    if matches_7 >= 2 or matches_30 >= 6:
        density_text = (f"{call_name} has played {matches_7} matches in the last 7 days "
                        f"and {matches_30} in the last 30 — fixture congestion is a factor.")
        _add_chunk(chunks, "match_density", density_text,
                   ["workload", "fixture", "congestion", "density"], 2.5)
    elif matches_30 > 0 and prev_injuries <= 2:
        density_text = (f"{call_name} has played {matches_30} matches in the last 30 days "
                        f"with a manageable schedule.")
        _add_chunk(chunks, "match_density", density_text,
                   ["workload", "fixture", "density"], 2.0)

    # Venue chunk
    is_home = (matchup.get("is_home")
               if matchup.get("is_home") is not None
               else extra_context.get("next_fixture", {}).get("is_home"))
    if opponent and is_home is not None:
        venue_text = (f"This is a home fixture for {call_name}'s side against {matchup_opponent_display or opponent}."
                      if is_home else
                      f"{call_name}'s side travel away to {matchup_opponent_display or opponent} for this one.")
        _add_chunk(chunks, "venue", venue_text,
                   ["fixture", "home", "away", "venue"], 1.8)

    # Market context
    market = extra_context.get("bookmaker_consensus") or {}
    market_prob = _safe_float(market.get("average_probability"), 0.0)
    market_decimal = _safe_float(market.get("average_decimal"), 0.0)
    if market_prob > 0 and market_decimal > 1:
        _add_chunk(
            chunks,
            "market",
            f"Bookmaker consensus implies about {round(market_prob * 100)}% probability ({market_decimal:.2f} decimal).",
            ["market", "odds", "probability"],
            2.3,
        )

    # "Heat": force one open-ended football question for narrative sections.
    open_question = _build_dynamic_open_question(
        call_name=call_name,
        team_name=player_data.get("team"),
        player_position=player_position,
        matchup_opponent=matchup_opponent_display,
        recent_form=recent_form,
        vs_opponent=vs_opponent,
        fixture_history=fixture_history,
        fixture_samples=fixture_samples,
        all_time_samples=all_time_samples,
        days_since=days_since,
        market_prob=market_prob,
        opponent_defense_avg_conceded=_safe_float(opponent_defense.get("avg_goals_conceded_last5", 0.0), 0.0),
        opponent_defense_samples=_safe_int(opponent_defense.get("samples", 0), 0),
    )
    if open_question:
        _add_chunk(
            chunks,
            "open_question",
            open_question,
            ["question", "narrative", "decision"],
            2.4,
        )

    return chunks


def retrieve_player_context(
    player_data: Dict[str, Any],
    extra_context: Optional[Dict[str, Any]] = None,
    query: str = DEFAULT_STORY_QUERY,
    top_k: int = 6,
    include_open_question: bool = True,
) -> List[Dict[str, Any]]:
    """Retrieve the top-k most relevant context chunks for a query."""
    chunks = build_player_context_chunks(player_data, extra_context=extra_context)
    if not include_open_question:
        chunks = [chunk for chunk in chunks if chunk.get("kind") != "open_question"]
    if not chunks:
        return []

    q_tokens = _tokenize(query)
    ranked: List[Dict[str, Any]] = []
    for chunk in chunks:
        text_tokens = _tokenize(chunk["text"])
        overlap = len((text_tokens | chunk["tags"]) & q_tokens)
        lexical = overlap * 0.7
        specificity = min(len(text_tokens), 20) * 0.02
        score = chunk["weight"] + lexical + specificity
        ranked.append({**chunk, "score": round(score, 4)})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    selected = ranked[: max(1, top_k)]

    # Keep one open-ended question chunk even if lexical ranking pushes it out.
    if include_open_question:
        question_chunk = next((c for c in ranked if c.get("kind") == "open_question"), None)
        if question_chunk and not any(c.get("kind") == "open_question" for c in selected):
            if len(selected) >= max(1, top_k):
                selected[-1] = question_chunk
            else:
                selected.append(question_chunk)
    return selected


SECTION_RAG_QUERIES = {
    "story": DEFAULT_STORY_QUERY,
    "scoring": "scoring odds injury adjusted probability goals assists availability fixture market form",
    "fpl": "fpl value minutes availability rotation injury risk goals assists fixture form",
}

SECTION_RAG_PRIORITIES = {
    "story": [
        "recent_form",
        "defensive_form",
        "fixture_history",
        "fixture_history_all_time",
        "fixture_latest",
        "opponent_defense",
        "workload",
        "recency",
        "market",
        "output",
        "history",
    ],
    "scoring": [
        "recent_form",
        "opponent_defense",
        "vs_opponent",
        "fixture",
        "market",
        "fixture_history",
        "fixture_latest",
        "workload",
        "recency",
        "output",
    ],
    "fpl": [
        "recent_form",
        "defensive_form",
        "vs_opponent",
        "output",
        "fixture",
        "recency",
        "market",
        "fixture_history",
        "workload",
        "history",
    ],
}

SECTION_RAG_PREFIXES = {
    "story": [
        "{fact}",
        "{fact}",
        "{fact}",
    ],
    "scoring": [
        "Form pulse: {fact}",
        "Anytime scorer read: {fact}",
        "Betting angle: {fact}",
    ],
    "fpl": [
        "{fact}",
        "{fact}",
        "{fact}",
    ],
}


def build_dynamic_rag_line(
    player_data: Dict[str, Any],
    extra_context: Optional[Dict[str, Any]] = None,
    section: str = "story",
    context_chunks: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build one dynamic context line for a section from retrieved RAG facts."""
    section_key = (section or "story").strip().lower()
    if section_key not in SECTION_RAG_QUERIES:
        section_key = "story"

    if context_chunks is None:
        context_chunks = retrieve_player_context(
            player_data,
            extra_context=extra_context,
            query=SECTION_RAG_QUERIES[section_key],
            top_k=12 if section_key == "story" else 10,
            include_open_question=(section_key == "story"),
        )

    chunk_by_kind: Dict[str, str] = {}
    for chunk in context_chunks:
        kind = chunk.get("kind")
        text = (chunk.get("text") or "").strip()
        if kind and text and kind not in chunk_by_kind:
            chunk_by_kind[kind] = text

    fact = ""
    disallow_phrases = {
        "scoring": ["attacking output sits at", "goals/90", "assists/90"],
    }
    blocked = [p.lower() for p in disallow_phrases.get(section_key, [])]
    for kind in SECTION_RAG_PRIORITIES.get(section_key, []):
        text = chunk_by_kind.get(kind, "").strip()
        if blocked and any(token in text.lower() for token in blocked):
            continue
        if text:
            fact = text
            break

    call_name = _call_name(player_data)
    if not fact:
        prob = _safe_float(player_data.get("ensemble_prob"), 0.5)
        g90 = _safe_float(player_data.get("goals_per_90"), 0.0)
        a90 = _safe_float(player_data.get("assists_per_90"), 0.0)
        candidates = [
            f"{call_name}'s injury-adjusted baseline sits around {round(prob * 100)}% risk with {g90:.2f} goals/90 and {a90:.2f} assists/90.",
            f"Underlying profile for {call_name}: {round(prob * 100)}% risk signal and {g90 + a90:.2f} goal involvements per 90.",
            f"Context remains mixed for {call_name}, balancing availability risk and attacking output.",
        ]
    else:
        cleaned_fact = _statement(fact)
        candidates = [
            template.format(fact=cleaned_fact)
            for template in SECTION_RAG_PREFIXES.get(section_key, SECTION_RAG_PREFIXES["story"])
        ]

    next_fixture = (extra_context or {}).get("next_fixture") or {}
    opponent = _display_team_name((next_fixture.get("opponent") or "").strip())
    key = f"{section_key}|{call_name.lower()}|{opponent.lower()}"
    selected = _choose_variation(
        candidates,
        salt=f"{key}:{round(_safe_float(player_data.get('ensemble_prob'), 0.0), 3)}:{len(context_chunks)}",
        avoid=_LAST_SECTION_RAG_LINE.get(key),
    )
    final_line = _statement(selected or "")
    if not final_line:
        final_line = _statement(f"Context note: {call_name}'s current profile stays matchup-sensitive this week.")
    _LAST_SECTION_RAG_LINE[key] = final_line
    return final_line
