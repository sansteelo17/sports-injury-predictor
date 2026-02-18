"""
Generate personalized injury risk stories for players.

Uses player data and risk factors to create human-readable narratives
that explain WHY a player has their specific risk level.
"""

import math
import re
from typing import Dict, List, Optional

from .context_rag import build_dynamic_rag_line, retrieve_player_context
from .llm_client import generate_grounded_narrative


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


def generate_player_story(player_data: Dict, extra_context: Optional[Dict] = None) -> str:
    """
    Generate a personalized risk story for a player.

    Args:
        player_data: Dict with player info including:
            - name, team, age, position
            - previous_injuries, total_days_lost, days_since_last_injury
            - archetype, ensemble_prob
            - last_injury_date (optional)

    Returns:
        Human-readable story explaining the player's risk
    """
    name = player_data.get("name", "This player")
    first_name = _call_name(player_data)

    prev_injuries = _safe_int(player_data.get("previous_injuries", 0))
    days_lost = _safe_float(player_data.get("total_days_lost", 0))
    days_since = _safe_int(player_data.get("days_since_last_injury", 365), 365)
    archetype = player_data.get("archetype", "Unknown")
    prob = _safe_float(player_data.get("ensemble_prob", 0.5), 0.5)
    age = _safe_int(player_data.get("age", 25), 25)

    # Calculate derived stats
    avg_days = days_lost / prev_injuries if prev_injuries > 0 else 0

    # Opening with percentile context
    percentile = player_data.get("risk_percentile")
    opening = ""
    if prob >= 0.45:
        opening = f"{first_name} currently profiles as high injury risk ({round(prob * 100)}%)."
    elif prob >= 0.30:
        opening = f"{first_name} currently profiles as moderate injury risk ({round(prob * 100)}%)."
    else:
        opening = f"{first_name} currently profiles as lower injury risk ({round(prob * 100)}%)."

    if percentile and percentile >= 0.85:
        top_bucket = max(1, 100 - int(float(percentile) * 100))
        opening = opening.rstrip(".") + f" That places them in roughly the top {top_bucket}% of risk in the current pool."

    # Retrieve context chunks (lightweight RAG over structured context)
    context_chunks = retrieve_player_context(
        player_data,
        extra_context=extra_context,
        top_k=12,
        include_open_question=True,
    )

    chunk_by_kind = {}
    for chunk in context_chunks:
        kind = chunk.get("kind")
        if kind and kind not in chunk_by_kind:
            chunk_by_kind[kind] = chunk.get("text", "")

    details: List[str] = []
    for kind in [
        "history",
        "sample_sensitivity",
        "severity_pattern",
        "recency",
        "fixture_history",
        "fixture_history_all_time",
        "fixture_latest",
        "recent_form",
        "vs_opponent",
        "opponent_defense",
        "workload",
        "fixture",
        "market",
        "output",
    ]:
        text = chunk_by_kind.get(kind, "")
        sentence = _as_sentence(text)
        if sentence and sentence not in details:
            details.append(sentence)

    # Age context (kept explicit because it is not always present in retrieved chunks)
    if age >= 32:
        details.append(f"At {age}, recovery windows can tighten under fixture congestion.")
    elif age <= 21:
        details.append(f"At {age}, physical development can be both protective and volatile.")

    # Archetype interpretation
    archetype_insights = {
        "Currently Vulnerable": "The return window is still open, so short-term re-injury risk remains elevated.",
        "Fragile": "When setbacks happen, they have tended to require longer recoveries.",
        "Injury Prone": "Frequency is the key concern rather than one isolated injury spell.",
        "Recurring": "The pattern is repeat events with relatively manageable layoff lengths.",
        "Durable": "Recent availability trends point to resilience under normal load.",
        "Clean Record": "Data history is light, which keeps uncertainty higher but baseline risk lower.",
        "Moderate Risk": "The profile is mixed, with no single dominant red flag.",
    }
    if archetype in archetype_insights:
        details.append(archetype_insights[archetype])

    # Guardrail for small-sample histories (e.g., 1-2 injuries with one severe event)
    if 0 < prev_injuries <= 2:
        if days_since >= 365:
            details.append(
                f"Only {prev_injuries} injuries are logged, so one major event can skew averages, and the {days_since}-day healthy run should carry real weight."
            )
        else:
            details.append(
                f"Only {prev_injuries} injuries are logged, so this profile is more sample-sensitive than most."
            )

    # Build final narrative with controlled length
    open_question = _as_sentence(_first_chunk_text(context_chunks, "open_question") or "")
    story_parts = [_as_sentence(opening)] + [d for d in details if d][:5]
    if open_question:
        story_parts.append(open_question)
    fallback_story = " ".join(part for part in story_parts if part).strip()
    fallback_story = fallback_story or f"{first_name} currently profiles as moderate injury risk."

    return generate_grounded_narrative(
        task="Write a risk analysis narrative that explains injury profile clearly for football users.",
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
    """Generate player-aware FPL manager tip (not team-level market copy)."""
    first_name = _call_name(player_data)
    if first_name and first_name[0].islower():
        first_name = first_name[0].upper() + first_name[1:]
    injury_prob = _safe_float(player_data.get("ensemble_prob", 0.5), 0.5)
    days_since = _safe_int(player_data.get("days_since_last_injury", 365), 365)
    minutes = _safe_int(player_data.get("minutes", 0), 0)
    position = str(player_data.get("position", "Unknown") or "Unknown")
    role = _position_group(position)
    goals_per_90 = _safe_float(player_data.get("goals_per_90", 0.0), 0.0)
    assists_per_90 = _safe_float(player_data.get("assists_per_90", 0.0), 0.0)

    extra_context = extra_context or {}
    matchup_context = extra_context.get("matchup_context") or {}
    recent_form = matchup_context.get("recent_form") or {}
    recent_samples = _safe_int(recent_form.get("samples", 0), 0)
    recent_goals = _safe_int(recent_form.get("goals", 0), 0)
    recent_assists = _safe_int(recent_form.get("assists", 0), 0)
    recent_returns = _safe_int(recent_form.get("returns", 0), 0)
    recent_clean_sheets = _safe_int(recent_form.get("clean_sheets", 0), 0)
    recent_goal_involvements = recent_goals + recent_assists

    opponent_defense = matchup_context.get("opponent_defense") or {}
    opp_conceded = _safe_float(opponent_defense.get("avg_goals_conceded_last5", 0.0), 0.0)
    opp_def_samples = _safe_int(opponent_defense.get("samples", 0), 0)

    next_fixture = extra_context.get("next_fixture") or {}
    opponent = _display_team_name((
        matchup_context.get("opponent")
        or next_fixture.get("opponent")
        or "this opponent"
    ))
    is_home = matchup_context.get("is_home")
    if is_home is None:
        is_home = next_fixture.get("is_home")
    venue_label = "at home" if is_home else "away" if is_home is not None else ""
    availability_pct = round((1 - injury_prob) * 100)

    # Baseline manager action (kept distinct from FPL value tiers)
    if minutes < 180:
        action = "Bench for now"
        reason = f"only {minutes} recent minutes, so starts are not secure yet"
    elif days_since < 21:
        action = "Bench unless needed"
        reason = f"just {days_since} days since the last injury, so availability is still volatile"
    elif injury_prob >= 0.45:
        action = "Start only with bench cover"
        reason = f"availability risk is still elevated ({round(injury_prob * 100)}%)"
    elif role == "defender":
        if recent_clean_sheets >= 2 and injury_prob < 0.35:
            action = "Start"
            reason = (
                f"{recent_clean_sheets} clean sheets in the last {recent_samples} and availability near {availability_pct}%"
                if recent_samples > 0
                else f"clean-sheet trajectory is positive with availability near {availability_pct}%"
            )
        else:
            action = "Playable, not a lock"
            if recent_samples > 0:
                reason = f"{recent_clean_sheets} clean sheets in the last {recent_samples}, so the clean-sheet floor is still matchup-dependent"
            else:
                reason = "defensive floor is still matchup-dependent"
    else:
        output_signal = goals_per_90 + assists_per_90
        if output_signal >= 0.55 and recent_returns >= 2 and injury_prob < 0.35:
            action = "Start with confidence"
            if recent_samples > 0:
                reason = (
                    f"{recent_goal_involvements} goal involvements in the last {recent_samples} "
                    f"with availability near {availability_pct}%"
                )
            else:
                reason = "output and availability both support the pick"
        elif output_signal < 0.2 and recent_returns == 0:
            action = "Bench/avoid this week"
            if recent_samples > 0:
                reason = f"no returns in the last {recent_samples}, and baseline output is {output_signal:.2f} involvements per 90"
            else:
                reason = "current output profile is too quiet"
        else:
            action = "Start if owned"
            if recent_samples > 0 and opp_def_samples > 0 and opp_conceded > 0:
                reason = (
                    f"{recent_goal_involvements} involvements in the last {recent_samples}; "
                    f"{opponent} are conceding {opp_conceded:.2f} goals per game lately"
                )
            elif recent_samples > 0:
                reason = f"{recent_goal_involvements} involvements in the last {recent_samples}, with steady minutes behind that profile"
            else:
                reason = "minutes are stable, but the return trend is mixed"

    # Align phrasing with FPL value tier to avoid contradictions.
    value_assessment = get_fpl_value_assessment(player_data, extra_context=extra_context)
    if value_assessment:
        tier = (value_assessment.get("tier") or "").strip().lower()
        if tier == "avoid":
            action = "Bench/avoid this week"
            reason = "overall availability-plus-output signal is weak right now"
        elif tier == "rotation":
            action = "Start only if needed"
            reason = "minutes are usable, but projection is closer to squad depth than a locked starter"

    insight = f"{action} for {first_name} {venue_label} vs {opponent}: {reason}."
    return _as_sentence(re.sub(r"\s{2,}", " ", insight).strip())


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
    fallback_analysis = (
        f"Yara estimates {player_call_name}'s chance to score at {round(score_prob * 100)}% after injury adjustment. "
        f"Baseline sits at {goals_per_90:.2f} goals/90 and {assists_per_90:.2f} assists/90 "
        f"with availability {availability:.2f}."
    )
    scoring_context = retrieve_player_context(
        player_data,
        extra_context=extra_context,
        query="scoring odds injury adjusted probability goals assists availability fixture market",
        top_k=10,
        include_open_question=False,
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

    if tier == "Avoid":
        if low_output and high_risk:
            verdict = (
                f"{first_name} is an avoid this week: output is too thin and availability drag is still heavy."
            )
        elif low_output and durability_signal:
            verdict = (
                f"{first_name} is an avoid this week: upside is too thin for the price. "
                f"Durability is not the main issue ({prev_injuries} prior injury, {days_since} days injury-free)."
            )
        elif low_output:
            verdict = f"{first_name} is an avoid this week: output is too thin for this price bracket."
        elif high_risk:
            verdict = f"{first_name} is an avoid this week: injury drag is too high for the expected return."
        else:
            verdict = f"{first_name} is an avoid this week: projected return is too low for the price."
    elif tier == "Rotation":
        if low_output:
            verdict = f"{first_name} is a rotation-only pick: minutes are usable, upside is limited."
        elif high_risk:
            verdict = f"{first_name} is a rotation-only pick until availability risk cools."
        else:
            verdict = f"{first_name} is fixture-dependent value, better as depth than a locked starter."
    elif tier == "Decent":
        if high_risk:
            verdict = f"{first_name} is decent value but still carries availability drag."
        else:
            verdict = f"{first_name} is decent value with balanced upside and risk."
    elif tier == "Strong":
        verdict = f"{first_name} is strong value right now with reliable return potential."
    else:
        verdict = f"{first_name} is premium value right now and can justify a starter slot."

    if archetype == "Currently Vulnerable" and days_since < 60:
        verdict = f"{first_name} is still in a vulnerable return window, which caps FPL trust this week."
    elif injury_prone and days_since < 120 and tier in {"Decent", "Strong", "Premium"}:
        verdict = (
            f"{first_name} has real upside, but volatility is high "
            f"({prev_injuries} injuries, {avg_days_per_injury:.0f} days average layoff)."
        )

    position_insight = None
    if role == "attacker" and ga_per_90 >= 0.5 and injury_prob < 0.20:
        position_insight = (
            f"As a forward with {goals} goals this season and low injury drag, {first_name} is captain-viable."
        )
    elif role == "midfielder" and assists_per_90 >= 0.3 and injury_prob < 0.20:
        position_insight = f"Creative midfield profile with {assists} assists - strong slot efficiency."
    elif role == "defender" and (recent_clean_sheet_rate >= 0.4 or ga_per_90 >= 0.2):
        position_insight = (
            f"Defender value is live when clean sheets hold and attacking threat ({ga_per_90:.2f} G+A/90) stays active."
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
    venue = "at home" if is_home else "away"
    edge = round((yara - market) * 100, 1)

    # Pick a contextual reason
    if days_since < 60:
        context = "The recent return adds uncertainty, but the underlying output is there."
    elif inj_prob < 0.25:
        context = f"Fitness looks solid — no red flags in the injury profile."
    elif g90 >= 0.5:
        context = f"The numbers back it up — {g90:.2f} goals per 90 this season."
    elif archetype == "Durable":
        context = "Durability is underpriced here."
    else:
        context = f"Playing {venue} against {opponent} favors the over."

    return f"I'm projecting {yara*100:.0f}%. Market has {market*100:.0f}% — I see value. {context}"


def _yara_market_generous(name, yara, market, opponent, inj_prob, archetype):
    """Market is more generous than Yara's model."""
    if inj_prob >= 0.45:
        reason = f"Injury risk is real — {inj_prob*100:.0f}% probability is non-trivial."
    elif archetype in ("Fragile", "Currently Vulnerable"):
        reason = f"The {archetype.lower()} tag means availability isn't guaranteed."
    else:
        reason = f"The underlying scoring rate doesn't justify the market price."

    return f"Market says {market*100:.0f}%, I'm at {yara*100:.0f}%. The market's generous — I'd pass. {reason}"


def _yara_aligned(name, yara, market, opponent, inj_prob):
    """Market and model agree."""
    if inj_prob < 0.3:
        note = "Low injury risk is the one tailwind not fully priced in."
    elif inj_prob >= 0.45:
        note = "Just watch the teamsheet — availability is the only variable."
    else:
        note = "No edge either way. Move on to better spots."

    return f"Market and my model are aligned at ~{yara*100:.0f}%. {note}"


def _yara_no_market(name, yara, inj_prob, g90, a90, archetype, days_since):
    """No market odds available — projection only."""
    if g90 >= 0.5 and inj_prob < 0.35:
        tone = f"No market line available, but I have {name} at {yara*100:.0f}% to score. The output is consistent and the body is holding up."
    elif g90 >= 0.3 and inj_prob < 0.4:
        tone = f"No market line to compare. My projection: {yara*100:.0f}%. Decent attacking output with manageable risk."
    elif inj_prob >= 0.45:
        tone = f"No market line, and I'm only at {yara*100:.0f}%. Injury risk ({inj_prob*100:.0f}%) is the headline here."
    elif days_since < 60:
        tone = f"No market data. Projecting {yara*100:.0f}%, but {name}'s still in the return window — treat with caution."
    else:
        involvement = (g90 + a90) * 100
        tone = f"No market line. I have {name} at {yara*100:.0f}% to score, {involvement:.0f}% for any goal involvement."

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
    context_driver = (
        _first_chunk_text(lab_context, "fixture_latest")
        or _first_chunk_text(lab_context, "fixture_history")
        or _first_chunk_text(lab_context, "recent_form")
        or _first_chunk_text(lab_context, "opponent_defense")
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
