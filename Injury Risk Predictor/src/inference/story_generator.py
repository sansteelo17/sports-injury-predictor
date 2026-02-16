"""
Generate personalized injury risk stories for players.

Uses player data and risk factors to create human-readable narratives
that explain WHY a player has their specific risk level.
"""

import math
from typing import Dict, List, Optional


def generate_player_story(player_data: Dict) -> str:
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
    first_name = name.split()[0] if name else "This player"

    prev_injuries = player_data.get("previous_injuries", 0)
    days_lost = player_data.get("total_days_lost", 0)
    days_since = player_data.get("days_since_last_injury", 365)
    archetype = player_data.get("archetype", "Unknown")
    prob = player_data.get("ensemble_prob", 0.5)
    age = player_data.get("age", 25)

    # Calculate derived stats
    avg_days = days_lost / prev_injuries if prev_injuries > 0 else 0

    # Build story components
    parts = []

    # Opening based on risk level with percentile context
    percentile = player_data.get("risk_percentile")
    pct_suffix = ""
    if percentile and percentile >= 0.85:
        pct_suffix = f", placing them in the top {100 - int(percentile * 100)}% of risk across the league"

    if prob >= 0.45:
        parts.append(f"{first_name} is currently at elevated risk{pct_suffix}")
    elif prob >= 0.30:
        parts.append(f"{first_name} has a moderate injury risk{pct_suffix}")
    else:
        parts.append(f"{first_name} has a relatively low injury risk")

    # Recency factor
    if days_since < 30:
        parts.append(f"having only recently returned from injury {days_since} days ago")
    elif days_since < 60:
        parts.append(f"still in the vulnerable window after returning from injury about {days_since} days ago")
    elif days_since < 90:
        parts.append("with a recent injury still in the recovery monitoring period")

    # History factor
    if prev_injuries == 0:
        parts.append("with no significant injury history on record")
    elif prev_injuries >= 15:
        parts.append(f"with an extensive injury history of {prev_injuries} recorded injuries")
    elif prev_injuries >= 10:
        parts.append(f"having dealt with {prev_injuries} injuries throughout their career")
    elif prev_injuries >= 5:
        parts.append(f"with a moderate injury history of {prev_injuries} previous injuries")

    # Severity factor
    if avg_days >= 50:
        parts.append(f"When injured, they typically miss significant time (averaging {avg_days:.0f} days per injury)")
    elif avg_days >= 30:
        parts.append(f"Their injuries tend to be moderately serious, averaging {avg_days:.0f} days out")
    elif avg_days >= 15 and prev_injuries > 0:
        parts.append(f"Most injuries have been relatively minor, averaging {avg_days:.0f} days recovery")

    # Age factor
    if age >= 32:
        parts.append(f"At {age}, recovery times may be extended compared to younger players")
    elif age <= 21:
        parts.append(f"At just {age}, they're still developing physically which can be both protective and risky")

    # Archetype-specific insight
    archetype_insights = {
        "Currently Vulnerable": "The recent return from injury places them in a high-risk window where re-injury is most likely.",
        "Fragile": "Their injury pattern shows a tendency toward serious injuries requiring extended recovery periods.",
        "Injury Prone": "Historical patterns suggest they're more susceptible to injuries than the average player.",
        "Recurring": "While injuries are frequent, they tend to recover quickly from each setback.",
        "Durable": "Their body has shown resilience, bouncing back well from physical demands.",
        "Clean Record": "Limited injury data makes prediction less certain, but the absence of issues is encouraging.",
        "Moderate Risk": "Their injury profile is fairly typical, without major red flags or notable resilience.",
    }

    if archetype in archetype_insights:
        parts.append(archetype_insights[archetype])

    # Combine into flowing narrative
    story = ". ".join(parts)
    if not story.endswith("."):
        story += "."

    return story


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


def get_fpl_insight(player_data: Dict) -> Optional[str]:
    """Generate FPL-specific insight for a player."""
    name = player_data.get("name", "This player")
    first_name = name.split()[0] if name else "This player"
    days_since = player_data.get("days_since_last_injury", 365)
    archetype = player_data.get("archetype", "Unknown")
    prob = player_data.get("ensemble_prob", 0.5)

    if days_since < 14:
        return f"FPL Warning: {first_name} just returned from injury. High rotation risk - may be used as impact sub rather than starter."

    if 14 <= days_since <= 28:
        return f"FPL Alert: {first_name}'s coach may limit minutes to reduce re-injury risk. Don't expect full 90s in the next few gameweeks."

    if archetype == "Currently Vulnerable" and days_since <= 60:
        return f"FPL Note: {first_name} is in a vulnerable window. Have bench cover ready in case of setback."

    if prob >= 0.45:
        return f"FPL Consideration: {first_name}'s elevated injury risk means rotation is likely during fixture congestion."

    if archetype == "Injury Prone" and prob >= 0.35:
        return f"FPL Tip: {first_name} has a history of injuries. Monitor news carefully before deadline."

    return None


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


def calculate_scoring_odds(player_data: Dict) -> Optional[Dict]:
    """
    Calculate odds for a player to score in the next match.

    Combines historical scoring rate with availability probability.
    """
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

    return {
        "score_probability": round(score_prob, 3),
        "involvement_probability": round(involvement_prob, 3),
        "goals_per_90": round(goals_per_90, 2),
        "assists_per_90": round(assists_per_90, 2),
        "american": odds["american"],
        "decimal": odds["decimal"],
        "fractional": odds["fractional"],
        "availability_factor": round(availability, 2),
    }


def get_fpl_value_assessment(player_data: Dict) -> Optional[Dict]:
    """
    Generate FPL value assessment combining injury risk with attacking output.
    """
    name = player_data.get("name", "Player")
    first_name = name.split()[0] if name else "Player"
    goals = player_data.get("goals", 0)
    assists = player_data.get("assists", 0)
    goals_per_90 = player_data.get("goals_per_90", 0)
    assists_per_90 = player_data.get("assists_per_90", 0)
    price = player_data.get("price", 0)
    injury_prob = player_data.get("ensemble_prob", 0.5)
    position = player_data.get("position", "Unknown")
    minutes = player_data.get("minutes", 0)
    archetype = player_data.get("archetype", "Unknown")

    if minutes < 270 or price == 0:
        return None

    ga_per_90 = goals_per_90 + assists_per_90
    risk_factor = 1 - injury_prob
    adjusted_value = ga_per_90 * risk_factor

    if adjusted_value >= 0.6:
        tier, emoji = "Premium", "💎"
    elif adjusted_value >= 0.4:
        tier, emoji = "Strong", "✅"
    elif adjusted_value >= 0.25:
        tier, emoji = "Decent", "👍"
    elif adjusted_value >= 0.1:
        tier, emoji = "Rotation", "🔄"
    else:
        tier, emoji = "Avoid", "⛔"

    # Injury history context for verdict
    prev_injuries = player_data.get("previous_injuries", player_data.get("player_injury_count", 0))
    days_since = player_data.get("days_since_last_injury", 365)
    total_days_lost = player_data.get("total_days_lost", 0)
    avg_days_per_injury = total_days_lost / prev_injuries if prev_injuries > 0 else 0
    injury_prone = prev_injuries >= 5 or (prev_injuries >= 3 and avg_days_per_injury >= 30)

    # Priority: archetype/history overrides, then model probability
    if archetype == "Currently Vulnerable" or (days_since < 45 and prev_injuries >= 2):
        verdict = f"{emoji} {first_name} is in a vulnerable period - monitor before investing."
    elif injury_prone and days_since < 120:
        verdict = f"{emoji} {first_name} has a concerning injury record ({prev_injuries} injuries, avg {avg_days_per_injury:.0f} days) - high risk pick."
    elif injury_prob < 0.15 and not injury_prone and ga_per_90 >= 0.5:
        verdict = f"{emoji} {first_name} is a reliable FPL asset - good returns with low injury risk."
    elif injury_prob < 0.15 and not injury_prone and ga_per_90 < 0.3:
        verdict = f"{emoji} {first_name} is durable but lacks attacking output for FPL value."
    elif injury_prob >= 0.30 and ga_per_90 >= 0.5:
        verdict = f"{emoji} {first_name} has great numbers but elevated injury risk - high reward, high risk."
    elif injury_prob >= 0.30:
        verdict = f"{emoji} {first_name} carries significant injury risk - consider alternatives."
    else:
        verdict = f"{emoji} {first_name} offers {tier.lower()} FPL value with moderate risk."

    position_insight = None
    if position == "FWD" and ga_per_90 >= 0.5 and injury_prob < 0.20:
        position_insight = f"As a forward with {goals} goals this season and low injury risk, {first_name} is a strong captaincy option."
    elif position == "MID" and assists_per_90 >= 0.3 and injury_prob < 0.20:
        position_insight = f"Creative midfielder with {assists} assists - good value in the midfield slot."
    elif position == "DEF" and ga_per_90 >= 0.2:
        position_insight = f"Attacking defender with goal threat - rare value if they stay fit."

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
    first_name = name.split()[0] if name else "This player"
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
    name = player_data.get("name", "Player").split()[0]
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


def generate_lab_notes(player_data: Dict) -> Optional[Dict]:
    """
    Generate Yara's Lab Notes — explainability in plain English + technical detail.

    Checks key features against thresholds and explains what's driving the risk score.

    Args:
        player_data: Dict with player info including model features

    Returns:
        Dict with summary, key_drivers, technical — or None if insufficient data
    """
    name = player_data.get("name", "This player")
    first_name = name.split()[0] if name else "This player"
    prob = player_data.get("ensemble_prob", 0.5)

    # Check key features and build driver list
    drivers = []

    # 1. ACWR (Acute:Chronic Workload Ratio)
    acwr = player_data.get("acwr", 0)
    if acwr and float(acwr) >= 1.3:
        drivers.append({
            "name": "Workload Ratio",
            "value": round(float(acwr), 2),
            "impact": "risk_increasing",
            "explanation": f"ACWR of {float(acwr):.2f} exceeds the 1.3 threshold — workload is spiking relative to baseline."
        })
    elif acwr and float(acwr) > 0:
        drivers.append({
            "name": "Workload Ratio",
            "value": round(float(acwr), 2),
            "impact": "protective",
            "explanation": f"ACWR of {float(acwr):.2f} is within safe range — workload is well managed."
        })

    # 2. Fatigue Index
    fatigue = player_data.get("fatigue_index", 0)
    if fatigue and float(fatigue) >= 1.0:
        drivers.append({
            "name": "Fatigue Index",
            "value": round(float(fatigue), 2),
            "impact": "risk_increasing",
            "explanation": "Acute load exceeding chronic load — accumulated fatigue is a concern."
        })

    # 3. Injury Count
    inj_count = player_data.get("player_injury_count", player_data.get("previous_injuries", 0))
    if inj_count and int(inj_count) >= 5:
        drivers.append({
            "name": "Injury History",
            "value": int(inj_count),
            "impact": "risk_increasing",
            "explanation": f"{int(inj_count)} previous injuries — historical pattern strongly influences the model."
        })
    elif inj_count and int(inj_count) >= 3:
        drivers.append({
            "name": "Injury History",
            "value": int(inj_count),
            "impact": "risk_increasing",
            "explanation": f"{int(inj_count)} previous injuries — moderate injury history noted."
        })
    elif int(inj_count or 0) == 0:
        drivers.append({
            "name": "Injury History",
            "value": 0,
            "impact": "protective",
            "explanation": "No recorded injuries — clean slate works strongly in the player's favor."
        })

    # 4. Days Since Last Injury
    days_since = player_data.get("days_since_last_injury", 365)
    if days_since and int(days_since) <= 30:
        drivers.append({
            "name": "Recency",
            "value": int(days_since),
            "impact": "risk_increasing",
            "explanation": f"Only {int(days_since)} days since last injury — re-injury window is still open."
        })
    elif days_since and int(days_since) <= 60:
        drivers.append({
            "name": "Recency",
            "value": int(days_since),
            "impact": "risk_increasing",
            "explanation": f"{int(days_since)} days since last injury — elevated risk period hasn't fully closed."
        })
    elif days_since and int(days_since) > 365:
        drivers.append({
            "name": "Recency",
            "value": int(days_since),
            "impact": "protective",
            "explanation": "Over a year injury-free — strong indicator of current durability."
        })

    # 5. Age
    age = player_data.get("age", 25)
    if age and int(age) >= 32:
        drivers.append({
            "name": "Age Factor",
            "value": int(age),
            "impact": "risk_increasing",
            "explanation": f"At {int(age)}, recovery capacity and tissue resilience decline — model accounts for this."
        })
    elif age and int(age) <= 21:
        drivers.append({
            "name": "Age Factor",
            "value": int(age),
            "impact": "neutral",
            "explanation": f"At {int(age)}, still developing physically — can go either way."
        })

    # 6. Acute Load
    acute = player_data.get("acute_load", 0)
    if acute and float(acute) >= 3:
        drivers.append({
            "name": "Match Congestion",
            "value": round(float(acute), 1),
            "impact": "risk_increasing",
            "explanation": f"3+ matches in 7 days — fixture congestion is a significant risk factor."
        })

    # 7. Injury Prone Flag
    is_ip = player_data.get("is_injury_prone", 0)
    if is_ip and int(is_ip) == 1:
        drivers.append({
            "name": "Injury Prone Profile",
            "value": "Yes",
            "impact": "risk_increasing",
            "explanation": "Historical injury frequency exceeds the threshold — model flags elevated baseline risk."
        })

    # 8. Spike Flag
    spike = player_data.get("spike_flag", 0)
    if spike and int(spike) == 1:
        drivers.append({
            "name": "Workload Spike",
            "value": "Detected",
            "impact": "risk_increasing",
            "explanation": "ACWR > 1.5 — dangerous workload spike that sharply increases injury probability."
        })

    if not drivers:
        return None

    # Sort: risk-increasing first, then protective
    impact_order = {"risk_increasing": 0, "neutral": 1, "protective": 2}
    drivers.sort(key=lambda d: impact_order.get(d["impact"], 1))

    # Build plain English summary
    risk_drivers = [d for d in drivers if d["impact"] == "risk_increasing"]
    protective_drivers = [d for d in drivers if d["impact"] == "protective"]

    if risk_drivers:
        driver_names = [d["name"].lower() for d in risk_drivers[:3]]
        if len(driver_names) == 1:
            summary = f"The main factor elevating {first_name}'s risk is {driver_names[0]}."
        else:
            summary = f"Key factors elevating {first_name}'s risk are {', '.join(driver_names[:-1])} and {driver_names[-1]}."
        if protective_drivers:
            prot_names = [d["name"].lower() for d in protective_drivers[:2]]
            summary += f" Working in their favor: {', '.join(prot_names)}."
    else:
        summary = f"{first_name}'s risk profile is clean — no significant risk factors detected by the model."

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
        "methodology": (
            "Ensemble of CatBoost, LightGBM, and XGBoost with stacking meta-learner. "
            "81 engineered features including workload ratios, injury history, "
            "match congestion, and temporal patterns. "
            "Trained on 10+ years of Premier League injury data."
        ),
        "feature_highlights": tech_features[:8],
    }

    return {
        "summary": summary,
        "key_drivers": drivers[:4],
        "technical": technical,
    }
