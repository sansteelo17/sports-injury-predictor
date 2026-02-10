"""
Generate personalized injury risk stories for players.

Uses player data and risk factors to create human-readable narratives
that explain WHY a player has their specific risk level.
"""

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

    # Opening based on risk level
    if prob >= 0.6:
        parts.append(f"{first_name} is currently at elevated risk")
    elif prob >= 0.35:
        parts.append(f"{first_name} has a moderate injury risk")
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

    if prob >= 0.6:
        return "With elevated risk factors, rotation with squad depth and careful fixture management during busy periods would help mitigate injury probability."

    if prob < 0.35:
        return "Current risk indicators are favorable. Maintaining the current training and recovery regime should support continued availability."

    return "Standard monitoring and recovery protocols are appropriate. No specific interventions indicated at this time."
