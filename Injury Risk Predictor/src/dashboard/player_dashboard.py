import pandas as pd

from ..inference.risk_card import generate_recommendation
from ..models.archetype import get_archetype_profile

# ------------------------------------------------------------
# Utility: Get latest snapshot
# ------------------------------------------------------------
def get_latest_snapshot(inference_df, player_name):
    """
    Get the most recent data snapshot for a player from the inference DataFrame.

    Args:
        inference_df: DataFrame with inference results (output from build_inference_df_with_ensemble
                      or build_full_inference_df)
        player_name: Name of the player to look up

    Returns:
        pandas.Series: The most recent row for the player, sorted by match_date

    Raises:
        ValueError: If player is not found in the DataFrame
        KeyError: If 'name' or 'match_date' columns are missing

    Example:
        >>> row = get_latest_snapshot(inference_df, "Mohamed Salah")
        >>> print(row["ensemble_prob"])  # or row["catboost_prob"]
        0.45
    """
    if "name" not in inference_df.columns:
        raise KeyError("inference_df must have a 'name' column")

    # Support both historical (match_date) and live (snapshot_date/injury_datetime) data
    date_col = None
    for col in ["match_date", "snapshot_date", "injury_datetime"]:
        if col in inference_df.columns:
            date_col = col
            break

    if date_col is None:
        raise KeyError("inference_df must have a date column (match_date, snapshot_date, or injury_datetime)")

    pdf = inference_df[inference_df["name"] == player_name]

    if pdf.empty:
        available_players = inference_df["name"].unique()[:10]
        raise ValueError(
            f"Player '{player_name}' not found in inference_df. "
            f"Available players (first 10): {list(available_players)}"
        )

    return pdf.sort_values(date_col).iloc[-1]


def _get_safe_value(row, col, default=None):
    """Safely get a value from row with fallback."""
    if col in row.index and pd.notna(row[col]):
        return row[col]
    return default


def _get_risk_probability(row):
    """
    Get the primary risk probability from the row.
    Supports both ensemble and individual model outputs.
    """
    # Priority: ensemble_prob > catboost_prob > average of available models
    if "ensemble_prob" in row.index and pd.notna(row["ensemble_prob"]):
        return float(row["ensemble_prob"]), "ensemble"

    if "catboost_prob" in row.index and pd.notna(row["catboost_prob"]):
        return float(row["catboost_prob"]), "catboost"

    # Fallback: average of available model probs
    prob_cols = ["lgb_prob", "xgb_prob"]
    available_probs = [float(row[c]) for c in prob_cols if c in row.index and pd.notna(row[c])]
    if available_probs:
        return sum(available_probs) / len(available_probs), "average"

    raise KeyError("No risk probability column found in row")


# ------------------------------------------------------------
# Panels
# ------------------------------------------------------------

def panel_player_overview(row):
    """Build player overview panel."""
    # Handle team column variations
    team = _get_safe_value(row, "player_team") or _get_safe_value(row, "team", "Unknown")

    # Handle age (may not always be present)
    age = _get_safe_value(row, "age")
    if age is not None:
        age = int(age)

    # Handle match_date
    match_date = _get_safe_value(row, "match_date")
    if match_date is not None:
        if hasattr(match_date, 'date'):
            match_date = str(match_date.date())
        else:
            match_date = str(match_date)

    return {
        "name": row["name"],
        "team": team,
        "position": _get_safe_value(row, "position", "Unknown"),
        "age": age,
        "archetype": _get_safe_value(row, "archetype", "Unknown"),
        "confidence": _get_safe_value(row, "confidence", "medium"),
        "last_match": match_date,
    }


def panel_injury_risk(row):
    """Build injury risk panel with model support details."""
    risk, source = _get_risk_probability(row)

    if risk >= 0.60:
        level = "High"
    elif risk >= 0.40:
        level = "Moderate"
    else:
        level = "Low"

    # Build model support dict based on available columns
    model_support = {
        "probability_source": source,
        "confidence": _get_safe_value(row, "confidence", "medium"),
    }

    # Add ensemble prob if available
    if "ensemble_prob" in row.index and pd.notna(row["ensemble_prob"]):
        model_support["ensemble"] = round(float(row["ensemble_prob"]), 3)

    # Add individual model probs if available
    model_cols = {
        "catboost_prob": "catboost",
        "lgb_prob": "lightgbm",
        "xgb_prob": "xgboost"
    }
    for col, name in model_cols.items():
        if col in row.index and pd.notna(row[col]):
            model_support[name] = round(float(row[col]), 3)

    # Add agreement score if available
    if "agreement" in row.index and pd.notna(row["agreement"]):
        model_support["agreement_score"] = int(row["agreement"])

    return {
        "risk_score": round(risk, 3),
        "risk_level": level,
        "model_support": model_support,
    }


def panel_severity_projection(row):
    """
    Build severity projection panel.

    Works with both:
    - Ground truth data (has 'severity_days')
    - Inference data (has 'predicted_severity_class')
    """
    # Check if we have ground truth or predicted severity
    if "severity_days" in row.index and pd.notna(row.get("severity_days")):
        sev = float(row["severity_days"])

        if sev >= 60:
            level = "Catastrophic"
        elif sev >= 21:
            level = "Major"
        elif sev >= 7:
            level = "Moderate"
        else:
            level = "Minor"

        return {
            "projected_days_lost": round(sev, 1),
            "severity_category": level,
            "source": "actual"
        }

    elif "predicted_severity_class" in row.index:
        # Map predicted class to estimated days and category
        pred_class = row["predicted_severity_class"]
        class_mapping = {
            "short": {"days": "0-21", "category": "Minor/Moderate", "midpoint": 10},
            "medium": {"days": "21-60", "category": "Major", "midpoint": 40},
            "long": {"days": "60+", "category": "Catastrophic", "midpoint": 90},
        }

        info = class_mapping.get(pred_class, class_mapping["short"])

        return {
            "projected_days_lost": info["days"],
            "severity_category": info["category"],
            "predicted_class": pred_class,
            "source": "predicted"
        }

    else:
        return {
            "projected_days_lost": "Unknown",
            "severity_category": "Unknown",
            "source": "missing"
        }


def panel_archetype(archetype):
    """
    Get detailed archetype profile using the centralized archetype definitions.

    Parameters
    ----------
    archetype : str
        Archetype name (short or full name)

    Returns
    -------
    dict
        Profile with description, training_focus, key_characteristics, etc.
    """
    profile = get_archetype_profile(archetype)

    # Return a subset of the profile for dashboard display
    return {
        "description": profile.get("description", "Unknown archetype profile"),
        "training_focus": profile.get("training_focus", "Consult with sports science staff"),
        "key_characteristics": profile.get("key_characteristics", []),
        "minutes_strategy": profile.get("minutes_strategy", "Standard management"),
        "risk_level": profile.get("risk_level", "unknown"),
    }

# ------------------------------------------------------------
# Top SHAP Drivers
# ------------------------------------------------------------
def panel_top_drivers(row, top_n=5):
    """
    Extract top SHAP drivers from the row.

    If shap_values column is missing (e.g., inference without SHAP computation),
    falls back to showing key risk factors from available columns.
    """
    # Check if shap_values exists
    if "shap_values" not in row.index or row["shap_values"] is None:
        # Fallback: generate synthetic drivers from key risk features
        return _fallback_risk_drivers(row, top_n)

    shap_vals = row["shap_values"]

    # Handle case where shap_vals is not iterable (e.g., NaN or scalar)
    try:
        if hasattr(shap_vals, '__len__') and len(shap_vals) > 0:
            feature_names = row.index.tolist()
            pairs = list(zip(feature_names, shap_vals))
            pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

            # Keep only model features
            model_features = [
                p for p in pairs
                if p[0] not in ["match_date", "player_team", "name", "position", "shap_values"]
            ]

            top = []
            for feat, val in model_features[:top_n]:
                top.append({
                    "feature": feat,
                    "impact": round(float(val), 4),
                    "direction": "increase_risk" if val > 0 else "decrease_risk"
                })
            return top
    except (TypeError, ValueError):
        pass

    # If we get here, shap_values exists but isn't usable
    return _fallback_risk_drivers(row, top_n)


def _fallback_risk_drivers(row, top_n=5):
    """
    Generate risk drivers from key features when SHAP values aren't available.
    """
    drivers = []

    # Key features that typically drive injury risk
    risk_features = [
        ("acwr", "Acute:Chronic Workload Ratio", lambda v: v > 1.3),
        ("acute_load", "Recent Match Load", lambda v: v > 3),
        ("age", "Age Factor", lambda v: v > 30),
        ("previous_injuries", "Previous Injury Count", lambda v: v > 2),
        ("fatigue_index", "Fatigue Index", lambda v: v > 1),
        ("spike_flag", "Workload Spike", lambda v: v == 1),
        ("strain", "Training Strain", lambda v: v > 10),
        ("monotony", "Training Monotony", lambda v: v > 2),
        ("days_since_last_injury", "Days Since Last Injury", lambda v: v < 60),
        ("reinjury_rate", "Re-injury Rate", lambda v: v > 0.1),
    ]

    for col, display_name, is_elevated in risk_features:
        if col in row.index:
            val = row[col]
            try:
                val = float(val)
                elevated = is_elevated(val)
                drivers.append({
                    "feature": display_name,
                    "impact": round(val, 4),
                    "direction": "increase_risk" if elevated else "decrease_risk"
                })
            except (TypeError, ValueError):
                continue

    # Sort by absolute impact and return top N
    drivers = sorted(drivers, key=lambda x: abs(x["impact"]), reverse=True)
    return drivers[:top_n]

# ------------------------------------------------------------
# Training Flag System
# ------------------------------------------------------------
def panel_training_flag(row):
    """
    Generate RAG (Red/Amber/Green) training flag based on risk and severity.

    Red: High risk (>=0.60) or severe projected injury (>=21 days)
    Amber: Moderate risk (>=0.40) or moderate projected injury (>=7 days)
    Green: Low risk and minor projected injury
    """
    risk, _ = _get_risk_probability(row)
    sev = float(_get_safe_value(row, "severity_days", 0))

    if risk >= 0.60 or sev >= 21:
        flag = "Red"
        description = "High injury risk - reduced training load recommended"
    elif risk >= 0.40 or sev >= 7:
        flag = "Amber"
        description = "Elevated injury risk - monitor closely and manage load"
    else:
        flag = "Green"
        description = "Normal training - continue current program"

    return {
        "training_flag": flag,
        "description": description,
        "risk_score": round(risk, 3),
        "severity_days": round(sev, 1),
    }


# ------------------------------------------------------------
# Match Minutes Guidance
# ------------------------------------------------------------
def panel_minutes_guidance(row):
    """Generate match minutes guidance based on risk and archetype."""
    risk, _ = _get_risk_probability(row)
    archetype = _get_safe_value(row, "archetype", "Unknown")

    if risk >= 0.60:
        if archetype == "Injury Prone":
            guidance = "Limit exposure; avoid full matches."
            max_minutes = 60
        elif archetype == "Fragile":
            guidance = "Very limited minutes; control tightly."
            max_minutes = 30
        else:
            guidance = "Reduce match load; avoid intense spikes."
            max_minutes = 70
    elif risk >= 0.40:
        guidance = "Manage workload; avoid rapid increases."
        max_minutes = 80
    else:
        guidance = "Full availability."
        max_minutes = 90

    return {
        "minutes_guidance": guidance,
        "suggested_max_minutes": max_minutes,
        "risk_level": "high" if risk >= 0.60 else "moderate" if risk >= 0.40 else "low",
    }


# ------------------------------------------------------------
# Recommendation Panel
# ------------------------------------------------------------
def panel_recommendation(row):
    """Generate professional recommendation based on risk factors."""
    risk, _ = _get_risk_probability(row)
    severity_days = float(_get_safe_value(row, "severity_days", 0))
    archetype = _get_safe_value(row, "archetype", "Unknown")
    confidence = _get_safe_value(row, "confidence", "medium")

    return {
        "recommendation": generate_recommendation(
            risk=risk,
            severity_days=severity_days,
            archetype=archetype,
            confidence=confidence
        )
    }

# ------------------------------------------------------------
# FINAL DASHBOARD BUILDER
# ------------------------------------------------------------
def build_player_dashboard(inference_df, player_name):
    row = get_latest_snapshot(inference_df, player_name)

    return {
        "overview": panel_player_overview(row),
        "injury_risk": panel_injury_risk(row),
        "severity_projection": panel_severity_projection(row),
        "archetype_profile": panel_archetype(row["archetype"]),
        "top_drivers": panel_top_drivers(row),
        "training_flag": panel_training_flag(row),
        "minutes_guidance": panel_minutes_guidance(row),
        "recommendation": panel_recommendation(row)
    }