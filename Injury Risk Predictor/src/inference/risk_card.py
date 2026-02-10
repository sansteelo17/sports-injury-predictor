import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from ..models.archetype import get_archetype_profile as get_central_archetype_profile

logger = get_logger(__name__)


def _get_risk_probability(player_row):
    """
    Get the primary risk probability from the player row.
    Supports both ensemble and individual model outputs.

    Priority: ensemble_prob > catboost_prob > lgb_prob > xgb_prob
    """
    prob_cols = ["ensemble_prob", "catboost_prob", "lgb_prob", "xgb_prob"]

    for col in prob_cols:
        if col in player_row.index and pd.notna(player_row[col]):
            return float(player_row[col]), col

    raise KeyError(
        f"No risk probability column found. Expected one of: {prob_cols}. "
        f"Available columns: {list(player_row.index)}"
    )


def _get_model_support(player_row):
    """
    Extract model agreement/support information for the risk card.
    Handles both ensemble and individual model outputs.
    """
    support = {}

    # Check for ensemble probability
    if "ensemble_prob" in player_row.index and pd.notna(player_row["ensemble_prob"]):
        support["ensemble"] = round(float(player_row["ensemble_prob"]), 3)
        support["model_type"] = "stacking_ensemble"

    # Check for individual model probabilities
    model_cols = {
        "catboost_prob": "catboost",
        "lgb_prob": "lightgbm",
        "xgb_prob": "xgboost"
    }

    for col, name in model_cols.items():
        if col in player_row.index and pd.notna(player_row[col]):
            support[name] = round(float(player_row[col]), 3)

    # Add agreement score if available
    if "agreement" in player_row.index and pd.notna(player_row["agreement"]):
        support["agreement_score"] = int(player_row["agreement"])

    # Determine model type if not already set
    if "model_type" not in support:
        if len([k for k in support if k in ["catboost", "lightgbm", "xgboost"]]) > 1:
            support["model_type"] = "individual_ensemble"
        elif "catboost" in support:
            support["model_type"] = "catboost_only"
        else:
            support["model_type"] = "unknown"

    return support


def _classify_risk_level(risk_prob):
    """Classify risk probability into human-readable level."""
    if risk_prob >= 0.70:
        return "critical"
    elif risk_prob >= 0.60:
        return "high"
    elif risk_prob >= 0.40:
        return "moderate"
    elif risk_prob >= 0.20:
        return "low"
    else:
        return "minimal"


def _classify_severity_level(severity_days):
    """Classify severity days into human-readable level."""
    if severity_days >= 30:
        return "catastrophic"
    elif severity_days >= 14:
        return "major"
    elif severity_days >= 7:
        return "moderate"
    else:
        return "minor"


def _extract_workload_summary(player_row):
    """Extract workload metrics into a summary dict."""
    workload = {}

    # Recent match load
    load_cols = ["matches_last_7", "matches_last_14", "matches_last_30"]
    for col in load_cols:
        if col in player_row.index and pd.notna(player_row[col]):
            workload[col] = int(player_row[col])

    # Workload ratios and metrics
    ratio_cols = ["acute_load", "chronic_load", "acwr", "fatigue_index", "strain"]
    for col in ratio_cols:
        if col in player_row.index and pd.notna(player_row[col]):
            workload[col] = round(float(player_row[col]), 2)

    # Spike flag
    if "spike_flag" in player_row.index:
        workload["spike_flag"] = bool(player_row["spike_flag"])

    # Rest days
    if "rest_days_before_injury" in player_row.index and pd.notna(player_row["rest_days_before_injury"]):
        workload["rest_days"] = int(player_row["rest_days_before_injury"])
    elif "rest_days" in player_row.index and pd.notna(player_row["rest_days"]):
        workload["rest_days"] = int(player_row["rest_days"])

    return workload if workload else None


def _extract_injury_history_summary(player_row):
    """Extract injury history into a summary dict."""
    history = {}

    if "previous_injuries" in player_row.index and pd.notna(player_row["previous_injuries"]):
        history["previous_injuries"] = int(player_row["previous_injuries"])

    if "days_since_last_injury" in player_row.index and pd.notna(player_row["days_since_last_injury"]):
        days = int(player_row["days_since_last_injury"])
        history["days_since_last_injury"] = days
        # Classify recency
        if days <= 30:
            history["recency"] = "very_recent"
        elif days <= 90:
            history["recency"] = "recent"
        elif days <= 365:
            history["recency"] = "moderate"
        elif days >= 999:
            history["recency"] = "no_history"
        else:
            history["recency"] = "distant"

    return history if history else None


def _get_safe_value(player_row, col, default=None):
    """Safely get a value from player_row with fallback."""
    if col in player_row.index and pd.notna(player_row[col]):
        return player_row[col]
    return default


def _extract_key_risk_factors(player_row, top_k=5):
    """
    Extract key risk factors when SHAP values aren't available.

    Uses domain knowledge to identify which features are elevated
    and contributing to injury risk.
    """
    factors = []

    # Define risk features with thresholds and display names
    risk_features = [
        # (column, display_name, threshold_for_elevated, higher_is_worse)
        ("acwr", "Acute:Chronic Workload Ratio", 1.3, True),
        ("acute_load", "Recent Match Load (7 days)", 3, True),
        ("fatigue_index", "Fatigue Index", 1.0, True),
        ("spike_flag", "Workload Spike Detected", 0.5, True),
        ("strain", "Training Strain", 10, True),
        ("monotony", "Training Monotony", 2.0, True),
        ("previous_injuries", "Previous Injury Count", 2, True),
        ("is_injury_prone", "Injury-Prone Flag", 0.5, True),
        ("age", "Age Factor", 30, True),
        ("days_since_last_injury", "Days Since Last Injury", 60, False),  # Lower is worse
        ("chronic_load", "Chronic Load (28 days)", 4, True),
        ("matches_last_7", "Matches Last 7 Days", 2, True),
        ("matches_last_14", "Matches Last 14 Days", 4, True),
        ("player_injury_count", "Total Career Injuries", 3, True),
    ]

    for col, display_name, threshold, higher_is_worse in risk_features:
        if col in player_row.index:
            val = player_row[col]
            try:
                val = float(val)
                if pd.isna(val):
                    continue

                # Determine if this factor is elevated (contributing to risk)
                if higher_is_worse:
                    is_elevated = val >= threshold
                else:
                    is_elevated = val <= threshold

                # Calculate relative impact (how far from threshold)
                if higher_is_worse:
                    impact = (val - threshold) / max(threshold, 1)
                else:
                    impact = (threshold - val) / max(threshold, 1)

                factors.append({
                    "feature": display_name,
                    "value": round(val, 2),
                    "impact": round(impact, 3),
                    "direction": "increase_risk" if is_elevated else "decrease_risk"
                })
            except (TypeError, ValueError):
                continue

    # Sort by absolute impact and return top_k
    factors = sorted(factors, key=lambda x: abs(x["impact"]), reverse=True)
    return factors[:top_k]


def build_risk_card(player_row, feature_cols=None, top_k=5):
    """
    Builds a comprehensive player risk card from the inference dataframe row.

    Supports both ensemble (build_inference_df_with_ensemble) and individual
    model (build_full_inference_df) outputs.

    Args:
        player_row: A pandas Series (row) from the inference DataFrame
        feature_cols: List of feature column names (optional, needed for SHAP factors)
        top_k: Number of top SHAP factors to include (default: 5)

    Returns:
        dict: Risk card with the following structure:
            - name: Player name
            - team: Team name
            - position: Playing position
            - risk_assessment: Risk probability, level, and model support
            - severity_projection: Days lost projection and severity level
            - archetype: Player's risk archetype and profile
            - workload_summary: Recent match load and fatigue metrics
            - injury_history: Previous injuries and recency
            - top_factors: Top SHAP contributors (if available)
            - training_flag: Red/Amber/Green status
            - recommendation: Actionable recommendation text

    Example:
        >>> row = get_latest_snapshot(inference_df, "Mohamed Salah")
        >>> card = build_risk_card(row, feature_cols=model.feature_names_)
        >>> print(card["training_flag"])
        'Amber'
    """
    # Get player identifiers
    name = player_row["name"]
    team = _get_safe_value(player_row, "team") or _get_safe_value(player_row, "player_team", "Unknown")
    position = _get_safe_value(player_row, "position", "Unknown")

    # Get risk probability (supports both ensemble and individual models)
    risk_prob, prob_source = _get_risk_probability(player_row)
    risk_level = _classify_risk_level(risk_prob)

    # Get severity
    severity_days = float(_get_safe_value(player_row, "severity_days", 0))
    severity_level = _classify_severity_level(severity_days)

    # Get archetype and confidence
    archetype = _get_safe_value(player_row, "archetype", "Unknown")
    confidence = _get_safe_value(player_row, "confidence", "medium")

    logger.debug(f"Generating risk card for {name} (risk={risk_prob:.3f} from {prob_source}, severity={severity_days:.1f}d)")

    # -----------------------------
    # 1. Risk Assessment
    # -----------------------------
    risk_assessment = {
        "probability": round(risk_prob, 3),
        "level": risk_level,
        "probability_source": prob_source,
        "model_support": _get_model_support(player_row),
        "confidence": confidence,
    }

    # -----------------------------
    # 2. Severity Projection
    # -----------------------------
    severity_projection = {
        "projected_days_lost": round(severity_days, 1),
        "severity_level": severity_level,
    }

    # -----------------------------
    # 3. Top risk factors (SHAP if available, otherwise key features)
    # -----------------------------
    top_factors = None

    # Try SHAP values first
    if feature_cols is not None and "shap_values" in player_row.index:
        shap_vals = player_row["shap_values"]
        if shap_vals is not None:
            shap_vals = np.array(shap_vals)
            # Zip feature names with SHAP impacts
            pairs = list(zip(feature_cols, shap_vals[:len(feature_cols)]))
            # Sort by absolute SHAP magnitude
            pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
            # Select top_k
            top_factors = [
                {
                    "feature": feat,
                    "impact": round(float(val), 4),
                    "direction": "increase_risk" if val > 0 else "decrease_risk"
                }
                for feat, val in pairs[:top_k]
            ]

    # Fallback: generate from key risk features if no SHAP
    if top_factors is None:
        top_factors = _extract_key_risk_factors(player_row, top_k)

    # -----------------------------
    # 4. Training Flag (RAG status)
    # -----------------------------
    if risk_prob >= 0.60 or severity_days >= 21:
        training_flag = "Red"
    elif risk_prob >= 0.40 or severity_days >= 7:
        training_flag = "Amber"
    else:
        training_flag = "Green"

    # -----------------------------
    # 5. Recommendation
    # -----------------------------
    recommendation = generate_recommendation(
        risk_prob,
        severity_days,
        archetype,
        confidence
    )

    # -----------------------------
    # 6. Build Card Output
    # -----------------------------
    risk_card = {
        "name": name,
        "team": team,
        "position": position,
        "risk_assessment": risk_assessment,
        "severity_projection": severity_projection,
        "archetype": {
            "name": archetype,
            "profile": _get_archetype_profile(archetype),
        },
        "workload_summary": _extract_workload_summary(player_row),
        "injury_history": _extract_injury_history_summary(player_row),
        "top_factors": top_factors,
        "training_flag": training_flag,
        "recommendation": recommendation,
    }

    logger.info(f"Risk card generated for {name}: {training_flag} flag, {archetype} archetype, {confidence} confidence")

    return risk_card


def _get_archetype_profile(archetype):
    """
    Get the archetype profile details using the centralized archetype definitions.

    Parameters
    ----------
    archetype : str
        Archetype name (short or full name)

    Returns
    -------
    dict
        Profile with description, training_focus, risk_tendency, etc.
    """
    profile = get_central_archetype_profile(archetype)

    return {
        "description": profile.get("description", "Unknown archetype profile."),
        "training_focus": profile.get("training_focus", "Standard monitoring recommended."),
        "risk_tendency": profile.get("risk_level", "unknown"),
        "key_characteristics": profile.get("key_characteristics", []),
        "minutes_strategy": profile.get("minutes_strategy", "Standard management"),
    }

def generate_recommendation(risk, severity_days, archetype, confidence="medium"):
    """
    Professional recommendation logic for high-performance staff.

    Uses the centralized archetype definitions to provide consistent,
    archetype-specific recommendations.

    Parameters
    ----------
    risk : float
        Risk probability (0-1)
    severity_days : float
        Projected days lost if injury occurs
    archetype : str
        Player's archetype name
    confidence : str
        Model confidence level ('very-high', 'high', 'medium', 'low')

    Returns
    -------
    str
        Detailed recommendation text
    """

    # ------------------------------------------------------------
    # 1. Risk Classification
    # ------------------------------------------------------------
    if risk >= 0.60:
        risk_level = "high"
    elif risk >= 0.40:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # ------------------------------------------------------------
    # 2. Severity Classification (days lost projection)
    # ------------------------------------------------------------
    if severity_days >= 30:
        sev_level = "catastrophic"
    elif severity_days >= 14:
        sev_level = "major"
    elif severity_days >= 7:
        sev_level = "moderate"
    else:
        sev_level = "minor"

    # ------------------------------------------------------------
    # 3. Get Archetype Profile from centralized definitions
    # ------------------------------------------------------------
    archetype_profile = get_central_archetype_profile(archetype)

    # Build profile dict for recommendation generation
    profile = {
        "tendencies": archetype_profile.get("description", ""),
        "interventions": archetype_profile.get("training_focus", ""),
        "minutes_strategy": archetype_profile.get("minutes_strategy", "Standard management"),
    }

    # ------------------------------------------------------------
    # 4. Confidence Interpretation
    # ------------------------------------------------------------
    if confidence == "very-high":
        confidence_msg = "Model agreement is strong; recommendation should be prioritized."
    elif confidence == "high":
        confidence_msg = "High model agreement; intervention is advised."
    elif confidence == "medium":
        confidence_msg = "Moderate confidence; monitor response and adjust as needed."
    else:
        confidence_msg = "Low confidence; treat this as informational rather than directive."

    # ------------------------------------------------------------
    # 5. Final Recommendation Logic
    # ------------------------------------------------------------
    if risk_level == "high":
        if sev_level in ["catastrophic", "major"]:
            rec = (
                f"High injury risk with a projected {sev_level} severity. "
                f"Training exposure should be reduced immediately. "
                f"Archetype insight: {profile['tendencies']}. "
                f"Recommended training approach: {profile['interventions']}. "
                f"Match-minute strategy: {profile['minutes_strategy']}. "
                f"{confidence_msg}"
            )
        else:
            rec = (
                f"High injury risk with a projected {sev_level} severity. "
                f"A reduction in workload is recommended to prevent escalation. "
                f"Archetype insight: {profile['tendencies']}. "
                f"Training approach: emphasize recovery and stable loading. "
                f"Interventions: {profile['interventions']}. "
                f"Match-minute strategy: {profile['minutes_strategy']}. "
                f"{confidence_msg}"
            )

    elif risk_level == "moderate":
        rec = (
            f"Moderate injury risk with a projected {sev_level} severity. "
            f"Archetype insight: {profile['tendencies']}. "
            f"Recommended adjustments: {profile['interventions']}. "
            f"Match-minute strategy: {profile['minutes_strategy']}. "
            f"{confidence_msg}"
        )

    else:  # low risk
        rec = (
            f"Low injury risk with a projected {sev_level} severity. "
            f"Player may remain on a normal training structure. "
            f"Archetype insight: {profile['tendencies']}. "
            f"Minutes strategy: {profile['minutes_strategy']}. "
            f"{confidence_msg}"
        )

    return rec