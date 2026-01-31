"""
Position-specific features for injury prediction.

Different positions have different injury patterns:
- Forwards: Hamstring injuries from sprinting, contact injuries
- Midfielders: Overuse injuries from high distance covered
- Defenders: Contact injuries, muscle strains from tackling
- Goalkeepers: Different injury profile altogether
"""

import pandas as pd
import numpy as np


# Position risk categories based on injury research
POSITION_INJURY_RISK = {
    # Forwards - high sprint load, hamstring risk
    "FW": {"base_risk": 0.7, "sprint_risk": 0.9, "contact_risk": 0.6},
    "CF": {"base_risk": 0.7, "sprint_risk": 0.9, "contact_risk": 0.7},
    "LW": {"base_risk": 0.75, "sprint_risk": 0.95, "contact_risk": 0.5},
    "RW": {"base_risk": 0.75, "sprint_risk": 0.95, "contact_risk": 0.5},
    "ST": {"base_risk": 0.7, "sprint_risk": 0.85, "contact_risk": 0.7},

    # Midfielders - high distance, overuse risk
    "MF": {"base_risk": 0.6, "sprint_risk": 0.6, "contact_risk": 0.5},
    "CM": {"base_risk": 0.55, "sprint_risk": 0.5, "contact_risk": 0.5},
    "CAM": {"base_risk": 0.65, "sprint_risk": 0.7, "contact_risk": 0.5},
    "CDM": {"base_risk": 0.6, "sprint_risk": 0.5, "contact_risk": 0.65},
    "LM": {"base_risk": 0.65, "sprint_risk": 0.75, "contact_risk": 0.45},
    "RM": {"base_risk": 0.65, "sprint_risk": 0.75, "contact_risk": 0.45},

    # Defenders - contact injuries, tackling
    "DF": {"base_risk": 0.55, "sprint_risk": 0.4, "contact_risk": 0.75},
    "CB": {"base_risk": 0.5, "sprint_risk": 0.3, "contact_risk": 0.8},
    "LB": {"base_risk": 0.6, "sprint_risk": 0.7, "contact_risk": 0.6},
    "RB": {"base_risk": 0.6, "sprint_risk": 0.7, "contact_risk": 0.6},
    "LWB": {"base_risk": 0.65, "sprint_risk": 0.8, "contact_risk": 0.55},
    "RWB": {"base_risk": 0.65, "sprint_risk": 0.8, "contact_risk": 0.55},

    # Goalkeepers - unique injury profile
    "GK": {"base_risk": 0.35, "sprint_risk": 0.1, "contact_risk": 0.4},
}


def normalize_position(pos: str) -> str:
    """Normalize position string to standard format."""
    if pd.isna(pos):
        return "MF"  # Default to midfielder

    pos = str(pos).upper().strip()

    # Handle compound positions (e.g., "DF,MF" -> take first)
    if "," in pos:
        pos = pos.split(",")[0].strip()

    # Common mappings
    mappings = {
        "FORWARD": "FW",
        "STRIKER": "ST",
        "WINGER": "LW",
        "MIDFIELDER": "MF",
        "ATTACKING MIDFIELDER": "CAM",
        "DEFENSIVE MIDFIELDER": "CDM",
        "CENTRAL MIDFIELDER": "CM",
        "DEFENDER": "DF",
        "CENTER BACK": "CB",
        "CENTRE BACK": "CB",
        "LEFT BACK": "LB",
        "RIGHT BACK": "RB",
        "FULL BACK": "RB",
        "GOALKEEPER": "GK",
        "KEEPER": "GK",
    }

    return mappings.get(pos, pos if pos in POSITION_INJURY_RISK else "MF")


def add_position_risk_features(df: pd.DataFrame,
                                position_column: str = "position") -> pd.DataFrame:
    """
    Add position-specific injury risk features.

    Args:
        df: DataFrame with player data
        position_column: Name of the position column

    Returns:
        DataFrame with additional position-based features
    """
    df = df.copy()

    if position_column not in df.columns:
        print(f"Warning: {position_column} not found. Skipping position features.")
        return df

    # Normalize positions
    df["position_normalized"] = df[position_column].apply(normalize_position)

    # Add risk scores from lookup
    df["position_base_risk"] = df["position_normalized"].apply(
        lambda x: POSITION_INJURY_RISK.get(x, POSITION_INJURY_RISK["MF"])["base_risk"]
    )
    df["position_sprint_risk"] = df["position_normalized"].apply(
        lambda x: POSITION_INJURY_RISK.get(x, POSITION_INJURY_RISK["MF"])["sprint_risk"]
    )
    df["position_contact_risk"] = df["position_normalized"].apply(
        lambda x: POSITION_INJURY_RISK.get(x, POSITION_INJURY_RISK["MF"])["contact_risk"]
    )

    # Position categories (one-hot style)
    df["is_forward"] = df["position_normalized"].isin(["FW", "CF", "LW", "RW", "ST"]).astype(int)
    df["is_midfielder"] = df["position_normalized"].isin(["MF", "CM", "CAM", "CDM", "LM", "RM"]).astype(int)
    df["is_defender"] = df["position_normalized"].isin(["DF", "CB", "LB", "RB", "LWB", "RWB"]).astype(int)
    df["is_goalkeeper"] = (df["position_normalized"] == "GK").astype(int)

    # Wide positions (more sprinting)
    df["is_wide_position"] = df["position_normalized"].isin(
        ["LW", "RW", "LM", "RM", "LB", "RB", "LWB", "RWB"]
    ).astype(int)

    # Central positions (more contact)
    df["is_central_position"] = df["position_normalized"].isin(
        ["CM", "CDM", "CB", "ST", "CF", "CAM"]
    ).astype(int)

    return df


def add_position_workload_interaction(df: pd.DataFrame,
                                       position_column: str = "position") -> pd.DataFrame:
    """
    Add interaction features between position and workload.

    High-risk combinations:
    - Winger + high sprint load = very high hamstring risk
    - Defender + high acute load = contact injury risk
    - Midfielder + high chronic load = overuse risk
    """
    df = df.copy()

    # Ensure position features exist
    if "position_sprint_risk" not in df.columns:
        df = add_position_risk_features(df, position_column)

    # Interaction features
    if "acute_load" in df.columns:
        # Sprint risk * acute load
        df["sprint_load_risk"] = df["position_sprint_risk"] * df["acute_load"]

        # Contact risk * acute load
        df["contact_load_risk"] = df["position_contact_risk"] * df["acute_load"]

    if "acwr" in df.columns:
        # Position-adjusted ACWR
        df["position_adjusted_acwr"] = df["acwr"] * df["position_base_risk"]

    if "matches_last_7" in df.columns:
        # Congestion risk for wide players (lots of sprinting)
        df["wide_player_congestion"] = df["is_wide_position"] * df["matches_last_7"]

        # Congestion risk for defenders (lots of contact)
        df["defender_congestion"] = df["is_defender"] * df["matches_last_7"]

    if "age" in df.columns:
        # Age * position interaction (older forwards = higher hamstring risk)
        df["age_forward_risk"] = (
            df["is_forward"].astype(int) *
            np.maximum(0, df["age"] - 28) / 10  # Risk increases after 28
        )

        # Age * defender (older defenders = higher muscle strain risk)
        df["age_defender_risk"] = (
            df["is_defender"].astype(int) *
            np.maximum(0, df["age"] - 30) / 10  # Risk increases after 30
        )

    return df
