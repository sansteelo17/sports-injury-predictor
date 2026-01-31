import pandas as pd
import numpy as np

BODY_MAP = {
    "ankle": ["ankle"],
    "knee": ["knee", "cruciate", "acl", "pcl", "lcl", "mcl", "meniscus", "ligament"],
    "calf": ["calf", "achilles", "tendon rupture", "tendon irritation"],
    "hamstring": ["hamstring"],
    "groin": ["groin", "adductor", "pubic"],
    "thigh": ["thigh", "quad", "quadricep"],
    "hip": ["hip"],
    "back": ["back" ],
    "shoulder": ["shoulder"],
    "arm": ["arm", "elbow", "hand", "wrist", "forearm"],
    "neck": ["neck"],
    "head": ["head", "concussion", "face", "facial", "broken cheekbone"],
    "chest": ["chest", "rib", "ribs"],
    "toe": ["toe"],
    "foot": ["foot", "metatarsal"],
    "leg": ["leg", "shin", "fibula", "tibia", "hairline fracture", "stress reaction of the bone",],
    "muscle": ["muscle", "muscular" ],
    "abdomen": ["abdominal"],
}

INJURY_TYPE_MAP = {
    "tear": ["torn", "tear", "rupture", "acl", "mcl", "lcl", "pcl", "meniscus"],
    "strain": ["strain", "pulled", "tightness", "muscular", "muscle injury"],
    "sprain": ["sprain"],
    "fracture": ["fracture", "broken", "break"],
    "contusion": ["contusion", "dead leg", "bruise"],
    "fatigue": ["fatigue", "fitness"],
    "inflammation": ["inflammation", "itis"],
    "tendon": ["tendon", "tendinopathy", "tendonitis"],
    "ligament": ["ligament"],
    "cramp": ["cramp", "spasm"],
    "illness": ["ill", "infection", "virus", "flu", "cold", "malaria"],
    "surgery": ["surgery", "appendectomy", "operation", "procedure"],
    "impact": ["collision", "contact"],
    "pain": ["pain", "soreness", "ache"],
    "problem": ["problem", "problems", "issue", "issues", "niggle", "knock"],
}

def normalize_injury(text):
    return str(text).strip().lower()

def build_severity_dataset(injury_df: pd.DataFrame,
                           team_matches: pd.DataFrame) -> pd.DataFrame:

    df = injury_df.copy()

    df["severity_days"] = (
        df["date_of_return"] - df["date_of_injury"]
    ).dt.days

    df = df.dropna(subset=["severity_days"])
    df = df[df["severity_days"] > 0]

    df = df.sort_values(["player_team", "injury_datetime"])

    results = []

    for team in df["player_team"].unique():
        left_team = df[df["player_team"] == team].copy()
        right_team = team_matches[team_matches["team"] == team].copy()

        merged = pd.merge_asof(
            left_team.sort_values("injury_datetime"),
            right_team.sort_values("match_date"),
            left_on="injury_datetime",
            right_on="match_date",
            direction="backward"
        )

        results.append(merged)

    return pd.concat(results, ignore_index=True)

def clean_severity_dataset(severity_final):
    df = severity_final.copy()

    # 1. Drop all `_x` versions of columns
    x_cols = [col for col in df.columns if col.endswith("_x")]
    df = df.drop(columns=x_cols, errors="ignore")

    # 2. Rename `_y` → clean names
    rename_map = {col: col.replace("_y", "") for col in df.columns if col.endswith("_y")}
    df = df.rename(columns=rename_map)

    # 3. Remove accidental duplicate columns after rename
    df = df.loc[:, ~df.columns.duplicated()]

    # 4. Ensure target exists
    if "severity_days" not in df.columns:
        raise ValueError("severity_days column missing — ensure you computed it first.")

    return df

def classify_injury_type(text: str) -> str:
    t = normalize_injury(text)

    # 1. Explicit keyword mapping first
    for label, keywords in INJURY_TYPE_MAP.items():
        if any(k in t for k in keywords):
            return label

    # 2. Illness / infection cases
    if any(k in t for k in ["fever", "shingles", "depression", "quarantine"]):
        return "illness"

    # 3. Stress reaction → inflammation
    if "stress reaction" in t:
        return "inflammation"

    # 4. Try to infer body area
    area = extract_body_area(t)

    # 5. Soft-tissue pattern "[body] injury" → strain
    if "injury" in t:
        return "strain"

    # 6. Ankle injury defaults to sprain
    if "ankle injury" in t or "injury to the ankle" in t:
        return "sprain"

    # 7. Head injury → contusion
    if "head injury" in t or "concussion" in t:
        return "contusion"

    # 8. Unknown injury
    if "unknown" in t:
        return "unknown"

    # 9. Otherwise
    return "other"

def extract_body_area(text):
    t = normalize_injury(text)

    # Handle multi-area injuries
    if " and " in t:
        for area, kws in BODY_MAP.items():
            if any(k in t for k in kws):
                return area

    # Direct "___ injury" matches
    for area in BODY_MAP:
        if area in t:
            return area

    # Keyword dictionary
    for area, kws in BODY_MAP.items():
        if any(k in t for k in kws):
            return area

    return "other"

def is_body_only_injury(text: str) -> int:
    """
    Injury with no mechanism keyword.
    e.g., 'ankle injury', 'knee injury' ⇒ body_only=1
    """
    t = normalize_injury(text)

    # If ANY mechanism keyword is present → it's NOT body-only
    if any(k in t for group in INJURY_TYPE_MAP.values() for k in group):
        return 0
    return 1


def build_injury_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["injury_clean"] = df["injury"].apply(normalize_injury)
    df["injury_type"] = df["injury_clean"].apply(classify_injury_type)
    df["body_area"] = df["injury_clean"].apply(extract_body_area)
    df["injury_is_body_only"] = df["injury_clean"].apply(is_body_only_injury)
    df["injury_complexity_flag"] = df["injury_clean"].apply(
        lambda x: 1 if " and " in x else 0
    )

    return df


# ============================================================================
# REGRESSION-SPECIFIC FEATURES FOR SEVERITY PREDICTION
# ============================================================================
#
# These features are designed to predict HOW LONG an injury will last.
# They capture factors that influence recovery time beyond just injury type.
#
# ============================================================================

# Historical severity by injury type (baseline expectations)
INJURY_TYPE_SEVERITY = {
    # Serious injuries (60+ days typical)
    "tear": 90,      # ACL tears, muscle tears - long recovery
    "fracture": 75,  # Broken bones need time to heal
    "surgery": 120,  # Post-surgical recovery is lengthy

    # Moderate injuries (20-60 days)
    "strain": 25,    # Muscle strains vary widely
    "sprain": 21,    # Ligament sprains
    "tendon": 35,    # Tendon issues can linger
    "ligament": 45,  # Non-tear ligament injuries

    # Minor injuries (7-20 days)
    "contusion": 10, # Bruises heal relatively quickly
    "inflammation": 14,
    "cramp": 7,
    "pain": 10,
    "fatigue": 7,

    # Variable
    "illness": 10,   # Depends on illness type
    "impact": 14,
    "problem": 14,
    "other": 21,
    "unknown": 21,
}

# Body area affects recovery time
BODY_AREA_SEVERITY = {
    # Lower body (weight-bearing) - longer recovery
    "knee": 1.3,      # Weight-bearing, complex joint
    "ankle": 1.1,     # Weight-bearing
    "hamstring": 1.2, # High recurrence risk
    "calf": 1.1,
    "thigh": 1.0,
    "groin": 1.15,    # Difficult to fully rest
    "hip": 1.2,
    "foot": 1.1,
    "leg": 1.0,
    "toe": 0.8,

    # Upper body - shorter recovery for footballers
    "shoulder": 0.9,
    "arm": 0.7,
    "chest": 0.9,
    "back": 1.1,      # Can affect movement
    "neck": 1.0,

    # Head - protocol dependent
    "head": 0.8,      # Concussion protocols

    # Other
    "muscle": 1.0,
    "abdomen": 0.9,
    "other": 1.0,
}


def add_severity_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features specifically designed for predicting injury DURATION.

    New Features Added:
    -------------------
    1. expected_severity_days: Baseline expectation based on injury type
       - Uses historical averages for each injury type
       - Helps model learn deviations from typical patterns

    2. body_area_multiplier: How body location affects recovery
       - Lower body injuries take longer (weight-bearing)
       - Upper body shorter for footballers (less critical)

    3. adjusted_expected_days: expected_severity * body_area_multiplier
       - Combined baseline prediction

    4. is_serious_injury: Binary flag for injuries typically 30+ days
       - Tears, fractures, surgery, ligament damage

    5. is_recurrence_prone: Body areas with high re-injury rates
       - Hamstring, groin, calf - often recur if rushed

    6. age_recovery_factor: Older players typically recover slower
       - Under 25: 0.9 (faster recovery)
       - 25-30: 1.0 (baseline)
       - Over 30: 1.1-1.2 (slower recovery)

    7. workload_at_injury: Recent match load when injured
       - High workload injuries may indicate fatigue-related issues
       - Often heal faster once rested

    8. season_timing: When in season injury occurred
       - Early season: More conservative treatment
       - Late season: May rush recovery for playoffs

    Args:
        df: DataFrame with injury_type, body_area, age columns

    Returns:
        DataFrame with new severity prediction features
    """
    df = df.copy()

    # 1. Expected severity based on injury type
    df["expected_severity_days"] = df["injury_type"].map(INJURY_TYPE_SEVERITY).fillna(21)

    # 2. Body area multiplier
    df["body_area_multiplier"] = df["body_area"].map(BODY_AREA_SEVERITY).fillna(1.0)

    # 3. Adjusted expected days
    df["adjusted_expected_days"] = df["expected_severity_days"] * df["body_area_multiplier"]

    # 4. Serious injury flag
    serious_types = ["tear", "fracture", "surgery", "ligament"]
    df["is_serious_injury"] = df["injury_type"].isin(serious_types).astype(int)

    # 5. Recurrence-prone body areas
    recurrence_areas = ["hamstring", "groin", "calf", "thigh"]
    df["is_recurrence_prone"] = df["body_area"].isin(recurrence_areas).astype(int)

    # 6. Age recovery factor
    if "age" in df.columns:
        df["age_recovery_factor"] = df["age"].apply(
            lambda x: 0.9 if x < 25 else (1.0 if x <= 30 else (1.1 if x <= 33 else 1.2))
        )
    else:
        df["age_recovery_factor"] = 1.0

    # 7. Workload at injury (if available)
    if "matches_last_7" in df.columns:
        df["workload_at_injury"] = df["matches_last_7"]
        df["high_workload_injury"] = (df["matches_last_7"] >= 3).astype(int)

    if "matches_last_30" in df.columns:
        df["monthly_load_at_injury"] = df["matches_last_30"]

    # 8. Season timing (if date available)
    date_cols = ["date_of_injury", "injury_datetime", "Date of Injury"]
    date_col = None
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df["injury_month"] = df[date_col].dt.month

        # Season phase: Aug-Oct (early), Nov-Feb (mid), Mar-May (late/playoffs)
        df["season_phase"] = df["injury_month"].apply(
            lambda m: "early" if m in [8, 9, 10] else ("late" if m in [3, 4, 5] else "mid")
        )

    # 9. Injury complexity (multiple body areas)
    if "injury_complexity_flag" not in df.columns and "injury_clean" in df.columns:
        df["injury_complexity_flag"] = df["injury_clean"].apply(
            lambda x: 1 if " and " in str(x) else 0
        )

    return df


def add_team_recovery_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add team-level features as a PROXY for medical staff quality.

    We can't directly measure medical staff quality, but we CAN measure:
    - How quickly players at each team historically recover
    - Whether a team tends to rush players back (higher re-injury rates)
    - Team's average injury severity

    These are computed from historical data and represent the team's
    "medical environment" - a combination of staff quality, facilities,
    and injury management philosophy.

    Features Added:
    ---------------
    1. team_avg_recovery_days: Historical average recovery time for this team
    2. team_recovery_ratio: Team's avg recovery / league avg (< 1 = faster than avg)
    3. team_injury_count: How many injuries this team has had (more data = more reliable)
    4. player_vs_team_recovery: Player's avg vs their team's avg

    Args:
        df: DataFrame with 'player_team' and 'severity_days' columns

    Returns:
        DataFrame with team recovery features added
    """
    df = df.copy()

    team_col = None
    for col in ['player_team', 'team', 'Team']:
        if col in df.columns:
            team_col = col
            break

    if team_col is None or 'severity_days' not in df.columns:
        print("Warning: Cannot compute team recovery features (missing team or severity_days)")
        return df

    # Compute team-level statistics
    team_stats = df.groupby(team_col)['severity_days'].agg([
        ('team_avg_recovery_days', 'mean'),
        ('team_median_recovery_days', 'median'),
        ('team_injury_count', 'count'),
        ('team_recovery_std', 'std')
    ]).reset_index()

    # League average for comparison
    league_avg = df['severity_days'].mean()
    team_stats['team_recovery_ratio'] = team_stats['team_avg_recovery_days'] / league_avg

    # Merge back
    df = df.merge(team_stats, on=team_col, how='left')

    # Fill missing values for teams with few injuries
    df['team_avg_recovery_days'] = df['team_avg_recovery_days'].fillna(league_avg)
    df['team_recovery_ratio'] = df['team_recovery_ratio'].fillna(1.0)
    df['team_injury_count'] = df['team_injury_count'].fillna(1)

    # Player vs team comparison (if player has multiple injuries)
    if 'name' in df.columns:
        player_avg = df.groupby('name')['severity_days'].transform('mean')
        df['player_avg_recovery'] = player_avg
        df['player_vs_team_recovery'] = df['player_avg_recovery'] / df['team_avg_recovery_days']
        df['player_vs_team_recovery'] = df['player_vs_team_recovery'].fillna(1.0)

    print(f"Added team recovery features. Teams with fastest recovery:")
    fast_teams = team_stats.nsmallest(5, 'team_avg_recovery_days')[[team_col, 'team_avg_recovery_days', 'team_injury_count']]
    print(fast_teams.to_string(index=False))

    return df


def add_player_injury_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add player-specific injury history features.

    These capture whether a player is "injury-prone" and their
    personal recovery patterns.

    Features Added:
    ---------------
    1. player_injury_count: Total injuries for this player
    2. player_avg_severity: Player's average injury duration
    3. is_injury_prone: 1 if player has 3+ injuries
    4. player_worst_injury: Longest injury duration for this player
    5. days_since_last_injury_same_area: Recurrence indicator

    Args:
        df: DataFrame with player and injury information

    Returns:
        DataFrame with player history features added
    """
    df = df.copy()

    if 'name' not in df.columns:
        print("Warning: Cannot compute player history features (missing 'name' column)")
        return df

    # Player-level statistics
    player_stats = df.groupby('name').agg({
        'severity_days': ['count', 'mean', 'max', 'std']
    }).reset_index()
    player_stats.columns = ['name', 'player_injury_count', 'player_avg_severity',
                            'player_worst_injury', 'player_severity_std']

    # Injury-prone flag
    player_stats['is_injury_prone'] = (player_stats['player_injury_count'] >= 3).astype(int)

    # Merge back
    df = df.merge(player_stats, on='name', how='left')

    # Fill missing
    df['player_injury_count'] = df['player_injury_count'].fillna(1)
    df['player_avg_severity'] = df['player_avg_severity'].fillna(df['severity_days'])
    df['is_injury_prone'] = df['is_injury_prone'].fillna(0)

    # Same body area recurrence (if we have body_area and dates)
    if 'body_area' in df.columns and 'date_of_injury' in df.columns:
        df = df.sort_values(['name', 'date_of_injury'])
        df['prev_injury_same_area'] = (
            (df.groupby(['name', 'body_area']).cumcount() > 0).astype(int)
        )

    print(f"Added player history features.")
    print(f"  Injury-prone players (3+ injuries): {df['is_injury_prone'].sum()}")

    return df


def get_severity_feature_descriptions() -> dict:
    """
    Returns descriptions of all severity prediction features.
    Useful for documentation and SHAP interpretation.
    """
    return {
        # Injury characteristics
        "injury_type": "Classified injury mechanism (tear, strain, fracture, etc.)",
        "body_area": "Body part affected (knee, hamstring, ankle, etc.)",
        "injury_is_body_only": "1 if injury has no specific mechanism (e.g., 'knee injury')",
        "injury_complexity_flag": "1 if multiple body parts affected",

        # Severity predictions
        "expected_severity_days": "Historical average days out for this injury type",
        "body_area_multiplier": "Recovery time multiplier based on body location",
        "adjusted_expected_days": "expected_severity * body_area_multiplier",

        # Risk flags
        "is_serious_injury": "1 for tears, fractures, surgery, ligament damage",
        "is_recurrence_prone": "1 for hamstring, groin, calf injuries (high re-injury risk)",

        # Player factors
        "age_recovery_factor": "Age-based recovery speed (younger=faster)",
        "age": "Player age at time of injury",

        # Workload context
        "workload_at_injury": "Matches in last 7 days when injured",
        "high_workload_injury": "1 if 3+ matches in week before injury",
        "monthly_load_at_injury": "Matches in last 30 days when injured",

        # Timing
        "injury_month": "Month of injury (1-12)",
        "season_phase": "early (Aug-Oct), mid (Nov-Feb), late (Mar-May)",

        # Target
        "severity_days": "Days out due to injury (prediction target)",
    }
