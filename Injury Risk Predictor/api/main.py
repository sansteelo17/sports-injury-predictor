"""
FastAPI backend for EPL Injury Risk Predictor.

Serves predictions from the trained ML models to the React frontend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
import os

from pathlib import Path
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path=None, *args, **kwargs):
        """Lightweight .env loader fallback when python-dotenv is unavailable."""
        env_path = Path(path) if path else Path(".env")
        if not env_path.exists():
            return False
        try:
            for raw_line in env_path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            return True
        except Exception:
            return False

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env")

from src.utils.logger import get_logger
logger = get_logger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.model_io import load_artifacts
from src.inference.story_generator import (
    generate_player_story,
    generate_risk_factors_list,
    get_recommendation_text,
    get_fpl_insight,
    calculate_scoring_odds,
    get_fpl_value_assessment,
    calculate_clean_sheet_odds,
    generate_yara_response,
    generate_lab_notes,
)
from src.data_loaders.fpl_api import FPLClient, get_fpl_insights
from src.data_loaders.football_data_api import FootballDataClient, get_standings_summary
from src.data_loaders.odds_api import OddsClient, get_clean_sheet_insight
# Archetype clustering imports are done lazily inside assign_hybrid_archetypes()
# to avoid importing all of src.models (which pulls in lightgbm, xgboost, etc.)

app = FastAPI(
    title="EPL Injury Risk Predictor API",
    description="ML-powered injury risk predictions for Premier League players",
    version="1.0.0",
)

# CORS for React frontend (local dev + Render deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://epl-injury-frontend.onrender.com",
        "https://injurywatch.onrender.com",
        "https://yara-sports-frontend.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FPL team names -> display names mapping
FPL_TEAM_DISPLAY_NAMES = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Wolves": "Wolverhampton",
    "Brighton": "Brighton Hove",
    "Nott'm Forest": "Nottingham",
    "Leeds": "Leeds United",
}


def _fpl_team_to_df_team(fpl_team: str) -> str:
    """Convert FPL team name to our display name."""
    return FPL_TEAM_DISPLAY_NAMES.get(fpl_team, fpl_team)


# Load models at startup
artifacts = None
inference_df = None
fpl_stats_cache = {}  # FPL stats indexed by name
fpl_team_ids = {}  # Team name -> FPL team ID (for badges)
fpl_player_codes = {}  # Player name -> FPL code (for photos)
fpl_players_by_team = {}  # Team name -> set of FPL player names (for filtering)
odds_client = None


@app.on_event("startup")
async def load_models():
    global artifacts, inference_df, fpl_stats_cache, fpl_team_ids, fpl_player_codes, fpl_players_by_team, odds_client
    artifacts = load_artifacts()
    if artifacts and "inference_df" in artifacts:
        inference_df = artifacts["inference_df"]
        print(f"Loaded {len(inference_df)} player predictions")
    else:
        print("WARNING: No trained models found. API will return errors.")

    # Load FPL stats
    try:
        client = FPLClient()
        all_stats = client.get_all_player_stats()
        # Index by multiple keys for matching
        for p in all_stats:
            name_lower = p["name"].lower()
            full_lower = p["full_name"].lower()
            fpl_stats_cache[name_lower] = p
            fpl_stats_cache[full_lower] = p
            # Index by last name
            last_name = p["full_name"].split()[-1].lower() if p["full_name"] else ""
            if last_name and len(last_name) >= 4:
                fpl_stats_cache[last_name] = p
            # Index by first + last (skipping middle names)
            parts = p["full_name"].split()
            if len(parts) >= 2:
                first_last = f"{parts[0]} {parts[-1]}".lower()
                fpl_stats_cache[first_last] = p
            # Store photo codes
            if p.get("photo_code"):
                fpl_player_codes[name_lower] = p["photo_code"]
                fpl_player_codes[full_lower] = p["photo_code"]
                if last_name and len(last_name) >= 4:
                    fpl_player_codes[last_name] = p["photo_code"]
            # Build per-team player sets for filtering
            team_name = p.get("team", "")
            if team_name:
                if team_name not in fpl_players_by_team:
                    fpl_players_by_team[team_name] = set()
                fpl_players_by_team[team_name].add(name_lower)
                fpl_players_by_team[team_name].add(full_lower)
                if last_name and len(last_name) >= 4:
                    fpl_players_by_team[team_name].add(last_name)

        # Build team ID lookup for badges — use "code" not "id"!
        # FPL "id" is sequential 1-20 (alphabetical, changes each season).
        # FPL "code" is the historical club identifier used by the PL CDN for badges.
        # e.g. Arsenal: id=1 but code=3 → badge URL must be t3@x2.png, not t1.
        teams = client.get_teams()
        for t in teams:
            fpl_team_ids[t["name"].lower()] = t["code"]
        print(f"Loaded FPL stats for {len(all_stats)} players, {len(teams)} teams")
    except Exception as e:
        print(f"WARNING: Failed to load FPL stats: {e}")

    # Initialize odds client
    odds_client = OddsClient()
    matches = odds_client.get_upcoming_matches()
    print(f"Loaded odds for {len(matches)} upcoming matches")

    # Filter inference_df: validate against FPL and correct team assignments.
    # FPL updates weekly (gameweek-level) so it reflects January transfers etc.
    # football-data.org squads can be stale.
    if inference_df is not None and fpl_stats_cache:
        pre_count = len(inference_df)
        valid_rows = []
        for idx, row in inference_df.iterrows():
            fpl = get_fpl_stats_for_player(row["name"], team_hint=row.get("team", ""))
            if fpl:
                valid_rows.append(idx)
                # Use FPL team assignment — it's updated weekly and reflects transfers
                fpl_team = fpl.get("team", "")
                if fpl_team:
                    inference_df.at[idx, "team"] = fpl_team
                    inference_df.at[idx, "player_team"] = fpl_team
        inference_df = inference_df.loc[valid_rows].copy()
        # Deduplicate: keep highest-risk entry per name+team
        inference_df = inference_df.sort_values("ensemble_prob", ascending=False)
        inference_df = inference_df.drop_duplicates(subset=["name", "team"], keep="first")
        print(f"Filtered players: {pre_count} -> {len(inference_df)} (matched to FPL squads)")

    # Enrich inference_df with scraped injury history from player_history.pkl
    if inference_df is not None:
        try:
            import pandas as pd
            history_path = os.path.join(PROJECT_ROOT, "models", "player_history.pkl")
            if os.path.exists(history_path):
                ph = pd.read_pickle(history_path)
                # Merge enriched injury columns into inference_df
                merge_cols = ["name"]
                enrich_cols = ["total_days_lost", "days_since_last_injury", "last_injury_date"]
                available = [c for c in enrich_cols if c in ph.columns]
                if available:
                    ph_subset = ph[["name"] + available].drop_duplicates(subset=["name"], keep="last")
                    # Also update player_injury_count and player_avg_severity from fresh scrape
                    for col in ["player_injury_count", "player_avg_severity", "player_worst_injury", "is_injury_prone"]:
                        if col in ph.columns:
                            available.append(col)
                    ph_subset = ph[["name"] + available].drop_duplicates(subset=["name"], keep="last")

                    pre_cols = set(inference_df.columns)
                    # Drop old columns that will be replaced
                    for col in available:
                        if col in inference_df.columns:
                            inference_df = inference_df.drop(columns=[col])
                    inference_df = inference_df.merge(ph_subset, on="name", how="left")
                    # Fill NaN for players not in history
                    inference_df["player_injury_count"] = inference_df["player_injury_count"].fillna(0)
                    inference_df["player_avg_severity"] = inference_df["player_avg_severity"].fillna(0)
                    inference_df["player_worst_injury"] = inference_df["player_worst_injury"].fillna(0)
                    inference_df["is_injury_prone"] = inference_df["is_injury_prone"].fillna(0)
                    inference_df["total_days_lost"] = inference_df["total_days_lost"].fillna(0)
                    inference_df["days_since_last_injury"] = inference_df["days_since_last_injury"].fillna(365)
                    has_data = (inference_df["player_injury_count"] > 0).sum()
                    print(f"Enriched injury history: {has_data}/{len(inference_df)} players have injury records")
        except Exception as e:
            print(f"WARNING: Failed to enrich injury history: {e}")

    # Re-assign archetypes: KMeans for players with per-injury detail, rule-based fallback
    if inference_df is not None:
        inference_df = assign_hybrid_archetypes(inference_df)
        archetype_counts = inference_df["archetype"].value_counts().to_dict()
        print(f"Hybrid archetypes: {archetype_counts}")


def assign_rule_based_archetypes(df):
    """Assign archetypes using rule-based logic from scraped injury summary stats.

    Order matters — first match wins. Designed to address:
    - Players with injuries incorrectly showing as "Clean Record"
    - Players with 2 injuries + high severity but >1yr gap misclassified as "Injury Prone"
    """
    import math

    def _safe_int(val, default=0):
        try:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return default
            return int(val)
        except (ValueError, TypeError):
            return default

    def _safe_float(val, default=0.0):
        try:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return default
            return float(val)
        except (ValueError, TypeError):
            return default

    def _classify(row):
        count = _safe_int(row.get("player_injury_count", 0))
        avg_sev = _safe_float(row.get("player_avg_severity", 0))
        worst = _safe_float(row.get("player_worst_injury", 0))
        days_since = _safe_float(row.get("days_since_last_injury", 9999), 9999)

        if count == 0:
            return "Clean Record"
        # Fragile: serious injuries (high severity + at least a pattern)
        if count >= 3 and avg_sev >= 40 and worst >= 60:
            return "Fragile"
        # Injury Prone: many injuries AND they're not just minor knocks
        if count >= 5 and avg_sev >= 20:
            return "Injury Prone"
        # Recurring Issues: frequent recent injuries
        if count >= 3 and days_since < 180:
            return "Recurring Issues"
        # Durable: few injuries, long time since last
        if count <= 2 and days_since > 365:
            return "Durable"
        # Durable: many minor knocks (avg < 20 days) with no recent issues
        if avg_sev < 20 and days_since > 180:
            return "Durable"
        return "Moderate Risk"

    df["archetype"] = df.apply(_classify, axis=1)
    return df


def assign_hybrid_archetypes(df):
    """Hybrid archetype assignment: HDBSCAN primary, KMeans fallback, rule-based last resort.

    1. Build archetype features from player_injuries_detail.pkl (per-injury records)
    2. HDBSCAN to find dense clusters — labels noise as -1
    3. KMeans on noise points to assign them to nearest cluster
    4. Label clusters by inspecting centroids (avg_severity, total_injuries, etc.)
    5. Rule-based fallback for players not in detail pkl (no per-injury data)
    """
    import os
    import pandas as pd
    import hdbscan
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from src.feature_engineering.archetype import build_player_archetype_features

    detail_path = os.path.join(os.path.dirname(__file__), "..", "models", "player_injuries_detail.pkl")
    detail_path = os.path.abspath(detail_path)

    if not os.path.exists(detail_path):
        print("No player_injuries_detail.pkl found — falling back to rule-based archetypes")
        return assign_rule_based_archetypes(df)

    try:
        detail_df = pd.read_pickle(detail_path)
        # Fix StringDtype columns
        for col in detail_df.columns:
            if "String" in str(detail_df[col].dtype) or str(detail_df[col].dtype) in ("string", "string[python]"):
                detail_df[col] = detail_df[col].astype(object)

        # Build per-player feature matrix from per-injury records
        feat_df = build_player_archetype_features(detail_df)
        print(f"Built archetype features for {len(feat_df)} players")

        # Use core features only (avoids curse of dimensionality with 44 sparse features)
        core_cols = [
            "total_injuries", "avg_severity", "median_severity", "max_severity",
            "high_severity_rate", "avg_days_between_injuries", "reinjury_rate",
            "body_area_entropy", "severity_cv", "severity_trend",
        ]
        core_cols = [c for c in core_cols if c in feat_df.columns]
        X = feat_df[core_cols].fillna(0).values

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- HDBSCAN ---
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3, cluster_selection_method="eom")
        labels = clusterer.fit_predict(X_scaled)
        n_clusters = len(set(labels) - {-1})
        noise_count = (labels == -1).sum()
        print(f"HDBSCAN: {n_clusters} clusters, {noise_count} noise points out of {len(labels)}")

        # --- KMeans fallback for noise points ---
        if n_clusters < 2:
            # HDBSCAN couldn't find structure — use KMeans entirely
            print("HDBSCAN found <2 clusters — using KMeans with 5 clusters")
            km = KMeans(n_clusters=5, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            n_clusters = 5
        elif noise_count > 0:
            # Assign noise points to nearest HDBSCAN cluster via KMeans
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            km_labels = km.fit_predict(X_scaled)
            for i in range(len(labels)):
                if labels[i] == -1:
                    labels[i] = km_labels[i]
            print(f"KMeans assigned {noise_count} noise points to clusters")

        feat_df["cluster"] = labels

        # --- Label clusters by ranking on key characteristics ---
        profiles = feat_df.groupby("cluster").agg({
            "total_injuries": "mean",
            "avg_severity": "mean",
            "max_severity": "mean",
            "high_severity_rate": "mean",
            "reinjury_rate": "mean",
            "severity_cv": "mean",
        })

        # Score each cluster for each archetype and assign greedily
        archetype_candidates = [
            ("Fragile", lambda r: r["avg_severity"] * 2 + r["max_severity"] - r["total_injuries"] * 3),
            ("Injury Prone", lambda r: r["total_injuries"] * 5 + r["high_severity_rate"] * 20),
            ("Unpredictable", lambda r: r["severity_cv"] * 30 + r["total_injuries"] * 2),
            ("Recurring Issues", lambda r: r["reinjury_rate"] * 100 + r["total_injuries"] * 3),
            ("Durable", lambda r: -(r["avg_severity"] * 2 + r["total_injuries"] * 5)),
        ]

        cluster_to_archetype = {}
        assigned_clusters = set()
        for archetype_name, score_fn in archetype_candidates:
            best_cid, best_score = None, float("-inf")
            for cid, row in profiles.iterrows():
                if cid in assigned_clusters:
                    continue
                score = score_fn(row)
                if score > best_score:
                    best_score = score
                    best_cid = cid
            if best_cid is not None:
                cluster_to_archetype[best_cid] = archetype_name
                assigned_clusters.add(best_cid)

        # Any remaining clusters get "Moderate Risk"
        for cid in profiles.index:
            if cid not in cluster_to_archetype:
                cluster_to_archetype[cid] = "Moderate Risk"

        feat_df["archetype"] = feat_df["cluster"].map(cluster_to_archetype)

        # Log cluster assignments
        for cid in sorted(cluster_to_archetype.keys()):
            row = profiles.loc[cid]
            count = (feat_df["cluster"] == cid).sum()
            print(f"  Cluster {cid} → {cluster_to_archetype[cid]} ({count} players, "
                  f"avg_inj={row['total_injuries']:.1f}, avg_sev={row['avg_severity']:.1f})")

        # Map back to inference_df by player name
        archetype_lookup = dict(zip(feat_df["name"], feat_df["archetype"]))
        matched = 0
        for idx, row in df.iterrows():
            name = row.get("name", "")
            if name in archetype_lookup:
                df.at[idx, "archetype"] = archetype_lookup[name]
                matched += 1

        # Rule-based fallback for unmatched players (not in detail pkl)
        unmatched_mask = ~df["name"].isin(archetype_lookup)
        unmatched_count = unmatched_mask.sum()
        if unmatched_count > 0:
            unmatched_df = assign_rule_based_archetypes(df.loc[unmatched_mask].copy())
            df.loc[unmatched_mask, "archetype"] = unmatched_df["archetype"].values

        # Recency-based overrides: clustering ignores time since last injury
        overrides = 0
        for idx, row in df.iterrows():
            days_since = float(row.get("days_since_last_injury", 9999))
            prev_inj = int(row.get("previous_injuries", row.get("player_injury_count", 0)))
            total_days = float(row.get("total_days_lost", 0))
            avg_sev = total_days / prev_inj if prev_inj > 0 else 0
            current = df.at[idx, "archetype"]

            # Recently injured/returning → Currently Vulnerable
            if days_since < 60 and current not in ("Fragile",):
                df.at[idx, "archetype"] = "Currently Vulnerable"
                overrides += 1
            # Long injury-free + low severity → Durable (not Injury Prone)
            elif days_since > 365 and avg_sev < 25 and current in ("Injury Prone", "Recurring Issues"):
                df.at[idx, "archetype"] = "Durable"
                overrides += 1

        counts = df["archetype"].value_counts().to_dict()
        print(f"Hybrid archetypes: {matched} clustered, {unmatched_count} rule-based, {overrides} recency overrides")
        print(f"  Distribution: {counts}")
        return df

    except Exception as e:
        print(f"WARNING: Hybrid clustering failed ({e}) — falling back to rule-based")
        import traceback
        traceback.print_exc()
        return assign_rule_based_archetypes(df)


# ============================================================
# Response Models
# ============================================================

class PlayerSummary(BaseModel):
    name: str
    team: str
    position: str
    risk_level: str
    risk_probability: float
    archetype: str
    minutes_played: int = 0
    is_starter: bool = False
    player_image_url: Optional[str] = None


class RiskFactors(BaseModel):
    previous_injuries: int
    total_days_lost: int
    days_since_last_injury: int
    avg_days_per_injury: float


class ModelPredictions(BaseModel):
    ensemble: float
    lgb: float
    xgb: float
    catboost: float


class ImpliedOdds(BaseModel):
    """Betting odds derived from injury probability."""
    american: str
    decimal: float
    fractional: str
    implied_prob: float


class ScoringOdds(BaseModel):
    """Odds for a player to score in next match."""
    score_probability: float
    involvement_probability: float
    goals_per_90: float
    assists_per_90: float
    american: str
    decimal: float
    fractional: str
    availability_factor: float


class FPLValue(BaseModel):
    """FPL value assessment for a player."""
    tier: str
    tier_emoji: str
    verdict: str
    position_insight: Optional[str] = None
    adjusted_value: float
    goals_per_90: float
    assists_per_90: float
    price: float
    risk_factor: float


class CleanSheetOdds(BaseModel):
    """Clean sheet odds for defenders/goalkeepers."""
    clean_sheet_probability: float
    goals_conceded_per_game: float
    american: str
    decimal: float
    availability_factor: float


class NextFixture(BaseModel):
    """Player's next match info with betting context."""
    opponent: str
    is_home: bool
    match_time: Optional[str] = None
    clean_sheet_odds: Optional[str] = None
    win_probability: Optional[float] = None
    fixture_insight: Optional[str] = None


class YaraResponse(BaseModel):
    """Yara's opinionated analysis comparing model vs market odds."""
    response_text: str
    fpl_tip: str
    market_probability: Optional[float] = None
    yara_probability: float
    market_odds_decimal: Optional[float] = None
    bookmaker: Optional[str] = None


class BookmakerOddsLine(BaseModel):
    """Single bookmaker line for scoring/clean-sheet market."""
    bookmaker: str
    decimal_odds: float
    implied_probability: float
    source: Optional[str] = None


class BookmakerConsensus(BaseModel):
    """Normalized three-bookie market summary for narrative explainability."""
    market_type: str  # "score" | "clean_sheet"
    market_label: str
    average_decimal: float
    average_probability: float
    summary_text: str
    market_line: str
    lines: List[BookmakerOddsLine]


class LabDriver(BaseModel):
    """A single key driver in Yara's Lab Notes."""
    name: str
    value: Any
    impact: str  # "risk_increasing", "protective", "neutral"
    explanation: str


class TechnicalDetails(BaseModel):
    """Technical details for the 'for builders' section."""
    model_agreement: float
    methodology: str
    feature_highlights: List[Dict[str, Any]]


class LabNotes(BaseModel):
    """Yara's Lab Notes — explainability for the risk score."""
    summary: str
    key_drivers: List[LabDriver]
    technical: TechnicalDetails


class PlayerRisk(BaseModel):
    name: str
    team: str
    position: str
    age: int
    risk_level: str
    risk_probability: float
    archetype: str
    archetype_description: str
    factors: RiskFactors
    model_predictions: ModelPredictions
    recommendations: List[str]
    story: str
    implied_odds: ImpliedOdds
    last_injury_date: Optional[str]
    fpl_insight: Optional[str] = None
    scoring_odds: Optional[ScoringOdds] = None
    fpl_value: Optional[FPLValue] = None
    clean_sheet_odds: Optional[CleanSheetOdds] = None
    next_fixture: Optional[NextFixture] = None
    bookmaker_consensus: Optional[BookmakerConsensus] = None
    yara_response: Optional[YaraResponse] = None
    lab_notes: Optional[LabNotes] = None
    risk_percentile: Optional[float] = None
    player_image_url: Optional[str] = None
    team_badge_url: Optional[str] = None


class TeamOverview(BaseModel):
    team: str
    total_players: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_risk: float
    players: List[PlayerSummary]
    team_badge_url: Optional[str] = None
    next_fixture: Optional[Dict[str, Any]] = None


class HealthCheck(BaseModel):
    status: str
    models_loaded: bool
    player_count: int


class LeagueStanding(BaseModel):
    id: int
    name: str
    short_name: str
    position: int
    played: int
    wins: int
    draws: int
    losses: int
    points: int
    form: Optional[str]
    strength: int


class GameweekSummary(BaseModel):
    gameweek: int
    name: str
    deadline: Optional[str]
    is_current: bool
    is_next: bool
    fixture_count: int
    double_gameweek_teams: List[str]
    featured_matches: List[str]


class FPLInsights(BaseModel):
    current_gameweek: Optional[int]
    standings: List[LeagueStanding]
    upcoming_gameweeks: List[GameweekSummary]
    has_double_gameweek: bool


class TeamStanding(BaseModel):
    name: str
    short_name: str
    position: Optional[int] = None
    points: int
    played: int
    form: Optional[str] = None
    distance_from_top: Optional[int] = None
    distance_from_safety: Optional[int] = None


class StandingsSummary(BaseModel):
    leader: TeamStanding
    second: TeamStanding
    gap_to_second: int
    safety_points: int = 0
    selected_team: Optional[TeamStanding] = None


# ============================================================
# Helper Functions
# ============================================================

ARCHETYPE_DESCRIPTIONS = {
    "Durable": "Resilient player with limited injury history and good recovery",
    "Fragile": "When injured, tends to be serious — requires extended recovery periods",
    "Injury Prone": "Frequently picks up injuries, though typically not severe",
    "Recurring Issues": "Recent pattern of repeated injuries — needs targeted management",
    "Moderate Risk": "Average injury profile with no major red flags",
    "Clean Record": "No significant injury history on record",
    "Currently Vulnerable": "Recently returned from injury, elevated re-injury risk",
    "Recurring": "Regular minor injuries but quick recoveries",
    "Unpredictable": "Varied injury patterns that are difficult to predict",
}

# Team name aliases for FPL badge lookup
# Maps all known team name variants → FPL full name (lowercase)
# FPL names: Arsenal, Aston Villa, Bournemouth, Brentford, Brighton,
#   Chelsea, Crystal Palace, Everton, Fulham, Ipswich, Leicester,
#   Liverpool, Man City, Man Utd, Newcastle, Nott'm Forest,
#   Southampton, Spurs, West Ham, Wolves
TEAM_BADGE_ALIASES = {
    # Arsenal
    "arsenal": "arsenal", "arsenal fc": "arsenal",
    # Aston Villa
    "aston villa": "aston villa", "villa": "aston villa",
    # Bournemouth
    "bournemouth": "bournemouth", "afc bournemouth": "bournemouth",
    # Brentford
    "brentford": "brentford", "brentford fc": "brentford",
    # Brighton
    "brighton": "brighton", "brighton hove": "brighton",
    "brighton & hove albion": "brighton", "brighton and hove albion": "brighton",
    "brighton & hove": "brighton",
    # Chelsea
    "chelsea": "chelsea", "chelsea fc": "chelsea",
    # Crystal Palace
    "crystal palace": "crystal palace", "palace": "crystal palace",
    # Everton
    "everton": "everton", "everton fc": "everton",
    # Fulham
    "fulham": "fulham", "fulham fc": "fulham",
    # Ipswich
    "ipswich": "ipswich", "ipswich town": "ipswich",
    # Leicester
    "leicester": "leicester", "leicester city": "leicester",
    # Liverpool
    "liverpool": "liverpool", "liverpool fc": "liverpool",
    # Man City
    "man city": "man city", "manchester city": "man city",
    "manchester city fc": "man city", "mcfc": "man city",
    # Man Utd
    "man utd": "man utd", "man united": "man utd",
    "manchester united": "man utd", "manchester united fc": "man utd",
    "mufc": "man utd",
    # Newcastle
    "newcastle": "newcastle", "newcastle united": "newcastle",
    "newcastle utd": "newcastle",
    # Nott'm Forest
    "nott'm forest": "nott'm forest", "nottm forest": "nott'm forest",
    "nottingham forest": "nott'm forest", "nottingham": "nott'm forest",
    # Southampton
    "southampton": "southampton", "southampton fc": "southampton",
    # Spurs
    "spurs": "spurs", "tottenham": "spurs",
    "tottenham hotspur": "spurs", "tottenham hotspurs": "spurs",
    # West Ham
    "west ham": "west ham", "west ham united": "west ham",
    # Wolves
    "wolves": "wolves", "wolverhampton": "wolves",
    "wolverhampton wanderers": "wolves",
    # Promoted / other
    "leeds": "leeds", "leeds united": "leeds",
    "sunderland": "sunderland", "sunderland afc": "sunderland",
    "burnley": "burnley", "burnley fc": "burnley",
    "luton": "luton", "luton town": "luton",
    "sheffield united": "sheffield utd", "sheffield utd": "sheffield utd",
}


def get_risk_level(prob: float, row=None) -> str:
    """Classify 2-week injury risk using percentile rank from model output.

    Uses percentile-based thresholds from the actual data distribution so that
    ~25% of players are High, ~45% Medium, ~30% Low. The model now incorporates
    injury history features (player_injury_count, player_avg_severity, etc.)
    so no manual overrides are needed.
    """
    if inference_df is not None and "ensemble_prob" in inference_df.columns:
        p75 = inference_df["ensemble_prob"].quantile(0.75)
        p30 = inference_df["ensemble_prob"].quantile(0.30)
        if prob >= p75:
            return "High"
        elif prob >= p30:
            return "Medium"
        else:
            return "Low"
    else:
        # Fallback absolute thresholds
        if prob >= 0.30:
            return "High"
        elif prob >= 0.15:
            return "Medium"
        else:
            return "Low"


def normalize_risk_score(prob: float) -> float:
    """Convert raw model probability to a 0-100 score based on percentile rank.

    The raw ensemble_prob is not calibrated (range ~0.04-0.92, median ~0.77),
    so showing it directly confuses users. Instead, show where the player
    ranks relative to all others: 50 = average risk, 90 = top 10%.
    """
    if inference_df is not None and "ensemble_prob" in inference_df.columns:
        percentile = float((inference_df["ensemble_prob"] <= prob).mean())
        return round(percentile * 100, 1)
    # Fallback: min-max normalize assuming typical range
    return round(max(0, min(100, (prob - 0.04) / (0.92 - 0.04) * 100)), 1)


def get_team_badge_url(team_name: str) -> Optional[str]:
    """Get Premier League badge URL for a team."""
    team_lower = team_name.lower().strip()
    search = TEAM_BADGE_ALIASES.get(team_lower, team_lower)

    # Exact match
    if search in fpl_team_ids:
        return f"https://resources.premierleague.com/premierleague/badges/50/t{fpl_team_ids[search]}@x2.png"

    # Substring match fallback — only against names >= 5 chars to avoid
    # false positives from short codes like "che", "ars", "lei"
    for key, tid in fpl_team_ids.items():
        if len(key) < 5:
            continue
        if search in key or key in search:
            return f"https://resources.premierleague.com/premierleague/badges/50/t{fpl_team_ids[key]}@x2.png"
    return None


def get_player_image_url(player_name: str) -> Optional[str]:
    """Get Premier League player photo URL."""
    name_lower = player_name.lower()
    # Try exact match
    if name_lower in fpl_player_codes:
        code = fpl_player_codes[name_lower]
        return f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png"
    # Try partial match on significant name parts
    parts = [p for p in name_lower.split() if len(p) >= 4]
    for part in parts:
        for key, code in fpl_player_codes.items():
            if part in key:
                return f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png"
    return None


def calculate_implied_odds(prob: float) -> ImpliedOdds:
    """Convert probability to betting odds formats."""
    prob = max(0.01, min(0.99, prob))
    decimal_odds = round(1 / prob, 2)
    if prob >= 0.5:
        american = int(-100 * prob / (1 - prob))
        american_str = str(american)
    else:
        american = int(100 * (1 - prob) / prob)
        american_str = f"+{american}"
    if decimal_odds >= 2:
        numerator = int(round((decimal_odds - 1) * 1))
        fractional = f"{numerator}/1"
    else:
        denominator = int(round(1 / (decimal_odds - 1)))
        fractional = f"1/{denominator}"
    return ImpliedOdds(
        american=american_str, decimal=decimal_odds,
        fractional=fractional, implied_prob=round(prob, 3),
    )


def get_fpl_stats_for_player(name: str, team_hint: str = None) -> Optional[Dict]:
    """Look up FPL stats for a player by name.

    Args:
        name: Player name to search for
        team_hint: Optional team name from inference_df to validate matches
    """
    if not fpl_stats_cache:
        return None
    name_lower = name.lower()

    # Exact match (most reliable)
    if name_lower in fpl_stats_cache:
        return fpl_stats_cache[name_lower]

    # Full name containment (e.g. "Bukayo Saka" in "Bukayo Saka" or vice versa)
    candidates = []
    for key, stats in fpl_stats_cache.items():
        if name_lower in key or key in name_lower:
            candidates.append(stats)

    # If team_hint provided, prefer candidates matching the team
    if candidates and team_hint:
        team_lower = team_hint.lower()
        team_matches = [s for s in candidates if _teams_match(s.get("team", ""), team_lower)]
        if team_matches:
            return team_matches[0]

    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return candidates[0]

    # Last resort: match by last name only (must be unique and long enough)
    parts = name_lower.split()
    if parts:
        last_name = parts[-1]
        if len(last_name) >= 5:  # Only match long last names to avoid false positives
            last_name_matches = []
            for key, stats in fpl_stats_cache.items():
                if key.endswith(last_name) or last_name == key.split()[-1] if " " in key else last_name == key:
                    last_name_matches.append(stats)
            if len(last_name_matches) == 1:
                return last_name_matches[0]
            # If multiple matches, use team hint
            if last_name_matches and team_hint:
                team_lower = team_hint.lower()
                team_filtered = [s for s in last_name_matches if _teams_match(s.get("team", ""), team_lower)]
                if len(team_filtered) == 1:
                    return team_filtered[0]
    return None


def _teams_match(fpl_team: str, team_hint: str) -> bool:
    """Check if FPL team name matches a team hint (fuzzy)."""
    fpl_lower = fpl_team.lower()
    hint_lower = team_hint.lower()
    # Direct match
    if fpl_lower == hint_lower:
        return True
    # Check display name mapping
    display = _fpl_team_to_df_team(fpl_team).lower()
    if display == hint_lower or hint_lower in display or display in hint_lower:
        return True
    # Short name containment (e.g. "wolves" in "wolverhampton")
    if len(fpl_lower) >= 4 and (fpl_lower in hint_lower or hint_lower in fpl_lower):
        return True
    return False


def get_next_fixture_for_team(team_name: str, injury_prob: float) -> Optional[Dict]:
    """Get next fixture info with betting odds for a team.

    Uses FPL API for fixture data (always has real kickoff times).
    Enriches with external odds if ODDS_API_KEY is available (not mock).
    """
    # Try external odds API only if we have a real API key (not mock data)
    odds_data = None
    if odds_client and odds_client.api_key:
        cs_data = odds_client.get_clean_sheet_odds(team_name)
        if cs_data:
            fixture_insight = get_clean_sheet_insight(team_name, injury_prob)
            odds_data = {
                "clean_sheet_odds": cs_data["american"],
                "win_probability": cs_data["win_probability"],
                "fixture_insight": fixture_insight,
            }
    # Add 1X2 bookie lines (works with live key and mock fallback)
    if odds_client:
        moneyline_data = odds_client.get_team_moneyline_1x2(team_name)
        if moneyline_data and moneyline_data.get("books"):
            if odds_data is None:
                odds_data = {}
            odds_data["moneyline_1x2"] = moneyline_data["books"]

    # Use FPL API fixtures (real kickoff times)
    try:
        client = FPLClient()
        gw = client.get_current_gameweek()
        if not gw:
            return None
        teams_list = client.get_teams()
        team_map = {t["id"]: t["name"] for t in teams_list}

        # Find team ID by matching name
        team_lower = team_name.lower()
        team_id = None
        for t in teams_list:
            if (team_lower in t["name"].lower() or t["name"].lower() in team_lower
                    or team_lower in t.get("short_name", "").lower()):
                team_id = t["id"]
                break
        if not team_id:
            search = TEAM_BADGE_ALIASES.get(team_lower, team_lower)
            for t in teams_list:
                if search in t["name"].lower() or t["name"].lower() in search:
                    team_id = t["id"]
                    break
        if not team_id:
            return None

        # Try current GW first, then next GW if all current fixtures are finished
        gw_ids_to_try = [gw["id"]]
        # Find next GW
        data = client.get_bootstrap()
        for event in data.get("events", []):
            if event.get("is_next") and event["id"] != gw["id"]:
                gw_ids_to_try.append(event["id"])
                break
            if event["id"] == gw["id"] + 1:
                gw_ids_to_try.append(event["id"])
                break

        for gw_id in gw_ids_to_try:
            fixtures = client.get_fixtures(gw_id)
            for f in fixtures:
                # Skip finished matches — only show upcoming/live
                if f.get("finished") or f.get("finished_provisional"):
                    continue

                fixture_result = None
                if f.get("team_h") == team_id:
                    fixture_result = {
                        "opponent": team_map.get(f["team_a"], "Unknown"),
                        "is_home": True,
                        "match_time": f.get("kickoff_time"),
                        "clean_sheet_odds": None,
                        "win_probability": None,
                        "fixture_insight": None,
                    }
                elif f.get("team_a") == team_id:
                    fixture_result = {
                        "opponent": team_map.get(f["team_h"], "Unknown"),
                        "is_home": False,
                        "match_time": f.get("kickoff_time"),
                        "clean_sheet_odds": None,
                        "win_probability": None,
                        "fixture_insight": None,
                    }
                if fixture_result:
                    # Merge external odds data if available
                    if odds_data:
                        fixture_result.update(odds_data)
                    return fixture_result
    except Exception as e:
        logger.warning(f"Failed to get FPL fixture for {team_name}: {e}")

    return None


def _is_defensive_position(position: str) -> bool:
    pos_lower = (position or "").lower()
    return any(p in pos_lower for p in ["def", "gk", "goalkeeper", "back"])


def _canonical_bookmaker_name(raw_name: str) -> Optional[str]:
    aliases = {
        "SkyBet": ["sky bet", "skybet"],
        "Paddy Power": ["paddy power", "paddypower"],
        "Betway": ["betway"],
    }
    haystack = (raw_name or "").lower()
    for canonical, keys in aliases.items():
        if any(k in haystack for k in keys):
            return canonical
    return None


def build_bookmaker_consensus(
    player_name: str,
    position: str,
    scoring_odds_data: Optional[Dict],
    clean_sheet_data: Optional[Dict],
    scorer_market_snapshot: Optional[Dict],
) -> Optional[Dict]:
    """Build direct-only market lines with preferred-first book selection."""
    defensive_market = _is_defensive_position(position) and clean_sheet_data is not None
    if defensive_market:
        # No direct clean-sheet market integration yet (only model estimate exists).
        return None

    if not scorer_market_snapshot:
        return None

    market_type = "score"
    market_label = "to score"
    target_books = ["SkyBet", "Paddy Power", "Betway"]
    preferred_map: Dict[str, Dict[str, Any]] = {}
    fallback_lines: List[Dict[str, Any]] = []
    seen_books = set()

    for raw_line in scorer_market_snapshot.get("lines", []):
        decimal = raw_line.get("decimal_odds")
        if not decimal or decimal <= 1:
            continue
        raw_book = raw_line.get("bookmaker", "Unknown")
        canonical = _canonical_bookmaker_name(raw_book)
        book_name = canonical or raw_book
        if book_name in seen_books:
            continue
        seen_books.add(book_name)

        normalized_line = {
            "bookmaker": book_name,
            "decimal_odds": round(float(decimal), 2),
            "implied_probability": round(1 / float(decimal), 3),
            "source": raw_line.get("source", "Live Market"),
        }
        if canonical in target_books:
            preferred_map[canonical] = normalized_line
        else:
            fallback_lines.append(normalized_line)

    lines = [preferred_map[book] for book in target_books if book in preferred_map]
    if len(lines) < 3:
        lines.extend(fallback_lines[: max(0, 3 - len(lines))])
    if not lines:
        return None

    average_decimal = round(sum(line["decimal_odds"] for line in lines) / len(lines), 2)
    average_probability = round(sum(line["implied_probability"] for line in lines) / len(lines), 3)
    probability_pct = round(average_probability * 100, 1)
    first_name = player_name.split()[0] if player_name else "Player"
    opponent = scorer_market_snapshot.get("opponent") if scorer_market_snapshot else None
    is_home = scorer_market_snapshot.get("is_home") if scorer_market_snapshot else None

    summary_text = (
        f"Bookies estimate {first_name}'s odds {market_label} is at an average of "
        f"{average_decimal:.2f}. A {probability_pct}% probability."
    )
    market_line = (
        f"{first_name} {'anytime scorer' if market_type == 'score' else 'clean sheet'} "
        f"- {average_decimal:.2f} (≈{round(average_probability * 100)}%)"
    )

    return {
        "market_type": market_type,
        "market_label": market_label,
        "average_decimal": average_decimal,
        "average_probability": average_probability,
        "summary_text": summary_text,
        "market_line": market_line,
        "lines": lines,
        "opponent": opponent,
        "is_home": is_home,
    }


def enhance_yara_response(
    base_response: Optional[Dict],
    player_row: Dict[str, Any],
    next_fixture_data: Optional[Dict],
    market_consensus: Optional[Dict],
    scoring_odds_data: Optional[Dict] = None,
    clean_sheet_data: Optional[Dict] = None,
) -> Optional[Dict]:
    """Blend model narrative with fixture context and bookmaker consensus."""
    if not base_response and not market_consensus:
        return None

    name = player_row.get("name", "This player")
    first_name = name.split()[0] if name else "This player"
    team = player_row.get("team", "their team")
    injury_prob = float(player_row.get("ensemble_prob", 0.5))
    goals_per_90 = float(player_row.get("goals_per_90", 0) or 0)
    assists_per_90 = float(player_row.get("assists_per_90", 0) or 0)

    opponent = None
    is_home = None
    if next_fixture_data:
        opponent = next_fixture_data.get("opponent")
        is_home = next_fixture_data.get("is_home")

    if market_consensus and not opponent:
        opponent = market_consensus.get("opponent")
        is_home = market_consensus.get("is_home")

    fallback_model_prob = 0.0
    if _is_defensive_position(player_row.get("position", "")) and clean_sheet_data:
        fallback_model_prob = float(clean_sheet_data.get("clean_sheet_probability", 0.0) or 0.0)
    elif scoring_odds_data:
        fallback_model_prob = float(scoring_odds_data.get("score_probability", 0.0) or 0.0)

    yara_probability = (
        base_response.get("yara_probability")
        if base_response
        else (fallback_model_prob or (market_consensus.get("average_probability") if market_consensus else 0.0))
    )
    market_probability = (
        market_consensus.get("average_probability")
        if market_consensus
        else (base_response.get("market_probability") if base_response else None)
    )

    venue_context = ""
    if opponent:
        venue = "host" if is_home else "travel to"
        venue_context = f" {team} {venue} {opponent} next."

    availability_context = "load risk is elevated" if injury_prob >= 0.4 else "availability profile is steady"
    if market_consensus and market_probability is not None:
        response_text = (
            f"I'm projecting {round(float(yara_probability) * 100)}%. "
            f"Bookies average {round(float(market_probability) * 100)}%. "
            f"{first_name} is at {goals_per_90:.2f} goals/90 and {assists_per_90:.2f} assists/90, "
            f"and the injury-adjusted view says {availability_context}.{venue_context}"
        )
    else:
        response_text = (
            f"I'm projecting {round(float(yara_probability) * 100)}%. "
            f"{first_name} is at {goals_per_90:.2f} goals/90 and {assists_per_90:.2f} assists/90; "
            f"{availability_context}.{venue_context}"
        )

    fpl_tip = base_response.get("fpl_tip") if base_response else None
    if not fpl_tip:
        if injury_prob >= 0.45:
            fpl_tip = "Start only with bench cover. Captain only if chasing upside."
        elif yara_probability >= 0.4:
            fpl_tip = "Start if you own. Captain only if fixture upside matches your rank goals."
        else:
            fpl_tip = "Playable, but not a captain priority this week."

    return {
        "response_text": response_text,
        "fpl_tip": fpl_tip,
        "market_probability": market_probability,
        "yara_probability": float(yara_probability),
        "market_odds_decimal": market_consensus.get("average_decimal") if market_consensus else (
            base_response.get("market_odds_decimal") if base_response else None
        ),
        "bookmaker": "Bookies Avg" if market_consensus else (base_response.get("bookmaker") if base_response else None),
    }


def get_personalized_insights(row: dict) -> List[str]:
    """Generate personalized insights using the story generator."""
    risk_factors = generate_risk_factors_list(row)
    insights = []
    main_rec = get_recommendation_text(row)
    if main_rec:
        insights.append(main_rec)
    for factor in risk_factors[:2]:
        if factor["impact"] in ["high_risk", "moderate_risk"]:
            insights.append(f"{factor['factor']}: {factor['description']}")
        elif factor["impact"] == "protective":
            insights.append(f"{factor['factor']}: {factor['description']}")
    return insights[:4]


def player_row_to_risk(row) -> PlayerRisk:
    """Convert a DataFrame row to a PlayerRisk response."""
    prob = row.get("ensemble_prob", row.get("calibrated_prob", 0.5))
    # Use player_injury_count (has data) instead of previous_injuries (all zeros in inference_df)
    prev_injuries = int(row.get("player_injury_count", row.get("previous_injuries", 0)))
    avg_severity = row.get("player_avg_severity", 0)
    total_days = int(prev_injuries * avg_severity) if prev_injuries > 0 else 0
    player_name = row.get("name", "Unknown")
    team = row.get("team", row.get("player_team", "Unknown"))

    # Build enriched row with corrected injury fields + FPL stats
    # inference_df is now enriched at startup with real Transfermarkt data,
    # but story generators read "previous_injuries" so we map it
    enriched_row = dict(row)
    enriched_row["previous_injuries"] = prev_injuries
    # Use total_days_lost from scraped data if available, otherwise estimate
    total_days_from_scrape = row.get("total_days_lost", 0)
    if total_days_from_scrape and int(total_days_from_scrape) > 0:
        total_days = int(total_days_from_scrape)
    enriched_row["total_days_lost"] = total_days
    # Use days_since_last_injury from scraped data (real value, not default 365)
    days_since = int(row.get("days_since_last_injury", 365))
    enriched_row["days_since_last_injury"] = days_since

    # Enrich with FPL stats
    fpl_stats = get_fpl_stats_for_player(player_name)
    if fpl_stats:
        enriched_row.update({
            "goals": fpl_stats.get("goals", 0),
            "assists": fpl_stats.get("assists", 0),
            "goals_per_90": fpl_stats.get("goals_per_90", 0),
            "assists_per_90": fpl_stats.get("assists_per_90", 0),
            "price": fpl_stats.get("price", 0),
            "form": fpl_stats.get("form", 0),
            "minutes": fpl_stats.get("minutes", 0),
        })

    # Compute risk percentile for context (e.g. "higher risk than 92% of players")
    risk_percentile = None
    if inference_df is not None:
        risk_percentile = round(float((inference_df["ensemble_prob"] <= prob).mean()), 3)
    enriched_row["risk_percentile"] = risk_percentile

    # Use enriched_row for ALL generators so they see correct injury history
    story = generate_player_story(enriched_row)
    scoring_odds_data = calculate_scoring_odds(enriched_row)
    fpl_value_data = get_fpl_value_assessment(enriched_row)
    clean_sheet_data = calculate_clean_sheet_odds(enriched_row)
    next_fixture_data = get_next_fixture_for_team(team, prob)

    scorer_market_snapshot = None
    if odds_client:
        scorer_market_snapshot = odds_client.get_anytime_scorer_market_snapshot(team, player_name)

    bookmaker_consensus_data = build_bookmaker_consensus(
        player_name=player_name,
        position=enriched_row.get("position", ""),
        scoring_odds_data=scoring_odds_data,
        clean_sheet_data=clean_sheet_data,
        scorer_market_snapshot=scorer_market_snapshot,
    )

    # Yara's response: compare model projection with consensus market line
    market_odds_for_yara = None
    if bookmaker_consensus_data:
        market_odds_for_yara = {
            "implied_probability": bookmaker_consensus_data["average_probability"],
            "decimal_odds": bookmaker_consensus_data["average_decimal"],
            "bookmaker": "Bookies Avg",
            "opponent": bookmaker_consensus_data.get("opponent"),
            "is_home": bookmaker_consensus_data.get("is_home"),
        }
    elif odds_client:
        market_odds_for_yara = odds_client.get_anytime_scorer_odds(team, player_name)

    yara_data = enhance_yara_response(
        base_response=generate_yara_response(enriched_row, market_odds_for_yara),
        player_row=enriched_row,
        next_fixture_data=next_fixture_data,
        market_consensus=bookmaker_consensus_data,
        scoring_odds_data=scoring_odds_data,
        clean_sheet_data=clean_sheet_data,
    )

    # Yara's Lab Notes: explainability
    lab_notes_data = generate_lab_notes(enriched_row)

    fpl_insight_text = get_fpl_insight(enriched_row)
    if not fpl_insight_text and next_fixture_data:
        fpl_insight_text = next_fixture_data.get("fixture_insight")

    return PlayerRisk(
        name=player_name,
        team=team,
        position=enriched_row.get("position", "Unknown"),
        age=int(enriched_row.get("age", 25)),
        risk_level=get_risk_level(prob, enriched_row),
        risk_probability=round(prob, 3),
        archetype=enriched_row.get("archetype", "Unknown"),
        archetype_description=ARCHETYPE_DESCRIPTIONS.get(
            enriched_row.get("archetype", ""), "Unknown injury profile"
        ),
        factors=RiskFactors(
            previous_injuries=int(prev_injuries),
            total_days_lost=int(total_days),
            days_since_last_injury=days_since,
            avg_days_per_injury=round(total_days / prev_injuries, 1) if prev_injuries > 0 else 0,
        ),
        model_predictions=ModelPredictions(
            ensemble=round(prob, 3),
            lgb=round(enriched_row.get("lgb_prob", prob), 3),
            xgb=round(enriched_row.get("xgb_prob", prob), 3),
            catboost=round(enriched_row.get("catboost_prob", prob), 3),
        ),
        recommendations=get_personalized_insights(enriched_row),
        story=story,
        implied_odds=calculate_implied_odds(prob),
        last_injury_date=str(enriched_row.get("last_injury_date")) if enriched_row.get("last_injury_date") else None,
        fpl_insight=fpl_insight_text,
        scoring_odds=ScoringOdds(**scoring_odds_data) if scoring_odds_data else None,
        fpl_value=FPLValue(**fpl_value_data) if fpl_value_data else None,
        clean_sheet_odds=CleanSheetOdds(**clean_sheet_data) if clean_sheet_data else None,
        next_fixture=NextFixture(**next_fixture_data) if next_fixture_data else None,
        bookmaker_consensus=BookmakerConsensus(**bookmaker_consensus_data) if bookmaker_consensus_data else None,
        yara_response=YaraResponse(**yara_data) if yara_data else None,
        lab_notes=LabNotes(
            summary=lab_notes_data["summary"],
            key_drivers=[LabDriver(**d) for d in lab_notes_data["key_drivers"]],
            technical=TechnicalDetails(**lab_notes_data["technical"]),
        ) if lab_notes_data else None,
        risk_percentile=risk_percentile,
        player_image_url=get_player_image_url(player_name),
        team_badge_url=get_team_badge_url(team),
    )


# ============================================================
# API Endpoints
# ============================================================

@app.get("/api/health", response_model=HealthCheck)
async def health_check():
    """Check API health and model status."""
    return HealthCheck(
        status="ok" if inference_df is not None else "no_models",
        models_loaded=inference_df is not None,
        player_count=len(inference_df) if inference_df is not None else 0,
    )


@app.get("/api/odds/status")
async def odds_status(team: str = "Arsenal", player: str = "Bukayo Saka"):
    """Inspect odds provider mode and direct market availability."""
    if odds_client is None:
        raise HTTPException(status_code=503, detail="Odds client not initialized")

    scorer_snapshot = odds_client.get_anytime_scorer_market_snapshot(team, player)
    team_moneyline = odds_client.get_team_moneyline_1x2(team)

    return {
        "provider_mode": getattr(odds_client, "odds_provider", "auto"),
        "has_odds_api_key": bool(getattr(odds_client, "api_key", "")),
        "has_api_football_key": bool(getattr(odds_client, "api_football_key", "")),
        "team": team,
        "player": player,
        "scorer_lines_count": len((scorer_snapshot or {}).get("lines", [])),
        "scorer_lines_books": [l.get("bookmaker") for l in (scorer_snapshot or {}).get("lines", [])],
        "moneyline_books_count": len((team_moneyline or {}).get("books", [])),
        "moneyline_books": [b.get("bookmaker") for b in (team_moneyline or {}).get("books", [])],
    }


@app.get("/api/players", response_model=List[PlayerSummary])
async def list_players(team: Optional[str] = None, risk_level: Optional[str] = None):
    """List all players with summary info."""
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    df = inference_df.copy()

    if team:
        df = df[df["team"].str.lower() == team.lower()]
    if risk_level:
        df["risk_level"] = df.apply(lambda r: get_risk_level(r.get("ensemble_prob", 0.5), r), axis=1)
        df = df[df["risk_level"].str.lower() == risk_level.lower()]

    df = df.sort_values("ensemble_prob", ascending=False)

    players = []
    for _, row in df.iterrows():
        prob = row.get("ensemble_prob", 0.5)
        name = row.get("name", "Unknown")
        fpl = get_fpl_stats_for_player(name)
        minutes = fpl.get("minutes", 0) if fpl else 0
        players.append(PlayerSummary(
            name=name,
            team=row.get("team", "Unknown"),
            position=row.get("position", "Unknown"),
            risk_level=get_risk_level(prob, row),
            risk_probability=round(prob, 3),
            archetype=row.get("archetype", "Unknown"),
            minutes_played=minutes,
            is_starter=minutes >= 900,
            player_image_url=get_player_image_url(name),
        ))

    return players


@app.get("/api/players/{player_name}/risk", response_model=PlayerRisk)
async def get_player_risk(player_name: str):
    """Get detailed risk prediction for a specific player."""
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    matches = inference_df[
        inference_df["name"].str.lower() == player_name.lower()
    ]
    if matches.empty:
        matches = inference_df[
            inference_df["name"].str.lower().str.contains(player_name.lower())
        ]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

    row = matches.iloc[0].to_dict()
    return player_row_to_risk(row)


@app.get("/api/teams", response_model=List[str])
async def list_teams():
    """List all available teams (filtered to current Premier League)."""
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    our_teams = inference_df["team"].unique().tolist()

    try:
        client = FootballDataClient()
        standings = client.get_standings()
        current_pl_teams = {t["name"].lower() for t in standings}
        current_pl_short = {t["short_name"].lower() for t in standings}

        from src.data_loaders.football_data_api import TEAM_ALIASES

        filtered = []
        for team in our_teams:
            team_lower = team.lower()
            search_term = TEAM_ALIASES.get(team_lower, team_lower)
            in_pl = any(
                search_term in pl_team or pl_team in search_term
                for pl_team in current_pl_teams
            ) or any(
                search_term in short or short in search_term
                for short in current_pl_short
            )
            if in_pl:
                filtered.append(team)

        return sorted(filtered)
    except Exception as e:
        logger.warning(f"Failed to filter teams by current PL: {e}")
        return sorted(our_teams)


@app.get("/api/teams/badges")
async def get_team_badges():
    """Get badge URLs for all teams."""
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    teams = inference_df["team"].unique().tolist()
    badges = {}
    for team in teams:
        url = get_team_badge_url(team)
        if url:
            badges[team] = url
    return badges


@app.get("/api/teams/{team_name}/overview", response_model=TeamOverview)
async def get_team_overview(team_name: str):
    """Get risk overview for an entire team."""
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    team_df = inference_df[
        inference_df["team"].str.lower() == team_name.lower()
    ]
    if team_df.empty:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")

    team_df = team_df.copy()
    team_df["risk_level"] = team_df.apply(lambda r: get_risk_level(r.get("ensemble_prob", 0.5), r), axis=1)

    high_risk = len(team_df[team_df["risk_level"] == "High"])
    medium_risk = len(team_df[team_df["risk_level"] == "Medium"])
    low_risk = len(team_df[team_df["risk_level"] == "Low"])

    team_df = team_df.sort_values("ensemble_prob", ascending=False)

    players = []
    for _, row in team_df.iterrows():
        prob = row.get("ensemble_prob", 0.5)
        name = row.get("name", "Unknown")
        fpl = get_fpl_stats_for_player(name)
        minutes = fpl.get("minutes", 0) if fpl else 0
        players.append(PlayerSummary(
            name=name,
            team=row.get("team", "Unknown"),
            position=row.get("position", "Unknown"),
            risk_level=get_risk_level(prob, row),
            risk_probability=round(prob, 3),
            archetype=row.get("archetype", "Unknown"),
            minutes_played=minutes,
            is_starter=minutes >= 900,
            player_image_url=get_player_image_url(name),
        ))

    actual_team = team_df.iloc[0]["team"]

    # Get next fixture for the team
    next_fixture_data = get_next_fixture_for_team(actual_team, team_df["ensemble_prob"].mean())

    return TeamOverview(
        team=actual_team,
        total_players=len(team_df),
        high_risk_count=high_risk,
        medium_risk_count=medium_risk,
        low_risk_count=low_risk,
        avg_risk=round(team_df["ensemble_prob"].mean(), 3),
        players=players,
        team_badge_url=get_team_badge_url(actual_team),
        next_fixture=next_fixture_data,
    )


@app.get("/api/archetypes")
async def list_archetypes():
    """List all player archetypes with descriptions."""
    return ARCHETYPE_DESCRIPTIONS


@app.get("/api/fpl/insights", response_model=FPLInsights)
async def get_fpl_insights_endpoint():
    """Get FPL insights including standings, fixtures, and double gameweeks."""
    try:
        client = FPLClient()
        standings = client.get_standings()
        upcoming = client.get_upcoming_fixtures_summary(3)
        current_gw = client.get_current_gameweek()

        return FPLInsights(
            current_gameweek=current_gw.get("id") if current_gw else None,
            standings=[LeagueStanding(**s) for s in standings],
            upcoming_gameweeks=[GameweekSummary(**gw) for gw in upcoming],
            has_double_gameweek=any(gw.get("double_gameweek_teams") for gw in upcoming),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FPL API error: {str(e)}")


@app.get("/api/fpl/standings", response_model=List[LeagueStanding])
async def get_league_standings():
    """Get current Premier League standings."""
    try:
        client = FPLClient()
        standings = client.get_standings()
        return [LeagueStanding(**s) for s in standings]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FPL API error: {str(e)}")


@app.get("/api/standings/summary")
async def get_standings_summary_endpoint(team: Optional[str] = None):
    """Get Premier League standings summary."""
    try:
        summary = get_standings_summary(team)
        if not summary:
            raise HTTPException(status_code=503, detail="Standings unavailable")

        return StandingsSummary(
            leader=TeamStanding(**summary["leader"]),
            second=TeamStanding(**summary["second"]),
            gap_to_second=summary["gap_to_second"],
            safety_points=summary.get("safety_points", 0),
            selected_team=TeamStanding(**summary["selected_team"]) if summary.get("selected_team") else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API_STANDINGS_ERROR] {repr(e)}")
        raise HTTPException(status_code=500, detail=f"Standings error: {str(e)}")


@app.get("/api/fpl/double-gameweeks")
async def get_double_gameweeks():
    """Get teams with upcoming double gameweeks."""
    try:
        client = FPLClient()
        teams = {t["id"]: t["name"] for t in client.get_teams()}
        double_gws = client.get_double_gameweek_teams()

        result = {}
        for gw, team_ids in double_gws.items():
            result[str(gw)] = [teams.get(tid, "Unknown") for tid in team_ids]

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FPL API error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
