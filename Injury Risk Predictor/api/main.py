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

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.model_io import load_artifacts

app = FastAPI(
    title="EPL Injury Risk Predictor API",
    description="ML-powered injury risk predictions for Premier League players",
    version="1.0.0",
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
artifacts = None
inference_df = None


@app.on_event("startup")
async def load_models():
    global artifacts, inference_df
    artifacts = load_artifacts()
    if artifacts and "inference_df" in artifacts:
        inference_df = artifacts["inference_df"]
        print(f"Loaded {len(inference_df)} player predictions")
    else:
        print("WARNING: No trained models found. API will return errors.")


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
    last_injury_date: Optional[str]


class TeamOverview(BaseModel):
    team: str
    total_players: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_risk: float
    players: List[PlayerSummary]


class HealthCheck(BaseModel):
    status: str
    models_loaded: bool
    player_count: int


# ============================================================
# Helper Functions
# ============================================================

ARCHETYPE_DESCRIPTIONS = {
    "Durable": "Resilient player who recovers quickly from minor setbacks",
    "Fragile": "Prone to serious injuries requiring extended recovery",
    "Currently Vulnerable": "Recently returned from injury, elevated re-injury risk",
    "Injury Prone": "Frequent injuries with moderate recovery times",
    "Recurring": "Regular minor injuries but quick recoveries",
    "Moderate Risk": "Average injury profile, no major concerns",
    "Clean Record": "Minimal injury history on record",
}


def get_risk_level(prob: float) -> str:
    if prob >= 0.6:
        return "High"
    elif prob >= 0.35:
        return "Medium"
    return "Low"


def get_recommendations(row: dict) -> List[str]:
    recommendations = []
    archetype = row.get("archetype", "")
    days_since = row.get("days_since_last_injury", 365)
    prev_injuries = row.get("previous_injuries", 0)

    if days_since < 60:
        recommendations.append("Gradual return-to-play protocol recommended")
        recommendations.append("Monitor workload carefully for 4-6 weeks")

    if archetype == "Fragile":
        recommendations.append("Consider reduced match minutes")
        recommendations.append("Prioritize recovery between fixtures")
    elif archetype == "Injury Prone":
        recommendations.append("Regular physiotherapy sessions advised")
        recommendations.append("Track training load closely")
    elif archetype == "Currently Vulnerable":
        recommendations.append("Avoid fixture congestion periods")
        recommendations.append("Consider rotation with backup players")
    elif archetype == "Recurring":
        recommendations.append("Address underlying movement patterns")
        recommendations.append("Strength and conditioning focus")

    if prev_injuries >= 10:
        recommendations.append("Comprehensive injury prevention program")

    if not recommendations:
        recommendations.append("Maintain current fitness regime")
        recommendations.append("Standard monitoring protocols")

    return recommendations[:4]  # Max 4 recommendations


def player_row_to_risk(row) -> PlayerRisk:
    """Convert a DataFrame row to a PlayerRisk response."""
    prob = row.get("ensemble_prob", row.get("calibrated_prob", 0.5))
    prev_injuries = row.get("previous_injuries", 0)
    total_days = row.get("total_days_lost", 0)

    return PlayerRisk(
        name=row.get("name", "Unknown"),
        team=row.get("team", row.get("player_team", "Unknown")),
        position=row.get("position", "Unknown"),
        age=int(row.get("age", 25)),
        risk_level=get_risk_level(prob),
        risk_probability=round(prob, 3),
        archetype=row.get("archetype", "Unknown"),
        archetype_description=ARCHETYPE_DESCRIPTIONS.get(
            row.get("archetype", ""), "Unknown injury profile"
        ),
        factors=RiskFactors(
            previous_injuries=int(prev_injuries),
            total_days_lost=int(total_days),
            days_since_last_injury=int(row.get("days_since_last_injury", 365)),
            avg_days_per_injury=round(total_days / prev_injuries, 1) if prev_injuries > 0 else 0,
        ),
        model_predictions=ModelPredictions(
            ensemble=round(prob, 3),
            lgb=round(row.get("lgb_prob", prob), 3),
            xgb=round(row.get("xgb_prob", prob), 3),
            catboost=round(row.get("catboost_prob", prob), 3),
        ),
        recommendations=get_recommendations(row),
        last_injury_date=str(row.get("last_injury_date")) if row.get("last_injury_date") else None,
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


@app.get("/api/players", response_model=List[PlayerSummary])
async def list_players(team: Optional[str] = None, risk_level: Optional[str] = None):
    """
    List all players with summary info.

    Optional filters:
    - team: Filter by team name
    - risk_level: Filter by High/Medium/Low
    """
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    df = inference_df.copy()

    # Apply filters
    if team:
        df = df[df["team"].str.lower() == team.lower()]

    if risk_level:
        df["risk_level"] = df["ensemble_prob"].apply(get_risk_level)
        df = df[df["risk_level"].str.lower() == risk_level.lower()]

    # Sort by risk (highest first)
    df = df.sort_values("ensemble_prob", ascending=False)

    players = []
    for _, row in df.iterrows():
        prob = row.get("ensemble_prob", 0.5)
        players.append(PlayerSummary(
            name=row.get("name", "Unknown"),
            team=row.get("team", "Unknown"),
            position=row.get("position", "Unknown"),
            risk_level=get_risk_level(prob),
            risk_probability=round(prob, 3),
            archetype=row.get("archetype", "Unknown"),
        ))

    return players


@app.get("/api/players/{player_name}/risk", response_model=PlayerRisk)
async def get_player_risk(player_name: str):
    """Get detailed risk prediction for a specific player."""
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Case-insensitive search
    matches = inference_df[
        inference_df["name"].str.lower() == player_name.lower()
    ]

    if matches.empty:
        # Try partial match
        matches = inference_df[
            inference_df["name"].str.lower().str.contains(player_name.lower())
        ]

    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

    row = matches.iloc[0].to_dict()
    return player_row_to_risk(row)


@app.get("/api/teams", response_model=List[str])
async def list_teams():
    """List all available teams."""
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    teams = sorted(inference_df["team"].unique().tolist())
    return teams


@app.get("/api/teams/{team_name}/overview", response_model=TeamOverview)
async def get_team_overview(team_name: str):
    """Get risk overview for an entire team."""
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Case-insensitive team match
    team_df = inference_df[
        inference_df["team"].str.lower() == team_name.lower()
    ]

    if team_df.empty:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")

    # Calculate risk levels
    team_df = team_df.copy()
    team_df["risk_level"] = team_df["ensemble_prob"].apply(get_risk_level)

    high_risk = len(team_df[team_df["risk_level"] == "High"])
    medium_risk = len(team_df[team_df["risk_level"] == "Medium"])
    low_risk = len(team_df[team_df["risk_level"] == "Low"])

    # Sort players by risk
    team_df = team_df.sort_values("ensemble_prob", ascending=False)

    players = []
    for _, row in team_df.iterrows():
        prob = row.get("ensemble_prob", 0.5)
        players.append(PlayerSummary(
            name=row.get("name", "Unknown"),
            team=row.get("team", "Unknown"),
            position=row.get("position", "Unknown"),
            risk_level=get_risk_level(prob),
            risk_probability=round(prob, 3),
            archetype=row.get("archetype", "Unknown"),
        ))

    return TeamOverview(
        team=team_df.iloc[0]["team"],
        total_players=len(team_df),
        high_risk_count=high_risk,
        medium_risk_count=medium_risk,
        low_risk_count=low_risk,
        avg_risk=round(team_df["ensemble_prob"].mean(), 3),
        players=players,
    )


@app.get("/api/archetypes")
async def list_archetypes():
    """List all player archetypes with descriptions."""
    return ARCHETYPE_DESCRIPTIONS


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
