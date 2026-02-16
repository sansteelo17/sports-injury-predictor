import pandas as pd
from pathlib import Path

# ALWAYS resolve root based on this file's location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "csv"

def load_injury_data():
    return pd.read_csv(DATA_DIR / "player_injuries_impact.csv")

def load_match_data():
    return pd.read_csv(DATA_DIR / "premier-league-matches.csv")

def load_player_stats():
    return pd.read_csv(DATA_DIR / "All_Players_1992-2025.csv")

def load_all():
    return {
        "injuries": load_injury_data(),
        "matches": load_match_data(),
        "stats": load_player_stats(),
    }