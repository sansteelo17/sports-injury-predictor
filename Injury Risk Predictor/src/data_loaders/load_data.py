import pandas as pd
from pathlib import Path

# ALWAYS resolve root based on this file's location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "csv"


def load_injury_data():
    return pd.read_csv(DATA_DIR / "player_injuries_impact.csv")


def load_match_data():
    df = pd.read_csv(DATA_DIR / "premier-league-matches.csv")
    if "league" not in df.columns:
        df["league"] = "Premier League"
    return df


def load_laliga_match_data():
    path = DATA_DIR / "la-liga-matches.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["Date"])
    if "league" not in df.columns:
        df["league"] = "La Liga"
    return df


def load_all_matches() -> pd.DataFrame:
    """Load EPL + La Liga match data combined."""
    epl = load_match_data()
    laliga = load_laliga_match_data()
    if laliga.empty:
        return epl
    combined = pd.concat([epl, laliga], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    return combined.sort_values("Date").reset_index(drop=True)


def load_player_stats():
    return pd.read_csv(DATA_DIR / "All_Players_1992-2025.csv", low_memory=False)


def load_all():
    return {
        "injuries": load_injury_data(),
        "matches": load_match_data(),
        "stats": load_player_stats(),
    }


def load_all_with_laliga():
    """Load everything including La Liga match data."""
    return {
        "injuries": load_injury_data(),
        "matches": load_all_matches(),
        "stats": load_player_stats(),
    }