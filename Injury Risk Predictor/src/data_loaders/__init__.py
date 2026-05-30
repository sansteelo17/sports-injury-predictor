"""Data loading utilities for injury risk prediction."""

from .load_data import (
    load_all,
    load_injury_data,
    load_match_data,
    load_player_stats,
)

# API client for live data (optional - requires API key)
try:
    from .api_client import (
        FootballDataClient,
        fetch_current_season_matches,
        fetch_historical_matches,
    )
except ImportError:
    pass  # requests not installed

# FBref season stats via soccerdata (optional - requires soccerdata + Chrome)
try:
    from .soccerdata_loader import load_player_season_stats
except ImportError:
    pass  # soccerdata not installed

__all__ = [
    "load_all",
    "load_injury_data",
    "load_match_data",
    "load_player_stats",
    # API client
    "FootballDataClient",
    "fetch_current_season_matches",
    "fetch_historical_matches",
    # FBref data via soccerdata
    "load_player_season_stats",
]
