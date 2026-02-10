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

# FBref scraper for player match logs (optional - requires beautifulsoup4)
try:
    from .fbref_scraper import (
        FBrefScraper,
        fetch_all_player_workloads,
    )
except ImportError:
    pass  # beautifulsoup4 not installed

__all__ = [
    "load_all",
    "load_injury_data",
    "load_match_data",
    "load_player_stats",
    # API client
    "FootballDataClient",
    "fetch_current_season_matches",
    "fetch_historical_matches",
    # FBref scraper
    "FBrefScraper",
    "fetch_all_player_workloads",
]
