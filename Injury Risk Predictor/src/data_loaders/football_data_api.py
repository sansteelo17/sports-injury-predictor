"""
Premier League standings via ESPN API (no auth required).
"""

import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)

# ESPN API - free, no auth required
ESPN_API_URL = "https://site.api.espn.com/apis/v2/sports/soccer/eng.1/standings"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"


class FootballDataClient:
    """Client for Premier League standings via ESPN."""

    def __init__(self):
        """Initialize client."""
        self.session = requests.Session()
        self._cache = {}
        self._cache_file = CACHE_DIR / "standings_cache.json"
        self._cache_ttl = timedelta(hours=1)  # Cache for 1 hour
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk."""
        try:
            if self._cache_file.exists():
                with open(self._cache_file) as f:
                    data = json.load(f)
                    # Check if cache is still valid
                    cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
                    if datetime.now() - cached_at < self._cache_ttl:
                        self._cache = data
        except Exception as e:
            logger.debug(f"Failed to load cache: {e}")

    def _save_cache(self):
        """Save cache to disk."""
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache["cached_at"] = datetime.now().isoformat()
            with open(self._cache_file, "w") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.debug(f"Failed to save cache: {e}")

    def get_standings(self) -> List[Dict]:
        """
        Get current Premier League standings from ESPN.

        Returns list of teams with:
        - position, name, short_name, played, won, draw, lost, points, form
        """
        # Check cache first
        if "standings" in self._cache:
            return self._cache["standings"]

        try:
            response = self.session.get(ESPN_API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()

            standings = []
            # ESPN nests under children[0].standings.entries
            children = data.get("children", [])
            if children:
                entries = children[0].get("standings", {}).get("entries", [])
            else:
                entries = data.get("standings", {}).get("entries", [])

            for entry in entries:
                team = entry.get("team", {})
                stats = {}
                for s in entry.get("stats", []):
                    name = s.get("name")
                    value = s.get("value")
                    if name and value is not None:
                        stats[name] = value

                standings.append({
                    "position": int(stats.get("rank", 0)),
                    "name": team.get("displayName", "Unknown"),
                    "short_name": team.get("abbreviation", "???"),
                    "played": int(stats.get("gamesPlayed", 0)),
                    "won": int(stats.get("wins", 0)),
                    "draw": int(stats.get("ties", 0)),
                    "lost": int(stats.get("losses", 0)),
                    "points": int(stats.get("points", 0)),
                    "goal_difference": int(stats.get("pointDifferential", 0)),
                    "form": "",  # ESPN doesn't provide form string
                })

            # Sort by position
            standings.sort(key=lambda x: x["position"])

            # Cache results
            self._cache["standings"] = standings
            self._save_cache()

            return standings

        except Exception as e:
            logger.warning(f"Failed to fetch standings: {e}")
            # Return cached data if available, even if expired
            return self._cache.get("standings", [])

    def get_team_position(self, team_name: str) -> Optional[Dict]:
        """
        Get a specific team's standing.

        Args:
            team_name: Team name (partial match supported)

        Returns:
            Team standing dict or None
        """
        standings = self.get_standings()
        team_lower = team_name.lower()

        for team in standings:
            if (team_lower in team["name"].lower() or
                team_lower in team["short_name"].lower()):
                return team

        return None

    def get_title_race_info(self, team_name: Optional[str] = None) -> Dict:
        """
        Get title race context.

        Args:
            team_name: Optional team to get distance for

        Returns:
            Dict with leader, second, gap, and optional team_distance
        """
        standings = self.get_standings()

        if len(standings) < 2:
            return {}

        leader = standings[0]
        second = standings[1]
        gap = leader["points"] - second["points"]

        result = {
            "leader": {
                "name": leader["name"],
                "short_name": leader["short_name"],
                "points": leader["points"],
                "played": leader["played"],
            },
            "second": {
                "name": second["name"],
                "short_name": second["short_name"],
                "points": second["points"],
            },
            "gap_to_second": gap,
        }

        # Add selected team's position if provided
        if team_name:
            team = self.get_team_position(team_name)
            if team:
                result["selected_team"] = {
                    "name": team["name"],
                    "short_name": team["short_name"],
                    "position": team["position"],
                    "points": team["points"],
                    "played": team["played"],
                    "form": team["form"],
                    "distance_from_top": leader["points"] - team["points"],
                }

        return result


def get_standings_summary(team_name: Optional[str] = None) -> Dict:
    """
    Get a summary of standings for the dashboard.

    Args:
        team_name: Optional team to include in summary

    Returns:
        Dict with leader, title_race, and optionally selected_team info
    """
    client = FootballDataClient()
    return client.get_title_race_info(team_name)
