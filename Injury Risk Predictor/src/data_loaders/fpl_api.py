"""
Fantasy Premier League API integration.

Fetches fixtures, gameweeks, and standings data from the FPL API.
"""

import requests
from typing import Dict, List, Optional
from datetime import datetime
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)

FPL_BASE_URL = "https://fantasy.premierleague.com/api"


class FPLClient:
    """Client for the Fantasy Premier League API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        })
        self._bootstrap_cache = None

    def _fetch(self, endpoint: str) -> Dict:
        """Fetch data from FPL API."""
        url = f"{FPL_BASE_URL}/{endpoint}"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"FPL API error: {e}")
            return {}

    def get_bootstrap(self) -> Dict:
        """Get the main bootstrap-static data (teams, players, events)."""
        if self._bootstrap_cache:
            return self._bootstrap_cache
        self._bootstrap_cache = self._fetch("bootstrap-static/")
        return self._bootstrap_cache

    def get_teams(self) -> List[Dict]:
        """Get all Premier League teams with their info."""
        data = self.get_bootstrap()
        return data.get("teams", [])

    def get_current_gameweek(self) -> Optional[Dict]:
        """Get the current gameweek info."""
        data = self.get_bootstrap()
        events = data.get("events", [])
        for event in events:
            if event.get("is_current"):
                return event
        # If no current, find next
        for event in events:
            if event.get("is_next"):
                return event
        return None

    def get_fixtures(self, gameweek: Optional[int] = None) -> List[Dict]:
        """
        Get fixtures, optionally filtered by gameweek.

        Args:
            gameweek: Specific gameweek number, or None for all fixtures
        """
        endpoint = "fixtures/"
        if gameweek:
            endpoint += f"?event={gameweek}"
        return self._fetch(endpoint) or []

    def get_standings(self) -> List[Dict]:
        """
        Get current Premier League standings.

        Returns list of teams sorted by position with:
        - name, short_name, position, played, wins, draws, losses, points, form
        """
        teams = self.get_teams()

        # Sort by position (lowest first)
        standings = sorted(teams, key=lambda t: t.get("position", 99))

        result = []
        for team in standings:
            result.append({
                "id": team.get("id"),
                "name": team.get("name"),
                "short_name": team.get("short_name"),
                "position": team.get("position", 0),
                "played": team.get("played", 0),
                "wins": team.get("win", 0),
                "draws": team.get("draw", 0),
                "losses": team.get("loss", 0),
                "points": team.get("points", 0),
                "form": team.get("form"),
                "strength": team.get("strength", 3),
            })
        return result

    def get_double_gameweek_teams(self) -> Dict[str, List[int]]:
        """
        Find teams with double gameweeks (2+ fixtures in a gameweek).

        Returns:
            Dict mapping gameweek number -> list of team IDs with double fixtures
        """
        fixtures = self.get_fixtures()
        teams = {t["id"]: t["name"] for t in self.get_teams()}

        # Count fixtures per team per gameweek
        gw_team_counts: Dict[int, Dict[int, int]] = {}

        for fixture in fixtures:
            gw = fixture.get("event")
            if not gw:
                continue

            if gw not in gw_team_counts:
                gw_team_counts[gw] = {}

            home = fixture.get("team_h")
            away = fixture.get("team_a")

            for team_id in [home, away]:
                if team_id:
                    gw_team_counts[gw][team_id] = gw_team_counts[gw].get(team_id, 0) + 1

        # Find gameweeks with double fixtures
        double_gws = {}
        for gw, team_counts in gw_team_counts.items():
            double_teams = [tid for tid, count in team_counts.items() if count >= 2]
            if double_teams:
                double_gws[gw] = double_teams

        return double_gws

    def get_upcoming_fixtures_summary(self, num_gameweeks: int = 3) -> List[Dict]:
        """
        Get summary of upcoming fixtures for FPL insights.

        Returns list of gameweek summaries with:
        - gameweek number
        - deadline
        - fixture count
        - double gameweek teams
        - featured matches (top 6 teams playing each other)
        """
        current_gw = self.get_current_gameweek()
        if not current_gw:
            return []

        current_id = current_gw.get("id", 1)
        teams = {t["id"]: t for t in self.get_teams()}
        top_6_ids = {t["id"] for t in self.get_teams() if t.get("position", 99) <= 6}

        fixtures = self.get_fixtures()
        double_gws = self.get_double_gameweek_teams()

        result = []
        for gw_num in range(current_id, current_id + num_gameweeks):
            gw_fixtures = [f for f in fixtures if f.get("event") == gw_num]

            featured = []
            for f in gw_fixtures:
                home_id = f.get("team_h")
                away_id = f.get("team_a")
                if home_id in top_6_ids and away_id in top_6_ids:
                    home_name = teams.get(home_id, {}).get("short_name", "???")
                    away_name = teams.get(away_id, {}).get("short_name", "???")
                    featured.append(f"{home_name} vs {away_name}")

            double_teams = []
            if gw_num in double_gws:
                double_teams = [
                    teams.get(tid, {}).get("name", "Unknown")
                    for tid in double_gws[gw_num]
                ]

            # Get event info
            events = self.get_bootstrap().get("events", [])
            event = next((e for e in events if e.get("id") == gw_num), {})

            result.append({
                "gameweek": gw_num,
                "name": event.get("name", f"Gameweek {gw_num}"),
                "deadline": event.get("deadline_time"),
                "is_current": event.get("is_current", False),
                "is_next": event.get("is_next", False),
                "fixture_count": len(gw_fixtures),
                "double_gameweek_teams": double_teams,
                "featured_matches": featured,
            })

        return result


def get_fpl_insights() -> Dict:
    """
    Get FPL insights for the dashboard.

    Returns summary data useful for injury risk context.
    """
    client = FPLClient()

    try:
        standings = client.get_standings()
        upcoming = client.get_upcoming_fixtures_summary(3)
        current_gw = client.get_current_gameweek()

        return {
            "current_gameweek": current_gw.get("id") if current_gw else None,
            "standings": standings[:6],  # Top 6 only
            "upcoming_gameweeks": upcoming,
            "has_double_gameweek": any(gw.get("double_gameweek_teams") for gw in upcoming),
        }
    except Exception as e:
        logger.error(f"Failed to get FPL insights: {e}")
        return {}
