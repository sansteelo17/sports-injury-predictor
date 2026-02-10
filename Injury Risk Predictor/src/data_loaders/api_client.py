"""
Football data API client for fetching live match data.

Supports football-data.org API (free tier: 10 requests/minute).
Get your free API key at: https://www.football-data.org/client/register

Usage:
    from src.data_loaders.api_client import FootballDataClient

    client = FootballDataClient(api_key="your_key")
    matches = client.get_premier_league_matches(season=2024)
    roster = client.get_team_squad("Arsenal")
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from ..utils.logger import get_logger

logger = get_logger(__name__)

# API Configuration
BASE_URL = "https://api.football-data.org/v4"
PREMIER_LEAGUE_ID = "PL"
RATE_LIMIT_DELAY = 6.5  # seconds between requests (free tier: 10/min)


class FootballDataClient:
    """
    Client for football-data.org API.

    Free tier limits:
    - 10 requests per minute
    - Premier League, Championship, and other major leagues
    - Match results, fixtures, standings, team squads
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the API client.

        Args:
            api_key: Your football-data.org API key.
                     Can also be set via FOOTBALL_DATA_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("FOOTBALL_DATA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Get one free at https://www.football-data.org/client/register\n"
                "Set via: FootballDataClient(api_key='...') or FOOTBALL_DATA_API_KEY env var"
            )

        self.session = requests.Session()
        self.session.headers.update({"X-Auth-Token": self.api_key})
        self._last_request_time = 0

    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            sleep_time = RATE_LIMIT_DELAY - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make a GET request to the API."""
        self._rate_limit()

        url = f"{BASE_URL}/{endpoint}"
        logger.debug(f"GET {url} params={params}")

        response = self.session.get(url, params=params)

        if response.status_code == 429:
            # Rate limited - wait and retry
            logger.warning("Rate limited, waiting 60s...")
            time.sleep(60)
            return self._get(endpoint, params)

        response.raise_for_status()
        return response.json()

    def get_premier_league_matches(
        self,
        season: int = 2024,
        status: str = "FINISHED",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch Premier League matches.

        Args:
            season: Season year (e.g., 2024 for 2024-25 season)
            status: Match status filter - SCHEDULED, LIVE, IN_PLAY,
                    PAUSED, FINISHED, POSTPONED, SUSPENDED, CANCELLED
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns matching our pipeline format:
            - Season_End_Year, Wk, Date, Home, Away, HomeGoals, AwayGoals, FTR
        """
        params = {"season": season}
        if status:
            params["status"] = status
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

        data = self._get(f"competitions/{PREMIER_LEAGUE_ID}/matches", params)
        matches = data.get("matches", [])

        logger.info(f"Fetched {len(matches)} matches from API")

        if not matches:
            return pd.DataFrame()

        # Convert to our format
        rows = []
        for m in matches:
            if m["status"] != "FINISHED":
                continue

            home_goals = m["score"]["fullTime"]["home"]
            away_goals = m["score"]["fullTime"]["away"]

            # Determine result
            if home_goals > away_goals:
                ftr = "H"
            elif away_goals > home_goals:
                ftr = "A"
            else:
                ftr = "D"

            rows.append({
                "Season_End_Year": season + 1,
                "Wk": m.get("matchday", 0),
                "Date": m["utcDate"][:10],
                "Home": self._normalize_team_name(m["homeTeam"]["shortName"]),
                "Away": self._normalize_team_name(m["awayTeam"]["shortName"]),
                "HomeGoals": home_goals,
                "AwayGoals": away_goals,
                "FTR": ftr,
            })

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        logger.info(f"Processed {len(df)} finished matches")
        return df

    def get_team_squad(self, team_name: str) -> pd.DataFrame:
        """
        Fetch current squad for a team.

        Args:
            team_name: Team name (e.g., "Arsenal", "Manchester United")

        Returns:
            DataFrame with player info: name, position, dateOfBirth, nationality
        """
        # First get team ID
        teams_data = self._get(f"competitions/{PREMIER_LEAGUE_ID}/teams")
        teams = teams_data.get("teams", [])

        team_id = None
        for t in teams:
            if team_name.lower() in t["name"].lower() or team_name.lower() in t["shortName"].lower():
                team_id = t["id"]
                break

        if not team_id:
            raise ValueError(f"Team '{team_name}' not found in Premier League")

        # Get squad
        team_data = self._get(f"teams/{team_id}")
        squad = team_data.get("squad", [])

        rows = []
        for p in squad:
            rows.append({
                "name": p["name"],
                "position": p.get("position", "Unknown"),
                "date_of_birth": p.get("dateOfBirth"),
                "nationality": p.get("nationality"),
            })

        return pd.DataFrame(rows)

    def get_all_team_squads(self) -> pd.DataFrame:
        """
        Fetch squads for all Premier League teams.

        Returns:
            DataFrame with all players: name, team, position, age
        """
        teams_data = self._get(f"competitions/{PREMIER_LEAGUE_ID}/teams")
        teams = teams_data.get("teams", [])

        all_players = []
        for team in teams:
            logger.info(f"Fetching squad for {team['shortName']}...")

            try:
                team_data = self._get(f"teams/{team['id']}")
                squad = team_data.get("squad", [])

                for p in squad:
                    dob = p.get("dateOfBirth")
                    age = None
                    if dob:
                        try:
                            birth_date = datetime.strptime(dob, "%Y-%m-%d")
                            age = (datetime.now() - birth_date).days // 365
                        except ValueError:
                            pass

                    all_players.append({
                        "name": p["name"],
                        "team": team["shortName"],
                        "position": p.get("position", "Unknown"),
                        "age": age,
                        "nationality": p.get("nationality"),
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch {team['shortName']}: {e}")

        df = pd.DataFrame(all_players)
        logger.info(f"Fetched {len(df)} players from {len(teams)} teams")
        return df

    def get_upcoming_fixtures(self, days_ahead: int = 14) -> pd.DataFrame:
        """
        Fetch upcoming fixtures.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            DataFrame with scheduled matches
        """
        date_from = datetime.now().strftime("%Y-%m-%d")
        date_to = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        params = {
            "dateFrom": date_from,
            "dateTo": date_to,
            "status": "SCHEDULED",
        }

        data = self._get(f"competitions/{PREMIER_LEAGUE_ID}/matches", params)
        matches = data.get("matches", [])

        rows = []
        for m in matches:
            rows.append({
                "date": m["utcDate"][:10],
                "time": m["utcDate"][11:16],
                "home": self._normalize_team_name(m["homeTeam"]["shortName"]),
                "away": self._normalize_team_name(m["awayTeam"]["shortName"]),
                "matchday": m.get("matchday"),
            })

        return pd.DataFrame(rows)

    def _normalize_team_name(self, name: str) -> str:
        """
        Normalize team names to match our historical data.

        The API uses slightly different names than our CSV data.
        """
        # Mapping from API names to our historical names
        name_map = {
            "Man United": "Manchester United",
            "Man City": "Manchester City",
            "Spurs": "Tottenham",
            "Tottenham Hotspur": "Tottenham",
            "Wolves": "Wolverhampton",
            "Brighton Hove": "Brighton",
            "Brighton & Hove Albion": "Brighton",
            "Newcastle Utd": "Newcastle",
            "West Ham Utd": "West Ham",
            "Nott'm Forest": "Nottingham Forest",
            "Nottingham": "Nottingham Forest",
            "Leicester City": "Leicester",
            "Leeds United": "Leeds",
            "Ipswich Town": "Ipswich",
        }
        return name_map.get(name, name)


def fetch_current_season_matches(api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to fetch all matches from current season.

    Args:
        api_key: Optional API key (uses env var if not provided)

    Returns:
        DataFrame with match data in pipeline format
    """
    client = FootballDataClient(api_key)

    # Determine current season
    now = datetime.now()
    if now.month >= 8:  # Season starts in August
        season = now.year
    else:
        season = now.year - 1

    return client.get_premier_league_matches(season=season)


def fetch_historical_matches(
    api_key: Optional[str] = None,
    seasons: List[int] = None,
) -> pd.DataFrame:
    """
    Fetch multiple seasons of match data.

    Args:
        api_key: Optional API key
        seasons: List of season years (e.g., [2022, 2023, 2024])

    Returns:
        Combined DataFrame with all matches
    """
    if seasons is None:
        seasons = [2022, 2023, 2024]

    client = FootballDataClient(api_key)

    all_matches = []
    for season in seasons:
        logger.info(f"Fetching {season}-{season+1} season...")
        df = client.get_premier_league_matches(season=season)
        all_matches.append(df)

    return pd.concat(all_matches, ignore_index=True)
