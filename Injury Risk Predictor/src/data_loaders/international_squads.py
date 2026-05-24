"""International (national team) data loaders.

Sources:
1. football-data.org for tournament squads and fixtures (free tier covers the
   FIFA World Cup under competition code ``WC``).
2. Transfermarkt national-team pages for caps/goals — used as a supplemental
   enrichment when a player is known to the club pipeline but their caps
   record needs filling in.

Why a separate module rather than extending ``api_client.FootballDataClient``:
the WC has a different shape (groups, knockouts, country squads) and reuses
none of the league-specific normalisation. Keeping it isolated makes the WC
data path easy to test and easy to swap if we change provider.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests

from .api_client import BASE_URL, RATE_LIMIT_DELAY
from ..utils.logger import get_logger

logger = get_logger(__name__)

WORLD_CUP_CODE = "WC"
WORLD_CUP_2026_SEASON = 2026  # football-data.org keys tournaments by start year


class WorldCupClient:
    """Thin client for World Cup squad and fixture data.

    Composed (not subclassed) so the rate-limit + auth boilerplate stays in
    one place if we later swap providers.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FOOTBALL_DATA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FOOTBALL_DATA_API_KEY required for World Cup data. "
                "Register at https://www.football-data.org/client/register"
            )
        self.session = requests.Session()
        self.session.headers.update({"X-Auth-Token": self.api_key})
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        import time
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        self._rate_limit()
        url = f"{BASE_URL}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def get_teams(self, season: int = WORLD_CUP_2026_SEASON) -> List[Dict]:
        """Return participating national teams for the tournament season."""
        data = self._get(f"competitions/{WORLD_CUP_CODE}/teams", {"season": season})
        return data.get("teams", [])

    def get_squads(self, season: int = WORLD_CUP_2026_SEASON) -> pd.DataFrame:
        """Fetch the official squad list for every participating team.

        Returns one row per player. Squads typically lock 7 days before
        kickoff; before that the API may return preliminary or empty squads,
        so an empty DataFrame is a valid (logged) outcome.
        """
        teams = self.get_teams(season=season)
        if not teams:
            logger.warning("World Cup teams endpoint returned no teams for season %s", season)
            return pd.DataFrame()

        rows: List[Dict] = []
        for team in teams:
            team_id = team.get("id")
            country = team.get("name") or team.get("shortName")
            try:
                team_data = self._get(f"teams/{team_id}")
            except requests.HTTPError as exc:
                logger.warning("Squad fetch failed for %s: %s", country, exc)
                continue
            squad = team_data.get("squad") or []
            if not squad:
                logger.info("No squad players returned for %s (likely pre-announcement)", country)
            for p in squad:
                dob = p.get("dateOfBirth")
                age: Optional[int] = None
                if dob:
                    try:
                        age = (datetime.now() - datetime.strptime(dob, "%Y-%m-%d")).days // 365
                    except ValueError:
                        age = None
                rows.append(
                    {
                        "name": p.get("name"),
                        "national_team": country,
                        "national_team_short": team.get("tla") or team.get("shortName"),
                        "position": p.get("position", "Unknown"),
                        "shirt_number": p.get("shirtNumber"),
                        "age": age,
                        "date_of_birth": dob,
                        "nationality": p.get("nationality"),
                    }
                )

        df = pd.DataFrame(rows)
        logger.info("Fetched %d international players from %d teams", len(df), len(teams))
        return df

    def get_fixtures(self, season: int = WORLD_CUP_2026_SEASON) -> pd.DataFrame:
        """Fetch all tournament fixtures (group stage + knockouts).

        The ``stage`` field separates ``GROUP_STAGE`` from ``LAST_16``,
        ``QUARTER_FINALS`` etc. The ``group`` field is only populated for
        group stage fixtures.
        """
        data = self._get(f"competitions/{WORLD_CUP_CODE}/matches", {"season": season})
        matches = data.get("matches") or []
        rows: List[Dict] = []
        for m in matches:
            score = (m.get("score") or {}).get("fullTime") or {}
            rows.append(
                {
                    "match_id": m.get("id"),
                    "utc_date": m.get("utcDate"),
                    "status": m.get("status"),
                    "stage": m.get("stage"),
                    "group": m.get("group"),
                    "matchday": m.get("matchday"),
                    "home_team": (m.get("homeTeam") or {}).get("name"),
                    "away_team": (m.get("awayTeam") or {}).get("name"),
                    "home_goals": score.get("home"),
                    "away_goals": score.get("away"),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df["utc_date"] = pd.to_datetime(df["utc_date"], errors="coerce")
            df = df.sort_values("utc_date").reset_index(drop=True)
        logger.info("Fetched %d World Cup fixtures (season %s)", len(df), season)
        return df


def build_groups_map(fixtures_df: pd.DataFrame) -> Dict[str, str]:
    """Derive ``{country: group}`` from group-stage fixtures.

    Knockout fixtures don't carry a ``group`` field, so we read it from group
    stage rows only.
    """
    if fixtures_df is None or fixtures_df.empty or "group" not in fixtures_df.columns:
        return {}
    groups: Dict[str, str] = {}
    gs = fixtures_df[fixtures_df["group"].notna()]
    for _, row in gs.iterrows():
        group = str(row["group"])
        for col in ("home_team", "away_team"):
            team = row.get(col)
            if isinstance(team, str) and team and team not in groups:
                groups[team] = group
    return groups


def next_fixture_for_country(
    fixtures_df: pd.DataFrame,
    country: str,
    now: Optional[datetime] = None,
) -> Optional[Dict]:
    """Return the next scheduled fixture for a country, or ``None``."""
    if fixtures_df is None or fixtures_df.empty:
        return None
    now = now or datetime.utcnow()
    upcoming = fixtures_df[
        ((fixtures_df["home_team"] == country) | (fixtures_df["away_team"] == country))
        & (fixtures_df["utc_date"] >= pd.Timestamp(now))
    ].sort_values("utc_date")
    if upcoming.empty:
        return None
    row = upcoming.iloc[0]
    opponent = row["away_team"] if row["home_team"] == country else row["home_team"]
    return {
        "opponent": opponent,
        "is_home": row["home_team"] == country,
        "utc_date": row["utc_date"].isoformat() if pd.notna(row["utc_date"]) else None,
        "stage": row.get("stage"),
        "group": row.get("group"),
        "matchday": row.get("matchday"),
    }
