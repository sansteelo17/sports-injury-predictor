"""
External betting odds integration.

Fetches real-time odds from The Odds API (aggregates multiple bookmakers).
Free tier: 500 requests/month.
"""

import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
import os

from ..utils.logger import get_logger

logger = get_logger(__name__)

# The Odds API - free tier available
ODDS_API_URL = "https://api.the-odds-api.com/v4"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"

# Premier League sport key
SPORT_KEY = "soccer_epl"

# Market types we care about
MARKETS = {
    "h2h": "Match Winner",
    "totals": "Over/Under Goals",
    "btts": "Both Teams to Score",
}


class OddsClient:
    """Client for fetching betting odds from external APIs."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize with API key.

        Get free API key at: https://the-odds-api.com/
        Set ODDS_API_KEY environment variable or pass directly.
        """
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")
        self.session = requests.Session()
        self._cache = {}
        self._cache_file = CACHE_DIR / "odds_cache.json"
        self._cache_ttl = timedelta(hours=2)
        self._load_cache()

    def _load_cache(self):
        """Load cached odds from disk."""
        try:
            if self._cache_file.exists():
                with open(self._cache_file) as f:
                    data = json.load(f)
                    cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
                    if datetime.now() - cached_at < self._cache_ttl:
                        self._cache = data
                        logger.info(f"Loaded odds cache from {cached_at}")
        except Exception as e:
            logger.debug(f"Failed to load odds cache: {e}")

    def _save_cache(self):
        """Save odds to disk cache."""
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache["cached_at"] = datetime.now().isoformat()
            with open(self._cache_file, "w") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.debug(f"Failed to save odds cache: {e}")

    def get_upcoming_matches(self) -> List[Dict]:
        """
        Get upcoming Premier League matches with odds.

        Returns list of matches with:
        - home_team, away_team
        - commence_time
        - odds from various bookmakers
        """
        if "matches" in self._cache:
            return self._cache["matches"]

        if not self.api_key:
            logger.warning("No ODDS_API_KEY set - using mock data")
            return self._get_mock_matches()

        try:
            response = self.session.get(
                f"{ODDS_API_URL}/sports/{SPORT_KEY}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": "uk,us",
                    "markets": "h2h",
                    "oddsFormat": "american",
                },
                timeout=10,
            )
            response.raise_for_status()
            matches = response.json()

            # Cache results
            self._cache["matches"] = matches
            self._save_cache()

            logger.info(f"Fetched odds for {len(matches)} upcoming matches")
            return matches

        except Exception as e:
            logger.error(f"Failed to fetch odds: {e}")
            return self._cache.get("matches", self._get_mock_matches())

    def _get_mock_matches(self) -> List[Dict]:
        """Return mock match data when API key not available."""
        return [
            {
                "home_team": "Manchester City",
                "away_team": "Ipswich Town",
                "commence_time": (datetime.now() + timedelta(days=3)).isoformat(),
                "bookmakers": [
                    {
                        "key": "mock",
                        "title": "Mock Odds",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Manchester City", "price": -450},
                                    {"name": "Ipswich Town", "price": 1200},
                                    {"name": "Draw", "price": 550},
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "home_team": "Arsenal",
                "away_team": "Wolverhampton Wanderers",
                "commence_time": (datetime.now() + timedelta(days=4)).isoformat(),
                "bookmakers": [
                    {
                        "key": "mock",
                        "title": "Mock Odds",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Arsenal", "price": -280},
                                    {"name": "Wolverhampton Wanderers", "price": 800},
                                    {"name": "Draw", "price": 400},
                                ],
                            }
                        ],
                    }
                ],
            },
        ]

    def get_team_next_match(self, team_name: str) -> Optional[Dict]:
        """
        Get a team's next match with odds.

        Args:
            team_name: Team name (partial match supported)

        Returns:
            Match dict with odds, or None
        """
        matches = self.get_upcoming_matches()
        team_lower = team_name.lower()

        # Team name aliases for matching
        aliases = {
            "man city": "manchester city",
            "man united": "manchester united",
            "man utd": "manchester united",
            "wolves": "wolverhampton",
            "spurs": "tottenham",
        }
        search_term = aliases.get(team_lower, team_lower)

        for match in matches:
            home = match.get("home_team", "").lower()
            away = match.get("away_team", "").lower()

            if search_term in home or search_term in away or team_lower in home or team_lower in away:
                return match

        return None

    def get_anytime_scorer_odds(self, team_name: str, player_name: str) -> Optional[Dict]:
        """
        Get anytime goal scorer odds for a player from The Odds API.

        Uses the player_goal_scorer_anytime market. Falls back to None
        if market unavailable or no API key.

        Args:
            team_name: Player's team name
            player_name: Player's name

        Returns:
            Dict with bookmaker, decimal_odds, implied_probability, or None
        """
        if not self.api_key:
            return None

        # Check cache
        cache_key = f"scorer_{player_name.lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            response = self.session.get(
                f"{ODDS_API_URL}/sports/{SPORT_KEY}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": "uk,us",
                    "markets": "player_goal_scorer_anytime",
                    "oddsFormat": "decimal",
                },
                timeout=10,
            )
            response.raise_for_status()
            matches = response.json()

            # Find the right match for this team
            team_lower = team_name.lower()
            player_lower = player_name.lower()
            player_last = player_name.split()[-1].lower() if player_name else ""

            for match in matches:
                home = match.get("home_team", "").lower()
                away = match.get("away_team", "").lower()

                if team_lower not in home and team_lower not in away:
                    continue

                # Search bookmakers for player odds
                for bookie in match.get("bookmakers", []):
                    for market in bookie.get("markets", []):
                        if market.get("key") != "player_goal_scorer_anytime":
                            continue
                        for outcome in market.get("outcomes", []):
                            outcome_name = outcome.get("name", "").lower()
                            if (player_lower in outcome_name or
                                    player_last in outcome_name or
                                    outcome_name in player_lower):
                                decimal_odds = outcome.get("price", 0)
                                if decimal_odds > 0:
                                    implied_prob = round(1 / decimal_odds, 3)
                                    result = {
                                        "bookmaker": bookie.get("title", "Unknown"),
                                        "decimal_odds": decimal_odds,
                                        "implied_probability": implied_prob,
                                        "player_matched": outcome.get("name", ""),
                                        "opponent": match.get("away_team") if team_lower in home else match.get("home_team"),
                                        "is_home": team_lower in home,
                                    }
                                    self._cache[cache_key] = result
                                    return result

            # Player not found in scorer markets
            self._cache[cache_key] = None
            return None

        except Exception as e:
            logger.debug(f"Failed to fetch scorer odds for {player_name}: {e}")
            return None

    def get_clean_sheet_odds(self, team_name: str) -> Optional[Dict]:
        """
        Estimate clean sheet odds for a team's next match.

        Uses match winner odds + opponent strength to estimate CS probability.

        Args:
            team_name: Team name

        Returns:
            Dict with clean_sheet_probability, american odds, insight
        """
        match = self.get_team_next_match(team_name)
        if not match:
            return None

        team_lower = team_name.lower()
        home_team = match.get("home_team", "")
        away_team = match.get("away_team", "")
        is_home = team_lower in home_team.lower()
        opponent = away_team if is_home else home_team

        # Get win odds
        bookmakers = match.get("bookmakers", [])
        if not bookmakers:
            return None

        outcomes = bookmakers[0].get("markets", [{}])[0].get("outcomes", [])
        team_odds = None
        for outcome in outcomes:
            if team_lower in outcome.get("name", "").lower():
                team_odds = outcome.get("price")
                break

        if team_odds is None:
            return None

        # Convert American odds to probability
        if team_odds < 0:
            win_prob = abs(team_odds) / (abs(team_odds) + 100)
        else:
            win_prob = 100 / (team_odds + 100)

        # Estimate CS probability (roughly 60-70% of wins are clean sheets for top teams)
        # Adjust based on being favorite/underdog
        if win_prob > 0.6:  # Strong favorite
            cs_multiplier = 0.65
        elif win_prob > 0.4:  # Slight favorite
            cs_multiplier = 0.50
        else:  # Underdog
            cs_multiplier = 0.35

        cs_prob = win_prob * cs_multiplier

        # Convert back to American odds
        if cs_prob >= 0.5:
            american = int(-100 * cs_prob / (1 - cs_prob))
            american_str = str(american)
        else:
            american = int(100 * (1 - cs_prob) / cs_prob)
            american_str = f"+{american}"

        decimal_odds = round(1 / cs_prob, 2)

        return {
            "team": team_name,
            "opponent": opponent,
            "is_home": is_home,
            "clean_sheet_probability": round(cs_prob, 3),
            "american": american_str,
            "decimal": decimal_odds,
            "win_probability": round(win_prob, 3),
            "match_time": match.get("commence_time"),
            "source": bookmakers[0].get("title", "Unknown"),
        }


def get_clean_sheet_insight(team_name: str, injury_prob: float) -> Optional[str]:
    """
    Generate insight combining bookmaker CS odds with injury risk.

    Args:
        team_name: Player's team
        injury_prob: Player's injury probability

    Returns:
        Insight string or None
    """
    client = OddsClient()
    cs_data = client.get_clean_sheet_odds(team_name)

    if not cs_data:
        return None

    cs_prob = cs_data["clean_sheet_probability"]
    opponent = cs_data["opponent"]
    american = cs_data["american"]
    is_home = cs_data["is_home"]
    venue = "at home" if is_home else "away"

    # High CS odds + Low injury risk = Great pick
    if cs_prob >= 0.35 and injury_prob < 0.20:
        return f"🛡️ Clean Sheet Alert: {team_name} are {american} for a CS {venue} vs {opponent}. With low injury risk, this is a solid defensive pick!"

    # High CS odds + High injury risk = Risky
    if cs_prob >= 0.35 and injury_prob >= 0.30:
        return f"⚠️ {team_name} have good CS odds ({american}) vs {opponent}, but this player's elevated injury risk means they might not be on the pitch to collect those points."

    # Low CS odds (tough fixture)
    if cs_prob < 0.25:
        return f"⚔️ Tough fixture: {team_name} face {opponent} - clean sheet unlikely ({american}). Consider bench fodder for this week."

    # Moderate situation
    if cs_prob >= 0.25:
        return f"📊 {team_name} are {american} for a CS vs {opponent}. {'Low injury risk makes this viable.' if injury_prob < 0.20 else 'Monitor fitness before committing.'}"

    return None
