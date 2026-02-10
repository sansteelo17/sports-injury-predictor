"""
FBref web scraper for player match logs.

Scrapes match-by-match player data including:
- Date of each match
- Minutes played
- Whether started or came off bench
- Goals, assists, cards

Usage:
    from src.data_loaders.fbref_scraper import FBrefScraper

    scraper = FBrefScraper()

    # Get all Premier League players with recent match logs
    players = scraper.get_premier_league_players()

    # Get match logs for a specific player
    logs = scraper.get_player_match_logs("Bruno Fernandes")

Rate limiting: 3 seconds between requests (be respectful to FBref)
"""

import os
import re
import time
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from urllib.parse import urljoin

import requests
import pandas as pd
from bs4 import BeautifulSoup

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Configuration
BASE_URL = "https://fbref.com"
PREMIER_LEAGUE_URL = f"{BASE_URL}/en/comps/9/Premier-League-Stats"
RATE_LIMIT_DELAY = 4.0  # seconds between requests (be respectful to FBref)
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# Cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "fbref"


class FBrefScraper:
    """
    Scraper for FBref player match logs.

    Includes caching to avoid redundant requests.
    """

    def __init__(self, cache_hours: int = 24):
        """
        Initialize the scraper.

        Args:
            cache_hours: Hours to cache responses (default 24)
        """
        self.session = requests.Session()
        # Use headers that look like a real browser to avoid 403 blocks
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        })
        self._last_request_time = 0
        self.cache_hours = cache_hours

        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self):
        """Ensure we don't overwhelm FBref."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            sleep_time = RATE_LIMIT_DELAY - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        return CACHE_DIR / f"{url_hash}.html"

    def _get_cached(self, url: str) -> Optional[str]:
        """Get cached response if fresh enough."""
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - mtime < timedelta(hours=self.cache_hours):
                logger.debug(f"Cache hit: {url[:50]}...")
                return cache_path.read_text()
        return None

    def _save_cache(self, url: str, content: str):
        """Save response to cache."""
        cache_path = self._get_cache_path(url)
        cache_path.write_text(content)

    def _get(self, url: str, use_cache: bool = True) -> str:
        """
        Fetch a URL with rate limiting and caching.

        Args:
            url: URL to fetch
            use_cache: Whether to use cached response

        Returns:
            HTML content
        """
        # Check cache first
        if use_cache:
            cached = self._get_cached(url)
            if cached:
                return cached

        # Rate limit
        self._rate_limit()

        logger.info(f"Fetching: {url[:60]}...")
        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        # Cache the response
        self._save_cache(url, response.text)

        return response.text

    def get_premier_league_teams(self) -> List[Dict]:
        """
        Get all current Premier League teams with their FBref IDs.

        Returns:
            List of dicts with: name, url, fbref_id
        """
        html = self._get(PREMIER_LEAGUE_URL)
        soup = BeautifulSoup(html, "html.parser")

        teams = []

        # Find the squad standard stats table
        table = soup.find("table", {"id": "stats_squads_standard_for"})
        if not table:
            # Try alternate table ID
            table = soup.find("table", {"class": "stats_table"})

        if not table:
            logger.warning("Could not find teams table on FBref")
            return teams

        # Parse team rows
        for row in table.find_all("tr"):
            th = row.find("th", {"data-stat": "team"})
            if th:
                link = th.find("a")
                if link and link.get("href"):
                    href = link["href"]
                    name = link.text.strip()

                    # Extract team ID from URL like /en/squads/18bb7c10/Arsenal-Stats
                    match = re.search(r"/squads/([a-f0-9]+)/", href)
                    fbref_id = match.group(1) if match else None

                    teams.append({
                        "name": name,
                        "url": urljoin(BASE_URL, href),
                        "fbref_id": fbref_id,
                    })

        logger.info(f"Found {len(teams)} Premier League teams")
        return teams

    def get_team_players(self, team_url: str) -> List[Dict]:
        """
        Get all players from a team page.

        Args:
            team_url: FBref team URL

        Returns:
            List of dicts with: name, url, fbref_id, position, age
        """
        html = self._get(team_url)
        soup = BeautifulSoup(html, "html.parser")

        players = []

        # Find the standard stats table
        table = soup.find("table", {"id": "stats_standard_9"})
        if not table:
            table = soup.find("table", {"id": lambda x: x and "stats_standard" in x})

        if not table:
            logger.warning(f"Could not find players table for {team_url}")
            return players

        # Parse player rows
        for row in table.find_all("tr"):
            th = row.find("th", {"data-stat": "player"})
            if th:
                link = th.find("a")
                if link and link.get("href"):
                    href = link["href"]
                    name = link.text.strip()

                    # Extract player ID from URL
                    match = re.search(r"/players/([a-f0-9]+)/", href)
                    fbref_id = match.group(1) if match else None

                    # Get position and age
                    pos_td = row.find("td", {"data-stat": "position"})
                    age_td = row.find("td", {"data-stat": "age"})

                    position = pos_td.text.strip() if pos_td else "Unknown"
                    age_text = age_td.text.strip() if age_td else ""

                    # Parse age (format: "25-123" for 25 years, 123 days)
                    age = None
                    if age_text:
                        age_match = re.match(r"(\d+)", age_text)
                        if age_match:
                            age = int(age_match.group(1))

                    # Get minutes played this season
                    min_td = row.find("td", {"data-stat": "minutes"})
                    minutes = 0
                    if min_td:
                        try:
                            minutes = int(min_td.text.strip().replace(",", ""))
                        except ValueError:
                            pass

                    players.append({
                        "name": name,
                        "url": urljoin(BASE_URL, href),
                        "fbref_id": fbref_id,
                        "position": position,
                        "age": age,
                        "season_minutes": minutes,
                    })

        logger.info(f"Found {len(players)} players")
        return players

    def get_player_match_logs(
        self,
        player_url: str,
        season: str = "2024-2025"
    ) -> pd.DataFrame:
        """
        Get match-by-match logs for a player.

        Args:
            player_url: FBref player URL
            season: Season string (e.g., "2024-2025")

        Returns:
            DataFrame with columns: date, opponent, home_away, minutes,
                                   started, goals, assists, cards
        """
        # Build match logs URL
        # Format: /en/players/{id}/matchlogs/{season}/summary/{name}-Match-Logs
        match = re.search(r"/players/([a-f0-9]+)/([^/]+)", player_url)
        if not match:
            logger.warning(f"Could not parse player URL: {player_url}")
            return pd.DataFrame()

        player_id = match.group(1)
        player_slug = match.group(2)

        logs_url = f"{BASE_URL}/en/players/{player_id}/matchlogs/{season}/summary/{player_slug}-Match-Logs"

        try:
            html = self._get(logs_url)
        except requests.HTTPError as e:
            logger.warning(f"Could not fetch match logs: {e}")
            return pd.DataFrame()

        soup = BeautifulSoup(html, "html.parser")

        # Find the match logs table
        table = soup.find("table", {"id": "matchlogs_all"})
        if not table:
            table = soup.find("table", {"class": "stats_table"})

        if not table:
            logger.warning(f"No match logs table found for {player_url}")
            return pd.DataFrame()

        rows = []
        for tr in table.find_all("tr"):
            # Skip header rows
            if tr.find("th", {"scope": "col"}):
                continue

            # Get date
            date_th = tr.find("th", {"data-stat": "date"})
            if not date_th or not date_th.text.strip():
                continue

            date_str = date_th.text.strip()

            # Parse other fields
            def get_stat(stat_name):
                td = tr.find("td", {"data-stat": stat_name})
                return td.text.strip() if td else ""

            comp = get_stat("comp")

            # Only include Premier League matches
            if comp and "Premier League" not in comp:
                continue

            venue = get_stat("venue")
            opponent = get_stat("opponent")
            result = get_stat("result")

            # Minutes
            minutes_str = get_stat("minutes")
            try:
                minutes = int(minutes_str) if minutes_str else 0
            except ValueError:
                minutes = 0

            # Started or sub
            started = get_stat("game_started") == "Y"

            # Goals, assists
            try:
                goals = int(get_stat("goals") or 0)
            except ValueError:
                goals = 0
            try:
                assists = int(get_stat("assists") or 0)
            except ValueError:
                assists = 0

            # Cards
            yellow = get_stat("cards_yellow")
            red = get_stat("cards_red")

            rows.append({
                "date": date_str,
                "opponent": opponent,
                "home_away": "H" if venue == "Home" else "A",
                "result": result,
                "minutes": minutes,
                "started": started,
                "goals": goals,
                "assists": assists,
                "yellow_card": yellow == "1",
                "red_card": red == "1",
            })

        df = pd.DataFrame(rows)

        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        logger.debug(f"Found {len(df)} Premier League matches")
        return df

    def get_all_premier_league_players(self) -> pd.DataFrame:
        """
        Get all current Premier League players with their team info.

        Returns:
            DataFrame with: name, team, position, age, season_minutes,
                           fbref_id, player_url
        """
        teams = self.get_premier_league_teams()

        all_players = []
        for team in teams:
            logger.info(f"Fetching players for {team['name']}...")

            players = self.get_team_players(team["url"])

            for player in players:
                player["team"] = team["name"]
                all_players.append(player)

        df = pd.DataFrame(all_players)

        # Rename url to player_url
        if "url" in df.columns:
            df = df.rename(columns={"url": "player_url"})

        logger.info(f"Total: {len(df)} players from {len(teams)} teams")
        return df

    def get_player_workload(
        self,
        player_url: str,
        as_of_date: datetime = None,
        season: str = "2024-2025"
    ) -> Dict:
        """
        Calculate workload metrics for a player based on their match logs.

        Computes all workload features needed by the model:
        - acute_load, chronic_load, acwr (basic)
        - monotony, strain, fatigue_index (advanced)
        - workload_slope, spike_flag (trend indicators)

        Args:
            player_url: FBref player URL
            as_of_date: Date to calculate workload as of (default: today)
            season: Season to get logs for

        Returns:
            Dict with all workload features matching model expectations
        """
        import numpy as np

        if as_of_date is None:
            as_of_date = datetime.now()

        logs = self.get_player_match_logs(player_url, season)

        # Default values when no data
        defaults = {
            "acute_load": 0,
            "chronic_load": 0,
            "acwr": 1.0,
            "monotony": 1.0,
            "strain": 0,
            "fatigue_index": 0,
            "workload_slope": 0,
            "spike_flag": 0,
            "matches_last_7": 0,
            "matches_last_14": 0,
            "matches_last_30": 0,
            "minutes_last_7": 0,
            "minutes_last_14": 0,
            "minutes_last_30": 0,
            "total_minutes": 0,
            "total_matches": 0,
            "avg_minutes": 0,
        }

        if len(logs) == 0:
            return defaults

        # Filter to matches before as_of_date
        logs = logs[logs["date"] < as_of_date].copy()

        if len(logs) == 0:
            return defaults

        # Calculate windows
        date_7 = as_of_date - timedelta(days=7)
        date_14 = as_of_date - timedelta(days=14)
        date_28 = as_of_date - timedelta(days=28)
        date_30 = as_of_date - timedelta(days=30)

        matches_7 = logs[logs["date"] >= date_7]
        matches_14 = logs[logs["date"] >= date_14]
        matches_28 = logs[logs["date"] >= date_28]
        matches_30 = logs[logs["date"] >= date_30]

        # Count matches where player actually played (minutes > 0)
        played_7 = len(matches_7[matches_7["minutes"] > 0])
        played_14 = len(matches_14[matches_14["minutes"] > 0])
        played_28 = len(matches_28[matches_28["minutes"] > 0])

        # Minutes in windows
        min_7 = matches_7["minutes"].sum()
        min_14 = matches_14["minutes"].sum()
        min_28 = matches_28["minutes"].sum()
        min_30 = matches_30["minutes"].sum()

        # Workload calculations (using minutes as load proxy)
        acute_load = min_7 / 90  # Convert to "90-minute equivalents"
        chronic_load = (min_28 / 90) / 4  # Weekly average

        # ACWR
        if chronic_load > 0:
            acwr = acute_load / chronic_load
        else:
            acwr = 1.0 if acute_load == 0 else 2.0

        # ---------------------------------
        # MONOTONY (14-day load regularity)
        # ---------------------------------
        # Calculate daily loads for last 14 days
        daily_loads = []
        for i in range(14):
            day = as_of_date - timedelta(days=i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            day_matches = logs[(logs["date"] >= day_start) & (logs["date"] < day_end)]
            daily_loads.append(day_matches["minutes"].sum() / 90)

        mean_load = np.mean(daily_loads)
        std_load = np.std(daily_loads)
        if std_load < 0.1:
            monotony = min(mean_load / 0.1, 5.0) if mean_load > 0 else 1.0
        else:
            monotony = min(mean_load / std_load, 5.0)

        # ---------------------------------
        # STRAIN = acute_load × monotony
        # ---------------------------------
        strain = acute_load * monotony

        # ---------------------------------
        # FATIGUE INDEX = acute - chronic
        # ---------------------------------
        fatigue_index = acute_load - chronic_load

        # ---------------------------------
        # WORKLOAD SLOPE (trend of last 5 matches)
        # ---------------------------------
        recent_matches = logs.nlargest(5, "date")
        if len(recent_matches) >= 2:
            # Use minutes normalized to 90 as load
            loads = (recent_matches["minutes"] / 90).values[::-1]  # oldest to newest
            x = np.arange(len(loads))
            workload_slope = np.polyfit(x, loads, 1)[0]
        else:
            workload_slope = 0

        # ---------------------------------
        # SPIKE FLAG (ACWR > 1.5 danger zone)
        # ---------------------------------
        spike_flag = 1 if acwr > 1.5 else 0

        return {
            "acute_load": round(acute_load, 2),
            "chronic_load": round(chronic_load, 2),
            "acwr": round(acwr, 3),
            "monotony": round(monotony, 3),
            "strain": round(strain, 3),
            "fatigue_index": round(fatigue_index, 3),
            "workload_slope": round(workload_slope, 4),
            "spike_flag": spike_flag,
            "matches_last_7": played_7,
            "matches_last_14": played_14,
            "matches_last_30": len(matches_30[matches_30["minutes"] > 0]),
            "minutes_last_7": min_7,
            "minutes_last_14": min_14,
            "minutes_last_30": min_30,
            "total_minutes": logs["minutes"].sum(),
            "total_matches": len(logs[logs["minutes"] > 0]),
            "avg_minutes": round(logs[logs["minutes"] > 0]["minutes"].mean(), 1) if len(logs[logs["minutes"] > 0]) > 0 else 0,
        }


def fetch_all_player_workloads(cache_hours: int = 12) -> pd.DataFrame:
    """
    Fetch current workload for all Premier League players.

    This is the main entry point for the refresh script.

    Args:
        cache_hours: Hours to cache FBref responses

    Returns:
        DataFrame with player info and workload metrics
    """
    scraper = FBrefScraper(cache_hours=cache_hours)

    # Get all players
    logger.info("Fetching all Premier League players...")
    players = scraper.get_all_premier_league_players()

    # Calculate workload for each player
    logger.info(f"Calculating workload for {len(players)} players...")

    workloads = []
    for i, player in players.iterrows():
        if i % 20 == 0:
            logger.info(f"Progress: {i}/{len(players)} players...")

        try:
            workload = scraper.get_player_workload(player["player_url"])
            workload["name"] = player["name"]
            workload["team"] = player["team"]
            workload["position"] = player["position"]
            workload["age"] = player["age"]
            workloads.append(workload)
        except Exception as e:
            logger.warning(f"Failed to get workload for {player['name']}: {e}")
            # Add with default values
            workloads.append({
                "name": player["name"],
                "team": player["team"],
                "position": player["position"],
                "age": player["age"],
                "acute_load": 0,
                "chronic_load": 0,
                "acwr": 1.0,
                "monotony": 1.0,
                "strain": 0,
                "fatigue_index": 0,
                "workload_slope": 0,
                "spike_flag": 0,
                "matches_last_7": 0,
                "matches_last_14": 0,
                "matches_last_30": 0,
                "minutes_last_7": 0,
                "minutes_last_14": 0,
                "minutes_last_30": 0,
                "total_minutes": 0,
                "total_matches": 0,
                "avg_minutes": 0,
            })

    df = pd.DataFrame(workloads)
    logger.info(f"Completed workload calculation for {len(df)} players")

    return df
