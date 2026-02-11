"""
Transfermarkt injury history scraper.

Fetches player injury records from Transfermarkt to supplement the local injury CSV.
Useful for players/teams not covered in the original dataset.

Usage:
    from src.data_loaders.transfermarkt_scraper import fetch_player_injuries

    injuries = fetch_player_injuries("erling-haaland", "418560")
    # Returns: [{'injury': 'Muscle injury', 'date': '2024-01-15', 'days_out': 21}, ...]
"""

import re
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

import requests
from bs4 import BeautifulSoup
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Configuration
BASE_URL = "https://www.transfermarkt.com"
RATE_LIMIT_DELAY = 3.0  # seconds between requests
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "transfermarkt"


class TransfermarktScraper:
    """
    Scrapes injury history from Transfermarkt.

    Respects rate limits and caches responses to minimize requests.
    """

    def __init__(self, cache_hours: int = 168):  # 1 week cache
        """
        Initialize the scraper.

        Args:
            cache_hours: Hours to cache responses (default 168 = 1 week)
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })
        self._last_request_time = 0
        self.cache_hours = cache_hours
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _cache_key(self, url: str) -> Path:
        """Generate cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
        return CACHE_DIR / f"{url_hash}.html"

    def _get_cached(self, url: str) -> Optional[str]:
        """Get cached response if still valid."""
        cache_file = self._cache_key(url)
        if cache_file.exists():
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mtime < timedelta(hours=self.cache_hours):
                return cache_file.read_text(encoding="utf-8")
        return None

    def _save_cache(self, url: str, content: str):
        """Save response to cache."""
        cache_file = self._cache_key(url)
        cache_file.write_text(content, encoding="utf-8")

    def _fetch(self, url: str) -> str:
        """Fetch URL with caching and rate limiting."""
        cached = self._get_cached(url)
        if cached:
            logger.debug(f"Cache hit: {url}")
            return cached

        self._rate_limit()
        logger.debug(f"Fetching: {url}")

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            self._save_cache(url, response.text)
            return response.text
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            raise

    def search_player(self, name: str) -> Optional[Dict]:
        """
        Search for a player by name.

        Returns:
            Dict with 'name', 'slug', 'player_id', 'team' or None if not found
        """
        search_url = f"{BASE_URL}/schnellsuche/ergebnis/schnellsuche?query={name.replace(' ', '+')}"

        try:
            html = self._fetch(search_url)
            soup = BeautifulSoup(html, "html.parser")

            # Find first player result
            player_row = soup.select_one("table.items tbody tr")
            if not player_row:
                return None

            # Extract player link
            link = player_row.select_one("td.hauptlink a")
            if not link:
                return None

            href = link.get("href", "")
            # Format: /player-name/profil/spieler/12345
            match = re.search(r"/([^/]+)/profil/spieler/(\d+)", href)
            if not match:
                return None

            slug, player_id = match.groups()

            # Get team
            team_cell = player_row.select("td.zentriert")
            team = ""
            for cell in team_cell:
                img = cell.select_one("img[alt]")
                if img and "title" in img.attrs:
                    team = img["title"]
                    break

            return {
                "name": link.get_text(strip=True),
                "slug": slug,
                "player_id": player_id,
                "team": team
            }
        except Exception as e:
            logger.warning(f"Search failed for '{name}': {e}")
            return None

    def get_injury_history(self, slug: str, player_id: str) -> List[Dict]:
        """
        Get injury history for a player.

        Args:
            slug: Player URL slug (e.g., "erling-haaland")
            player_id: Transfermarkt player ID (e.g., "418560")

        Returns:
            List of injury records with keys: injury, date, days_out, games_missed
        """
        url = f"{BASE_URL}/{slug}/verletzungen/spieler/{player_id}"

        try:
            html = self._fetch(url)
            soup = BeautifulSoup(html, "html.parser")

            injuries = []
            table = soup.select_one("table.items")
            if not table:
                return []

            rows = table.select("tbody tr")
            for row in rows:
                cells = row.select("td")
                if len(cells) < 5:
                    continue

                try:
                    # Parse injury type
                    injury_type = cells[1].get_text(strip=True)

                    # Parse dates (format varies: "Jan 15, 2024" or "15/01/2024")
                    date_from = cells[2].get_text(strip=True)
                    date_to = cells[3].get_text(strip=True)

                    # Parse days out
                    days_text = cells[4].get_text(strip=True)
                    days_match = re.search(r"(\d+)", days_text)
                    days_out = int(days_match.group(1)) if days_match else 0

                    # Parse games missed
                    games_text = cells[5].get_text(strip=True) if len(cells) > 5 else "0"
                    games_match = re.search(r"(\d+)", games_text)
                    games_missed = int(games_match.group(1)) if games_match else 0

                    # Parse date
                    injury_date = self._parse_date(date_from)

                    injuries.append({
                        "injury": injury_type,
                        "date": injury_date.strftime("%Y-%m-%d") if injury_date else date_from,
                        "date_to": date_to,
                        "days_out": days_out,
                        "games_missed": games_missed
                    })
                except Exception as e:
                    logger.debug(f"Failed to parse injury row: {e}")
                    continue

            return injuries
        except Exception as e:
            logger.warning(f"Failed to get injuries for {slug}: {e}")
            return []

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string in various formats."""
        formats = [
            "%b %d, %Y",  # Jan 15, 2024
            "%d/%m/%Y",   # 15/01/2024
            "%Y-%m-%d",   # 2024-01-15
            "%d.%m.%Y",   # 15.01.2024
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    def get_player_stats(self, slug: str, player_id: str, season: str = "2024") -> Dict:
        """
        Get player's season statistics including minutes played.

        Args:
            slug: Player URL slug
            player_id: Transfermarkt player ID
            season: Season year (e.g., "2024" for 2024/25)

        Returns:
            Dict with: minutes_played, appearances, goals, assists
        """
        url = f"{BASE_URL}/{slug}/leistungsdaten/spieler/{player_id}/saison/{season}/plus/1"

        try:
            html = self._fetch(url)
            soup = BeautifulSoup(html, "html.parser")

            stats = {
                "minutes_played": 0,
                "appearances": 0,
                "goals": 0,
                "assists": 0,
                "is_starter": False,
            }

            # Find the stats table (tfoot has totals)
            footer = soup.select_one("table.items tfoot tr")
            if footer:
                cells = footer.select("td")
                # Typical columns: Appearances, Goals, Assists, Yellow, Red, Minutes
                for i, cell in enumerate(cells):
                    text = cell.get_text(strip=True).replace(".", "").replace(",", "").replace("'", "")

                    # Minutes is usually the last numeric column with high values
                    if text.isdigit():
                        val = int(text)
                        if val > 100:  # Likely minutes
                            stats["minutes_played"] = val
                        elif val < 50 and i < 3:  # Early columns: appearances
                            if stats["appearances"] == 0:
                                stats["appearances"] = val

            # Alternative: look for specific labeled rows
            info_items = soup.select("li.data-header__label")
            for item in info_items:
                label = item.get_text(strip=True).lower()
                value_span = item.select_one("span.data-header__content")
                if value_span:
                    value = value_span.get_text(strip=True).replace(".", "").replace(",", "").replace("'", "")
                    if "minute" in label and value.isdigit():
                        stats["minutes_played"] = int(value)
                    elif "appearance" in label and value.isdigit():
                        stats["appearances"] = int(value)

            # Parse from the performance data box if available
            perf_box = soup.select("div.data-header__box--small span")
            for span in perf_box:
                text = span.get_text(strip=True)
                if "'" in text:  # Minutes often shown as "1,234'"
                    minutes_text = text.replace("'", "").replace(".", "").replace(",", "")
                    if minutes_text.isdigit():
                        stats["minutes_played"] = int(minutes_text)

            # Determine if player is a regular starter (>1000 minutes = ~11 full games)
            stats["is_starter"] = stats["minutes_played"] >= 900

            return stats
        except Exception as e:
            logger.debug(f"Failed to get stats for {slug}: {e}")
            return {
                "minutes_played": 0,
                "appearances": 0,
                "goals": 0,
                "assists": 0,
                "is_starter": False,
            }

    def get_current_team(self, player_name: str) -> Optional[str]:
        """
        Get player's current team from Transfermarkt.

        This accounts for loans and recent transfers.

        Args:
            player_name: Player's full name

        Returns:
            Current team name or None if not found
        """
        player = self.search_player(player_name)
        if player:
            return player.get("team")
        return None

    def get_team_squad(self, team_slug: str, team_id: str) -> List[Dict]:
        """
        Get all players in a team's squad.

        Args:
            team_slug: Team URL slug (e.g., "fc-sunderland")
            team_id: Transfermarkt team ID (e.g., "289")

        Returns:
            List of player dicts with name, slug, player_id, position, age
        """
        url = f"{BASE_URL}/{team_slug}/kader/verein/{team_id}/saison_id/2025/plus/1"

        try:
            html = self._fetch(url)
            soup = BeautifulSoup(html, "html.parser")

            players = []
            # Find player rows in the squad table
            rows = soup.select("table.items tbody tr.odd, table.items tbody tr.even")

            for row in rows:
                try:
                    # Get player link
                    player_link = row.select_one("td.hauptlink a")
                    if not player_link:
                        continue

                    href = player_link.get("href", "")
                    match = re.search(r"/([^/]+)/profil/spieler/(\d+)", href)
                    if not match:
                        continue

                    slug, player_id = match.groups()
                    name = player_link.get_text(strip=True)

                    # Get position
                    position_cell = row.select_one("td.posrela table tr:last-child td")
                    position = position_cell.get_text(strip=True) if position_cell else "Unknown"

                    # Get age from the row
                    age = None
                    cells = row.select("td.zentriert")
                    for cell in cells:
                        text = cell.get_text(strip=True)
                        if text.isdigit() and 15 <= int(text) <= 45:
                            age = int(text)
                            break

                    players.append({
                        "name": name,
                        "slug": slug,
                        "player_id": player_id,
                        "position": position,
                        "age": age,
                    })
                except Exception as e:
                    logger.debug(f"Failed to parse player row: {e}")
                    continue

            logger.info(f"Found {len(players)} players in squad")
            return players
        except Exception as e:
            logger.warning(f"Failed to get squad for {team_slug}: {e}")
            return []

    def get_player_age(self, slug: str, player_id: str) -> Optional[int]:
        """
        Get player's age from their profile page.

        Args:
            slug: Player URL slug
            player_id: Transfermarkt player ID

        Returns:
            Age as integer, or None if not found
        """
        url = f"{BASE_URL}/{slug}/profil/spieler/{player_id}"

        try:
            html = self._fetch(url)
            soup = BeautifulSoup(html, "html.parser")

            # Method 1: Look for age in info-table (format: "DD/MM/YYYY (age)")
            info_items = soup.select(".info-table__content--bold")
            for item in info_items:
                text = item.get_text(strip=True)
                # Match "DD/MM/YYYY (29)" or similar date with age in parens
                age_match = re.search(r"\((\d{1,2})\)$", text)
                if age_match:
                    age = int(age_match.group(1))
                    if 15 <= age <= 45:
                        return age

            # Method 2: Look for standalone age number
            info_table = soup.select("span.info-table__content")
            for span in info_table:
                text = span.get_text(strip=True)
                if text.isdigit():
                    age = int(text)
                    if 15 <= age <= 45:
                        return age

            # Method 3: Look in header data
            header = soup.select_one("[data-header-datum]")
            if header:
                text = header.get_text()
                age_match = re.search(r"(\d{1,2})\s*years?", text, re.IGNORECASE)
                if age_match:
                    return int(age_match.group(1))

            # Method 4: Parse from birth date
            birth_span = soup.select_one("span[itemprop='birthDate']")
            if birth_span:
                birth_text = birth_span.get_text(strip=True)
                birth_date = self._parse_date(birth_text)
                if birth_date:
                    age = (datetime.now() - birth_date).days // 365
                    if 15 <= age <= 45:
                        return age

            return None
        except Exception as e:
            logger.debug(f"Failed to get age for {slug}: {e}")
            return None

    def fetch_player_injuries(self, player_name: str, include_stats: bool = True) -> Optional[Dict]:
        """
        Fetch complete injury history for a player by name.

        Args:
            player_name: Full player name (e.g., "Erling Haaland")
            include_stats: Whether to also fetch season stats (minutes played)

        Returns:
            Dict with:
                - name: Player name as found
                - injuries: List of injury records
                - total_injuries: Count
                - total_days_out: Sum of days missed
                - last_injury_date: Most recent injury date
                - days_since_last: Days since last injury
                - age: Player's current age
                - minutes_played: Season minutes (if include_stats=True)
                - is_starter: Whether player is a regular starter
        """
        # Search for player
        player = self.search_player(player_name)
        if not player:
            logger.debug(f"Player not found: {player_name}")
            return None

        # Get player age from profile page
        age = self.get_player_age(player["slug"], player["player_id"])

        # Get season stats (minutes played)
        stats = {}
        if include_stats:
            stats = self.get_player_stats(player["slug"], player["player_id"])

        # Get injury history
        injuries = self.get_injury_history(player["slug"], player["player_id"])

        if not injuries:
            return {
                "name": player["name"],
                "team": player["team"],
                "age": age,
                "injuries": [],
                "total_injuries": 0,
                "total_days_out": 0,
                "last_injury_date": None,
                "days_since_last": 365 * 5,  # 5 years default for "never injured"
                "minutes_played": stats.get("minutes_played", 0),
                "appearances": stats.get("appearances", 0),
                "is_starter": stats.get("is_starter", False),
            }

        # Calculate stats
        total_days = sum(i["days_out"] for i in injuries)

        # Find most recent injury
        last_date = None
        for injury in injuries:
            try:
                d = datetime.strptime(injury["date"], "%Y-%m-%d")
                if last_date is None or d > last_date:
                    last_date = d
            except ValueError:
                continue

        days_since = (datetime.now() - last_date).days if last_date else 365 * 5

        return {
            "name": player["name"],
            "team": player["team"],
            "age": age,
            "injuries": injuries,
            "total_injuries": len(injuries),
            "total_days_out": total_days,
            "last_injury_date": last_date.strftime("%Y-%m-%d") if last_date else None,
            "days_since_last": max(0, days_since),
            "minutes_played": stats.get("minutes_played", 0),
            "appearances": stats.get("appearances", 0),
            "is_starter": stats.get("is_starter", False),
        }


def fetch_injuries_for_players(player_names: List[str], progress_callback=None) -> Dict[str, Dict]:
    """
    Fetch injury history for multiple players.

    Args:
        player_names: List of player names
        progress_callback: Optional callback(current, total, name) for progress updates

    Returns:
        Dict mapping player name -> injury data
    """
    scraper = TransfermarktScraper()
    results = {}

    for i, name in enumerate(player_names):
        if progress_callback:
            progress_callback(i + 1, len(player_names), name)

        data = scraper.fetch_player_injuries(name)
        if data:
            results[name] = data

    return results


def build_injury_lookup(player_names: List[str]) -> Dict[str, Dict]:
    """
    Build injury history lookup dict for model inference.

    Args:
        player_names: List of player names to look up

    Returns:
        Dict mapping player name -> {
            'previous_injuries': int,
            'days_since_last_injury': int,
            'total_days_lost': int
        }
    """
    raw_data = fetch_injuries_for_players(player_names)

    lookup = {}
    for name, data in raw_data.items():
        lookup[name] = {
            "previous_injuries": data["total_injuries"],
            "days_since_last_injury": data["days_since_last"],
            "total_days_lost": data["total_days_out"],
            "last_injury_date": data["last_injury_date"]
        }

    return lookup


def update_players_from_transfermarkt(
    df: pd.DataFrame,
    player_col: str = "name",
    team_col: str = "team",
    age_col: str = "age",
    update_teams: bool = True,
    update_ages: bool = True,
    progress_callback=None
) -> pd.DataFrame:
    """
    Update player data (team, age) from Transfermarkt.

    This accounts for loans, transfers, and current ages.

    Args:
        df: DataFrame with player names
        player_col: Column containing player names
        team_col: Column to update with current team
        age_col: Column to update with current age
        update_teams: Whether to update teams
        update_ages: Whether to update ages
        progress_callback: Optional callback(current, total, name) for progress updates

    Returns:
        DataFrame with updated columns
    """
    scraper = TransfermarktScraper()
    df = df.copy()

    unique_players = df[player_col].unique()
    updates = {}

    for i, name in enumerate(unique_players):
        if progress_callback:
            progress_callback(i + 1, len(unique_players), name)

        # Use fetch_player_injuries which gets both team and age
        data = scraper.fetch_player_injuries(name, include_stats=False)
        if data:
            updates[name] = {
                "team": data.get("team"),
                "age": data.get("age"),
            }

    # Apply updates
    for player, info in updates.items():
        if update_teams and info.get("team"):
            df.loc[df[player_col] == player, team_col] = info["team"]
        if update_ages and info.get("age"):
            df.loc[df[player_col] == player, age_col] = info["age"]

    logger.info(f"Updated {len(updates)}/{len(unique_players)} players from Transfermarkt")
    return df


def update_teams_from_transfermarkt(
    df: pd.DataFrame,
    player_col: str = "player",
    team_col: str = "team",
    progress_callback=None
) -> pd.DataFrame:
    """
    Update team assignments in a DataFrame using current Transfermarkt data.

    This accounts for loans and recent transfers.

    Args:
        df: DataFrame with player names
        player_col: Column containing player names
        team_col: Column to update with current team
        progress_callback: Optional callback(current, total, name) for progress updates

    Returns:
        DataFrame with updated team column
    """
    scraper = TransfermarktScraper()
    df = df.copy()

    unique_players = df[player_col].unique()
    team_updates = {}

    for i, name in enumerate(unique_players):
        if progress_callback:
            progress_callback(i + 1, len(unique_players), name)

        current_team = scraper.get_current_team(name)
        if current_team:
            team_updates[name] = current_team
            logger.info(f"Updated team for {name}: {current_team}")

    # Apply updates
    for player, team in team_updates.items():
        df.loc[df[player_col] == player, team_col] = team

    logger.info(f"Updated teams for {len(team_updates)}/{len(unique_players)} players")
    return df
