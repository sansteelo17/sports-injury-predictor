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
import re
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path=None, *args, **kwargs):
        """Lightweight .env loader fallback when python-dotenv is unavailable."""
        env_path = Path(path) if path else Path(".env")
        if not env_path.exists():
            return False
        try:
            for raw_line in env_path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            return True
        except Exception:
            return False

from ..utils.logger import get_logger

logger = get_logger(__name__)

# The Odds API - free tier available
ODDS_API_URL = "https://api.the-odds-api.com/v4"
API_FOOTBALL_URL = "https://v3.football.api-sports.io"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
load_dotenv(Path(__file__).parent.parent.parent / ".env")

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
        self.api_football_key = os.environ.get("API_FOOTBALL_KEY", "")
        self.odds_provider = os.environ.get("ODDS_PROVIDER", "auto").strip().lower()
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

    def get_upcoming_matches(self, force_odds_api: bool = False) -> List[Dict]:
        """
        Get upcoming Premier League matches with odds.

        Returns list of matches with:
        - home_team, away_team
        - commence_time
        - odds from various bookmakers
        """
        cache_key = "matches_forced_odds_api" if force_odds_api else "matches"
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not force_odds_api and "matches" in self._cache:
            return self._cache["matches"]

        if not force_odds_api and self.odds_provider == "api_football":
            logger.info("ODDS_PROVIDER=api_football - using API-Football/Polymarket direct mode")
            return self._cache.get("matches", [])

        if not self.api_key:
            if self.api_football_key:
                logger.info("No ODDS_API_KEY set - using API-Football/Polymarket direct mode")
                return self._cache.get("matches", [])
            logger.warning("No ODDS_API_KEY set - using mock data")
            return self._get_mock_matches()

        try:
            response = self.session.get(
                f"{ODDS_API_URL}/sports/{SPORT_KEY}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": "uk,us,eu",
                    "markets": "h2h",
                    "oddsFormat": "american",
                },
                timeout=10,
            )
            response.raise_for_status()
            matches = response.json()

            # Cache results
            if force_odds_api:
                self._cache["matches_forced_odds_api"] = matches
            else:
                self._cache["matches"] = matches
            self._save_cache()

            logger.info(f"Fetched odds for {len(matches)} upcoming matches")
            return matches

        except Exception as e:
            logger.error(f"Failed to fetch odds: {e}")
            if force_odds_api:
                return self._cache.get("matches_forced_odds_api", self._get_mock_matches())
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

    def get_team_next_match(self, team_name: str, force_odds_api: bool = False) -> Optional[Dict]:
        """
        Get a team's next match with odds.

        Args:
            team_name: Team name (partial match supported)

        Returns:
            Match dict with odds, or None
        """
        matches = self.get_upcoming_matches(force_odds_api=force_odds_api)
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
        market_snapshot = self.get_anytime_scorer_market_snapshot(team_name, player_name)
        if market_snapshot and market_snapshot.get("lines"):
            preferred_order = ["SkyBet", "Paddy Power", "Betway"]
            lines = market_snapshot["lines"]
            best_line = None
            for preferred in preferred_order:
                best_line = next((line for line in lines if line.get("bookmaker") == preferred), None)
                if best_line:
                    break
            if not best_line:
                best_line = lines[0]
            return {
                "bookmaker": best_line.get("bookmaker", "Unknown"),
                "decimal_odds": best_line.get("decimal_odds"),
                "implied_probability": best_line.get("implied_probability"),
                "player_matched": best_line.get("player_matched", player_name),
                "opponent": market_snapshot.get("opponent"),
                "is_home": market_snapshot.get("is_home"),
            }

        # Backward-compatible fallback if snapshot is unavailable
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
                    "regions": "uk,us,eu",
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

    @staticmethod
    def _bookmaker_aliases() -> Dict[str, List[str]]:
        """Canonical sportsbook names used in the UI and their known aliases."""
        return {
            "SportyBet": ["sportybet", "sportsbet", "sportingbet"],
            "PolyMarket": ["polymarket", "poly market"],
            "Bet365": ["bet365"],
            "SkyBet": ["sky bet", "skybet"],
            "Paddy Power": ["paddy power", "paddypower"],
            "Betway": ["betway"],
        }

    def _canonicalize_bookmaker(self, raw_name: str, raw_key: str = "") -> Optional[str]:
        """Map provider bookmaker labels to canonical names."""
        haystack = f"{raw_name} {raw_key}".lower()
        for canonical, aliases in self._bookmaker_aliases().items():
            if any(alias in haystack for alias in aliases):
                return canonical
        return None

    def get_anytime_scorer_market_snapshot(self, team_name: str, player_name: str) -> Optional[Dict]:
        """
        Get normalized scorer market lines for target bookies.

        Returns:
            {
                opponent, is_home,
                lines: [{bookmaker, decimal_odds, implied_probability, source, player_matched}],
                average_decimal_odds, average_probability
            }
        """
        cache_key = (
            f"scorer_market_snapshot_{self.odds_provider}_"
            f"{team_name.lower()}_{player_name.lower()}"
        )
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached is not None:
                return cached

        if not self.api_key:
            poly = self._get_polymarket_anytime_scorer_snapshot(team_name, player_name)
            self._cache[cache_key] = poly
            return poly

        try:
            response = self.session.get(
                f"{ODDS_API_URL}/sports/{SPORT_KEY}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": "uk,us,eu",
                    "markets": "player_goal_scorer_anytime",
                    "oddsFormat": "decimal",
                },
                timeout=10,
            )
            response.raise_for_status()
            matches = response.json()

            team_lower = team_name.lower()
            player_lower = player_name.lower()
            player_last = player_name.split()[-1].lower() if player_name else ""

            for match in matches:
                home = match.get("home_team", "").lower()
                away = match.get("away_team", "").lower()
                if team_lower not in home and team_lower not in away:
                    continue

                opponent = match.get("away_team") if team_lower in home else match.get("home_team")
                is_home = team_lower in home

                all_lines: Dict[str, Dict] = {}

                for bookie in match.get("bookmakers", []):
                    bookie_title = bookie.get("title", "Unknown")
                    bookie_key = bookie.get("key", "")
                    matched_outcome = None

                    for market in bookie.get("markets", []):
                        if market.get("key") != "player_goal_scorer_anytime":
                            continue
                        for outcome in market.get("outcomes", []):
                            outcome_name = outcome.get("name", "").lower()
                            if (
                                player_lower in outcome_name
                                or player_last in outcome_name
                                or outcome_name in player_lower
                            ):
                                matched_outcome = outcome
                                break
                        if matched_outcome:
                            break

                    if not matched_outcome:
                        continue

                    decimal_odds = matched_outcome.get("price", 0)
                    if not decimal_odds or decimal_odds <= 1:
                        continue

                    book_name = self._display_bookmaker_name(bookie_title, bookie_key)
                    if book_name in all_lines:
                        continue

                    line = {
                        "bookmaker": book_name,
                        "decimal_odds": round(float(decimal_odds), 2),
                        "implied_probability": round(1 / float(decimal_odds), 3),
                        "source": "Live Market",
                        "player_matched": matched_outcome.get("name", ""),
                    }
                    all_lines[book_name] = line

                target_books = ["SkyBet", "Paddy Power", "Betway"]
                preferred = [all_lines[b] for b in target_books if b in all_lines]
                others = [line for name, line in all_lines.items() if name not in target_books]
                merged_lines = (preferred + others)[:3]

                if not merged_lines:
                    # In API-Football mode, fall back to Polymarket only if no target lines exist.
                    if self.odds_provider == "api_football":
                        poly = self._get_polymarket_anytime_scorer_snapshot(team_name, player_name)
                        self._cache[cache_key] = poly
                        return poly
                    continue

                avg_decimal = round(sum(line["decimal_odds"] for line in merged_lines) / len(merged_lines), 2)
                avg_probability = round(sum(line["implied_probability"] for line in merged_lines) / len(merged_lines), 3)

                result = {
                    "opponent": opponent,
                    "is_home": is_home,
                    "lines": merged_lines,
                    "average_decimal_odds": avg_decimal,
                    "average_probability": avg_probability,
                }
                self._cache[cache_key] = result
                return result

            self._cache[cache_key] = None
            return None

        except Exception as e:
            logger.debug(f"Failed to fetch scorer market snapshot for {player_name}: {e}")
            return None

    @staticmethod
    def _parse_poly_jsonish(value):
        """Polymarket often returns arrays as JSON strings."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []

    def _get_polymarket_anytime_scorer_snapshot(self, team_name: str, player_name: str) -> Optional[Dict]:
        """
        Direct no-key fallback using Polymarket Gamma API.

        Looks for active events mentioning player + scorer language and extracts
        the YES price as implied scoring probability.
        """
        player_lower = (player_name or "").lower()
        player_last = player_name.split()[-1].lower() if player_name else ""
        team_lower = (team_name or "").lower()
        search_terms = [player_lower, player_last]

        try:
            response = self.session.get(
                f"{POLYMARKET_GAMMA_URL}/events",
                params={
                    "closed": "false",
                    "limit": 200,
                    "order": "id",
                    "ascending": "false",
                },
                timeout=10,
            )
            response.raise_for_status()
            events = response.json()

            for event in events:
                title = (event.get("title") or "").lower()
                if not title:
                    continue
                if not any(term and term in title for term in search_terms):
                    continue
                if "score" not in title and "scorer" not in title:
                    continue
                if team_lower and team_lower not in title:
                    continue

                for market in event.get("markets", []):
                    outcomes = self._parse_poly_jsonish(market.get("outcomes"))
                    outcome_prices = self._parse_poly_jsonish(market.get("outcomePrices"))
                    if not outcomes or not outcome_prices or len(outcomes) != len(outcome_prices):
                        continue

                    yes_price = None
                    for idx, outcome_label in enumerate(outcomes):
                        label = str(outcome_label).lower()
                        if label == "yes":
                            try:
                                yes_price = float(outcome_prices[idx])
                            except (ValueError, TypeError):
                                yes_price = None
                            break

                    if yes_price is None or yes_price <= 0 or yes_price >= 1:
                        continue

                    decimal = round(1 / yes_price, 2)
                    return {
                        "opponent": None,
                        "is_home": None,
                        "lines": [
                            {
                                "bookmaker": "PolyMarket",
                                "decimal_odds": decimal,
                                "implied_probability": round(yes_price, 3),
                                "source": "Live Market",
                                "player_matched": player_name,
                            }
                        ],
                        "average_decimal_odds": decimal,
                        "average_probability": round(yes_price, 3),
                    }
            return None
        except Exception as e:
            logger.debug(f"Failed to fetch Polymarket scorer snapshot for {player_name}: {e}")
            return None

    @staticmethod
    def _prob_to_decimal_str(prob: float) -> Optional[str]:
        """Convert a probability to decimal odds text."""
        if prob <= 0 or prob >= 1:
            return None
        return f"{(1.0 / prob):.2f}"

    def _parse_polymarket_event_teams(self, title: str) -> Optional[Dict[str, str]]:
        """Best-effort parser for 'Team A vs Team B' style event titles."""
        if not title:
            return None
        clean = re.sub(r"\s+", " ", title.strip())
        m = re.search(r"(.+?)\s+vs\.?\s+(.+)", clean, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"(.+?)\s+v\s+(.+)", clean, flags=re.IGNORECASE)
        if not m:
            return None
        left = m.group(1).strip(" -|")
        right = m.group(2).split("|")[0].split("-")[0].strip()
        if not left or not right:
            return None
        return {"home_team": left, "away_team": right}

    def _get_polymarket_team_moneyline_1x2(self, team_name: str) -> Optional[Dict]:
        """
        Best-effort team match moneyline from Polymarket event markets.

        Notes:
        - Polymarket often provides binary or 3-way structures depending on market.
        - We surface direct lines only; if draw is unavailable, draw remains "-".
        """
        team_lower = (team_name or "").lower()
        try:
            response = self.session.get(
                f"{POLYMARKET_GAMMA_URL}/events",
                params={"closed": "false", "limit": 250, "order": "id", "ascending": "false"},
                timeout=10,
            )
            response.raise_for_status()
            events = response.json()

            for event in events:
                title = event.get("title", "")
                title_lower = title.lower()
                if team_lower not in title_lower:
                    continue
                if "score" in title_lower or "scorer" in title_lower:
                    continue

                parsed_teams = self._parse_polymarket_event_teams(title) or {
                    "home_team": team_name,
                    "away_team": "Opponent",
                }
                home_team = parsed_teams["home_team"]
                away_team = parsed_teams["away_team"]
                home_prob = None
                draw_prob = None
                away_prob = None

                for market in event.get("markets", []):
                    outcomes = self._parse_poly_jsonish(market.get("outcomes"))
                    prices = self._parse_poly_jsonish(market.get("outcomePrices"))
                    if not outcomes or not prices or len(outcomes) != len(prices):
                        continue

                    for idx, label in enumerate(outcomes):
                        label_l = str(label).lower()
                        try:
                            p = float(prices[idx])
                        except (TypeError, ValueError):
                            continue
                        if p <= 0 or p >= 1:
                            continue
                        if "draw" in label_l:
                            draw_prob = p
                        elif home_team.lower() in label_l:
                            home_prob = p
                        elif away_team.lower() in label_l:
                            away_prob = p
                        elif team_lower in label_l:
                            if team_lower in home_team.lower():
                                home_prob = p
                            elif team_lower in away_team.lower():
                                away_prob = p

                if home_prob is None and away_prob is None:
                    continue

                books = [{
                    "bookmaker": "PolyMarket",
                    "home": self._prob_to_decimal_str(home_prob) or "-",
                    "draw": self._prob_to_decimal_str(draw_prob) or "-",
                    "away": self._prob_to_decimal_str(away_prob) or "-",
                    "source": "Live Market",
                }]

                return {
                    "home_team": home_team,
                    "away_team": away_team,
                    "is_home": team_lower in home_team.lower(),
                    "books": books,
                }
            return None
        except Exception as e:
            logger.debug(f"Failed to fetch Polymarket team moneyline for {team_name}: {e}")
            return None

    @staticmethod
    def _format_american(price) -> Optional[str]:
        """Format odds as an American-style string."""
        if price is None:
            return None
        try:
            val = float(price)
        except (TypeError, ValueError):
            return None
        rounded = int(round(val))
        if rounded > 0:
            return f"+{rounded}"
        return str(rounded)

    def _display_bookmaker_name(self, raw_name: str, raw_key: str = "") -> str:
        """Prefer canonical label for known books, otherwise keep provider name."""
        canonical = self._canonicalize_bookmaker(raw_name, raw_key)
        if canonical:
            return canonical
        return (raw_name or "Unknown").strip() or "Unknown"

    @staticmethod
    def _american_to_decimal_str(price) -> Optional[str]:
        """Convert American odds (or decimal passthrough) to decimal odds string."""
        if price is None:
            return None
        try:
            val = float(price)
        except (TypeError, ValueError):
            return None
        if val > 1 and val < 20:
            return f"{val:.2f}"
        if val <= -100:
            dec = 1.0 + (100.0 / abs(val))
            return f"{dec:.2f}"
        if val >= 100:
            dec = 1.0 + (val / 100.0)
            return f"{dec:.2f}"
        return None

    def _moneyline_target_books(self) -> List[str]:
        return ["SkyBet", "Paddy Power", "Betway"]

    def _extract_match_moneyline_books(self, match: Dict) -> List[Dict]:
        """Extract direct 1X2 decimal lines, preferring target bookmakers first."""
        home_team = match.get("home_team", "")
        away_team = match.get("away_team", "")
        preferred_by_name: Dict[str, Dict] = {}
        fallback_books: List[Dict] = []
        seen = set()
        target_order = self._moneyline_target_books()
        target_books = set(target_order)

        for bookie in match.get("bookmakers", []):
            title = bookie.get("title", "Unknown")
            key = bookie.get("key", "")
            h2h_market = None
            for market in bookie.get("markets", []):
                if market.get("key") == "h2h":
                    h2h_market = market
                    break
            if not h2h_market:
                continue

            home_price = None
            away_price = None
            draw_price = None
            for outcome in h2h_market.get("outcomes", []):
                name = outcome.get("name", "").lower()
                price = outcome.get("price")
                if not name:
                    continue
                if home_team.lower() in name:
                    home_price = self._american_to_decimal_str(price)
                elif away_team.lower() in name:
                    away_price = self._american_to_decimal_str(price)
                elif "draw" in name:
                    draw_price = self._american_to_decimal_str(price)

            if not home_price or not away_price:
                continue

            book_name = self._display_bookmaker_name(title, key)
            if book_name in seen:
                continue
            seen.add(book_name)
            source_label = "Mock Data" if "mock" in title.lower() else "Live Market"
            row = {
                "bookmaker": book_name,
                "home": home_price,
                "draw": draw_price or "-",
                "away": away_price,
                "source": source_label,
            }

            if book_name in target_books:
                preferred_by_name[book_name] = row
            else:
                fallback_books.append(row)

        ordered_preferred = [preferred_by_name[name] for name in target_order if name in preferred_by_name]
        if len(ordered_preferred) < 3:
            ordered_preferred.extend(fallback_books[: max(0, 3 - len(ordered_preferred))])
        return ordered_preferred[:3]

    def _build_moneyline_result_from_match(self, team_name: str, match: Dict) -> Optional[Dict]:
        """Build normalized moneyline result from an Odds API match payload."""
        if not match:
            return None
        home_team = match.get("home_team", "")
        away_team = match.get("away_team", "")
        team_lower = (team_name or "").lower()
        books = self._extract_match_moneyline_books(match)
        if not books:
            return None
        return {
            "home_team": home_team,
            "away_team": away_team,
            "is_home": team_lower in home_team.lower(),
            "books": books,
        }

    def get_team_moneyline_1x2(self, team_name: str) -> Optional[Dict]:
        """
        Get 1X2 moneyline rows for a team's next match.

        Returns:
            {
                "home_team": "...",
                "away_team": "...",
                "is_home": bool,
                "books": [
                    {"bookmaker": "SportyBet", "home": "-120", "draw": "+250", "away": "+310", "source": "..."}
                ]
            }
        """
        cache_key = f"moneyline_1x2_{self.odds_provider}_{team_name.lower()}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached is not None:
                return cached

        # Explicit provider override.
        if self.odds_provider == "api_football":
            api_football_data = self._get_team_moneyline_1x2_api_football(team_name)
            if not api_football_data and self.api_key:
                api_football_data = self._build_moneyline_result_from_match(
                    team_name,
                    self.get_team_next_match(team_name, force_odds_api=True),
                )
            self._cache[cache_key] = api_football_data
            return api_football_data
        if self.odds_provider == "odds_api":
            result = self._build_moneyline_result_from_match(
                team_name,
                self.get_team_next_match(team_name, force_odds_api=True),
            )
            if not result:
                self._cache[cache_key] = None
                return None
            self._cache[cache_key] = result
            return result

        # API-Football direct feed (option 4) when key exists and Odds API key is absent.
        if not self.api_key and self.api_football_key:
            api_football_data = self._get_team_moneyline_1x2_api_football(team_name)
            self._cache[cache_key] = api_football_data
            return api_football_data

        result = self._build_moneyline_result_from_match(
            team_name,
            self.get_team_next_match(team_name),
        )
        if not result:
            self._cache[cache_key] = None
            return None

        self._cache[cache_key] = result
        return result

    def _api_football_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Minimal API-Football v3 client."""
        if not self.api_football_key:
            return None
        try:
            response = self.session.get(
                f"{API_FOOTBALL_URL}/{endpoint.lstrip('/')}",
                params=params,
                headers={"x-apisports-key": self.api_football_key},
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"API-Football request failed for {endpoint}: {e}")
            return None

    def _get_team_moneyline_1x2_api_football(self, team_name: str) -> Optional[Dict]:
        """
        Get direct 1X2 lines from API-Football (no synthesis).
        """
        fixtures_payload = self._api_football_request(
            "fixtures",
            {"league": 39, "next": 20},  # EPL
        )
        if not fixtures_payload:
            return None

        fixtures = fixtures_payload.get("response", [])
        if not fixtures:
            return None

        team_lower = (team_name or "").lower()
        fixture = None
        for item in fixtures:
            home_name = item.get("teams", {}).get("home", {}).get("name", "")
            away_name = item.get("teams", {}).get("away", {}).get("name", "")
            home_lower = home_name.lower()
            away_lower = away_name.lower()
            if team_lower in home_lower or team_lower in away_lower:
                fixture = item
                break

        if not fixture:
            return None

        fixture_id = fixture.get("fixture", {}).get("id")
        if not fixture_id:
            return None

        odds_payload = self._api_football_request("odds", {"fixture": fixture_id})
        if not odds_payload:
            return None

        response_rows = odds_payload.get("response", [])
        if not response_rows:
            return None

        target_order = self._moneyline_target_books()
        target_books = set(target_order)
        preferred_by_name: Dict[str, Dict] = {}
        fallback_lines: List[Dict] = []
        seen_books = set()
        for row in response_rows:
            for bookmaker in row.get("bookmakers", []):
                raw_name = bookmaker.get("name", "")
                book_name = self._display_bookmaker_name(raw_name, "")
                if book_name in seen_books:
                    continue

                match_winner = None
                for bet in bookmaker.get("bets", []):
                    bet_name = (bet.get("name") or "").lower()
                    if "match winner" in bet_name or bet_name == "1x2":
                        match_winner = bet
                        break
                if not match_winner:
                    continue

                home_val = "-"
                draw_val = "-"
                away_val = "-"
                for value in match_winner.get("values", []):
                    key = (value.get("value") or "").lower()
                    odd = value.get("odd")
                    odd_text = f"{float(odd):.2f}" if odd is not None else "-"
                    if key in ("home", "1"):
                        home_val = odd_text
                    elif key in ("draw", "x"):
                        draw_val = odd_text
                    elif key in ("away", "2"):
                        away_val = odd_text

                if home_val == "-" and away_val == "-":
                    continue

                seen_books.add(book_name)
                line = {
                    "bookmaker": book_name,
                    "home": home_val,
                    "draw": draw_val,
                    "away": away_val,
                    "source": "Live Market",
                }
                if book_name in target_books:
                    preferred_by_name[book_name] = line
                else:
                    fallback_lines.append(line)

        books = [preferred_by_name[name] for name in target_order if name in preferred_by_name]
        if len(books) < 3:
            books.extend(fallback_lines[: max(0, 3 - len(books))])
        books = books[:3]
        if not books:
            return None

        home_name = fixture.get("teams", {}).get("home", {}).get("name", "")
        away_name = fixture.get("teams", {}).get("away", {}).get("name", "")
        return {
            "home_team": home_name,
            "away_team": away_name,
            "is_home": team_lower in home_name.lower(),
            "books": books,
        }

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
        return (
            f"Clean Sheet Alert: {team_name} are {american} for a CS {venue} vs {opponent}. "
            "With low injury risk, this is a solid defensive pick."
        )

    # High CS odds + High injury risk = Risky
    if cs_prob >= 0.35 and injury_prob >= 0.30:
        return (
            f"{team_name} have good CS odds ({american}) vs {opponent}, but this player's elevated injury "
            "risk means they might not be on the pitch to collect those points."
        )

    # Low CS odds (tough fixture)
    if cs_prob < 0.25:
        return (
            f"Tough fixture: {team_name} face {opponent} - clean sheet unlikely ({american}). "
            "Consider bench fodder for this week."
        )

    # Moderate situation
    if cs_prob >= 0.25:
        return (
            f"{team_name} are {american} for a CS vs {opponent}. "
            + ("Low injury risk makes this viable." if injury_prob < 0.20 else "Monitor fitness before committing.")
        )

    return None
