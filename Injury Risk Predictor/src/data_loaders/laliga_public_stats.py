"""
Public LaLiga stats loader.

Uses official public LaLiga leaderboard pages (no auth required) to enrich
La Liga players with current-season production and minutes. FBref can then be
used as the fallback for broader minutes coverage.
"""

from __future__ import annotations

import io
import json
import re
import time
import unicodedata
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..utils.logger import get_logger

logger = get_logger(__name__)

BASE_URL = "https://www.laliga.com"
LEADERBOARD_URL = (
    f"{BASE_URL}/en-US/leaderboard/all-leaders"
    "?stat={stat}&stat_competition=laliga-easports"
)
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "laliga_public"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_SERVICE_FALLBACK_URL = "https://apim.laliga.com/public-service"
PUBLIC_SERVICE_SUBSCRIPTION_FALLBACK = "c13c3a8e2f6b46da9c5c425cf61fab3e"
PUBLIC_SERVICE_COMPETITION = "laliga-easports"
PLAYER_RANKINGS_PAGE_SIZE = 100
MIN_USABLE_CACHE_ROWS = 200

STAT_CONFIG: Dict[str, Dict[str, str]] = {
    "minutes_played": {"stat": "total_mins_played_ranking", "label": "MINUTES PLAYED"},
    "goals": {"stat": "total_goals_ranking", "label": "GOALS"},
    "assists": {"stat": "total_assists_ranking", "label": "ASSISTS"},
    "shots": {"stat": "total_shots_ranking", "label": "SHOTS"},
    "saves": {"stat": "total_saves_ranking", "label": "SAVES"},
    "yellow_cards": {"stat": "total_yellow_cards_ranking", "label": "YELLOW CARDS"},
    "red_cards": {"stat": "total_red_cards_ranking", "label": "RED CARDS"},
}

TEAM_CODE_MAP: Dict[str, str] = {
    "ALA": "Alaves",
    "ATH": "Athletic Club",
    "ATM": "Atletico Madrid",
    "BAR": "Barcelona",
    "BET": "Real Betis",
    "CEL": "Celta Vigo",
    "ELC": "Elche",
    "ESP": "Espanyol",
    "GET": "Getafe",
    "GIR": "Girona",
    "LEG": "Leganes",
    "LEV": "Levante",
    "LPA": "Las Palmas",
    "MLL": "Mallorca",
    "OSA": "Osasuna",
    "OVI": "Real Oviedo",
    "RAY": "Rayo Vallecano",
    "RMA": "Real Madrid",
    "RSO": "Real Sociedad",
    "SEV": "Sevilla",
    "VAL": "Valencia",
    "VIL": "Villarreal",
    "VLL": "Valladolid",
    "CAD": "Cadiz",
    "GRA": "Granada",
    "ALM": "Almeria",
}

TEAM_ALIAS_MAP: Dict[str, str] = {
    "athletic": "Athletic Club",
    "athletic club": "Athletic Club",
    "athletic bilbao": "Athletic Club",
    "atleti": "Atletico Madrid",
    "atletico": "Atletico Madrid",
    "atletico de madrid": "Atletico Madrid",
    "club atletico de madrid": "Atletico Madrid",
    "barca": "Barcelona",
    "fc barcelona": "Barcelona",
    "real betis": "Real Betis",
    "betis": "Real Betis",
    "rc celta de vigo": "Celta Vigo",
    "celta": "Celta Vigo",
    "celta vigo": "Celta Vigo",
    "deportivo alaves": "Alaves",
    "alaves": "Alaves",
    "cd leganes": "Leganes",
    "leganes": "Leganes",
    "ud almeria": "Almeria",
    "almeria": "Almeria",
    "cadiz cf": "Cadiz",
    "cadiz": "Cadiz",
    "sevilla fc": "Sevilla",
    "real sociedad de futbol": "Real Sociedad",
    "rayo vallecano de madrid": "Rayo Vallecano",
    "rcd espanyol de barcelona": "Espanyol",
    "rcd mallorca": "Mallorca",
    "real madrid cf": "Real Madrid",
    "villarreal cf": "Villarreal",
    "valencia cf": "Valencia",
    "ca osasuna": "Osasuna",
    "getafe cf": "Getafe",
    "girona fc": "Girona",
}

TOKEN_ALIASES: Dict[str, str] = {
    "jr": "junior",
    "jnr": "junior",
    "sr": "senior",
}

PLAYER_ALIASES: Dict[str, str] = {
    "vini jr": "vinicius junior",
    "vini": "vinicius junior",
    "fede": "federico valverde",
    "giuliano": "giuliano simeone",
    "barrene": "ander barrenetxea",
    "a moleiro": "alberto moleiro",
    "l milla": "luis milla",
    "s cardona": "sergi cardona",
    "l rioja": "luis rioja",
}


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_team_name(name: str) -> str:
    key = _normalize_text(name)
    if not key:
        return ""
    if key.upper() in TEAM_CODE_MAP:
        return TEAM_CODE_MAP[key.upper()]
    mapped = TEAM_ALIAS_MAP.get(key)
    if mapped:
        return mapped
    if len(key) <= 4 and key.upper() in TEAM_CODE_MAP:
        return TEAM_CODE_MAP[key.upper()]
    return " ".join(part.capitalize() for part in key.split())


def normalize_player_name(name: str) -> str:
    key = _normalize_text(name)
    if not key:
        return ""
    key = PLAYER_ALIASES.get(key, key)
    tokens = [TOKEN_ALIASES.get(tok, tok) for tok in key.split()]
    return " ".join(tokens)


def _player_tokens(name: str) -> List[str]:
    return [tok for tok in normalize_player_name(name).split() if tok]


def _looks_like_team_code(value: str) -> bool:
    raw = str(value or "").strip().upper()
    return raw in TEAM_CODE_MAP or (2 <= len(raw) <= 4 and raw.isalpha())


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip().replace(",", "")
        if not text:
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _cache_age_hours(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    delta = datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime)
    return delta.total_seconds() / 3600


def _parse_next_data(html: str) -> Dict[str, Any]:
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def _score_player_match(target_name: str, candidate_name: str) -> float:
    target = normalize_player_name(target_name)
    candidate = normalize_player_name(candidate_name)
    if not target or not candidate:
        return 0.0
    if target == candidate:
        return 1.0

    target_tokens = _player_tokens(target)
    candidate_tokens = _player_tokens(candidate)
    if not target_tokens or not candidate_tokens:
        return 0.0

    score = SequenceMatcher(None, target, candidate).ratio() * 0.45

    target_last = target_tokens[-1]
    candidate_last = candidate_tokens[-1]
    if target_last == candidate_last:
        score += 0.28
    elif target_last.startswith(candidate_last) or candidate_last.startswith(target_last):
        if min(len(target_last), len(candidate_last)) >= 4:
            score += 0.18

    target_first = target_tokens[0]
    candidate_first = candidate_tokens[0]
    if target_first == candidate_first:
        score += 0.18
    elif target_first[:1] == candidate_first[:1]:
        score += 0.12

    overlap = 0
    for token in target_tokens:
        for other in candidate_tokens:
            if token == other or token.startswith(other) or other.startswith(token):
                overlap += 1
                break
    score += min(overlap / max(len(target_tokens), len(candidate_tokens)), 1.0) * 0.25

    if len(candidate_tokens) == 1 and (
        candidate_tokens[0] in target_tokens
        or any(tok.startswith(candidate_tokens[0]) for tok in target_tokens)
    ):
        score += 0.12
    if len(target_tokens) == 1 and (
        target_tokens[0] in candidate_tokens
        or any(tok.startswith(target_tokens[0]) for tok in candidate_tokens)
    ):
        score += 0.12

    return min(score, 1.0)


class LaLigaPublicStatsLoader:
    def __init__(self, cache_hours: int = 6):
        self.cache_hours = cache_hours
        self._runtime_config_cache: Dict[int, Dict[str, str]] = {}
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }
        )

    def _season_key(self) -> int:
        now = datetime.utcnow()
        return now.year if now.month >= 8 else now.year - 1

    def _html_cache_path(self, field_name: str, season: int) -> Path:
        return CACHE_DIR / f"{field_name}_{season}.html"

    def _merged_cache_path(self, season: int) -> Path:
        return CACHE_DIR / f"laliga_public_stats_{season}.csv"

    def _rankings_cache_path(self, season: int, offset: int) -> Path:
        return CACHE_DIR / f"players_rankings_{season}_{offset}.json"

    def _cache_fresh(self, path: Path) -> bool:
        age_hours = _cache_age_hours(path)
        return age_hours is not None and age_hours < self.cache_hours

    def _load_cached_merge(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists() or not self._cache_fresh(path):
            return None
        try:
            cached = pd.read_csv(path)
        except Exception:
            return None
        if len(cached) < MIN_USABLE_CACHE_ROWS:
            logger.info(
                "Ignoring undersized cached LaLiga merge (%d rows) at %s",
                len(cached),
                path,
            )
            return None
        required_columns = {"name", "team", "normalized_name", "normalized_team", "normalized_nickname"}
        if not required_columns.issubset(set(cached.columns)):
            logger.info(
                "Ignoring outdated cached LaLiga merge missing columns %s at %s",
                ",".join(sorted(required_columns - set(cached.columns))),
                path,
            )
            return None
        return cached

    def _fetch_html(self, field_name: str, season: int, force_refresh: bool = False) -> str:
        cache_path = self._html_cache_path(field_name, season)
        if not force_refresh and self._cache_fresh(cache_path):
            return cache_path.read_text(encoding="utf-8")

        config = STAT_CONFIG[field_name]
        url = LEADERBOARD_URL.format(stat=config["stat"])
        logger.info("Fetching public LaLiga %s leaderboard", field_name)
        started = time.perf_counter()
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        html = response.text
        cache_path.write_text(html, encoding="utf-8")
        logger.info(
            "Fetched public LaLiga %s leaderboard in %.2fs",
            field_name,
            time.perf_counter() - started,
        )
        return html

    def _parse_table_html(self, html: str, field_name: str) -> Optional[pd.DataFrame]:
        config = STAT_CONFIG[field_name]
        try:
            tables = pd.read_html(io.StringIO(html))
        except Exception:
            return None

        wanted = _normalize_text(config["label"])
        for table in tables:
            columns = [_normalize_text(col) for col in table.columns]
            if "player" not in columns or "team" not in columns:
                continue

            player_col = table.columns[columns.index("player")]
            team_col = table.columns[columns.index("team")]

            value_col = None
            for idx, normalized in enumerate(columns):
                if normalized == wanted or normalized.endswith(wanted):
                    value_col = table.columns[idx]
                    break
            if value_col is None:
                numeric_candidates = [
                    table.columns[idx]
                    for idx, normalized in enumerate(columns)
                    if normalized not in {"player", "team", "country", "rank", "pos"}
                ]
                if numeric_candidates:
                    value_col = numeric_candidates[-1]
            if value_col is None:
                continue

            parsed = pd.DataFrame(
                {
                    "name": table[player_col].astype(str).str.strip(),
                    "team_raw": table[team_col].astype(str).str.strip(),
                    field_name: pd.to_numeric(table[value_col], errors="coerce").fillna(0),
                }
            )
            parsed = parsed[parsed["name"].astype(str).ne("")]
            if not parsed.empty:
                return parsed
        return None

    def _parse_text_fallback(self, html: str, field_name: str) -> pd.DataFrame:
        config = STAT_CONFIG[field_name]
        soup = BeautifulSoup(html, "html.parser")
        lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]
        wanted = config["label"].upper()

        header_idx = None
        for i in range(len(lines) - 3):
            if (
                lines[i].upper() == "PLAYER"
                and lines[i + 1].upper() == "TEAM"
                and lines[i + 3].upper() == wanted
            ):
                header_idx = i
                break

        if header_idx is None:
            return pd.DataFrame(columns=["name", "team_raw", field_name])

        rows: List[Dict[str, Any]] = []
        i = header_idx + 4
        while i + 4 < len(lines):
            current = lines[i]
            if current == "..." or "REGISTERED YET" in current.upper():
                break
            if not re.fullmatch(r"\d+", current):
                i += 1
                continue

            player = lines[i + 1]
            team_raw = lines[i + 2]
            value_raw = lines[i + 4]
            if not player or not team_raw or not _looks_like_team_code(team_raw):
                i += 1
                continue

            rows.append(
                {
                    "name": player,
                    "team_raw": team_raw,
                    field_name: _safe_int(value_raw, 0),
                }
            )
            i += 5

        return pd.DataFrame(rows)

    def _parse_leaderboard(self, html: str, field_name: str) -> pd.DataFrame:
        parsed = self._parse_table_html(html, field_name)
        if parsed is None or parsed.empty:
            parsed = self._parse_text_fallback(html, field_name)
        if parsed.empty:
            logger.warning("No rows parsed from public LaLiga %s leaderboard", field_name)
            return parsed

        parsed["team"] = parsed["team_raw"].map(lambda raw: normalize_team_name(TEAM_CODE_MAP.get(str(raw).upper(), raw)))
        parsed["normalized_name"] = parsed["name"].map(normalize_player_name)
        parsed["normalized_team"] = parsed["team"].map(_normalize_text)
        parsed[field_name] = parsed[field_name].map(lambda value: _safe_int(value, 0))
        parsed = parsed.drop_duplicates(subset=["normalized_name", "normalized_team"], keep="first")
        return parsed[["name", "team", "normalized_name", "normalized_team", field_name]]

    def _runtime_config(self, season: int, force_refresh: bool = False) -> Dict[str, str]:
        if not force_refresh and season in self._runtime_config_cache:
            return self._runtime_config_cache[season]
        html = self._fetch_html("minutes_played", season, force_refresh=force_refresh)
        next_data = _parse_next_data(html)
        runtime = next_data.get("runtimeConfig", {}) or {}
        query = next_data.get("query", {}) or {}
        competition = str(query.get("stat_competition") or PUBLIC_SERVICE_COMPETITION).strip() or PUBLIC_SERVICE_COMPETITION
        config = {
            "backend_url": str(runtime.get("backendUrl") or PUBLIC_SERVICE_FALLBACK_URL).rstrip("/"),
            "backend_subscription": str(
                runtime.get("backendSubscription") or PUBLIC_SERVICE_SUBSCRIPTION_FALLBACK
            ).strip(),
            "competition": competition,
        }
        self._runtime_config_cache[season] = config
        return config

    def _fetch_rankings_page(
        self,
        season: int,
        offset: int,
        force_refresh: bool = False,
        limit: int = PLAYER_RANKINGS_PAGE_SIZE,
    ) -> Dict[str, Any]:
        cache_path = self._rankings_cache_path(season, offset)
        if not force_refresh and self._cache_fresh(cache_path):
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        runtime = self._runtime_config(season, force_refresh=force_refresh)
        url = (
            f"{runtime['backend_url']}/api/v1/subscriptions/"
            f"{runtime['competition']}-{season}/players/rankings"
        )
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Origin": BASE_URL,
            "Referer": f"{BASE_URL}/",
            "Ocp-Apim-Subscription-Key": runtime["backend_subscription"],
        }
        params = {
            "limit": limit,
            "offset": offset,
            "orderField": "stat.total_mins_played_ranking",
            "orderType": "ASC",
        }
        response = self.session.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _extract_stat_value(self, stats_map: Dict[str, Any], *keys: str) -> int:
        for key in keys:
            value = stats_map.get(key)
            if value not in (None, ""):
                return _safe_int(value, 0)
        return 0

    def _load_rankings_frame(self, season: int, force_refresh: bool = False) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        offset = 0
        total = None

        while total is None or offset < total:
            payload = self._fetch_rankings_page(season, offset, force_refresh=force_refresh)
            total = _safe_int(payload.get("total"), total or 0)
            players = payload.get("player_rankings", []) or []
            if not players:
                break
            rows.extend(players)
            offset += len(players)
            if len(players) < PLAYER_RANKINGS_PAGE_SIZE:
                break

        if not rows:
            return pd.DataFrame()

        parsed_rows: List[Dict[str, Any]] = []
        for player in rows:
            team_info = player.get("team") or {}
            stats_map = {
                str(item.get("name") or "").strip(): item.get("stat")
                for item in (player.get("stats") or [])
                if item.get("name")
            }
            team_name = normalize_team_name(
                str(team_info.get("nickname") or team_info.get("name") or team_info.get("shortname") or "")
            )
            name = str(player.get("name") or player.get("nickname") or "").strip()
            if not name or not team_name:
                continue

            parsed_rows.append(
                {
                    "name": name,
                    "nickname": str(player.get("nickname") or "").strip(),
                    "team": team_name,
                    "normalized_name": normalize_player_name(name),
                    "normalized_nickname": normalize_player_name(player.get("nickname") or ""),
                    "normalized_team": _normalize_text(team_name),
                    "minutes_played": self._extract_stat_value(stats_map, "total_mins_played"),
                    "goals": self._extract_stat_value(stats_map, "total_goals"),
                    "assists": self._extract_stat_value(stats_map, "total_assists"),
                    "shots": self._extract_stat_value(stats_map, "total_shots", "total_scoring_att", "total_attempt"),
                    "saves": self._extract_stat_value(stats_map, "total_saves"),
                    "yellow_cards": self._extract_stat_value(stats_map, "total_yellow_cards", "total_yellow_card"),
                    "red_cards": self._extract_stat_value(stats_map, "total_red_cards", "total_red_card"),
                    "appearances": self._extract_stat_value(stats_map, "total_games"),
                    "shirt_number": _safe_int(player.get("shirt_number"), 0),
                }
            )

        if not parsed_rows:
            return pd.DataFrame()
        parsed = pd.DataFrame(parsed_rows)
        parsed = parsed.drop_duplicates(subset=["normalized_name", "normalized_team"], keep="first")
        return parsed

    def load_stats(
        self,
        force_refresh: bool = False,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        season = self._season_key()
        merged_cache = self._merged_cache_path(season)
        cached = None if force_refresh else self._load_cached_merge(merged_cache)
        if cached is not None:
            return cached

        requested_fields = [field for field in (fields or list(STAT_CONFIG.keys())) if field in STAT_CONFIG]
        merged: Optional[pd.DataFrame] = None
        try:
            merged = self._load_rankings_frame(season, force_refresh=force_refresh)
            if not merged.empty:
                logger.info(
                    "Loaded %d players from public LaLiga rankings endpoint",
                    len(merged),
                )
        except Exception as e:
            logger.warning("Public LaLiga rankings endpoint failed, falling back to leaderboard HTML: %s", e)

        if merged is None or merged.empty:
            loaded_fields: List[str] = []
            failed_fields: List[str] = []
            merged = None
            for field_name in requested_fields:
                try:
                    html = self._fetch_html(field_name, season, force_refresh=force_refresh)
                    stat_df = self._parse_leaderboard(html, field_name)
                except Exception as e:
                    logger.warning("Skipping public LaLiga %s leaderboard: %s", field_name, e)
                    failed_fields.append(field_name)
                    continue
                if stat_df.empty:
                    failed_fields.append(field_name)
                    continue
                loaded_fields.append(field_name)
                if merged is None:
                    merged = stat_df.copy()
                else:
                    merged = merged.merge(
                        stat_df,
                        on=["normalized_name", "normalized_team"],
                        how="outer",
                        suffixes=("", "_dup"),
                    )
                    if "name_dup" in merged.columns:
                        merged["name"] = merged["name"].fillna(merged["name_dup"])
                        merged = merged.drop(columns=["name_dup"])
                    if "team_dup" in merged.columns:
                        merged["team"] = merged["team"].fillna(merged["team_dup"])
                        merged = merged.drop(columns=["team_dup"])

            if merged is None:
                if merged_cache.exists():
                    logger.warning(
                        "Public LaLiga stats refresh returned no usable tables; reusing cached merge from %s",
                        merged_cache,
                    )
                    return pd.read_csv(merged_cache)
                merged = pd.DataFrame(
                    columns=[
                        "name",
                        "team",
                        "normalized_name",
                        "normalized_team",
                        *STAT_CONFIG.keys(),
                    ]
                )
            else:
                logger.info(
                    "Public LaLiga stats loaded fields=%s missing=%s",
                    ",".join(loaded_fields) if loaded_fields else "none",
                    ",".join(failed_fields) if failed_fields else "none",
                )

        for field_name in STAT_CONFIG:
            if field_name not in merged.columns:
                merged[field_name] = 0
            merged[field_name] = pd.to_numeric(merged[field_name], errors="coerce").fillna(0).astype(int)

        if "minutes_played" not in merged.columns:
            merged["minutes_played"] = 0
        if "appearances" not in merged.columns:
            merged["appearances"] = 0
        merged["appearances"] = merged.apply(
            lambda row: _safe_int(row.get("appearances"), 0)
            or (max(1, round(_safe_int(row.get("minutes_played"), 0) / 78)) if _safe_int(row.get("minutes_played"), 0) > 0 else 0),
            axis=1,
        )

        def _per90(count_col: str) -> List[float]:
            values: List[float] = []
            for _, row in merged.iterrows():
                minutes = _safe_int(row.get("minutes_played", 0), 0)
                count = _safe_int(row.get(count_col, 0), 0)
                values.append(round((count * 90) / minutes, 2) if minutes > 0 else 0.0)
            return values

        merged["goals_per_90"] = _per90("goals")
        merged["assists_per_90"] = _per90("assists")
        merged["shots_per_90"] = _per90("shots")
        merged["saves_per_90"] = _per90("saves")
        merged["stats_source"] = "laliga_official_public"

        merged = merged.sort_values(["minutes_played", "goals", "assists"], ascending=False).reset_index(drop=True)
        merged.to_csv(merged_cache, index=False)
        return merged


def resolve_public_player_stats(
    stats_df: pd.DataFrame,
    player_name: str,
    team_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if stats_df is None or stats_df.empty:
        return None

    name_key = normalize_player_name(player_name)
    team_key = _normalize_text(normalize_team_name(team_hint or ""))

    exact_mask = stats_df["normalized_name"].eq(name_key)
    if "normalized_nickname" in stats_df.columns:
        exact_mask |= stats_df["normalized_nickname"].eq(name_key)
    if team_key:
        exact_mask &= stats_df["normalized_team"].eq(team_key)
    exact = stats_df[exact_mask]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    candidates = stats_df
    if team_key:
        team_candidates = stats_df[stats_df["normalized_team"].eq(team_key)]
        if not team_candidates.empty:
            candidates = team_candidates

    best_row = None
    best_score = 0.0
    for _, row in candidates.iterrows():
        score = _score_player_match(player_name, str(row.get("name", "")))
        nickname_score = _score_player_match(player_name, str(row.get("nickname", "")))
        score = max(score, nickname_score)
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None or best_score < 0.58:
        return None
    return best_row.to_dict()
