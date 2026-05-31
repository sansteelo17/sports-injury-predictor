#!/usr/bin/env python3
"""
Refresh predictions with current data.

Uses football-data.org for team-level workload (the only mode that still works
end-to-end). Per-player season minutes/goals come from the soccerdata library
(FBref data, accessed via undetected chromedriver) on top of the league-specific
public-stats loaders.

Usage:
    python scripts/refresh_predictions.py --api-key YOUR_KEY
    python scripts/refresh_predictions.py --dry-run
    python scripts/refresh_predictions.py --transfermarkt   # also scrape TM injuries

Get a free football-data.org API key at https://www.football-data.org/client/register.
"""

import os
import sys
import argparse
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Load .env if it exists (so API keys don't need to be passed on command line)
_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip().strip('"').strip("'"))
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger
from src.utils.model_io import load_artifacts, save_artifacts

logger = get_logger(__name__)


# =============================================================================
# FBREF MODE: Scrape actual player match logs
# =============================================================================

# =============================================================================
# API MODE: Team-level workload (the only mode that still works)
# =============================================================================

# Normalize team names between squad API and match API
_TEAM_NAME_MAP = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Brighton Hove": "Brighton",
    "Leeds United": "Leeds",
    "Nottingham": "Nottingham Forest",
}

_LA_LIGA_TEAM_KEY_MAP = {
    "athletic": "athletic club",
    "athletic bilbao": "athletic club",
    "athletic club de bilbao": "athletic club",
    "atleti": "atletico madrid",
    "atletico": "atletico madrid",
    "atletico de madrid": "atletico madrid",
    "club atletico de madrid": "atletico madrid",
    "barca": "barcelona",
    "fc barcelona": "barcelona",
    "real betis balompie": "real betis",
    "betis": "real betis",
    "real sociedad de futbol": "real sociedad",
    "deportivo alaves": "alaves",
    "alaves": "alaves",
    "almeria": "almeria",
    "ud almeria": "almeria",
    "cd leganes": "leganes",
    "leganes": "leganes",
    "leganes cf": "leganes",
    "espanyol": "espanyol",
    "rcd espanyol de barcelona": "espanyol",
    "rc celta de vigo": "celta vigo",
    "celta": "celta vigo",
    "sevilla fc": "sevilla",
    "valencia cf": "valencia",
    "villarreal cf": "villarreal",
    "ca osasuna": "osasuna",
    "getafe cf": "getafe",
    "rayo vallecano de madrid": "rayo vallecano",
    "mallorca": "mallorca",
    "rcd mallorca": "mallorca",
    "levante ud": "levante",
    "cadiz cf": "cadiz",
    "real madrid cf": "real madrid",
}

_PLAYER_NAME_CANONICAL_MAP = {
    # La Liga / football-data common call-name mismatches
    "gavi": "pablo gavira",
    "pablo martin paez gavira": "pablo gavira",
    "pablo martín páez gavira": "pablo gavira",
    "alejandro balde": "alex balde",
    "alex balde": "alex balde",
    "pedro gonzalez lopez": "pedri",
    "pedro gonzalez lópez": "pedri",
    "pedri": "pedri",
    # Common top-level aliases used across providers
    "vinicius jose paixao de oliveira junior": "vinicius junior",
    "vinicius jose passador de oliveira": "vinicius junior",
    "vinicius jr": "vinicius junior",
    "vinicius junior": "vinicius junior",
    "martin odegaard": "martin odegaard",
    "martin ødegaard": "martin odegaard",
}


def _normalize_match_team(team: str) -> str:
    """Map squad team name to match data team name."""
    return _TEAM_NAME_MAP.get(team, team)


def _normalize_lookup_text(value: str) -> str:
    """Fold accents/punctuation for cross-provider matching."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(text.split())


def _normalize_player_key(name: str) -> str:
    return _normalize_lookup_text(name)


def _canonical_player_key(name: str) -> str:
    key = _normalize_player_key(name)
    return _PLAYER_NAME_CANONICAL_MAP.get(key, key)


def _normalize_team_key(team: str) -> str:
    key = _normalize_lookup_text(team)
    return _LA_LIGA_TEAM_KEY_MAP.get(key, key)


# Surname-only matching is inherently ambiguous: two players can share a surname
# (Anthony vs Kaide Gordon, Bruno vs Mateus Fernandes). A flat surname key lets
# the second registration overwrite the first, so a later lookup serves one
# player's stats to another. Namespace surname keys and disable any that collide
# so a wrong match is never returned — a miss is better than another player's
# appearances/minutes/goals.
_AMBIGUOUS = object()


def _register_surname_key(lookup: dict, normalized_name: str, payload: dict) -> None:
    parts = normalized_name.split()
    if len(parts) < 2:
        return  # mononyms are already covered by full-name keys
    key = ("__surname__", parts[-1])
    if key not in lookup:
        lookup[key] = payload
    elif lookup[key] is not payload:
        lookup[key] = _AMBIGUOUS


def _surname_candidate(normalized_name: str):
    parts = normalized_name.split()
    return ("__surname__", parts[-1]) if len(parts) >= 2 else None


def _register_minutes_payload(lookup: dict, name: str, payload: dict, team: str | None = None):
    """Register a minutes payload under robust match keys."""
    if not name:
        return

    payload = {
        "ratio": payload.get("ratio"),
        "minutes_played": int(payload.get("minutes_played", 0) or 0),
        "appearances": int(payload.get("appearances", 0) or 0),
        "source": str(payload.get("source", "unknown")),
    }

    normalized_name = _normalize_player_key(name)
    canonical_name = _canonical_player_key(name)
    normalized_team = _normalize_team_key(team) if team else ""

    lookup[name] = payload
    lookup[name.lower()] = payload
    lookup[normalized_name] = payload
    lookup[canonical_name] = payload
    if team:
        lookup[(normalized_name, normalized_team)] = payload
        lookup[(canonical_name, normalized_team)] = payload


def _lookup_minutes_payload(lookup: dict, player_name: str, team_name: str | None = None):
    """Resolve the best minutes payload using team-aware keys first."""
    normalized_name = _normalize_player_key(player_name)
    canonical_name = _canonical_player_key(player_name)
    normalized_team = _normalize_team_key(team_name) if team_name else ""

    candidates = []
    if normalized_team:
        candidates.append((normalized_name, normalized_team))
        candidates.append((canonical_name, normalized_team))
    candidates.extend([
        player_name,
        player_name.lower(),
        normalized_name,
        canonical_name,
    ])
    # Surname-only fallback removed: with no team binding it served a namesake's
    # stats across clubs (Kaide Gordon picked up Anthony Gordon's numbers because
    # only one "Gordon" was in the source). A miss beats a wrong match.

    for key in candidates:
        if key is None:
            continue
        val = lookup.get(key)
        if val is not None and val is not _AMBIGUOUS:
            return val
    return None


def _register_signal_payload(lookup: dict, name: str, payload: dict, team: str | None = None):
    if not name:
        return

    normalized_name = _normalize_player_key(name)
    canonical_name = _canonical_player_key(name)
    normalized_team = _normalize_team_key(team) if team else ""

    keys = [name, name.lower(), normalized_name, canonical_name]
    if normalized_team:
        keys.extend([
            (normalized_name, normalized_team),
            (canonical_name, normalized_team),
        ])
    for key in keys:
        lookup[key] = payload


def _lookup_signal_payload(lookup: dict, player_name: str, team_name: str | None = None):
    normalized_name = _normalize_player_key(player_name)
    canonical_name = _canonical_player_key(player_name)
    normalized_team = _normalize_team_key(team_name) if team_name else ""

    candidates = []
    if normalized_team:
        candidates.extend([
            (normalized_name, normalized_team),
            (canonical_name, normalized_team),
        ])
    candidates.extend([player_name, player_name.lower(), normalized_name, canonical_name])
    # Surname-only fallback removed: with no team binding it served a namesake's
    # stats across clubs (Kaide Gordon picked up Anthony Gordon's numbers because
    # only one "Gordon" was in the source). A miss beats a wrong match.

    for key in candidates:
        if key is None:
            continue
        val = lookup.get(key)
        if val is not None and val is not _AMBIGUOUS:
            return val
    return None


def _register_injury_payload(lookup: dict, name: str, payload: dict):
    """Register injury history under exact and canonicalized keys."""
    if not name:
        return
    normalized_name = _normalize_player_key(name)
    canonical_name = _canonical_player_key(name)
    lookup[name] = payload
    lookup[name.lower()] = payload
    lookup[normalized_name] = payload
    lookup[canonical_name] = payload


def _lookup_injury_payload(lookup: dict, player_name: str):
    """Resolve injury history using exact, normalized, and canonical keys."""
    normalized_name = _normalize_player_key(player_name)
    canonical_name = _canonical_player_key(player_name)
    for key in (player_name, player_name.lower(), normalized_name, canonical_name):
        if key in lookup:
            return lookup[key]
    return None


def _count_minutes_matches(players_df: pd.DataFrame, lookup: dict) -> int:
    matched = 0
    for _, player in players_df[["name", "team"]].drop_duplicates().iterrows():
        entry = _lookup_minutes_payload(lookup, str(player["name"]), str(player["team"]))
        if entry and int(entry.get("minutes_played", 0) or 0) > 0:
            matched += 1
    return matched


def _merge_minutes_lookup(base_lookup: dict, incoming_lookup: dict) -> dict:
    """
    Merge minutes lookups without letting stale fallback rows overwrite stronger data.

    Priority rule:
    - keep existing populated minutes if they are already present
    - allow incoming to replace empty/zero-minute placeholders
    """
    merged = dict(base_lookup)
    for key, payload in (incoming_lookup or {}).items():
        if not isinstance(payload, dict):
            continue  # skip sentinels/placeholders (e.g. ambiguity markers)
        existing = merged.get(key)
        existing_minutes = int(existing.get("minutes_played", 0) or 0) if isinstance(existing, dict) else 0
        incoming_minutes = int(payload.get("minutes_played", 0) or 0)

        if existing is None or existing_minutes <= 0 < incoming_minutes or existing_minutes <= 0:
            merged[key] = payload
    return merged


def build_artifact_minutes_lookup(players_df, artifact_df, league_name: str):
    """Seed minutes from the existing inference artifact when external sources miss."""
    if not isinstance(artifact_df, pd.DataFrame) or artifact_df.empty:
        return {}

    artifact = artifact_df.copy()
    if "league" in artifact.columns:
        artifact = artifact[artifact["league"].fillna("") == league_name]
    if artifact.empty:
        return {}

    wanted = {
        (_normalize_player_key(str(row["name"])), _normalize_team_key(str(row["team"])))
        for _, row in players_df[["name", "team"]].drop_duplicates().iterrows()
    }

    lookup = {}
    matched = 0
    for _, row in artifact.iterrows():
        name = str(row.get("name", "")).strip()
        team = str(row.get("team", "") or row.get("player_team", "")).strip()
        key = (_normalize_player_key(name), _normalize_team_key(team))
        if key not in wanted:
            continue

        minutes_played = int(row.get("minutes_played", 0) or 0)
        appearances = int(row.get("appearances", 0) or 0)
        if minutes_played <= 0 and appearances <= 0:
            continue

        payload = {
            "ratio": row.get("playing_time_ratio"),
            "minutes_played": minutes_played,
            "appearances": appearances,
            "source": "artifact",
        }
        _register_minutes_payload(lookup, name, payload, team=team)
        matched += 1

    if matched:
        logger.info(f"Seeded {matched} {league_name} players from cached artifact minutes")
    return lookup


def build_artifact_signal_lookup(players_df, artifact_df, league_name: str):
    """Seed goals/assists from the existing inference artifact when the live
    stat source (FBref) is unavailable.

    On Render the refresh runs without Chrome, so FBref returns nothing and the
    PL block would otherwise fall back to FPL — whose goals/assists differ from
    what fans verify against. Preferring the previously-shipped (FBref-built)
    artifact stats over FPL keeps the displayed numbers correct across refreshes.
    """
    if not isinstance(artifact_df, pd.DataFrame) or artifact_df.empty:
        return {}

    artifact = artifact_df.copy()
    if "league" in artifact.columns:
        artifact = artifact[artifact["league"].fillna("") == league_name]
    if artifact.empty:
        return {}

    wanted = {
        (_normalize_player_key(str(row["name"])), _normalize_team_key(str(row["team"])))
        for _, row in players_df[["name", "team"]].drop_duplicates().iterrows()
    }

    lookup = {}
    matched = 0
    for _, row in artifact.iterrows():
        name = str(row.get("name", "")).strip()
        team = str(row.get("team", "") or row.get("player_team", "")).strip()
        key = (_normalize_player_key(name), _normalize_team_key(team))
        if key not in wanted:
            continue

        goals = int(row.get("goals", 0) or 0)
        assists = int(row.get("assists", 0) or 0)
        minutes = int(row.get("minutes_played", 0) or 0)
        if goals <= 0 and assists <= 0 and minutes <= 0:
            continue

        _register_signal_payload(
            lookup,
            name,
            {
                "goals": goals,
                "assists": assists,
                "goals_per_90": float(row.get("goals_per_90", 0) or 0.0),
                "assists_per_90": float(row.get("assists_per_90", 0) or 0.0),
                "minutes": minutes,
            },
            team=team,
        )
        matched += 1

    if matched:
        logger.info(f"Seeded {matched} {league_name} players from cached artifact signals")
    return lookup


def compute_team_workload(matches_df, team, as_of_date):
    """
    Compute team-level workload metrics.

    Returns all workload features expected by the model:
    - acute_load, chronic_load, acwr (basic)
    - monotony, strain, fatigue_index (advanced)
    - workload_slope, spike_flag (trend indicators)
    """
    team = _normalize_match_team(team)
    team_matches = matches_df[
        ((matches_df["Home"] == team) | (matches_df["Away"] == team)) &
        (matches_df["Date"] < as_of_date)
    ].copy()

    team_matches = team_matches.sort_values("Date")

    if len(team_matches) == 0:
        return {
            "acute_load": 0, "chronic_load": 0, "acwr": 1.0,
            "monotony": 1.0, "strain": 0, "fatigue_index": 0,
            "workload_slope": 0, "spike_flag": 0,
            "matches_last_7": 0, "matches_last_14": 0, "matches_last_30": 0,
        }

    date_7 = as_of_date - timedelta(days=7)
    date_14 = as_of_date - timedelta(days=14)
    date_28 = as_of_date - timedelta(days=28)
    date_30 = as_of_date - timedelta(days=30)

    matches_7 = len(team_matches[team_matches["Date"] >= date_7])
    matches_14 = len(team_matches[team_matches["Date"] >= date_14])
    matches_28 = len(team_matches[team_matches["Date"] >= date_28])
    matches_30 = len(team_matches[team_matches["Date"] >= date_30])

    acute_load = matches_7
    chronic_load = matches_28 / 4

    if chronic_load > 0:
        acwr = acute_load / chronic_load
    else:
        acwr = 1.0 if acute_load == 0 else 2.0

    # Monotony: 14-day load regularity
    daily_loads = []
    for i in range(14):
        day = as_of_date - timedelta(days=i)
        day_matches = len(team_matches[team_matches["Date"].dt.date == day.date()])
        daily_loads.append(day_matches)

    mean_load = np.mean(daily_loads)
    std_load = np.std(daily_loads)
    if std_load < 0.1:
        monotony = min(mean_load / 0.1, 5.0) if mean_load > 0 else 1.0
    else:
        monotony = min(mean_load / std_load, 5.0)

    strain = acute_load * monotony
    fatigue_index = acute_load - chronic_load
    spike_flag = 1 if acwr > 1.8 else 0

    # Workload slope: trend in match density over last 5 weeks
    # Positive = schedule getting busier, negative = easing off
    if len(team_matches) >= 3:
        weekly_loads = []
        for w in range(5):
            week_start = as_of_date - timedelta(days=7 * (w + 1))
            week_end = as_of_date - timedelta(days=7 * w)
            week_count = len(team_matches[
                (team_matches["Date"] >= week_start) &
                (team_matches["Date"] < week_end)
            ])
            weekly_loads.append(week_count)
        # Reverse so index 0 = oldest week, 4 = most recent
        weekly_loads = weekly_loads[::-1]
        x = np.arange(len(weekly_loads))
        workload_slope = np.polyfit(x, weekly_loads, 1)[0]
    else:
        workload_slope = 0

    return {
        "acute_load": acute_load, "chronic_load": chronic_load,
        "acwr": round(acwr, 3), "monotony": round(monotony, 3),
        "strain": round(strain, 3), "fatigue_index": round(fatigue_index, 3),
        "workload_slope": round(workload_slope, 4), "spike_flag": spike_flag,
        "matches_last_7": matches_7, "matches_last_14": matches_14,
        "matches_last_30": matches_30,
    }


def load_player_minutes_lookup():
    """
    Load player playing time ratios from FPL API (live minutes data).

    FPL provides actual minutes played this season for every PL player.
    Converts to a ratio: minutes / max_possible_minutes.

    Returns a dict mapping player name -> metadata dict with:
    - ratio: playing time ratio (0-1)
    - minutes_played: actual minutes played this season
    """
    try:
        from src.data_loaders.fpl_api import FPLClient
        client = FPLClient()
        all_stats = client.get_all_player_stats()

        # Current GW determines max possible minutes
        current_gw = client.get_current_gameweek()
        gw_num = current_gw.get("id", 20) if current_gw else 20
        max_minutes = gw_num * 90  # Maximum possible minutes so far

        lookup = {}
        for p in all_stats:
            minutes = p.get("minutes", 0)
            if max_minutes > 0:
                ratio = min(minutes / max_minutes, 1.0)
            else:
                ratio = 0.5

            payload = {
                "ratio": ratio,
                "minutes_played": minutes,
                "appearances": p.get("minutes", 0) // 90 if minutes else 0,
                "source": "fpl",
            }

            # Index by multiple name formats for matching
            name = p.get("name", "")
            full_name = p.get("full_name", "")
            team = p.get("team", "")
            if name:
                _register_minutes_payload(lookup, name, payload, team=team)
            if full_name:
                _register_minutes_payload(lookup, full_name, payload, team=team)

        logger.info(f"Loaded playing time ratios for {len(all_stats)} players from FPL API (GW{gw_num})")
        return lookup
    except Exception as e:
        logger.warning(f"Failed to load FPL minutes: {e}")
        return {}


def load_fpl_live_signal_lookup():
    """Load live EPL player stats that can proxy output and role importance."""
    try:
        from src.data_loaders.fpl_api import FPLClient
        client = FPLClient()
        all_stats = client.get_all_player_stats()
        lookup = {}
        for player in all_stats:
            payload = {
                "goals": int(player.get("goals", 0) or 0),
                "assists": int(player.get("assists", 0) or 0),
                "goals_per_90": float(player.get("goals_per_90", 0) or 0.0),
                "assists_per_90": float(player.get("assists_per_90", 0) or 0.0),
                "minutes": int(player.get("minutes", 0) or 0),
                "price": float(player.get("price", 0) or 0.0),
                "selected_by": float(player.get("selected_by", 0) or 0.0),
                "points_per_game": float(player.get("points_per_game", 0) or 0.0),
                "form": float(player.get("form", 0) or 0.0),
            }
            team = str(player.get("team", "")).strip()
            for name_key in [player.get("name"), player.get("full_name")]:
                if name_key:
                    _register_signal_payload(lookup, str(name_key), payload, team=team)
        logger.info("Loaded live FPL signal lookup for %s players", len(all_stats))
        return lookup
    except Exception as e:
        logger.warning(f"Failed to load FPL live signal lookup: {e}")
        return {}


def load_laliga_public_signal_lookup(players_df):
    """Load public La Liga player output signals for current-season projections."""
    from src.data_loaders.laliga_public_stats import (
        LaLigaPublicStatsLoader,
        resolve_public_player_stats,
    )

    try:
        stats_df = LaLigaPublicStatsLoader(cache_hours=6).load_stats(
            fields=["minutes_played", "goals", "assists", "shots", "saves"]
        )
    except Exception as e:
        logger.warning(f"Failed to load public La Liga signal lookup: {e}")
        return {}

    lookup = {}
    for _, player in players_df[["name", "team"]].drop_duplicates().iterrows():
        name = str(player.get("name", "")).strip()
        team = str(player.get("team", "")).strip()
        stats_row = resolve_public_player_stats(stats_df, name, team)
        if not stats_row:
            continue
        payload = {
            "goals": int(stats_row.get("goals", 0) or 0),
            "assists": int(stats_row.get("assists", 0) or 0),
            "goals_per_90": float(stats_row.get("goals_per_90", 0) or 0.0),
            "assists_per_90": float(stats_row.get("assists_per_90", 0) or 0.0),
            "shots_per_90": float(stats_row.get("shots_per_90", 0) or 0.0),
            "saves_per_90": float(stats_row.get("saves_per_90", 0) or 0.0),
            "minutes": int(stats_row.get("minutes_played", 0) or 0),
        }
        _register_signal_payload(lookup, name, payload, team=team)

    logger.info("Loaded public La Liga signal lookup for %s players", len(lookup))
    return lookup


def load_api_football_minutes_lookup(players_df, season: int, league_name: str = "La Liga", league_id: int = 140):
    """
    Load season minutes from API-Football player statistics.

    Uses a CSV cache to avoid paginating through the entire league every run.
    """
    import requests

    api_key = os.environ.get("API_FOOTBALL_KEY", "").strip()
    if not api_key:
        logger.info(f"No API_FOOTBALL_KEY set - skipping {league_name} API-Football minutes")
        return {}

    cache_dir = Path(PROJECT_ROOT) / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"api_football_minutes_{league_name.lower().replace(' ', '_')}_{season}.csv"

    def _read_cache():
        if not cache_file.exists():
            return {}
        try:
            cache_df = pd.read_csv(cache_file)
        except Exception as e:
            logger.warning(f"Failed to read API-Football minutes cache: {e}")
            return {}

        lookup = {}
        for _, row in cache_df.iterrows():
            payload = {
                "minutes_played": int(row.get("minutes_played", 0) or 0),
                "appearances": int(row.get("appearances", 0) or 0),
                "source": str(row.get("source", "api-football-cache")),
            }
            _register_minutes_payload(
                lookup,
                str(row.get("name", "")).strip(),
                payload,
                team=str(row.get("team", "")).strip(),
            )
        return lookup

    if cache_file.exists():
        age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
        if age_hours < 12:
            cached_lookup = _read_cache()
            if cached_lookup:
                matched = _count_minutes_matches(players_df, cached_lookup)
                print(f"   API-Football cache matched {matched}/{len(players_df)} {league_name} players")
                return cached_lookup

    print(f"   Loading {league_name} minutes from API-Football...")

    session = requests.Session()
    headers = {"x-apisports-key": api_key}
    rows = []
    page = 1
    total_pages = 1

    try:
        while page <= total_pages:
            resp = session.get(
                "https://v3.football.api-sports.io/players",
                params={"league": league_id, "season": season, "page": page},
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            errors = data.get("errors", {})
            if errors:
                logger.warning(f"API-Football player stats error for {league_name}: {errors}")
                break

            total_pages = int(data.get("paging", {}).get("total", page) or page)
            response_rows = data.get("response", []) or []
            if not response_rows:
                break

            for item in response_rows:
                player = item.get("player", {}) or {}
                name = str(player.get("name") or "").strip()
                if not name:
                    continue

                total_minutes = 0
                total_appearances = 0
                primary_team = ""

                for stat in item.get("statistics", []) or []:
                    games = stat.get("games", {}) or {}
                    minutes = games.get("minutes")
                    appearances = games.get("appearences", games.get("appearances"))
                    team_name = str((stat.get("team", {}) or {}).get("name") or "").strip()

                    try:
                        total_minutes += int(minutes or 0)
                    except (TypeError, ValueError):
                        pass
                    try:
                        total_appearances += int(appearances or 0)
                    except (TypeError, ValueError):
                        pass

                    if team_name and not primary_team:
                        primary_team = team_name

                if total_minutes <= 0 and total_appearances <= 0:
                    continue

                rows.append({
                    "name": name,
                    "team": primary_team,
                    "minutes_played": total_minutes,
                    "appearances": total_appearances,
                    "source": "api-football",
                })

            page += 1
    except Exception as e:
        logger.warning(f"Failed to fetch {league_name} minutes from API-Football: {e}")

    if not rows:
        cached_lookup = _read_cache()
        if cached_lookup:
            matched = _count_minutes_matches(players_df, cached_lookup)
            print(f"   API-Football cache fallback matched {matched}/{len(players_df)} {league_name} players")
            return cached_lookup
        return {}

    try:
        pd.DataFrame(rows).drop_duplicates(subset=["name", "team"], keep="first").to_csv(cache_file, index=False)
    except Exception as e:
        logger.warning(f"Failed to write API-Football minutes cache: {e}")

    lookup = {}
    for row in rows:
        _register_minutes_payload(
            lookup,
            str(row.get("name", "")).strip(),
            {
                "minutes_played": int(row.get("minutes_played", 0) or 0),
                "appearances": int(row.get("appearances", 0) or 0),
                "source": "api-football",
            },
            team=str(row.get("team", "")).strip(),
        )

    matched = _count_minutes_matches(players_df, lookup)
    print(f"   API-Football matched {matched}/{len(players_df)} {league_name} players")
    return lookup


def load_fbref_minutes_lookup(players_df, league_name: str = "La Liga", season_start_year: int | None = None):
    """Load season minutes from FBref via the soccerdata library.

    soccerdata wraps FBref behind undetected chromedriver so we get past the
    bot block our plain-requests scraper kept hitting (403). The shape of the
    returned lookup is unchanged so the rest of the pipeline doesn't care.

    season_start_year defaults to the current season inferred from today's
    date (Aug+ → this year, else previous).
    """
    from src.data_loaders.soccerdata_loader import load_player_season_stats

    if season_start_year is None:
        now = datetime.now()
        season_start_year = now.year if now.month >= 8 else now.year - 1

    print(f"   Loading {league_name} minutes via soccerdata (FBref, season {season_start_year})...")
    fbref_df = load_player_season_stats(league_name, season_start_year)
    if fbref_df is None or fbref_df.empty:
        print(f"   soccerdata returned no rows for {league_name}; continuing with other sources.")
        return {}

    lookup = {}
    for _, row in fbref_df.iterrows():
        minutes_played = int(row.get("season_minutes", 0) or 0)
        appearances = int(row.get("appearances", 0) or 0)
        if minutes_played <= 0 and appearances <= 0:
            continue
        _register_minutes_payload(
            lookup,
            str(row.get("name", "")).strip(),
            {
                "minutes_played": minutes_played,
                "appearances": appearances,
                "source": "soccerdata-fbref",
            },
            team=str(row.get("team", "")).strip(),
        )

    matched = _count_minutes_matches(players_df, lookup)
    print(f"   soccerdata/FBref matched {matched}/{len(players_df)} {league_name} players")
    return lookup


def load_laliga_public_minutes_lookup(players_df):
    """Load current La Liga minutes from public official LaLiga leaderboard pages."""
    from src.data_loaders.laliga_public_stats import (
        LaLigaPublicStatsLoader,
        resolve_public_player_stats,
    )

    print("   Loading La Liga minutes from official public LaLiga leaderboards...")
    try:
        stats_df = LaLigaPublicStatsLoader(cache_hours=6).load_stats(
            fields=["minutes_played", "goals", "assists"]
        )
    except Exception as e:
        logger.warning(f"Failed to load public LaLiga stats: {e}")
        return {}, 0

    lookup = {}
    matched = 0
    for _, player in players_df[["name", "team"]].drop_duplicates().iterrows():
        stats_row = resolve_public_player_stats(
            stats_df,
            str(player.get("name", "")).strip(),
            str(player.get("team", "")).strip(),
        )
        if not stats_row:
            continue

        minutes_played = int(stats_row.get("minutes_played", 0) or 0)
        appearances = int(stats_row.get("appearances", 0) or 0)
        if minutes_played <= 0 and appearances <= 0:
            continue

        matched += 1
        _register_minutes_payload(
            lookup,
            str(player.get("name", "")).strip(),
            {
                "minutes_played": minutes_played,
                "appearances": appearances,
                "source": str(stats_row.get("stats_source", "laliga-official-public")),
            },
            team=str(player.get("team", "")).strip(),
        )

    print(f"   Public LaLiga leaderboards matched {matched}/{len(players_df)} La Liga players")
    return lookup, matched


def load_transfermarkt_minutes_lookup(players_df, season: int, league_name: str = "La Liga"):
    """
    Load season minutes for players from Transfermarkt.

    Uses the scraper's HTML cache, plus a light CSV cache keyed by player + team,
    so the first run may be slow but subsequent refreshes stay reasonable.
    """
    from src.data_loaders.transfermarkt_scraper import TransfermarktScraper

    scraper = TransfermarktScraper(cache_hours=168)
    cache_dir = Path(PROJECT_ROOT) / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"transfermarkt_minutes_{league_name.lower().replace(' ', '_')}_{season}.csv"

    cached_rows = {}
    if cache_file.exists():
        try:
            cache_df = pd.read_csv(cache_file)
            for _, row in cache_df.iterrows():
                key = (
                    str(row.get("name", "")).strip().lower(),
                    str(row.get("team", "")).strip().lower(),
                )
                cached_rows[key] = {
                    "minutes_played": int(row.get("minutes_played", 0) or 0),
                    "appearances": int(row.get("appearances", 0) or 0),
                    "source": "transfermarkt-cache",
                }
        except Exception as e:
            logger.warning(f"Failed to read Transfermarkt minutes cache: {e}")

    lookup = {}
    updated_rows = []
    matched = 0

    players = players_df[["name", "team"]].drop_duplicates()
    print(f"   Loading {league_name} minutes from Transfermarkt cache/scraper...")

    for _, player in players.iterrows():
        name = str(player.get("name", "")).strip()
        team = str(player.get("team", "")).strip()
        key = (name.lower(), team.lower())

        entry = cached_rows.get(key)
        if entry is None:
            try:
                match = scraper.search_player(name, team_hint=team)
                if match:
                    stats = scraper.get_player_stats(
                        match["slug"],
                        match["player_id"],
                        season=str(season),
                    )
                    entry = {
                        "minutes_played": int(stats.get("minutes_played", 0) or 0),
                        "appearances": int(stats.get("appearances", 0) or 0),
                        "source": "transfermarkt",
                    }
                else:
                    entry = {
                        "minutes_played": 0,
                        "appearances": 0,
                        "source": "transfermarkt",
                    }
            except Exception as e:
                logger.debug(f"Transfermarkt minutes lookup failed for {name} ({team}): {e}")
                entry = {
                    "minutes_played": 0,
                    "appearances": 0,
                    "source": "transfermarkt",
                }

        if entry.get("minutes_played", 0) > 0:
            matched += 1

        payload = {
            "minutes_played": int(entry.get("minutes_played", 0) or 0),
            "appearances": int(entry.get("appearances", 0) or 0),
            "source": str(entry.get("source", "transfermarkt")),
        }
        # Register through the shared helper so the surname key is ambiguity-safe
        # (two players sharing a surname no longer overwrite each other's stats).
        _register_minutes_payload(lookup, name, payload, team)

        updated_rows.append({
            "name": name,
            "team": team,
            **payload,
        })

    try:
        pd.DataFrame(updated_rows).to_csv(cache_file, index=False)
    except Exception as e:
        logger.warning(f"Failed to write Transfermarkt minutes cache: {e}")

    print(f"   Matched Transfermarkt minutes for {matched}/{len(players)} {league_name} players")
    return lookup


def load_fbref_stats_lookups(players_df, league_name: str = "Premier League", season_start_year: int | None = None):
    """Build BOTH a minutes lookup (real appearances + minutes) and a signal
    lookup (goals + assists + per-90) from FBref via soccerdata.

    FBref reports real appearances (not FPL's minutes/90 proxy) and official
    goals/assists, so cards match what fans verify against. One request covers
    the whole league. Transfermarkt is no longer usable for this: its stats
    pages are JS-rendered, so the plain-requests scraper returns an empty shell.
    Returns ``(minutes_lookup, signal_lookup)``; the signal lookup only holds
    players FBref found, so callers can fall back to FPL for the rest.

    ``players_df`` (the squad) is used to re-key FBref payloads under the squad's
    team strings: FBref says "Newcastle United" while the squad/FPL say
    "Newcastle", and the team-aware lookup is keyed on the squad name. Without
    this alignment the FBref payload sits under a team key that's never queried,
    so the FPL fallback wins. Matched by normalized full name (unique per league).
    """
    from src.data_loaders.soccerdata_loader import load_player_season_stats

    if season_start_year is None:
        now = datetime.now()
        season_start_year = now.year if now.month >= 8 else now.year - 1

    squad_team_by_name = {}
    if players_df is not None and "name" in getattr(players_df, "columns", []):
        for _, p in players_df[["name", "team"]].drop_duplicates().iterrows():
            squad_team_by_name[_normalize_player_key(str(p["name"]))] = str(p["team"])

    print(f"   Loading {league_name} season stats from FBref (apps/goals/assists/minutes)...")
    df = load_player_season_stats(league_name, season_start_year)
    if df is None or df.empty:
        print(f"   FBref returned no rows for {league_name}; stats fall back to FPL.")
        return {}, {}

    minutes_lookup: dict = {}
    signal_lookup: dict = {}
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        # Prefer the squad's team string so keys line up with FPL/the squad.
        team = squad_team_by_name.get(_normalize_player_key(name), str(row.get("team", "")).strip())
        mins = int(row.get("season_minutes", 0) or 0)
        apps = int(row.get("appearances", 0) or 0)
        goals = int(row.get("goals", 0) or 0)
        assists = int(row.get("assists", 0) or 0)

        _register_minutes_payload(
            minutes_lookup,
            name,
            {"minutes_played": mins, "appearances": apps, "source": "fbref"},
            team,
        )
        per90 = 90.0 / mins if mins > 0 else 0.0
        _register_signal_payload(
            signal_lookup,
            name,
            {
                "goals": goals,
                "assists": assists,
                "goals_per_90": round(goals * per90, 3),
                "assists_per_90": round(assists * per90, 3),
                "minutes": mins,
            },
            team,
        )

    print(f"   FBref season stats: {len(df)} {league_name} players")
    return minutes_lookup, signal_lookup


def load_injury_history_lookup():
    """
    Load real injury history from scraped Transfermarkt data.

    CSVs are for training only — inference uses scraped pkl files exclusively.

    Sources (in priority order):
    1. player_injuries_detail.pkl — per-injury records with dates (best)
    2. player_history.pkl — summary stats (fallback)

    Returns a dict mapping player name -> {
        'previous_injuries': count,
        'days_since_last_injury': days since most recent injury,
        'last_injury_date': date of most recent injury,
        'total_days_lost': total days lost to injury,
        'avg_severity': average days per injury
    }
    """
    today = datetime.now()
    lookup = {}
    detail_players = 0
    summary_players = 0

    def _read_pickle_compat(path: str):
        try:
            return pd.read_pickle(path)
        except (NotImplementedError, Exception) as e:
            logger.warning(f"Pickle compat issue for {os.path.basename(path)}: {e}")
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)

    def _merge_payload(name: str, payload: dict, overwrite: bool = False):
        existing = _lookup_injury_payload(lookup, name)
        if not existing or overwrite:
            _register_injury_payload(lookup, name, payload)
            return

        merged = dict(existing)
        if int(merged.get("previous_injuries", 0) or 0) <= 0 and int(payload.get("previous_injuries", 0) or 0) > 0:
            merged["previous_injuries"] = int(payload.get("previous_injuries", 0) or 0)
        if float(merged.get("avg_severity", 0) or 0) <= 0 and float(payload.get("avg_severity", 0) or 0) > 0:
            merged["avg_severity"] = float(payload.get("avg_severity", 0) or 0)
        if int(merged.get("total_days_lost", 0) or 0) <= 0 and int(payload.get("total_days_lost", 0) or 0) > 0:
            merged["total_days_lost"] = int(payload.get("total_days_lost", 0) or 0)
        existing_days = merged.get("days_since_last_injury")
        incoming_days = payload.get("days_since_last_injury")
        if incoming_days is not None and (existing_days is None or int(existing_days) >= 365):
            merged["days_since_last_injury"] = int(incoming_days)
        if not merged.get("last_injury_date") and payload.get("last_injury_date"):
            merged["last_injury_date"] = payload.get("last_injury_date")
        _register_injury_payload(lookup, name, merged)

    # --- Source 1: Detailed injury records (primary, includes real recency) ---
    detail_paths = [
        os.path.join(PROJECT_ROOT, "models", "player_injuries_detail.pkl"),
        os.path.join(PROJECT_ROOT, "models", "laliga_injuries_detail.pkl"),
    ]
    for detail_path in detail_paths:
        if not os.path.exists(detail_path):
            continue
        try:
            detail_df = _read_pickle_compat(detail_path)
        except Exception as e:
            logger.warning(f"Failed to read injury detail data from {os.path.basename(detail_path)}: {e}")
            continue

        if not hasattr(detail_df, "columns") or "name" not in detail_df.columns:
            continue

        for col in detail_df.select_dtypes(include=["string"]).columns:
            detail_df[col] = detail_df[col].astype(object)
        if "name" in detail_df.columns and hasattr(detail_df["name"].dtype, "name") and "String" in str(detail_df["name"].dtype):
            detail_df["name"] = detail_df["name"].astype(str)

        if "injury_datetime" not in detail_df.columns:
            continue

        detail_df["injury_datetime"] = pd.to_datetime(detail_df["injury_datetime"], errors="coerce")
        detail_df = detail_df.dropna(subset=["injury_datetime"])
        if detail_df.empty:
            continue

        severity_col = "severity_days" if "severity_days" in detail_df.columns else "days_out" if "days_out" in detail_df.columns else None
        grouped_count = 0
        for player, group in detail_df.groupby("name"):
            injuries = group.sort_values("injury_datetime", ascending=False)
            last_injury = injuries.iloc[0]["injury_datetime"]
            total_lost = int(injuries[severity_col].sum()) if severity_col else 0
            avg_sev = float(injuries[severity_col].mean()) if severity_col else 0.0
            payload = {
                "previous_injuries": len(injuries),
                "days_since_last_injury": max(0, (today - last_injury).days),
                "last_injury_date": last_injury.strftime("%Y-%m-%d"),
                "total_days_lost": total_lost,
                "avg_severity": avg_sev,
            }
            _merge_payload(str(player), payload, overwrite=True)
            grouped_count += 1
        detail_players += grouped_count

    # --- Source 2: Summary history pkls (fill gaps when detail is missing) ---
    history_paths = [
        os.path.join(PROJECT_ROOT, "models", "player_history.pkl"),
        os.path.join(PROJECT_ROOT, "models", "laliga_player_history.pkl"),
    ]
    for history_path in history_paths:
        if not os.path.exists(history_path):
            continue
        try:
            history_df = _read_pickle_compat(history_path)
        except Exception as e:
            logger.warning(f"Failed to read injury history data from {os.path.basename(history_path)}: {e}")
            continue

        if not hasattr(history_df, "columns") or "name" not in history_df.columns:
            continue

        loaded = 0
        for _, row in history_df.iterrows():
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            count = int(row.get("player_injury_count", 0) or 0)
            avg_sev = float(row.get("player_avg_severity", 0) or 0)
            total_days_lost = row.get("total_days_lost")
            if pd.isna(total_days_lost) or total_days_lost is None:
                total_days_lost = int(round(count * avg_sev))
            days_since = row.get("days_since_last_injury")
            if pd.isna(days_since) or days_since is None:
                days_since = None
            payload = {
                "previous_injuries": count,
                "days_since_last_injury": None if days_since is None else int(days_since),
                "last_injury_date": row.get("last_injury_date") if pd.notna(row.get("last_injury_date")) else None,
                "total_days_lost": int(total_days_lost or 0),
                "avg_severity": avg_sev,
            }
            _merge_payload(name, payload, overwrite=False)
            loaded += 1
        summary_players += loaded

    if lookup:
        logger.info(
            "Loaded injury history for %s players (%s detail records, %s summary rows)",
            len({k for k in lookup.keys() if isinstance(k, str) and k == k.lower()}),
            detail_players,
            summary_players,
        )
        return lookup

    logger.warning("No scraped injury data found. Run: python scripts/scrape_injuries.py")
    return {}


def fetch_transfermarkt_injuries(player_names: list, existing_lookup: dict) -> dict:
    """
    Fetch injury history from Transfermarkt for players not in local CSV.

    Args:
        player_names: List of all player names
        existing_lookup: Dict of players already found in local CSV

    Returns:
        Updated lookup with Transfermarkt data merged in
    """
    from src.data_loaders.transfermarkt_scraper import TransfermarktScraper

    # Find players missing from local data
    missing = [name for name in player_names if name not in existing_lookup]

    if not missing:
        logger.info("All players found in local CSV, skipping Transfermarkt")
        return existing_lookup

    print(f"   Fetching {len(missing)} players from Transfermarkt...")
    scraper = TransfermarktScraper()

    lookup = existing_lookup.copy()
    found = 0
    errors = 0

    for i, name in enumerate(missing):
        if i % 20 == 0 and i > 0:
            print(f"   Progress: {i}/{len(missing)} ({found} found, {errors} not found)")

        try:
            data = scraper.fetch_player_injuries(name)
            if data and data["total_injuries"] > 0:
                lookup[name] = {
                    'previous_injuries': data["total_injuries"],
                    'days_since_last_injury': data["days_since_last"],
                    'last_injury_date': data["last_injury_date"],
                    'total_days_lost': data.get("total_days_out", 0)
                }
                found += 1
            elif data:
                # Player found but no injuries - still useful info
                lookup[name] = {
                    'previous_injuries': 0,
                    'days_since_last_injury': 365 * 3,  # 3 years for "never injured"
                    'last_injury_date': None,
                    'total_days_lost': 0
                }
                found += 1
        except Exception as e:
            logger.debug(f"Failed to fetch {name}: {e}")
            errors += 1

    print(f"   Transfermarkt: found {found}/{len(missing)} players")
    logger.info(f"Transfermarkt lookup complete: {found} found, {errors} errors")

    return lookup


def compute_team_form(matches_df, team, as_of_date, n_matches=5):
    """
    Compute team form features from recent match results.

    Returns dict with: goals_for_last_5, goals_against_last_5, goal_diff_last_5,
    avg_goal_diff_last_5, form_last_5, form_avg_last_5, win_ratio_last_5,
    win_streak, loss_streak, rest_days_before_injury, avg_rest_last_5.
    """
    team = _normalize_match_team(team)
    team_matches = matches_df[
        ((matches_df["Home"] == team) | (matches_df["Away"] == team)) &
        (matches_df["Date"] < as_of_date)
    ].sort_values("Date", ascending=False)

    if len(team_matches) == 0:
        return {
            "goals_for_last_5": 7, "goals_against_last_5": 5,
            "goal_diff_last_5": 2, "avg_goal_diff_last_5": 0.4,
            "form_last_5": 7, "form_avg_last_5": 1.4,
            "win_ratio_last_5": 0.4, "win_streak": 0, "loss_streak": 0,
            "rest_days_before_injury": 4, "avg_rest_last_5": 4,
        }

    recent = team_matches.head(n_matches)

    goals_for = 0
    goals_against = 0
    points = 0
    wins = 0
    results = []  # W/D/L sequence for streak calculation

    for _, match in recent.iterrows():
        is_home = match["Home"] == team
        gf = match["HomeGoals"] if is_home else match["AwayGoals"]
        ga = match["AwayGoals"] if is_home else match["HomeGoals"]
        goals_for += gf
        goals_against += ga

        if gf > ga:
            points += 3
            wins += 1
            results.append("W")
        elif gf == ga:
            points += 1
            results.append("D")
        else:
            results.append("L")

    n = len(recent)
    goal_diff = goals_for - goals_against

    # Streaks (from most recent)
    win_streak = 0
    for r in results:
        if r == "W":
            win_streak += 1
        else:
            break

    loss_streak = 0
    for r in results:
        if r == "L":
            loss_streak += 1
        else:
            break

    # Rest days: days since last match
    last_match_date = team_matches.iloc[0]["Date"]
    rest_days = max(1, (as_of_date - last_match_date).days)

    # Average rest between last 5 matches
    if n >= 2:
        dates = recent["Date"].tolist()
        gaps = [(dates[i] - dates[i + 1]).days for i in range(len(dates) - 1)]
        avg_rest = np.mean(gaps)
    else:
        avg_rest = 4

    return {
        "goals_for_last_5": int(goals_for),
        "goals_against_last_5": int(goals_against),
        "goal_diff_last_5": int(goal_diff),
        "avg_goal_diff_last_5": round(goal_diff / n, 2) if n > 0 else 0,
        "form_last_5": int(points),
        "form_avg_last_5": round(points / n, 2) if n > 0 else 0,
        "win_ratio_last_5": round(wins / n, 2) if n > 0 else 0,
        "win_streak": win_streak,
        "loss_streak": loss_streak,
        "rest_days_before_injury": round(rest_days, 1),
        "avg_rest_last_5": round(avg_rest, 1),
    }


def build_next_fixture_map(fixtures_df: pd.DataFrame) -> dict:
    """Map each team to its next scheduled fixture."""
    if fixtures_df is None or fixtures_df.empty:
        return {}

    fixtures = fixtures_df.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    fixtures = fixtures.sort_values(["date", "time"], na_position="last").reset_index(drop=True)

    next_map = {}
    for _, row in fixtures.iterrows():
        home = str(row.get("home", "")).strip()
        away = str(row.get("away", "")).strip()
        date_val = row.get("date")
        if home and home not in next_map:
            next_map[home] = {
                "opponent": away,
                "is_home": True,
                "date": date_val,
            }
        if away and away not in next_map:
            next_map[away] = {
                "opponent": home,
                "is_home": False,
                "date": date_val,
            }
    return next_map


def compute_fixture_context(match_df: pd.DataFrame, team: str, opponent: str, as_of_date: datetime) -> dict:
    """Build live opponent-strength and H2H features for the next fixture."""
    if not opponent:
        return {
            "opp_form_avg_last_5": 0.0,
            "opp_goal_diff_last_5": 0.0,
            "opp_win_ratio_last_5": 0.0,
            "h2h_matches_played": 0.0,
            "h2h_win_ratio": 0.0,
            "h2h_points_per_match": 0.0,
            "fixture_edge_score": 0.0,
        }

    team_form = compute_team_form(match_df, team, as_of_date)
    opp_form = compute_team_form(match_df, opponent, as_of_date)

    team_norm = _normalize_match_team(team)
    opp_norm = _normalize_match_team(opponent)
    h2h = match_df[
        (
            ((match_df["Home"] == team_norm) & (match_df["Away"] == opp_norm)) |
            ((match_df["Home"] == opp_norm) & (match_df["Away"] == team_norm))
        ) &
        (match_df["Date"] < as_of_date)
    ].sort_values("Date", ascending=False).head(6)

    wins = 0
    points = 0
    goal_diff_total = 0
    for _, match in h2h.iterrows():
        is_home = match["Home"] == team_norm
        gf = int(match["HomeGoals"] if is_home else match["AwayGoals"])
        ga = int(match["AwayGoals"] if is_home else match["HomeGoals"])
        goal_diff_total += (gf - ga)
        if gf > ga:
            wins += 1
            points += 3
        elif gf == ga:
            points += 1

    samples = len(h2h)
    h2h_win_ratio = round(wins / samples, 3) if samples else 0.0
    h2h_points_per_match = round(points / samples, 3) if samples else 0.0
    h2h_goal_diff_avg = round(goal_diff_total / samples, 3) if samples else 0.0

    fixture_edge_score = (
        (team_form["form_avg_last_5"] - opp_form["form_avg_last_5"])
        + ((team_form["goal_diff_last_5"] - opp_form["goal_diff_last_5"]) * 0.15)
        + (h2h_points_per_match * 0.25)
    )

    return {
        "opp_form_avg_last_5": opp_form["form_avg_last_5"],
        "opp_goal_diff_last_5": opp_form["goal_diff_last_5"],
        "opp_win_ratio_last_5": opp_form["win_ratio_last_5"],
        "h2h_matches_played": float(samples),
        "h2h_win_ratio": h2h_win_ratio,
        "h2h_points_per_match": h2h_points_per_match,
        "h2h_goal_diff_avg": h2h_goal_diff_avg,
        "fixture_edge_score": round(fixture_edge_score, 3),
    }


def refresh_with_api(artifacts, api_key, dry_run=False):
    """
    Refresh predictions using football-data.org API.

    Uses team schedule + player's historical playing time ratio to estimate
    individual workload. More accurate than pure team-level approximation.
    """
    from src.data_loaders.api_client import FootballDataClient

    print("\n2. Connecting to football-data.org API...")
    client = FootballDataClient(api_key)
    print("   Connected!")

    # Load player minutes lookup for scaling
    print("\n3. Loading player playing time data...")
    minutes_lookup = load_player_minutes_lookup()
    print(f"   Loaded playing time ratios for {len(minutes_lookup)} players")
    epl_signal_lookup = load_fpl_live_signal_lookup()

    # Fetch matches (PL + all other competitions for accurate workload)
    print("\n4. Fetching current season matches (PL + cups/European)...")
    now = datetime.now()
    season = now.year if now.month >= 8 else now.year - 1
    matches = client.get_all_matches_for_pl_teams(season=season)
    if "competition" in matches.columns:
        comp_counts = matches["competition"].value_counts().to_dict()
        comp_str = ", ".join(f"{v} {k}" for k, v in comp_counts.items())
        print(f"   Fetched {len(matches)} matches ({comp_str}) from {season}-{season+1}")
    else:
        print(f"   Fetched {len(matches)} matches from {season}-{season+1}")

    # Fetch squads
    print("\n5. Fetching current Premier League squads...")
    players = client.get_all_team_squads()
    print(f"   Fetched {len(players)} players from {players['team'].nunique()} teams")

    try:
        upcoming_epl = client.get_upcoming_fixtures(days_ahead=14)
    except Exception as e:
        logger.warning(f"Failed to fetch upcoming EPL fixtures for live context: {e}")
        upcoming_epl = pd.DataFrame()
    epl_next_fixture_map = build_next_fixture_map(upcoming_epl)

    # Build snapshots with player-scaled workload + team form
    print("\n6. Computing player workloads (team schedule × playing time)...")

    # Break detection: if no matches in the last 5 days, use the day after
    # the last match as the snapshot date. This prevents workload features from
    # collapsing to zero during international breaks. 5 days is safe because
    # even the longest midweek-to-weekend gap is only 4 days.
    now = datetime.now()
    if "Date" in matches.columns and len(matches) > 0:
        last_match_date = matches["Date"].max()
        if hasattr(last_match_date, "to_pydatetime"):
            last_match_date = last_match_date.to_pydatetime()
        days_since_last = (now - last_match_date).days
        if days_since_last > 5:
            snapshot_date = last_match_date + timedelta(days=1)
            print(f"   Break detected: last match was {days_since_last} days ago "
                  f"({last_match_date.strftime('%Y-%m-%d')})")
            print(f"   Using snapshot date {snapshot_date.strftime('%Y-%m-%d')} "
                  f"instead of today to preserve workload features")
        else:
            snapshot_date = now
    else:
        snapshot_date = now

    def _resolve_playing_time(player_name, team_name, match_df, team_snapshot, minutes_lkp):
        """Resolve player playing-time payload into a usable ratio + raw minutes."""
        play_entry = _lookup_minutes_payload(minutes_lkp, player_name, team_name)

        team_norm = _normalize_match_team(team_name)
        team_matches_played = len(match_df[
            ((match_df["Home"] == team_norm) | (match_df["Away"] == team_norm)) &
            (match_df["Date"] < team_snapshot)
        ])
        max_minutes = max(team_matches_played * 90, 90)

        if isinstance(play_entry, dict):
            minutes_played = int(play_entry.get("minutes_played", 0) or 0)
            appearances = int(play_entry.get("appearances", 0) or 0)
            ratio = play_entry.get("ratio")
            if ratio is None:
                ratio = min(minutes_played / max_minutes, 1.0) if minutes_played > 0 else 0.5
            return float(ratio), minutes_played, appearances

        if play_entry is not None:
            return float(play_entry), 0, 0

        return 0.5, 0, 0

    def _build_player_rows(
        squad_df,
        match_df,
        league_name,
        snapshot_dt,
        minutes_lkp,
        signal_lkp,
        next_fixture_map,
        ref_now=None,
    ):
        """Build workload snapshot rows for all players in a squad."""
        if ref_now is None:
            ref_now = datetime.now()
        team_workload_cache = {}
        team_form_cache = {}
        team_snapshot_cache = {}
        rows = []
        matched = 0
        for _, player in squad_df.iterrows():
            team = player["team"]
            if team not in team_workload_cache:
                # Per-team snapshot: if this team's last match was >5 days ago
                # (e.g. they played earlier in the GW), use day after their last
                # match instead of the global snapshot to avoid acute_load collapse.
                if team not in team_snapshot_cache:
                    team_norm = _normalize_match_team(team)
                    team_dates = match_df[
                        ((match_df["Home"] == team_norm) | (match_df["Away"] == team_norm)) &
                        (match_df["Date"] < ref_now)
                    ]["Date"]
                    if len(team_dates) > 0:
                        last_team_match = team_dates.max()
                        if hasattr(last_team_match, "to_pydatetime"):
                            last_team_match = last_team_match.to_pydatetime()
                        days_gap = (ref_now - last_team_match).days
                        team_snapshot_cache[team] = (
                            last_team_match + timedelta(days=1) if days_gap > 5 else snapshot_dt
                        )
                    else:
                        team_snapshot_cache[team] = snapshot_dt
                team_snap = team_snapshot_cache[team]
                team_workload_cache[team] = compute_team_workload(match_df, team, team_snap)
                team_form_cache[team] = compute_team_form(match_df, team, team_snap)

            team_workload = team_workload_cache[team]
            team_form = team_form_cache[team]
            team_snap = team_snapshot_cache.get(team, snapshot_dt)

            player_name = player["name"]
            play_ratio, minutes_played, appearances = _resolve_playing_time(
                player_name,
                team,
                match_df,
                team_snap,
                minutes_lkp,
            )
            signal_payload = _lookup_signal_payload(signal_lkp, player_name, team) or {}
            next_fixture = next_fixture_map.get(team, {})
            fixture_context = compute_fixture_context(
                match_df,
                team,
                str(next_fixture.get("opponent", "")).strip(),
                team_snap,
            )

            player_acute = team_workload["acute_load"]
            player_chronic = team_workload["chronic_load"]
            team_acwr = team_workload["acwr"]

            if play_ratio >= 0.15:
                if play_ratio >= 0.6:
                    player_acwr = team_acwr
                else:
                    spike_factor = 1.0 + 0.15 * (1.0 - abs(2 * play_ratio - 0.8))
                    player_acwr = team_acwr * max(1.0, spike_factor)
            else:
                player_acwr = 0.9

            player_acwr = max(0.5, min(2.5, player_acwr))
            player_fatigue = player_acute - player_chronic

            scaled_workload = {
                "acute_load": round(player_acute, 2),
                "chronic_load": round(player_chronic, 2),
                "acwr": round(player_acwr, 3),
                "monotony": team_workload["monotony"],
                "strain": team_workload["strain"],
                "fatigue_index": round(player_fatigue, 2),
                "workload_slope": team_workload["workload_slope"],
                "spike_flag": 1 if player_acwr > 1.8 else 0,
                "matches_last_7": team_workload["matches_last_7"],
                "matches_last_14": team_workload["matches_last_14"],
                "matches_last_30": team_workload["matches_last_30"],
            }

            if _lookup_minutes_payload(minutes_lkp, player_name, team):
                matched += 1

            rest_days = team_form["rest_days_before_injury"]
            rows.append({
                "name": player_name,
                "player_team": player["team"],
                "team": player["team"],
                "position": player.get("position", "Unknown"),
                "age": player.get("age", 25),
                "league": league_name,
                "injury_datetime": team_snap,
                "snapshot_date": team_snap.strftime("%Y-%m-%d"),
                "playing_time_ratio": play_ratio,
                "minutes_played": minutes_played,
                "appearances": appearances,
                "goals": int(signal_payload.get("goals", 0) or 0),
                "assists": int(signal_payload.get("assists", 0) or 0),
                "goals_per_90": float(signal_payload.get("goals_per_90", 0) or 0.0),
                "assists_per_90": float(signal_payload.get("assists_per_90", 0) or 0.0),
                "shots_per_90": float(signal_payload.get("shots_per_90", 0) or 0.0),
                "saves_per_90": float(signal_payload.get("saves_per_90", 0) or 0.0),
                "days_since_last_match": rest_days,
                **fixture_context,
                **scaled_workload,
                **team_form,
            })
        print(f"   {matched}/{len(squad_df)} {league_name} players matched with playing time data")
        return rows

    # --- Premier League ---
    # Displayed season stats (apps/goals/assists/minutes) come from FBref so PL
    # cards show real appearances and official goals/assists (matching what fans
    # verify against), not FPL's minutes/90 proxy. FPL fills only where FBref has
    # no record, and stays the live source for price/ownership/form at serve time.
    pl_fb_minutes, pl_fb_signal = load_fbref_stats_lookups(players, league_name="Premier League")
    # Precedence FBref > artifact(prior pkl) > FPL. On Render (no Chrome) FBref
    # returns nothing, so the previously-shipped correct stats must beat FPL's
    # minutes/90 "appearances" and divergent assists rather than reverting.
    pl_artifact_minutes = build_artifact_minutes_lookup(players, artifacts.get("inference_df"), "Premier League")
    pl_artifact_signal = build_artifact_signal_lookup(players, artifacts.get("inference_df"), "Premier League")
    pl_minutes_lookup = _merge_minutes_lookup(
        _merge_minutes_lookup(pl_fb_minutes, pl_artifact_minutes),  # FBref over artifact
        minutes_lookup,  # FPL fills only players neither had
    )
    pl_signal_lookup = {**epl_signal_lookup, **pl_artifact_signal, **pl_fb_signal}  # fbref > artifact > fpl
    epl_rows = _build_player_rows(
        players,
        matches,
        "Premier League",
        snapshot_date,
        pl_minutes_lookup,
        pl_signal_lookup,
        epl_next_fixture_map,
        ref_now=now,
    )

    # --- La Liga (+ UCL for La Liga teams) ---
    print("\n   Fetching La Liga squads and matches...")
    try:
        la_liga_players = client.get_all_la_liga_squads(season=season)
        la_liga_matches = client.get_la_liga_matches(season=season, status="FINISHED")
        print(f"   La Liga: {len(la_liga_players)} players, {len(la_liga_matches)} La Liga matches")
        try:
            upcoming_laliga = client.get_upcoming_la_liga_fixtures(days_ahead=14)
        except Exception as e:
            logger.warning(f"Failed to fetch upcoming La Liga fixtures for live context: {e}")
            upcoming_laliga = pd.DataFrame()
        laliga_next_fixture_map = build_next_fixture_map(upcoming_laliga)

        # Add UCL matches for La Liga teams (same API call we already do for EPL)
        try:
            cl_all = client.get_champions_league_matches(season=season)
            if len(cl_all) > 0:
                la_liga_teams = set(la_liga_matches["Home"].unique()) | set(la_liga_matches["Away"].unique())
                # UCL uses same name normalisation as La Liga via _normalize_la_liga_team in get_la_liga_matches
                # CL uses _normalize_team_name (EPL normaliser) — map CL names to La Liga format
                cl_la = cl_all[
                    cl_all["Home"].isin(la_liga_teams) | cl_all["Away"].isin(la_liga_teams)
                ].copy()
                if len(cl_la) > 0:
                    if "league" not in cl_la.columns:
                        cl_la["league"] = "La Liga"
                    la_liga_matches = pd.concat([la_liga_matches, cl_la], ignore_index=True)
                    la_liga_matches = la_liga_matches.drop_duplicates(
                        subset=["Date", "Home", "Away"], keep="first"
                    )
                    print(f"   + {len(cl_la)} UCL matches → {len(la_liga_matches)} total La Liga workload matches")
        except Exception as cl_e:
            print(f"   Warning: UCL fetch failed ({cl_e}), using La Liga only")

        public_minutes_lookup, public_matches = load_laliga_public_minutes_lookup(la_liga_players)
        fbref_minutes_lookup = load_fbref_minutes_lookup(
            la_liga_players,
            league_name="La Liga",
        )
        artifact_minutes_lookup = build_artifact_minutes_lookup(
            la_liga_players,
            artifacts.get("inference_df"),
            "La Liga",
        )
        laliga_signal_lookup = load_laliga_public_signal_lookup(la_liga_players)

        la_liga_minutes_lookup = {}
        la_liga_minutes_lookup = _merge_minutes_lookup(la_liga_minutes_lookup, public_minutes_lookup)
        la_liga_minutes_lookup = _merge_minutes_lookup(la_liga_minutes_lookup, fbref_minutes_lookup)
        la_liga_minutes_lookup = _merge_minutes_lookup(la_liga_minutes_lookup, artifact_minutes_lookup)

        total_players = len(la_liga_players)
        fbref_matches = _count_minutes_matches(la_liga_players, fbref_minutes_lookup)
        artifact_matches = _count_minutes_matches(la_liga_players, artifact_minutes_lookup)
        final_matches = _count_minutes_matches(la_liga_players, la_liga_minutes_lookup)
        print(
            f"   La Liga minutes coverage: {final_matches}/{total_players} players "
            f"(public {public_matches}, FBref {fbref_matches}, artifact {artifact_matches})"
        )
        la_liga_rows = _build_player_rows(
            la_liga_players,
            la_liga_matches,
            "La Liga",
            snapshot_date,
            la_liga_minutes_lookup,
            laliga_signal_lookup,
            laliga_next_fixture_map,
            ref_now=now,
        )
    except Exception as e:
        print(f"   Warning: could not fetch La Liga data ({e}). Falling back to existing La Liga predictions.")
        # Preserve existing La Liga rows from the last good pkl so the saved
        # artifact is never EPL-only. Without this, a transient API failure
        # wipes all La Liga players from the live inference_df.
        existing_df = artifacts.get("inference_df")
        if existing_df is not None and "league" in existing_df.columns:
            existing_la_liga = existing_df[existing_df["league"] == "La Liga"]
            la_liga_rows = existing_la_liga.to_dict("records") if not existing_la_liga.empty else []
            print(f"   Preserved {len(la_liga_rows)} existing La Liga predictions.")
        else:
            la_liga_rows = []

    # --- Bundesliga (+ UCL for Bundesliga teams) ---
    # Same shape as La Liga: squads + matches + minutes lookup → workload rows.
    # No public-stats fallback (Bundesliga has no equivalent of LaLiga official
    # leaderboards), so we lean on FBref minutes + the prior artifact pickle.
    print("\n   Fetching Bundesliga squads and matches...")
    try:
        bl_players = client.get_all_bundesliga_squads(season=season)
        bl_matches = client.get_bundesliga_matches(season=season, status="FINISHED")
        print(f"   Bundesliga: {len(bl_players)} players, {len(bl_matches)} Bundesliga matches")

        # Add UCL matches for Bundesliga teams to capture European workload.
        try:
            cl_all = client.get_champions_league_matches(season=season)
            if len(cl_all) > 0:
                bl_teams = set(bl_matches["Home"].unique()) | set(bl_matches["Away"].unique())
                cl_bl = cl_all[cl_all["Home"].isin(bl_teams) | cl_all["Away"].isin(bl_teams)].copy()
                if len(cl_bl) > 0:
                    if "league" not in cl_bl.columns:
                        cl_bl["league"] = "Bundesliga"
                    bl_matches = pd.concat([bl_matches, cl_bl], ignore_index=True)
                    bl_matches = bl_matches.drop_duplicates(
                        subset=["Date", "Home", "Away"], keep="first"
                    )
                    print(f"   + {len(cl_bl)} UCL matches → {len(bl_matches)} total Bundesliga workload matches")
        except Exception as cl_e:
            print(f"   Warning: UCL fetch failed ({cl_e}), using Bundesliga only")

        fbref_minutes_lookup = load_fbref_minutes_lookup(bl_players, league_name="Bundesliga")
        artifact_minutes_lookup = build_artifact_minutes_lookup(
            bl_players, artifacts.get("inference_df"), "Bundesliga",
        )

        bl_minutes_lookup = {}
        bl_minutes_lookup = _merge_minutes_lookup(bl_minutes_lookup, fbref_minutes_lookup)
        bl_minutes_lookup = _merge_minutes_lookup(bl_minutes_lookup, artifact_minutes_lookup)

        total_bl = len(bl_players)
        fbref_bl_matches = _count_minutes_matches(bl_players, fbref_minutes_lookup)
        artifact_bl_matches = _count_minutes_matches(bl_players, artifact_minutes_lookup)
        final_bl_matches = _count_minutes_matches(bl_players, bl_minutes_lookup)
        print(
            f"   Bundesliga minutes coverage: {final_bl_matches}/{total_bl} players "
            f"(FBref {fbref_bl_matches}, artifact {artifact_bl_matches})"
        )

        bl_rows = _build_player_rows(
            bl_players,
            bl_matches,
            "Bundesliga",
            snapshot_date,
            bl_minutes_lookup,
            {},   # no public output-signal lookup yet for Bundesliga
            {},   # no next-fixture map yet; UCL/BL fixtures wired in a follow-up
            ref_now=now,
        )
    except Exception as e:
        print(f"   Warning: could not fetch Bundesliga data ({e}). Preserving existing predictions.")
        existing_df = artifacts.get("inference_df")
        if existing_df is not None and "league" in existing_df.columns:
            existing_bl = existing_df[existing_df["league"] == "Bundesliga"]
            bl_rows = existing_bl.to_dict("records") if not existing_bl.empty else []
            print(f"   Preserved {len(bl_rows)} existing Bundesliga predictions.")
        else:
            bl_rows = []

    # --- Serie A (+ UCL for Serie A teams) ---
    # Same shape as Bundesliga: squads + matches + minutes lookup → workload rows.
    print("\n   Fetching Serie A squads and matches...")
    try:
        sa_players = client.get_all_serie_a_squads(season=season)
        sa_matches = client.get_serie_a_matches(season=season, status="FINISHED")
        print(f"   Serie A: {len(sa_players)} players, {len(sa_matches)} Serie A matches")

        # Add UCL matches for Serie A teams to capture European workload.
        try:
            cl_all = client.get_champions_league_matches(season=season)
            if len(cl_all) > 0:
                sa_teams = set(sa_matches["Home"].unique()) | set(sa_matches["Away"].unique())
                cl_sa = cl_all[cl_all["Home"].isin(sa_teams) | cl_all["Away"].isin(sa_teams)].copy()
                if len(cl_sa) > 0:
                    if "league" not in cl_sa.columns:
                        cl_sa["league"] = "Serie A"
                    sa_matches = pd.concat([sa_matches, cl_sa], ignore_index=True)
                    sa_matches = sa_matches.drop_duplicates(
                        subset=["Date", "Home", "Away"], keep="first"
                    )
                    print(f"   + {len(cl_sa)} UCL matches → {len(sa_matches)} total Serie A workload matches")
        except Exception as cl_e:
            print(f"   Warning: UCL fetch failed ({cl_e}), using Serie A only")

        fbref_minutes_lookup = load_fbref_minutes_lookup(sa_players, league_name="Serie A")
        artifact_minutes_lookup = build_artifact_minutes_lookup(
            sa_players, artifacts.get("inference_df"), "Serie A",
        )

        sa_minutes_lookup = {}
        sa_minutes_lookup = _merge_minutes_lookup(sa_minutes_lookup, fbref_minutes_lookup)
        sa_minutes_lookup = _merge_minutes_lookup(sa_minutes_lookup, artifact_minutes_lookup)

        total_sa = len(sa_players)
        fbref_sa_matches = _count_minutes_matches(sa_players, fbref_minutes_lookup)
        artifact_sa_matches = _count_minutes_matches(sa_players, artifact_minutes_lookup)
        final_sa_matches = _count_minutes_matches(sa_players, sa_minutes_lookup)
        print(
            f"   Serie A minutes coverage: {final_sa_matches}/{total_sa} players "
            f"(FBref {fbref_sa_matches}, artifact {artifact_sa_matches})"
        )

        sa_rows = _build_player_rows(
            sa_players,
            sa_matches,
            "Serie A",
            snapshot_date,
            sa_minutes_lookup,
            {},   # no public output-signal lookup yet for Serie A
            {},   # no next-fixture map yet
            ref_now=now,
        )
    except Exception as e:
        print(f"   Warning: could not fetch Serie A data ({e}). Preserving existing predictions.")
        existing_df = artifacts.get("inference_df")
        if existing_df is not None and "league" in existing_df.columns:
            existing_sa = existing_df[existing_df["league"] == "Serie A"]
            sa_rows = existing_sa.to_dict("records") if not existing_sa.empty else []
            print(f"   Preserved {len(sa_rows)} existing Serie A predictions.")
        else:
            sa_rows = []

    all_rows = epl_rows + la_liga_rows + bl_rows + sa_rows
    snapshots_df = pd.DataFrame(all_rows)
    print(
        f"\n   Total: {len(epl_rows)} EPL + {len(la_liga_rows)} La Liga + "
        f"{len(bl_rows)} Bundesliga + {len(sa_rows)} Serie A = {len(snapshots_df)} players"
    )
    return snapshots_df


# =============================================================================
# INFERENCE
# =============================================================================

def assign_archetype_heuristic(row):
    """
    Assign injury-focused archetype based on injury patterns, not raw counts.

    Key insight: Total injuries means nothing. What matters is:
    - Severity (avg days per injury)
    - Recency (how recently injured)
    - Impact (total days lost)

    Archetypes:
    - Durable: Rarely injured, quick recovery when it happens
    - Fragile: When injured, out for long periods (high severity)
    - Recurring: Frequent minor injuries, often returns quickly
    - Currently Vulnerable: Recently injured or returning
    - Moderate Risk: Average injury profile
    - Clean Record: No significant injury history
    """
    prev_injuries = row.get("previous_injuries", 0)
    days_since = row.get("days_since_last_injury", 365)
    total_days_lost = row.get("total_days_lost", 0)
    age = row.get("age", 25)

    # Calculate severity metrics
    avg_days_per_injury = total_days_lost / prev_injuries if prev_injuries > 0 else 0

    # Priority 1: Currently vulnerable (recently injured or returning)
    if days_since < 60:
        return "Currently Vulnerable"

    # Priority 2: No injury history = clean record
    if prev_injuries == 0:
        return "Clean Record"

    # Priority 3: Classify by injury SEVERITY (avg days per injury)
    # Low severity (<20 days avg) = minor injuries, quick recovery
    # High severity (>40 days avg) = serious injuries, long absences

    if avg_days_per_injury >= 40:
        # Severe injuries when they happen
        return "Fragile"

    if avg_days_per_injury >= 25:
        # Moderate severity injuries
        if prev_injuries >= 10:
            return "Injury Prone"
        else:
            return "Moderate Risk"

    if avg_days_per_injury < 20:
        # Quick recovery from injuries
        if prev_injuries >= 10 and days_since < 180:
            return "Recurring"  # Frequent but minor
        else:
            return "Durable"  # Minor issues, handles them well

    # Default: moderate risk
    return "Moderate Risk"


def calibrate_live_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a conservative live-data calibration layer.

    The retrained ensemble can rank players well while still running too hot on
    current snapshots because live rows do not perfectly mirror the historical
    training distribution. Blend the model with a feature-grounded baseline, then
    shrink only when the pool-level mean clearly drifts too high.
    """
    out = df.copy()
    raw = pd.to_numeric(out.get("ensemble_prob", 0.5), errors="coerce").fillna(0.5).clip(0.01, 0.995)
    out["ensemble_prob_raw"] = raw

    recent_pressure = pd.to_numeric(out.get("recent_injury_pressure", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    injury_burden = pd.to_numeric(out.get("injury_burden_index", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    acwr = pd.to_numeric(out.get("acwr", 1.0), errors="coerce").fillna(1.0)
    player_importance = pd.to_numeric(out.get("player_importance_score", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    fixture_edge = pd.to_numeric(out.get("fixture_edge_score", 0.0), errors="coerce").fillna(0.0)

    load_pressure = ((acwr - 1.0) / 0.9).clip(0.0, 1.0)
    fixture_pressure = ((-fixture_edge) / 3.0).clip(0.0, 1.0)

    baseline = (
        0.06
        + recent_pressure * 0.28
        + injury_burden * 0.24
        + load_pressure * 0.18
        + player_importance * 0.12
        + fixture_pressure * 0.06
    ).clip(0.03, 0.82)

    calibrated = ((raw * 0.62) + (baseline * 0.38)).clip(0.02, 0.95)
    current_mean = float(calibrated.mean())
    if current_mean > 0.55:
        target_mean = 0.42
        try:
            gamma = max(1.0, min(8.0, float(np.log(target_mean) / np.log(current_mean))))
        except (ValueError, ZeroDivisionError):
            gamma = 2.5
        calibrated = calibrated.pow(gamma).clip(0.02, 0.95)

    out["ensemble_prob"] = calibrated.round(6)
    return out


def run_inference(snapshots_df, ensemble, severity_clf, player_history, archetype_df, use_transfermarkt=False):
    """
    Run model inference on player snapshots.

    Applies full feature engineering pipeline to match training data format.

    Args:
        use_transfermarkt: If True, fetch injury history from Transfermarkt for
                          players not found in local CSV (slower but more complete)
    """
    from src.feature_engineering.classification import add_contextual_classification_features
    from src.inference.inference_pipeline import apply_all_feature_engineering, add_player_history_features

    print("\n7. Running inference...")

    df = snapshots_df.copy()

    # Ensure required columns exist for feature engineering
    if "player_team" not in df.columns and "team" in df.columns:
        df["player_team"] = df["team"]

    # Apply full feature engineering (temporal, position, interactions)
    print("   Applying feature engineering...")
    df = apply_all_feature_engineering(
        df,
        date_column="injury_datetime",
        position_column="position",
        team_column="player_team"
    )

    # Load real injury history from CSV (not just the saved lookup)
    print("   Loading injury history from source data...")
    injury_history = load_injury_history_lookup()
    print(f"   Found injury records for {len(injury_history)} players in CSV")

    # Fetch from Transfermarkt for players not in local CSV
    player_names = df["name"].dropna().unique().tolist()
    if use_transfermarkt:
        injury_history = fetch_transfermarkt_injuries(player_names, injury_history)
        print(f"   Total injury records after Transfermarkt: {len(injury_history)}")

    # Compute full injury history features from detail pkl (most accurate source)
    detail_stats = {}
    for detail_path in [
        os.path.join(PROJECT_ROOT, "models", "player_injuries_detail.pkl"),
        os.path.join(PROJECT_ROOT, "models", "laliga_injuries_detail.pkl"),
    ]:
        if not os.path.exists(detail_path):
            continue
        try:
            detail_df = pd.read_pickle(detail_path)
            severity_col = "severity_days" if "severity_days" in detail_df.columns else "days_out" if "days_out" in detail_df.columns else None
            if severity_col and "name" in detail_df.columns:
                for name, grp in detail_df.groupby("name"):
                    payload = {
                        "worst": grp[severity_col].max(),
                        "std": grp[severity_col].std() if len(grp) > 1 else 0.0,
                    }
                    detail_stats[name] = payload
                    detail_stats[_canonical_player_key(name)] = payload
        except Exception as e:
            logger.debug(f"Failed to load detail stats from {os.path.basename(detail_path)}: {e}")

    # Merge real injury history (scraped data is the primary source)
    matched_injuries = 0
    for idx, row in df.iterrows():
        player_name = row.get("name", "")
        hist = _lookup_injury_payload(injury_history, str(player_name))
        if hist:
            count = hist["previous_injuries"]
            avg_sev = hist.get("avg_severity", 0)
            total_lost = hist.get("total_days_lost", 0)
            ds = detail_stats.get(player_name, {}) or detail_stats.get(_canonical_player_key(player_name), {})
            df.at[idx, "previous_injuries"] = count
            if hist.get("days_since_last_injury") is not None:
                df.at[idx, "days_since_last_injury"] = hist["days_since_last_injury"]
            df.at[idx, "total_days_lost"] = total_lost
            df.at[idx, "player_injury_count"] = count
            df.at[idx, "player_avg_severity"] = avg_sev
            df.at[idx, "player_worst_injury"] = ds.get("worst", avg_sev)
            df.at[idx, "player_severity_std"] = ds.get("std", 0.0)
            df.at[idx, "is_injury_prone"] = 1 if count >= 3 else 0
            matched_injuries += 1

    print(f"   Matched {matched_injuries}/{len(df)} players with injury history")

    # Set defaults for players without injury history
    if "previous_injuries" not in df.columns:
        df["previous_injuries"] = 0
    df["previous_injuries"] = df["previous_injuries"].fillna(0)

    if "days_since_last_injury" not in df.columns:
        df["days_since_last_injury"] = 365  # 1 year default for unknown
    df["days_since_last_injury"] = df["days_since_last_injury"].fillna(365)

    # --- Derived injury pattern features ---
    # overdue_ratio: how far past the player's average inter-injury gap they are.
    # > 1.0 means they are statistically "due" for another injury.
    if "avg_inter_injury_gap" in df.columns:
        gap = df["avg_inter_injury_gap"].replace(0, np.nan)
        df["overdue_ratio"] = (df["days_since_last_injury"] / gap).clip(0, 5).fillna(0)
    else:
        df["overdue_ratio"] = 0.0

    # age_x_injury_count: amplifies risk for older players with high injury burden
    if "player_injury_count" in df.columns and "age" in df.columns:
        df["age_x_injury_count"] = (
            df["age"].fillna(25) * df["player_injury_count"].fillna(0)
        ).clip(0, 500)
    else:
        df["age_x_injury_count"] = 0.0

    # acwr_x_play_ratio: high workload spike + high playing time = amplified risk
    if "acwr" in df.columns and "playing_time_ratio" in df.columns:
        df["acwr_x_play_ratio"] = (
            df["acwr"].fillna(1.0) * df["playing_time_ratio"].fillna(0.5)
        ).clip(0, 5)
    else:
        df["acwr_x_play_ratio"] = 0.0

    # Assign archetypes using heuristic (instead of just merging old assignments)
    print("   Assigning player archetypes...")
    df["archetype"] = df.apply(assign_archetype_heuristic, axis=1)
    df = add_contextual_classification_features(df)

    # Get expected features from model
    feature_cols = ensemble.feature_names_
    missing_features = [c for c in feature_cols if c not in df.columns]

    # Set sensible defaults for known missing features (not just zeros)
    feature_defaults = {
        "fifa_rating": 70,  # Average rating
        "rest_days_before_injury": 4,  # Typical rest between matches
        "avg_rest_last_5": 4,
        "goals_for_last_5": 7,  # ~1.4 per match
        "goals_against_last_5": 5,
        "goal_diff_last_5": 2,
        "avg_goal_diff_last_5": 0.4,
        "form_last_5": 7,  # ~1.4 pts per match
        "form_avg_last_5": 1.4,
        "win_ratio_last_5": 0.4,
        "win_streak": 0,
        "loss_streak": 0,
        "opp_form_avg_last_5": 1.4,
        "opp_goal_diff_last_5": 0.0,
        "opp_win_ratio_last_5": 0.4,
        "h2h_matches_played": 0.0,
        "h2h_win_ratio": 0.0,
        "h2h_points_per_match": 0.0,
        "fixture_edge_score": 0.0,
        "minutes_share": 0.45,
        "starter_ratio": 0.5,
        "goal_involvement_per90": 0.2,
        "shot_volume_per90": 0.8,
        "creative_actions_per90": 0.8,
        "player_importance_score": 0.45,
        "days_since_last_injury_capped": 180,
        "recent_injury_pressure": 0.0,
        "injury_burden_index": 0.0,
        "overdue_ratio": 0.0,
        "age_x_injury_count": 0.0,
        "acwr_x_play_ratio": 0.0,
        "avg_inter_injury_gap": 0.0,
        "min_inter_injury_gap": 0.0,
        "same_area_recurrence": 0,
        "recurring_injury_type": 0,
        "injuries_last_12m": 0,
        "injuries_last_24m": 0,
        "days_lost_last_12m": 0.0,
    }

    if missing_features:
        print(f"   Warning: {len(missing_features)} missing features, filling with defaults")
        logger.warning(f"Missing features after engineering: {missing_features}")
        for col in missing_features:
            df[col] = feature_defaults.get(col, 0)

    X = df[feature_cols].fillna(0)

    # Ensemble predictions
    print("   Running ensemble predictions...")

    # Suppress sklearn feature name warnings (expected when ensemble converts to numpy)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        probs = ensemble.predict_proba(X)
        df["ensemble_prob"] = probs[:, 1] if len(probs.shape) > 1 else probs

        # Individual base model predictions (using get_base_predictions method)
        base_preds = ensemble.get_base_predictions(X)
        df["lgb_prob"] = base_preds["lgb_prob"]
        df["xgb_prob"] = base_preds["xgb_prob"]
        df["catboost_prob"] = base_preds["catboost_prob"]

    raw_mean = df["ensemble_prob"].mean()

    # Apply isotonic calibrator if available — reshapes the bimodal output
    # into a proper probability distribution before heuristic mean-correction.
    _cal_path = os.path.join(PROJECT_ROOT, "models", "probability_calibrator.pkl")
    if os.path.exists(_cal_path):
        try:
            import pickle as _pkl
            with open(_cal_path, "rb") as _f:
                _calibrator = _pkl.load(_f)
            df["ensemble_prob"] = _calibrator.predict(df["ensemble_prob"].values).clip(0.01, 0.99)
            print(f"   Isotonic calibration applied: mean {raw_mean:.1%} → {df['ensemble_prob'].mean():.1%}")
        except Exception as _cal_e:
            print(f"   WARNING: Calibrator load failed ({_cal_e}), using raw probs")

    df = calibrate_live_probabilities(df)
    print(f"   Post-calibration probs: mean={df['ensemble_prob'].mean():.1%}, "
          f"range=[{df['ensemble_prob'].min():.1%}, {df['ensemble_prob'].max():.1%}]")
    if abs(df["ensemble_prob"].mean() - raw_mean) > 0.05:
        print(f"   Live calibration adjusted pool mean from {raw_mean:.1%} to {df['ensemble_prob'].mean():.1%}")

    # Agreement: how many models agree the player is above-median risk
    median_prob = df["ensemble_prob"].median()
    model_probs = np.column_stack([df["lgb_prob"], df["xgb_prob"], df["catboost_prob"]])
    df["agreement"] = (model_probs > median_prob).sum(axis=1)
    df["confidence"] = df["agreement"].map({3: "very-high", 2: "high", 1: "medium", 0: "low"})

    # Severity prediction using heuristic for live data
    # (Severity classifier requires 202 features including match-level data we don't have)
    # Heuristic: based on workload spike, age, and injury history
    def estimate_severity(row):
        """Estimate severity class based on available features."""
        score = 0

        # High workload spike → more severe injuries
        if row.get("spike_flag", 0) == 1:
            score += 1
        if row.get("acwr", 1.0) > 1.8:
            score += 1

        # Age factor (older players → longer recovery)
        age = row.get("age", 25)
        if age >= 32:
            score += 1
        elif age >= 28:
            score += 0.5

        # Injury history (injury-prone players → more severe)
        if row.get("is_injury_prone", 0) == 1:
            score += 1
        if row.get("previous_injuries", 0) >= 5:
            score += 0.5

        # Map score to severity class
        if score >= 2.5:
            return "long"
        elif score >= 1:
            return "medium"
        else:
            return "short"

    df["predicted_severity_class"] = df.apply(estimate_severity, axis=1)

    print(f"   Inference complete. Mean risk: {df['ensemble_prob'].mean():.1%}")
    return df


# =============================================================================
# MAIN
# =============================================================================

def run_injury_scraper(max_age_days: int = 1):
    """Run the injury scraper to update player_injuries_detail.pkl.

    Only re-scrapes players whose data is older than max_age_days,
    so this is fast when run frequently (only new/stale players).
    """
    import subprocess
    script = os.path.join(os.path.dirname(__file__), "scrape_injuries.py")
    if not os.path.exists(script):
        logger.warning("scrape_injuries.py not found, skipping injury update")
        return False

    print(f"\n   Running injury scraper (max-age={max_age_days} days)...")
    try:
        result = subprocess.run(
            [sys.executable, script, "--max-age", str(max_age_days)],
            capture_output=True, text=True, timeout=1800  # 30 min max
        )
        # Print summary lines only
        for line in result.stdout.splitlines():
            if any(kw in line.lower() for kw in ["done!", "scraped", "final", "players to scrape", "nothing to do"]):
                print(f"   {line.strip()}")
        if result.returncode != 0:
            logger.warning(f"Scraper exited with code {result.returncode}")
            if result.stderr:
                logger.warning(result.stderr[-500:])
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Injury scraper timed out after 30 minutes")
        return False
    except Exception as e:
        logger.warning(f"Injury scraper failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Refresh predictions with live data")
    # --mode kept for backwards-compat with the Render cron's CLI call but
    # only ``api`` is supported. The legacy ``fbref`` per-player workload mode
    # was removed when our custom FBref scraper started 403ing; FBref data
    # now flows through ``soccerdata`` for season minutes only.
    parser.add_argument("--mode", choices=["api"], default="api",
                        help="Data source (only 'api' is supported now)")
    parser.add_argument("--api-key", help="Football-data.org API key")
    parser.add_argument("--dry-run", action="store_true", help="Don't save, just preview")
    parser.add_argument("--players", help="Comma-separated player names to filter to")
    parser.add_argument("--transfermarkt", action="store_true",
                        help="Fetch injury history from Transfermarkt for missing players (slower)")
    parser.add_argument("--scrape-injuries", action="store_true",
                        help="Update injury history from Transfermarkt before refreshing predictions")
    parser.add_argument("--scrape-max-age", type=int, default=1,
                        help="Max age in days before re-scraping a player's injury data (default: 1)")
    args = parser.parse_args()

    print("=" * 60)
    print("INJURY RISK PREDICTOR - Live Data Refresh")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")

    # Step 0: Update injury history if requested
    if args.scrape_injuries:
        print("\n0. Updating injury history from Transfermarkt...")
        run_injury_scraper(max_age_days=args.scrape_max_age)

    # Load models
    print("\n1. Loading trained models...")
    artifacts = load_artifacts()
    if artifacts is None or "ensemble" not in artifacts:
        print("ERROR: No trained models found. Run the notebook first.")
        sys.exit(1)

    ensemble = artifacts["ensemble"]
    severity_clf = artifacts.get("severity_clf")
    player_history = artifacts.get("player_history", {})
    archetype_df = artifacts.get("df_clusters")
    print(f"   Loaded models with {len(player_history)} player histories")

    api_key = args.api_key or os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        print("ERROR: API key required.")
        print("Set FOOTBALL_DATA_API_KEY env var or use --api-key")
        sys.exit(1)
    snapshots_df = refresh_with_api(artifacts, api_key, args.dry_run)

    # Run inference
    inference_df = run_inference(
        snapshots_df, ensemble, severity_clf, player_history, archetype_df,
        use_transfermarkt=args.transfermarkt
    )

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    high_risk = inference_df[inference_df["ensemble_prob"] >= 0.6]
    medium_risk = inference_df[(inference_df["ensemble_prob"] >= 0.4) & (inference_df["ensemble_prob"] < 0.6)]
    low_risk = inference_df[inference_df["ensemble_prob"] < 0.4]

    print(f"\nRisk distribution ({len(inference_df)} players):")
    print(f"  High risk (>=60%):    {len(high_risk):3d} ({len(high_risk)/len(inference_df)*100:.1f}%)")
    print(f"  Medium risk (40-60%): {len(medium_risk):3d} ({len(medium_risk)/len(inference_df)*100:.1f}%)")
    print(f"  Low risk (<40%):      {len(low_risk):3d} ({len(low_risk)/len(inference_df)*100:.1f}%)")

    print(f"\nTop 10 highest risk players:")
    top_risk = inference_df.nlargest(10, "ensemble_prob")[["name", "player_team", "ensemble_prob", "acwr"]]
    for _, row in top_risk.iterrows():
        acwr_str = f"ACWR={row.get('acwr', 'N/A')}" if pd.notna(row.get('acwr')) else ""
        print(f"  {row['ensemble_prob']:.1%} - {row['name']} ({row.get('player_team', 'Unknown')}) {acwr_str}")

    # Guard: never overwrite the pkl with an EPL-only result.
    # If La Liga fetch failed, the snapshots_df has no La Liga rows.
    # Saving it would wipe La Liga from the live inference_df permanently
    # until the next successful refresh. Skip the save instead.
    has_la_liga = (
        "league" in inference_df.columns
        and "La Liga" in inference_df["league"].values
    )
    if not has_la_liga and not args.dry_run:
        print("\nWARNING: La Liga players missing from result — skipping save to preserve existing pkl.")
        print("La Liga fetch must have failed. Existing predictions remain unchanged.")
        sys.exit(0)

    # Per-league preservation guard: if a league's squad fetch degraded badly
    # (e.g. football-data 403s on most Bundesliga teams), the fresh frame has far
    # fewer players than before. Rather than ship that regression, keep the
    # existing rows for any league that came back below 60% of its prior count.
    # Mirrors the La Liga guard above but per-league and non-fatal.
    existing_idf = artifacts.get("inference_df")
    if existing_idf is not None and "league" in getattr(existing_idf, "columns", []) and not args.dry_run:
        prior_counts = existing_idf["league"].value_counts().to_dict()
        fresh_counts = inference_df["league"].value_counts().to_dict()
        preserved = []
        for league, prior_n in prior_counts.items():
            fresh_n = fresh_counts.get(league, 0)
            if prior_n >= 50 and fresh_n < 0.6 * prior_n:
                print(f"\nWARNING: {league} returned {fresh_n}/{prior_n} players (<60%); "
                      f"preserving existing {league} rows instead of shipping a regression.")
                inference_df = inference_df[inference_df["league"] != league]
                preserved.append(existing_idf[existing_idf["league"] == league])
        if preserved:
            inference_df = pd.concat([inference_df] + preserved, ignore_index=True)

    # Save
    if args.dry_run:
        print("\n[DRY RUN - not saving]")
    else:
        print("\n7. Saving updated predictions...")
        save_artifacts(
            ensemble=ensemble,
            severity_clf=severity_clf,
            df_clusters=archetype_df,
            player_history=player_history,
            inference_df=inference_df,
        )
        print(f"   Saved {len(inference_df)} player predictions")

    print("\nDone! Run 'streamlit run app.py' to see live predictions.")


if __name__ == "__main__":
    main()
