"""
FastAPI backend for EPL Injury Risk Predictor.

Serves predictions from the trained ML models to the React frontend.
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Set
import sys
import os
import json
from datetime import datetime, timedelta
import asyncio
import subprocess
import threading
import re
import time
import math

import unicodedata
from pathlib import Path
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

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
load_dotenv(ROOT / ".env")

from src.utils.logger import get_logger
logger = get_logger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.model_io import load_artifacts
from src.inference.story_generator import (
    generate_player_story,
    generate_risk_factors_list,
    get_recommendation_text,
    get_fpl_insight,
    calculate_scoring_odds,
    get_fpl_value_assessment,
    calculate_clean_sheet_odds,
    generate_yara_response,
    generate_lab_notes,
)
from src.inference.llm_client import generate_grounded_narrative
from src.data_loaders.fpl_api import FPLClient, get_fpl_insights
from src.data_loaders.football_data_api import FootballDataClient, get_standings_summary
from src.data_loaders.api_client import FootballDataClient as MatchHistoryApiClient, LA_LIGA_ID
from src.data_loaders.odds_api import OddsClient, get_clean_sheet_insight
# Archetype clustering imports are done lazily inside assign_hybrid_archetypes()
# to avoid importing all of src.models (which pulls in lightgbm, xgboost, etc.)

app = FastAPI(
    title="EPL Injury Risk Predictor API",
    description="ML-powered injury risk predictions for Premier League players",
    version="1.0.0",
)

# CORS for React frontend (local dev + Render deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://epl-injury-frontend.onrender.com",
        "https://injurywatch.onrender.com",
        "https://yara-sports-frontend.onrender.com",
        "https://yaraspeaks.com",
        "https://www.yaraspeaks.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FPL team names -> display names mapping
FPL_TEAM_DISPLAY_NAMES = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Wolves": "Wolverhampton",
    "Brighton": "Brighton Hove",
    "Nott'm Forest": "Nottingham",
    "Leeds": "Leeds United",
}


def _fpl_team_to_df_team(fpl_team: str) -> str:
    """Convert FPL team name to our display name."""
    return FPL_TEAM_DISPLAY_NAMES.get(fpl_team, fpl_team)


# Load models at startup
artifacts = None
inference_df = None
injury_detail_df = None  # Per-injury records from player_injuries_detail.pkl
fpl_stats_cache = {}  # FPL stats indexed by name
fpl_team_ids = {}  # Team name -> FPL team ID (for badges)
fpl_team_name_to_id = {}  # Team name -> FPL numeric team ID
fpl_team_names_by_id = {}  # FPL numeric team ID -> team name
fpl_team_meta = {}  # Team name -> full FPL team metadata
fpl_team_recent_defense = {}  # Team ID -> recent defensive snapshot
fpl_player_codes = {}  # Player name -> FPL code (for photos)
fpl_player_team = {}  # Player name -> team name (for photo disambiguation)
fpl_players_by_team = {}  # Team name -> set of FPL player names (for filtering)
fpl_element_lookup: Dict[int, Dict] = {}  # FPL element ID -> player stats dict
_tm_photo_map: Dict[str, str] = {}  # Normalised player name -> Transfermarkt photo URL
_tm_photo_bytes_cache: Dict[str, bytes] = {}  # photo URL -> raw image bytes (in-memory, avoids repeat fetches)
shirt_numbers_by_team: Dict[str, Dict[str, int]] = {}  # Normalized team -> normalized player name -> shirt number
shirt_number_lookup_attempted: Set[str] = set()  # Teams we already tried loading for shirt numbers
_startup_complete: bool = False  # Set True after load_models finishes; suppresses API calls during startup
historical_matches_df = None  # Local historical PL fixtures from csv
fixture_history_cache = {}  # (team, opponent, years) -> summary
odds_client = None
refresh_state = {
    "running": False,
    "last_status": "idle",
    "last_started_at": None,
    "last_finished_at": None,
    "last_mode": None,
    "last_error": None,
    "last_exit_code": None,
    "last_log_tail": "",
}
refresh_state_lock = threading.Lock()

# TTL caches for expensive per-request external API calls
_pl_standings_cache: Dict[str, Any] = {"data": None, "expires": None}  # football-data.org PL standings
_fpl_insights_cache: Dict[str, Any] = {"data": None, "expires": None}  # FPL insights (standings + fixtures + gw)
_player_risk_cache: Dict[str, Dict[str, Any]] = {}  # player name (lower) -> {"data": ..., "expires": ...}
_la_liga_standings_cache: Dict[str, Any] = {"data": None, "expires": None}
_la_liga_fixture_dataset_cache: Dict[str, Any] = {"season": None, "data": None, "expires": None}
_la_liga_team_fixtures_cache: Dict[str, Dict[str, Any]] = {}
_la_liga_moneyline_cache: Dict[str, Dict[str, Any]] = {}
_la_liga_team_context_cache: Dict[str, Dict[str, Any]] = {}
_tm_player_profile_cache: Dict[str, Dict[str, Any]] = {}
_tm_scraper_instance = None
_PL_STANDINGS_TTL = timedelta(hours=6)   # team list barely changes mid-season
_FPL_INSIGHTS_TTL = timedelta(minutes=15)  # GW fixture data refreshes each round
_PLAYER_RISK_TTL = timedelta(minutes=10)  # odds + FPL data; stale-ish is fine for repeated views
_LA_LIGA_STANDINGS_TTL = timedelta(minutes=12)
_LA_LIGA_FIXTURE_DATASET_TTL = timedelta(minutes=12)
_LA_LIGA_TEAM_FIXTURES_TTL = timedelta(minutes=10)
_LA_LIGA_MONEYLINE_TTL = timedelta(minutes=7)
_LA_LIGA_TEAM_CONTEXT_TTL = timedelta(minutes=5)
_TM_PLAYER_PROFILE_TTL = timedelta(hours=24)


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _cache_entry_is_fresh(entry: Optional[Dict[str, Any]]) -> bool:
    if not entry:
        return False
    expires = entry.get("expires")
    return bool(expires and expires > datetime.utcnow())


def _normalize_cache_key(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _require_refresh_token(x_refresh_token: Optional[str]) -> None:
    expected = (os.environ.get("REFRESH_CRON_TOKEN") or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Refresh endpoint is not configured. Set REFRESH_CRON_TOKEN.",
        )
    provided = (x_refresh_token or "").strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid refresh token")


def _run_refresh_job(mode: str = "api") -> None:
    """Run refresh_predictions and hot-reload artifacts into API memory."""
    global artifacts, inference_df

    cmd = [sys.executable, "scripts/refresh_predictions.py", "--mode", mode]
    output = ""
    exit_code = 1
    status = "failed"
    error = None

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        exit_code = int(proc.returncode)
        output = "\n".join(
            [part for part in [(proc.stdout or "").strip(), (proc.stderr or "").strip()] if part]
        )
        if exit_code != 0:
            error = f"refresh_predictions exited with code {exit_code}"
        else:
            # Reload all runtime artifacts/caches the same way startup does.
            asyncio.run(load_models())
            status = "success"
    except Exception as exc:
        error = str(exc)

    log_tail = "\n".join((output or "").splitlines()[-40:])
    with refresh_state_lock:
        refresh_state["running"] = False
        refresh_state["last_status"] = status
        refresh_state["last_finished_at"] = _utc_now_iso()
        refresh_state["last_exit_code"] = exit_code
        refresh_state["last_error"] = error
        refresh_state["last_log_tail"] = log_tail


@app.on_event("startup")
async def load_models():
    """Startup event — offloads blocking work to a thread so the event loop stays
    responsive to health checks while models are loading."""
    await asyncio.to_thread(_load_models_blocking)


def _load_models_blocking():
    global artifacts, inference_df, fpl_stats_cache, fpl_team_ids, fpl_team_name_to_id
    global fpl_team_names_by_id, fpl_team_meta, fpl_team_recent_defense
    global fpl_player_codes, fpl_players_by_team, fpl_element_lookup, historical_matches_df, fixture_history_cache, odds_client
    global shirt_numbers_by_team, shirt_number_lookup_attempted, _startup_complete, _player_risk_cache
    global _la_liga_standings_cache, _la_liga_fixture_dataset_cache, _la_liga_team_fixtures_cache
    global _la_liga_moneyline_cache, _la_liga_team_context_cache, _tm_player_profile_cache
    _startup_complete = False
    # Reset caches so manual/cron refreshes don't accumulate stale keys.
    _player_risk_cache = {}
    _la_liga_standings_cache = {"data": None, "expires": None}
    _la_liga_fixture_dataset_cache = {"season": None, "data": None, "expires": None}
    _la_liga_team_fixtures_cache = {}
    _la_liga_moneyline_cache = {}
    _la_liga_team_context_cache = {}
    _tm_player_profile_cache = {}
    fpl_stats_cache = {}
    fpl_team_ids = {}
    fpl_team_name_to_id = {}
    fpl_team_names_by_id = {}
    fpl_team_meta = {}
    fpl_team_recent_defense = {}
    fpl_player_codes = {}
    fpl_players_by_team = {}
    fpl_element_lookup = {}
    shirt_numbers_by_team = {}
    shirt_number_lookup_attempted = set()
    fixture_history_cache = {}

    artifacts = load_artifacts()
    if artifacts and "inference_df" in artifacts:
        inference_df = artifacts["inference_df"]
        print(f"Loaded {len(inference_df)} player predictions")
    else:
        print("WARNING: No trained models found. API will return errors.")

    # Load FPL stats
    try:
        client = FPLClient()
        all_stats = client.get_all_player_stats()
        # Index by multiple keys for matching
        for p in all_stats:
            name_lower = p["name"].lower()
            full_lower = p["full_name"].lower()
            fpl_stats_cache[name_lower] = p
            fpl_stats_cache[full_lower] = p
            # Also index accent-stripped versions
            for k in [name_lower, full_lower]:
                ascii_k = _strip_accents(k)
                if ascii_k != k:
                    fpl_stats_cache[ascii_k] = p
            # Index by last name
            last_name = p["full_name"].split()[-1].lower() if p["full_name"] else ""
            if last_name and len(last_name) >= 4:
                fpl_stats_cache[last_name] = p
                ascii_ln = _strip_accents(last_name)
                if ascii_ln != last_name:
                    fpl_stats_cache[ascii_ln] = p
            # Index by first + last (skipping middle names)
            parts = p["full_name"].split()
            if len(parts) >= 2:
                first_last = f"{parts[0]} {parts[-1]}".lower()
                fpl_stats_cache[first_last] = p
                ascii_fl = _strip_accents(first_last)
                if ascii_fl != first_last:
                    fpl_stats_cache[ascii_fl] = p
            # Store photo codes with team for disambiguation
            if p.get("photo_code"):
                code = p["photo_code"]
                for k in [name_lower, full_lower]:
                    fpl_player_codes[k] = code
                    ascii_k = _strip_accents(k)
                    if ascii_k != k:
                        fpl_player_codes[ascii_k] = code
                if last_name and len(last_name) >= 4:
                    fpl_player_codes[last_name] = code
                    ascii_ln = _strip_accents(last_name)
                    if ascii_ln != last_name:
                        fpl_player_codes[ascii_ln] = code
                if len(parts) >= 2:
                    fl = f"{parts[0]} {parts[-1]}".lower()
                    fpl_player_codes[fl] = code
                    ascii_fl = _strip_accents(fl)
                    if ascii_fl != fl:
                        fpl_player_codes[ascii_fl] = code
                # Store team mapping for disambiguation
                player_team = p.get("team", "")
                if player_team:
                    for k in [name_lower, full_lower]:
                        fpl_player_team[k] = player_team
                        ascii_k = _strip_accents(k)
                        if ascii_k != k:
                            fpl_player_team[ascii_k] = player_team
                    if last_name and len(last_name) >= 4:
                        fpl_player_team[last_name] = player_team
            # Build per-team player sets for filtering
            team_name = p.get("team", "")
            if team_name:
                if team_name not in fpl_players_by_team:
                    fpl_players_by_team[team_name] = set()
                fpl_players_by_team[team_name].add(name_lower)
                fpl_players_by_team[team_name].add(full_lower)
                if last_name and len(last_name) >= 4:
                    fpl_players_by_team[team_name].add(last_name)
            # Index by FPL element ID for squad sync
            element_id = p.get("player_id")
            if element_id is not None:
                fpl_element_lookup[element_id] = p

        logger.info(f"FPL element lookup: {len(fpl_element_lookup)} players indexed by ID")

        # Build team ID lookup for badges — use "code" not "id"!
        # FPL "id" is sequential 1-20 (alphabetical, changes each season).
        # FPL "code" is the historical club identifier used by the PL CDN for badges.
        # e.g. Arsenal: id=1 but code=3 → badge URL must be t3@x2.png, not t1.
        teams = client.get_teams()
        for t in teams:
            fpl_team_ids[t["name"].lower()] = t["code"]
            fpl_team_name_to_id[t["name"].lower()] = t["id"]
            fpl_team_names_by_id[t["id"]] = t["name"]
            fpl_team_meta[t["name"].lower()] = t

        # Build recent defensive form cache from completed fixtures
        try:
            all_fixtures = client.get_fixtures()
            conceded_by_team = {}
            for fx in all_fixtures:
                if not (fx.get("finished") or fx.get("finished_provisional")):
                    continue
                home = fx.get("team_h")
                away = fx.get("team_a")
                home_score = fx.get("team_h_score")
                away_score = fx.get("team_a_score")
                kickoff = fx.get("kickoff_time") or ""
                if home is None or away is None or home_score is None or away_score is None:
                    continue
                conceded_by_team.setdefault(home, []).append((kickoff, int(away_score)))
                conceded_by_team.setdefault(away, []).append((kickoff, int(home_score)))

            for team_id, rows in conceded_by_team.items():
                rows = sorted(rows, key=lambda r: r[0])
                last5 = rows[-5:]
                if not last5:
                    continue
                avg_conceded = round(sum(v for _, v in last5) / len(last5), 2)
                clean_sheets = sum(1 for _, v in last5 if v == 0)
                fpl_team_recent_defense[team_id] = {
                    "avg_goals_conceded_last5": avg_conceded,
                    "clean_sheets_last5": clean_sheets,
                    "samples": len(last5),
                }
        except Exception as defense_err:
            logger.warning(f"Failed to build team recent-defense cache: {defense_err}")
        print(f"Loaded FPL stats for {len(all_stats)} players, {len(teams)} teams")
    except Exception as e:
        print(f"WARNING: Failed to load FPL stats: {e}")

    # Initialize odds client
    odds_client = OddsClient()
    matches = odds_client.get_upcoming_matches()
    print(f"Loaded odds for {len(matches)} upcoming matches")

    # Load historical fixture data — live cache first (freshest), then local CSV fallback.
    # Run API refresh/backfill in the background so startup remains fast.
    try:
        historical_matches_df = _load_fixture_history_live_cached_first()
        fixture_history_cache = {}
        if historical_matches_df is not None and not historical_matches_df.empty:
            latest_date = historical_matches_df["Date"].max()
            latest_str = str(latest_date.date()) if hasattr(latest_date, "date") else str(latest_date)[:10]
            print(
                f"Loaded historical fixtures (live-first): {len(historical_matches_df)} rows "
                f"(latest={latest_str})"
            )
        else:
            print("No cached/live or local fixture history — context stays limited until backfill completes")
        # Kick off API refresh/backfill in a background thread (may take 60s+ due to rate limits)
        # Skip entirely if backfill is disabled — avoids 3-retry × 60s startup delay.
        if _env_flag("FIXTURE_HISTORY_ENABLE_API_BACKFILL", default=True):
            backfill_thread = threading.Thread(
                target=_backfill_fixture_history_async,
                daemon=True,
                name="fixture-history-backfill",
            )
            backfill_thread.start()
        else:
            print("Fixture history API backfill disabled (FIXTURE_HISTORY_ENABLE_API_BACKFILL=false)")
    except Exception as e:
        print(f"WARNING: Failed to load historical fixture data: {e}")

    # Filter inference_df: validate EPL players against FPL and correct team assignments.
    # FPL updates weekly (gameweek-level) so it reflects January transfers etc.
    # football-data.org squads can be stale.
    # La Liga players are not in FPL — keep them unconditionally.
    if inference_df is not None and fpl_stats_cache:
        pre_count = len(inference_df)
        valid_rows = []
        for idx, row in inference_df.iterrows():
            row_league = row.get("league", "Premier League")
            if row_league != "Premier League":
                # Non-EPL leagues: no FPL validation, keep as-is
                valid_rows.append(idx)
                continue
            fpl = get_fpl_stats_for_player(row["name"], team_hint=row.get("team", ""))
            if fpl:
                valid_rows.append(idx)
                # Use FPL team assignment — it's updated weekly and reflects transfers
                fpl_team = fpl.get("team", "")
                if fpl_team:
                    inference_df.at[idx, "team"] = fpl_team
                    inference_df.at[idx, "player_team"] = fpl_team
        inference_df = inference_df.loc[valid_rows].copy()
        # Deduplicate: keep highest-risk entry per name+team
        inference_df = inference_df.sort_values("ensemble_prob", ascending=False)
        inference_df = inference_df.drop_duplicates(subset=["name", "team"], keep="first")
        print(f"Filtered players: {pre_count} -> {len(inference_df)} (EPL validated via FPL, La Liga kept)")

    # Enrich inference_df with scraped injury history from player_history.pkl
    # Also merges laliga_player_history.pkl for La Liga players.
    if inference_df is not None:
        try:
            import pandas as pd
            history_path = os.path.join(PROJECT_ROOT, "models", "player_history.pkl")
            laliga_history_path = os.path.join(PROJECT_ROOT, "models", "laliga_player_history.pkl")
            if os.path.exists(history_path):
                ph = pd.read_pickle(history_path)
                # Merge La Liga history (richer — has total_days_lost, days_since_last_injury)
                if os.path.exists(laliga_history_path):
                    try:
                        ll_ph = pd.read_pickle(laliga_history_path)
                        # Apply known name aliases before merging.
                        # Transfermarkt records players by their full legal name;
                        # football-data.org uses the commonly known name.
                        TM_NAME_ALIASES = {
                            "daniel carvajal": "dani carvajal",
                            "jose ignacio fernandez iglesias": "nacho",
                            "jose maria gimenez de vargas": "jose gimenez",
                            "vinicius jose passador de oliveira": "vinicius junior",
                            "vinicius jr": "vinicius junior",
                            "vinicius junior": "vinicius junior",
                            # Gavi: TM uses "Gavi", football-data.org uses "Pablo Gavira"
                            "gavi": "pablo gavira",
                            "pablo martín páez gavira": "pablo gavira",
                            "pablo martin paez gavira": "pablo gavira",
                        }
                        ll_ph["name"] = ll_ph["name"].apply(
                            lambda n: TM_NAME_ALIASES.get(n.lower(), n) if isinstance(n, str) else n
                        )
                        # La Liga history takes precedence for La Liga players; EPL history wins for EPL
                        ph = pd.concat([ph, ll_ph], ignore_index=True)
                        # Keep highest injury count per player (most complete record)
                        ph = ph.sort_values("player_injury_count", ascending=False)
                        ph = ph.drop_duplicates(subset=["name"], keep="first")
                        print(f"Merged La Liga history: {len(ph)} players total")
                    except Exception as ll_e:
                        logger.warning(f"Failed to load La Liga player history: {ll_e}")
                # Merge enriched injury columns into inference_df.
                # Transfermarkt capitalises all name words ("Frenkie De Jong") while
                # football-data.org uses natural case ("Frenkie de Jong").
                # Build a lowercase lookup so we match case-insensitively.
                enrich_cols = ["total_days_lost", "days_since_last_injury", "last_injury_date"]
                available = [c for c in enrich_cols if c in ph.columns]
                for col in ["player_injury_count", "player_avg_severity", "player_worst_injury", "is_injury_prone"]:
                    if col in ph.columns:
                        available.append(col)

                if available:
                    ph_subset = ph[["name"] + available].copy()
                    ph_subset = ph_subset.sort_values("player_injury_count", ascending=False)
                    ph_subset = ph_subset.drop_duplicates(subset=["name"], keep="first")
                    # Build normalised key: lowercase + accent-stripped for case- and accent-insensitive join
                    # Transfermarkt: "Fran Garcia", football-data.org: "Fran García" — need to match these
                    ph_subset["_name_lower"] = ph_subset["name"].apply(lambda n: _strip_accents(n.lower()) if isinstance(n, str) else "")
                    ph_subset = ph_subset.drop_duplicates(subset=["_name_lower"], keep="first")

                    inference_df["_name_lower"] = inference_df["name"].apply(lambda n: _strip_accents(n.lower()) if isinstance(n, str) else "")

                    # Stash original values before dropping — used as fallback for players
                    # not covered by player_history.pkl (e.g. EPL players not in Transfermarkt pkl)
                    fallback_cols = ["total_days_lost", "days_since_last_injury"]
                    _fallback = {}
                    for col in fallback_cols:
                        if col in inference_df.columns:
                            _fallback[col] = inference_df.set_index("_name_lower")[col].to_dict()

                    # Drop old columns before merge
                    for col in available:
                        if col in inference_df.columns:
                            inference_df = inference_df.drop(columns=[col])

                    ph_join = ph_subset.drop(columns=["name"]).rename(columns={"_name_lower": "_name_lower"})
                    inference_df = inference_df.merge(ph_join, on="_name_lower", how="left")
                    inference_df = inference_df.drop(columns=["_name_lower"])

                    # Fill NaN for players not in history
                    inference_df["player_injury_count"] = inference_df["player_injury_count"].fillna(0)
                    inference_df["player_avg_severity"] = inference_df["player_avg_severity"].fillna(0)
                    inference_df["player_worst_injury"] = inference_df["player_worst_injury"].fillna(0)
                    inference_df["is_injury_prone"] = inference_df["is_injury_prone"].fillna(0)
                    # Restore original values for players not found in player_history.pkl
                    inference_df["_name_lower_tmp"] = inference_df["name"].apply(lambda n: _strip_accents(n.lower()) if isinstance(n, str) else "")
                    if "total_days_lost" in inference_df.columns:
                        if "total_days_lost" in _fallback:
                            mask = inference_df["total_days_lost"].isna()
                            inference_df.loc[mask, "total_days_lost"] = inference_df.loc[mask, "_name_lower_tmp"].map(_fallback["total_days_lost"])
                        inference_df["total_days_lost"] = inference_df["total_days_lost"].fillna(0)
                    else:
                        inference_df["total_days_lost"] = inference_df["_name_lower_tmp"].map(_fallback.get("total_days_lost", {})).fillna(0)
                    if "days_since_last_injury" in inference_df.columns:
                        if "days_since_last_injury" in _fallback:
                            mask = inference_df["days_since_last_injury"].isna()
                            inference_df.loc[mask, "days_since_last_injury"] = inference_df.loc[mask, "_name_lower_tmp"].map(_fallback["days_since_last_injury"])
                        inference_df["days_since_last_injury"] = inference_df["days_since_last_injury"].fillna(365)
                    else:
                        inference_df["days_since_last_injury"] = inference_df["_name_lower_tmp"].map(_fallback.get("days_since_last_injury", {})).fillna(365)
                    inference_df = inference_df.drop(columns=["_name_lower_tmp"])
                    has_data = (inference_df["player_injury_count"] > 0).sum()
                    print(f"Enriched injury history: {has_data}/{len(inference_df)} players have injury records")
        except Exception as e:
            print(f"WARNING: Failed to enrich injury history: {e}")

    # Load per-injury detail records for narrative enrichment
    global injury_detail_df
    try:
        import pandas as pd
        detail_path = os.path.join(PROJECT_ROOT, "models", "player_injuries_detail.pkl")
        if os.path.exists(detail_path):
            injury_detail_df = pd.read_pickle(detail_path)
            print(f"Loaded {len(injury_detail_df)} injury detail records for {injury_detail_df['name'].nunique()} players")
    except Exception as e:
        print(f"WARNING: Failed to load injury detail: {e}")

    # Load Transfermarkt player photo map (covers La Liga + recently-transferred EPL players)
    global _tm_photo_map
    try:
        photo_map_path = os.path.join(PROJECT_ROOT, "models", "player_photo_map.json")
        if os.path.exists(photo_map_path):
            with open(photo_map_path, "r", encoding="utf-8") as _f:
                _tm_photo_map = json.load(_f)
            print(f"Loaded TM photo map: {len(_tm_photo_map)} entries")
        else:
            print("INFO: player_photo_map.json not found — run scripts/build_player_photo_map.py to generate")
    except Exception as e:
        print(f"WARNING: Failed to load TM photo map: {e}")

    # Re-assign archetypes: KMeans for players with per-injury detail, rule-based fallback
    if inference_df is not None:
        inference_df = assign_hybrid_archetypes(inference_df)
        archetype_counts = inference_df["archetype"].value_counts().to_dict()
        print(f"Hybrid archetypes: {archetype_counts}")

    _startup_complete = True


def assign_rule_based_archetypes(df):
    """Assign archetypes using rule-based logic from scraped injury summary stats.

    Order matters — first match wins. Designed to address:
    - Players with injuries incorrectly showing as "Clean Record"
    - Players with 2 injuries + high severity but >1yr gap misclassified as "Injury Prone"
    """
    import math

    def _safe_int(val, default=0):
        try:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return default
            return int(val)
        except (ValueError, TypeError):
            return default

    def _safe_float(val, default=0.0):
        try:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return default
            return float(val)
        except (ValueError, TypeError):
            return default

    def _classify(row):
        count = _safe_int(row.get("player_injury_count", 0))
        avg_sev = _safe_float(row.get("player_avg_severity", 0))
        worst = _safe_float(row.get("player_worst_injury", 0))
        days_since = _safe_float(row.get("days_since_last_injury", 9999), 9999)

        if count == 0:
            return "Clean Record"
        # Fragile: repeated severe spells, not just one outlier event.
        if ((count >= 4 and avg_sev >= 45 and worst >= 75) or
                (count >= 3 and avg_sev >= 60 and worst >= 75)):
            return "Fragile"
        # Recurring Issues: frequent, recent, AND not just minor knocks (check before Injury Prone)
        if count >= 5 and days_since < 90 and avg_sev >= 20:
            return "Recurring Issues"
        # Injury Prone: many injuries AND they're not just minor knocks
        if count >= 5 and avg_sev >= 20:
            return "Injury Prone"
        # Durable: few injuries, long time since last
        if count <= 2 and days_since > 365:
            return "Durable"
        # Durable: many minor knocks (avg < 20 days) with no recent issues
        if avg_sev < 20 and days_since > 180:
            return "Durable"
        return "Moderate Risk"

    df["archetype"] = df.apply(_classify, axis=1)
    return df


def _apply_archetype_overrides(df):
    """Recency-based post-processing applied after both cluster and cache paths."""
    import math
    overrides = 0
    for idx, row in df.iterrows():
        try:
            days_since = float(row.get("days_since_last_injury", 9999))
        except (TypeError, ValueError):
            days_since = 9999.0
        try:
            prev_inj = int(float(row.get("previous_injuries", row.get("player_injury_count", 0)) or 0))
        except (TypeError, ValueError):
            prev_inj = 0
        try:
            total_days = float(row.get("total_days_lost", 0) or 0)
        except (TypeError, ValueError):
            total_days = 0.0
        avg_sev = total_days / prev_inj if prev_inj > 0 else 0
        current = df.at[idx, "archetype"]

        if days_since < 60 and current not in ("Fragile",):
            df.at[idx, "archetype"] = "Currently Vulnerable"
            overrides += 1
        elif days_since > 365 and avg_sev < 25 and current in ("Injury Prone", "Recurring Issues", "Fragile"):
            df.at[idx, "archetype"] = "Durable"
            overrides += 1
        elif prev_inj <= 2 and days_since > 365 and current in ("Fragile", "Injury Prone"):
            df.at[idx, "archetype"] = "Moderate Risk" if avg_sev >= 45 else "Durable"
            overrides += 1
    if overrides:
        print(f"Archetype recency overrides applied: {overrides}")
    return df


def assign_hybrid_archetypes(df):
    """Hybrid archetype assignment: HDBSCAN primary, KMeans fallback, rule-based last resort.

    1. Build archetype features from player_injuries_detail.pkl (per-injury records)
    2. HDBSCAN to find dense clusters — labels noise as -1
    3. KMeans on noise points to assign them to nearest cluster
    4. Label clusters by inspecting centroids (avg_severity, total_injuries, etc.)
    5. Rule-based fallback for players not in detail pkl (no per-injury data)

    Results are cached to models/archetype_cache.pkl keyed on the mtime of
    player_injuries_detail.pkl. On subsequent startups the cache is loaded
    directly, skipping HDBSCAN entirely (~20-30s saved on cold start).
    """
    import os
    import pickle
    import pandas as pd
    import hdbscan
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from src.feature_engineering.archetype import build_player_archetype_features

    detail_path = os.path.join(os.path.dirname(__file__), "..", "models", "player_injuries_detail.pkl")
    detail_path = os.path.abspath(detail_path)
    cache_path = os.path.join(os.path.dirname(detail_path), "archetype_cache.pkl")

    if not os.path.exists(detail_path):
        print("No player_injuries_detail.pkl found — falling back to rule-based archetypes")
        return assign_rule_based_archetypes(df)

    # --- Cache check ---
    # Skip HDBSCAN if we have a fresh cache keyed on the detail pkl's mtime.
    try:
        detail_mtime = os.path.getmtime(detail_path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as _cf:
                _cached = pickle.load(_cf)
            if _cached.get("detail_mtime") == detail_mtime:
                archetype_lookup = _cached["archetype_lookup"]
                sparse_names = _cached.get("sparse_names", set())
                matched = 0
                for idx, row in df.iterrows():
                    name = row.get("name", "")
                    if name in archetype_lookup:
                        df.at[idx, "archetype"] = archetype_lookup[name]
                        matched += 1
                unmatched_mask = ~df["name"].isin(archetype_lookup) | df["name"].isin(sparse_names)
                if unmatched_mask.sum() > 0:
                    unmatched_df = assign_rule_based_archetypes(df.loc[unmatched_mask].copy())
                    df.loc[unmatched_mask, "archetype"] = unmatched_df["archetype"].values
                print(f"Archetypes loaded from cache ({matched} matched, {unmatched_mask.sum()} rule-based)")
                # Still apply recency overrides
                return _apply_archetype_overrides(df)
    except Exception as _ce:
        print(f"Archetype cache miss or corrupt — recomputing: {_ce}")

    try:
        detail_df = pd.read_pickle(detail_path)
        # Fix StringDtype columns
        for col in detail_df.columns:
            if "String" in str(detail_df[col].dtype) or str(detail_df[col].dtype) in ("string", "string[python]"):
                detail_df[col] = detail_df[col].astype(object)

        # Build per-player feature matrix from per-injury records
        feat_df = build_player_archetype_features(detail_df)
        print(f"Built archetype features for {len(feat_df)} players")

        # Avoid over-clustering tiny samples: players with sparse injury history
        # are better handled by rule-based logic.
        min_injuries_for_clustering = max(2, int(os.getenv("ARCHETYPE_CLUSTER_MIN_INJURIES", "3")))
        feat_df["total_injuries"] = feat_df["total_injuries"].fillna(0)
        clustered_feat_df = feat_df[feat_df["total_injuries"] >= min_injuries_for_clustering].copy()
        sparse_feat_df = feat_df[feat_df["total_injuries"] < min_injuries_for_clustering].copy()
        print(
            f"Archetype clustering scope: {len(clustered_feat_df)} clustered, "
            f"{len(sparse_feat_df)} sparse-history fallback"
        )

        if len(clustered_feat_df) < 25:
            print("Insufficient clustered sample size — falling back to rule-based archetypes")
            return assign_rule_based_archetypes(df)

        # Use core features only (avoids curse of dimensionality with sparse one-hot columns)
        core_cols = [
            "total_injuries", "avg_severity", "median_severity", "max_severity",
            "high_severity_rate", "avg_days_between_injuries", "reinjury_rate",
            "body_area_entropy", "severity_cv", "severity_trend",
        ]
        core_cols = [c for c in core_cols if c in clustered_feat_df.columns]
        X = clustered_feat_df[core_cols].fillna(0).values

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- HDBSCAN ---
        min_cluster_size = max(8, int(os.getenv("ARCHETYPE_HDBSCAN_MIN_CLUSTER_SIZE", "10")))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=3, cluster_selection_method="eom")
        labels = clusterer.fit_predict(X_scaled)
        n_clusters = len(set(labels) - {-1})
        noise_count = (labels == -1).sum()
        print(f"HDBSCAN: {n_clusters} clusters, {noise_count} noise points out of {len(labels)}")

        # --- KMeans fallback for noise points ---
        if n_clusters < 2:
            # HDBSCAN couldn't find structure — use KMeans entirely
            fallback_clusters = max(2, min(5, len(clustered_feat_df) // 12))
            print(f"HDBSCAN found <2 clusters — using KMeans with {fallback_clusters} clusters")
            km = KMeans(n_clusters=fallback_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            n_clusters = fallback_clusters
        elif noise_count > 0:
            # Assign noise points to nearest HDBSCAN cluster via KMeans
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            km_labels = km.fit_predict(X_scaled)
            for i in range(len(labels)):
                if labels[i] == -1:
                    labels[i] = km_labels[i]
            print(f"KMeans assigned {noise_count} noise points to clusters")

        clustered_feat_df["cluster"] = labels

        # --- Label clusters by ranking on key characteristics ---
        profiles = clustered_feat_df.groupby("cluster").agg({
            "total_injuries": "mean",
            "avg_severity": "mean",
            "max_severity": "mean",
            "high_severity_rate": "mean",
            "reinjury_rate": "mean",
            "severity_cv": "mean",
            "avg_days_between_injuries": "mean",
            "body_area_entropy": "mean",
            "severity_trend": "mean",
        })

        # Independent per-cluster scoring (do not force one-to-one archetype labels).
        # Forcing unique labels across clusters was causing obvious misclassifications.
        def _score_cluster(row):
            injuries = float(row.get("total_injuries", 0) or 0)
            avg_sev = float(row.get("avg_severity", 0) or 0)
            max_sev = float(row.get("max_severity", 0) or 0)
            high_sev = float(row.get("high_severity_rate", 0) or 0)
            reinjury = float(row.get("reinjury_rate", 0) or 0)
            variability = float(row.get("severity_cv", 0) or 0)
            avg_gap = float(row.get("avg_days_between_injuries", 0) or 0)
            entropy = float(row.get("body_area_entropy", 0) or 0)
            trend = float(row.get("severity_trend", 0) or 0)

            scores = {
                "Fragile": (avg_sev * 1.5) + (max_sev * 0.8) + (high_sev * 35) - (injuries * 1.4),
                "Injury Prone": (injuries * 7.0) + (high_sev * 12) + (max(0, 140 - avg_gap) * 0.06),
                "Recurring Issues": (reinjury * 110) + (injuries * 3.5) + (max(0, 120 - avg_gap) * 0.05) + (max(0, trend) * 8),
                "Unpredictable": (variability * 32) + (entropy * 8) + (abs(trend) * 4) + (injuries * 2),
                "Durable": (max(0, 28 - avg_sev) * 1.2) + (max(0, 4 - injuries) * 8) + (avg_gap * 0.08) + (max(0, 0.35 - reinjury) * 40),
            }
            return max(scores, key=scores.get)

        cluster_to_archetype = {cid: _score_cluster(row) for cid, row in profiles.iterrows()}
        clustered_feat_df["archetype"] = clustered_feat_df["cluster"].map(cluster_to_archetype)

        # Log cluster assignments
        for cid in sorted(cluster_to_archetype.keys()):
            row = profiles.loc[cid]
            count = (clustered_feat_df["cluster"] == cid).sum()
            print(f"  Cluster {cid} → {cluster_to_archetype[cid]} ({count} players, "
                  f"avg_inj={row['total_injuries']:.1f}, avg_sev={row['avg_severity']:.1f})")

        # Map back to inference_df by player name
        archetype_lookup = dict(zip(clustered_feat_df["name"], clustered_feat_df["archetype"]))

        # Persist cache so next startup skips HDBSCAN
        try:
            with open(cache_path, "wb") as _cf:
                pickle.dump({
                    "detail_mtime": detail_mtime,
                    "archetype_lookup": archetype_lookup,
                    "sparse_names": set(sparse_feat_df["name"].tolist()),
                }, _cf)
            print(f"Archetype cache saved ({len(archetype_lookup)} players)")
        except Exception as _ce:
            print(f"Could not save archetype cache: {_ce}")

        matched = 0
        for idx, row in df.iterrows():
            name = row.get("name", "")
            if name in archetype_lookup:
                df.at[idx, "archetype"] = archetype_lookup[name]
                matched += 1

        sparse_names = set(sparse_feat_df["name"].tolist())

        # Rule-based fallback for unmatched players (not in detail pkl or sparse history)
        unmatched_mask = ~df["name"].isin(archetype_lookup) | df["name"].isin(sparse_names)
        unmatched_count = unmatched_mask.sum()
        if unmatched_count > 0:
            unmatched_df = assign_rule_based_archetypes(df.loc[unmatched_mask].copy())
            df.loc[unmatched_mask, "archetype"] = unmatched_df["archetype"].values

        df = _apply_archetype_overrides(df)

        # Collapse guard: if one archetype dominates unnaturally, fallback to
        # rule-based assignment to preserve interpretability.
        counts = df["archetype"].value_counts().to_dict()
        dominant_ratio = (max(counts.values()) / len(df)) if counts else 0.0
        if dominant_ratio >= float(os.getenv("ARCHETYPE_DOMINANCE_MAX_RATIO", "0.55")):
            print(
                f"Archetype collapse detected ({dominant_ratio:.1%} dominant) — "
                "falling back to rule-based archetypes"
            )
            df = assign_rule_based_archetypes(df)
            counts = df["archetype"].value_counts().to_dict()
            dominant_ratio = (max(counts.values()) / len(df)) if counts else 0.0

        # Final guardrail: if data quality causes a second collapse, enforce a
        # rank-based spread so the UI doesn't show one archetype for everyone.
        if dominant_ratio >= 0.70:
            print(
                f"Archetype collapse persisted ({dominant_ratio:.1%}) — "
                "applying quantile-based fallback archetypes"
            )
            tmp = df.copy()
            inj = pd.to_numeric(
                tmp.get("previous_injuries", tmp.get("player_injury_count", 0)),
                errors="coerce",
            ).fillna(0)
            days_lost = pd.to_numeric(tmp.get("total_days_lost", 0), errors="coerce").fillna(0)
            days_since = pd.to_numeric(tmp.get("days_since_last_injury", 365), errors="coerce").fillna(365)
            avg_sev = days_lost.divide(inj.where(inj > 0, 1))

            composite = (
                inj * 0.65
                + avg_sev * 0.08
                + (days_since < 60).astype(float) * 6.0
                + (days_since < 180).astype(float) * 2.0
                - (days_since > 365).astype(float) * 2.5
            )
            rank = composite.rank(pct=True)

            def _assign(i, sev, since, pct):
                if i <= 0:
                    return "Clean Record"
                if since < 60:
                    return "Currently Vulnerable"
                if pct >= 0.88:
                    return "Fragile" if sev >= 45 else "Injury Prone"
                if pct >= 0.70:
                    return "Injury Prone"
                if pct >= 0.52:
                    return "Recurring Issues"
                if pct >= 0.32:
                    return "Moderate Risk"
                return "Durable"

            tmp["archetype"] = [
                _assign(i, sev, since, pct)
                for i, sev, since, pct in zip(inj.tolist(), avg_sev.tolist(), days_since.tolist(), rank.tolist())
            ]
            df["archetype"] = tmp["archetype"].values
            counts = df["archetype"].value_counts().to_dict()

        counts = df["archetype"].value_counts().to_dict()
        print(f"Hybrid archetypes: {matched} clustered, {unmatched_count} rule-based")
        print(f"  Distribution: {counts}")
        return df

    except Exception as e:
        print(f"WARNING: Hybrid clustering failed ({e}) — falling back to rule-based")
        import traceback
        traceback.print_exc()
        return assign_rule_based_archetypes(df)


# ============================================================
# Response Models
# ============================================================

class PlayerSummary(BaseModel):
    name: str
    team: str
    position: str
    shirt_number: Optional[int] = None
    risk_level: str
    risk_probability: float
    archetype: str
    minutes_played: int = 0
    is_starter: bool = False
    player_image_url: Optional[str] = None
    days_since_last_injury: int = 365
    is_currently_injured: bool = False
    injury_news: Optional[str] = None
    chance_of_playing: Optional[int] = None


class RiskFactors(BaseModel):
    previous_injuries: int
    total_days_lost: int
    days_since_last_injury: int
    avg_days_per_injury: float


class ModelPredictions(BaseModel):
    ensemble: float
    lgb: float
    xgb: float
    catboost: float


class ImpliedOdds(BaseModel):
    """Betting odds derived from injury probability."""
    american: str
    decimal: float
    fractional: str
    implied_prob: float


class ScoringOdds(BaseModel):
    """Odds for a player to score in next match."""
    score_probability: float
    involvement_probability: float
    goals_per_90: float
    assists_per_90: float
    american: str
    decimal: float
    fractional: str
    availability_factor: float
    analysis: Optional[str] = None


class FPLValue(BaseModel):
    """FPL value assessment for a player."""
    tier: str
    tier_emoji: str
    verdict: str
    position_insight: Optional[str] = None
    adjusted_value: float
    goals_per_90: float
    assists_per_90: float
    price: float
    risk_factor: float


class FPLPointsProjection(BaseModel):
    """Expected FPL points for next gameweek, discounted by injury risk."""
    expected_points: float
    base_points: float
    injury_discount_pct: float
    fixture_multiplier: float
    confidence: str
    breakdown: str


class RiskComparison(BaseModel):
    """Player risk compared to squad and position averages."""
    squad_avg_risk: float
    position_avg_risk: float
    squad_rank: int
    squad_total: int
    position_group: str
    position_rank: int
    position_total: int


class PlayerImportance(BaseModel):
    """Fan-facing importance context for narrative explainability."""
    score: float
    tier: str
    ownership_pct: Optional[float] = None
    price: Optional[float] = None
    price_tier: Optional[str] = None
    captaincy_proxy_pct: Optional[float] = None
    role_importance: Optional[str] = None
    form_signal: Optional[str] = None
    h2h_signal: Optional[str] = None
    summary: str


class CleanSheetOdds(BaseModel):
    """Clean sheet odds for defenders/goalkeepers."""
    clean_sheet_probability: float
    goals_conceded_per_game: float
    american: str
    decimal: float
    availability_factor: float


class NextFixture(BaseModel):
    """Player's next match info with betting context."""
    opponent: str
    is_home: bool
    match_time: Optional[str] = None
    clean_sheet_odds: Optional[str] = None
    win_probability: Optional[float] = None
    fixture_insight: Optional[str] = None


class YaraResponse(BaseModel):
    """Yara's opinionated analysis comparing model vs market odds."""
    response_text: str
    fpl_tip: str
    market_probability: Optional[float] = None
    yara_probability: float
    market_odds_decimal: Optional[float] = None
    bookmaker: Optional[str] = None


class BookmakerOddsLine(BaseModel):
    """Single bookmaker line for scoring/clean-sheet market."""
    bookmaker: str
    decimal_odds: float
    implied_probability: float
    source: Optional[str] = None


class BookmakerConsensus(BaseModel):
    """Normalized three-bookie market summary for narrative explainability."""
    market_type: str  # "score" | "clean_sheet"
    market_label: str
    average_decimal: float
    average_probability: float
    summary_text: str
    market_line: str
    lines: List[BookmakerOddsLine]


class LabDriver(BaseModel):
    """A single key driver in Yara's Lab Notes."""
    name: str
    value: Any
    impact: str  # "risk_increasing", "protective", "neutral"
    explanation: str


class TechnicalDetails(BaseModel):
    """Technical details for the 'for builders' section."""
    model_agreement: float
    methodology: str
    feature_highlights: List[Dict[str, Any]]


class LabNotes(BaseModel):
    """Yara's Lab Notes — explainability for the risk score."""
    summary: str
    key_drivers: List[LabDriver]
    technical: TechnicalDetails


class UpcomingFixture(BaseModel):
    """A single upcoming fixture with FDR difficulty."""
    opponent: str
    is_home: bool
    difficulty: int = 3
    match_time: Optional[str] = None


class InjuryRecord(BaseModel):
    date: Optional[str] = None
    body_area: str = "unknown"
    injury_type: str = "unknown"
    injury_raw: str = ""
    severity_days: int = 0
    games_missed: int = 0


class PlayerRisk(BaseModel):
    name: str
    team: str
    position: str
    league: str = "Premier League"
    shirt_number: Optional[int] = None
    age: int
    risk_level: str
    risk_probability: float
    archetype: str
    archetype_description: str
    factors: RiskFactors
    model_predictions: ModelPredictions
    recommendations: List[str]
    story: str
    implied_odds: ImpliedOdds
    last_injury_date: Optional[str]
    fpl_insight: Optional[str] = None
    scoring_odds: Optional[ScoringOdds] = None
    fpl_value: Optional[FPLValue] = None
    clean_sheet_odds: Optional[CleanSheetOdds] = None
    next_fixture: Optional[NextFixture] = None
    bookmaker_consensus: Optional[BookmakerConsensus] = None
    yara_response: Optional[YaraResponse] = None
    lab_notes: Optional[LabNotes] = None
    risk_percentile: Optional[float] = None
    player_image_url: Optional[str] = None
    team_badge_url: Optional[str] = None
    is_currently_injured: bool = False
    injury_news: Optional[str] = None
    chance_of_playing: Optional[int] = None
    upcoming_fixtures: Optional[List[UpcomingFixture]] = None
    injury_records: List[InjuryRecord] = []
    acwr: Optional[float] = None
    acute_load: Optional[float] = None
    chronic_load: Optional[float] = None
    spike_flag: Optional[bool] = None
    fpl_points_projection: Optional[FPLPointsProjection] = None
    risk_comparison: Optional[RiskComparison] = None
    player_importance: Optional[PlayerImportance] = None


class WhatIfProjection(BaseModel):
    """Result of a what-if scenario projection."""
    player_name: str
    current_risk: float
    projected_risk: float
    scenario: str
    delta: float
    acwr_current: float
    acwr_projected: float


class TeamOverview(BaseModel):
    team: str
    total_players: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_risk: float
    players: List[PlayerSummary]
    team_badge_url: Optional[str] = None
    next_fixture: Optional[Dict[str, Any]] = None


class FPLSquadPlayer(PlayerSummary):
    """PlayerSummary extended with FPL squad position metadata."""
    is_captain: bool = False
    is_vice_captain: bool = False
    squad_position: int = 0
    multiplier: int = 1


class FPLSquadEntry(BaseModel):
    """FPL manager info."""
    team_name: str
    manager_name: str
    total_points: int
    gameweek: int
    gameweek_points: int


class FPLSquadSync(BaseModel):
    """Full squad sync response."""
    entry: FPLSquadEntry
    players: List[FPLSquadPlayer]
    unmatched: List[str]
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_risk: float
    is_gw_finished: bool = False


class HealthCheck(BaseModel):
    status: str
    models_loaded: bool
    player_count: int


class LeagueStanding(BaseModel):
    id: int
    name: str
    short_name: str
    position: int
    played: int
    wins: int
    draws: int
    losses: int
    points: int
    form: Optional[str]
    strength: int


class GameweekSummary(BaseModel):
    gameweek: int
    name: str
    deadline: Optional[str]
    is_current: bool
    is_next: bool
    fixture_count: int
    double_gameweek_teams: List[str]
    featured_matches: List[str]


class FPLInsights(BaseModel):
    current_gameweek: Optional[int]
    standings: List[LeagueStanding]
    upcoming_gameweeks: List[GameweekSummary]
    has_double_gameweek: bool


class TeamStanding(BaseModel):
    name: str
    short_name: str
    position: Optional[int] = None
    points: int
    played: int
    form: Optional[str] = None
    distance_from_top: Optional[int] = None
    distance_from_safety: Optional[int] = None


class StandingsSummary(BaseModel):
    leader: TeamStanding
    second: TeamStanding
    gap_to_second: int
    safety_points: int = 0
    selected_team: Optional[TeamStanding] = None


# ============================================================
# Helper Functions
# ============================================================

ARCHETYPE_DESCRIPTIONS = {
    "Durable": "Resilient player with limited injury history and good recovery",
    "Fragile": "When injured, tends to be serious. Requires extended recovery periods",
    "Injury Prone": "Frequently picks up injuries, though typically not severe",
    "Recurring Issues": "Recent pattern of repeated injuries. Needs targeted management",
    "Moderate Risk": "Average injury profile with no major red flags",
    "Clean Record": "No significant injury history on record",
    "Currently Vulnerable": "Recently returned from injury, elevated re-injury risk",
    "Recurring": "Regular minor injuries but quick recoveries",
    "Unpredictable": "Varied injury patterns that are difficult to predict",
}

# Team name aliases for FPL badge lookup
# Maps all known team name variants → FPL full name (lowercase)
# FPL names: Arsenal, Aston Villa, Bournemouth, Brentford, Brighton,
#   Chelsea, Crystal Palace, Everton, Fulham, Ipswich, Leicester,
#   Liverpool, Man City, Man Utd, Newcastle, Nott'm Forest,
#   Southampton, Spurs, West Ham, Wolves
TEAM_BADGE_ALIASES = {
    # Arsenal
    "arsenal": "arsenal", "arsenal fc": "arsenal",
    # Aston Villa
    "aston villa": "aston villa", "villa": "aston villa",
    # Bournemouth
    "bournemouth": "bournemouth", "afc bournemouth": "bournemouth",
    # Brentford
    "brentford": "brentford", "brentford fc": "brentford",
    # Brighton
    "brighton": "brighton", "brighton hove": "brighton",
    "brighton & hove albion": "brighton", "brighton and hove albion": "brighton",
    "brighton & hove": "brighton",
    # Chelsea
    "chelsea": "chelsea", "chelsea fc": "chelsea",
    # Crystal Palace
    "crystal palace": "crystal palace", "palace": "crystal palace",
    # Everton
    "everton": "everton", "everton fc": "everton",
    # Fulham
    "fulham": "fulham", "fulham fc": "fulham",
    # Ipswich
    "ipswich": "ipswich", "ipswich town": "ipswich",
    # Leicester
    "leicester": "leicester", "leicester city": "leicester",
    # Liverpool
    "liverpool": "liverpool", "liverpool fc": "liverpool",
    # Man City
    "man city": "man city", "manchester city": "man city",
    "manchester city fc": "man city", "mcfc": "man city",
    # Man Utd
    "man utd": "man utd", "man united": "man utd",
    "manchester united": "man utd", "manchester united fc": "man utd",
    "mufc": "man utd",
    # Newcastle
    "newcastle": "newcastle", "newcastle united": "newcastle",
    "newcastle utd": "newcastle",
    # Nott'm Forest
    "nott'm forest": "nott'm forest", "nottm forest": "nott'm forest",
    "nottingham forest": "nott'm forest", "nottingham": "nott'm forest",
    # Southampton
    "southampton": "southampton", "southampton fc": "southampton",
    # Spurs
    "spurs": "spurs", "tottenham": "spurs",
    "tottenham hotspur": "spurs", "tottenham hotspurs": "spurs",
    # West Ham
    "west ham": "west ham", "west ham united": "west ham",
    # Wolves
    "wolves": "wolves", "wolverhampton": "wolves",
    "wolverhampton wanderers": "wolves",
    # Promoted / other
    "leeds": "leeds", "leeds united": "leeds",
    "sunderland": "sunderland", "sunderland afc": "sunderland",
    "burnley": "burnley", "burnley fc": "burnley",
    "luton": "luton", "luton town": "luton",
    "sheffield united": "sheffield utd", "sheffield utd": "sheffield utd",
}


def _league_prob_series(league: Optional[str] = None):
    """Return the ensemble_prob series for percentile calculation.

    Normalises within the player's own league so that EPL and La Liga
    each have a meaningful high/medium/low distribution rather than
    La Liga players all sitting in the bottom half of a combined ranking.
    Falls back to all players if league is unknown or inference_df has no
    league column.
    """
    if inference_df is None or "ensemble_prob" not in inference_df.columns:
        return None
    if league and "league" in inference_df.columns:
        sub = inference_df[inference_df["league"].str.lower() == league.lower()]
        if len(sub) >= 10:
            return sub["ensemble_prob"]
    return inference_df["ensemble_prob"]


def get_risk_level(prob: float, row=None) -> str:
    """Classify 2-week injury risk using within-league percentile ranking.

    Ranks player relative to others in the same league:
    - High: top 20% (80th percentile and above)
    - Medium: 40th-80th percentile
    - Low: bottom 40%
    """
    league = row.get("league") if isinstance(row, dict) else (getattr(row, "get", lambda k, d=None: d)("league") if row is not None else None)
    series = _league_prob_series(league)
    if series is not None:
        percentile = float((series <= prob).mean())
        if percentile >= 0.80:
            return "High"
        elif percentile >= 0.40:
            return "Medium"
        else:
            return "Low"
    else:
        if prob >= 0.20:
            return "High"
        elif prob >= 0.10:
            return "Medium"
        else:
            return "Low"


def normalize_risk_score(prob: float, league: Optional[str] = None) -> float:
    """Convert raw model probability to a 0-100 score based on within-league percentile rank.

    Normalises within the player's own league so La Liga and EPL each span 0-100.
    50 = average risk for that league, 90 = top 10% in that league.
    """
    series = _league_prob_series(league)
    if series is not None:
        percentile = float((series <= prob).mean())
        return round(min(99.0, percentile * 100), 1)
    return round(max(0, min(99, (prob - 0.04) / (0.92 - 0.04) * 100)), 1)


# La Liga team badge URLs from football-data.org crests (static — IDs are stable)
LA_LIGA_BADGE_MAP: Dict[str, str] = {
    "athletic club": "https://crests.football-data.org/77.png",
    "atletico madrid": "https://crests.football-data.org/78.png",
    "osasuna": "https://crests.football-data.org/79.png",
    "espanyol": "https://crests.football-data.org/80.png",
    "barcelona": "https://crests.football-data.org/81.png",
    "getafe": "https://crests.football-data.org/82.png",
    "real madrid": "https://crests.football-data.org/86.png",
    "rayo vallecano": "https://crests.football-data.org/87.png",
    "mallorca": "https://crests.football-data.org/89.png",
    "real betis": "https://crests.football-data.org/90.png",
    "real sociedad": "https://crests.football-data.org/92.png",
    "villarreal": "https://crests.football-data.org/94.png",
    "valencia": "https://crests.football-data.org/95.png",
    "valladolid": "https://crests.football-data.org/250.png",
    "alaves": "https://crests.football-data.org/263.png",
    "las palmas": "https://crests.football-data.org/275.png",
    "girona": "https://crests.football-data.org/298.png",
    "celta vigo": "https://crests.football-data.org/558.png",
    "sevilla": "https://crests.football-data.org/559.png",
    "leganes": "https://crests.football-data.org/745.png",
}


def get_team_badge_url(team_name: str) -> Optional[str]:
    """Get badge URL for a team (EPL via FPL CDN, La Liga via football-data.org crests)."""
    team_lower = team_name.lower().strip()

    # La Liga — check static map first (no API dependency)
    if team_lower in LA_LIGA_BADGE_MAP:
        return LA_LIGA_BADGE_MAP[team_lower]

    search = TEAM_BADGE_ALIASES.get(team_lower, team_lower)

    # Exact match
    if search in fpl_team_ids:
        return f"https://resources.premierleague.com/premierleague/badges/50/t{fpl_team_ids[search]}@x2.png"

    # Substring match fallback — only against names >= 5 chars to avoid
    # false positives from short codes like "che", "ars", "lei"
    for key, tid in fpl_team_ids.items():
        if len(key) < 5:
            continue
        if search in key or key in search:
            return f"https://resources.premierleague.com/premierleague/badges/50/t{fpl_team_ids[key]}@x2.png"
    return None


def _normalize_team(t: str) -> str:
    """Normalize team name for fuzzy comparison (preserves 'city'/'united' to avoid Man City/Man Utd collision)."""
    return t.lower().replace(" fc", "").replace("afc ", "").strip()


def _strip_accents(s: str) -> str:
    """Remove diacritics: 'Guimarães' → 'Guimaraes', 'Martín' → 'Martin'."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _normalize_player_key(name: str) -> str:
    """Normalize player names for cross-provider matching."""
    base = _strip_accents((name or "").lower())
    base = re.sub(r"[^a-z0-9 ]+", " ", base)
    return re.sub(r"\s+", " ", base).strip()


def _add_shirt_number_entry(mapping: Dict[str, int], player_name: str, shirt_value: Any) -> None:
    """Insert canonical + alias keys for one player's shirt number."""
    shirt = _safe_int(shirt_value, 0)
    if shirt <= 0:
        return

    full_name = _normalize_player_key(str(player_name or ""))
    if not full_name:
        return

    mapping.setdefault(full_name, shirt)
    parts = full_name.split()
    if parts:
        last = parts[-1]
        if len(last) >= 4:
            mapping.setdefault(last, shirt)
        if len(parts) >= 2:
            mapping.setdefault(f"{parts[0]} {parts[-1]}", shirt)


def _build_team_shirt_number_map(squad_df) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if squad_df is None or len(getattr(squad_df, "index", [])) == 0:
        return mapping

    for _, row in squad_df.iterrows():
        _add_shirt_number_entry(mapping, row.get("name", ""), row.get("shirt_number"))
    return mapping


def _api_football_team_search_terms(team_name: Optional[str]) -> List[str]:
    """Generate likely API-Football team search terms from our normalized names."""
    raw = str(team_name or "").strip()
    if not raw:
        return []

    canonical_key = TEAM_BADGE_ALIASES.get(raw.lower(), raw.lower())
    key_to_api_name = {
        "man city": "Manchester City",
        "man utd": "Manchester United",
        "nott'm forest": "Nottingham Forest",
        "spurs": "Tottenham",
        "wolves": "Wolverhampton Wanderers",
        "brighton": "Brighton",
        "west ham": "West Ham",
        "newcastle": "Newcastle",
        "leicester": "Leicester",
        "ipswich": "Ipswich",
    }
    pretty_from_key = key_to_api_name.get(
        canonical_key,
        " ".join(p.capitalize() for p in canonical_key.replace("'", "").split()),
    )

    terms: List[str] = []
    for candidate in [raw, pretty_from_key, canonical_key]:
        candidate = str(candidate or "").strip()
        if candidate and candidate.lower() not in {t.lower() for t in terms}:
            terms.append(candidate)
    return terms


def _pick_api_football_team_id(response_rows: List[Dict], team_name: str) -> Optional[int]:
    """Pick best matching API-Football team ID from /teams response rows."""
    if not response_rows:
        return None

    target = _normalize_player_key(team_name)
    best_id: Optional[int] = None
    best_score = -1

    for row in response_rows:
        team = row.get("team", {})
        team_id = _safe_int(team.get("id"), 0)
        if team_id <= 0:
            continue
        name = _normalize_player_key(str(team.get("name", "")))
        if not name:
            continue

        overlap = len(set(target.split()) & set(name.split()))
        contains_bonus = 3 if (target in name or name in target) else 0
        score = overlap + contains_bonus

        if score > best_score:
            best_score = score
            best_id = team_id

    if best_id:
        return best_id

    first = response_rows[0].get("team", {})
    return _safe_int(first.get("id"), 0) or None


def _fetch_api_football_shirt_numbers(team_name: Optional[str]) -> Dict[str, int]:
    """
    Fetch shirt numbers from API-Football using `players/squads`.
    Requires odds client with API_FOOTBALL_KEY configured.
    """
    if not team_name or odds_client is None:
        return {}
    if not getattr(odds_client, "api_football_key", ""):
        return {}
    if not hasattr(odds_client, "_api_football_request"):
        return {}

    search_terms = _api_football_team_search_terms(team_name)
    if not search_terms:
        return {}

    now_year = datetime.utcnow().year
    team_id: Optional[int] = None
    for season in (now_year, now_year - 1):
        for term in search_terms:
            payload = odds_client._api_football_request(
                "teams",
                {"league": 39, "season": season, "search": term},
            )
            rows = (payload or {}).get("response", [])
            if not rows:
                continue
            team_id = _pick_api_football_team_id(rows, team_name)
            if team_id:
                break
        if team_id:
            break

    if not team_id:
        return {}

    squad_payload = odds_client._api_football_request("players/squads", {"team": team_id})
    squads = (squad_payload or {}).get("response", [])
    if not squads:
        return {}

    mapping: Dict[str, int] = {}
    for squad in squads:
        for player in squad.get("players", []):
            _add_shirt_number_entry(mapping, player.get("name", ""), player.get("number"))
    return mapping


def _ensure_team_shirt_numbers_loaded(team_name: Optional[str]) -> None:
    """Lazy-load squad numbers from football-data.org for one team."""
    if not team_name:
        return
    # Skip API lookups during startup — avoids rate-limit hits from the FPL validation loop.
    if not _startup_complete:
        return

    key = TEAM_BADGE_ALIASES.get(str(team_name).lower().strip(), str(team_name).lower().strip())
    if not key:
        return
    if key in shirt_numbers_by_team or key in shirt_number_lookup_attempted:
        return

    shirt_number_lookup_attempted.add(key)
    api_key = (os.getenv("FOOTBALL_DATA_API_KEY") or "").strip()
    if api_key:
        try:
            client = MatchHistoryApiClient(api_key=api_key)
            squad_df = client.get_team_squad(team_name)
            team_map = _build_team_shirt_number_map(squad_df)
            if team_map:
                shirt_numbers_by_team[key] = team_map
                logger.info(f"Loaded {len(team_map)} shirt numbers for {team_name}")
        except Exception as e:
            logger.debug(f"Shirt-number lookup failed for {team_name}: {e}")

    if key in shirt_numbers_by_team:
        return

    try:
        team_map = _fetch_api_football_shirt_numbers(team_name)
        if team_map:
            shirt_numbers_by_team[key] = team_map
            logger.info(f"Loaded {len(team_map)} shirt numbers for {team_name} via API-Football")
    except Exception as e:
        logger.debug(f"API-Football shirt-number lookup failed for {team_name}: {e}")


def _lookup_shirt_number(player_name: str, team_hint: Optional[str] = None) -> Optional[int]:
    """Find shirt number from cached/lazy-loaded squad data."""
    if not player_name:
        return None

    candidate_keys: List[str] = []
    normalized = _normalize_player_key(player_name)
    if normalized:
        candidate_keys.append(normalized)
        parts = normalized.split()
        if len(parts) >= 2:
            candidate_keys.append(f"{parts[0]} {parts[-1]}")
        if parts:
            candidate_keys.append(parts[-1])

    teams_to_check: List[str] = []
    if team_hint:
        _ensure_team_shirt_numbers_loaded(team_hint)
        norm_team = TEAM_BADGE_ALIASES.get(str(team_hint).lower().strip(), str(team_hint).lower().strip())
        if norm_team:
            teams_to_check.append(norm_team)

    # Fallback: search already-loaded teams
    for team_key in shirt_numbers_by_team.keys():
        if team_key not in teams_to_check:
            teams_to_check.append(team_key)

    for team_key in teams_to_check:
        mapping = shirt_numbers_by_team.get(team_key, {})
        for key in candidate_keys:
            shirt = mapping.get(key)
            if isinstance(shirt, int) and shirt > 0:
                return shirt
    return None


def _attach_shirt_number(stats: Optional[Dict], player_name: str, team_hint: Optional[str] = None) -> Optional[Dict]:
    """Attach shirt number to stats if missing, using fallback squad sources."""
    if not stats:
        return stats
    current = _safe_int(stats.get("shirt_number"), 0)
    if current > 0:
        return stats

    fallback = _lookup_shirt_number(player_name, team_hint=team_hint or stats.get("team"))
    if fallback:
        stats["shirt_number"] = fallback
    return stats


def _resolve_shirt_number(player_name: str, fpl_stats: Optional[Dict], team_hint: Optional[str]) -> Optional[int]:
    """Return shirt number from FPL stats, then fallback squad sources."""
    if fpl_stats:
        from_fpl = _safe_int(fpl_stats.get("shirt_number"), 0)
        if from_fpl > 0:
            return from_fpl
    return _lookup_shirt_number(player_name, team_hint=team_hint)


def get_player_image_url(player_name: str, team_name: Optional[str] = None) -> Optional[str]:
    """Get player photo URL. Checks Transfermarkt map first (covers La Liga + transferred EPL players),
    then falls back to the FPL CDN."""
    _photo_url = lambda code: f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png"
    name_lower = player_name.lower()

    # TM map: covers La Liga players and EPL players whose FPL photo is stale post-transfer
    if _tm_photo_map:
        name_stripped = _strip_accents(name_lower)
        tm_url = _tm_photo_map.get(name_lower) or _tm_photo_map.get(name_stripped)
        if tm_url:
            return f"/api/player-photo/tm?name={player_name}"

    # Try exact match first
    if name_lower in fpl_player_codes:
        code = fpl_player_codes[name_lower]
        return _photo_url(code)

    # Try last name exact match (must be an exact key, not substring)
    last_name = name_lower.split()[-1] if name_lower.split() else ""
    if last_name and len(last_name) >= 4 and last_name in fpl_player_codes:
        # If team provided, verify it matches to avoid cross-team collisions
        # e.g. "bruno" exists for both Fernandes (Man Utd) and Guimarães (Newcastle)
        matched_team = fpl_player_team.get(last_name, "")
        if team_name and matched_team and _normalize_team(team_name) != _normalize_team(matched_team):
            pass  # Skip — wrong team
        else:
            return _photo_url(fpl_player_codes[last_name])

    # Try first+last name combo
    parts = name_lower.split()
    if len(parts) >= 2:
        first_last = f"{parts[0]} {parts[-1]}"
        if first_last in fpl_player_codes:
            return _photo_url(fpl_player_codes[first_last])

    # Strict partial: only match if a name part IS a key (not substring of a key)
    for part in parts:
        if len(part) >= 5 and part in fpl_player_codes:
            matched_team = fpl_player_team.get(part, "")
            if team_name and matched_team and _normalize_team(team_name) != _normalize_team(matched_team):
                continue
            return _photo_url(fpl_player_codes[part])

    # Containment fallback: search name in key or key in search name
    # Score by overlap length to pick the best match
    best_code, best_overlap = None, 0
    name_ascii = _strip_accents(name_lower)
    for key, code in fpl_player_codes.items():
        if name_lower in key or name_ascii in key:
            overlap = max(len(name_lower), len(key))
            if overlap > best_overlap:
                # Team check if available
                matched_team = fpl_player_team.get(key, "")
                if team_name and matched_team and _normalize_team(team_name) != _normalize_team(matched_team):
                    continue
                best_code, best_overlap = code, overlap
    if best_code:
        return _photo_url(best_code)

    return None


def _market_value_to_fantasy_price(market_value_millions: float) -> float:
    """Map TM market value into a fantasy-game-like price band for non-FPL leagues."""
    if market_value_millions <= 0:
        return 0.0
    scaled = 4.0 + min(10.0, math.log1p(market_value_millions) * 2.0)
    return round(min(max(scaled, 4.0), 14.0), 1)


def _get_transfermarkt_scraper():
    global _tm_scraper_instance
    if _tm_scraper_instance is None:
        from src.data_loaders.transfermarkt_scraper import TransfermarktScraper
        _tm_scraper_instance = TransfermarktScraper(cache_hours=168)
    return _tm_scraper_instance


def _get_transfermarkt_player_profile_cached(player_name: str, team_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch and cache season-level TM stats for leagues without FPL coverage."""
    global _tm_player_profile_cache
    cache_key = f"{_normalize_cache_key(player_name)}|{_normalize_cache_key(team_hint)}|{_current_season_year()}"
    cache_entry = _tm_player_profile_cache.get(cache_key)
    if _cache_entry_is_fresh(cache_entry):
        return cache_entry.get("data")

    data = None
    try:
        scraper = _get_transfermarkt_scraper()
        player = scraper.search_player(player_name, team_hint=team_hint)
        if player:
            stats = scraper.get_player_stats(
                player["slug"],
                player["player_id"],
                season=str(_current_season_year()),
            ) or {}
            market_value = scraper.get_player_market_value(player["slug"], player["player_id"])
            minutes_played = _safe_int(stats.get("minutes_played", 0), 0)
            goals = _safe_int(stats.get("goals", 0), 0)
            assists = _safe_int(stats.get("assists", 0), 0)
            per_90 = minutes_played / 90 if minutes_played > 0 else 0

            data = {
                "team": player.get("team"),
                "minutes_played": minutes_played,
                "appearances": _safe_int(stats.get("appearances", 0), 0),
                "goals": goals,
                "assists": assists,
                "goals_per_90": round(goals / per_90, 2) if per_90 > 0 else 0.0,
                "assists_per_90": round(assists / per_90, 2) if per_90 > 0 else 0.0,
                "market_value_m": market_value,
                "fantasy_price": _market_value_to_fantasy_price(market_value or 0.0),
                "is_starter": bool(stats.get("is_starter", False)),
            }
    except Exception as e:
        logger.debug(f"TM season profile unavailable for {player_name}: {e}")

    _tm_player_profile_cache[cache_key] = {
        "data": data,
        "expires": datetime.utcnow() + _TM_PLAYER_PROFILE_TTL,
    }
    return data


POPULAR_NAME_OVERRIDES = {
    "mohamed salah": "Salah",
    "bruno fernandes": "Bruno",
    "heung-min son": "Son",
    "son heung-min": "Son",
    "bukayo saka": "Saka",
    "erling haaland": "Haaland",
    "alexander isak": "Isak",
    "martin odegaard": "Odegaard",
    "martin ødegaard": "Odegaard",
    # La Liga popular names
    "pablo gavira": "Gavi",
    "gavi": "Gavi",
}


def get_popular_player_name(player_name: str, fpl_stats: Optional[Dict[str, Any]] = None) -> str:
    """Get natural football call-name used in narrative text."""
    full = (player_name or "").strip()
    lowered = full.lower()
    if lowered in POPULAR_NAME_OVERRIDES:
        return POPULAR_NAME_OVERRIDES[lowered]

    if fpl_stats:
        web_name = (fpl_stats.get("name") or "").strip()
        if web_name:
            return web_name

    if not full:
        return "This player"
    parts = full.split()
    if len(parts) == 1:
        return parts[0]
    particles = {"van", "von", "de", "da", "di", "del", "der", "dos", "le", "la"}
    if len(parts) >= 2 and parts[-2].lower() in particles:
        return f"{parts[-2]} {parts[-1]}"
    return parts[-1]


def calculate_implied_odds(prob: float) -> ImpliedOdds:
    """Convert probability to betting odds formats."""
    prob = max(0.01, min(0.99, prob))
    decimal_odds = round(1 / prob, 2)
    if prob >= 0.5:
        american = int(-100 * prob / (1 - prob))
        american_str = str(american)
    else:
        american = int(100 * (1 - prob) / prob)
        american_str = f"+{american}"
    if decimal_odds >= 2:
        numerator = int(round((decimal_odds - 1) * 1))
        fractional = f"{numerator}/1"
    else:
        denominator = int(round(1 / (decimal_odds - 1)))
        fractional = f"1/{denominator}"
    return ImpliedOdds(
        american=american_str, decimal=decimal_odds,
        fractional=fractional, implied_prob=round(prob, 3),
    )


def _resolve_player_from_element(element_id: int):
    """Map an FPL element ID to the matching inference_df row.

    Returns (row, fpl_data) tuple or (None, fpl_data) if no inference match.
    fpl_data may also be None if the element ID is unknown.
    """
    fpl_data = fpl_element_lookup.get(element_id)
    if not fpl_data or inference_df is None:
        return None, fpl_data

    web_name = (fpl_data.get("name") or "").strip()
    full_name = (fpl_data.get("full_name") or "").strip()
    team = (fpl_data.get("team") or "").strip()

    # 1. Exact match on web_name
    mask = inference_df["name"].str.lower() == web_name.lower()
    matches = inference_df[mask]
    if len(matches) == 1:
        return matches.iloc[0], fpl_data
    if len(matches) > 1 and team:
        team_m = matches[matches["team"].str.lower() == team.lower()]
        if not team_m.empty:
            return team_m.iloc[0], fpl_data

    # 2. Full name match
    mask = inference_df["name"].str.lower() == full_name.lower()
    matches = inference_df[mask]
    if len(matches) == 1:
        return matches.iloc[0], fpl_data
    if len(matches) > 1 and team:
        team_m = matches[matches["team"].str.lower() == team.lower()]
        if not team_m.empty:
            return team_m.iloc[0], fpl_data

    # 3. Word-overlap matching (accent-stripped)
    # Score candidates by how many name-words overlap, prefer team match
    web_ascii = _strip_accents(web_name.lower())
    full_ascii = _strip_accents(full_name.lower())
    fpl_words = set(full_ascii.split()) | set(web_ascii.split())
    # Remove very short words (initials like "b", "j") that cause false positives
    fpl_words = {w for w in fpl_words if len(w) >= 3}

    best_row = None
    best_score = 0
    for _, row in inference_df.iterrows():
        row_name = row.get("name", "").lower()
        row_ascii = _strip_accents(row_name)
        row_words = set(row_ascii.split())
        overlap = fpl_words & row_words
        if not overlap:
            continue
        # Score: number of overlapping words, bonus for team match
        score = len(overlap)
        row_team = row.get("team", "").lower()
        team_ok = not team or team.lower() in row_team or row_team in team.lower()
        if team_ok:
            score += 5  # strong team bonus
        if score > best_score:
            best_score = score
            best_row = row

    # Require at least 1 word overlap + team match, OR 2+ word overlap without team
    if best_row is not None and best_score >= 2:
        return best_row, fpl_data

    return None, fpl_data


def get_fpl_stats_for_player(name: str, team_hint: str = None) -> Optional[Dict]:
    """Look up FPL stats for a player by name.

    Args:
        name: Player name to search for
        team_hint: Optional team name from inference_df to validate matches
    """
    if not fpl_stats_cache:
        return None
    name_lower = name.lower()

    # Exact match (most reliable) — try with and without accents
    if name_lower in fpl_stats_cache:
        return _attach_shirt_number(fpl_stats_cache[name_lower], name, team_hint=team_hint)
    name_ascii = _strip_accents(name_lower)
    if name_ascii != name_lower and name_ascii in fpl_stats_cache:
        return _attach_shirt_number(fpl_stats_cache[name_ascii], name, team_hint=team_hint)

    # Full name containment (e.g. "Bukayo Saka" in "Bukayo Saka" or vice versa)
    # Score candidates by match quality to avoid false positives
    candidates = []
    for key, stats in fpl_stats_cache.items():
        if name_lower in key or key in name_lower:
            # Score: longer overlap = better match
            overlap = len(key) if key in name_lower else len(name_lower)
            candidates.append((overlap, stats))
    # Sort by match quality (longest overlap first)
    candidates.sort(key=lambda x: -x[0])

    # If team_hint provided, prefer candidates matching the team
    if candidates and team_hint:
        team_lower = team_hint.lower()
        team_matches = [(o, s) for o, s in candidates if _teams_match(s.get("team", ""), team_lower)]
        if team_matches:
            return _attach_shirt_number(team_matches[0][1], name, team_hint=team_hint)  # Already sorted by overlap

    if len(candidates) == 1:
        return _attach_shirt_number(candidates[0][1], name, team_hint=team_hint)
    if candidates:
        return _attach_shirt_number(candidates[0][1], name, team_hint=team_hint)

    # Last resort: match by last name only (must be unique and long enough)
    parts = name_lower.split()
    if parts:
        last_name = parts[-1]
        if len(last_name) >= 5:  # Only match long last names to avoid false positives
            last_name_matches = []
            for key, stats in fpl_stats_cache.items():
                if key.endswith(last_name) or last_name == key.split()[-1] if " " in key else last_name == key:
                    last_name_matches.append(stats)
            if len(last_name_matches) == 1:
                return _attach_shirt_number(last_name_matches[0], name, team_hint=team_hint)
            # If multiple matches, use team hint
            if last_name_matches and team_hint:
                team_lower = team_hint.lower()
                team_filtered = [s for s in last_name_matches if _teams_match(s.get("team", ""), team_lower)]
                if len(team_filtered) == 1:
                    return _attach_shirt_number(team_filtered[0], name, team_hint=team_hint)
    return None


def _teams_match(fpl_team: str, team_hint: str) -> bool:
    """Check if FPL team name matches a team hint (fuzzy)."""
    fpl_lower = fpl_team.lower()
    hint_lower = team_hint.lower()
    # Direct match
    if fpl_lower == hint_lower:
        return True
    # Check display name mapping
    display = _fpl_team_to_df_team(fpl_team).lower()
    if display == hint_lower or hint_lower in display or display in hint_lower:
        return True
    # Short name containment (e.g. "wolves" in "wolverhampton")
    if len(fpl_lower) >= 4 and (fpl_lower in hint_lower or hint_lower in fpl_lower):
        return True
    return False


def _safe_int(val, default=0) -> int:
    try:
        if val is None:
            return default
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _safe_float(val, default=0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _resolve_fpl_team_id(team_name: Optional[str]) -> Optional[int]:
    """Resolve a display team name to FPL numeric team ID."""
    if not team_name:
        return None
    team_lower = team_name.lower().strip()
    search = TEAM_BADGE_ALIASES.get(team_lower, team_lower)

    # Direct match
    if search in fpl_team_name_to_id:
        return fpl_team_name_to_id[search]

    # Fuzzy containment
    for key, tid in fpl_team_name_to_id.items():
        if len(key) < 5:
            continue
        if search in key or key in search:
            return tid
    return None


HISTORY_TEAM_ALIASES = {
    "arsenal": "Arsenal",
    "aston villa": "Aston Villa",
    "bournemouth": "Bournemouth",
    "brentford": "Brentford",
    "brighton": "Brighton",
    "brighton hove": "Brighton",
    "brighton & hove albion": "Brighton",
    "chelsea": "Chelsea",
    "crystal palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "ipswich": "Ipswich",
    "leicester": "Leicester",
    "liverpool": "Liverpool",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "man utd": "Manchester United",
    "man united": "Manchester United",
    "manchester united": "Manchester United",
    "newcastle": "Newcastle",
    "newcastle united": "Newcastle",
    "nottingham": "Nott'ham Forest",
    "nottingham forest": "Nott'ham Forest",
    "nott'm forest": "Nott'ham Forest",
    "southampton": "Southampton",
    "spurs": "Tottenham",
    "tottenham": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "west ham": "West Ham",
    "west ham united": "West Ham",
    "wolves": "Wolverhampton",
    "wolverhampton": "Wolverhampton",
    "wolverhampton wanderers": "Wolverhampton",
    "leeds": "Leeds",
    "leeds united": "Leeds",
    "burnley": "Burnley",
    "sunderland": "Sunderland",
}


def _normalize_history_team_name(team_name: Optional[str]) -> Optional[str]:
    if not team_name:
        return None
    key = team_name.lower().strip()
    key = TEAM_BADGE_ALIASES.get(key, key)
    return HISTORY_TEAM_ALIASES.get(key, team_name)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _safe_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _fixture_history_cache_path() -> Path:
    return Path(PROJECT_ROOT) / "data" / "cache" / "fixture_history_live.csv"


def _normalize_fixture_history_df(df):
    import pandas as pd

    expected_cols = ["Date", "Home", "Away", "HomeGoals", "AwayGoals"]
    if df is None or df.empty:
        return pd.DataFrame(columns=expected_cols)

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        logger.warning(f"Fixture history frame missing columns {missing}; skipping those rows")
        return pd.DataFrame(columns=expected_cols)

    out = df[expected_cols].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", "Home", "Away"]).copy()
    out["Home"] = out["Home"].astype(str).map(lambda x: _normalize_history_team_name(x) or x)
    out["Away"] = out["Away"].astype(str).map(lambda x: _normalize_history_team_name(x) or x)
    out["HomeGoals"] = pd.to_numeric(out["HomeGoals"], errors="coerce").fillna(0).astype(int)
    out["AwayGoals"] = pd.to_numeric(out["AwayGoals"], errors="coerce").fillna(0).astype(int)
    out = out.drop_duplicates(subset=["Date", "Home", "Away", "HomeGoals", "AwayGoals"])
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def _compute_history_backfill_seasons(local_df, max_seasons: int) -> List[int]:
    import pandas as pd

    now = datetime.utcnow()
    current_season = now.year if now.month >= 8 else now.year - 1
    if local_df is not None and not local_df.empty:
        local_last = pd.to_datetime(local_df["Date"], errors="coerce").max()
        if pd.notna(local_last):
            # Season year start: Aug->year, Jan-May->year-1
            start = local_last.year if local_last.month >= 8 else local_last.year - 1
        else:
            start = current_season - 4
    else:
        start = current_season - 4

    start = max(2010, int(start))
    if start > current_season:
        start = current_season
    seasons = list(range(start, current_season + 1))
    if len(seasons) > max_seasons:
        seasons = seasons[-max_seasons:]
    return seasons


def _load_cached_live_fixture_history(cache_path: Path):
    import pandas as pd

    if not cache_path.exists():
        return pd.DataFrame(), False
    try:
        cached = pd.read_csv(cache_path, usecols=["Date", "Home", "Away", "HomeGoals", "AwayGoals"])
        cached = _normalize_fixture_history_df(cached)
        ttl_hours = _safe_env_int("FIXTURE_HISTORY_CACHE_TTL_HOURS", 24)
        max_age = timedelta(hours=max(1, ttl_hours))
        age = datetime.utcnow() - datetime.utcfromtimestamp(cache_path.stat().st_mtime)
        is_fresh = age <= max_age
        return cached, is_fresh
    except Exception as e:
        logger.warning(f"Failed reading cached live fixture history: {e}")
        return pd.DataFrame(), False


def _fetch_live_fixture_history_backfill(local_df):
    import pandas as pd

    if not _env_flag("FIXTURE_HISTORY_ENABLE_API_BACKFILL", default=True):
        return pd.DataFrame()

    api_key = (
        os.getenv("FOOTBALL_DATA_API_KEY")
        or os.getenv("FOOTBALL_DATA_KEY")
        or os.getenv("FOOTBALL_DATA_TOKEN")
    )
    if not api_key:
        return pd.DataFrame()

    max_seasons = max(1, _safe_env_int("FIXTURE_HISTORY_MAX_API_SEASONS", 5))
    seasons = _compute_history_backfill_seasons(local_df, max_seasons=max_seasons)
    if not seasons:
        return pd.DataFrame()

    try:
        client = MatchHistoryApiClient(api_key=api_key)
    except Exception as e:
        logger.warning(f"Could not initialize live fixture backfill client: {e}")
        return pd.DataFrame()

    frames = []
    for season in seasons:
        try:
            df = client.get_premier_league_matches(season=season, status="FINISHED")
            if df is not None and not df.empty:
                frames.append(df[["Date", "Home", "Away", "HomeGoals", "AwayGoals"]].copy())
        except Exception as e:
            logger.warning(f"Failed fetching season {season}-{season+1} for fixture backfill: {e}")

    if not frames:
        return pd.DataFrame()

    live_df = pd.concat(frames, ignore_index=True)
    live_df = _normalize_fixture_history_df(live_df)
    if live_df.empty:
        return live_df

    # Keep mainly the post-local tail plus overlap window for safe dedupe.
    if local_df is not None and not local_df.empty:
        local_last = pd.to_datetime(local_df["Date"], errors="coerce").max()
        if pd.notna(local_last):
            overlap_cutoff = local_last - pd.Timedelta(days=380)
            live_df = live_df[live_df["Date"] >= overlap_cutoff].copy()

    return live_df.reset_index(drop=True)


def _load_fixture_history_local_only():
    """Load only the local CSV fixture history (fast, no API calls)."""
    import pandas as pd

    history_path = Path(PROJECT_ROOT) / "csv" / "premier-league-matches.csv"
    if not history_path.exists():
        return pd.DataFrame(columns=["Date", "Home", "Away", "HomeGoals", "AwayGoals"])
    try:
        local_df = pd.read_csv(
            history_path,
            usecols=["Date", "Home", "Away", "HomeGoals", "AwayGoals"],
        )
        return _normalize_fixture_history_df(local_df)
    except Exception as e:
        logger.warning(f"Failed loading local fixture history CSV: {e}")
        return pd.DataFrame(columns=["Date", "Home", "Away", "HomeGoals", "AwayGoals"])


def _load_fixture_history_live_cached_first():
    """Load fixture history with live cache precedence, then local CSV fallback."""
    import pandas as pd

    local_df = _load_fixture_history_local_only()
    cache_path = _fixture_history_cache_path()
    cached_live_df, _ = _load_cached_live_fixture_history(cache_path)

    parts = []
    if cached_live_df is not None and not cached_live_df.empty:
        parts.append(cached_live_df)
    if local_df is not None and not local_df.empty:
        parts.append(local_df)

    if not parts:
        return pd.DataFrame(columns=["Date", "Home", "Away", "HomeGoals", "AwayGoals"])

    return _normalize_fixture_history_df(pd.concat(parts, ignore_index=True))


def _backfill_fixture_history_async():
    """Run API backfill in background thread, then merge into global df."""
    global historical_matches_df, fixture_history_cache
    try:
        full_df = _load_fixture_history_dataset()
        if full_df is not None and not full_df.empty:
            historical_matches_df = full_df
            fixture_history_cache = {}  # Clear cache so new data is used
            latest_date = full_df["Date"].max()
            latest_str = str(latest_date.date()) if hasattr(latest_date, "date") else str(latest_date)[:10]
            print(
                f"Fixture history backfill complete: {len(full_df)} rows "
                f"(latest={latest_str})"
            )
    except Exception as e:
        logger.warning(f"Background fixture history backfill failed: {e}")


def _load_fixture_history_dataset():
    import pandas as pd

    history_path = Path(PROJECT_ROOT) / "csv" / "premier-league-matches.csv"
    local_df = pd.DataFrame(columns=["Date", "Home", "Away", "HomeGoals", "AwayGoals"])
    if history_path.exists():
        try:
            local_df = pd.read_csv(
                history_path,
                usecols=["Date", "Home", "Away", "HomeGoals", "AwayGoals"],
            )
            local_df = _normalize_fixture_history_df(local_df)
        except Exception as e:
            logger.warning(f"Failed loading local fixture history CSV: {e}")
            local_df = pd.DataFrame(columns=["Date", "Home", "Away", "HomeGoals", "AwayGoals"])

    cache_path = _fixture_history_cache_path()
    cached_live_df, cache_fresh = _load_cached_live_fixture_history(cache_path)
    live_df = cached_live_df.copy()

    if not cache_fresh:
        fetched_live_df = _fetch_live_fixture_history_backfill(local_df)
        if fetched_live_df is not None and not fetched_live_df.empty:
            live_df = fetched_live_df
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                live_df.to_csv(cache_path, index=False)
                logger.info(
                    f"Refreshed live fixture history cache: {len(live_df)} rows -> {cache_path}"
                )
            except Exception as e:
                logger.warning(f"Failed writing fixture history cache: {e}")

    parts = []
    if live_df is not None and not live_df.empty:
        parts.append(live_df)
    if local_df is not None and not local_df.empty:
        parts.append(local_df)

    if not parts:
        return pd.DataFrame(columns=["Date", "Home", "Away", "HomeGoals", "AwayGoals"])

    combined = _normalize_fixture_history_df(pd.concat(parts, ignore_index=True))
    if local_df is not None and not local_df.empty and (live_df is None or live_df.empty):
        local_last = pd.to_datetime(local_df["Date"], errors="coerce").max()
        if pd.notna(local_last):
            stale_days = (datetime.utcnow() - local_last.to_pydatetime()).days
            if stale_days > 365:
                logger.warning(
                    f"Fixture history appears stale by {stale_days} days; "
                    "set FOOTBALL_DATA_API_KEY to enable live backfill."
                )
    return combined


def get_fixture_history_context(
    team_name: Optional[str],
    opponent_name: Optional[str],
    years: int = 5,
    recent_n: int = 6,
) -> Optional[Dict[str, Any]]:
    """Summarize team-v-team fixture history from blended local+live dataset."""
    if historical_matches_df is None or team_name is None or opponent_name is None:
        return None

    team = _normalize_history_team_name(team_name)
    opp = _normalize_history_team_name(opponent_name)
    if not team or not opp:
        return None
    if team == opp:
        return None

    cache_key = (team, opp, years, recent_n)
    if cache_key in fixture_history_cache:
        return fixture_history_cache[cache_key]

    matches = historical_matches_df[
        ((historical_matches_df["Home"] == team) & (historical_matches_df["Away"] == opp)) |
        ((historical_matches_df["Home"] == opp) & (historical_matches_df["Away"] == team))
    ].copy()
    if matches.empty:
        fixture_history_cache[cache_key] = None
        return None

    # Prefer last N years; fallback to all-time if sparse because local csv may stop at 2023.
    cutoff = datetime.utcnow() - timedelta(days=365 * years)
    recent_window = matches[matches["Date"] >= cutoff].copy()
    if recent_window.empty:
        period_matches = matches.copy()
        period_label = "all available seasons"
    else:
        period_matches = recent_window
        period_label = f"last {years} years"

    period_matches = period_matches.sort_values("Date")
    if len(period_matches) > recent_n:
        period_matches = period_matches.tail(recent_n).copy()

    wins = draws = losses = goals_for = goals_against = 0
    meetings = []
    for _, m in period_matches.iterrows():
        home = m["Home"]
        away = m["Away"]
        hg = int(m.get("HomeGoals", 0))
        ag = int(m.get("AwayGoals", 0))
        date_str = str(m["Date"])[:10]
        if home == team:
            gf, ga = hg, ag
        else:
            gf, ga = ag, hg
        goals_for += gf
        goals_against += ga
        if gf > ga:
            wins += 1
            result = "W"
        elif gf < ga:
            losses += 1
            result = "L"
        else:
            draws += 1
            result = "D"
        meetings.append({
            "date": date_str,
            "result": result,
            "score": f"{gf}-{ga}",
            "home": home,
            "away": away,
        })

    # All-time PL context for cases where recent window is thin (e.g., promoted teams).
    all_time_wins = all_time_draws = all_time_losses = 0
    all_time_goals_for = all_time_goals_against = 0
    for _, m in matches.sort_values("Date").iterrows():
        home = m["Home"]
        hg = int(m.get("HomeGoals", 0))
        ag = int(m.get("AwayGoals", 0))
        if home == team:
            gf, ga = hg, ag
        else:
            gf, ga = ag, hg
        all_time_goals_for += gf
        all_time_goals_against += ga
        if gf > ga:
            all_time_wins += 1
        elif gf < ga:
            all_time_losses += 1
        else:
            all_time_draws += 1

    samples = len(period_matches)
    summary = {
        "team": team,
        "opponent": opp,
        "period_label": period_label,
        "samples": samples,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "recent_meetings": meetings[-min(3, len(meetings)):],
        "latest_meeting_date": meetings[-1]["date"] if meetings else None,
        "all_time_samples": int(len(matches)),
        "all_time_wins": all_time_wins,
        "all_time_draws": all_time_draws,
        "all_time_losses": all_time_losses,
        "all_time_goals_for": all_time_goals_for,
        "all_time_goals_against": all_time_goals_against,
    }
    fixture_history_cache[cache_key] = summary
    return summary


def get_player_matchup_context(
    player_name: str,
    team_hint: Optional[str] = None,
    next_fixture_data: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build opponent-specific context: recent form + historical output vs next opponent."""
    try:
        fpl_stats = get_fpl_stats_for_player(player_name, team_hint=team_hint or "")
        if not fpl_stats:
            return None
        player_id = _safe_int(fpl_stats.get("player_id"), 0)
        if player_id <= 0:
            return None

        client = FPLClient()
        summary = client.get_player_summary(player_id)
        history = summary.get("history", []) if summary else []
        if not history:
            return None

        played = [h for h in history if _safe_int(h.get("minutes"), 0) > 0]
        if not played:
            return None
        played = sorted(played, key=lambda h: h.get("kickoff_time") or "")

        recent = played[-5:]
        minutes_last_5 = [_safe_int(h.get("minutes"), 0) for h in played[-5:]]
        recent_goals = sum(_safe_int(h.get("goals_scored"), 0) for h in recent)
        recent_assists = sum(_safe_int(h.get("assists"), 0) for h in recent)
        recent_clean_sheets = sum(_safe_int(h.get("clean_sheets"), 0) for h in recent)
        recent_returns = sum(
            1 for h in recent
            if (_safe_int(h.get("goals_scored"), 0) + _safe_int(h.get("assists"), 0)) > 0
        )
        recent_avg_points = round(
            sum(_safe_int(h.get("total_points"), 0) for h in recent) / max(1, len(recent)),
            2,
        )
        recent_total_xg = round(
            sum(_safe_float(h.get("expected_goals"), 0.0) for h in recent),
            2,
        )

        # Past seasons summary for richer H2H context
        history_past = summary.get("history_past", [])
        past_season_minutes = sum(_safe_int(s.get("total_points"), 0) for s in history_past)
        seasons_played = len(history_past)

        opponent = next_fixture_data.get("opponent") if next_fixture_data else None
        opponent_id = _resolve_fpl_team_id(opponent)
        vs_opponent = []
        if opponent_id:
            vs_opponent = [
                h for h in played
                if _safe_int(h.get("opponent_team"), -1) == opponent_id
            ]

        vs_context = None
        if opponent:
            if vs_opponent:
                vs_recent = vs_opponent[-5:]
                vs_goals = sum(_safe_int(h.get("goals_scored"), 0) for h in vs_recent)
                vs_assists = sum(_safe_int(h.get("assists"), 0) for h in vs_recent)
                vs_clean_sheets = sum(_safe_int(h.get("clean_sheets"), 0) for h in vs_recent)
                vs_returns = sum(
                    1 for h in vs_recent
                    if (_safe_int(h.get("goals_scored"), 0) + _safe_int(h.get("assists"), 0)) > 0
                )
                vs_avg_points = round(
                    sum(_safe_int(h.get("total_points"), 0) for h in vs_recent) / max(1, len(vs_recent)),
                    2,
                )
                vs_context = {
                    "samples": len(vs_opponent),
                    "goals": vs_goals,
                    "assists": vs_assists,
                    "clean_sheets": vs_clean_sheets,
                    "returns": vs_returns,
                    "avg_points_recent": vs_avg_points,
                    "last_meeting_kickoff": vs_opponent[-1].get("kickoff_time"),
                }
            else:
                vs_context = {
                    "samples": 0,
                    "goals": 0,
                    "assists": 0,
                    "clean_sheets": 0,
                    "returns": 0,
                    "avg_points_recent": None,
                    "last_meeting_kickoff": None,
                }

        opponent_defense = None
        if opponent_id:
            team_meta = fpl_team_meta.get((fpl_team_names_by_id.get(opponent_id, "") or "").lower())
            defense_form = fpl_team_recent_defense.get(opponent_id)
            if team_meta or defense_form:
                opponent_defense = {
                    "strength_defence_home": _safe_int((team_meta or {}).get("strength_defence_home"), 0),
                    "strength_defence_away": _safe_int((team_meta or {}).get("strength_defence_away"), 0),
                    "avg_goals_conceded_last5": _safe_float((defense_form or {}).get("avg_goals_conceded_last5"), 0.0),
                    "clean_sheets_last5": _safe_int((defense_form or {}).get("clean_sheets_last5"), 0),
                    "samples": _safe_int((defense_form or {}).get("samples"), 0),
                }

        return {
            "recent_form": {
                "samples": len(recent),
                "goals": recent_goals,
                "assists": recent_assists,
                "clean_sheets": recent_clean_sheets,
                "returns": recent_returns,
                "avg_points": recent_avg_points,
                "total_xg": recent_total_xg,
            },
            "minutes_last_5": minutes_last_5,
            "games_played": len(played),
            "seasons_played": seasons_played,
            "opponent": opponent,
            "is_home": bool(next_fixture_data.get("is_home")) if next_fixture_data else None,
            "vs_opponent": vs_context,
            "opponent_defense": opponent_defense,
        }
    except Exception as e:
        logger.warning(f"Failed building matchup context for {player_name}: {e}")
        return None


TEAM_NICKNAME_OVERRIDES = {
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Tottenham": "Spurs",
    "Wolverhampton": "Wolves",
    "West Ham": "West Ham",
    "Brighton Hove": "Brighton",
    "Nottingham": "Nott'm Forest",
    "Nott'ham Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Newcastle United": "Newcastle",
}


def _team_nickname(name: Optional[str]) -> str:
    if not name:
        return "Team"
    return TEAM_NICKNAME_OVERRIDES.get(name, name)


def _parse_decimal_odds(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    try:
        val = float(text)
    except (TypeError, ValueError):
        return None
    if val <= 1.0:
        return None
    return round(val, 2)


def _avg_decimal_odds(lines: List[Dict[str, Any]], side: str) -> Optional[float]:
    values = []
    for line in lines:
        dec = _parse_decimal_odds(line.get(side))
        if dec is not None:
            values.append(dec)
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _classify_price_state(
    team_implied: Optional[float],
    opp_implied: Optional[float],
    team_side_avg: Optional[float],
    opp_side_avg: Optional[float],
    draw_avg: Optional[float] = None,
) -> str:
    # Primary classification is relative: team price vs opponent price.
    # edge = team_implied - opp_implied (positive = team favored)
    # A high draw probability means the market sees this as tight,
    # so we dampen the edge accordingly.
    draw_implied = (1.0 / draw_avg) if draw_avg and draw_avg > 1 else None
    if team_implied is not None and opp_implied is not None:
        edge = team_implied - opp_implied
        if draw_implied is not None and draw_implied >= 0.27:
            edge *= 0.5
        if abs(edge) < 0.10:
            return "in a balanced market"
        if edge >= 0.25:
            return "firm favorites"
        if edge >= 0.08:
            return "slight favorites"
        if edge <= -0.25:
            return "clear underdogs"
        if edge <= -0.08:
            return "slight underdogs"
        return "in a balanced market"

    # Decimal fallback if implied probabilities are unavailable.
    if team_side_avg is not None and opp_side_avg is not None:
        diff = opp_side_avg - team_side_avg
        if abs(diff) < 0.30:
            return "in a balanced market"
        if diff >= 1.00:
            return "firm favorites"
        if diff > 0.30:
            return "slight favorites"
        if diff <= -1.00:
            return "clear underdogs"
        if diff <= -0.30:
            return "slight underdogs"
        return "in a balanced market"

    # Last resort when only one side is known.
    if team_implied is not None:
        if team_implied >= 0.62:
            return "firm favorites"
        if team_implied >= 0.55:
            return "slight favorites"
        if team_implied >= 0.45:
            return "in a balanced market"
        if team_implied >= 0.38:
            return "slight underdogs"
        return "clear underdogs"

    return "awaiting stronger live pricing"


def _top_team_risk_names(team_name: str, top_n: int = 3, threshold: float = 0.45) -> List[str]:
    if inference_df is None:
        return []
    try:
        team_rows = inference_df[inference_df["team"].str.lower() == team_name.lower()].copy()
        if team_rows.empty:
            return []
        team_rows = team_rows.sort_values("ensemble_prob", ascending=False)
        names = []
        for _, row in team_rows.iterrows():
            prob = _safe_float(row.get("ensemble_prob"), 0.0)
            if prob < threshold:
                continue
            player_name = row.get("name", "")
            if not player_name:
                continue
            fpl = get_fpl_stats_for_player(player_name, team_hint=team_name)
            display = get_popular_player_name(player_name, fpl)
            if display not in names:
                names.append(display)
            if len(names) >= top_n:
                break
        return names
    except Exception:
        return []


def _build_team_market_insight(
    team_name: str,
    opponent_name: Optional[str],
    is_home: bool,
    moneyline_rows: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    team_short = _team_nickname(team_name)
    opponent_short = _team_nickname(opponent_name) if opponent_name else "opponent"
    odds_rows = moneyline_rows or []

    home_avg = _avg_decimal_odds(odds_rows, "home")
    draw_avg = _avg_decimal_odds(odds_rows, "draw")
    away_avg = _avg_decimal_odds(odds_rows, "away")

    team_side_avg = home_avg if is_home else away_avg
    opp_side_avg = away_avg if is_home else home_avg

    team_implied = (1.0 / team_side_avg) if team_side_avg and team_side_avg > 1 else None
    opp_implied = (1.0 / opp_side_avg) if opp_side_avg and opp_side_avg > 1 else None
    price_state = _classify_price_state(team_implied, opp_implied, team_side_avg, opp_side_avg, draw_avg)

    h2h = get_fixture_history_context(team_name, opponent_name, years=5, recent_n=6) if opponent_name else None
    h2h_part = ""
    if h2h and _safe_int(h2h.get("samples"), 0) > 0:
        period = str(h2h.get("period_label", "recent")).replace(" years", "y")
        h2h_part = (
            " H2H "
            f"{_safe_int(h2h.get('wins'), 0)}-"
            f"{_safe_int(h2h.get('draws'), 0)}-"
            f"{_safe_int(h2h.get('losses'), 0)} "
            f"({period})."
        )

    risk_names = _top_team_risk_names(team_name)
    risk_part = f" Risk watch: {', '.join(risk_names)}." if risk_names else ""

    headline = f"Market Insight: {team_short} {price_state} vs {opponent_short}."
    return f"{headline}{h2h_part}{risk_part}".replace("..", ".")


def get_next_fixture_for_team(team_name: str, injury_prob: float) -> Optional[Dict]:
    """Get next fixture info with betting odds for a team.

    Uses FPL API for fixture data (always has real kickoff times).
    Enriches with external odds if ODDS_API_KEY is available (not mock).
    """
    # Try external odds API only if we have a real API key (not mock data)
    odds_data = None
    if odds_client and odds_client.api_key:
        cs_data = odds_client.get_clean_sheet_odds(team_name)
        if cs_data:
            fixture_insight = get_clean_sheet_insight(team_name, injury_prob)
            odds_data = {
                "clean_sheet_odds": cs_data["american"],
                "win_probability": cs_data["win_probability"],
                "fixture_insight": fixture_insight,
            }
    # Add 1X2 bookie lines (works with live key and mock fallback)
    if odds_client:
        moneyline_data = odds_client.get_team_moneyline_1x2(team_name)
        if moneyline_data and moneyline_data.get("books"):
            if odds_data is None:
                odds_data = {}
            odds_data["moneyline_1x2"] = moneyline_data["books"]

    # Use FPL API fixtures (real kickoff times)
    try:
        client = FPLClient()
        gw = client.get_current_gameweek()
        if not gw:
            return None
        teams_list = client.get_teams()
        team_map = {t["id"]: t["name"] for t in teams_list}

        # Find team ID by matching name
        team_lower = team_name.lower()
        team_id = None
        for t in teams_list:
            if (team_lower in t["name"].lower() or t["name"].lower() in team_lower
                    or team_lower in t.get("short_name", "").lower()):
                team_id = t["id"]
                break
        if not team_id:
            search = TEAM_BADGE_ALIASES.get(team_lower, team_lower)
            for t in teams_list:
                if search in t["name"].lower() or t["name"].lower() in search:
                    team_id = t["id"]
                    break
        if not team_id:
            return None

        # Try current GW first, then next GW if all current fixtures are finished
        gw_ids_to_try = [gw["id"]]
        # Find next GW
        data = client.get_bootstrap()
        for event in data.get("events", []):
            if event.get("is_next") and event["id"] != gw["id"]:
                gw_ids_to_try.append(event["id"])
                break
            if event["id"] == gw["id"] + 1:
                gw_ids_to_try.append(event["id"])
                break

        for gw_id in gw_ids_to_try:
            fixtures = client.get_fixtures(gw_id)
            for f in fixtures:
                # Skip finished matches — only show upcoming/live
                if f.get("finished") or f.get("finished_provisional"):
                    continue

                fixture_result = None
                if f.get("team_h") == team_id:
                    fixture_result = {
                        "opponent": team_map.get(f["team_a"], "Unknown"),
                        "is_home": True,
                        "match_time": f.get("kickoff_time"),
                        "clean_sheet_odds": None,
                        "win_probability": None,
                        "fixture_insight": None,
                        "difficulty": f.get("team_h_difficulty"),
                    }
                elif f.get("team_a") == team_id:
                    fixture_result = {
                        "opponent": team_map.get(f["team_h"], "Unknown"),
                        "is_home": False,
                        "match_time": f.get("kickoff_time"),
                        "clean_sheet_odds": None,
                        "win_probability": None,
                        "fixture_insight": None,
                        "difficulty": f.get("team_a_difficulty"),
                    }
                if fixture_result:
                    # Merge external odds data if available
                    if odds_data:
                        fixture_result.update(odds_data)
                    fixture_result["fixture_insight"] = _build_team_market_insight(
                        team_name=team_name,
                        opponent_name=fixture_result.get("opponent"),
                        is_home=bool(fixture_result.get("is_home")),
                        moneyline_rows=fixture_result.get("moneyline_1x2"),
                    )
                    return fixture_result
    except Exception as e:
        logger.warning(f"Failed to get FPL fixture for {team_name}: {e}")

    return None


def get_upcoming_fixtures(team_name: str, count: int = 5) -> List[Dict]:
    """Get the next N fixtures for a team with FDR difficulty ratings."""
    try:
        client = FPLClient()
        teams_list = client.get_teams()
        team_map = {t["id"]: t["short_name"] for t in teams_list}

        team_lower = team_name.lower()
        team_id = None
        for t in teams_list:
            if (team_lower in t["name"].lower() or t["name"].lower() in team_lower
                    or team_lower in t.get("short_name", "").lower()):
                team_id = t["id"]
                break
        if not team_id:
            search = TEAM_BADGE_ALIASES.get(team_lower, team_lower)
            for t in teams_list:
                if search in t["name"].lower() or t["name"].lower() in search:
                    team_id = t["id"]
                    break
        if not team_id:
            return []

        # Get all remaining fixtures for the season
        data = client.get_bootstrap()
        current_gw = None
        for event in data.get("events", []):
            if event.get("is_current"):
                current_gw = event["id"]
                break
        if not current_gw:
            return []

        upcoming = []
        for gw_id in range(current_gw, current_gw + 10):
            if len(upcoming) >= count:
                break
            try:
                fixtures = client.get_fixtures(gw_id)
            except Exception:
                break
            for f in fixtures:
                if f.get("finished") or f.get("finished_provisional"):
                    continue
                if f.get("team_h") == team_id:
                    upcoming.append({
                        "opponent": team_map.get(f["team_a"], "???"),
                        "is_home": True,
                        "difficulty": f.get("team_h_difficulty", 3),
                        "match_time": f.get("kickoff_time"),
                    })
                elif f.get("team_a") == team_id:
                    upcoming.append({
                        "opponent": team_map.get(f["team_h"], "???"),
                        "is_home": False,
                        "difficulty": f.get("team_a_difficulty", 3),
                        "match_time": f.get("kickoff_time"),
                    })
                if len(upcoming) >= count:
                    break
        return upcoming
    except Exception as e:
        logger.warning(f"Failed to get upcoming fixtures for {team_name}: {e}")
        return []


def _build_la_liga_standings_rows(table: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    client = MatchHistoryApiClient()
    rows = []
    for row in table:
        team = row.get("team", {})
        short_name = client._normalize_la_liga_team(team.get("shortName", team.get("name", "")))
        rows.append({
            "position": row.get("position"),
            "name": short_name,
            "full_name": team.get("name", ""),
            "badge_url": LA_LIGA_BADGE_MAP.get(short_name.lower()),
            "played": row.get("playedGames", 0),
            "won": row.get("won", 0),
            "draw": row.get("draw", 0),
            "lost": row.get("lost", 0),
            "goals_for": row.get("goalsFor", 0),
            "goals_against": row.get("goalsAgainst", 0),
            "goal_difference": row.get("goalDifference", 0),
            "points": row.get("points", 0),
            "form": row.get("form"),
        })
    return rows


def _fetch_la_liga_standings_live() -> List[Dict[str, Any]]:
    started = time.perf_counter()
    client = MatchHistoryApiClient()
    data = client._get(f"competitions/{LA_LIGA_ID}/standings")
    table = data.get("standings", [{}])[0].get("table", [])
    rows = _build_la_liga_standings_rows(table)
    logger.info(
        "La Liga standings cache miss: fetched %s rows in %.2fs",
        len(rows),
        time.perf_counter() - started,
    )
    return rows


def _get_la_liga_standings_cached() -> List[Dict[str, Any]]:
    global _la_liga_standings_cache
    if _cache_entry_is_fresh(_la_liga_standings_cache):
        return _la_liga_standings_cache["data"] or []

    rows = _fetch_la_liga_standings_live()
    _la_liga_standings_cache = {
        "data": rows,
        "expires": datetime.utcnow() + _LA_LIGA_STANDINGS_TTL,
    }
    return rows


def _current_season_year() -> int:
    now = datetime.utcnow()
    return now.year if now.month >= 8 else now.year - 1


def _fetch_la_liga_fixture_dataset_live() -> Dict[str, Any]:
    started = time.perf_counter()
    client = MatchHistoryApiClient()
    season = _current_season_year()
    data = client._get(f"competitions/{LA_LIGA_ID}/matches", {"season": season})
    matches = data.get("matches", [])

    normalized_matches = []
    for match in matches:
        home_name = client._normalize_la_liga_team(
            match.get("homeTeam", {}).get("shortName", match.get("homeTeam", {}).get("name", ""))
        )
        away_name = client._normalize_la_liga_team(
            match.get("awayTeam", {}).get("shortName", match.get("awayTeam", {}).get("name", ""))
        )
        normalized_matches.append({
            "status": match.get("status"),
            "utcDate": match.get("utcDate"),
            "home": home_name,
            "away": away_name,
        })

    logger.info(
        "La Liga fixture dataset cache miss: fetched %s matches for %s-%s in %.2fs",
        len(normalized_matches),
        season,
        season + 1,
        time.perf_counter() - started,
    )
    return {"season": season, "matches": normalized_matches}


def _get_la_liga_fixture_dataset_cached() -> Dict[str, Any]:
    global _la_liga_fixture_dataset_cache, _la_liga_team_fixtures_cache, _la_liga_team_context_cache
    season = _current_season_year()
    cache_entry = _la_liga_fixture_dataset_cache
    if (
        cache_entry.get("season") == season
        and _cache_entry_is_fresh(cache_entry)
        and cache_entry.get("data") is not None
    ):
        return {"season": season, "matches": cache_entry["data"]}

    live_dataset = _fetch_la_liga_fixture_dataset_live()
    _la_liga_fixture_dataset_cache = {
        "season": live_dataset["season"],
        "data": live_dataset["matches"],
        "expires": datetime.utcnow() + _LA_LIGA_FIXTURE_DATASET_TTL,
    }
    _la_liga_team_fixtures_cache = {}
    _la_liga_team_context_cache = {}
    return live_dataset


def _get_la_liga_team_fixtures_cached(team_name: str, count: int = 5) -> List[Dict[str, Any]]:
    global _la_liga_team_fixtures_cache
    team_key = _normalize_cache_key(team_name)
    cache_key = f"{team_key}|{count}"
    cache_entry = _la_liga_team_fixtures_cache.get(cache_key)
    if _cache_entry_is_fresh(cache_entry):
        return cache_entry["data"] or []

    dataset = _get_la_liga_fixture_dataset_cached()
    matches = dataset.get("matches") or []
    today = str(datetime.utcnow().date())
    upcoming: List[Dict[str, Any]] = []

    for match in sorted(matches, key=lambda item: item.get("utcDate") or ""):
        if len(upcoming) >= count:
            break
        status = match.get("status")
        if status not in ("SCHEDULED", "TIMED"):
            continue
        match_date = str(match.get("utcDate") or "")[:10]
        if match_date and match_date < today:
            continue

        home = str(match.get("home", "") or "")
        away = str(match.get("away", "") or "")
        if _normalize_cache_key(home) == team_key:
            upcoming.append({
                "opponent": away,
                "is_home": True,
                "difficulty": 3,
                "match_time": match.get("utcDate"),
            })
        elif _normalize_cache_key(away) == team_key:
            upcoming.append({
                "opponent": home,
                "is_home": False,
                "difficulty": 3,
                "match_time": match.get("utcDate"),
            })

    _la_liga_team_fixtures_cache[cache_key] = {
        "data": upcoming,
        "expires": datetime.utcnow() + _LA_LIGA_TEAM_FIXTURES_TTL,
    }
    return upcoming


def _get_la_liga_moneyline_cached(team_name: str) -> Optional[Dict[str, Any]]:
    global _la_liga_moneyline_cache
    team_key = _normalize_cache_key(team_name)
    cache_entry = _la_liga_moneyline_cache.get(team_key)
    if _cache_entry_is_fresh(cache_entry):
        return cache_entry.get("data")

    data = None
    if odds_client:
        started = time.perf_counter()
        data = odds_client.get_la_liga_moneyline_1x2(team_name)
        logger.info(
            "La Liga moneyline cache miss for %s fetched in %.2fs",
            team_name,
            time.perf_counter() - started,
        )

    _la_liga_moneyline_cache[team_key] = {
        "data": data,
        "expires": datetime.utcnow() + _LA_LIGA_MONEYLINE_TTL,
    }
    return data


def _build_la_liga_next_fixture_data(team_name: str, upcoming_fixture: Optional[Dict[str, Any]], moneyline_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not upcoming_fixture:
        return None

    books = moneyline_data.get("books") if moneyline_data and moneyline_data.get("books") else None
    opponent = upcoming_fixture.get("opponent")
    is_home = bool(upcoming_fixture.get("is_home"))
    fixture_insight = (
        _build_team_market_insight(team_name, opponent, is_home, moneyline_rows=books)
        if books else None
    )
    return {
        "opponent": opponent,
        "is_home": is_home,
        "match_time": upcoming_fixture.get("match_time"),
        "clean_sheet_odds": None,
        "win_probability": None,
        "fixture_insight": fixture_insight,
        "moneyline_1x2": books,
    }


def _get_la_liga_team_context_cached(team_name: str, count: int = 1) -> Dict[str, Any]:
    global _la_liga_team_context_cache
    team_key = _normalize_cache_key(team_name)
    cache_key = f"{team_key}|{count}"
    cache_entry = _la_liga_team_context_cache.get(cache_key)
    if _cache_entry_is_fresh(cache_entry):
        return cache_entry["data"] or {"upcoming_fixtures": [], "next_fixture_data": None}

    upcoming = _get_la_liga_team_fixtures_cached(team_name, count=max(1, count))
    moneyline_data = _get_la_liga_moneyline_cached(team_name)
    next_fixture_data = _build_la_liga_next_fixture_data(
        team_name,
        upcoming[0] if upcoming else None,
        moneyline_data,
    )
    payload = {
        "upcoming_fixtures": upcoming,
        "next_fixture_data": next_fixture_data,
    }
    _la_liga_team_context_cache[cache_key] = {
        "data": payload,
        "expires": datetime.utcnow() + _LA_LIGA_TEAM_CONTEXT_TTL,
    }
    return payload


def get_upcoming_fixtures_la_liga(team_name: str, count: int = 5) -> List[Dict]:
    """Get the next N La Liga fixtures for a team using the shared cached dataset."""
    try:
        return _get_la_liga_team_fixtures_cached(team_name, count=count)
    except Exception as e:
        logger.warning(f"Failed to get La Liga upcoming fixtures for {team_name}: {e}")
        return []


def _is_defensive_position(position: str) -> bool:
    pos_lower = (position or "").lower()
    return any(p in pos_lower for p in ["def", "gk", "goalkeeper", "back"])


def _canonical_bookmaker_name(raw_name: str) -> Optional[str]:
    aliases = {
        "SkyBet": ["sky bet", "skybet"],
        "Paddy Power": ["paddy power", "paddypower"],
        "Betway": ["betway"],
    }
    haystack = (raw_name or "").lower()
    for canonical, keys in aliases.items():
        if any(k in haystack for k in keys):
            return canonical
    return None


def build_bookmaker_consensus(
    player_name: str,
    position: str,
    scoring_odds_data: Optional[Dict],
    clean_sheet_data: Optional[Dict],
    scorer_market_snapshot: Optional[Dict],
) -> Optional[Dict]:
    """Build direct-only market lines with preferred-first book selection."""
    defensive_market = _is_defensive_position(position) and clean_sheet_data is not None
    if defensive_market:
        # No direct clean-sheet market integration yet (only model estimate exists).
        return None

    if not scorer_market_snapshot:
        return None

    market_type = "score"
    market_label = "to score"
    target_books = ["SkyBet", "Paddy Power", "Betway"]
    preferred_map: Dict[str, Dict[str, Any]] = {}
    fallback_lines: List[Dict[str, Any]] = []
    seen_books = set()

    for raw_line in scorer_market_snapshot.get("lines", []):
        decimal = raw_line.get("decimal_odds")
        if not decimal or decimal <= 1:
            continue
        raw_book = raw_line.get("bookmaker", "Unknown")
        canonical = _canonical_bookmaker_name(raw_book)
        book_name = canonical or raw_book
        if book_name in seen_books:
            continue
        seen_books.add(book_name)

        normalized_line = {
            "bookmaker": book_name,
            "decimal_odds": round(float(decimal), 2),
            "implied_probability": round(1 / float(decimal), 3),
            "source": raw_line.get("source", "Live Market"),
        }
        if canonical in target_books:
            preferred_map[canonical] = normalized_line
        else:
            fallback_lines.append(normalized_line)

    lines = [preferred_map[book] for book in target_books if book in preferred_map]
    if len(lines) < 3:
        lines.extend(fallback_lines[: max(0, 3 - len(lines))])
    if not lines:
        return None

    average_decimal = round(sum(line["decimal_odds"] for line in lines) / len(lines), 2)
    average_probability = round(sum(line["implied_probability"] for line in lines) / len(lines), 3)
    probability_pct = round(average_probability * 100, 1)
    first_name = player_name.split()[0] if player_name else "Player"
    opponent = scorer_market_snapshot.get("opponent") if scorer_market_snapshot else None
    is_home = scorer_market_snapshot.get("is_home") if scorer_market_snapshot else None

    summary_text = (
        f"Bookies estimate {first_name}'s odds {market_label} is at an average of "
        f"{average_decimal:.2f}. A {probability_pct}% probability."
    )
    market_line = (
        f"{first_name} {'anytime scorer' if market_type == 'score' else 'clean sheet'} "
        f"- {average_decimal:.2f} (≈{round(average_probability * 100)}%)"
    )

    return {
        "market_type": market_type,
        "market_label": market_label,
        "average_decimal": average_decimal,
        "average_probability": average_probability,
        "summary_text": summary_text,
        "market_line": market_line,
        "lines": lines,
        "opponent": opponent,
        "is_home": is_home,
    }


def enhance_yara_response(
    base_response: Optional[Dict],
    player_row: Dict[str, Any],
    next_fixture_data: Optional[Dict],
    market_consensus: Optional[Dict],
    scoring_odds_data: Optional[Dict] = None,
    clean_sheet_data: Optional[Dict] = None,
    injury_records: Optional[List[Dict]] = None,
) -> Optional[Dict]:
    """Blend model narrative with fixture context and bookmaker consensus."""
    if not base_response and not market_consensus:
        return None

    name = player_row.get("name", "This player")
    first_name = name.split()[0] if name else "This player"
    team = player_row.get("team", "their team")
    injury_prob = float(player_row.get("ensemble_prob", 0.5))
    goals_per_90 = float(player_row.get("goals_per_90", 0) or 0)
    assists_per_90 = float(player_row.get("assists_per_90", 0) or 0)

    opponent = None
    is_home = None
    if next_fixture_data:
        opponent = next_fixture_data.get("opponent")
        is_home = next_fixture_data.get("is_home")

    if market_consensus and not opponent:
        opponent = market_consensus.get("opponent")
        is_home = market_consensus.get("is_home")

    fallback_model_prob = 0.0
    if _is_defensive_position(player_row.get("position", "")) and clean_sheet_data:
        fallback_model_prob = float(clean_sheet_data.get("clean_sheet_probability", 0.0) or 0.0)
    elif scoring_odds_data:
        fallback_model_prob = float(scoring_odds_data.get("score_probability", 0.0) or 0.0)

    yara_probability = (
        base_response.get("yara_probability")
        if base_response
        else (fallback_model_prob or (market_consensus.get("average_probability") if market_consensus else 0.0))
    )
    market_probability = (
        market_consensus.get("average_probability")
        if market_consensus
        else (base_response.get("market_probability") if base_response else None)
    )

    is_defensive = _is_defensive_position(player_row.get("position", ""))
    market_type_label = "to keep a clean sheet" if is_defensive else "to score"
    yara_pct = round(float(yara_probability) * 100)
    risk_pct = round(injury_prob * 100)

    # Stat-first template fallback — OptaJoe voice
    if market_consensus and market_probability is not None:
        market_pct = round(float(market_probability) * 100)
        edge = yara_pct - market_pct
        if edge > 0:
            template_text = (
                f"{abs(edge)}% gap between me and the bookies on {first_name}. "
                f"I have {yara_pct}% {market_type_label}, they have {market_pct}%. "
                f"{goals_per_90:.2f} goals/90 at {risk_pct}% injury risk."
            )
        elif edge < 0:
            template_text = (
                f"{market_pct}% — Bookies are generous on {first_name}. "
                f"I'm at {yara_pct}% {market_type_label}. "
                f"The scoring rate ({goals_per_90:.2f}/90) doesn't justify the market price."
            )
        else:
            template_text = (
                f"~{yara_pct}% — Market and model agree on {first_name} {market_type_label}. "
                f"{goals_per_90:.2f} goals/90 and {assists_per_90:.2f} assists/90 at {risk_pct}% risk."
            )
    else:
        template_text = (
            f"{yara_pct}% — My projection for {first_name} {market_type_label}. "
            f"{goals_per_90:.2f} goals/90, {assists_per_90:.2f} assists/90, {risk_pct}% injury risk."
        )
    if opponent:
        venue = "home" if is_home else "away"
        template_text += f" Next: {venue} vs {opponent}."

    # Use RAG context for richer LLM enrichment
    response_text = template_text
    try:
        from src.inference.context_rag import retrieve_player_context
        context_chunks = retrieve_player_context(
            player_row,
            extra_context={"next_fixture": {"opponent": opponent, "is_home": is_home}} if opponent else None,
            query="scoring odds injury availability fixture market opponent goals assists",
            top_k=10,
            include_open_question=False,
        )
        # Add market data as a chunk if available
        if market_consensus and market_probability is not None:
            context_chunks.append({
                "kind": "market",
                "text": f"Bookmaker average: {round(float(market_probability) * 100)}% {market_type_label}. Yara model: {yara_pct}%.",
                "tags": set(),
                "weight": 1.5,
            })

        llm_task = (
            f"Write Yara's market take on {first_name} in 2-3 sentences. "
            f"Lead with the gap between your number and the bookmakers. "
            f"Say whether the market is sleeping on him or has it right, and why. "
            f"Reference scoring rate and the opponent. Confident, direct, no hedging."
        )
        enriched = generate_grounded_narrative(
            task=llm_task,
            player_name=name,
            context_chunks=context_chunks,
            fallback_text=template_text,
            require_open_question=False,
        )
        if enriched and enriched != template_text:
            response_text = enriched
    except Exception:
        pass  # Fall back to template text

    fpl_tip = base_response.get("fpl_tip") if base_response else None
    if not fpl_tip:
        if injury_prob >= 0.45:
            fpl_tip = f"I would have bench cover ready. {risk_pct}% risk is too high to go in blind."
        elif yara_probability >= 0.4:
            fpl_tip = f"Start him. I have {yara_pct}% confidence in the output this week."
        else:
            fpl_tip = "Playable but not a captaincy shout. Steady floor, limited ceiling."

    return {
        "response_text": response_text,
        "fpl_tip": fpl_tip,
        "market_probability": market_probability,
        "yara_probability": float(yara_probability),
        "market_odds_decimal": market_consensus.get("average_decimal") if market_consensus else (
            base_response.get("market_odds_decimal") if base_response else None
        ),
        "bookmaker": "Bookies Avg" if market_consensus else (base_response.get("bookmaker") if base_response else None),
    }


def get_personalized_insights(row: dict) -> List[str]:
    """Generate personalized insights using the story generator."""
    risk_factors = generate_risk_factors_list(row)
    insights = []
    main_rec = get_recommendation_text(row)
    if main_rec:
        insights.append(main_rec)
    for factor in risk_factors[:2]:
        if factor["impact"] in ["high_risk", "moderate_risk"]:
            insights.append(f"{factor['factor']}: {factor['description']}")
        elif factor["impact"] == "protective":
            insights.append(f"{factor['factor']}: {factor['description']}")
    return insights[:4]


def calculate_expected_points(
    enriched_row: Dict,
    fpl_stats: Optional[Dict],
    next_fixture: Optional[Dict],
    injury_prob: float,
    scoring_odds_data: Optional[Dict] = None,
    clean_sheet_data: Optional[Dict] = None,
) -> Optional[Dict]:
    """Calculate FPL expected points for next gameweek, discounted by injury risk."""
    position = str(enriched_row.get("position", "")).lower()
    if any(t in position for t in ("midfield", "mid", "playmaker", "am", "cm", "dm")):
        pos_baseline = 3.6
    elif any(t in position for t in ("forward", "striker", "winger", "centre-forward", "center-forward", "attacker", "fwd")):
        pos_baseline = 4.0
    elif any(t in position for t in ("def", "back")):
        pos_baseline = 3.8
    elif any(t in position for t in ("gk", "goalkeeper")):
        pos_baseline = 3.2
    else:
        pos_baseline = 3.5

    if fpl_stats:
        status = str(fpl_stats.get("status", "a") or "a").lower()
        chance = fpl_stats.get("chance_of_playing")
        chance_int = _safe_int(chance, -1) if chance is not None else None
        if status in {"i", "u"} or chance_int == 0:
            return None

        ppg = _safe_float(fpl_stats.get("points_per_game", 0))
        form = _safe_float(fpl_stats.get("form", 0))
        minutes = _safe_int(fpl_stats.get("minutes", 0))

        if minutes < 270 or ppg == 0:
            return None

        base = ppg * 0.50 + form * 0.35 + pos_baseline * 0.15
        projection_source = "fpl"
    else:
        minutes = _safe_int(enriched_row.get("minutes", enriched_row.get("minutes_played", 0)), 0)
        if minutes < 270:
            return None

        goals_per_90 = _safe_float(enriched_row.get("goals_per_90", 0), 0.0)
        assists_per_90 = _safe_float(enriched_row.get("assists_per_90", 0), 0.0)
        availability = max(0.35, 1 - (injury_prob * 0.45))
        score_prob = _safe_float((scoring_odds_data or {}).get("score_probability", 0.0), 0.0)
        clean_sheet_prob = _safe_float((clean_sheet_data or {}).get("clean_sheet_probability", 0.0), 0.0)

        if any(t in position for t in ("def", "back", "goalkeeper", "gk")):
            goal_points = 6.0
            cs_points = 4.0
        elif any(t in position for t in ("midfield", "mid", "playmaker", "am", "cm", "dm")):
            goal_points = 5.0
            cs_points = 1.0
        else:
            goal_points = 4.0
            cs_points = 0.0

        assist_points = assists_per_90 * 3.0 * availability
        attacking_points = score_prob * goal_points + assist_points
        defensive_points = clean_sheet_prob * cs_points
        base = pos_baseline * 0.45 + attacking_points + defensive_points
        ppg = base
        form = goals_per_90 + assists_per_90 + clean_sheet_prob
        projection_source = "model"

    fdr = int(next_fixture.get("difficulty", 3)) if next_fixture else 3
    fdr_map = {1: 1.15, 2: 1.05, 3: 1.0, 4: 0.90, 5: 0.80}
    fixture_multiplier = fdr_map.get(fdr, 1.0)

    risk_weight = 0.6 if injury_prob >= 0.4 else (0.4 if injury_prob >= 0.2 else 0.25)
    injury_discount = injury_prob * risk_weight

    expected = round(base * fixture_multiplier * (1 - injury_discount), 1)

    # Confidence should reflect sample quality + risk stability, not just "played enough".
    established_sample = minutes >= 1600
    strong_output_signal = ppg >= 3.8 and form >= 1.0 if projection_source == "model" else ppg >= 3.8 and form >= 3.5
    stable_availability = injury_prob < 0.30
    usable_sample = minutes >= 900 and ppg >= 2.2

    if established_sample and strong_output_signal and stable_availability:
        confidence = "high"
    elif usable_sample and injury_prob < 0.50:
        confidence = "medium"
    else:
        confidence = "low"

    opponent = next_fixture.get("opponent", "opponent") if next_fixture else "their next opponent"
    venue = "at home" if next_fixture and next_fixture.get("is_home") else "away"
    if projection_source == "fpl":
        breakdown = (
            f"Base {base:.1f}pts (PPG {ppg:.1f}, form {form:.1f}). "
            f"Fixture vs {opponent} ({venue}, FDR {fdr}) gives {fixture_multiplier:.2f}x. "
            f"Injury risk {round(injury_prob * 100)}% applies {round(injury_discount * 100)}% discount."
        )
    else:
        breakdown = (
            f"Model baseline {base:.1f}pts from season output and role profile. "
            f"Fixture vs {opponent} ({venue}) applies {fixture_multiplier:.2f}x. "
            f"Injury risk {round(injury_prob * 100)}% applies {round(injury_discount * 100)}% discount."
        )

    return {
        "expected_points": max(0.5, expected),
        "base_points": round(base, 1),
        "injury_discount_pct": round(injury_discount, 3),
        "fixture_multiplier": fixture_multiplier,
        "confidence": confidence,
        "breakdown": breakdown,
    }


def _position_importance_role(position: str) -> str:
    """Map raw position text into a stable role bucket for importance scoring."""
    pos = str(position or "").strip().lower()
    if any(t in pos for t in ("goalkeeper", "keeper", "gk")):
        return "goalkeeper"
    if any(t in pos for t in ("def", "back")):
        return "defender"
    if any(t in pos for t in ("midfield", "mid", "playmaker", "am", "cm", "dm")):
        return "midfielder"
    if any(t in pos for t in ("forward", "striker", "winger", "attacker", "fwd")):
        return "attacker"
    return "other"


def calculate_player_importance(
    player_row: Dict[str, Any],
    fpl_stats: Optional[Dict[str, Any]] = None,
    matchup_context: Optional[Dict[str, Any]] = None,
    fixture_history: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build a fan-facing importance score from FPL exposure + football context.

    This is not a talent rating; it measures practical decision impact for fan picks:
    ownership/price exposure, reliability, recent form, and fixture/H2H signal.
    """
    stats = fpl_stats or {}
    matchup = matchup_context or {}
    fixture = fixture_history or {}

    ownership = _safe_float(stats.get("selected_by", player_row.get("selected_by")), 0.0)
    price = _safe_float(stats.get("price", player_row.get("price")), 0.0)
    ppg = _safe_float(stats.get("points_per_game", player_row.get("points_per_game")), 0.0)
    form = _safe_float(stats.get("form", player_row.get("form")), 0.0)
    minutes = _safe_int(stats.get("minutes", player_row.get("minutes")), 0)
    role = _position_importance_role(player_row.get("position", ""))

    # If no useful signal is available at all, don't emit noisy placeholders.
    if (
        ownership <= 0
        and price <= 0
        and ppg <= 0
        and form <= 0
        and minutes <= 0
        and not matchup
        and not fixture
    ):
        return None

    if price >= 11.0:
        price_tier = "Premium"
    elif price >= 8.0:
        price_tier = "High"
    elif price >= 6.0:
        price_tier = "Mid"
    else:
        price_tier = "Budget"

    role_weights = {
        "attacker": 1.06,
        "midfielder": 1.02,
        "defender": 0.96,
        "goalkeeper": 0.90,
        "other": 0.95,
    }

    if ownership >= 25 or price >= 10.5:
        role_importance = "Talisman"
    elif role == "attacker" and ownership >= 10:
        role_importance = "Primary attacker"
    elif role == "midfielder" and (ownership >= 8 or ppg >= 4.5):
        role_importance = "Creative hub"
    elif role in {"defender", "goalkeeper"} and ppg >= 4:
        role_importance = "Defensive anchor"
    elif ownership <= 5:
        role_importance = "Differential"
    else:
        role_importance = "Squad option"

    recent_form = matchup.get("recent_form") or {}
    recent_samples = _safe_int(recent_form.get("samples", 0), 0)
    recent_goals = _safe_int(recent_form.get("goals", 0), 0)
    recent_assists = _safe_int(recent_form.get("assists", 0), 0)
    recent_clean_sheets = _safe_int(recent_form.get("clean_sheets", 0), 0)
    if recent_samples > 0:
        if role in {"defender", "goalkeeper"}:
            form_signal_value = min(recent_clean_sheets / max(1, recent_samples), 1.0)
            form_signal = f"{recent_clean_sheets} clean sheets in last {recent_samples}"
        else:
            recent_gi = recent_goals + recent_assists
            form_signal_value = min((recent_gi / max(1, recent_samples)) / 1.2, 1.0)
            form_signal = f"{recent_goals} goals + {recent_assists} assists in last {recent_samples}"
    else:
        form_signal_value = min(form / 10.0, 1.0)
        form_signal = "Form sample still building"

    vs_opponent = matchup.get("vs_opponent") or {}
    vs_samples = _safe_int(vs_opponent.get("samples", 0), 0)
    opponent = matchup.get("opponent") or fixture.get("opponent") or "the opponent"
    if vs_samples > 0:
        if role in {"defender", "goalkeeper"}:
            vs_clean_sheets = _safe_int(vs_opponent.get("clean_sheets", 0), 0)
            h2h_value = min(vs_clean_sheets / max(1, vs_samples), 1.0)
            h2h_signal = f"{vs_clean_sheets} clean sheets in {vs_samples} vs {opponent}"
        else:
            vs_gi = _safe_int(vs_opponent.get("goals", 0), 0) + _safe_int(vs_opponent.get("assists", 0), 0)
            h2h_value = min((vs_gi / max(1, vs_samples)) / 1.2, 1.0)
            h2h_signal = f"{vs_gi} returns in {vs_samples} vs {opponent}"
    else:
        fixture_samples = _safe_int(fixture.get("samples", 0), 0)
        if fixture_samples >= 3:
            wins = _safe_int(fixture.get("wins", 0), 0)
            draws = _safe_int(fixture.get("draws", 0), 0)
            losses = _safe_int(fixture.get("losses", 0), 0)
            h2h_value = min(wins / max(1, fixture_samples), 1.0)
            h2h_signal = f"Team trend {wins}-{draws}-{losses} vs {opponent} ({fixture_samples} recent)"
        else:
            h2h_value = 0.45
            h2h_signal = f"Direct H2H signal still thin vs {opponent}"

    ownership_score = min(max(ownership / 40.0, 0.0), 1.0)
    price_score = min(max(price / 12.5, 0.0), 1.0)
    ppg_score = min(max(ppg / 8.0, 0.0), 1.0)
    minutes_score = min(max(minutes / 2500.0, 0.0), 1.0)
    role_score = role_weights.get(role, 0.95)

    composite = (
        ownership_score * 0.30
        + price_score * 0.18
        + ppg_score * 0.16
        + minutes_score * 0.10
        + role_score * 0.10
        + form_signal_value * 0.10
        + h2h_value * 0.06
    )
    score = round(min(max(composite * 100, 0.0), 100.0), 1)

    if score >= 75:
        tier = "Core"
    elif score >= 60:
        tier = "High"
    elif score >= 45:
        tier = "Balanced"
    else:
        tier = "Differential"

    captaincy_proxy_pct = min(
        100.0,
        max(
            0.0,
            ownership * 1.25
            + max(price - 8.0, 0.0) * 6.0
            + max(ppg - 4.0, 0.0) * 10.0,
        ),
    )

    call_name = get_popular_player_name(player_row.get("name", "This player"), fpl_stats=stats or None)
    summary = (
        f"{call_name} grades as {tier.lower()} importance "
        f"({ownership:.1f}% owned, {price_tier.lower()} price tier at £{price:.1f}m, {ppg:.1f} PPG). "
        f"Form: {form_signal}. H2H: {h2h_signal}."
    )

    return {
        "score": score,
        "tier": tier,
        "ownership_pct": round(ownership, 2),
        "price": round(price, 1) if price > 0 else None,
        "price_tier": price_tier,
        "captaincy_proxy_pct": round(captaincy_proxy_pct, 1),
        "role_importance": role_importance,
        "form_signal": form_signal,
        "h2h_signal": h2h_signal,
        "summary": summary,
    }


def calculate_risk_comparison(player_name: str, team: str, position: str, prob: float) -> Optional[Dict]:
    """Compare player's risk to squad and position averages."""
    if inference_df is None:
        return None

    team_players = inference_df[inference_df["team"] == team]
    if team_players.empty:
        return None

    squad_avg = round(float(team_players["ensemble_prob"].mean()), 3)
    squad_rank = int((team_players["ensemble_prob"] >= prob).sum())
    squad_total = len(team_players)

    pos_lower = position.lower()
    if any(t in pos_lower for t in ("midfield", "mid", "playmaker", "am", "cm", "dm")):
        pos_group = "Midfielder"
        pos_filter = inference_df["position"].str.lower().str.contains("midfield|\\bmid\\b|playmaker|\\bam\\b|\\bcm\\b|\\bdm\\b", regex=True, na=False)
    elif any(t in pos_lower for t in ("forward", "striker", "winger", "centre-forward", "center-forward", "attacker", "fwd")):
        pos_group = "Forward"
        pos_filter = inference_df["position"].str.lower().str.contains(
            "forward|striker|winger|centre-forward|center-forward|\\battacker\\b|\\bfwd\\b",
            regex=True,
            na=False,
        )
    elif any(t in pos_lower for t in ("def", "back")):
        pos_group = "Defender"
        pos_filter = inference_df["position"].str.lower().str.contains("def|back", regex=True, na=False)
    else:
        pos_group = "Goalkeeper"
        pos_filter = inference_df["position"].str.lower().str.contains("gk|goalkeeper", regex=True, na=False)

    pos_players = inference_df[pos_filter]
    pos_avg = round(float(pos_players["ensemble_prob"].mean()), 3) if not pos_players.empty else squad_avg
    pos_rank = int((pos_players["ensemble_prob"] >= prob).sum()) if not pos_players.empty else 1
    pos_total = len(pos_players) if not pos_players.empty else 1

    return {
        "squad_avg_risk": squad_avg,
        "position_avg_risk": pos_avg,
        "squad_rank": max(1, squad_rank),
        "squad_total": squad_total,
        "position_group": pos_group,
        "position_rank": max(1, pos_rank),
        "position_total": pos_total,
    }


def player_row_to_risk(row) -> PlayerRisk:
    """Convert a DataFrame row to a PlayerRisk response."""
    prob = _safe_float(row.get("ensemble_prob", row.get("calibrated_prob", 0.5)), 0.5)
    # Use player_injury_count (has data) instead of previous_injuries (all zeros in inference_df)
    prev_injuries = _safe_int(row.get("player_injury_count", row.get("previous_injuries", 0)))
    avg_severity = _safe_float(row.get("player_avg_severity", 0))
    total_days = int(prev_injuries * avg_severity) if prev_injuries > 0 else 0
    player_name = row.get("name", "Unknown")
    team = row.get("team", row.get("player_team", "Unknown"))
    player_league = row.get("league", "Premier League")
    is_la_liga = player_league == "La Liga"

    # Build enriched row with corrected injury fields + FPL stats
    # inference_df is now enriched at startup with real Transfermarkt data,
    # but story generators read "previous_injuries" so we map it
    enriched_row = dict(row)
    enriched_row["previous_injuries"] = prev_injuries
    # Inject normalized percentile score so story generators use the same % the frontend shows
    enriched_row["risk_score_pct"] = round(normalize_risk_score(prob, row.get("league")))
    # Use total_days_lost from scraped data if available, otherwise estimate
    total_days_from_scrape = _safe_int(row.get("total_days_lost", 0))
    if total_days_from_scrape > 0:
        total_days = total_days_from_scrape
    enriched_row["total_days_lost"] = total_days
    # Use days_since_last_injury from scraped data (real value, not default 365)
    days_since = _safe_int(row.get("days_since_last_injury", 365), 365)
    enriched_row["days_since_last_injury"] = days_since

    # Enrich with FPL stats — La Liga players are not in FPL so skip to avoid
    # matching La Liga players to wrong EPL entries (e.g. Fermín López → "H.Bueno")
    fpl_stats = get_fpl_stats_for_player(player_name, team_hint=team) if not is_la_liga else None
    tm_profile = _get_transfermarkt_player_profile_cached(player_name, team_hint=team) if is_la_liga else None
    enriched_row["story_name"] = get_popular_player_name(player_name, fpl_stats)
    if fpl_stats:
        enriched_row.update({
            "goals": fpl_stats.get("goals", 0),
            "assists": fpl_stats.get("assists", 0),
            "goals_per_90": fpl_stats.get("goals_per_90", 0),
            "assists_per_90": fpl_stats.get("assists_per_90", 0),
            "price": fpl_stats.get("price", 0),
            "form": fpl_stats.get("form", 0),
            "minutes": fpl_stats.get("minutes", 0),
            "shirt_number": fpl_stats.get("shirt_number"),
            "display_name": fpl_stats.get("name", ""),
            "popular_name": get_popular_player_name(player_name, fpl_stats),
        })
    elif tm_profile:
        enriched_row.update({
            "goals": tm_profile.get("goals", 0),
            "assists": tm_profile.get("assists", 0),
            "goals_per_90": tm_profile.get("goals_per_90", 0),
            "assists_per_90": tm_profile.get("assists_per_90", 0),
            "price": tm_profile.get("fantasy_price", 0),
            "market_value_m": tm_profile.get("market_value_m"),
            "minutes": tm_profile.get("minutes_played", 0),
            "minutes_played": tm_profile.get("minutes_played", 0),
            "appearances": tm_profile.get("appearances", 0),
            "popular_name": get_popular_player_name(player_name, None),
        })
    if not is_la_liga and _safe_int(enriched_row.get("shirt_number"), 0) <= 0:
        enriched_row["shirt_number"] = _lookup_shirt_number(player_name, team_hint=team)

    # Compute within-league risk percentile for context (e.g. "higher risk than 92% of players")
    risk_percentile = None
    player_league = enriched_row.get("league")
    _pctile_series = _league_prob_series(player_league)
    if _pctile_series is not None:
        risk_percentile = round(float((_pctile_series <= prob).mean()), 3)
    enriched_row["risk_percentile"] = risk_percentile

    player_league = enriched_row.get("league", "Premier League")
    if player_league == "La Liga":
        next_fixture_data = _get_la_liga_team_context_cached(team, count=1).get("next_fixture_data")
    else:
        next_fixture_data = get_next_fixture_for_team(team, prob)
    fixture_history_data = get_fixture_history_context(
        team_name=team,
        opponent_name=next_fixture_data.get("opponent") if next_fixture_data else None,
        years=5,
    )
    clean_sheet_data = calculate_clean_sheet_odds(enriched_row)
    matchup_context_data = get_player_matchup_context(
        player_name=player_name,
        team_hint=team,
        next_fixture_data=next_fixture_data,
    )
    importance_data = calculate_player_importance(
        player_row=enriched_row,
        fpl_stats=fpl_stats,
        matchup_context=matchup_context_data,
        fixture_history=fixture_history_data,
    )

    scorer_market_snapshot = None
    if odds_client:
        scorer_market_snapshot = odds_client.get_anytime_scorer_market_snapshot(
            team,
            player_name,
            league=player_league,
        )

    bookmaker_consensus_data = build_bookmaker_consensus(
        player_name=player_name,
        position=enriched_row.get("position", ""),
        scoring_odds_data=None,
        clean_sheet_data=clean_sheet_data,
        scorer_market_snapshot=scorer_market_snapshot,
    )

    # Look up per-injury detail records for this player
    injury_records = []
    if injury_detail_df is not None:
        player_detail = injury_detail_df[injury_detail_df["name"].str.lower() == player_name.lower()]
        if not player_detail.empty:
            for _, irow in player_detail.sort_values("injury_datetime", ascending=False).iterrows():
                injury_records.append({
                    "date": str(irow.get("injury_datetime", ""))[:10] if irow.get("injury_datetime") is not None else None,
                    "body_area": irow.get("body_area", "unknown"),
                    "injury_type": irow.get("injury_type", "unknown"),
                    "injury_raw": irow.get("injury_raw", ""),
                    "severity_days": int(irow.get("severity_days", 0)),
                    "games_missed": int(irow.get("games_missed", 0)),
                })

    # If player_injury_count was missing from player_history.pkl (filled to 0),
    # derive the count and total days from injury_records (actual injury data).
    if prev_injuries == 0 and injury_records:
        prev_injuries = len(injury_records)
        computed_days = sum(r.get("severity_days", 0) for r in injury_records)
        if total_days == 0 and computed_days > 0:
            total_days = computed_days
        enriched_row["previous_injuries"] = prev_injuries
        enriched_row["total_days_lost"] = total_days

    narrative_context = {
        "next_fixture": next_fixture_data,
        "fixture_history": fixture_history_data,
        "bookmaker_consensus": bookmaker_consensus_data,
        "matchup_context": matchup_context_data,
        "injury_records": injury_records,
        "player_importance": importance_data,
    }
    clean_sheet_data = calculate_clean_sheet_odds(enriched_row, extra_context=narrative_context)
    scoring_odds_data = calculate_scoring_odds(enriched_row, extra_context=narrative_context)
    fpl_value_data = get_fpl_value_assessment(enriched_row, extra_context=narrative_context)

    # Use enriched_row for ALL generators so they see corrected injury history.
    # Story receives fixture + market context through the lightweight retrieval layer.
    story = generate_player_story(
        enriched_row,
        extra_context=narrative_context,
    )

    # Yara's response: compare model projection with consensus market line
    market_odds_for_yara = None
    if bookmaker_consensus_data:
        market_odds_for_yara = {
            "implied_probability": bookmaker_consensus_data["average_probability"],
            "decimal_odds": bookmaker_consensus_data["average_decimal"],
            "bookmaker": "Bookies Avg",
            "opponent": bookmaker_consensus_data.get("opponent"),
            "is_home": bookmaker_consensus_data.get("is_home"),
        }
    elif odds_client:
        market_odds_for_yara = odds_client.get_anytime_scorer_odds(team, player_name, league=player_league)

    yara_data = enhance_yara_response(
        base_response=generate_yara_response(enriched_row, market_odds_for_yara),
        player_row=enriched_row,
        next_fixture_data=next_fixture_data,
        market_consensus=bookmaker_consensus_data,
        scoring_odds_data=scoring_odds_data,
        clean_sheet_data=clean_sheet_data,
        injury_records=injury_records,
    )

    # Yara's Lab Notes: explainability
    lab_notes_data = generate_lab_notes(enriched_row, extra_context=narrative_context)

    fpl_insight_text = get_fpl_insight(enriched_row, extra_context=narrative_context)

    # FPL Points Projection
    fpl_points_data = calculate_expected_points(
        enriched_row=enriched_row,
        fpl_stats=fpl_stats,
        next_fixture=next_fixture_data,
        injury_prob=prob,
        scoring_odds_data=scoring_odds_data,
        clean_sheet_data=clean_sheet_data,
    )

    # Risk Comparison vs squad/position
    risk_comparison_data = calculate_risk_comparison(
        player_name=player_name,
        team=team,
        position=enriched_row.get("position", "Unknown"),
        prob=prob,
    )

    return PlayerRisk(
        name=player_name,
        team=team,
        position=enriched_row.get("position", "Unknown"),
        shirt_number=_safe_int(enriched_row.get("shirt_number"), 0) or None,
        age=int(enriched_row.get("age", 25)),
        risk_level=get_risk_level(prob, enriched_row),
        risk_probability=round(normalize_risk_score(prob, enriched_row.get("league")) / 100, 3),
        archetype=enriched_row.get("archetype", "Unknown"),
        archetype_description=ARCHETYPE_DESCRIPTIONS.get(
            enriched_row.get("archetype", ""), "Unknown injury profile"
        ),
        factors=RiskFactors(
            previous_injuries=int(prev_injuries),
            total_days_lost=int(total_days),
            days_since_last_injury=days_since,
            avg_days_per_injury=round(total_days / prev_injuries, 1) if prev_injuries > 0 else 0,
        ),
        model_predictions=ModelPredictions(
            ensemble=round(prob, 3),
            lgb=round(enriched_row.get("lgb_prob", prob), 3),
            xgb=round(enriched_row.get("xgb_prob", prob), 3),
            catboost=round(enriched_row.get("catboost_prob", prob), 3),
        ),
        recommendations=get_personalized_insights(enriched_row),
        story=story,
        implied_odds=calculate_implied_odds(prob),
        last_injury_date=str(enriched_row.get("last_injury_date")) if enriched_row.get("last_injury_date") else None,
        fpl_insight=fpl_insight_text,
        scoring_odds=ScoringOdds(**scoring_odds_data) if scoring_odds_data else None,
        fpl_value=FPLValue(**fpl_value_data) if fpl_value_data else None,
        clean_sheet_odds=CleanSheetOdds(**clean_sheet_data) if clean_sheet_data else None,
        next_fixture=NextFixture(**next_fixture_data) if next_fixture_data else None,
        bookmaker_consensus=BookmakerConsensus(**bookmaker_consensus_data) if bookmaker_consensus_data else None,
        yara_response=YaraResponse(**yara_data) if yara_data else None,
        lab_notes=LabNotes(
            summary=lab_notes_data["summary"],
            key_drivers=[LabDriver(**d) for d in lab_notes_data["key_drivers"]],
            technical=TechnicalDetails(**lab_notes_data["technical"]),
        ) if lab_notes_data else None,
        league=enriched_row.get("league", "Premier League"),
        risk_percentile=risk_percentile,
        player_image_url=get_player_image_url(player_name, team),
        team_badge_url=get_team_badge_url(team),
        is_currently_injured=fpl_stats.get("status", "a") in ("i", "u") if fpl_stats else False,
        injury_news=fpl_stats.get("news") or None if fpl_stats else None,
        chance_of_playing=fpl_stats.get("chance_of_playing") if fpl_stats else None,
        upcoming_fixtures=[UpcomingFixture(**uf) for uf in (
            get_upcoming_fixtures_la_liga(team, count=5) if player_league == "La Liga"
            else get_upcoming_fixtures(team, count=5)
        )] or None,
        injury_records=[InjuryRecord(**ir) for ir in injury_records],
        acwr=round(float(enriched_row.get("acwr", 0)), 2) if enriched_row.get("acwr") is not None else None,
        acute_load=round(float(enriched_row.get("acute_load", 0)), 1) if enriched_row.get("acute_load") is not None else None,
        chronic_load=round(float(enriched_row.get("chronic_load", 0)), 1) if enriched_row.get("chronic_load") is not None else None,
        spike_flag=bool(enriched_row.get("spike_flag", 0)),
        fpl_points_projection=FPLPointsProjection(**fpl_points_data) if fpl_points_data else None,
        risk_comparison=RiskComparison(**risk_comparison_data) if risk_comparison_data else None,
        player_importance=PlayerImportance(**importance_data) if importance_data else None,
    )


# ============================================================
# API Endpoints
# ============================================================

@app.get("/api/model-metrics")
def get_model_metrics():
    """Return model performance metrics from the saved model card."""
    import json as _json
    card_path = Path("models/model_card.json")
    if not card_path.exists():
        raise HTTPException(status_code=404, detail="Model card not found. Run retrain_model.py to generate it.")
    with open(card_path) as f:
        return _json.load(f)


@app.get("/api/health", response_model=HealthCheck)
async def health_check():
    """Check API health and model status."""
    return HealthCheck(
        status="ok" if inference_df is not None else "no_models",
        models_loaded=inference_df is not None,
        player_count=len(inference_df) if inference_df is not None else 0,
    )


@app.post("/api/admin/refresh-predictions")
def trigger_prediction_refresh(
    mode: str = "api",
    x_refresh_token: Optional[str] = Header(default=None, alias="X-Refresh-Token"),
):
    """Trigger background prediction refresh and hot-reload artifacts."""
    _require_refresh_token(x_refresh_token)
    mode_key = (mode or "api").strip().lower()
    if mode_key not in {"api", "fbref"}:
        raise HTTPException(status_code=400, detail="mode must be 'api' or 'fbref'")

    with refresh_state_lock:
        if refresh_state["running"]:
            return {
                "status": "running",
                "message": "Refresh already in progress",
                "state": dict(refresh_state),
            }
        refresh_state["running"] = True
        refresh_state["last_status"] = "running"
        refresh_state["last_started_at"] = _utc_now_iso()
        refresh_state["last_mode"] = mode_key
        refresh_state["last_error"] = None
        refresh_state["last_exit_code"] = None
        refresh_state["last_log_tail"] = ""

    worker = threading.Thread(
        target=_run_refresh_job,
        args=(mode_key,),
        daemon=True,
        name="refresh-predictions-worker",
    )
    worker.start()

    return {
        "status": "started",
        "message": "Prediction refresh started",
        "mode": mode_key,
        "started_at": refresh_state["last_started_at"],
    }


@app.get("/api/admin/refresh-status")
def get_prediction_refresh_status(
    x_refresh_token: Optional[str] = Header(default=None, alias="X-Refresh-Token"),
):
    """Check status of the last/active prediction refresh run."""
    _require_refresh_token(x_refresh_token)
    with refresh_state_lock:
        return dict(refresh_state)


@app.get("/api/odds/status")
def odds_status(team: str = "Arsenal", player: str = "Bukayo Saka"):
    """Inspect odds provider mode and direct market availability."""
    if odds_client is None:
        raise HTTPException(status_code=503, detail="Odds client not initialized")

    scorer_snapshot = odds_client.get_anytime_scorer_market_snapshot(team, player)
    team_moneyline = odds_client.get_team_moneyline_1x2(team)

    return {
        "provider_mode": getattr(odds_client, "odds_provider", "auto"),
        "has_odds_api_key": bool(getattr(odds_client, "api_key", "")),
        "has_api_football_key": bool(getattr(odds_client, "api_football_key", "")),
        "team": team,
        "player": player,
        "scorer_lines_count": len((scorer_snapshot or {}).get("lines", [])),
        "scorer_lines_books": [l.get("bookmaker") for l in (scorer_snapshot or {}).get("lines", [])],
        "moneyline_books_count": len((team_moneyline or {}).get("books", [])),
        "moneyline_books": [b.get("bookmaker") for b in (team_moneyline or {}).get("books", [])],
    }


@app.get("/api/players", response_model=List[PlayerSummary])
def list_players(team: Optional[str] = None, risk_level: Optional[str] = None):
    """List all players with summary info."""
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    df = inference_df.copy()

    if team:
        df = df[df["team"].str.lower() == team.lower()]
    if risk_level:
        df["risk_level"] = df.apply(lambda r: get_risk_level(r.get("ensemble_prob", 0.5), r), axis=1)
        df = df[df["risk_level"].str.lower() == risk_level.lower()]

    df = df.sort_values("ensemble_prob", ascending=False)

    players = []
    for _, row in df.iterrows():
        prob = row.get("ensemble_prob", 0.5)
        name = row.get("name", "Unknown")
        row_league = row.get("league", "Premier League")
        is_la_liga = row_league == "La Liga"
        fpl = None if is_la_liga else get_fpl_stats_for_player(name, team_hint=row.get("team", ""))
        minutes = fpl.get("minutes", 0) if fpl else 0
        fpl_status = fpl.get("status", "a") if fpl else "a"
        display_prob = round(normalize_risk_score(prob, row_league) / 100, 3)
        players.append(PlayerSummary(
            name=name,
            team=row.get("team", "Unknown"),
            position=row.get("position", "Unknown"),
            shirt_number=(
                _safe_int(row.get("shirt_number"), 0) or None
                if is_la_liga
                else _resolve_shirt_number(name, fpl, row.get("team", "Unknown"))
            ),
            risk_level=get_risk_level(prob, row),
            risk_probability=display_prob,
            archetype=row.get("archetype", "Unknown"),
            minutes_played=minutes,
            is_starter=minutes >= 900,
            player_image_url=get_player_image_url(name, row.get("team")),
            days_since_last_injury=_safe_int(row.get("days_since_last_injury", 365), 365),
            is_currently_injured=fpl_status in ("i", "u"),
            injury_news=fpl.get("news") or None if fpl else None,
            chance_of_playing=fpl.get("chance_of_playing") if fpl else None,
        ))

    return players


def _prediction_risk_label(score: float) -> str:
    if score >= 0.65:
        return "HIGH"
    elif score >= 0.35:
        return "ELEVATED"
    return "LOW"


def _build_player_reasons(row: dict, matchup: Optional[dict], fpl: Optional[dict]) -> List[str]:
    """Build 2-3 short, specific reasons from actual model features."""
    reasons: List[str] = []
    name = row.get("name", "Player")
    first = name.split()[-1] if name else "Player"

    # Minutes / workload
    minutes_list = (matchup or {}).get("minutes_last_5", [])
    full_90s = sum(1 for m in minutes_list if m >= 85)
    matches_30 = _safe_int(row.get("matches_last_30", 0), 0)
    if full_90s >= 4 and len(minutes_list) >= 5:
        reasons.append(f"played 90 mins in {full_90s} of last {len(minutes_list)} games")
    elif matches_30 >= 6:
        reasons.append(f"{matches_30} matches in 30 days")
    elif matches_30 >= 4:
        reasons.append(f"{matches_30} games in a month, limited rest")

    # Injury recency
    days_since = _safe_int(row.get("days_since_last_injury", 365), 365)
    if days_since < 30:
        reasons.append(f"only {days_since} days since last injury")
    elif days_since < 60:
        reasons.append(f"still in recovery window at {days_since} days back")

    # Injury history
    prev = _safe_int(row.get("previous_injuries", 0), 0)
    total_lost = _safe_int(row.get("total_days_lost", 0), 0)
    if prev >= 8:
        reasons.append(f"{prev} career injuries, {total_lost} total days lost")
    elif prev >= 4:
        reasons.append(f"{prev} previous injuries on record")

    # ACWR / workload spike
    acwr = _safe_float(row.get("acwr", 0.0), 0.0)
    if acwr >= 1.5:
        reasons.append(f"ACWR at {acwr:.2f}, workload spike flagged")
    elif acwr >= 1.2:
        reasons.append(f"elevated workload ratio at {acwr:.2f}")

    # Age factor
    age = _safe_int(row.get("age", 25), 25)
    if age >= 32 and prev >= 3:
        reasons.append(f"age {age} with injury history raises baseline risk")

    # FPL injury news
    if fpl and fpl.get("news"):
        news = fpl["news"][:50]
        reasons.append(news.rstrip(". ").lower())

    # Cap at 3, ensure at least 2
    if len(reasons) < 2:
        prob = _safe_float(row.get("ensemble_prob", 0.5), 0.5)
        if prob >= 0.5:
            reasons.append("model flags elevated baseline from combined features")
        else:
            reasons.append("clean recent profile supports availability")

    return reasons[:3]


def _build_odds_movement(row: dict, fpl: Optional[dict]) -> str:
    """Describe odds movement based on available data."""
    prob = _safe_float(row.get("ensemble_prob", 0.5), 0.5)
    implied = calculate_implied_odds(prob)
    decimal_odds = implied.decimal if hasattr(implied, "decimal") else 1 / max(prob, 0.01)

    # No historical odds tracking yet, so describe current position
    if prob >= 0.65:
        return f"short at {decimal_odds:.1f}, model sees high risk"
    elif prob >= 0.45:
        return f"sitting at {decimal_odds:.1f}, elevated concern"
    elif prob >= 0.25:
        return f"drifting at {decimal_odds:.1f}, moderate risk"
    return f"long at {decimal_odds:.1f}, low concern"


@app.get("/api/predictions")
def get_predictions(team: str):
    """Get injury risk predictions for an entire squad. Used by the auto-poster."""
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Match team name flexibly
    team_lower = team.lower()
    df = inference_df[inference_df["team"].str.lower() == team_lower]
    if df.empty:
        # Try partial match
        df = inference_df[inference_df["team"].str.lower().str.contains(team_lower)]
    if df.empty:
        raise HTTPException(status_code=404, detail={"error": "Team not found"})

    team_name = df.iloc[0]["team"]

    # Get current gameweek
    try:
        client = FPLClient()
        gw_data = client.get_current_gameweek()
        gameweek = gw_data.get("id", 0) if gw_data else 0
    except Exception:
        gameweek = 0

    # Get next fixture for the team
    next_fixture_data = get_next_fixture_for_team(team_name, 0.5)
    next_opponent = (next_fixture_data or {}).get("opponent", "Unknown")

    players_out = []
    for _, row in df.sort_values("ensemble_prob", ascending=False).iterrows():
        row_dict = row.to_dict()
        name = row_dict.get("name", "Unknown")
        prob = _safe_float(row_dict.get("ensemble_prob", 0.5), 0.5)

        # Get FPL stats
        fpl = get_fpl_stats_for_player(name, team_hint=team_name)

        # Get matchup context (includes minutes_last_5 after the fix)
        matchup = get_player_matchup_context(
            player_name=name,
            team_hint=team_name,
            next_fixture_data=next_fixture_data,
        )

        minutes_last_5 = (matchup or {}).get("minutes_last_5", [])
        games_last_5 = len([m for m in minutes_last_5 if m > 0])

        # H2H injury rate vs opponent (from fixture history)
        h2h_injury_rate = 0.0
        vs_opp = (matchup or {}).get("vs_opponent") or {}
        vs_samples = _safe_int(vs_opp.get("samples", 0), 0)
        if vs_samples > 0:
            # Approximate: if player missed any of these H2H games, injury rate is higher
            h2h_injury_rate = round(max(0, 1 - (vs_samples / max(vs_samples + 1, 1))), 2)

        reasons = _build_player_reasons(row_dict, matchup, fpl)
        odds_movement = _build_odds_movement(row_dict, fpl)

        players_out.append({
            "name": name,
            "team": team_name,
            "risk_score": round(prob, 3),
            "risk_label": _prediction_risk_label(prob),
            "reasons": reasons,
            "minutes_last_5": minutes_last_5,
            "games_last_5": games_last_5,
            "odds_movement": odds_movement,
            "h2h_next_opponent": next_opponent,
            "h2h_injury_rate_vs_opponent": h2h_injury_rate,
        })

    return {
        "team": team_name,
        "gameweek": gameweek,
        "players": players_out,
    }


@app.get("/api/players/{player_name}/risk", response_model=PlayerRisk)
def get_player_risk(player_name: str):
    """Get detailed risk prediction for a specific player."""
    started = time.perf_counter()
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    cache_key = player_name.lower()
    now = datetime.utcnow()
    cached = _player_risk_cache.get(cache_key)
    if cached and cached["expires"] > now:
        logger.info(
            "Player risk for %s served from cache in %.2fs",
            player_name,
            time.perf_counter() - started,
        )
        return cached["data"]

    matches = inference_df[
        inference_df["name"].str.lower() == player_name.lower()
    ]
    if matches.empty:
        matches = inference_df[
            inference_df["name"].str.lower().str.contains(player_name.lower())
        ]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

    row = matches.iloc[0].to_dict()
    result = player_row_to_risk(row)
    _player_risk_cache[cache_key] = {"data": result, "expires": now + _PLAYER_RISK_TTL}
    logger.info(
        "Player risk for %s (%s) built in %.2fs",
        player_name,
        row.get("league", "Unknown"),
        time.perf_counter() - started,
    )
    return result


@app.get("/api/teams", response_model=List[str])
def list_teams(league: Optional[str] = None):
    """List all available teams, optionally filtered by league."""
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if league and "league" in inference_df.columns:
        league_df = inference_df[inference_df["league"].str.lower() == league.lower()]
        our_teams = league_df["team"].unique().tolist()
    else:
        our_teams = inference_df["team"].unique().tolist()

    # For non-EPL leagues, skip PL standings validation — return all teams directly
    is_epl = not league or league.lower() in ("premier league", "epl")
    if not is_epl:
        return sorted(our_teams)

    try:
        now = datetime.utcnow()
        if _pl_standings_cache["data"] is None or (
            _pl_standings_cache["expires"] and now > _pl_standings_cache["expires"]
        ):
            client = FootballDataClient()
            standings = client.get_standings()
            _pl_standings_cache["data"] = standings
            _pl_standings_cache["expires"] = now + _PL_STANDINGS_TTL
        else:
            standings = _pl_standings_cache["data"]
        current_pl_teams = {t["name"].lower() for t in standings}
        current_pl_short = {t["short_name"].lower() for t in standings}

        from src.data_loaders.football_data_api import TEAM_ALIASES

        filtered = []
        for team in our_teams:
            team_lower = team.lower()
            search_term = TEAM_ALIASES.get(team_lower, team_lower)
            in_pl = any(
                search_term in pl_team or pl_team in search_term
                for pl_team in current_pl_teams
            ) or any(
                search_term in short or short in search_term
                for short in current_pl_short
            )
            if in_pl:
                filtered.append(team)

        return sorted(filtered)
    except Exception as e:
        logger.warning(f"Failed to filter teams by current PL: {e}")
        return sorted(our_teams)


@app.get("/api/teams/badges")
def get_team_badges():
    """Get badge URLs for all teams."""
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    teams = inference_df["team"].unique().tolist()
    badges = {}
    for team in teams:
        url = get_team_badge_url(team)
        if url:
            badges[team] = url
    return badges


@app.get("/api/teams/{team_name}/overview", response_model=TeamOverview)
def get_team_overview(team_name: str):
    """Get risk overview for an entire team."""
    started = time.perf_counter()
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    team_df = inference_df[
        inference_df["team"].str.lower() == team_name.lower()
    ]
    if team_df.empty:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")

    team_df = team_df.copy()
    team_df["risk_level"] = team_df.apply(lambda r: get_risk_level(r.get("ensemble_prob", 0.5), r), axis=1)
    actual_team = team_df.iloc[0]["team"]
    team_league = team_df.iloc[0].get("league", "Premier League")
    is_la_liga = team_league == "La Liga"

    high_risk = len(team_df[team_df["risk_level"] == "High"])
    medium_risk = len(team_df[team_df["risk_level"] == "Medium"])
    low_risk = len(team_df[team_df["risk_level"] == "Low"])

    team_df = team_df.sort_values("ensemble_prob", ascending=False)

    players = []
    for _, row in team_df.iterrows():
        prob = row.get("ensemble_prob", 0.5)
        name = row.get("name", "Unknown")
        fpl = None if is_la_liga else get_fpl_stats_for_player(name)
        minutes = fpl.get("minutes", 0) if fpl else 0
        fpl_status = fpl.get("status", "a") if fpl else "a"
        row_league = row.get("league")
        display_prob = round(normalize_risk_score(prob, row_league) / 100, 3)
        players.append(PlayerSummary(
            name=name,
            team=row.get("team", "Unknown"),
            position=row.get("position", "Unknown"),
            shirt_number=(
                _safe_int(row.get("shirt_number"), 0) or None
                if is_la_liga
                else _resolve_shirt_number(name, fpl, row.get("team", "Unknown"))
            ),
            risk_level=get_risk_level(prob, row),
            risk_probability=display_prob,
            archetype=row.get("archetype", "Unknown"),
            minutes_played=minutes,
            is_starter=minutes >= 900,
            player_image_url=get_player_image_url(name, row.get("team")),
            days_since_last_injury=_safe_int(row.get("days_since_last_injury", 365), 365),
            is_currently_injured=fpl_status in ("i", "u"),
            injury_news=fpl.get("news") or None if fpl else None,
            chance_of_playing=fpl.get("chance_of_playing") if fpl else None,
        ))

    # Get next fixture for the team
    if is_la_liga:
        next_fixture_data = _get_la_liga_team_context_cached(actual_team, count=1).get("next_fixture_data")
    else:
        next_fixture_data = get_next_fixture_for_team(actual_team, team_df["ensemble_prob"].mean())

    response = TeamOverview(
        team=actual_team,
        total_players=len(team_df),
        high_risk_count=high_risk,
        medium_risk_count=medium_risk,
        low_risk_count=low_risk,
        avg_risk=round(normalize_risk_score(team_df["ensemble_prob"].mean(), team_league) / 100, 3),
        players=players,
        team_badge_url=get_team_badge_url(actual_team),
        next_fixture=next_fixture_data,
    )
    logger.info(
        "Team overview for %s (%s) built in %.2fs",
        actual_team,
        team_league,
        time.perf_counter() - started,
    )
    return response


@app.get("/api/archetypes")
def list_archetypes():
    """List all player archetypes with descriptions."""
    return ARCHETYPE_DESCRIPTIONS


@app.get("/api/fpl/insights", response_model=FPLInsights)
def get_fpl_insights_endpoint():
    """Get FPL insights including standings, fixtures, and double gameweeks."""
    try:
        now = datetime.utcnow()
        if _fpl_insights_cache["data"] is None or (
            _fpl_insights_cache["expires"] and now > _fpl_insights_cache["expires"]
        ):
            client = FPLClient()
            standings = client.get_standings()
            upcoming = client.get_upcoming_fixtures_summary(3)
            current_gw = client.get_current_gameweek()
            _fpl_insights_cache["data"] = {
                "standings": standings,
                "upcoming": upcoming,
                "current_gw": current_gw,
            }
            _fpl_insights_cache["expires"] = now + _FPL_INSIGHTS_TTL
        else:
            standings = _fpl_insights_cache["data"]["standings"]
            upcoming = _fpl_insights_cache["data"]["upcoming"]
            current_gw = _fpl_insights_cache["data"]["current_gw"]

        return FPLInsights(
            current_gameweek=current_gw.get("id") if current_gw else None,
            standings=[LeagueStanding(**s) for s in standings],
            upcoming_gameweeks=[GameweekSummary(**gw) for gw in upcoming],
            has_double_gameweek=any(gw.get("double_gameweek_teams") for gw in upcoming),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FPL API error: {str(e)}")


@app.get("/api/fpl/standings", response_model=List[LeagueStanding])
def get_league_standings():
    """Get current Premier League standings."""
    try:
        client = FPLClient()
        standings = client.get_standings()
        return [LeagueStanding(**s) for s in standings]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FPL API error: {str(e)}")


@app.get("/api/la-liga/standings")
def get_la_liga_standings():
    """Get current La Liga standings from football-data.org."""
    started = time.perf_counter()
    try:
        rows = _get_la_liga_standings_cached()
        logger.info(
            "La Liga standings served in %.2fs",
            time.perf_counter() - started,
        )
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"La Liga standings unavailable: {str(e)}")


@app.get("/api/player-photo/tm")
def proxy_tm_player_photo(name: str):
    """Proxy a Transfermarkt player photo by player name.

    Looks up the pre-built photo map, then fetches the image server-side
    (required because img.a.transfermarkt.technology blocks browser requests).
    Returns the image as image/jpeg.
    """
    from fastapi.responses import Response
    import unicodedata

    name_lower = name.lower().strip()
    stripped = "".join(
        c for c in unicodedata.normalize("NFD", name_lower)
        if unicodedata.category(c) != "Mn"
    )

    photo_url = _tm_photo_map.get(name_lower) or _tm_photo_map.get(stripped)
    if not photo_url:
        raise HTTPException(status_code=404, detail="No photo found for this player")

    if photo_url in _tm_photo_bytes_cache:
        return Response(content=_tm_photo_bytes_cache[photo_url], media_type="image/jpeg")

    try:
        from src.data_loaders.transfermarkt_scraper import TransfermarktScraper
        scraper = TransfermarktScraper(cache_hours=168)
        resp = scraper.session.get(photo_url, timeout=8)
        if resp.status_code != 200:
            raise HTTPException(status_code=404, detail="Photo unavailable")
        _tm_photo_bytes_cache[photo_url] = resp.content
        return Response(content=resp.content, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Photo fetch failed: {str(e)}")


@app.get("/api/standings/summary")
def get_standings_summary_endpoint(team: Optional[str] = None):
    """Get Premier League standings summary."""
    try:
        summary = get_standings_summary(team)
        if not summary:
            raise HTTPException(status_code=503, detail="Standings unavailable")

        return StandingsSummary(
            leader=TeamStanding(**summary["leader"]),
            second=TeamStanding(**summary["second"]),
            gap_to_second=summary["gap_to_second"],
            safety_points=summary.get("safety_points", 0),
            selected_team=TeamStanding(**summary["selected_team"]) if summary.get("selected_team") else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API_STANDINGS_ERROR] {repr(e)}")
        raise HTTPException(status_code=500, detail=f"Standings error: {str(e)}")


@app.get("/api/fpl/double-gameweeks")
def get_double_gameweeks():
    """Get teams with upcoming double gameweeks."""
    try:
        client = FPLClient()
        teams = {t["id"]: t["name"] for t in client.get_teams()}
        double_gws = client.get_double_gameweek_teams()

        result = {}
        for gw, team_ids in double_gws.items():
            result[str(gw)] = [teams.get(tid, "Unknown") for tid in team_ids]

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FPL API error: {str(e)}")


@app.get("/api/fpl/squad/{team_id}", response_model=FPLSquadSync)
def get_fpl_squad(team_id: int):
    """Sync an FPL manager's squad and return Yara risk data for each player."""
    if inference_df is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    client = FPLClient()

    # 1. Get current gameweek
    gw_data = client.get_current_gameweek()
    if not gw_data:
        raise HTTPException(status_code=503, detail="Could not determine current gameweek")
    current_gw = gw_data["id"]
    gw_finished = gw_data.get("finished", False)

    # 2. Fetch manager entry
    try:
        entry_data = client.get_entry(team_id)
    except Exception:
        raise HTTPException(status_code=503, detail="FPL servers are currently unavailable. Try again shortly.")
    if not entry_data or "id" not in entry_data:
        raise HTTPException(status_code=404, detail="FPL team not found. Check your Team ID.")

    manager_name = f"{entry_data.get('player_first_name', '')} {entry_data.get('player_last_name', '')}".strip()
    team_name = entry_data.get("name", "Unknown")

    # 3. Fetch picks for current gameweek (fall back to previous if empty)
    picks_data = client.get_picks(team_id, current_gw)
    if not picks_data or not picks_data.get("picks"):
        if current_gw > 1:
            picks_data = client.get_picks(team_id, current_gw - 1)
            current_gw -= 1
    if not picks_data or not picks_data.get("picks"):
        raise HTTPException(status_code=404, detail="No picks found for this gameweek.")

    picks = picks_data["picks"]
    entry_history = picks_data.get("entry_history", {})

    # 3a. Apply auto-subs — swap positions so lineup matches what the user saw
    auto_subs = picks_data.get("automatic_subs", [])
    if auto_subs:
        pick_by_element = {p["element"]: p for p in picks}
        for sub in auto_subs:
            el_in = sub.get("element_in")    # bench player coming in
            el_out = sub.get("element_out")  # starter going out
            p_in = pick_by_element.get(el_in)
            p_out = pick_by_element.get(el_out)
            if p_in and p_out:
                p_in["position"], p_out["position"] = p_out["position"], p_in["position"]
                p_in["multiplier"], p_out["multiplier"] = p_out["multiplier"], p_in["multiplier"]

    # 3b. If GW is finished, apply pending transfers for the next GW
    if gw_finished:
        try:
            transfers = client.get_transfers(team_id)
            next_gw = current_gw + 1
            pending = [t for t in transfers if t.get("event", 0) == next_gw]
            if pending:
                pick_elements = {p["element"] for p in picks}
                for t in pending:
                    out_id = t["element_out"]
                    in_id = t["element_in"]
                    if out_id in pick_elements:
                        # Swap: find the pick with the outgoing player, replace element
                        for p in picks:
                            if p["element"] == out_id:
                                p["element"] = in_id
                                break
                        pick_elements.discard(out_id)
                        pick_elements.add(in_id)
        except Exception:
            pass  # transfers are best-effort

    # 4. Resolve each pick to a player with risk data
    players = []
    unmatched = []

    for pick in picks:
        element_id = pick["element"]
        row, fpl_data = _resolve_player_from_element(element_id)

        if row is not None:
            prob = row.get("ensemble_prob", 0.5)
            name = row.get("name", "Unknown")
            fpl = get_fpl_stats_for_player(name)
            minutes = fpl.get("minutes", 0) if fpl else 0
            fpl_status = fpl.get("status", "a") if fpl else "a"
            # Use FPL element type position (GK/DEF/MID/FWD) for clean grouping
            fpl_pos = fpl_data.get("position", "") if fpl_data else ""
            _POS_MAP = {"GK": "Goalkeeper", "DEF": "Defender", "MID": "Midfielder", "FWD": "Forward"}
            display_pos = _POS_MAP.get(fpl_pos, row.get("position", "Unknown"))

            players.append(FPLSquadPlayer(
                name=name,
                team=row.get("team", "Unknown"),
                position=display_pos,
                shirt_number=_resolve_shirt_number(name, fpl, row.get("team", "Unknown")),
                risk_level=get_risk_level(prob, row),
                risk_probability=round(normalize_risk_score(prob, row.get("league")) / 100, 3),
                archetype=row.get("archetype", "Unknown"),
                minutes_played=minutes,
                is_starter=minutes >= 900,
                player_image_url=get_player_image_url(name, row.get("team")),
                days_since_last_injury=_safe_int(row.get("days_since_last_injury", 365), 365),
                is_currently_injured=fpl_status in ("i", "u"),
                injury_news=fpl.get("news") or None if fpl else None,
                chance_of_playing=fpl.get("chance_of_playing") if fpl else None,
                is_captain=pick.get("is_captain", False),
                is_vice_captain=pick.get("is_vice_captain", False),
                squad_position=pick.get("position", 0),
                multiplier=pick.get("multiplier", 1),
            ))
        elif fpl_data:
            unmatched.append(fpl_data.get("name", f"element_{element_id}"))
        else:
            unmatched.append(f"element_{element_id}")

    # 5. Sort: starters by risk desc, then bench by risk desc
    starters = sorted(
        [p for p in players if p.squad_position <= 11],
        key=lambda p: p.risk_probability, reverse=True
    )
    bench = sorted(
        [p for p in players if p.squad_position > 11],
        key=lambda p: p.risk_probability, reverse=True
    )
    sorted_players = starters + bench

    # 6. Risk counts
    probs = [p.risk_probability for p in players]
    high = sum(1 for p in probs if p >= 0.6)
    medium = sum(1 for p in probs if 0.4 <= p < 0.6)
    low = sum(1 for p in probs if p < 0.4)
    avg = round(sum(probs) / len(probs), 3) if probs else 0.0

    return FPLSquadSync(
        entry=FPLSquadEntry(
            team_name=team_name,
            manager_name=manager_name,
            total_points=entry_data.get("summary_overall_points", 0),
            gameweek=current_gw,
            gameweek_points=entry_history.get("points", 0),
        ),
        players=sorted_players,
        unmatched=unmatched,
        high_risk_count=high,
        medium_risk_count=medium,
        low_risk_count=low,
        avg_risk=avg,
        is_gw_finished=gw_finished,
    )


@app.get("/api/players/{player_name}/what-if", response_model=WhatIfProjection)
def what_if_projection(player_name: str, rest_next: bool = False, play_all: bool = False):
    """Project how resting or playing all matches affects a player's injury risk."""
    if inference_df is None or artifacts is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    ensemble = artifacts.get("ensemble")
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Ensemble model not loaded")

    matches = inference_df[inference_df["name"].str.lower() == player_name.lower()]
    if matches.empty:
        matches = inference_df[inference_df["name"].str.lower().str.contains(player_name.lower())]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

    try:
        import pandas as pd

        row = matches.iloc[0]
        current_prob = float(row.get("ensemble_prob", 0.5))
        acwr_current = float(row.get("acwr", 1.0))

        # Clone the row and modify workload features
        modified = row.copy()
        feature_cols = ensemble.feature_names_

        if rest_next:
            acute = float(modified.get("acute_load", 1))
            modified["acute_load"] = max(0.5, acute - 1)
            modified["rest_days_before_injury"] = float(modified.get("rest_days_before_injury", 3)) + 5
            modified["avg_rest_last_5"] = float(modified.get("avg_rest_last_5", 4)) + 2
            modified["matches_last_7"] = max(0, float(modified.get("matches_last_7", 1)) - 1)
            modified["matches_last_14"] = max(1, float(modified.get("matches_last_14", 2)) - 1)
            if "strain" in modified.index:
                modified["strain"] = max(0.1, float(modified.get("strain", 0)) * 0.5)
            if "monotony" in modified.index:
                modified["monotony"] = max(0.2, float(modified.get("monotony", 0)) * 0.6)
            if "spike_flag" in modified.index:
                modified["spike_flag"] = 0
            scenario = "Rest next match"
        elif play_all:
            acute = float(modified.get("acute_load", 1))
            modified["acute_load"] = acute + 2
            modified["rest_days_before_injury"] = max(2, float(modified.get("rest_days_before_injury", 3)) - 2)
            modified["avg_rest_last_5"] = max(2, float(modified.get("avg_rest_last_5", 4)) - 1.5)
            modified["matches_last_7"] = float(modified.get("matches_last_7", 1)) + 2
            modified["matches_last_14"] = float(modified.get("matches_last_14", 2)) + 3
            if "strain" in modified.index:
                modified["strain"] = float(modified.get("strain", 0)) * 2.0
            if "monotony" in modified.index:
                modified["monotony"] = min(5, float(modified.get("monotony", 0)) * 1.8)
            scenario = "Play all upcoming matches"
        else:
            raise HTTPException(status_code=400, detail="Specify rest_next=true or play_all=true")

        # Recalculate ACWR
        chronic = float(modified.get("chronic_load", 1))
        if chronic > 0:
            modified["acwr"] = float(modified["acute_load"]) / chronic
        acwr_projected = float(modified.get("acwr", acwr_current))

        # Recalculate derived features if present
        if "fatigue_index" in modified.index:
            modified["fatigue_index"] = float(modified["acute_load"]) - chronic
        if "spike_flag" in modified.index:
            modified["spike_flag"] = 1 if acwr_projected > 1.8 else 0

        # Patch sklearn compat: older pickled LogisticRegression may have removed attrs
        meta = getattr(ensemble, "meta_model_", None)
        if meta and not hasattr(meta, "multi_class"):
            meta.multi_class = "auto"

        # Predict with modified features
        modified_df = pd.DataFrame([modified])
        X = modified_df[feature_cols].copy()
        projected_prob = float(ensemble.predict_proba(X)[0, 1])

        return WhatIfProjection(
            player_name=str(row.get("name", player_name)),
            current_risk=round(current_prob, 3),
            projected_risk=round(projected_prob, 3),
            scenario=scenario,
            delta=round(projected_prob - current_prob, 3),
            acwr_current=round(acwr_current, 2),
            acwr_projected=round(acwr_projected, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"What-if projection failed for {player_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Projection failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
