"""soccerdata-backed player season stats loader.

Replaces our custom FBref scraper (which always 403s) with the community
``soccerdata`` library. soccerdata uses seleniumbase + undetected chromedriver
under the hood, so FBref's bot filter doesn't trip it.

This module exists because:
  - FBref data (Min, MP, Starts, Gls, Ast per player per season) is the cleanest
    free source for per-player Bundesliga minutes — no other free API exposes it.
  - Plain requests against FBref return 403. soccerdata works around that.
  - bundesliga.com's bapi doesn't expose a minutes ranking; Sofascore/FotMob also
    block our requests. soccerdata was the only path that actually returned data.

Caveats:
  - First run downloads ChromeDriver (~10MB) and spins up a headless browser.
    Subsequent fetches hit soccerdata's on-disk cache (``~/soccerdata/data/FBref``)
    so they're fast. Set ``SOCCERDATA_DIR`` to relocate the cache.
  - Render's stock Python runtime has no Chrome. Cron refreshes will need a
    Chromium-bearing Docker image or to run this step locally and ship the
    artifact pickle.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


# Map our internal league names to soccerdata's FBref league codes.
# soccerdata uses two-digit season codes like "2526" = 2025-26.
_LEAGUE_CODES = {
    "Bundesliga": "GER-Bundesliga",
    "Premier League": "ENG-Premier League",
    "La Liga": "ESP-La Liga",
    "Serie A": "ITA-Serie A",
    "Ligue 1": "FRA-Ligue 1",
}


def _season_code(season_start_year: int) -> str:
    """Convert 2025 → "2526" (soccerdata's compact season code)."""
    end = (season_start_year + 1) % 100
    return f"{season_start_year % 100:02d}{end:02d}"


# FBref ships position as compact codes (GK/DF/MF/FW), sometimes combined like
# "MF,FW" with the primary role first. Translate the primary code to the full
# word the narrative layer's ``_position_group`` recognises — its token list
# matches "goalkeeper"/"defender"/"midfielder"/"forward" but not "DF"/"MF"/"FW".
_FBREF_POS_TO_WORD = {
    "GK": "Goalkeeper",
    "DF": "Defender",
    "MF": "Midfielder",
    "FW": "Forward",
}


def _fbref_position_word(pos) -> str:
    primary = str(pos or "").split(",")[0].strip().upper()
    return _FBREF_POS_TO_WORD.get(primary, "Unknown")


def _fbref_age_int(age) -> int:
    """FBref age is usually a plain int (35); some stat tables ship "35-103"
    (years-days). Take the leading integer; default 25 when unparseable."""
    text = str(age or "").strip()
    if not text:
        return 25
    head = text.split("-")[0].strip()
    try:
        return int(float(head))
    except (TypeError, ValueError):
        return 25


def load_player_season_stats(
    league_name: str,
    season_start_year: int,
    stat_type: str = "standard",
) -> Optional[pd.DataFrame]:
    """Fetch per-player season stats for ``league_name`` via soccerdata/FBref.

    Returns a flat DataFrame with columns including ``player``, ``team``,
    ``minutes``, ``appearances``, ``starts``, ``goals``, ``assists``. Returns
    None on failure (caller falls back to whatever it had).

    league_name uses our internal label ("Bundesliga", "Premier League", ...).
    season_start_year is the year the season started (2025 for 2025-26).
    """
    code = _LEAGUE_CODES.get(league_name)
    if not code:
        logger.warning("soccerdata: no league code mapping for %s", league_name)
        return None

    season = _season_code(season_start_year)
    logger.info("soccerdata: fetching FBref %s player season stats (%s)", code, season)

    try:
        import soccerdata as sd  # local import — heavy dep with selenium
    except ImportError:
        logger.warning("soccerdata not installed; pip install soccerdata")
        return None

    try:
        fb = sd.FBref(leagues=code, seasons=season)
        df = fb.read_player_season_stats(stat_type=stat_type)
    except Exception as e:
        logger.warning("soccerdata fetch failed for %s %s: %s", league_name, season, e)
        return None

    if df is None or df.empty:
        return None

    # Flatten the MultiIndex columns FBref ships ("Playing Time" → "Min", etc.)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["__".join(c).strip("_") for c in df.columns.values]
    df = df.reset_index()

    # Normalise the column subset the rest of the pipeline expects. The exact
    # column names FBref ships under "standard" stat_type are stable enough to
    # hardcode; if soccerdata's schema shifts we'll surface it via a KeyError.
    out = pd.DataFrame({
        "name": df["player"].astype(str),
        "team": df["team"].astype(str),
        # position/age make FBref usable as a roster source (needed for Ligue 1,
        # where football-data's /teams squad endpoint 403s so there's no other
        # source for player position/age).
        "position": df.get("pos").map(_fbref_position_word) if "pos" in df.columns else "Unknown",
        "age": df.get("age").map(_fbref_age_int) if "age" in df.columns else 25,
        "appearances": pd.to_numeric(df.get("Playing Time__MP"), errors="coerce").fillna(0).astype(int),
        "starts": pd.to_numeric(df.get("Playing Time__Starts"), errors="coerce").fillna(0).astype(int),
        "season_minutes": pd.to_numeric(df.get("Playing Time__Min"), errors="coerce").fillna(0).astype(int),
        "goals": pd.to_numeric(df.get("Performance__Gls"), errors="coerce").fillna(0).astype(int),
        "assists": pd.to_numeric(df.get("Performance__Ast"), errors="coerce").fillna(0).astype(int),
    })
    out = out[out["season_minutes"] > 0].reset_index(drop=True)
    logger.info("soccerdata: %s %s → %d players with minutes", league_name, season, len(out))
    return out


__all__ = ["load_player_season_stats"]
