#!/usr/bin/env python3
"""
Build a player name → Transfermarkt photo URL map for La Liga and EPL players.

Searches Transfermarkt for each active squad player, extracts their profile
photo URL from the cached/fetched HTML, and saves to:
  models/player_photo_map.json  — { "player name lowercase": "https://..." }

This map is loaded at API startup for get_player_image_url().

Usage:
    python scripts/build_player_photo_map.py
    python scripts/build_player_photo_map.py --league laliga
    python scripts/build_player_photo_map.py --league epl
"""

import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
_env = ROOT / ".env"
if _env.exists():
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from bs4 import BeautifulSoup
from src.data_loaders.transfermarkt_scraper import TransfermarktScraper, BASE_URL
from src.utils.logger import get_logger

logger = get_logger(__name__)

OUT_PATH = ROOT / "models" / "player_photo_map.json"


def get_photo_url_from_profile(scraper, player_id: str, slug: str) -> str | None:
    """Fetch player profile and extract photo URL."""
    url = f"{BASE_URL}/{slug}/profil/spieler/{player_id}"
    html = scraper._fetch(url)
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    img = soup.select_one("img.data-header__profile-image")
    if img:
        return img.get("src") or img.get("data-src")
    return None


def get_photo_url_for_name(scraper, name: str) -> tuple[str | None, str | None, str | None]:
    """Search TM for player, return (photo_url, player_id, slug)."""
    result = scraper.search_player(name)
    if not result:
        return None, None, None
    player_id = result["player_id"]
    slug = result["slug"]
    photo_url = get_photo_url_from_profile(scraper, player_id, slug)
    return photo_url, player_id, slug


def load_existing() -> dict:
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=["laliga", "epl", "all"], default="all")
    args = parser.parse_args()

    from datetime import datetime
    now = datetime.now()
    season = now.year if now.month >= 8 else now.year - 1

    scraper = TransfermarktScraper(cache_hours=168)  # 1-week cache for profiles
    photo_map = load_existing()
    print(f"Loaded {len(photo_map)} existing entries from photo map")

    player_names = []

    if args.league in ("laliga", "all"):
        from src.data_loaders.api_client import FootballDataClient
        client = FootballDataClient()
        print("Fetching La Liga squads...")
        la_df = client.get_all_la_liga_squads(season=season)
        player_names += la_df["name"].unique().tolist()
        print(f"  {len(la_df['name'].unique())} La Liga players")

    if args.league in ("epl", "all"):
        from src.data_loaders.api_client import FootballDataClient
        client = FootballDataClient()
        print("Fetching EPL squads...")
        epl_df = client.get_all_team_squads()
        player_names += epl_df["name"].unique().tolist()
        print(f"  {len(epl_df['name'].unique())} EPL players")

    # Deduplicate and skip already cached
    player_names = list(dict.fromkeys(player_names))  # preserve order, deduplicate
    new_names = [n for n in player_names if n.lower() not in photo_map]
    print(f"\n{len(player_names)} total players, {len(new_names)} need photo lookup")

    found = 0
    not_found = 0
    errors = 0

    for i, name in enumerate(new_names, 1):
        try:
            photo_url, player_id, slug = get_photo_url_for_name(scraper, name)
            key = name.lower()
            if photo_url:
                photo_map[key] = photo_url
                # Also add accent-stripped variant
                import unicodedata
                stripped = "".join(
                    c for c in unicodedata.normalize("NFD", key)
                    if unicodedata.category(c) != "Mn"
                )
                if stripped != key:
                    photo_map[stripped] = photo_url
                found += 1
                if i % 50 == 0 or i <= 5:
                    print(f"  [{i}/{len(new_names)}] {name}: OK (id={player_id})")
            else:
                not_found += 1
                if i % 100 == 0:
                    print(f"  [{i}/{len(new_names)}] {name}: not found")
        except Exception as e:
            errors += 1
            logger.debug(f"Error for {name}: {e}")

    # Save
    with open(OUT_PATH, "w") as f:
        json.dump(photo_map, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {found} found, {not_found} not found, {errors} errors")
    print(f"Total in map: {len(photo_map)}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
