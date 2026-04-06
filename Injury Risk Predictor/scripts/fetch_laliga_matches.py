#!/usr/bin/env python3
"""
Fetch historical La Liga match data from football-data.org and save to csv/la-liga-matches.csv.

Usage:
    python scripts/fetch_laliga_matches.py
    python scripts/fetch_laliga_matches.py --seasons 2022 2023 2024
    python scripts/fetch_laliga_matches.py --dry-run   # print row count only
"""

import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_env = ROOT / ".env"
if _env.exists():
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                import os; os.environ.setdefault(k.strip(), v.strip())

import pandas as pd
from src.data_loaders.api_client import FootballDataClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

OUT_PATH = ROOT / "csv" / "la-liga-matches.csv"
DEFAULT_SEASONS = [2020, 2021, 2022, 2023, 2024]


def fetch_all_seasons(seasons: list[int], dry_run: bool = False) -> pd.DataFrame:
    client = FootballDataClient()
    frames = []

    for season in seasons:
        logger.info(f"Fetching La Liga {season}-{season+1}...")
        try:
            df = client.get_la_liga_matches(season=season)
            if len(df) == 0:
                logger.warning(f"  No matches returned for {season}")
                continue
            frames.append(df)
            logger.info(f"  {len(df)} matches fetched")
        except Exception as e:
            logger.error(f"  Failed for season {season}: {e}")
        # Be gentle with the rate limiter between seasons
        if season != seasons[-1]:
            time.sleep(2)

    if not frames:
        logger.error("No data fetched across any season.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "Home", "Away"]).sort_values("Date").reset_index(drop=True)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Fetch La Liga historical matches")
    parser.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS,
                        help="Season years to fetch (e.g. 2022 2023 2024)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, don't save")
    args = parser.parse_args()

    print(f"Fetching La Liga seasons: {args.seasons}")
    df = fetch_all_seasons(args.seasons, dry_run=args.dry_run)

    if df.empty:
        print("Nothing to save.")
        return

    print(f"\n{len(df)} total matches across {df['Season_End_Year'].nunique()} seasons")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Teams: {sorted(set(df['Home'].unique()) | set(df['Away'].unique()))}")

    if args.dry_run:
        print("\n[dry-run] Not saving.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing file if present
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH, parse_dates=["Date"])
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date", "Home", "Away"]).sort_values("Date").reset_index(drop=True)
        combined.to_csv(OUT_PATH, index=False)
        print(f"\nMerged with existing file → {len(combined)} total rows saved to {OUT_PATH}")
    else:
        df.to_csv(OUT_PATH, index=False)
        print(f"\nSaved {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
