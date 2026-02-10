#!/usr/bin/env python3
"""
Enrich inference_df with minutes played data from Transfermarkt.

This is a one-time batch job that adds minutes_played to each player.
Takes ~25-30 minutes due to rate limiting (3s per player, 500+ players).

Usage:
    python scripts/enrich_with_minutes.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.data_loaders.transfermarkt_scraper import TransfermarktScraper
from src.utils.model_io import load_artifacts, save_artifacts


def enrich_with_minutes(inference_df: pd.DataFrame, resume_from: int = 0) -> pd.DataFrame:
    """
    Add minutes_played data to inference_df by scraping Transfermarkt.

    Args:
        inference_df: DataFrame with player predictions
        resume_from: Index to resume from (for interrupted runs)

    Returns:
        Enriched DataFrame with minutes_played column
    """
    scraper = TransfermarktScraper()

    # Initialize columns if they don't exist
    if 'minutes_played' not in inference_df.columns:
        inference_df['minutes_played'] = 0
    if 'appearances' not in inference_df.columns:
        inference_df['appearances'] = 0

    players = inference_df['name'].unique()
    print(f"Enriching {len(players)} players with minutes data...")
    print(f"Estimated time: {len(players) * 3 / 60:.0f} minutes")

    # Create a cache file to save progress
    cache_file = PROJECT_ROOT / "data" / "cache" / "minutes_cache.csv"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache if resuming
    cached = {}
    if cache_file.exists() and resume_from > 0:
        cache_df = pd.read_csv(cache_file)
        cached = dict(zip(cache_df['name'], zip(cache_df['minutes_played'], cache_df['appearances'])))
        print(f"Loaded {len(cached)} cached entries")

    results = []

    for i, player_name in enumerate(tqdm(players, desc="Fetching minutes")):
        if i < resume_from:
            continue

        # Check cache first
        if player_name in cached:
            minutes, apps = cached[player_name]
            results.append({'name': player_name, 'minutes_played': minutes, 'appearances': apps})
            continue

        try:
            # Search for player and get stats
            player = scraper.search_player(player_name)
            if player:
                stats = scraper.get_player_stats(player['slug'], player['player_id'])
                minutes = stats.get('minutes_played', 0)
                apps = stats.get('appearances', 0)
            else:
                minutes = 0
                apps = 0

            results.append({'name': player_name, 'minutes_played': minutes, 'appearances': apps})

            # Save progress every 50 players
            if len(results) % 50 == 0:
                pd.DataFrame(results).to_csv(cache_file, index=False)
                print(f"\nSaved progress: {len(results)} players processed")

        except Exception as e:
            print(f"\nError fetching {player_name}: {e}")
            results.append({'name': player_name, 'minutes_played': 0, 'appearances': 0})

    # Final save
    pd.DataFrame(results).to_csv(cache_file, index=False)

    # Merge results into inference_df
    results_df = pd.DataFrame(results)

    # Update the inference_df
    for _, row in results_df.iterrows():
        mask = inference_df['name'] == row['name']
        inference_df.loc[mask, 'minutes_played'] = row['minutes_played']
        inference_df.loc[mask, 'appearances'] = row['appearances']

    return inference_df


def main():
    print("Loading model artifacts...")
    artifacts = load_artifacts()

    if not artifacts or 'inference_df' not in artifacts:
        print("ERROR: No inference_df found. Run the notebook first to generate predictions.")
        sys.exit(1)

    inference_df = artifacts['inference_df']
    print(f"Loaded inference_df with {len(inference_df)} rows")

    # Check if already enriched
    if 'minutes_played' in inference_df.columns and inference_df['minutes_played'].sum() > 0:
        print(f"Already has minutes data. Total minutes: {inference_df['minutes_played'].sum()}")
        response = input("Re-fetch all minutes data? (y/N): ")
        if response.lower() != 'y':
            print("Exiting.")
            return

    # Enrich with minutes
    enriched_df = enrich_with_minutes(inference_df)

    # Update artifacts
    artifacts['inference_df'] = enriched_df

    # Save back
    print("\nSaving enriched artifacts...")
    save_artifacts(
        models_dir=PROJECT_ROOT / "models",
        ensemble=artifacts.get('ensemble'),
        severity_clf=artifacts.get('severity_clf'),
        df_clusters=artifacts.get('df_clusters'),
        player_history=artifacts.get('player_history'),
        inference_df=enriched_df
    )

    # Print stats
    starters = enriched_df[enriched_df['minutes_played'] >= 900]['name'].nunique()
    print(f"\nDone! {starters} players identified as regular starters (900+ minutes)")
    print(f"Total minutes scraped: {enriched_df['minutes_played'].sum():,}")


if __name__ == "__main__":
    main()
