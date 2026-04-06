#!/usr/bin/env python3
"""
Scrape injury history for La Liga players from Transfermarkt.

Extracts unique La Liga players from All_Players_1992-2025.csv (last N seasons),
then runs the Transfermarkt scraper on each one. Saves results to:
  models/laliga_player_history.pkl   — per-player injury summary
  models/laliga_injuries_detail.pkl  — per-injury records

Usage:
    python scripts/scrape_laliga_injuries.py
    python scripts/scrape_laliga_injuries.py --seasons 3    # last 3 seasons only
    python scripts/scrape_laliga_injuries.py --test 10      # 10 players only
    python scripts/scrape_laliga_injuries.py --force        # re-scrape all
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

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
from src.data_loaders.transfermarkt_scraper import TransfermarktScraper
from src.utils.logger import get_logger

sys.path.insert(0, str(ROOT / "scripts"))
from extract_injury_detail import classify_injury

logger = get_logger(__name__)

MODELS_DIR = ROOT / "models"
STATS_CSV = ROOT / "csv" / "All_Players_1992-2025.csv"
OUT_HISTORY = MODELS_DIR / "laliga_player_history.pkl"
OUT_DETAIL = MODELS_DIR / "laliga_injuries_detail.pkl"
META_FILE = MODELS_DIR / "laliga_scrape_meta.pkl"


def load_laliga_players(seasons_back: int = 5) -> list[str]:
    """Extract unique La Liga player names from the stats CSV."""
    df = pd.read_csv(STATS_CSV, low_memory=False)
    la_liga = df[df["League"] == "La Liga"].copy()

    # Season is "YYYY-YYYY" — extract end year for comparison
    la_liga["season_end_year"] = la_liga["Season"].str.split("-").str[-1].astype(int)
    max_year = la_liga["season_end_year"].max()
    cutoff_year = max_year - seasons_back
    recent = la_liga[la_liga["season_end_year"] > cutoff_year]

    players = sorted(recent["Player"].dropna().unique().tolist())
    logger.info(f"La Liga players (last {seasons_back} seasons, end year >{cutoff_year}): {len(players)}")
    return players


def scrape_player(scraper: TransfermarktScraper, name: str) -> tuple:
    """Scrape a single player. Returns (summary_dict, detail_records)."""
    data = scraper.fetch_player_injuries(name, include_stats=False)
    if not data:
        return {
            "name": name,
            "player_injury_count": 0,
            "player_avg_severity": 0.0,
            "player_worst_injury": 0.0,
            "player_severity_std": 0.0,
            "is_injury_prone": 0,
            "total_days_lost": 0,
            "days_since_last_injury": 365 * 5,
            "last_injury_date": None,
            "scrape_status": "not_found",
        }, []

    injuries = data.get("injuries", [])
    total_injuries = data.get("total_injuries", 0)
    total_days = data.get("total_days_out", 0)
    days_since = data.get("days_since_last", 365 * 5)

    days_list = [i["days_out"] for i in injuries if i.get("days_out", 0) > 0]
    avg_sev = sum(days_list) / len(days_list) if days_list else 0.0
    worst = max(days_list) if days_list else 0.0
    std = float(pd.Series(days_list).std()) if len(days_list) > 1 else 0.0

    summary = {
        "name": name,
        "player_injury_count": total_injuries,
        "player_avg_severity": round(avg_sev, 2),
        "player_worst_injury": worst,
        "player_severity_std": round(std, 2),
        "is_injury_prone": int(total_injuries >= 3),
        "total_days_lost": total_days,
        "days_since_last_injury": days_since,
        "last_injury_date": data.get("last_injury_date"),
        "scrape_status": "found",
    }

    detail_records = []
    for inj in injuries:
        body_area, inj_type = classify_injury(inj.get("injury", ""))
        try:
            inj_date = datetime.strptime(inj["date"], "%Y-%m-%d")
        except (ValueError, KeyError):
            continue
        detail_records.append({
            "name": name,
            "injury_datetime": inj_date,
            "injury_type": inj_type,
            "body_area": body_area,
            "days_out": inj.get("days_out", 0),
            "season": inj.get("season", ""),
        })

    return summary, detail_records


def main():
    parser = argparse.ArgumentParser(description="Scrape La Liga player injuries from Transfermarkt")
    parser.add_argument("--seasons", type=int, default=5, help="How many recent seasons to include (default 5)")
    parser.add_argument("--test", type=int, default=0, help="Only scrape N players (for testing)")
    parser.add_argument("--force", action="store_true", help="Re-scrape all players")
    args = parser.parse_args()

    players = load_laliga_players(seasons_back=args.seasons)

    # Load existing
    existing_history = pd.read_pickle(OUT_HISTORY) if OUT_HISTORY.exists() else pd.DataFrame()
    meta = pd.read_pickle(META_FILE) if META_FILE.exists() else {}
    existing_names = set(existing_history["name"].tolist()) if not existing_history.empty else set()

    now = datetime.now()
    if args.force:
        to_scrape = players
    else:
        to_scrape = []
        for name in players:
            if name not in existing_names:
                to_scrape.append(name)
            elif name in meta and (now - meta[name]).days < 7:
                pass  # Fresh enough
            else:
                to_scrape.append(name)

    if args.test > 0:
        to_scrape = to_scrape[:args.test]

    print(f"\nLa Liga players total: {len(players)}")
    print(f"To scrape: {len(to_scrape)}")
    print(f"Already cached: {len(players) - len(to_scrape)}")

    if not to_scrape:
        print("Nothing to scrape. Use --force to re-scrape all.")
        return

    scraper = TransfermarktScraper()
    summaries = existing_history.to_dict("records") if not existing_history.empty else []
    detail_records = []

    if (ROOT / "models" / "laliga_injuries_detail.pkl").exists():
        existing_detail = pd.read_pickle(ROOT / "models" / "laliga_injuries_detail.pkl")
        detail_records = existing_detail.to_dict("records")

    found = 0
    not_found = 0

    for i, name in enumerate(to_scrape):
        print(f"[{i+1}/{len(to_scrape)}] {name}...", end=" ", flush=True)
        summary, details = scrape_player(scraper, name)
        status = summary["scrape_status"]
        print(f"{status} ({summary['player_injury_count']} injuries)")

        summaries = [s for s in summaries if s["name"] != name]
        summaries.append(summary)
        detail_records = [d for d in detail_records if d["name"] != name]
        detail_records.extend(details)
        meta[name] = now

        if status == "found":
            found += 1
        else:
            not_found += 1

        # Save checkpoint every 50 players
        if (i + 1) % 50 == 0:
            _save(summaries, detail_records, meta)
            print(f"  [checkpoint saved at {i+1}]")

    _save(summaries, detail_records, meta)

    print(f"\nDone. Found: {found} | Not found: {not_found}")
    print(f"Total in history: {len(summaries)} players")
    print(f"Saved to {OUT_HISTORY}")


def _save(summaries, detail_records, meta):
    history_df = pd.DataFrame(summaries)
    pd.to_pickle(history_df, OUT_HISTORY)
    if detail_records:
        detail_df = pd.DataFrame(detail_records)
        pd.to_pickle(detail_df, OUT_DETAIL)
    pd.to_pickle(meta, META_FILE)


if __name__ == "__main__":
    main()
