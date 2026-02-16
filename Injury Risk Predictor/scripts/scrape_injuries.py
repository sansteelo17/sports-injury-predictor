"""
Bulk scrape injury history from Transfermarkt for all players in inference_df.

Saves results to models/player_history.pkl (enriched with real injury data).
Uses caching to avoid re-scraping players already fetched recently.

Usage:
    python scripts/scrape_injuries.py              # Scrape all missing/stale players
    python scripts/scrape_injuries.py --force      # Re-scrape everything
    python scripts/scrape_injuries.py --test 5     # Test with 5 players only
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.data_loaders.transfermarkt_scraper import TransfermarktScraper


MODELS_DIR = ROOT / "models"
HISTORY_FILE = MODELS_DIR / "player_history.pkl"
DETAIL_FILE = MODELS_DIR / "player_injuries_detail.pkl"
SCRAPE_META_FILE = MODELS_DIR / "injury_scrape_meta.pkl"

# Import injury classification from extract script
sys.path.insert(0, str(ROOT / "scripts"))
from extract_injury_detail import classify_injury


def load_existing_history() -> pd.DataFrame:
    """Load existing player_history.pkl if it exists."""
    if HISTORY_FILE.exists():
        df = pd.read_pickle(HISTORY_FILE)
        print(f"Loaded existing history: {len(df)} players")
        return df
    return pd.DataFrame()


def load_scrape_metadata() -> dict:
    """Load metadata about when each player was last scraped."""
    if SCRAPE_META_FILE.exists():
        return pd.read_pickle(SCRAPE_META_FILE)
    return {}


def save_scrape_metadata(meta: dict):
    """Save scrape metadata."""
    pd.to_pickle(meta, SCRAPE_META_FILE)


def get_players_to_scrape(inference_df: pd.DataFrame, existing: pd.DataFrame,
                          meta: dict, force: bool = False, max_age_days: int = 7) -> list:
    """Determine which players need scraping."""
    all_names = inference_df["name"].unique().tolist()

    if force:
        return all_names

    existing_names = set(existing["name"].tolist()) if not existing.empty else set()
    now = datetime.now()

    to_scrape = []
    for name in all_names:
        if name not in existing_names:
            to_scrape.append(name)
        elif name in meta:
            scraped_at = meta[name]
            if (now - scraped_at).days >= max_age_days:
                to_scrape.append(name)
        else:
            # In existing but no metadata - treat as stale
            to_scrape.append(name)

    return to_scrape


def scrape_player(scraper: TransfermarktScraper, name: str) -> tuple:
    """Scrape a single player's injury data.

    Returns:
        (summary_dict, detail_records_list) - summary stats and per-injury records
    """
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

    # Calculate severity stats
    days_list = [i["days_out"] for i in injuries if i["days_out"] > 0]
    avg_severity = sum(days_list) / len(days_list) if days_list else 0
    worst_injury = max(days_list) if days_list else 0
    severity_std = pd.Series(days_list).std() if len(days_list) > 1 else 0

    summary = {
        "name": name,
        "player_injury_count": total_injuries,
        "player_avg_severity": round(avg_severity, 2),
        "player_worst_injury": float(worst_injury),
        "player_severity_std": round(float(severity_std), 2),
        "is_injury_prone": int(total_injuries >= 3),
        "total_days_lost": total_days,
        "days_since_last_injury": days_since,
        "last_injury_date": data.get("last_injury_date"),
        "scrape_status": "found",
    }

    # Build per-injury detail records for KMeans clustering
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
            "severity_days": inj.get("days_out", 0),
            "body_area": body_area,
            "injury_type": inj_type,
            "injury_raw": inj.get("injury", ""),
            "games_missed": inj.get("games_missed", 0),
        })

    return summary, detail_records


def main():
    parser = argparse.ArgumentParser(description="Scrape injury history from Transfermarkt")
    parser.add_argument("--force", action="store_true", help="Re-scrape all players")
    parser.add_argument("--test", type=int, default=0, help="Only scrape N players (for testing)")
    parser.add_argument("--max-age", type=int, default=7, help="Max age in days before re-scraping")
    args = parser.parse_args()

    # Load inference_df to get player list
    idf_path = MODELS_DIR / "inference_df.pkl"
    if not idf_path.exists():
        print("ERROR: models/inference_df.pkl not found. Run notebook first.")
        sys.exit(1)

    inference_df = pd.read_pickle(idf_path)
    print(f"Inference DF: {len(inference_df)} players")

    # Load existing data
    existing = load_existing_history()
    meta = load_scrape_metadata()

    # Determine who to scrape
    to_scrape = get_players_to_scrape(inference_df, existing, meta,
                                       force=args.force, max_age_days=args.max_age)

    if args.test > 0:
        to_scrape = to_scrape[:args.test]

    print(f"Players to scrape: {len(to_scrape)}")
    if not to_scrape:
        print("Nothing to do!")
        return

    # Estimate time (3 seconds per request, ~2 requests per player: search + injury page)
    est_minutes = (len(to_scrape) * 6) / 60
    print(f"Estimated time: ~{est_minutes:.0f} minutes (rate limited to avoid blocking)")

    # Scrape
    scraper = TransfermarktScraper()
    results = []
    all_detail_records = []
    errors = 0
    start_time = time.time()

    for i, name in enumerate(to_scrape):
        pct = ((i + 1) / len(to_scrape)) * 100
        elapsed = time.time() - start_time
        rate = (i + 1) / max(elapsed, 1) * 60
        print(f"  [{i+1}/{len(to_scrape)}] ({pct:.0f}%) {name}... ", end="", flush=True)

        try:
            result, detail_records = scrape_player(scraper, name)
            results.append(result)
            all_detail_records.extend(detail_records)
            meta[name] = datetime.now()

            status = result["scrape_status"]
            inj_count = result["player_injury_count"]
            if status == "found":
                print(f"{inj_count} injuries")
            else:
                print("not found on Transfermarkt")

        except Exception as e:
            errors += 1
            print(f"ERROR: {e}")
            results.append({
                "name": name,
                "player_injury_count": 0,
                "player_avg_severity": 0.0,
                "player_worst_injury": 0.0,
                "player_severity_std": 0.0,
                "is_injury_prone": 0,
                "total_days_lost": 0,
                "days_since_last_injury": 365 * 5,
                "last_injury_date": None,
                "scrape_status": "error",
            })

        # Save progress every 50 players
        if (i + 1) % 50 == 0:
            _save_progress(existing, results, meta)
            print(f"  --- Progress saved ({i+1} done) ---")

    # Final save
    _save_progress(existing, results, meta)

    # Save per-injury detail for KMeans clustering
    _save_detail(all_detail_records)

    elapsed = time.time() - start_time
    print(f"\nDone! Scraped {len(results)} players in {elapsed/60:.1f} minutes")
    print(f"  Found: {sum(1 for r in results if r['scrape_status'] == 'found')}")
    print(f"  Not found: {sum(1 for r in results if r['scrape_status'] == 'not_found')}")
    print(f"  Errors: {errors}")

    # Show final stats
    final = pd.read_pickle(HISTORY_FILE)
    has_injuries = (final["player_injury_count"] > 0).sum()
    print(f"\nFinal player_history.pkl: {len(final)} players, {has_injuries} with injury records")
    if DETAIL_FILE.exists():
        detail = pd.read_pickle(DETAIL_FILE)
        print(f"Per-injury detail: {len(detail)} records for KMeans clustering")


def _save_progress(existing: pd.DataFrame, new_results: list, meta: dict):
    """Merge new results with existing and save."""
    new_df = pd.DataFrame(new_results)

    if not existing.empty:
        # Add columns that may not exist in old data
        for col in ["total_days_lost", "days_since_last_injury", "last_injury_date", "scrape_status"]:
            if col not in existing.columns:
                existing[col] = None

        # Remove old entries for players we just re-scraped
        scraped_names = set(new_df["name"])
        existing = existing[~existing["name"].isin(scraped_names)]

        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["name"], keep="last")
    # Strip StringDtype for cross-version pickle compatibility
    from src.utils.model_io import _strip_stringdtype
    combined = _strip_stringdtype(combined)
    import pickle
    with open(HISTORY_FILE, "wb") as f:
        pickle.dump(combined, f, protocol=4)
    save_scrape_metadata(meta)


def _save_detail(new_detail_records: list):
    """Merge new per-injury detail records with existing and save."""
    if not new_detail_records:
        return

    new_detail = pd.DataFrame(new_detail_records)
    scraped_names = set(new_detail["name"])

    if DETAIL_FILE.exists():
        existing_detail = pd.read_pickle(DETAIL_FILE)
        # Remove old records for re-scraped players
        existing_detail = existing_detail[~existing_detail["name"].isin(scraped_names)]
        combined = pd.concat([existing_detail, new_detail], ignore_index=True)
    else:
        combined = new_detail

    from src.utils.model_io import _strip_stringdtype
    combined = _strip_stringdtype(combined)
    import pickle
    with open(DETAIL_FILE, "wb") as f:
        pickle.dump(combined, f, protocol=4)
    print(f"  Saved {len(combined)} per-injury detail records to {DETAIL_FILE}")


if __name__ == "__main__":
    main()
