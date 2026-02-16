#!/usr/bin/env python3
"""
Scheduled retraining pipeline for YaraSports Injury Risk Predictor.

Orchestrates the full pipeline:
1. Scrape fresh injury data from Transfermarkt
2. Refresh predictions with latest workload/match data
3. Re-cluster archetypes (HDBSCAN + KMeans)
4. Save versioned metadata

Designed to run on a schedule (Tue/Fri) via cron.

Usage:
    python scripts/retrain.py                    # Fast mode (API-based workload)
    python scripts/retrain.py --accurate         # Accurate mode (FBref scraping)
    python scripts/retrain.py --skip-scrape      # Skip injury re-scrape
    python scripts/retrain.py --dry-run          # Preview only, don't save
"""

import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env if it exists
_env_path = ROOT / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
METADATA_FILE = MODELS_DIR / "metadata.json"
RETRAIN_LOG = LOGS_DIR / "retrain.log"

# Max number of historical runs to keep in metadata
MAX_RUN_HISTORY = 20


def setup_logging():
    """Configure logging to both stdout and retrain.log."""
    LOGS_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger("retrain")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(RETRAIN_LOG, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # Stream handler (stdout)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    return logger


def load_metadata() -> dict:
    """Load existing metadata or return empty structure."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            return json.load(f)
    return {"runs": []}


def save_metadata(meta: dict):
    """Save metadata to JSON, keeping last N runs."""
    meta["runs"] = meta.get("runs", [])[-MAX_RUN_HISTORY:]
    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def step_scrape_injuries(logger, force=False, max_age=7):
    """Step 1: Scrape fresh injury data from Transfermarkt."""
    logger.info("=" * 60)
    logger.info("STEP 1: Scraping injury data from Transfermarkt")
    logger.info("=" * 60)

    import pandas as pd
    from scripts.scrape_injuries import (
        load_existing_history,
        load_scrape_metadata,
        save_scrape_metadata,
        get_players_to_scrape,
        scrape_player,
        _save_progress,
        _save_detail,
        MODELS_DIR as SCRAPE_MODELS_DIR,
    )
    from src.data_loaders.transfermarkt_scraper import TransfermarktScraper

    # Load inference_df for player list
    idf_path = MODELS_DIR / "inference_df.pkl"
    if not idf_path.exists():
        logger.warning("No inference_df.pkl found — skipping scrape")
        return 0

    inference_df = pd.read_pickle(idf_path)
    existing = load_existing_history()
    meta = load_scrape_metadata()

    to_scrape = get_players_to_scrape(
        inference_df, existing, meta, force=force, max_age_days=max_age
    )

    if not to_scrape:
        logger.info("All players up to date — nothing to scrape")
        return 0

    logger.info(f"Scraping {len(to_scrape)} players...")
    scraper = TransfermarktScraper()
    results = []
    all_detail_records = []
    errors = 0

    for i, name in enumerate(to_scrape):
        try:
            result, detail_records = scrape_player(scraper, name)
            results.append(result)
            all_detail_records.extend(detail_records)
            meta[name] = datetime.now()

            if result["scrape_status"] == "found":
                logger.info(f"  [{i+1}/{len(to_scrape)}] {name}: {result['player_injury_count']} injuries")
            else:
                logger.info(f"  [{i+1}/{len(to_scrape)}] {name}: not found")
        except Exception as e:
            errors += 1
            logger.warning(f"  [{i+1}/{len(to_scrape)}] {name}: ERROR - {e}")
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

    _save_progress(existing, results, meta)
    _save_detail(all_detail_records)
    save_scrape_metadata(meta)

    scraped_count = len(results)
    found = sum(1 for r in results if r["scrape_status"] == "found")
    logger.info(f"Scraping complete: {found}/{scraped_count} found, {errors} errors")
    return scraped_count


def step_refresh_predictions(logger, mode="api", dry_run=False):
    """Step 2: Refresh predictions with latest data."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"STEP 2: Refreshing predictions (mode={mode})")
    logger.info("=" * 60)

    from src.utils.model_io import load_artifacts, save_artifacts
    from scripts.refresh_predictions import (
        refresh_with_api,
        refresh_with_fbref,
        run_inference,
    )

    # Load trained models
    artifacts = load_artifacts()
    if artifacts is None:
        logger.error("No trained models found. Cannot refresh predictions.")
        return None

    ensemble = artifacts["ensemble"]
    severity_clf = artifacts["severity_clf"]
    player_history = artifacts["player_history"]
    archetype_df = artifacts["df_clusters"]
    logger.info(f"Loaded models ({len(player_history)} player histories)")

    # Get player snapshots
    if mode == "fbref":
        snapshots_df = refresh_with_fbref(artifacts, dry_run)
    else:
        api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
        if not api_key:
            logger.error("FOOTBALL_DATA_API_KEY not set — cannot use API mode")
            # Try fbref as fallback
            logger.info("Falling back to FBref mode...")
            snapshots_df = refresh_with_fbref(artifacts, dry_run)
        else:
            snapshots_df = refresh_with_api(artifacts, api_key, dry_run)

    if snapshots_df is None or len(snapshots_df) == 0:
        logger.error("No player snapshots generated")
        return None

    # Run inference
    inference_df = run_inference(
        snapshots_df, ensemble, severity_clf, player_history, archetype_df
    )

    logger.info(f"Generated predictions for {len(inference_df)} players")

    if not dry_run:
        save_artifacts(
            ensemble=ensemble,
            severity_clf=severity_clf,
            df_clusters=archetype_df,
            player_history=player_history,
            inference_df=inference_df,
        )
        logger.info("Artifacts saved")

    return inference_df


def step_recluster_archetypes(logger, inference_df, dry_run=False):
    """Step 3: Re-cluster archetypes using HDBSCAN + KMeans."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Re-clustering archetypes (HDBSCAN + KMeans)")
    logger.info("=" * 60)

    # Import the hybrid function from the API
    from api.main import assign_hybrid_archetypes

    inference_df = assign_hybrid_archetypes(inference_df)

    counts = inference_df["archetype"].value_counts().to_dict()
    logger.info(f"Archetype distribution: {counts}")

    if not dry_run:
        import pandas as pd
        # Save updated inference_df with new archetypes
        idf_path = MODELS_DIR / "inference_df.pkl"
        import pickle
        # Strip StringDtype for compatibility
        from src.utils.model_io import _strip_stringdtype
        df_clean = _strip_stringdtype(inference_df.copy())
        with open(idf_path, "wb") as f:
            pickle.dump(df_clean, f, protocol=4)
        logger.info(f"Updated inference_df saved with new archetypes")

    return inference_df


def step_save_metadata(logger, inference_df, mode, duration_sec, scraped_count):
    """Step 4: Save versioned metadata."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Saving metadata")
    logger.info("=" * 60)

    import pandas as pd

    meta = load_metadata()

    # Compute stats
    high = int((inference_df["ensemble_prob"] >= 0.6).sum()) if "ensemble_prob" in inference_df.columns else 0
    medium = int(((inference_df["ensemble_prob"] >= 0.4) & (inference_df["ensemble_prob"] < 0.6)).sum()) if "ensemble_prob" in inference_df.columns else 0
    low = int((inference_df["ensemble_prob"] < 0.4).sum()) if "ensemble_prob" in inference_df.columns else 0
    mean_risk = round(float(inference_df["ensemble_prob"].mean()), 4) if "ensemble_prob" in inference_df.columns else 0

    top_player = ""
    if "ensemble_prob" in inference_df.columns and "name" in inference_df.columns:
        top_idx = inference_df["ensemble_prob"].idxmax()
        top_player = inference_df.loc[top_idx, "name"]

    archetype_dist = inference_df["archetype"].value_counts().to_dict() if "archetype" in inference_df.columns else {}

    # Count injury records
    detail_path = MODELS_DIR / "player_injuries_detail.pkl"
    injury_records = 0
    if detail_path.exists():
        try:
            detail_df = pd.read_pickle(detail_path)
            injury_records = len(detail_df)
        except Exception:
            pass

    now = datetime.utcnow().isoformat() + "Z"
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    run_entry = {
        "timestamp": now,
        "mode": mode,
        "players": len(inference_df),
        "scraped": scraped_count,
        "duration_sec": round(duration_sec, 1),
        "mean_risk": mean_risk,
    }

    meta.update({
        "last_retrain": now,
        "mode": mode,
        "player_count": len(inference_df),
        "injury_records": injury_records,
        "risk_distribution": {"high": high, "medium": medium, "low": low},
        "archetype_distribution": {str(k): int(v) for k, v in archetype_dist.items()},
        "mean_risk": mean_risk,
        "top_risk_player": top_player,
        "version": version,
    })
    meta.setdefault("runs", []).append(run_entry)

    save_metadata(meta)
    logger.info(f"Metadata saved: version={version}, {len(inference_df)} players")
    logger.info(f"Risk distribution: High={high}, Medium={medium}, Low={low}")


def main():
    parser = argparse.ArgumentParser(
        description="YaraSports retraining pipeline (scheduled Tue/Fri)"
    )
    parser.add_argument(
        "--accurate", action="store_true",
        help="Use FBref scraping for accurate workload (slower, ~20 min)"
    )
    parser.add_argument(
        "--skip-scrape", action="store_true",
        help="Skip Transfermarkt injury scraping"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview only — don't save anything"
    )
    parser.add_argument(
        "--force-scrape", action="store_true",
        help="Re-scrape all players (ignore cache age)"
    )
    parser.add_argument(
        "--max-scrape-age", type=int, default=7,
        help="Max age in days before re-scraping a player (default: 7)"
    )
    args = parser.parse_args()

    logger = setup_logging()
    mode = "fbref" if args.accurate else "api"
    start_time = time.time()

    logger.info("")
    logger.info("=" * 60)
    logger.info("YARASPORTS RETRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode: {mode.upper()}")
    logger.info(f"Scrape: {'SKIP' if args.skip_scrape else 'YES'}")
    logger.info(f"Dry run: {'YES' if args.dry_run else 'NO'}")

    # Step 1: Scrape injuries
    scraped_count = 0
    if not args.skip_scrape:
        try:
            scraped_count = step_scrape_injuries(
                logger, force=args.force_scrape, max_age=args.max_scrape_age
            )
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            logger.info("Continuing with existing injury data...")
    else:
        logger.info("\nSkipping injury scrape (--skip-scrape)")

    # Step 2: Refresh predictions
    inference_df = None
    try:
        inference_df = step_refresh_predictions(logger, mode=mode, dry_run=args.dry_run)
    except Exception as e:
        logger.error(f"Prediction refresh failed: {e}")
        if args.dry_run:
            logger.info("Dry run: falling back to existing inference_df...")
        else:
            import traceback
            traceback.print_exc()

    if inference_df is None:
        # Try loading existing inference_df as fallback
        import pandas as pd
        idf_path = MODELS_DIR / "inference_df.pkl"
        if idf_path.exists():
            inference_df = pd.read_pickle(idf_path)
            logger.info(f"Loaded existing inference_df ({len(inference_df)} players)")
        else:
            logger.error("No predictions generated and no existing data — aborting")
            sys.exit(1)

    # Step 3: Re-cluster archetypes
    try:
        inference_df = step_recluster_archetypes(logger, inference_df, dry_run=args.dry_run)
    except Exception as e:
        logger.error(f"Archetype clustering failed: {e}")
        logger.info("Continuing with existing archetypes...")

    # Step 4: Save metadata
    duration = time.time() - start_time
    if not args.dry_run:
        step_save_metadata(logger, inference_df, mode, duration, scraped_count)
    else:
        logger.info("\n[DRY RUN — metadata not saved]")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("RETRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
    logger.info(f"Players: {len(inference_df)}")
    logger.info(f"Scraped: {scraped_count}")
    if "ensemble_prob" in inference_df.columns:
        logger.info(f"Mean risk: {inference_df['ensemble_prob'].mean():.1%}")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
