#!/usr/bin/env python3
"""
Append Transfermarkt photo URLs for World Cup 2026 players into the existing
player_photo_map.json that the API loads at startup. WC players from PL/La
Liga are already covered by build_player_photo_map.py; this script handles
the ~970 players from leagues we don't otherwise track (Bundesliga, Serie A,
Saudi PL, MLS, plus a long tail for smaller nations).

Usage:
    python scripts/build_wc_player_photo_map.py
    python scripts/build_wc_player_photo_map.py --limit 50      # smoke test
    python scripts/build_wc_player_photo_map.py --save-every 25 # checkpoint
"""

import sys
import json
import argparse
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env so TM scraper headers/cookies populate correctly.
import os
_env = ROOT / ".env"
if _env.exists():
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

import pandas as pd

# Reuse the helpers from the EPL/La Liga photo-map script — same scraper,
# same JSON schema, same normalisation. Don't duplicate.
from scripts.build_player_photo_map import (
    OUT_PATH, get_photo_url_for_name, load_existing,
)
from src.data_loaders.transfermarkt_scraper import TransfermarktScraper


WC_PICKLE = ROOT / "data" / "processed" / "inference_international_world_cup_2026.pkl"


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many new lookups (smoke test).")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Checkpoint the JSON every N successful lookups.")
    args = parser.parse_args()

    if not WC_PICKLE.exists():
        print(f"Missing {WC_PICKLE}. Run scripts/build_international_predictions.py first.")
        return 1

    df = pd.read_pickle(WC_PICKLE)
    if "name" not in df.columns:
        print(f"WC pickle has no 'name' column. Columns: {list(df.columns)}")
        return 1

    photo_map = load_existing()
    print(f"Loaded {len(photo_map)} existing entries from {OUT_PATH.name}")

    # Deduplicate by name, prefer rows that carry a 'team' (national team) and
    # 'club_team' for the team_hint that TM search uses to disambiguate.
    seen: dict[str, dict] = {}
    for _, row in df.iterrows():
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        if name not in seen:
            seen[name] = {
                "country": row.get("team"),
                "club": row.get("club_team"),
            }

    todo = []
    for name, hints in seen.items():
        key = name.lower()
        stripped = _strip_accents(key)
        if key in photo_map or stripped in photo_map:
            continue
        todo.append((name, hints))

    print(f"{len(seen)} unique WC players, {len(seen) - len(todo)} already covered, {len(todo)} to look up")

    if args.limit:
        todo = todo[: args.limit]
        print(f"--limit set: only attempting first {len(todo)}")

    scraper = TransfermarktScraper(cache_hours=168)

    found = 0
    not_found = 0
    errors = 0
    since_checkpoint = 0

    for i, (name, hints) in enumerate(todo, 1):
        try:
            photo_url, player_id, _slug = get_photo_url_for_name(scraper, name)
            key = name.lower()
            if photo_url:
                photo_map[key] = photo_url
                stripped = _strip_accents(key)
                if stripped != key:
                    photo_map[stripped] = photo_url
                found += 1
                since_checkpoint += 1
                if i <= 5 or i % 25 == 0:
                    print(f"  [{i}/{len(todo)}] {name} ({hints.get('country')}): OK (id={player_id})")
                if since_checkpoint >= args.save_every:
                    with open(OUT_PATH, "w", encoding="utf-8") as f:
                        json.dump(photo_map, f, indent=2, ensure_ascii=False)
                    since_checkpoint = 0
            else:
                not_found += 1
                if i % 50 == 0:
                    print(f"  [{i}/{len(todo)}] {name}: not found")
        except KeyboardInterrupt:
            print("\nInterrupted — saving progress and exiting.")
            break
        except Exception as e:
            errors += 1
            if errors <= 5 or errors % 25 == 0:
                print(f"  [{i}/{len(todo)}] {name}: error ({e})")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(photo_map, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {found} found, {not_found} not found, {errors} errors")
    print(f"Total in map: {len(photo_map)}")
    print(f"Saved to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
