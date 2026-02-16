"""
Extract per-injury records from Transfermarkt cached data.

Re-reads cached HTML (no network requests) and saves per-injury detail
needed for KMeans archetype clustering.

Saves to models/player_injuries_detail.pkl with columns:
    name, injury_datetime, severity_days, body_area, injury_type

Usage:
    python scripts/extract_injury_detail.py
"""

import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.data_loaders.transfermarkt_scraper import TransfermarktScraper

MODELS_DIR = ROOT / "models"
HISTORY_FILE = MODELS_DIR / "player_history.pkl"
DETAIL_FILE = MODELS_DIR / "player_injuries_detail.pkl"

# Map Transfermarkt injury descriptions to body_area and injury_type
# These are the most common patterns from Transfermarkt
INJURY_MAPPING = {
    # Muscle/strain injuries
    "muscle": ("muscle", "strain"),
    "hamstring": ("hamstring", "strain"),
    "calf": ("calf", "strain"),
    "thigh": ("thigh", "strain"),
    "groin": ("groin", "strain"),
    "quadricep": ("thigh", "strain"),
    "adductor": ("groin", "strain"),
    "abductor": ("hip", "strain"),
    "strain": ("muscle", "strain"),
    "pulled": ("muscle", "strain"),

    # Knee injuries
    "knee": ("knee", "injury"),
    "acl": ("knee", "tear"),
    "cruciate": ("knee", "tear"),
    "meniscus": ("knee", "tear"),
    "patella": ("knee", "injury"),
    "mcl": ("knee", "tear"),

    # Ankle/foot
    "ankle": ("ankle", "injury"),
    "foot": ("foot", "injury"),
    "metatarsal": ("foot", "fracture"),
    "achilles": ("achilles", "tear"),
    "plantar": ("foot", "strain"),

    # Ligament/tendon
    "ligament": ("knee", "tear"),
    "tendon": ("tendon", "tear"),
    "torn": ("muscle", "tear"),
    "rupture": ("muscle", "tear"),

    # Upper body
    "shoulder": ("shoulder", "injury"),
    "arm": ("arm", "injury"),
    "wrist": ("wrist", "injury"),
    "hand": ("hand", "injury"),
    "finger": ("hand", "injury"),
    "elbow": ("elbow", "injury"),
    "collarbone": ("shoulder", "fracture"),
    "rib": ("torso", "fracture"),

    # Back/hip
    "back": ("back", "injury"),
    "hip": ("hip", "injury"),
    "pelvi": ("hip", "injury"),
    "spine": ("back", "injury"),
    "lumbar": ("back", "injury"),

    # Head
    "head": ("head", "injury"),
    "concuss": ("head", "concussion"),
    "jaw": ("head", "fracture"),
    "nose": ("head", "fracture"),
    "facial": ("head", "injury"),

    # Other
    "fracture": ("bone", "fracture"),
    "broken": ("bone", "fracture"),
    "bruise": ("soft_tissue", "contusion"),
    "contusion": ("soft_tissue", "contusion"),
    "ill": ("illness", "illness"),
    "flu": ("illness", "illness"),
    "sick": ("illness", "illness"),
    "covid": ("illness", "illness"),
    "virus": ("illness", "illness"),
    "surgery": ("surgical", "surgery"),
    "operation": ("surgical", "surgery"),
}


def classify_injury(injury_text: str) -> tuple:
    """Map a Transfermarkt injury description to (body_area, injury_type)."""
    text = injury_text.lower()
    for keyword, (body_area, inj_type) in INJURY_MAPPING.items():
        if keyword in text:
            return body_area, inj_type
    return "unknown", "unknown"


def main():
    # Load player names from history
    if not HISTORY_FILE.exists():
        print("ERROR: models/player_history.pkl not found. Run scrape_injuries.py first.")
        sys.exit(1)

    history = pd.read_pickle(HISTORY_FILE)
    player_names = history["name"].tolist()
    print(f"Re-extracting injury detail for {len(player_names)} players from cache...")

    scraper = TransfermarktScraper()
    all_records = []
    players_with_detail = 0
    cache_hits = 0
    cache_misses = 0

    for i, name in enumerate(player_names):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(player_names)}] processed...")

        try:
            data = scraper.fetch_player_injuries(name, include_stats=False)
            if not data or not data.get("injuries"):
                continue

            players_with_detail += 1
            for inj in data["injuries"]:
                body_area, inj_type = classify_injury(inj.get("injury", ""))
                try:
                    inj_date = datetime.strptime(inj["date"], "%Y-%m-%d")
                except (ValueError, KeyError):
                    continue

                all_records.append({
                    "name": name,
                    "injury_datetime": inj_date,
                    "severity_days": inj.get("days_out", 0),
                    "body_area": body_area,
                    "injury_type": inj_type,
                    "injury_raw": inj.get("injury", ""),
                    "games_missed": inj.get("games_missed", 0),
                })
        except Exception as e:
            # Skip silently - player probably wasn't found
            continue

    detail_df = pd.DataFrame(all_records)
    detail_df.to_pickle(DETAIL_FILE)

    print(f"\nDone!")
    print(f"  Players with injury detail: {players_with_detail}")
    print(f"  Total injury records: {len(detail_df)}")
    if not detail_df.empty:
        print(f"  Body areas found: {detail_df['body_area'].nunique()}")
        print(f"  Injury types: {detail_df['injury_type'].nunique()}")
        print(f"  Players with 3+ injuries (eligible for KMeans): "
              f"{(detail_df.groupby('name').size() >= 3).sum()}")
    print(f"\nSaved to: {DETAIL_FILE}")


if __name__ == "__main__":
    main()
