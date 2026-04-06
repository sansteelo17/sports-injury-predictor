#!/usr/bin/env python3
"""
Convert scraped La Liga injury detail data into injury CSV format
and merge with player_injuries_impact.csv.

Reads:
  models/laliga_injuries_detail.pkl  — per-injury events from Transfermarkt
  csv/All_Players_1992-2025.csv      — for team/position lookup

Writes:
  csv/player_injuries_impact.csv     — updated with La Liga events appended

Usage:
    python scripts/build_laliga_injury_csv.py
    python scripts/build_laliga_injury_csv.py --dry-run
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np


DETAIL_PKL = ROOT / "models" / "laliga_injuries_detail.pkl"
STATS_CSV = ROOT / "csv" / "All_Players_1992-2025.csv"
INJURY_CSV = ROOT / "csv" / "player_injuries_impact.csv"


def build_laliga_injury_events() -> pd.DataFrame:
    detail = pd.read_pickle(DETAIL_PKL)

    # Filter: only events with a real date and meaningful days_out
    detail = detail[detail["injury_datetime"].notna()].copy()
    # Minimum 7 days — removes minor knocks and precautionary rests that
    # Transfermarkt records but don't represent true injury severity signals
    detail = detail[detail["days_out"] >= 7].copy()
    # Cap at 365 days — anything longer is likely a data error or compound re-injury
    detail = detail[detail["days_out"] <= 365].copy()

    # Filter to match our La Liga match data coverage (2023-25 seasons).
    # Injuries outside this window can't be paired with workload features,
    # which would feed the severity model null ACWR/load values.
    detail = detail[detail["injury_datetime"] >= "2023-08-01"].copy()

    # Load La Liga player info for team/position lookup
    stats = pd.read_csv(STATS_CSV, low_memory=False)
    la = stats[stats["League"] == "La Liga"].copy()

    # Extract end year from season string for sorting
    la["season_end_year"] = la["Season"].str.split("-").str[-1].astype(int)
    # Keep most recent entry per player for stable team/position
    la_latest = la.sort_values("season_end_year").drop_duplicates("Player", keep="last")
    player_info = la_latest.set_index("Player")[["Squad", "Pos", "Age"]].to_dict("index")

    rows = []
    unmatched = 0
    for _, row in detail.iterrows():
        name = row["name"]
        info = player_info.get(name, {})
        if not info:
            # Try case-insensitive
            name_lower = name.lower()
            for k, v in player_info.items():
                if k.lower() == name_lower:
                    info = v
                    break

        if not info:
            unmatched += 1
            continue

        inj_date = row["injury_datetime"]
        # Estimate return date from days_out
        return_date = inj_date + pd.Timedelta(days=int(row["days_out"]))

        # Map injury_type/body_area to an injury description
        body = row.get("body_area", "unknown")
        itype = row.get("injury_type", "unknown")
        if body and body != "unknown":
            injury_desc = f"{body.title()} {itype}" if itype and itype != "unknown" else body.title()
        else:
            injury_desc = itype.title() if itype and itype != "unknown" else "Unknown injury"

        # Derive season string from injury date
        year = inj_date.year
        month = inj_date.month
        season_start = year if month >= 8 else year - 1
        season_str = f"{season_start}/{str(season_start + 1)[2:]}"

        rows.append({
            "Name": name,
            "Team Name": info.get("Squad", "Unknown"),
            "Position": _normalize_position(info.get("Pos", "Unknown")),
            "Age": info.get("Age", np.nan),
            "Season": season_str,
            "FIFA rating": np.nan,
            "Injury": injury_desc,
            "Date of Injury": inj_date.strftime("%b %-d, %Y"),
            "Date of return": return_date.strftime("%b %-d, %Y"),
            "league": "La Liga",
        })

    print(f"  Built {len(rows)} La Liga injury events ({unmatched} skipped — no team match)")
    return pd.DataFrame(rows)


def _normalize_position(pos: str) -> str:
    if not isinstance(pos, str):
        return "Unknown"
    pos = pos.upper().strip()
    mapping = {
        "GK": "Goalkeeper",
        "DF": "Defender",
        "MF": "Midfielder",
        "FW": "Forward",
        "DF,MF": "Defender",
        "MF,FW": "Midfielder",
        "FW,MF": "Forward",
        "DF,FW": "Defender",
    }
    return mapping.get(pos, pos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Building La Liga injury events from Transfermarkt detail data...")
    laliga_df = build_laliga_injury_events()

    # Load existing EPL injury CSV
    existing = pd.read_csv(INJURY_CSV)
    existing["league"] = existing.get("league", pd.Series(["Premier League"] * len(existing), index=existing.index))
    if "league" not in existing.columns:
        existing["league"] = "Premier League"

    # Drop any existing La Liga rows (avoid duplicates on re-run)
    if "league" in existing.columns:
        existing = existing[existing["league"] != "La Liga"].copy()

    # Align columns
    all_cols = list(existing.columns)
    for col in all_cols:
        if col not in laliga_df.columns:
            laliga_df[col] = np.nan
    laliga_df = laliga_df[all_cols]

    combined = pd.concat([existing, laliga_df], ignore_index=True)

    print(f"\nEPL events:     {len(existing)}")
    print(f"La Liga events: {len(laliga_df)}")
    print(f"Total combined: {len(combined)}")

    if args.dry_run:
        print("\n[dry-run] Not saving.")
        return

    combined.to_csv(INJURY_CSV, index=False)
    print(f"\nSaved to {INJURY_CSV}")


if __name__ == "__main__":
    main()
