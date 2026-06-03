"""Build the Special Players inference frame.

A curated, flat watch-list of marquee players OUTSIDE our covered leagues
(Messi, Ronaldo, Neymar, …). We have no workload data for their leagues, so the
ensemble cannot score them. Instead we source each player's injury history from
Transfermarkt and carry the history + age fields; the API derives a
history/age-based risk read (not the ensemble) at serve time.

Reads:  src/competitions/special_players_roster.json (curated by George)
Writes: data/processed/inference_special_players.pkl  (merged at API load)

Run:  python scripts/build_special_players.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.competitions.registry import SPECIAL_PLAYERS  # noqa: E402
from src.data_loaders.transfermarkt_scraper import TransfermarktScraper  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

ROSTER_PATH = ROOT / "src" / "competitions" / "special_players_roster.json"
OUT_PATH = ROOT / "data" / "processed" / "inference_special_players.pkl"

# Coarse body-area tag from a Transfermarkt injury label, for the injury map.
_AREA_KEYWORDS = [
    "hamstring", "knee", "ankle", "thigh", "calf", "groin", "foot", "hip",
    "back", "shoulder", "achilles", "muscle", "muscular", "adductor",
    "metatarsal", "cruciate", "ligament", "toe", "heel", "rib",
]


def _body_area(label: str) -> str:
    low = (label or "").lower()
    for kw in _AREA_KEYWORDS:
        if kw in low:
            return "muscle" if kw == "muscular" else kw
    return "unknown"


def _to_iso(d: str) -> str | None:
    """TM dates are dd/mm/YYYY or already YYYY-mm-dd; normalise to ISO."""
    d = (d or "").strip()
    if not d:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
        return d
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", d)
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None


def build() -> pd.DataFrame:
    roster = json.loads(ROSTER_PATH.read_text())["players"]
    scraper = TransfermarktScraper()
    rows = []
    detail_rows = []
    for entry in roster:
        name = entry["name"]
        try:
            r = scraper.fetch_player_injuries(name, include_stats=False)
        except Exception as e:
            logger.warning("Special Players: TM fetch failed for %s: %s", name, e)
            r = None
        if not r:
            logger.warning("Special Players: no injury data for %s — skipping", name)
            continue

        injuries = r.get("injuries") or []
        count = int(r.get("total_injuries") or len(injuries) or 0)
        days_lost = int(r.get("total_days_out") or sum(int(i.get("days_out") or 0) for i in injuries))
        days_since = r.get("days_since")
        if days_since is None:
            days_since = r.get("days_since_last")
        worst = max((int(i.get("days_out") or 0) for i in injuries), default=0)
        avg_sev = round(days_lost / count, 1) if count else 0.0

        club = entry.get("club") or r.get("team") or "Unknown"
        rows.append({
            "name": name,
            "team": club,
            "player_team": club,
            "club_team": club,
            "club_league": entry.get("league"),
            "league": SPECIAL_PLAYERS.name,
            "competition_id": SPECIAL_PLAYERS.id,
            "competition_type": SPECIAL_PLAYERS.type,
            "position": entry.get("position") or "Unknown",
            "age": r.get("age") or entry.get("age"),
            "nationality": entry.get("country"),
            "has_risk_features": False,
            "ensemble_prob": float("nan"),
            # Injury-history fields the serve path reads for the history/age read.
            "player_injury_count": count,
            "previous_injuries": count,
            "total_days_lost": days_lost,
            "days_since_last_injury": int(days_since) if days_since is not None else 365,
            "player_avg_severity": avg_sev,
            "player_worst_injury": worst,
            "last_injury_date": r.get("last_injury_date"),
        })
        for inj in injuries:
            iso = _to_iso(inj.get("date"))
            detail_rows.append({
                "name": name,
                "injury_datetime": iso,
                "date": iso,
                "body_area": _body_area(inj.get("injury")),
                "injury_raw": inj.get("injury"),
                "injury_type": inj.get("injury"),
                "severity_days": int(inj.get("days_out") or 0),
                "games_missed": int(inj.get("games_missed") or 0),
            })
        logger.info("Special Players: %s — %d injuries, %d days, last %s",
                    name, count, days_lost, r.get("last_injury_date"))

    df = pd.DataFrame(rows)
    detail_df = pd.DataFrame(detail_rows)
    return df, detail_df


def main() -> int:
    df, detail_df = build()
    if df.empty:
        logger.error("No Special Players rows built; aborting.")
        return 1
    print("\n=== SPECIAL PLAYERS FRAME ===")
    print("players:", len(df))
    print("injury-history coverage:", int((df["player_injury_count"] > 0).sum()), "/", len(df))
    print(df[["name", "team", "age", "player_injury_count", "total_days_lost",
              "days_since_last_injury"]].to_string(index=False))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUT_PATH)
    detail_path = OUT_PATH.with_name("special_players_injuries_detail.pkl")
    detail_df.to_pickle(detail_path)
    print(f"\nWROTE {OUT_PATH} ({len(df)} players)")
    print(f"WROTE {detail_path} ({len(detail_df)} injury records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
