"""Build the UEFA Champions League inference frame.

UCL is a cross-club competition, so it is modelled like the World Cup frame
(scripts/build_international_predictions.py) rather than like a domestic league:

  - Clubs already in our covered leagues (~22 of 36) reuse their existing club
    inference rows verbatim — their UCL match workload is already folded into
    acute load — re-tagged with competition_id=champions-league. Real risk.
  - The remaining clubs (~14: Ajax, Benfica, Sporting, PSV, Galatasaray, …) are
    not in any covered league, so we have no domestic workload/minutes for them.
    They get baseline, identity-only rows (has_risk_features=False,
    ensemble_prob=NaN) built from the football-data CL squad lists, exactly like
    WC baseline rows. The API surfaces these as risk_level="Unknown".

Output: data/processed/inference_champions_league_2026.pkl, merged at API load
the same way the WC frame is.

Run:  python scripts/build_ucl_predictions.py --season 2025
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.competitions.registry import CHAMPIONS_LEAGUE  # noqa: E402
from src.data_loaders.api_client import FootballDataClient  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

OUT_PATH = ROOT / "data" / "processed" / "inference_champions_league_2026.pkl"

# The 22 clubs that ARE in our covered leagues, keyed by their EXACT
# inference_df ``team`` string (verified against the live pkl). Their rows are
# reused as-is, so badges/stats already resolve.
COVERED_CLUB_TEAMS = {
    "Arsenal", "Atalanta", "Athletic Club", "Atletico Madrid", "Barcelona",
    "Bayern Munich", "Chelsea", "Borussia Dortmund", "Eintracht Frankfurt",
    "Internazionale", "Juventus", "Bayer Leverkusen", "Liverpool", "Man City",
    "Marseille", "Monaco", "Napoli", "Newcastle", "Paris Saint-Germain",
    "Real Madrid", "Tottenham", "Villarreal",
}

# football-data CL ``shortName`` values for those same 22 covered clubs, so we
# can tell which CL squads to SKIP (covered rows come from inference_df, not the
# football-data squad). Everything else in the 36 is a baseline club.
COVERED_CL_SHORTNAMES = {
    "Arsenal", "Atalanta", "Athletic", "Atleti", "Barça", "Bayern", "Chelsea",
    "Dortmund", "Frankfurt", "Inter", "Juventus", "Leverkusen", "Liverpool",
    "Man City", "Marseille", "Monaco", "Napoli", "Newcastle", "PSG",
    "Real Madrid", "Tottenham", "Villarreal",
}


def _age_from_dob(dob: str | None) -> int | None:
    if not dob:
        return None
    try:
        birth = datetime.strptime(dob, "%Y-%m-%d")
        return (datetime.now() - birth).days // 365
    except (ValueError, TypeError):
        return None


def build_covered_rows(club_df: pd.DataFrame) -> pd.DataFrame:
    """Reuse club inference rows for the covered UCL clubs (real risk)."""
    cov = club_df[club_df["team"].astype(str).isin(COVERED_CLUB_TEAMS)].copy()
    if cov.empty:
        return cov
    cov["club_team"] = cov["team"]
    cov["club_league"] = cov["league"]
    cov["team"] = cov["team"]  # keep the club string so badges resolve
    cov["league"] = CHAMPIONS_LEAGUE.name
    cov["competition_id"] = CHAMPIONS_LEAGUE.id
    cov["competition_type"] = CHAMPIONS_LEAGUE.type
    cov["has_risk_features"] = True
    logger.info(
        "UCL covered rows: %d players across %d clubs",
        len(cov), cov["club_team"].nunique(),
    )
    return cov


def build_baseline_rows(client: FootballDataClient, season: int) -> pd.DataFrame:
    """Identity-only rows for the non-covered CL clubs from football-data squads."""
    data = client._get("competitions/CL/teams", {"season": season})
    teams = data.get("teams", [])
    rows = []
    baseline_clubs = 0
    for t in teams:
        short = (t.get("shortName") or t.get("name") or "").strip()
        if short in COVERED_CL_SHORTNAMES:
            continue  # covered — comes from inference_df instead
        squad = t.get("squad") or []
        if not squad:
            # Squad not embedded in the competition-level payload; fetch the team.
            try:
                squad = (client._get(f"teams/{t['id']}") or {}).get("squad", []) or []
            except Exception as e:
                logger.warning("UCL baseline squad fetch failed for %s: %s", short, e)
                squad = []
        if not squad:
            continue
        baseline_clubs += 1
        club_display = short
        for p in squad:
            rows.append({
                "name": p.get("name"),
                "team": club_display,
                "player_team": club_display,
                "club_team": club_display,
                "club_league": None,
                "league": CHAMPIONS_LEAGUE.name,
                "competition_id": CHAMPIONS_LEAGUE.id,
                "competition_type": CHAMPIONS_LEAGUE.type,
                "position": p.get("position") or "Unknown",
                "shirt_number": p.get("shirtNumber"),
                "age": _age_from_dob(p.get("dateOfBirth")),
                "date_of_birth": p.get("dateOfBirth"),
                "nationality": p.get("nationality"),
                "has_risk_features": False,
                "ensemble_prob": float("nan"),
                "archetype": "Unknown",
            })
    df = pd.DataFrame(rows)
    logger.info("UCL baseline rows: %d players across %d clubs", len(df), baseline_clubs)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2025, help="CL season start year (2025 = 2025-26)")
    parser.add_argument("--club-pickle", default=str(ROOT / "models" / "inference_df.pkl"))
    parser.add_argument("--dry-run", action="store_true", help="Build and report, do not write")
    args = parser.parse_args()

    # Local convenience: load .env if the API key isn't already in the environment
    # (on Render it is set directly).
    if not os.environ.get("FOOTBALL_DATA_API_KEY"):
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    club_df = pd.read_pickle(args.club_pickle)
    logger.info("Loaded club inference_df: %d rows", len(club_df))

    covered = build_covered_rows(club_df)
    client = FootballDataClient()
    baseline = build_baseline_rows(client, args.season)

    combined = pd.concat([covered, baseline], ignore_index=True, sort=False)
    if combined.empty:
        logger.error("No UCL rows built; aborting.")
        return 1

    with_risk = int(combined["has_risk_features"].fillna(False).sum())
    print("\n=== UCL FRAME ===")
    print("total rows:", len(combined))
    print("with risk (covered):", with_risk, "| baseline:", len(combined) - with_risk)
    print("clubs:", combined["club_team"].nunique())
    print("competition_id unique:", combined["competition_id"].unique().tolist())
    cov_probs = combined.loc[combined["has_risk_features"] == True, "ensemble_prob"]
    print("covered ensemble_prob: nunique=", cov_probs.nunique(), "NaN=", int(cov_probs.isna().sum()),
          "min/max=", round(float(cov_probs.min()), 3), round(float(cov_probs.max()), 3))
    base_probs = combined.loc[combined["has_risk_features"] == False, "ensemble_prob"]
    print("baseline ensemble_prob all-NaN:", bool(base_probs.isna().all()))

    if args.dry_run:
        print("\n[DRY RUN — not writing]")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_pickle(OUT_PATH)
    print(f"\nWROTE {OUT_PATH} ({len(combined)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
