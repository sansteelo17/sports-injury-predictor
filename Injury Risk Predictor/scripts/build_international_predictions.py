"""Build the international (World Cup 2026) inference DataFrame.

Strategy: a World Cup player's *injury risk* is driven almost entirely by their
club season (acute/chronic workload, prior injuries, recent congestion). The
trained model already encodes that signal — so we reuse each player's club
inference row as the basis, overlay tournament identity (national team, group,
next fixture, caps), and override the competition + venue fields. We do NOT
re-train. We DO renormalise risk within the tournament cohort downstream
(handled by ``_competition_prob_series`` in the API).

Players with no matching club row are skipped: predicting injury risk from
position + age alone is worse than admitting we don't know. The skipped count
is logged so we can decide whether to add Transfermarkt-based fallbacks in a
follow-up.

Run:
    python scripts/build_international_predictions.py

Output:
    data/processed/inference_international_world_cup_2026.pkl
    data/processed/world_cup_2026_fixtures.pkl
    data/processed/world_cup_2026_groups.json
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.competitions import WORLD_CUP_2026  # noqa: E402
from src.data_loaders.international_squads import (  # noqa: E402
    WorldCupClient,
    WORLD_CUP_2026_SEASON,
    build_groups_map,
    next_fixture_for_country,
)
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def _norm_name(name: str) -> str:
    """Lowercase + accent-strip — matches the join key used by api/main.py."""
    if not isinstance(name, str):
        return ""
    stripped = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    return stripped.lower().strip()


def join_squads_to_club_rows(
    squads_df: pd.DataFrame,
    club_inference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner join on normalised name. One row per (club row, national team)
    assignment so a player only appears once per tournament."""
    if squads_df.empty:
        return pd.DataFrame()
    squads = squads_df.copy()
    club = club_inference_df.copy()
    squads["_join_name"] = squads["name"].apply(_norm_name)
    club["_join_name"] = club["name"].apply(_norm_name)

    # Drop duplicates on the squad side (a player should only appear in one
    # national team) and on the club side keep the highest-minutes row.
    squads = squads.drop_duplicates(subset=["_join_name"], keep="first")
    if "minutes_played" in club.columns:
        club = club.sort_values("minutes_played", ascending=False)
    club = club.drop_duplicates(subset=["_join_name"], keep="first")

    merged = squads.merge(club, on="_join_name", how="inner", suffixes=("_intl", "_club"))
    logger.info(
        "Joined %d/%d international squad players to club inference rows",
        len(merged),
        len(squads),
    )
    return merged


def build_international_inference(
    merged: pd.DataFrame,
    groups: Dict[str, str],
    fixtures_df: pd.DataFrame,
) -> pd.DataFrame:
    """Project the merged frame into the inference_df shape, with WC overlay."""
    if merged.empty:
        return pd.DataFrame()

    rows = []
    for _, src in merged.iterrows():
        row = src.to_dict()
        # Resolve the canonical name + position from the squad side; the rest
        # of the features (acwr, ensemble_prob, …) come from the club row.
        name = row.get("name_intl") or row.get("name")
        national_team = row.get("national_team")
        nxt = next_fixture_for_country(fixtures_df, national_team) or {}

        out = {k: v for k, v in row.items() if not k.endswith("_intl") and k != "_join_name"}
        # Overrides for tournament identity. The club row's ``team`` / ``league``
        # are preserved as ``club_team`` / ``club_league`` so narrative can say
        # "his City form" while the row routes as a national-team entry.
        out["name"] = name
        out["club_team"] = row.get("team_club") or row.get("team")
        out["club_league"] = row.get("league_club") or row.get("league")
        out["team"] = national_team
        out["player_team"] = national_team
        out["league"] = WORLD_CUP_2026.name
        out["competition_id"] = WORLD_CUP_2026.id
        out["competition_type"] = WORLD_CUP_2026.type
        out["position"] = row.get("position_intl") or row.get("position")
        if row.get("shirt_number_intl") is not None:
            out["shirt_number"] = row["shirt_number_intl"]
        out["nationality"] = row.get("nationality_intl") or row.get("nationality")
        out["national_group"] = groups.get(national_team)
        out["next_intl_opponent"] = nxt.get("opponent")
        out["next_intl_is_home"] = nxt.get("is_home")
        out["next_intl_utc_date"] = nxt.get("utc_date")
        out["next_intl_stage"] = nxt.get("stage")
        rows.append(out)

    df = pd.DataFrame(rows)
    logger.info("Built %d international inference rows", len(df))
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=WORLD_CUP_2026_SEASON)
    parser.add_argument(
        "--club-pickle",
        default=str(ROOT / "models" / "inference_df.pkl"),
        help="Source club inference_df to join injury features from",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "processed"),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = WorldCupClient()
    logger.info("Fetching World Cup %s fixtures...", args.season)
    fixtures_df = client.get_fixtures(season=args.season)
    groups = build_groups_map(fixtures_df)
    fixtures_path = out_dir / f"world_cup_{args.season}_fixtures.pkl"
    fixtures_df.to_pickle(fixtures_path)
    logger.info("Wrote %s (%d fixtures)", fixtures_path, len(fixtures_df))

    groups_path = out_dir / f"world_cup_{args.season}_groups.json"
    groups_path.write_text(json.dumps(groups, indent=2, ensure_ascii=False))
    logger.info("Wrote %s (%d teams placed in groups)", groups_path, len(groups))

    logger.info("Fetching World Cup squads...")
    squads_df = client.get_squads(season=args.season)
    if squads_df.empty:
        logger.warning(
            "No squad rows returned. Tournament squads typically lock 7 days "
            "before kickoff; rerun closer to the tournament. Wrote fixtures + "
            "groups only."
        )
        return 0

    club_df = pd.read_pickle(args.club_pickle)
    logger.info("Loaded club inference_df: %d rows", len(club_df))
    merged = join_squads_to_club_rows(squads_df, club_df)
    intl_df = build_international_inference(merged, groups, fixtures_df)

    if intl_df.empty:
        logger.warning("Joined frame is empty; nothing to write.")
        return 0

    out_path = out_dir / f"inference_international_world_cup_{args.season}.pkl"
    intl_df.to_pickle(out_path)
    logger.info("Wrote %s (%d players, %d national teams)",
                out_path, len(intl_df), intl_df["team"].nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
