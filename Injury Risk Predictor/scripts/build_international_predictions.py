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
from typing import Dict, List, Optional

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
from src.data_loaders.national_team_caps import build_caps_lookup  # noqa: E402
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
        out["has_risk_features"] = True
        rows.append(out)

    df = pd.DataFrame(rows)
    logger.info("Built %d international inference rows (with risk features)", len(df))
    return df


def build_baseline_rows(
    squads_df: pd.DataFrame,
    joined_names: set,
    groups: Dict[str, str],
    fixtures_df: pd.DataFrame,
) -> pd.DataFrame:
    """Identity-only rows for squad players with no club inference row.

    These let WC team views render every player on the announced 26-man squad
    instead of only the ones playing in PL/La Liga. Risk features are NaN —
    the model has no signal for an MLS or Saudi Pro League player — and the
    API surfaces ``risk_level="Unknown"`` for these rows.
    """
    if squads_df.empty:
        return pd.DataFrame()
    unmatched = squads_df[~squads_df["name"].apply(_norm_name).isin(joined_names)]
    rows = []
    for _, p in unmatched.iterrows():
        national_team = p.get("national_team")
        nxt = next_fixture_for_country(fixtures_df, national_team) or {}
        rows.append(
            {
                "name": p.get("name"),
                "team": national_team,
                "player_team": national_team,
                "club_team": None,
                "club_league": None,
                "league": WORLD_CUP_2026.name,
                "competition_id": WORLD_CUP_2026.id,
                "competition_type": WORLD_CUP_2026.type,
                "position": p.get("position") or "Unknown",
                "shirt_number": p.get("shirt_number"),
                "age": p.get("age"),
                # Carry DOB so the API can derive age consistently (and as a
                # fallback when the squad age column is missing).
                "date_of_birth": p.get("date_of_birth"),
                "nationality": p.get("nationality"),
                "national_group": groups.get(national_team),
                "next_intl_opponent": nxt.get("opponent"),
                "next_intl_is_home": nxt.get("is_home"),
                "next_intl_utc_date": nxt.get("utc_date"),
                "next_intl_stage": nxt.get("stage"),
                # Mark the row so the API knows to surface "Unknown" risk and
                # skip these from the percentile cohort.
                "has_risk_features": False,
                "ensemble_prob": float("nan"),
                "archetype": "Unknown",
            }
        )
    df = pd.DataFrame(rows)
    logger.info("Built %d baseline (identity-only) WC rows for unmatched squad players", len(df))
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
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse on-disk fixtures + squads pickles instead of hitting the API. "
             "Useful while iterating on the join/overlay logic.",
    )
    parser.add_argument(
        "--enrich-caps",
        action="store_true",
        help="Scrape Wikipedia 'Current squad' tables for caps + international "
             "goals (~30s for 48 countries, cached 24h). Adds caps / intl_goals.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixtures_path = out_dir / f"world_cup_{args.season}_fixtures.pkl"
    groups_path = out_dir / f"world_cup_{args.season}_groups.json"
    squads_path = out_dir / f"world_cup_{args.season}_squads.pkl"

    client: Optional[WorldCupClient] = None

    if args.reuse_cache and fixtures_path.exists():
        fixtures_df = pd.read_pickle(fixtures_path)
        groups = json.loads(groups_path.read_text()) if groups_path.exists() else build_groups_map(fixtures_df)
        logger.info("Reusing cached fixtures (%d rows) + groups (%d teams)", len(fixtures_df), len(groups))
    else:
        client = WorldCupClient()
        logger.info("Fetching World Cup %s fixtures...", args.season)
        fixtures_df = client.get_fixtures(season=args.season)
        groups = build_groups_map(fixtures_df)
        fixtures_df.to_pickle(fixtures_path)
        logger.info("Wrote %s (%d fixtures)", fixtures_path, len(fixtures_df))
        groups_path.write_text(json.dumps(groups, indent=2, ensure_ascii=False))
        logger.info("Wrote %s (%d teams placed in groups)", groups_path, len(groups))

    if args.reuse_cache and squads_path.exists():
        squads_df = pd.read_pickle(squads_path)
        logger.info("Reusing cached squads (%d players)", len(squads_df))
    else:
        if client is None:
            client = WorldCupClient()
        logger.info("Fetching World Cup squads (rate-limited ~6.5s per team)...")
        squads_df = client.get_squads(season=args.season)
        if not squads_df.empty:
            squads_df.to_pickle(squads_path)
            logger.info("Cached squads to %s", squads_path)
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
    joined_names = set(intl_df["name"].apply(_norm_name)) if not intl_df.empty else set()
    baseline_df = build_baseline_rows(squads_df, joined_names, groups, fixtures_df)

    if intl_df.empty and baseline_df.empty:
        logger.warning("No rows produced; nothing to write.")
        return 0

    combined = pd.concat([intl_df, baseline_df], ignore_index=True, sort=False)

    if args.enrich_caps and not combined.empty:
        countries = sorted(combined["team"].dropna().unique().tolist())
        logger.info("Enriching caps from Wikipedia (%d countries)...", len(countries))
        caps_lookup = build_caps_lookup(countries)
        if caps_lookup:
            # Per-country fallback indices so name surface drift between
            # football-data.org and Wikipedia still joins:
            #   - ``full``: exact normalised string
            #   - ``firstlast``: first + last token, drops middle names
            #   - ``last``: last token (surname) — only for unambiguous surnames
            #   - ``token_bag``: tokens sorted as a tuple — handles Korean/Japanese
            #     order swaps ("Kim Seung-gyu" vs "Seung-Gyu Kim")
            by_country: Dict[str, Dict[str, Dict]] = {}
            for (country, full_norm), stats in caps_lookup.items():
                bucket = by_country.setdefault(country, {"full": {}, "last": {}, "firstlast": {}, "bag": {}})
                bucket["full"][full_norm] = stats
                tokens = full_norm.split()
                if tokens:
                    last = tokens[-1]
                    if len(last) >= 4:
                        bucket["last"][last] = None if last in bucket["last"] else stats
                if len(tokens) >= 2:
                    bucket["firstlast"][f"{tokens[0]} {tokens[-1]}"] = stats
                bag_key = tuple(sorted(tokens))
                if bag_key:
                    bucket["bag"][bag_key] = None if bag_key in bucket["bag"] else stats

            def _lookup(row):
                country = row["team"]
                bucket = by_country.get(country)
                if not bucket:
                    return {}
                full = _norm_name(row["name"])
                if full in bucket["full"]:
                    return bucket["full"][full]
                tokens = full.split()
                bag_key = tuple(sorted(tokens))
                if bag_key in bucket["bag"] and bucket["bag"][bag_key] is not None:
                    return bucket["bag"][bag_key]
                if len(tokens) >= 2:
                    fl = f"{tokens[0]} {tokens[-1]}"
                    if fl in bucket["full"]:
                        return bucket["full"][fl]
                    if fl in bucket["firstlast"]:
                        return bucket["firstlast"][fl]
                    last = tokens[-1]
                    cand = bucket["last"].get(last)
                    if cand:
                        return cand
                return {}

            stats = combined.apply(_lookup, axis=1)
            combined["caps"] = [s.get("caps") for s in stats]
            combined["intl_goals"] = [s.get("intl_goals") for s in stats]
            matched = int(combined["caps"].notna().sum())
            goals_matched = int(combined["intl_goals"].notna().sum())
            logger.info("Caps enrichment matched %d/%d players (intl_goals: %d)",
                        matched, len(combined), goals_matched)
        else:
            logger.warning("Caps lookup is empty — skipping enrichment columns")

    out_path = out_dir / f"inference_international_world_cup_{args.season}.pkl"
    combined.to_pickle(out_path)
    with_risk = int(combined["has_risk_features"].sum()) if "has_risk_features" in combined.columns else len(intl_df)
    logger.info(
        "Wrote %s (%d total players, %d national teams, %d with risk features, %d baseline)",
        out_path,
        len(combined),
        combined["team"].nunique(),
        with_risk,
        len(combined) - with_risk,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
