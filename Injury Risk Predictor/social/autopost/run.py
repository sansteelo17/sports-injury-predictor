"""Orchestrator. One entry point per fired trigger.

    python -m social.autopost.run --format matchday_board \
        --competition world-cup-2026 --matchday MD1 [--dry-run]

--dry-run renders the card + prints the copy, sends no email. Drop --dry-run to
email the draft to DRAFT_EMAIL_TO.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from . import config, copywriter, editorial, emailer, fetch, photos, render

LEAGUE_NAMES = {
    "world-cup-2026": "FIFA World Cup 2026",
    "premier-league": "Premier League",
    "la-liga": "La Liga",
    "bundesliga": "Bundesliga",
    "serie-a": "Serie A",
    "ligue-1": "Ligue 1",
    "champions-league": "UEFA Champions League",
}

# Per-format config. day_scoped boards restrict to who plays that date; the XI
# is a tournament-wide pool.
FORMAT_SPEC = {
    "matchday_board": {"count": 5, "day_scoped": True, "pool": 50},
    "riskiest_xi": {"count": 11, "day_scoped": False, "pool": 60},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", default="matchday_board",
                    choices=["matchday_board", "riskiest_xi"])
    ap.add_argument("--competition", default="world-cup-2026")
    ap.add_argument("--matchday", default="MD1", help="round label e.g. MD1, R16, GW1")
    ap.add_argument("--league", default=None, help="display name; defaults from competition")
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD. International boards restrict to teams playing that day. Default: today (UTC).")
    ap.add_argument("--all-day", action="store_true", help="ignore the day filter (whole competition pool)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = FORMAT_SPEC[args.format]
    league = args.league or LEAGUE_NAMES.get(args.competition, args.competition)
    if spec["day_scoped"] and not args.all_day:
        date = args.date or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    else:
        date = None

    print(f"[run] {args.format} | {league} | {args.matchday} | day={date or 'all'} | API={config.API_BASE}")
    candidates = fetch.fetch_candidates(args.competition, pool=spec["pool"], date=date)
    print(f"[fetch] {len(candidates)} risk-featured candidates"
          + (f" playing {date}" if date else ""))

    payload = editorial.select(candidates, args.matchday, league,
                               count=spec["count"], post_type=args.format)
    if not payload.get("should_post"):
        print("[run] should_post=false (nobody playing / no candidates). Nothing rendered or sent.")
        return 0

    # The editorial returns its own player dicts; merge the photo + club back in
    # by name so the card can render the image.
    by_name = {str(c["player_name"]).lower(): c for c in candidates}
    for p in payload["top_5"]:
        c = by_name.get(str(p.get("player_name", "")).lower())
        if c:
            p["image_url"] = c.get("image_url")
            p.setdefault("team", c.get("team"))
        # Broad fallback for players the FPL/TM source misses (Liga MX, MLS,
        # Serie A, etc.) so the card isn't half silhouettes, cached.
        if not p.get("image_url"):
            p["image_url"] = photos.wikipedia_photo(p.get("player_name", ""))
    # Ranked by injury probability - enforce strict order.
    payload["top_5"].sort(key=lambda p: p.get("risk_score_pct", 0), reverse=True)

    if args.format == "riskiest_xi":
        payload["title"] = "Riskiest XI"
        payload["subtitle"] = f"Eleven players carrying the most risk into {args.matchday}"
        payload["takeaway"] = payload.get("narrative_spine", "")

    print("[editorial] picked: " + ", ".join(
        f"{p['player_name']} {p['risk_score_pct']}%" for p in payload["top_5"]))

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = config.OUT_DIR / f"{args.format}_{args.competition}_{stamp}.png"
    render.render_card(args.format, payload, out_png)
    print(f"[render] {out_png}")

    x_post, reddit_post = copywriter.write(payload)
    print("\n--- X POST (" + str(len(x_post)) + " chars) ---\n" + x_post)
    print("\n--- REDDIT ---\n" + reddit_post + "\n")

    if args.dry_run:
        print(f"[dry-run] no email sent. card at {out_png}")
        return 0

    emailer.send_draft(payload, x_post, reddit_post, out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
