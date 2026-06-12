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

from . import (archetype, battle, config, copywriter, editorial, emailer, fetch,
               memory, photos, render, wc_schedule)

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
# is a tournament-wide pool on a raw (differentiated) risk scale.
FORMAT_SPEC = {
    "matchday_board": {"count": 5, "day_scoped": True, "pool": 50, "scale": "normalize"},
    "riskiest_xi": {"count": 11, "day_scoped": False, "pool": 60, "scale": "raw"},
}


def _run_battle(args) -> int:
    """Marquee head-to-head: next big fixture, top-risk star from each side."""
    league = args.league or LEAGUE_NAMES.get(args.competition, args.competition)
    print(f"[run] battle_card | {league} | API={config.API_BASE}")
    data = battle.build()
    if not data:
        print("[run] no marquee fixture / players found. Nothing built.")
        return 0
    fx, a, b = data["fixture"], data["left"], data["right"]
    for p in (a, b):
        if not p.get("image_url"):
            p["image_url"] = photos.wikipedia_photo(p.get("player_name", ""))
        p["signal_one_liner"] = p.get("x_factor", "")
    payload = {
        "post_type": "battle_card",
        "label": f"Battle Card · {fx['stage']}",
        "title": f"{fx['home']} vs {fx['away']}",
        "when": fx["utc"].strftime("%a %d %b"),
        "league": f"{league} · {fx['stage']}",
        "left": a, "right": b,
        "footnote": "Injury risk probabilities are model outputs, not medical advice",
        # For the copywriter.
        "top_5": [a, b],
        "matchday": fx["stage"],
        "narrative_spine": (f"{a['player_name']} and {b['player_name']} both carry real "
                            f"injury risk into {fx['home']} against {fx['away']}."),
    }
    print(f"[battle] {fx['home']} ({a['player_name']} {a['risk_score_pct']}%) vs "
          f"{fx['away']} ({b['player_name']} {b['risk_score_pct']}%)")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = config.OUT_DIR / f"battle_card_{args.competition}_{stamp}.png"
    render.render_card("battle_card", payload, out_png)
    print(f"[render] {out_png}")
    x_post, reddit_post = copywriter.write(payload)
    print("\n--- X POST (" + str(len(x_post)) + " chars) ---\n" + x_post)
    print("\n--- REDDIT ---\n" + reddit_post + "\n")
    if args.dry_run:
        print(f"[dry-run] no email sent. card at {out_png}")
        return 0
    emailer.send_draft(payload, x_post, reddit_post, out_png)
    return 0


def _run_archetype(args) -> int:
    """Explain one archetype and show this week's marquee examples."""
    league = args.league or LEAGUE_NAMES.get(args.competition, args.competition)
    print(f"[run] archetype | {league} | API={config.API_BASE}")
    data = archetype.build(args.competition)
    if not data:
        print("[run] no archetype with enough examples. Nothing to post.")
        return 0
    payload = {
        "post_type": "archetype", "league": league, "subtitle": "This week's vocabulary",
        "archetype": data["archetype"], "description": data["description"],
        "triggers": data["triggers"], "examples": data["examples"],
        "top_5": data["examples"],
        "narrative_spine": f"This week's archetype is {data['archetype']}. {data['description']}",
    }
    print(f"[archetype] {data['archetype']}: "
          + ", ".join(e["player_name"] for e in data["examples"]))
    return _emit(args, "archetype", payload)


def _emit(args, fmt: str, payload: Dict) -> int:
    """Shared tail: render, write copy, dry-run or email, record the post."""
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = config.OUT_DIR / f"{fmt}_{args.competition}_{stamp}.png"
    render.render_card(fmt, payload, out_png)
    print(f"[render] {out_png}")
    x_post, reddit_post = copywriter.write(payload)
    print("\n--- X POST (" + str(len(x_post)) + " chars) ---\n" + x_post)
    print("\n--- REDDIT ---\n" + reddit_post + "\n")
    if args.dry_run:
        print(f"[dry-run] no email sent. card at {out_png}")
        return 0
    emailer.send_draft(payload, x_post, reddit_post, out_png)
    for p in payload.get("top_5", []):
        memory.record_post(p.get("player_name", ""), fmt)
    print(f"[memory] recorded {fmt} post")
    return 0


def _run_spike(args) -> int:
    """Players whose risk jumped 15+ points since their last logged reading."""
    league = args.league or LEAGUE_NAMES.get(args.competition, args.competition)
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    print(f"[run] risk_spike | {league} | API={config.API_BASE}")
    cands = fetch.fetch_candidates(args.competition, pool=100)
    spikers = []
    for c in cands:
        prev = memory.last_risk(c["player_name"], args.competition, today)
        cur = c.get("risk_score_pct")
        if prev and cur is not None and (cur - prev[0]) >= 15:
            if memory.on_cooldown(c["player_name"], "risk_spike"):
                continue
            d = dict(c)
            d["delta_pct"] = cur - prev[0]
            d["signal_one_liner"] = (c.get("injury_news")
                                     or f"Up {cur - prev[0]} points since the last read.")
            spikers.append(d)
    if not spikers:
        print("[run] no risk spikes (>=15) since last reading. Nothing to post.")
        return 0
    spikers.sort(key=lambda p: p["delta_pct"], reverse=True)
    spikers = spikers[:5]
    for p in spikers:
        if not p.get("image_url"):
            p["image_url"] = photos.wikipedia_photo(p.get("player_name", ""))
    payload = {
        "post_type": "risk_spike", "matchday": "This week", "league": league,
        "title": "Biggest Risk Risers",
        "subtitle": "Sharpest week-over-week jumps in injury risk",
        "top_5": spikers,
        "narrative_spine": (f"{spikers[0]['player_name']} leads the risers, up "
                            f"{spikers[0]['delta_pct']} points since the last read."),
    }
    print("[spike] " + ", ".join(f"{p['player_name']} +{p['delta_pct']}" for p in spikers))
    return _emit(args, "risk_spike", payload)


def _run_accountability(args) -> int:
    """Score the calls logged for a past matchday and post hits and misses."""
    league = args.league or LEAGUE_NAMES.get(args.competition, args.competition)
    date = args.date or (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"[run] accountability | {league} | scoring {date} | API={config.API_BASE}")
    calls = memory.pending_calls(args.competition, date)
    if not calls:
        print(f"[run] no pending calls for {date}. Nothing to score.")
        return 0
    # Current status of the called players decides the outcome.
    status = {c["player_name"].lower(): c
              for c in fetch.fetch_candidates(args.competition, pool=100, date=date)}
    hits, misses = [], []
    for call in calls:
        nm = call["player_name"]
        pred = call["predicted_risk"] or 0
        cur = status.get((nm or "").lower(), {})
        # Outcome from the news agent: did a post-match report flag an injury?
        # Same grounded, attributed feed that powers the pre-match narratives.
        news = fetch.player_news(nm, cur.get("team", ""))
        flagged = news.get("injury_signal", False)
        matched = news.get("matched") or {}
        landed = flagged if pred >= 60 else (not flagged)  # high call lands if it broke down
        if flagged and matched:
            outcome = "flagged after the game"
            note = (f"{matched.get('source', '')}: {matched.get('headline', '')}"
                    ).strip(": ")[:120]
        else:
            outcome = "came through clean"
            note = "No injury news surfaced after the game."
        memory.score_call(call["id"], outcome, landed)
        rec = {"player_name": nm, "team": cur.get("team", ""),
               "predicted_risk": pred, "outcome": outcome, "note": note}
        (hits if landed else misses).append(rec)
    total = len(hits) + len(misses)
    payload = {
        "post_type": "accountability", "matchday": date, "league": league,
        "score": len(hits), "out_of": total, "hits": hits, "misses": misses,
        "top_5": hits + misses,
        "narrative_spine": f"{len(hits)} of {total} calls landed on {date}.",
    }
    print(f"[accountability] {len(hits)}/{total} landed")
    return _emit(args, "accountability", payload)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", default="matchday_board",
                    choices=["matchday_board", "riskiest_xi", "battle_card",
                             "risk_spike", "accountability", "archetype"])
    ap.add_argument("--competition", default="world-cup-2026")
    ap.add_argument("--matchday", default="auto",
                    help="round label; 'auto' derives the WC stage from the fixtures "
                         "(Group Stage -> Round of 32 -> ... -> Final)")
    ap.add_argument("--league", default=None, help="display name; defaults from competition")
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD. International boards restrict to teams playing that day. Default: today (UTC).")
    ap.add_argument("--all-day", action="store_true", help="ignore the day filter (whole competition pool)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Auto stage label so the tournament switches itself, no manual --matchday.
    if args.matchday == "auto":
        if args.competition == "world-cup-2026":
            ref = args.date or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
            args.matchday = wc_schedule.stage_label_for(ref)
        else:
            args.matchday = "Matchday"

    if args.format == "battle_card":
        return _run_battle(args)
    if args.format == "risk_spike":
        return _run_spike(args)
    if args.format == "accountability":
        return _run_accountability(args)
    if args.format == "archetype":
        return _run_archetype(args)

    spec = FORMAT_SPEC[args.format]
    league = args.league or LEAGUE_NAMES.get(args.competition, args.competition)
    if spec["day_scoped"] and not args.all_day:
        date = args.date or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    else:
        date = None

    print(f"[run] {args.format} | {league} | {args.matchday} | day={date or 'all'} | API={config.API_BASE}")
    candidates = fetch.fetch_candidates(args.competition, pool=spec["pool"],
                                        date=date, scale=spec["scale"])
    print(f"[fetch] {len(candidates)} risk-featured candidates"
          + (f" playing {date}" if date else ""))

    payload = editorial.select(candidates, args.matchday, league,
                               count=spec["count"], post_type=args.format)
    if not payload.get("should_post"):
        print("[run] should_post=false (nobody playing / no candidates). Nothing rendered or sent.")
        return 0

    # Pad to the format's count if the editorial under-delivered, using the next
    # highest-risk candidates not already picked (keeps the marquee at the top).
    picked = {str(p.get("player_name", "")).lower() for p in payload["top_5"]}
    for c in candidates:
        if len(payload["top_5"]) >= spec["count"]:
            break
        if str(c["player_name"]).lower() not in picked:
            payload["top_5"].append(dict(c))
            picked.add(str(c["player_name"]).lower())

    # Merge photo + club back in by name so the card can render the image.
    by_name = {str(c["player_name"]).lower(): c for c in candidates}
    for p in payload["top_5"]:
        c = by_name.get(str(p.get("player_name", "")).lower())
        if c:
            p["image_url"] = c.get("image_url")
            p.setdefault("team", c.get("team"))
            p.setdefault("risk_score_pct", c.get("risk_score_pct"))
            p.setdefault("position", c.get("position"))
        # Broad fallback for players the FPL/TM source misses, cached.
        if not p.get("image_url"):
            p["image_url"] = photos.wikipedia_photo(p.get("player_name", ""))
    # Ranked by injury probability - enforce strict order.
    payload["top_5"].sort(key=lambda p: p.get("risk_score_pct", 0), reverse=True)

    # Real week-over-week delta from memory (powers the board's wk column and
    # the spike trigger). None until we have a prior reading for the player.
    log_date = date or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    for p in payload["top_5"]:
        prev = memory.last_risk(p.get("player_name", ""), args.competition, log_date)
        cur = p.get("risk_score_pct")
        p["delta_pct"] = (cur - prev[0]) if (prev and cur is not None) else None

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
    # Remember this post: risk history for deltas, calls for accountability,
    # and the per-player cooldown.
    memory.log_board(payload["top_5"], args.competition, log_date, args.format)
    memory.record_calls(payload["top_5"], args.competition, log_date)
    for p in payload["top_5"]:
        memory.record_post(p.get("player_name", ""), args.format)
    print(f"[memory] logged {len(payload['top_5'])} picks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
