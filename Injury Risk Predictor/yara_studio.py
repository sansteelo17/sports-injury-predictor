"""
Yara Studio — local orchestrator
=================================
Single-script replacement for the Gumloop "matchday data prep" flow plus the
downstream editorial/render loop. Everything runs locally in Python:

    fetch teams -> team overviews -> flatten/normalize candidates
      -> cap top N by risk -> per-player /risk enrich
      -> OpenAI "journalist" scoring (structured output)
      -> merge + rank by story_score
      -> render PIL cards (graceful) -> write JSON artifact
      -> optional post (text + image) via yara_autoposter clients

Why not Gumloop: the Ask AI node is a lossy wrapper. Here we call the OpenAI
SDK directly with JSON output and parse tolerantly. Posting is done through the
Twitter/Reddit API clients already in yara_autoposter.py — no RPA needed.

Run:
    python yara_studio.py --league "Premier League" --top-n 30 --dry-run
    python yara_studio.py --post            # actually post the top candidate

Env:
    YARA_API_BASE      (default https://www.yaraspeaks.com/api)
    OPENAI_API_KEY     (required for the journalist step; skip with --no-journalist)
    OPENAI_MODEL       (default gpt-5)
    YARA_FONT_DIR      (Inter fonts; card degrades gracefully if absent)
    YARA_CUTOUT_DIR    (optional per-player transparent PNGs: "<name>.png")
"""

import os
import re
import sys
import json
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

import yara_autoposter as yap
from yara_card import build_risk_card

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("yara.studio")

API_BASE = os.getenv("YARA_API_BASE", "https://www.yaraspeaks.com/api")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
CUTOUT_DIR = os.getenv("YARA_CUTOUT_DIR", "")
OUTPUT_DIR = os.getenv("YARA_OUTPUT_DIR", "studio_output")

# Fields kept from /players/{name}/risk to keep the journalist payload small.
RISK_KEEP = [
    "story", "fpl_insight", "fpl_points_projection", "scoring_odds",
    "spike_flag", "acwr", "risk_comparison", "player_importance", "fpl_value",
]

JOURNALIST_PROMPT = """\
You are a sports journalist for Yara, a Premier League and La Liga injury risk \
intelligence platform. Audience: FPL managers, analysts, bettors who already \
know the basics and can smell filler.

You receive a JSON array of candidate players. Each has injury risk, ownership, \
FPL projection, fixtures, betting odds, and a short Yara story. The data is good \
but not enough alone. Risk without stakes is noise.

YOUR JOB
For every candidate, return one scored entry. Do not omit boring candidates, \
score them low.
1. Reject already-public stories (long-term absentees, fourth-choice keepers, \
players nobody owns with no template angle). Score low.
2. Score story_score 0-100: high ownership x high risk = high; sharp risk change \
either direction = high; marquee fixture (top-six, televised, title or \
relegation) = bonus; clear beneficiary elsewhere = bonus; absence changes team \
shape = bonus; low ownership, routine risk, no fixture stakes = floor.
3. story_hooks: short tags under 32 chars each, e.g. captaincy_at_risk, \
premium_doubt, template_pivot, differential_with_upside, \
bench_fodder_promoted, beneficiary_of_<name>, marquee_fixture, \
short_price_anytime_scorer, return_from_injury, form_collapse, form_spike.
4. journalist_notes: 2 sentences MAX, under 240 chars total. Terse, \
analyst-to-analyst.
5. why_short: one lowercase clause, under 100 chars.
6. sources: array of URLs you actually used. Empty array is fine.

HARD RULES
- No exclamation marks. No em dashes or en dashes. Use commas or full stops.
- No tabloid words: BREAKING, HUGE, URGENT, SHOCK, BOMBSHELL.
- player_name must exactly match a candidate. No fabrication. One entry per candidate.

OUTPUT
Return a JSON object: {"scored_players": [ ... ]}. Each item:
{"player_name": string, "story_score": integer 0-100, "story_hooks": string[], \
"why_short": string, "journalist_notes": string, "sources": string[]}
"""


# ── API ────────────────────────────────────────────────────────────────────

def _get(path: str, **params):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params or None,
                         headers={"Accept": "application/json"}, timeout=15)
        if r.status_code == 200:
            return r.json()
        log.warning("GET %s -> %s", path, r.status_code)
    except Exception as e:
        log.warning("GET %s failed: %s", path, e)
    return None


def fetch_teams(league: str) -> list[str]:
    data = _get("/teams", league=league)
    if isinstance(data, list):
        return [t if isinstance(t, str) else t.get("name") for t in data if t]
    return list(yap.EPL_TEAMS)  # fallback to the known EPL list


def fetch_overviews(teams: list[str], workers: int = 6) -> list[dict]:
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_get, f"/teams/{t}/overview"): t for t in teams}
        for fut in as_completed(futs):
            ov = fut.result()
            if ov:
                out.append(ov)
    return out


def flatten_candidates(overviews: list[dict]) -> list[dict]:
    """Flatten team overviews into candidate dicts using REAL API field names."""
    cands = []
    for ov in overviews:
        team = ov.get("team", "")
        next_fixture = ov.get("next_fixture")
        for p in ov.get("players", []):
            # Skip already-public absences (confirmed out).
            if p.get("is_currently_injured") and p.get("chance_of_playing") == 0:
                continue
            prob = float(p.get("risk_probability", 0) or 0)
            cands.append({
                "player_name": p.get("name"),
                "team": team,
                "position": p.get("position"),
                "risk_score_pct": round(prob * 100),
                "risk_probability": prob,
                "risk_level": p.get("risk_level"),
                "archetype": p.get("archetype"),
                "minutes_played": p.get("minutes_played"),
                "is_starter": p.get("is_starter"),
                "days_since_last_injury": p.get("days_since_last_injury"),
                "is_currently_injured": p.get("is_currently_injured"),
                "injury_news": p.get("injury_news"),
                "chance_of_playing": p.get("chance_of_playing"),
                "player_image_url": p.get("player_image_url"),
                "team_next_fixture": next_fixture,
            })
    cands.sort(key=lambda c: c["risk_score_pct"], reverse=True)
    return cands


def enrich_candidates(cands: list[dict], workers: int = 5) -> list[dict]:
    """Merge a slim slice of /players/{name}/risk into each candidate."""
    def fetch(c):
        risk = _get(f"/players/{c['player_name']}/risk")
        if risk:
            for k in RISK_KEEP:
                if k in risk:
                    c[k] = risk[k]
            uf = risk.get("upcoming_fixtures") or []
            c["upcoming_fixtures"] = uf[:2]
            ir = risk.get("injury_records") or []
            c["injury_records"] = ir[:5]
            ln = risk.get("lab_notes") or {}
            c["lab_notes_summary"] = ln.get("summary") if isinstance(ln, dict) else None
        return c

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fetch, cands))
    return cands


# ── Journalist (OpenAI) ──────────────────────────────────────────────────────

def score_candidates(cands: list[dict], gameweek, league: str) -> dict:
    """Call OpenAI once to score every candidate. Returns {name: scored_entry}."""
    try:
        from openai import OpenAI
    except ImportError:
        log.error("openai not installed. Run: pip install openai")
        return {}
    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set; skipping journalist step.")
        return {}

    client = OpenAI()
    user_msg = (
        f"gameweek: {gameweek}\nleague: {league}\n"
        f"candidates_json:\n{json.dumps(cands, default=str)}"
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JOURNALIST_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        log.error("Journalist call failed: %s", e)
        return {}

    scored = _parse_scored(raw)
    if not scored:
        log.warning("Journalist returned no parseable entries.")
    return {s.get("player_name"): s for s in scored if s.get("player_name")}


def _parse_scored(raw: str) -> list[dict]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        log.error("Could not parse journalist JSON (truncated?). First 200 chars: %s",
                  cleaned[:200])
        return []
    if isinstance(parsed, list):
        return parsed
    return parsed.get("scored_players", []) if isinstance(parsed, dict) else []


# ── Card rendering ───────────────────────────────────────────────────────────

def _cutout_for(name: str) -> str:
    if not CUTOUT_DIR or not name:
        return ""
    p = os.path.join(CUTOUT_DIR, f"{name}.png")
    return p if os.path.exists(p) else ""


def _signals(c: dict) -> list:
    sig = []
    acwr = c.get("acwr")
    if acwr is not None:
        flagged = c.get("spike_flag")
        sig.append(("LOAD INDEX (ACWR)", f"{float(acwr):.2f}",
                    "Spike" if flagged else "Normal",
                    "danger" if flagged else "muted"))
    nf = c.get("team_next_fixture") or {}
    if isinstance(nf, dict) and nf.get("opponent"):
        ha = "(H)" if nf.get("is_home") else "(A)"
        sig.append(("NEXT MATCH", f"{nf['opponent']} {ha}", "Fixture", "muted"))
    dsi = c.get("days_since_last_injury")
    if dsi is not None and dsi < 365:
        sig.append(("LAST INJURY", f"{dsi} days ago", "Recent", "amber"))
    proj = c.get("fpl_points_projection") or {}
    if isinstance(proj, dict) and proj.get("expected_points") is not None:
        sig.append(("FPL PROJECTION", f"{proj['expected_points']:.1f} pts",
                    proj.get("confidence", ""), "accent"))
    return sig


def render_card(c: dict, gameweek, league: str) -> str:
    name = c.get("player_name") or "unknown"
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    out = os.path.join(OUTPUT_DIR, f"{safe}.png")
    imp = c.get("player_importance") or {}
    own = imp.get("ownership_pct") if isinstance(imp, dict) else None
    strip = f"{own:.1f}% ownership" if isinstance(own, (int, float)) else ""
    data = {
        "name": name,
        "team": c.get("team", ""),
        "position": c.get("position", ""),
        "league": league,
        "gw": gameweek,
        "risk_pct": c.get("risk_score_pct", 0),
        "archetype": c.get("archetype", ""),
        "cutout_path": _cutout_for(name),
        "signals": _signals(c),
        "strip_text": strip,
    }
    return build_risk_card(data, out)


# ── Post text ────────────────────────────────────────────────────────────────

def build_post_text(c: dict, gameweek) -> str:
    name = c["player_name"]
    pct = c.get("risk_score_pct", 0)
    emoji = yap.get_risk_emoji((pct or 0) / 100.0)
    why = (c.get("scored", {}).get("why_short") or c.get("story") or "").strip()
    why = why.rstrip(".")
    lines = [
        f"{emoji} GW{gameweek} INJURY WATCH",
        f"{name} ({c.get('team','')}) — {pct}% injury risk",
    ]
    if why:
        lines.append(why[:180])
    lines.append("yaraspeaks.com")
    return "\n".join(lines)


# ── Orchestration ────────────────────────────────────────────────────────────

def run(league: str, top_n: int, gameweek, do_journalist: bool,
        do_render: bool, post: bool, dry_run: bool):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if gameweek is None:
        gameweek = yap.get_current_gameweek()

    log.info("Fetching teams for %s ...", league)
    teams = fetch_teams(league)
    log.info("%d teams", len(teams))

    log.info("Fetching team overviews ...")
    overviews = fetch_overviews(teams)
    cands = flatten_candidates(overviews)
    log.info("%d candidates after flatten", len(cands))

    cands = cands[:top_n]
    log.info("Enriching top %d ...", len(cands))
    enrich_candidates(cands)

    if do_journalist:
        log.info("Scoring with %s ...", OPENAI_MODEL)
        scores = score_candidates(cands, gameweek, league)
        for c in cands:
            s = scores.get(c["player_name"])
            if s:
                c["scored"] = s
                c["story_score"] = s.get("story_score", 0)
        cands.sort(key=lambda c: c.get("story_score", 0), reverse=True)
    else:
        for c in cands:
            c["story_score"] = c.get("risk_score_pct", 0)

    if do_render:
        log.info("Rendering cards ...")
        for c in cands[:5]:
            try:
                c["card_path"] = render_card(c, gameweek, league)
            except Exception as e:
                log.warning("Card render failed for %s: %s", c.get("player_name"), e)

    artifact = os.path.join(OUTPUT_DIR, "candidates.json")
    with open(artifact, "w") as f:
        json.dump(cands, f, indent=2, default=str)
    log.info("Wrote %s (%d candidates)", artifact, len(cands))

    if post and cands:
        top = cands[0]
        text = build_post_text(top, gameweek)
        card = top.get("card_path")
        if dry_run:
            log.info("[DRY RUN] Would post (card=%s):\n%s", card, text)
        else:
            _post_with_card(text, card)

    return cands


def _post_with_card(text: str, card_path: str | None):
    """Post text + image to Twitter (image via v1.1 media upload) and Reddit."""
    if card_path and os.path.exists(card_path) and not yap.DRY_RUN:
        try:
            import tweepy
            auth = tweepy.OAuth1UserHandler(
                os.getenv("TWITTER_API_KEY"), os.getenv("TWITTER_API_SECRET"),
                os.getenv("TWITTER_ACCESS_TOKEN"), os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
            )
            api_v1 = tweepy.API(auth)
            media = api_v1.media_upload(card_path)
            client = yap.get_twitter_client()
            client.create_tweet(text=text[:280], media_ids=[media.media_id])
            log.info("Posted tweet with card image.")
        except Exception as e:
            log.error("Image tweet failed (%s); falling back to text.", e)
            yap.post_twitter(text)
    else:
        yap.post_twitter(text)
    yap.post_reddit(title=text.split("\n")[1] if "\n" in text else text[:120], body=text)


def main():
    ap = argparse.ArgumentParser(description="Yara Studio local orchestrator")
    ap.add_argument("--league", default="Premier League")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--gameweek", type=int, default=None)
    ap.add_argument("--no-journalist", action="store_true", help="skip OpenAI scoring")
    ap.add_argument("--no-render", action="store_true", help="skip card rendering")
    ap.add_argument("--post", action="store_true", help="post the top candidate")
    ap.add_argument("--dry-run", action="store_true", help="log posts instead of sending")
    args = ap.parse_args()

    run(
        league=args.league,
        top_n=args.top_n,
        gameweek=args.gameweek,
        do_journalist=not args.no_journalist,
        do_render=not args.no_render,
        post=args.post,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
