"""
Yara Signal → Draft Post  —  7-Node Workflow
=============================================
Node 1  HTTP Fetcher      yaraspeaks.com API  (team → players list)
Node 2  JSON Parser       splits response into individual player dicts  (Loop)
Node 3  Airtable Updater  long-term memory, upsert per player
Node 4  AI Reasoning      GPT-4o finds the story angle
Node 5  Image Generator   Playwright renders 1600×900 PNG (3 templates)
Node 6  AI Copywriting    GPT-4o drafts post in Yara voice
Node 7  Notion            saves X + Reddit drafts for human approval

Required env vars:
    AIRTABLE_API_KEY
    OPENAI_API_KEY
    NOTION_API_KEY
    NOTION_DATABASE_ID
    NOTION_REDDIT_DB_ID     (optional, falls back to NOTION_DATABASE_ID)

yaraspeaks.com is a public API — no key required.
Airtable AI Assist handles signal generation on stored records automatically.

Run:
    python3 yara_workflow.py --trigger matchday_board
    python3 yara_workflow.py --trigger risk_spike   --player "Mohamed Salah"
    python3 yara_workflow.py --trigger captain_advisory
    python3 yara_workflow.py --trigger lineup_reaction --player "Bukayo Saka"
    python3 yara_workflow.py --trigger we_called_it    --player "Kevin De Bruyne"
    python3 yara_workflow.py --trigger halftime_reaction --player "Erling Haaland"
"""

import os, sys, json, argparse, requests
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from scripts.node3_risk_card      import build_risk_card
from scripts.node3_matchday_board import build_matchday_board, build_ufc_card

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Clients ─────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ── Airtable constants ───────────────────────────────────────────────────────
AT_BASE_ID     = "appeonHoOAtZhnqHf"
AT_PLAYERS_TBL = "tblNU8EkUkIH1ZxWl"
AT_KEY         = os.getenv("AIRTABLE_API_KEY", "")
AT_BASE_URL    = f"https://api.airtable.com/v0/{AT_BASE_ID}"
AT_HEADERS     = {"Authorization": f"Bearer {AT_KEY}",
                  "Content-Type": "application/json"}

# ── yaraspeaks API — public, no auth ────────────────────────────────────────
YARA_API = "https://api.yaraspeaks.com/api"

# ── Notion ───────────────────────────────────────────────────────────────────
NOTION_KEY       = os.getenv("NOTION_API_KEY", "")
NOTION_DB        = os.getenv("NOTION_DATABASE_ID", "")
NOTION_REDDIT_DB = os.getenv("NOTION_REDDIT_DB_ID", NOTION_DB)
NOTION_HDR       = {"Authorization": f"Bearer {NOTION_KEY}",
                    "Notion-Version": "2022-06-28",
                    "Content-Type": "application/json"}

# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — HTTP Fetcher
# Pulls live data from yaraspeaks.com. Team lookup first, then players.
# ─────────────────────────────────────────────────────────────────────────────

TRIGGER_TEAMS = {
    "matchday_board":   ["Arsenal", "Chelsea", "Liverpool", "Manchester City",
                         "Manchester United", "Tottenham", "Aston Villa",
                         "Newcastle", "West Ham", "Brighton"],
    "captain_advisory": ["Arsenal", "Chelsea", "Liverpool", "Manchester City",
                         "Manchester United", "Tottenham", "Aston Villa", "Newcastle"],
}

def _yara_get(endpoint: str, params: dict | None = None) -> dict | list | None:
    """Public API — no auth header required."""
    try:
        r = requests.get(f"{YARA_API}{endpoint}",
                         params=params or {}, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[Node 1] WARN: {endpoint} → {e}")
        return None


def node1_fetch(trigger_type: str,
                player_name: str | None = None,
                teams: list[str] | None = None) -> dict:
    """
    Node 1 — HTTP Fetcher.
    Team endpoint:   GET /teams/{TeamName}/overview
    Player endpoint: GET /players/{Player Name}/risk  (URL-encoded)
    """
    ctx: dict = {
        "trigger_type": trigger_type,
        "matchday": datetime.now().strftime("%A %-d %B").upper(),
    }

    if trigger_type in ("matchday_board", "captain_advisory"):
        all_players = []
        for team in (teams or TRIGGER_TEAMS.get(trigger_type, [])):
            data = _yara_get(f"/teams/{team}/overview")
            if data and "players" in data:
                # Team overview gives a lightweight list — we'll enrich
                # top players individually via the /risk endpoint
                all_players.extend(data["players"])
            if trigger_type == "captain_advisory":
                ctx["fpl_insights"] = _yara_get("/fpl/insights")
        ctx["raw_players"] = all_players

    elif player_name:
        import urllib.parse
        encoded = urllib.parse.quote(player_name)
        ctx["raw_players"] = [_yara_get(f"/players/{encoded}/risk") or {}]

    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — JSON Parser
# Splits the raw API response into individual player dicts.
# Enables the loop — every downstream node receives one player at a time.
# ─────────────────────────────────────────────────────────────────────────────

def node2_parse(ctx: dict) -> list[dict]:
    """
    Node 2 — JSON Parser.
    Maps the raw API response to the canonical player dict.
    Strips ALL pre-written narrative fields so Node 4 reasons from numbers only.
    """
    raw = ctx.get("raw_players", [])
    players = []
    for p in raw:
        if not p:
            continue

        factors       = p.get("factors") or {}
        scoring       = p.get("scoring_odds") or {}
        next_fix      = p.get("next_fixture") or {}
        fpl_proj      = p.get("fpl_points_projection") or {}
        importance    = p.get("player_importance") or {}

        players.append({
            # ── Identity ────────────────────────────────────────────────────
            "player_name":     p.get("name", ""),
            "player_team":     p.get("team", ""),
            "position":        _normalise_position(p.get("position", "")),
            "age":             p.get("age"),
            "shirt_number":    p.get("shirt_number"),
            "league":          p.get("league", ""),
            "player_image_url": p.get("player_image_url", ""),

            # ── Risk (raw numbers only — no pre-written copy) ───────────────
            "risk_probability":   p.get("risk_probability"),      # 0.0–1.0
            "risk_tier":          p.get("risk_level", ""),        # High/Medium/Low
            "archetype":          p.get("archetype", ""),
            "risk_percentile":    p.get("risk_percentile"),
            "is_currently_injured": p.get("is_currently_injured", False),
            "chance_of_playing":  p.get("chance_of_playing"),

            # ── Injury history (factors) ────────────────────────────────────
            "player_injury_count":          factors.get("previous_injuries"),
            "total_days_lost":              factors.get("total_days_lost"),
            "days_since_last_injury_capped": factors.get("days_since_last_injury"),
            "avg_days_per_injury":          factors.get("avg_days_per_injury"),
            "injury_news":                  p.get("injury_news", ""),

            # ── Load model ──────────────────────────────────────────────────
            "acwr":         p.get("acwr"),
            "acute_load":   p.get("acute_load"),
            "chronic_load": p.get("chronic_load"),
            "spike_flag":   bool(p.get("spike_flag")),

            # ── Scoring / form ──────────────────────────────────────────────
            "goals_per_90":     scoring.get("goals_per_90"),
            "assists_per_90":   scoring.get("assists_per_90"),
            "score_probability": scoring.get("score_probability"),
            "involvement_prob":  scoring.get("involvement_probability"),

            # ── Fixture context ─────────────────────────────────────────────
            "next_opponent":  next_fix.get("opponent", ""),
            "is_home":        next_fix.get("is_home"),
            "fixture_difficulty": next_fix.get("difficulty"),
            "upcoming_fixtures": p.get("upcoming_fixtures"),

            # ── FPL signals ─────────────────────────────────────────────────
            "ownership_percent": _to_decimal(p.get("ownership_percent")),
            "transfer_delta":    p.get("transfer_delta"),
            "fpl_price":         p.get("fpl_value", {}).get("price") if isinstance(p.get("fpl_value"), dict) else None,
            "fpl_pts_projection": fpl_proj.get("projected_points"),

            # ── Player importance ───────────────────────────────────────────
            "player_importance_score": importance.get("score") if isinstance(importance, dict) else p.get("player_importance"),
            "minutes_played":   p.get("minutes_played"),
            "is_starter":       p.get("is_starter"),

            # ── Workflow status ─────────────────────────────────────────────
            "post_status":  "Available",
            "last_updated": datetime.now(timezone.utc).isoformat(),
        })
    return players


def _normalise_position(pos: str) -> str:
    m = {"goalkeeper":"GK","defender":"DEF","midfielder":"MID","forward":"FWD",
         "gk":"GK","def":"DEF","mid":"MID","fwd":"FWD"}
    return m.get(str(pos).lower().strip(), pos.upper()[:3])


def _to_decimal(v) -> float | None:
    """Convert 78 or 0.78 or '78%' → 0.78"""
    if v is None:
        return None
    try:
        f = float(str(v).replace("%", ""))
        return f / 100 if f > 1 else f
    except:
        return None


def _risk_tier(v) -> str:
    if v is None:
        return "Low"
    try:
        pct = float(str(v).replace("%", ""))
        pct = pct * 100 if pct <= 1 else pct
    except:
        return "Low"
    if pct >= 65:
        return "High"
    if pct >= 40:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — Airtable Updater
# Long-term memory and persistence layer.
# Upserts every player from Node 2 into Airtable.
# Airtable AI Assist then automatically computes signal fields:
#   post_trigger, risk_label_calc, risk_tier, and any derived columns.
# The workflow trusts those computed values — it does not overwrite them.
# ─────────────────────────────────────────────────────────────────────────────

def node3_airtable_upsert(players: list[dict]) -> dict[str, str]:
    """
    Node 3 — Batched Airtable upsert for all players.
    1. Fetch all existing records in one call → build name→id map
    2. Split into creates and updates
    3. Batch create / update in groups of 10
    Returns {player_name: record_id}
    """
    if not players:
        return {}

    def _fields(p: dict) -> dict:
        """Build clean field dict — only writable Airtable fields."""
        SKIP = {
            # Not in schema
            "is_currently_injured", "minutes_played", "player_image_url",
            "league", "shirt_number", "total_days_lost", "avg_days_per_injury",
            "goals_per_90", "assists_per_90", "score_probability",
            "involvement_prob", "upcoming_fixtures", "fpl_pts_projection",
            "fixture_difficulty", "is_starter",
            # AI Assist / formula fields — Airtable computes these
            "archetype", "risk_tier", "risk_label_calc", "post_trigger",
        }
        out = {}
        for k, v in p.items():
            if v is None or k in SKIP:
                continue
            # Percent fields: Airtable stores 0.0–1.0
            if k in ("risk_probability", "chance_of_playing", "ownership_percent"):
                fv = float(v)
                out[k] = fv / 100 if fv > 1 else fv
            # Integer fields
            elif k in ("days_since_last_injury_capped", "player_injury_count",
                       "matches_last_7", "matches_last_30", "gw_number",
                       "player_importance_score"):
                try: out[k] = int(float(str(v)))
                except: pass
            # Boolean fields
            elif k in ("is_home", "spike_flag"):
                out[k] = bool(v)
            else:
                out[k] = v
        return out

    # ── Step 1: fetch all existing records in one call ───────────────────────
    r = requests.get(
        f"{AT_BASE_URL}/{AT_PLAYERS_TBL}",
        headers=AT_HEADERS,
        params={"fields[]": "player_name"},
        timeout=15,
    )
    existing_records = r.json().get("records", []) if r.ok else []
    existing_map = {
        rec["fields"].get("player_name", ""): rec["id"]
        for rec in existing_records
        if rec.get("fields", {}).get("player_name")
    }
    print(f"[Node 3] {len(existing_map)} existing records found")

    # ── Step 2: split into creates and updates ───────────────────────────────
    to_create, to_update = [], []
    rec_ids: dict[str, str] = {}

    for p in players:
        name   = p.get("player_name", "").strip()
        fields = _fields(p)
        if name in existing_map:
            to_update.append({"id": existing_map[name], "fields": fields})
            rec_ids[name] = existing_map[name]
        else:
            to_create.append({"fields": fields, "_name": name})

    # ── Step 3: batch create (10 per request) ───────────────────────────────
    for i in range(0, len(to_create), 10):
        batch = to_create[i:i+10]
        payload = [{"fields": b["fields"]} for b in batch]
        r = requests.post(
            f"{AT_BASE_URL}/{AT_PLAYERS_TBL}",
            headers=AT_HEADERS,
            json={"records": payload},
            timeout=15,
        )
        if r.ok:
            for j, rec in enumerate(r.json().get("records", [])):
                name = batch[j]["_name"]
                rec_ids[name] = rec["id"]
            print(f"[Node 3] + created {len(batch)} records")
        else:
            print(f"[Node 3] create error: {r.text[:100]}")

    # ── Step 4: batch update (10 per request) ───────────────────────────────
    for i in range(0, len(to_update), 10):
        batch = to_update[i:i+10]
        r = requests.patch(
            f"{AT_BASE_URL}/{AT_PLAYERS_TBL}",
            headers=AT_HEADERS,
            json={"records": batch},
            timeout=15,
        )
        if r.ok:
            print(f"[Node 3] ↑ updated {len(batch)} records")
        else:
            print(f"[Node 3] update error: {r.text[:100]}")

    print(f"[Node 3] Done — {len(to_create)} created, {len(to_update)} updated")
    return rec_ids


def node3_update_post_status(rec_id: str, trigger: str,
                              status: str = "Drafted") -> None:
    """Marks a single player record as Drafted after Node 6."""
    if not rec_id:
        return
    requests.patch(
        f"{AT_BASE_URL}/{AT_PLAYERS_TBL}",
        headers=AT_HEADERS,
        json={"records": [{"id": rec_id, "fields": {
            "post_trigger": trigger,
            "post_status":  status,
        }}]},
        timeout=10,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GATEKEEPER — runs after Node 3, before Nodes 4–7
# Prevents burning OpenAI + Playwright credits on players nobody owns.
# ─────────────────────────────────────────────────────────────────────────────

def should_process_content(player_data: dict) -> bool:
    """
    Only spend credits on players FPL managers actually care about.
    Risk probability >= 0.25 (scaled: 0.25 = Moderate threshold)
    OR spike_flag is True (sudden jump always warrants a post)
    AND owned by more than 5% of FPL managers.
    """
    risk_flag = (
        player_data.get("risk_probability", 0) >= 0.25
        or bool(player_data.get("spike_flag", False))
    )
    owned      = player_data.get("ownership_percent", 0) or 0
    # ownership stored as decimal (0.05 = 5%)
    ownership_flag = owned > 0.05

    return risk_flag and ownership_flag
# GPT-4o finds the angle. Not a summary — a story.
# ─────────────────────────────────────────────────────────────────────────────

NODE4_SYSTEM = """
You are Yara's editorial brain. You receive structured injury risk data and decide what the story is.
Your job is not to summarise numbers. Your job is to find the angle.

Rules:
- Start from a specific number, then build the angle around it.
- Ask: what is the manager seeing that aligns or conflicts with the model?
- Ask: is there a contradiction worth calling out?
- Flag model-vs-bookmaker disagreement when present.
- Flag FRAGILE archetype + FPL captain pick — that tension is always worth a post.
- Never speculate without a number behind it.
- If nothing is genuinely interesting, return {"should_post": false, "reason": "NO_STORY"}.
- For MATCHDAY_BOARD: rank top-5 players by risk. If the fixture is Top-6 or a derby, set marquee_matchup=true and name the two headline players. Generate one gladiator_stat for each — a unique, one-line historical or load fact.
- NO exclamation marks. NO em dashes. NO BREAKING. NO URGENT.

Output ONLY valid JSON:
{
  "post_type": "matchday_board|captain_advisory|risk_spike|lineup_reaction|we_called_it|halftime_reaction|odds_disagreement",
  "headline_player": "string or null",
  "headline_angle": "one sentence — the core story",
  "supporting_signals": ["signal 1", "signal 2"],
  "model_vs_manager_tension": true|false,
  "risk_score_pct": number,
  "should_post": true|false,
  "marquee_matchup": true|false,
  "marquee_players": ["p1", "p2"] or null,
  "gladiator_stats": {"p1_name": "stat", "p2_name": "stat"} or null,
  "top5_players": [
    {"name":"","club":"","position":"","risk_pct":0,"archetype":"","major_signal":"","delta_pct":0}
  ] or null,
  "verdict": "one cinematic sentence for the battle card" or null,
  "vitals": {
    "p1_name": {"acwr":"","strain":"","recurrence":"","load_slope":"","fatigue":"","goals_per_game":"","assists_per_game":"","form":""},
    "p2_name": {...}
  } or null,
  "captain_picks": [{"name":"","risk_pct":0,"fixture_rating":"","verdict":"START|COVER|AVOID"}] or null,
  "reddit_angle": "3-4 paragraph plain-text for r/FantasyPL" or null
}
""".strip()


def node4_reason(players: list[dict], trigger_type: str,
                 extra_ctx: dict | None = None) -> dict:
    """
    Runs GPT-4o over the full player list for this trigger.
    Returns parsed reasoning JSON.
    """
    payload = {
        "trigger_type": trigger_type,
        "players": players,
        **(extra_ctx or {}),
    }
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": NODE4_SYSTEM},
                {"role": "user",   "content": json.dumps(payload, default=str)},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"[Node 4] ERROR: {e}")
        return {"should_post": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 — Image Generator
# Uses the three pre-built Playwright templates.
# ─────────────────────────────────────────────────────────────────────────────

def node5_generate_image(players: list[dict],
                         reasoning: dict,
                         trigger_type: str) -> str | None:
    """
    Picks the right template and renders a 1600×900 PNG.
    Returns the local file path.
    """
    gw     = (players[0].get("gw_number") if players else None) or "—"
    league = "Premier League"

    # ── MATCHDAY BOARD ───────────────────────────────────────────────────────
    if trigger_type == "matchday_board":

        if reasoning.get("marquee_matchup") and reasoning.get("marquee_players"):
            return _render_ufc_card(reasoning, players, gw, league)
        else:
            return _render_matchday_board(reasoning, gw, league)

    # ── SINGLE-PLAYER RISK CARD ──────────────────────────────────────────────
    else:
        return _render_risk_card(players, reasoning, gw, league)


def _render_matchday_board(reasoning: dict, gw, league: str) -> str | None:
    top5 = reasoning.get("top5_players") or []
    if not top5:
        return None

    board_players = []
    for p in top5[:5]:
        name_parts = (p.get("name") or "").split()
        board_players.append({
            "first_name":   name_parts[0] if name_parts else "",
            "last_name":    " ".join(name_parts[1:]) if len(name_parts) > 1 else "",
            "club":         p.get("club", ""),
            "position":     p.get("position", "MID"),
            "risk_pct":     p.get("risk_pct", 0),
            "archetype":    p.get("archetype", "MONITOR"),
            "major_signal": p.get("major_signal", ""),
            "delta_pct":    p.get("delta_pct", 0),
        })

    out = str(OUTPUT_DIR / "matchday_board.png")
    return build_matchday_board(
        {"gw": gw, "league": league,
         "matchday": datetime.now().strftime("%A %-d %B").upper(),
         "players": board_players},
        output_path=out,
    )


def _render_ufc_card(reasoning: dict, players: list[dict], gw, league: str) -> str | None:
    mp = reasoning["marquee_players"]
    gs = reasoning.get("gladiator_stats") or {}
    vt = reasoning.get("vitals") or {}
    top5 = reasoning.get("top5_players") or []

    def _find(name):
        for p in top5:
            if name.lower() in (p.get("name") or "").lower():
                return p
        for p in players:
            if name.lower() in (p.get("player_name") or "").lower():
                return p
        return {}

    def _build_side(full_name: str) -> dict:
        raw  = _find(full_name)
        parts = full_name.split()
        return {
            "first_name":    parts[0] if parts else "",
            "last_name":     " ".join(parts[1:]) if len(parts) > 1 else "",
            "club":          raw.get("club") or raw.get("player_team", ""),
            "position":      raw.get("position", "MID"),
            "risk_pct":      raw.get("risk_pct") or int((raw.get("risk_probability") or 0) * 100),
            "archetype":     raw.get("archetype", "FRAGILE"),
            "gladiator_stat": gs.get(full_name, ""),
            "vitals":        vt.get(full_name, {}),
        }

    out = str(OUTPUT_DIR / "ufc_battle_card.png")
    return build_ufc_card(
        {
            "gw": gw, "league": league,
            "matchday": datetime.now().strftime("%A %-d %B").upper(),
            "fixture_label": f"{mp[0]} vs {mp[1]}",
            "badge_type":    "TOP-6 CLASH",
            "p1":            _build_side(mp[0]),
            "p2":            _build_side(mp[1]),
            "verdict":       reasoning.get("verdict", ""),
        },
        output_path=out,
    )


def _render_risk_card(players: list[dict], reasoning: dict,
                      gw, league: str) -> str | None:
    player = players[0] if players else {}
    if not player:
        return None

    name   = player.get("player_name", "")
    parts  = name.split()
    risk   = int((player.get("risk_probability") or 0) * 100)
    arch   = player.get("archetype", "MONITOR")

    supporting = reasoning.get("supporting_signals", [])

    signals = [
        {"type": "Minutes Trend",
         "text": str(player.get("workload_slope", "")),
         "value": "", "direction": "down"},
        {"type": "Load Index",
         "text": f"ACWR {player.get('acwr', '—')}",
         "value": f"+{int((player.get('risk_probability',0))*100)}%",
         "direction": "up"},
        {"type": "Next Match",
         "text": f"{player.get('next_opponent','—')} ({'H' if player.get('is_home') else 'A'})  ·  GW{gw}",
         "value": "", "direction": "neutral"},
        {"type": "Congestion",
         "text": f"{player.get('matches_last_7', '—')} fixtures in next 7 days",
         "value": "Flagged" if player.get("spike_flag") else "Clear",
         "direction": "flag" if player.get("spike_flag") else "neutral"},
    ]
    # Override with AI supporting signals if richer
    if supporting:
        signals = [
            {"type": "Signal", "text": s, "value": "", "direction": "neutral"}
            for s in supporting[:4]
        ]

    out = str(OUTPUT_DIR / "risk_card.png")
    return build_risk_card(
        {
            "first_name":     parts[0] if parts else "",
            "last_name":      " ".join(parts[1:]) if len(parts) > 1 else "",
            "club":           player.get("player_team", ""),
            "position":       player.get("position", "MID"),
            "gw":             gw,
            "league":         league,
            "risk_pct":       risk,
            "archetype":      arch.upper(),
            "signals":        signals,
            "fpl_ownership":  round((player.get("ownership_percent") or 0) * 100, 1),
            "transfers":      int(player.get("transfer_delta") or 0),
            "xmins":          int(player.get("matches_last_7") or 0) * 90 // 3,
        },
        output_path=out,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6 — AI Copywriting
# Drafts the post in Yara's voice.
# ─────────────────────────────────────────────────────────────────────────────

NODE6_SYSTEM = """
You are Yara. AI injury risk analyst for football. First person. Not a bot.

Voice rules:
- No exclamation marks. Ever.
- No emoji except national flag glyphs where genuinely relevant.
- No BREAKING or URGENT.
- Always lead with a number. Never just a vibe.
- Sentence case. Normal punctuation.
- No em dashes. Use commas or periods.
- Dry humor allowed. Warmth allowed. Hype is not.
- You may push back on manager decisions with data, not opinion.
- You may call out your own misses. Accountability builds trust.

Format by post_type:

MATCHDAY_BOARD: open with one line framing the day's risk landscape. List 5 players: name, risk%, one-sentence signal. Close with one line for FPL managers.

CAPTAIN_ADVISORY: name top 3 picks. For each: risk score, fixture rating, verdict (START / COVER / AVOID). One honest line about uncertainty.

RISK_SPIKE: one player, one number, one angle. What changed, what it means for the next fixture. Max 4 sentences.

LINEUP_REACTION: react to confirmed XI within Yara's model. If high-risk starts: "Yara had [name] at [X]%. [Manager] sees something different. Watch [signal] at 60 minutes." Max 3 sentences.

WE_CALLED_IT: state what Yara said pre-match and what happened. No gloating. If model was wrong: "Yara had [name] at [X]% and he ran 90 minutes. The model noted it. We noted it too."

ODDS_DISAGREEMENT: "Yara has [name] at [X]% risk. The market has him at [implied odds]. One of these is wrong." Then one line on what the model sees that the market may not.

Output: post text only. No preamble.
Max 280 chars for single post, or up to 4 tweets separated by /// for threads.
""".strip()


def node6_draft_post(reasoning: dict) -> dict:
    """Returns {x_post, reddit_post}."""
    prompt = (
        f"post_type: {reasoning.get('post_type')}\n"
        f"headline_angle: {reasoning.get('headline_angle')}\n"
        f"supporting_signals: {reasoning.get('supporting_signals')}\n"
        f"risk_score_pct: {reasoning.get('risk_score_pct')}\n"
        f"model_vs_manager_tension: {reasoning.get('model_vs_manager_tension')}\n"
        f"headline_player: {reasoning.get('headline_player')}\n"
        f"captain_picks: {reasoning.get('captain_picks')}\n"
        f"top5_players: {reasoning.get('top5_players')}\n"
    )
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": NODE6_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.5,
        )
        x_post = resp.choices[0].message.content.strip()
    except Exception as e:
        x_post = f"[Node 6 ERROR] {e}"

    return {
        "x_post":      x_post,
        "reddit_post": reasoning.get("reddit_angle") or "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 7 — Notion
# Saves X + Reddit drafts for human approval. Nothing posts automatically.
# ─────────────────────────────────────────────────────────────────────────────

def node7_save_draft(reasoning: dict, post: dict,
                     image_path: str | None) -> dict:
    """Creates Notion pages for X and Reddit drafts. Returns page URLs."""
    if not NOTION_KEY or not NOTION_DB:
        print("[Node 7] Notion not configured. Skipping.")
        return {}

    ts        = datetime.now().strftime("%Y-%m-%d %H:%M")
    post_type = reasoning.get("post_type", "unknown")
    player    = reasoning.get("headline_player") or "Board"
    risk      = float(reasoning.get("risk_score_pct") or 0)
    img_url   = image_path or ""

    def _create(db_id: str, title: str, body: str, platform: str) -> str | None:
        payload = {
            "parent":     {"database_id": db_id},
            "properties": {
                "Name":           {"title": [{"text": {"content": title}}]},
                "post_type":      {"select": {"name": post_type}},
                "risk_score_pct": {"number": risk},
                "platform":       {"select": {"name": platform}},
                "Status":         {"select": {"name": "Pending Review"}},
            },
            "children": [{
                "object": "block", "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": body[:2000]}}]},
            }],
        }
        if img_url:
            payload["properties"]["image_url"] = {"url": img_url}
        try:
            r = requests.post("https://api.notion.com/v1/pages",
                              headers=NOTION_HDR, json=payload, timeout=10)
            r.raise_for_status()
            url = r.json().get("url")
            print(f"[Node 7] {platform} draft → {url}")
            return url
        except Exception as e:
            print(f"[Node 7] ERROR ({platform}): {e}")
            return None

    x_url = _create(NOTION_DB,
                    f"[{post_type.upper()}] {player} — {ts}",
                    post["x_post"], "x")

    reddit_url = None
    if post.get("reddit_post"):
        reddit_url = _create(NOTION_REDDIT_DB,
                             f"[REDDIT][{post_type.upper()}] {player} — {ts}",
                             post["reddit_post"], "reddit")

    return {"x_page_url": x_url, "reddit_page_url": reddit_url}


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run(trigger_type: str,
        player_name: str | None = None,
        matchday:    str | None = None) -> dict:

    print(f"\n{'='*60}")
    print(f"  YARA  |  trigger={trigger_type}  |  player={player_name}")
    print(f"{'='*60}\n")

    # NODE 1 — Fetch
    print("[Node 1] Fetching from yaraspeaks API...")
    ctx = node1_fetch(trigger_type, player_name=player_name)
    if matchday:
        ctx["matchday"] = matchday.upper()

    # NODE 2 — Parse into player list
    print("[Node 2] Parsing player data...")
    players = node2_parse(ctx)
    print(f"[Node 2] {len(players)} player(s) parsed.")

    # NODE 3 — Airtable upsert (batched — all players in one pass)
    print("[Node 3] Upserting to Airtable...")
    rec_ids = node3_airtable_upsert(players)

    # ── GATEKEEPER — filter before expensive Nodes 4–7 ──────────────────────
    eligible = [p for p in players if should_process_content(p)]
    skipped  = len(players) - len(eligible)
    print(f"[Gatekeeper] {len(eligible)} eligible / {skipped} skipped "
          f"(risk < 25% and no spike, or ownership ≤ 5%)")

    if not eligible:
        print("[Gatekeeper] No players meet threshold. Stopping.")
        return {"status": "gated", "reason": "no_eligible_players"}

    # NODE 4 — AI Reasoning
    print("[Node 4] Reasoning...")
    reasoning = node4_reason(eligible, trigger_type,
                              extra_ctx={"matchday": ctx.get("matchday"),
                                         "fpl_insights": ctx.get("fpl_insights")})

    if not reasoning.get("should_post", True):
        print(f"[Node 4] NO_STORY → {reasoning.get('reason','')}")
        return {"status": "no_story", "reasoning": reasoning}

    print(f"[Node 4] Angle: {reasoning.get('headline_angle','')}")

    # NODE 5 — Generate image
    print("[Node 5] Rendering image...")
    image_path = node5_generate_image(eligible, reasoning, trigger_type)
    print(f"[Node 5] → {image_path}")

    # NODE 6 — Draft post
    print("[Node 6] Drafting post...")
    post = node6_draft_post(reasoning)
    print(f"\n── X POST ──────────────────────────────\n{post['x_post']}\n")

    # Update Airtable post_status for headline player
    headline_player = reasoning.get("headline_player") or ""
    if headline_player in rec_ids:
        node3_update_post_status(rec_ids[headline_player], trigger_type)

    # NODE 7 — Save to Notion
    print("[Node 7] Saving draft to Notion...")
    notion = node7_save_draft(reasoning, post, image_path)

    return {
        "status":        "drafted",
        "trigger":       trigger_type,
        "image_path":    image_path,
        "x_post":        post["x_post"],
        "reddit_post":   post.get("reddit_post", ""),
        "notion_x":      notion.get("x_page_url"),
        "notion_reddit": notion.get("reddit_page_url"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Yara Signal → Draft Post")
    ap.add_argument("--trigger", required=True,
                    choices=["matchday_board", "risk_spike", "lineup_reaction",
                             "halftime_reaction", "captain_advisory",
                             "we_called_it"])
    ap.add_argument("--player",   default=None)
    ap.add_argument("--matchday", default=None)
    args = ap.parse_args()

    result = run(args.trigger, player_name=args.player, matchday=args.matchday)
    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("reasoning",)},
        indent=2, default=str
    ))
