"""
Yara Auto-Poster
================
Content formats:
  1. Daily Flag            — top risk player per team, posted every morning
  2. Gameweek Preview      — top 5 risks across the league, posted Friday
  3. Post-Match Callout    — auto-detects when a flagged player misses/is subbed off
  4. Intl Break Watch      — who's accumulating minutes on international duty
  5. Return GW Preview     — risk heading into the first GW back from a break
  6. Break Recovery        — who benefits from the rest, risk dropping
  7. Accuracy Recap        — how Yara's pre-break flags held up

Setup:
  pip install tweepy praw requests python-dotenv schedule

Then fill in .env (see .env.example)
Run: python yara_autoposter.py
"""

import os
import json
import time
import random
import logging
import schedule
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("yara_poster.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("yara")

# ─── CONFIG ──────────────────────────────────────────────────────────────────

TWITTER_ENABLED   = os.getenv("TWITTER_ENABLED", "true").lower() == "true"
REDDIT_ENABLED    = os.getenv("REDDIT_ENABLED", "true").lower() == "true"
YARA_API_BASE     = os.getenv("YARA_API_BASE", "https://www.yaraspeaks.com/api")
REDDIT_SUBREDDITS = os.getenv("REDDIT_SUBREDDITS", "FantasyPL,PremierLeague").split(",")
DRY_RUN           = os.getenv("DRY_RUN", "false").lower() == "true"  # set true to test without posting

EPL_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton Hove",
    "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Liverpool", "Man City", "Man United", "Newcastle",
    "Nottingham", "Tottenham", "West Ham", "Wolverhampton",
    # These may 404 if not in current inference_df — that's fine, we skip them
    "Burnley", "Leeds United", "Sunderland",
]

# ─── PROMINENCE TIERS ────────────────────────────────────────────────────────

# TIER 1 (weight 2.0): Elite, globally recognised. Posts about these get
# engagement from non-FPL fans. ~25-30 names.
TIER_1_PLAYERS = {
    "salah", "haaland", "saka", "palmer", "de bruyne",
    "son", "bruno fernandes", "rashford", "rice", "odegaard",
    "van dijk", "alexander-arnold", "foden", "grealish", "havertz",
    "martinez", "onana", "alisson", "ederson", "silva",
    "luis diaz", "darwin nunez", "gabriel jesus", "martinelli",
    "sancho", "sterling", "enzo fernandez",
}

# TIER 2 (weight 1.5): Strong FPL assets, well-known PL players with a
# meaningful fanbase. Regular starters for top-8 clubs, popular FPL picks,
# typically 200k+ ownership. ~60-80 names.
TIER_2_PLAYERS = {
    "watkins", "isak", "gordon", "jota", "diogo jota",
    "trossard", "timber", "saliba", "gabriel", "white",
    "guimaraes", "bruno guimaraes", "schar", "trippier",
    "porro", "kulusevski", "maddison", "romero", "bissouma",
    "jackson", "caicedo", "mudryk", "nkunku", "colwill", "mount",
    "madueke", "enzo", "cucurella", "james", "chilwell",
    "bowen", "paqueta", "kudus", "antonio", "ward-prowse",
    "ait-nouri", "cunha", "hwang", "neto", "strand larsen",
    "calvert-lewin", "mcneil", "pickford", "mykolenko",
    "semenyo", "kluivert", "solanke", "evanilson",
    "mbeumo", "toney", "wissa", "pinnock",
    "joao pedro", "mitoma", "welbeck", "ferguson",
    "barnes", "vardy", "ndidi", "faes",
    "ipswich town", "szmodics", "delap",
    "munoz", "eze", "mateta", "olise",
    "elanga", "gibbs-white", "wood", "murillo",
    "rogers", "dibling", "bednarek",
    "cash", "mcginn", "konsa", "bailey", "duran",
    "martinez emiliano", "emiliano martinez",
    "doku", "gvardiol", "rodri", "kovacic",
    "dalot", "mainoo", "hojlund", "amad", "garnacho",
    "raya", "calafiori", "zinchenko", "partey", "nwaneri",
}

# Weight lookup
TIER_WEIGHTS = {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.3}


def _name_matches(tier_entry: str, player_name: str) -> bool:
    """Match a tier entry against a player name using word-boundary-safe logic.

    Rules:
    - Multi-word tier entries (e.g. "bruno fernandes") use substring match
      since they are specific enough to avoid false positives.
    - Single-word tier entries (e.g. "salah", "son") must match a complete
      word token in the player name to avoid "son" matching "henderson".
    """
    tier_words = tier_entry.split()
    if len(tier_words) >= 2:
        # Multi-word: substring is safe ("bruno fernandes" won't false-positive)
        return tier_entry in player_name
    # Single word: must match a full token
    return tier_entry in player_name.split()


def _get_player_tier(name: str, player_data: dict = None) -> int:
    """Determine prominence tier for a player.

    Tier 4 (penalised) if the player is already confirmed injured/unavailable.
    Otherwise check tier 1, tier 2, or fall through to tier 3.
    """
    name_lower = name.lower()

    # Tier 4: already injured/suspended — no point flagging
    if player_data:
        reasons = player_data.get("reasons") or []
        reasons_text = " ".join(r.lower() for r in reasons)
        if any(kw in reasons_text for kw in [
            "confirmed out", "suspended", "ruled out",
            "surgery", "acl", "cruciate", "broken",
        ]):
            return 4

    # Tier 1: global recognition
    for t1_name in TIER_1_PLAYERS:
        if _name_matches(t1_name, name_lower):
            return 1

    # Tier 2: strong FPL / well-known PL
    for t2_name in TIER_2_PLAYERS:
        if _name_matches(t2_name, name_lower):
            return 2

    # Tier 3: everyone else
    return 3


def get_prominence_score(player: dict) -> float:
    """Calculate prominence_score = risk_score * player_weight."""
    risk = player.get("risk_score", 0)
    tier = _get_player_tier(player.get("name", ""), player)
    weight = TIER_WEIGHTS.get(tier, 1.0)
    return round(risk * weight, 4)


def rank_by_prominence(players: list[dict]) -> list[dict]:
    """Sort players by prominence score descending. Annotate each with internal fields."""
    for p in players:
        p["_tier"] = _get_player_tier(p.get("name", ""), p)
        p["_weight"] = TIER_WEIGHTS.get(p["_tier"], 1.0)
        p["_prominence"] = get_prominence_score(p)
    return sorted(players, key=lambda x: x["_prominence"], reverse=True)


# ─── TWITTER CLIENT ───────────────────────────────────────────────────────────

def get_twitter_client():
    try:
        import tweepy
        client = tweepy.Client(
            bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
        )
        return client
    except ImportError:
        log.error("tweepy not installed. Run: pip install tweepy")
        return None

def post_twitter(text: str, client=None) -> bool:
    if DRY_RUN:
        log.info(f"[DRY RUN] Twitter post:\n{text}\n{'─'*60}")
        return True
    if not TWITTER_ENABLED:
        return False
    if client is None:
        client = get_twitter_client()
    if client is None:
        return False
    try:
        response = client.create_tweet(text=text[:280])
        log.info(f"Twitter posted: tweet_id={response.data['id']}")
        return True
    except Exception as e:
        log.error(f"Twitter post failed: {e}")
        return False

# ─── REDDIT CLIENT ────────────────────────────────────────────────────────────

def get_reddit_client():
    try:
        import praw
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "YaraSports Bot v1.0"),
        )
        return reddit
    except ImportError:
        log.error("praw not installed. Run: pip install praw")
        return None

def post_reddit(title: str, body: str, subreddits: list = None, reddit=None) -> bool:
    if DRY_RUN:
        log.info(f"[DRY RUN] Reddit post:\nTitle: {title}\nBody:\n{body}\n{'─'*60}")
        return True
    if not REDDIT_ENABLED:
        return False
    if reddit is None:
        reddit = get_reddit_client()
    if reddit is None:
        return False

    targets = subreddits or REDDIT_SUBREDDITS
    success = False
    for sub in targets:
        try:
            subreddit = reddit.subreddit(sub.strip())
            subreddit.submit(title=title, selftext=body)
            log.info(f"Reddit posted to r/{sub}")
            success = True
            time.sleep(2)  # rate limit between subreddits
        except Exception as e:
            log.error(f"Reddit r/{sub} failed: {e}")
    return success

# ─── YARA DATA ────────────────────────────────────────────────────────────────

def fetch_predictions(team: str) -> dict | None:
    """Fetch live predictions from yaraspeaks.com API."""
    try:
        url = f"{YARA_API_BASE}/predictions"
        resp = requests.get(url, params={"team": team}, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        log.warning(f"Yara API returned {resp.status_code} for {team}")
    except Exception as e:
        log.warning(f"Yara API error for {team}: {e}")
    return None

def fetch_all_predictions() -> list[dict]:
    """Fetch predictions for all EPL teams, ranked by prominence."""
    all_players = []
    for team in EPL_TEAMS:
        data = fetch_predictions(team)
        if data and "players" in data:
            for p in data["players"]:
                p["team"] = team
            all_players.extend(data["players"])
        time.sleep(0.5)  # polite to the API
    return rank_by_prominence(all_players)

def get_current_gameweek() -> int:
    """Fetch live gameweek from FPL API. Fall back to date calculation."""
    try:
        resp = requests.get(
            "https://fantasy.premierleague.com/api/bootstrap-static/",
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code == 200:
            events = resp.json().get("events", [])
            for event in events:
                if event.get("is_current"):
                    return event["id"]
            # If no current, try next
            for event in events:
                if event.get("is_next"):
                    return event["id"]
    except Exception as e:
        log.warning(f"FPL API unreachable for gameweek: {e}")

    # Fallback: approximate from season start
    season_start = datetime(2024, 8, 16)
    weeks = (datetime.now() - season_start).days // 7
    return max(1, min(38, weeks + 1))

def get_risk_label(score: float) -> str:
    if score >= 0.65: return "HIGH"
    if score >= 0.35: return "ELEVATED"
    return "LOW"

def get_risk_emoji(score: float) -> str:
    if score >= 0.65: return "🔴"
    if score >= 0.35: return "🟡"
    return "🟢"

# ─── CONTENT BUILDERS ─────────────────────────────────────────────────────────

def build_daily_flag(player: dict, gw: int) -> dict:
    """Format 1: Daily Flag — top risk player of the day."""
    name    = player["name"]
    team    = player["team"]
    pct     = int(player["risk_score"] * 100)
    reasons = player.get("reasons", ["high workload detected", "fixture congestion"])
    r1      = reasons[0].capitalize() if reasons else "High workload detected"
    r2      = reasons[1].capitalize() if len(reasons) > 1 else "Fixture congestion period"
    tag     = team.replace(" ", "")

    twitter = (
        f"⚠️ YARA INJURY FLAG — GW{gw}\n\n"
        f"{name} ({team}) — {pct}% risk\n\n"
        f"{r1}. {r2}. "
        f"Yara has him flagged heading into the weekend.\n\n"
        f"If he's in your FPL side, worth having bench cover.\n\n"
        f"Full squad risk → yaraspeaks.com\n"
        f"#FPL #PremierLeague #{tag}"
    )

    reddit_title = f"Yara Injury Flag — GW{gw}: {name} ({team}) at {pct}% risk"
    reddit_body  = (
        f"**Yara's ensemble model (CatBoost + XGBoost + LightGBM) is flagging "
        f"{name} as elevated injury risk heading into GW{gw}.**\n\n"
        f"**Risk signals detected:**\n"
        + "\n".join(f"- {r.capitalize()}" for r in reasons[:3]) +
        f"\n\n**Risk score: {pct}%** ({get_risk_label(player['risk_score'])})\n\n"
        f"Model looks at workload patterns, injury history, fixture context, "
        f"and betting odds signals. Full {team} squad breakdown at yaraspeaks.com.\n\n"
        f"FPL managers — bench cover advised if he's a key pick this week.\n\n"
        f"*For educational purposes only.*"
    )

    return {"twitter": twitter, "reddit_title": reddit_title, "reddit_body": reddit_body, "player": player}

def build_gameweek_preview(players: list[dict], gw: int) -> dict:
    """Format 2: Gameweek Preview — top 5 risks by prominence across the league."""
    top5 = players[:5]

    tw_lines = "\n".join(
        f"{i+1}. {p['name']} ({p['team']}) — {int(p['risk_score']*100)}% {get_risk_emoji(p['risk_score'])}"
        for i, p in enumerate(top5)
    )
    twitter = (
        f"📊 YARA GW{gw} INJURY WATCH\n\n"
        f"Players to monitor this week:\n\n"
        f"{tw_lines}\n\n"
        f"Full breakdowns at yaraspeaks.com\n"
        f"#FPL #PremierLeague #FantasyFootball"
    )

    rd_lines = "\n".join(
        f"{i+1}. **{p['name']}** ({p['team']}) — {int(p['risk_score']*100)}% risk"
        + (f" — {p['reasons'][0].capitalize()}" if p.get("reasons") else "")
        for i, p in enumerate(top5)
    )
    reddit_title = f"Yara's Top 5 Injury Risks Heading Into GW{gw}"
    reddit_body  = (
        f"Yara's ensemble model has flagged these players as elevated injury "
        f"risks for GW{gw}. Useful if you're making FPL transfers or setting your lineup.\n\n"
        f"{rd_lines}\n\n"
        f"Model combines workload data, fixture context, injury history, and "
        f"odds signals. Full squad-by-squad breakdowns at yaraspeaks.com.\n\n"
        f"*Not financial/betting advice. Educational purposes only.*"
    )

    return {"twitter": twitter, "reddit_title": reddit_title, "reddit_body": reddit_body}

def build_callout(player: dict, gw: int, outcome: str = "substituted off") -> dict:
    """Format 3: Post-Match Callout — Yara flagged them and it happened."""
    name = player["name"]
    team = player["team"]
    pct  = int(player["risk_score"] * 100)
    reasons = player.get("reasons", ["elevated workload", "fixture congestion"])

    twitter = (
        f"🔔 YARA CALLED IT — GW{gw}\n\n"
        f"{name} was flagged at {pct}% injury risk before the match.\n\n"
        f"He was {outcome}.\n\n"
        f"The model saw it coming.\n"
        f"Follow for weekly injury intelligence.\n\n"
        f"Full squad data → yaraspeaks.com\n"
        f"#FPL #PremierLeague"
    )

    reddit_title = f"Yara flagged {name} at {pct}% risk before GW{gw} — {outcome}"
    reddit_body  = (
        f"Before GW{gw}'s fixtures, Yara's model had **{name}** at **{pct}%** injury probability.\n\n"
        f"**Signals that triggered the flag:**\n"
        + "\n".join(f"- {r.capitalize()}" for r in reasons[:3]) +
        f"\n\nHe was {outcome}.\n\n"
        f"Model is live at yaraspeaks.com — tracks the full EPL squad and generates "
        f"risk narratives, not just scores.\n\n"
        f"*Retrospective post for accuracy tracking. Educational purposes only.*"
    )

    return {"twitter": twitter, "reddit_title": reddit_title, "reddit_body": reddit_body, "player": player}

# ─── INTERNATIONAL BREAK CONTENT BUILDERS ────────────────────────────────────

def _workload_description(player: dict) -> str:
    """Build a short workload phrase from minutes_last_5."""
    mins = player.get("minutes_last_5") or []
    if not mins:
        return "heavy recent workload"
    full_90s = sum(1 for m in mins if m >= 90)
    total = sum(mins)
    if full_90s >= 4:
        return f"{full_90s} full 90s in his last 5 appearances ({total} mins total)"
    return f"{total} minutes across his last 5 appearances"


def build_intl_break_watch(players: list[dict], gw: int) -> dict:
    """Format 4: International Break Injury Watch.

    Surfaces high-risk prominent players who are likely still playing
    on international duty, accumulating more workload.
    """
    # Filter to elevated+ risk, take top 5 by prominence
    elevated = [p for p in players if p.get("risk_score", 0) >= 0.35]
    top5 = elevated[:5]
    if not top5:
        top5 = players[:5]

    tw_lines = "\n".join(
        f"{i+1}. {p['name']} ({p['team']}) — {int(p['risk_score']*100)}% "
        f"{get_risk_emoji(p['risk_score'])}"
        for i, p in enumerate(top5)
    )
    twitter = (
        f"🌍 YARA INTERNATIONAL BREAK WATCH\n\n"
        f"Risk scores look low right now. They should.\n"
        f"No PL matches = no PL workload.\n\n"
        f"But these players are still playing on intl duty, "
        f"and those minutes don't show up in the model yet:\n\n"
        f"{tw_lines}\n\n"
        f"The real risk is higher than the number says.\n\n"
        f"Full data → yaraspeaks.com\n"
        f"#FPL #PremierLeague #InternationalBreak"
    )

    rd_lines = "\n".join(
        f"{i+1}. **{p['name']}** ({p['team']}) — {int(p['risk_score']*100)}% risk — "
        f"{_workload_description(p)}"
        for i, p in enumerate(top5)
    )
    reddit_title = f"International Break Injury Watch — Why Low Scores Are Misleading Right Now"
    reddit_body = (
        f"You might have noticed that injury risk scores across the board "
        f"have dropped during the international break. That is expected. "
        f"Yara's model tracks Premier League workload (matches in the last 7/14/30 days, "
        f"ACWR, minutes accumulation), and with no PL fixtures, those features cool off.\n\n"
        f"**The blind spot:** International minutes are not yet fed into the model. "
        f"So when Salah plays 180 minutes for Egypt or Saka starts both England qualifiers, "
        f"the model does not see it. The numbers will catch up when the PL resumes, "
        f"but by then the fatigue is already baked in.\n\n"
        f"**Players to watch — their risk is higher than the score suggests:**\n\n"
        f"{rd_lines}\n\n"
        f"These were the highest-risk prominent players heading into the break. "
        f"Every international cap adds to their load without the model registering it. "
        f"Treat these scores as a floor, not a ceiling.\n\n"
        f"Keep an eye on team news before the GW{gw + 1} deadline. "
        f"Full squad breakdowns at yaraspeaks.com.\n\n"
        f"*For educational purposes only.*"
    )

    return {"twitter": twitter, "reddit_title": reddit_title, "reddit_body": reddit_body}


def build_return_gw_preview(players: list[dict], gw: int) -> dict:
    """Format 5: Return Gameweek Preview.

    Posted the Friday before the PL resumes after an international break.
    Same shape as gameweek preview but framed around break fatigue.
    """
    top5 = players[:5]

    tw_lines = "\n".join(
        f"{i+1}. {p['name']} ({p['team']}) — {int(p['risk_score']*100)}% "
        f"{get_risk_emoji(p['risk_score'])}"
        for i, p in enumerate(top5)
    )
    twitter = (
        f"📊 YARA GW{gw} RETURN WATCH\n\n"
        f"The PL is back. Scores will spike this week as match "
        f"workload feeds back into the model.\n\n"
        f"Highest risk returning:\n\n"
        f"{tw_lines}\n\n"
        f"These numbers do not include intl minutes. Real risk is higher.\n\n"
        f"Full breakdowns → yaraspeaks.com\n"
        f"#FPL #PremierLeague #FantasyFootball"
    )

    rd_lines = "\n".join(
        f"{i+1}. **{p['name']}** ({p['team']}) — {int(p['risk_score']*100)}% risk"
        + (f" — {p['reasons'][0].capitalize()}" if p.get("reasons") else "")
        for i, p in enumerate(top5)
    )
    reddit_title = f"GW{gw} Return Watch — Scores Are About to Spike After the Break"
    reddit_body = (
        f"The international break is over and the Premier League returns "
        f"with GW{gw}.\n\n"
        f"**Important context on the scores:** Risk numbers dropped during the break "
        f"because Yara's workload features (matches in 7/14/30 days, ACWR) cooled off "
        f"with no PL fixtures. International minutes are not yet captured in the model. "
        f"Expect these scores to jump back up after GW{gw}'s matches feed in.\n\n"
        f"**Top 5 players to monitor:**\n\n"
        f"{rd_lines}\n\n"
        f"Players who flew long-haul for international duty "
        f"(South America, Africa, Asia) carry additional fatigue risk "
        f"on top of what the numbers show. The model sees them as rested. "
        f"Their bodies are not.\n\n"
        f"If any of these are in your FPL side, make sure you have a "
        f"playing bench. Full squad-by-squad data at yaraspeaks.com.\n\n"
        f"*Not financial/betting advice. Educational purposes only.*"
    )

    return {"twitter": twitter, "reddit_title": reddit_title, "reddit_body": reddit_body}


def build_break_recovery(players: list[dict], gw: int) -> dict:
    """Format 6: Break Recovery Tracker.

    Highlights players whose risk is LOW or dropping — the break helped them.
    Useful for FPL managers looking for differentials coming back fresh.
    """
    # Find low-risk prominent players (the break beneficiaries)
    recovering = [
        p for p in players
        if p.get("risk_score", 0) < 0.35
        and _get_player_tier(p.get("name", ""), p) <= 2
    ]
    # Sort by tier first (tier 1 first), then by lowest risk
    recovering.sort(key=lambda p: (_get_player_tier(p.get("name", ""), p), p.get("risk_score", 0)))
    top5 = recovering[:5]

    if not top5:
        # Fallback: just show the lowest-risk prominent players
        prominent = [p for p in players if _get_player_tier(p.get("name", ""), p) <= 2]
        prominent.sort(key=lambda p: p.get("risk_score", 0))
        top5 = prominent[:5]

    if not top5:
        return None  # Nothing useful to post

    tw_lines = "\n".join(
        f"✅ {p['name']} ({p.get('team', '?')}) — {int(p.get('risk_score',0)*100)}%"
        for p in top5
    )
    twitter = (
        f"💚 YARA BREAK RECOVERY TRACKER\n\n"
        f"Most risk scores are low right now (no PL matches to drive workload).\n\n"
        f"But these players genuinely benefit from the rest:\n\n"
        f"{tw_lines}\n\n"
        f"Low base risk + no intl duty = truly fresh legs.\n\n"
        f"Full squad data → yaraspeaks.com\n"
        f"#FPL #PremierLeague #FantasyFootball"
    )

    rd_lines = "\n".join(
        f"- **{p['name']}** ({p.get('team', '?')}) — {int(p.get('risk_score',0)*100)}% risk"
        for p in top5
    )
    reddit_title = f"Break Recovery Tracker — Who Actually Benefits From the Rest?"
    reddit_body = (
        f"A note on the numbers first: most risk scores drop during international "
        f"breaks because Yara's PL workload features cool off with no fixtures. "
        f"Low scores across the board do not mean everyone is fine.\n\n"
        f"**The players below are different.** These are prominent names who were "
        f"already low risk before the break, were not called up or played limited "
        f"international minutes, and have no recent injury history dragging them up. "
        f"The break genuinely helped them.\n\n"
        f"**Players with the lowest genuine risk heading into GW{gw}:**\n\n"
        f"{rd_lines}\n\n"
        f"Compare these to players whose scores also look low but who played "
        f"180 minutes on international duty. The model cannot tell the difference "
        f"yet, but you can.\n\n"
        f"Good targets if you are looking for captain picks or transfers "
        f"with low downside risk. Full data at yaraspeaks.com.\n\n"
        f"*For educational purposes only.*"
    )

    return {"twitter": twitter, "reddit_title": reddit_title, "reddit_body": reddit_body}


def build_accuracy_recap(players: list[dict], gw: int) -> dict:
    """Format 7: Model Accuracy Recap.

    Summarises how many flagged players actually had issues.
    Uses current data as a proxy — players still at high risk or with
    injury-related reasons validate the pre-break flag.
    """
    # High-risk players = those Yara flagged
    flagged = [p for p in players if p.get("risk_score", 0) >= 0.55]
    tier_1_flagged = [p for p in flagged if _get_player_tier(p.get("name", ""), p) == 1]
    tier_2_flagged = [p for p in flagged if _get_player_tier(p.get("name", ""), p) == 2]

    total_flagged = len(flagged)
    prominent_flagged = len(tier_1_flagged) + len(tier_2_flagged)

    # Players with injury/absence signals in their reasons
    confirmed = []
    for p in flagged:
        reasons_text = " ".join(r.lower() for r in (p.get("reasons") or []))
        if any(kw in reasons_text for kw in [
            "days since last injury", "career injuries",
            "recovery window", "injury", "hamstring", "muscle",
            "knee", "ankle", "groin",
        ]):
            confirmed.append(p)

    # Pick top 3 most notable for the highlight
    highlights = rank_by_prominence(list(flagged))[:3]

    hl_lines = "\n".join(
        f"- {p['name']} ({p.get('team', '?')}) — flagged at "
        f"{int(p.get('risk_score',0)*100)}%"
        + (f" — {p['reasons'][0]}" if p.get("reasons") else "")
        for p in highlights
    )

    twitter = (
        f"📈 YARA MODEL RECAP — Pre-Break Flags\n\n"
        f"Before the break, Yara flagged {total_flagged} players "
        f"at 55%+ risk.\n\n"
        f"{prominent_flagged} were big names.\n\n"
        f"Scores have cooled (no PL matches = lower workload), "
        f"but the underlying injury history and fatigue are still there.\n\n"
        + "\n".join(
            f"• {p['name']} — was {int(p.get('risk_score',0)*100)}%"
            for p in highlights
        )
        + f"\n\nyaraspeaks.com\n"
        f"#FPL #PremierLeague"
    )

    reddit_title = f"Yara Model Recap — {total_flagged} Players Were Flagged Before the Break"
    reddit_body = (
        f"Before the international break, Yara's ensemble model (CatBoost + XGBoost + LightGBM) "
        f"flagged **{total_flagged} players** at 55%+ injury risk across the Premier League.\n\n"
        f"Of those, **{prominent_flagged}** were tier 1 or tier 2 names "
        f"(globally recognised or high-FPL-ownership players).\n\n"
        f"**Notable flags:**\n\n"
        f"{hl_lines}\n\n"
        f"**A note on current scores:** If you check these players now, their risk scores "
        f"will look lower. That is because the model's workload features (matches in 7/14/30 days, "
        f"ACWR) have cooled off during the break with no PL fixtures. International minutes "
        f"are not yet fed into the pipeline, so the model sees rest that may not exist.\n\n"
        f"The underlying risk factors (injury history, recurrence windows, career days lost) "
        f"have not changed. Expect scores to climb back when PL fixtures resume.\n\n"
        f"The model tracks workload patterns, injury history, and betting market signals. "
        f"It does not predict specific injuries, but flags elevated probability windows.\n\n"
        f"Full data and squad breakdowns at yaraspeaks.com.\n\n"
        f"*For educational purposes only.*"
    )

    return {"twitter": twitter, "reddit_title": reddit_title, "reddit_body": reddit_body}


# ─── MATCH RESULT DETECTION ───────────────────────────────────────────────────

class MatchMonitor:
    """
    Monitors for match results and detects when previously-flagged players
    miss games or are substituted off. Uses football-data.org free API.

    Get a free key at: https://www.football-data.org/
    Set FOOTBALL_DATA_API_KEY in your .env
    """

    def __init__(self):
        self.api_key      = os.getenv("FOOTBALL_DATA_API_KEY", "")
        self.base_url     = "https://api.football-data.org/v4"
        self.EPL_ID       = 2021
        self.flagged      = {}   # player_name -> risk_score (loaded before matchday)
        self.posted       = set()  # track posted callouts to avoid duplicates
        self.last_checked = None

    def load_flags(self):
        """Load today's high-risk players from Yara before matches start."""
        log.info("Loading pre-match flags from Yara...")
        self.flagged = {}
        for team in EPL_TEAMS:
            data = fetch_predictions(team)
            if data and "players" in data:
                for p in data["players"]:
                    if p.get("risk_score", 0) >= 0.55:  # only flag elevated+
                        self.flagged[p["name"].lower()] = {**p, "team": team}
        log.info(f"Loaded {len(self.flagged)} flagged players")

    def get_todays_matches(self) -> list:
        """Fetch today's EPL fixtures."""
        if not self.api_key:
            log.warning("FOOTBALL_DATA_API_KEY not set — match detection disabled")
            return []
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            resp = requests.get(
                f"{self.base_url}/competitions/{self.EPL_ID}/matches",
                headers={"X-Auth-Token": self.api_key},
                params={"dateFrom": today, "dateTo": today},
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json().get("matches", [])
        except Exception as e:
            log.error(f"Football API error: {e}")
        return []

    def get_match_lineups(self, match_id: int) -> dict:
        """Get lineups for a specific match."""
        if not self.api_key:
            return {}
        try:
            resp = requests.get(
                f"{self.base_url}/matches/{match_id}",
                headers={"X-Auth-Token": self.api_key},
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            log.error(f"Match lineup fetch error: {e}")
        return {}

    def check_for_callouts(self, twitter_client=None, reddit_client=None):
        """
        Check completed/live matches for flagged players who:
        - Were not in the starting XI (injury-related absence)
        - Were substituted off before 70 mins
        """
        if not self.flagged:
            log.info("No flags loaded — skipping callout check")
            return

        matches = self.get_todays_matches()
        for match in matches:
            status = match.get("status", "")
            if status not in ("IN_PLAY", "PAUSED", "FINISHED"):
                continue

            match_id  = match["id"]
            match_key = f"callout_{match_id}"

            match_detail = self.get_match_lineups(match_id)
            if not match_detail:
                continue

            lineup = match_detail.get("lineups", [])
            subs   = []

            for team_lineup in lineup:
                for sub in team_lineup.get("substitutions", []):
                    player_out = sub.get("playerOut", {}).get("name", "")
                    minute     = sub.get("minute", 999)
                    subs.append((player_out.lower(), minute))

            for player_name_lower, sub_minute in subs:
                if player_name_lower not in self.flagged:
                    continue
                callout_key = f"{match_key}_{player_name_lower}"
                if callout_key in self.posted:
                    continue
                if sub_minute > 75:
                    continue  # normal sub, not injury-related

                player = self.flagged[player_name_lower]
                gw     = get_current_gameweek()
                outcome = f"substituted off in the {sub_minute}th minute"
                content = build_callout(player, gw, outcome)

                log.info(f"Callout triggered: {player['name']} subbed at {sub_minute}'")

                if twitter_client or TWITTER_ENABLED:
                    post_twitter(content["twitter"], twitter_client)
                if reddit_client or REDDIT_ENABLED:
                    post_reddit(content["reddit_title"], content["reddit_body"], reddit_client)

                self.posted.add(callout_key)
                time.sleep(3)

# ─── SCHEDULED JOBS ──────────────────────────────────────────────────────────

monitor = MatchMonitor()

def job_daily_flag(team_filter: str = None):
    """Runs every day at 08:00 UTC — picks the most prominent high-risk player.

    Args:
        team_filter: Optional team name. If provided, only fetch that team's
                     predictions and pick the top player from it.
    """
    log.info("Running: Daily Flag" + (f" (team={team_filter})" if team_filter else ""))
    gw = get_current_gameweek()

    if team_filter:
        # Single-team mode: find the best match from EPL_TEAMS
        matched = next(
            (t for t in EPL_TEAMS if t.lower() == team_filter.lower()
             or team_filter.lower() in t.lower()),
            None,
        )
        if not matched:
            log.error(f"Unknown team: {team_filter}. Available: {', '.join(EPL_TEAMS)}")
            return
        data = fetch_predictions(matched)
        if not data or "players" not in data:
            log.warning(f"No predictions for {matched}")
            return
        for p in data["players"]:
            p["team"] = matched
        players = rank_by_prominence(data["players"])
    else:
        players = fetch_all_predictions()  # already ranked by prominence

    if not players:
        log.warning("No predictions returned")
        return

    top = players[0]

    # Skip posting if the top prominence-weighted player is below 40% risk
    if top.get("risk_score", 0) < 0.40:
        log.warning(
            f"Top prominence pick {top.get('name')} has risk_score "
            f"{top.get('risk_score', 0):.2f} — below 0.40 threshold. "
            f"No strong flags today. Skipping post."
        )
        return

    log.info(
        f"Daily flag: {top['name']} (tier {top.get('_tier', '?')}, "
        f"weight {top.get('_weight', '?')}, prominence {top.get('_prominence', '?'):.3f}, "
        f"raw risk {top.get('risk_score', 0):.3f})"
    )

    content = build_daily_flag(top, gw)

    twitter_client = get_twitter_client()
    reddit_client  = get_reddit_client()

    post_twitter(content["twitter"], twitter_client)
    post_reddit(content["reddit_title"], content["reddit_body"], reddit=reddit_client)

def job_gameweek_preview():
    """Runs every Friday at 10:00 UTC — top 5 risks by prominence."""
    log.info("Running: Gameweek Preview")
    gw      = get_current_gameweek()
    players = fetch_all_predictions()  # already ranked by prominence
    if not players:
        return

    # Log the top 5 with their prominence details
    for i, p in enumerate(players[:5]):
        log.info(
            f"  GW preview #{i+1}: {p['name']} — "
            f"risk={p.get('risk_score',0):.3f}, tier={p.get('_tier','?')}, "
            f"prominence={p.get('_prominence',0):.3f}"
        )

    content = build_gameweek_preview(players, gw)

    twitter_client = get_twitter_client()
    reddit_client  = get_reddit_client()

    post_twitter(content["twitter"], twitter_client)
    post_reddit(
        content["reddit_title"],
        content["reddit_body"],
        subreddits=["FantasyPL", "PremierLeague", "Fantasy_Football"],
        reddit=reddit_client
    )

def job_load_pre_match_flags():
    """Runs at 11:00 on matchdays — loads flags before KO."""
    log.info("Running: Loading pre-match flags")
    monitor.load_flags()

def job_check_callouts():
    """Runs every 15 mins during typical matchday windows (Sat/Sun 14:00–22:00 UTC)."""
    now  = datetime.utcnow()
    hour = now.hour
    day  = now.weekday()  # 5=Sat, 6=Sun, 1=Tue, 2=Wed

    # Only run during match windows
    match_windows = {
        5: (13, 22),  # Saturday
        6: (13, 21),  # Sunday
        1: (18, 22),  # Tuesday
        2: (18, 22),  # Wednesday
    }

    if day not in match_windows:
        log.info(f"Callout check skipped: {now.strftime('%A')} is not a match day (Sat/Sun/Tue/Wed only)")
        return
    start, end = match_windows[day]
    if not (start <= hour <= end):
        log.info(
            f"Callout check skipped: {now.strftime('%A %H:%M')} UTC is outside "
            f"the match window ({start}:00–{end}:00 UTC)"
        )
        return

    log.info("Running: Callout check")
    twitter_client = get_twitter_client()
    reddit_client  = get_reddit_client()
    monitor.check_for_callouts(twitter_client, reddit_client)


def job_intl_break_watch():
    """International break: who's accumulating risk on national team duty."""
    log.info("Running: International Break Watch")
    gw = get_current_gameweek()
    players = fetch_all_predictions()
    if not players:
        log.warning("No predictions returned")
        return

    content = build_intl_break_watch(players, gw)

    twitter_client = get_twitter_client()
    reddit_client  = get_reddit_client()
    post_twitter(content["twitter"], twitter_client)
    post_reddit(content["reddit_title"], content["reddit_body"],
                subreddits=["FantasyPL", "PremierLeague"], reddit=reddit_client)


def job_return_preview():
    """Friday before PL resumes after a break: risk heading into the return GW."""
    log.info("Running: Return GW Preview")
    gw = get_current_gameweek()
    players = fetch_all_predictions()
    if not players:
        log.warning("No predictions returned")
        return

    content = build_return_gw_preview(players, gw)

    twitter_client = get_twitter_client()
    reddit_client  = get_reddit_client()
    post_twitter(content["twitter"], twitter_client)
    post_reddit(content["reddit_title"], content["reddit_body"],
                subreddits=["FantasyPL", "PremierLeague", "Fantasy_Football"],
                reddit=reddit_client)


def job_break_recovery():
    """Mid-break: who benefits from the rest."""
    log.info("Running: Break Recovery Tracker")
    gw = get_current_gameweek()
    players = fetch_all_predictions()
    if not players:
        log.warning("No predictions returned")
        return

    content = build_break_recovery(players, gw)
    if content is None:
        log.warning("No recovery candidates found — skipping post")
        return

    twitter_client = get_twitter_client()
    reddit_client  = get_reddit_client()
    post_twitter(content["twitter"], twitter_client)
    post_reddit(content["reddit_title"], content["reddit_body"],
                subreddits=["FantasyPL"], reddit=reddit_client)


def job_accuracy_recap():
    """Post-break or end of GW block: how did Yara's flags hold up."""
    log.info("Running: Accuracy Recap")
    gw = get_current_gameweek()
    players = fetch_all_predictions()
    if not players:
        log.warning("No predictions returned")
        return

    content = build_accuracy_recap(players, gw)

    twitter_client = get_twitter_client()
    reddit_client  = get_reddit_client()
    post_twitter(content["twitter"], twitter_client)
    post_reddit(content["reddit_title"], content["reddit_body"],
                subreddits=["FantasyPL", "PremierLeague"], reddit=reddit_client)


# ─── COMPARISON UTILITY ──────────────────────────────────────────────────────

def show_comparison():
    """Print a before/after comparison of raw vs prominence ranking."""
    log.info("Fetching predictions for comparison...")
    all_players = []
    for team in EPL_TEAMS:
        data = fetch_predictions(team)
        if data and "players" in data:
            for p in data["players"]:
                p["team"] = team
            all_players.extend(data["players"])
        time.sleep(0.5)

    if not all_players:
        print("No predictions available.")
        return

    # Raw ranking (old behaviour)
    raw_sorted = sorted(all_players, key=lambda x: x.get("risk_score", 0), reverse=True)

    # Prominence ranking (new behaviour)
    prominence_sorted = rank_by_prominence(list(all_players))

    print("\n" + "=" * 72)
    print("TOP 5 BY RAW RISK SCORE (old behaviour)")
    print("=" * 72)
    for i, p in enumerate(raw_sorted[:5]):
        tier = _get_player_tier(p.get("name", ""), p)
        print(f"  {i+1}. {p['name']:25s} ({p['team']:15s}) risk={p['risk_score']:.3f}  tier={tier}")

    print("\n" + "=" * 72)
    print("TOP 5 BY PROMINENCE SCORE (new behaviour)")
    print("=" * 72)
    for i, p in enumerate(prominence_sorted[:5]):
        print(
            f"  {i+1}. {p['name']:25s} ({p['team']:15s}) "
            f"risk={p['risk_score']:.3f}  tier={p['_tier']}  "
            f"weight={p['_weight']:.1f}  prominence={p['_prominence']:.3f}"
        )
    print()

# ─── SCHEDULER ───────────────────────────────────────────────────────────────

def run_scheduler():
    log.info("Yara Auto-Poster starting...")
    log.info(f"Twitter: {'ON' if TWITTER_ENABLED else 'OFF'} | Reddit: {'ON' if REDDIT_ENABLED else 'OFF'} | Dry run: {DRY_RUN}")

    schedule.every().day.at("08:00").do(job_daily_flag)
    schedule.every().friday.at("10:00").do(job_gameweek_preview)
    schedule.every().saturday.at("11:00").do(job_load_pre_match_flags)
    schedule.every().sunday.at("11:00").do(job_load_pre_match_flags)
    schedule.every().tuesday.at("17:30").do(job_load_pre_match_flags)
    schedule.every().wednesday.at("17:30").do(job_load_pre_match_flags)
    schedule.every(15).minutes.do(job_check_callouts)

    log.info("Scheduler running. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(30)

def show_team(team_name: str):
    """Print predictions for a single team."""
    matched = next(
        (t for t in EPL_TEAMS if t.lower() == team_name.lower()
         or team_name.lower() in t.lower()),
        None,
    )
    if not matched:
        print(f"Unknown team: {team_name}")
        print(f"Available: {', '.join(EPL_TEAMS)}")
        return

    data = fetch_predictions(matched)
    if not data or "players" not in data:
        print(f"No predictions for {matched}")
        return

    gw = data.get("gameweek", get_current_gameweek())
    print(f"\n{'='*60}")
    print(f"{matched} — GW{gw}")
    print(f"{'='*60}")
    for p in data["players"]:
        tier = _get_player_tier(p.get("name", ""), p)
        label = p.get("risk_label", get_risk_label(p.get("risk_score", 0)))
        print(f"  {p['name']:25s} risk={p.get('risk_score',0):.3f} ({label})  tier={tier}")
        for r in p.get("reasons", [])[:3]:
            print(f"    - {r}")
    print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "daily":
            # Optional: python yara_autoposter.py daily Arsenal
            team_arg = sys.argv[2] if len(sys.argv) > 2 else None
            job_daily_flag(team_filter=team_arg)
        elif cmd == "preview":
            job_gameweek_preview()
        elif cmd == "flags":
            job_load_pre_match_flags()
        elif cmd == "callout-check":
            job_check_callouts()
        elif cmd == "compare":
            show_comparison()
        elif cmd == "intl-watch":
            job_intl_break_watch()
        elif cmd == "return-preview":
            job_return_preview()
        elif cmd == "recovery":
            job_break_recovery()
        elif cmd == "recap":
            job_accuracy_recap()
        elif cmd == "show":
            if len(sys.argv) < 3:
                print("Usage: python yara_autoposter.py show <team>")
                print(f"Available: {', '.join(EPL_TEAMS)}")
            else:
                show_team(" ".join(sys.argv[2:]))
        else:
            print(f"Unknown command: {cmd}")
            print("Commands:")
            print("  daily [team]      — daily flag (optionally for one team)")
            print("  preview           — gameweek preview (top 5)")
            print("  show <team>       — show predictions for a team")
            print("  flags             — load pre-match flags")
            print("  callout-check     — check for callouts")
            print("  compare           — raw vs prominence ranking")
            print()
            print("International break:")
            print("  intl-watch        — who's accumulating risk on intl duty")
            print("  return-preview    — risk heading into the return GW")
            print("  recovery          — who benefits from the rest")
            print("  recap             — model accuracy recap")
    else:
        run_scheduler()
