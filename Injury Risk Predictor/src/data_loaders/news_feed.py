"""Grounded football news retrieval, per player.

Hybrid-architecture data source. We query Google News (an aggregator over
hundreds of outlets) for the specific player, so a marquee name on a big team
reliably has recent, attributed coverage — not just the few players who happen
to appear on a homepage feed. Every item carries its originating outlet and a
link the reader can open: that attribution is the "verified" in verified news.
The narrative layer may summarise these items but must never source or invent
them. If it is not returned here, it does not appear.

Dependency-light: Google News exposes RSS, parsed with the stdlib ElementTree.
Results are cached per player so a view never re-hits the network within the
TTL.
"""

from __future__ import annotations

import html
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
_TTL_SECONDS = 1800  # 30 min
_REQUEST_TIMEOUT = 6.0
# normalized "name|team" -> {"expires": epoch, "items": List[dict]}
_news_cache: Dict[str, Dict] = {}

# Outlets we are comfortable presenting as "verified" reporting. Items from
# these are kept ahead of the rest; everything still shows its source so the
# reader can judge, but reputable sources lead.
REPUTABLE_SOURCES = {
    "bbc", "the guardian", "guardian", "sky sports", "espn", "the athletic",
    "reuters", "the times", "the telegraph", "independent", "goal", "bbc sport",
    "fabrizio romano", "athletic", "evening standard", "manchester evening news",
    "liverpool echo", "football.london", "the mirror", "mirror",
}


def _normalize(text: str) -> str:
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.lower()


def _strip_html(text: str) -> str:
    if not text:
        return ""
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


@dataclass
class NewsItem:
    title: str
    summary: str
    source: str
    url: str
    published: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def _parse_google_news(xml_bytes: bytes) -> List[NewsItem]:
    """Parse a Google News RSS search result. Never raises."""
    try:
        root = ET.fromstring(xml_bytes)
    except Exception as e:
        logger.warning("Google News parse failed: %s", e)
        return []

    items: List[NewsItem] = []
    for node in root.iter():
        if node.tag.split("}")[-1].lower() != "item":
            continue
        title = link = published = source = ""
        for child in node:
            ctag = child.tag.split("}")[-1].lower()
            if ctag == "title":
                title = (child.text or "").strip()
            elif ctag == "link":
                link = (child.text or "").strip() or child.attrib.get("href", "").strip()
            elif ctag == "pubdate":
                published = (child.text or "").strip()
            elif ctag == "source":
                # Google News carries the outlet in a <source> element.
                source = (child.text or "").strip()
        # Google News titles are "Headline - Outlet"; fall back to that split.
        if not source and " - " in title:
            source = title.rsplit(" - ", 1)[-1].strip()
        headline = title.rsplit(" - ", 1)[0].strip() if (" - " in title and source) else title
        if headline and link:
            items.append(
                NewsItem(
                    title=_strip_html(headline),
                    summary="",
                    source=source or "Google News",
                    url=link,
                    published=published or None,
                )
            )
    return items


def _query_for(player_name: str, team: Optional[str]) -> str:
    # Quote the name so we match the person, not loose tokens; add team +
    # "football" to disambiguate common names.
    parts = [f'"{player_name}"']
    if team:
        parts.append(team)
    parts.append("football")
    return " ".join(parts)


def _rank(item: NewsItem) -> int:
    return 0 if _normalize(item.source) in REPUTABLE_SOURCES else 1


def fetch_player_news(player_name: str, team: Optional[str] = None, limit: int = 4) -> List[Dict]:
    """Return up to ``limit`` attributed news items about the player.

    Reputable outlets lead; every item keeps its source and link. Empty list on
    no result or any failure — the card shows nothing rather than fabricating.
    """
    if not player_name or len(player_name.strip()) < 3:
        return []
    cache_key = f"{_normalize(player_name)}|{_normalize(team or '')}"
    now = time.time()
    cached = _news_cache.get(cache_key)
    if cached and cached["expires"] > now:
        return cached["items"]

    url = (
        f"{GOOGLE_NEWS_RSS}?q={quote_plus(_query_for(player_name, team))}"
        "&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        resp = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "YaraSports/1.0 (+https://yaraspeaks.com)"},
        )
        items = _parse_google_news(resp.content) if resp.status_code == 200 else []
    except Exception as e:
        logger.warning("News fetch failed for %s: %s", player_name, e)
        items = []

    # Drop obvious dupes, prefer reputable sources, cap to limit.
    seen = set()
    deduped: List[NewsItem] = []
    for it in sorted(items, key=_rank):
        key = it.url
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    result = [d.to_dict() for d in deduped[:limit]]

    _news_cache[cache_key] = {"expires": now + _TTL_SECONDS, "items": result}
    return result


# Availability/fitness terms: this is the matchup-relevant angle our structured
# data cannot know (the dossier knows form, workload, H2H — not whether he is
# actually passed fit to start). Used both to target the query and to flag which
# returned items are matchup-relevant.
_FITNESS_TERMS = (
    "injury", "injured", "fitness", "fit", "doubt", "doubtful", "return",
    "returns", "sidelined", "out", "ruled out", "knock", "setback", "lineup",
    "line-up", "available", "availability", "suspended", "suspension", "rested",
)


def _matchup_query_for(player_name: str, team: Optional[str]) -> str:
    """Query targeted at the player's availability/fitness for the next match.

    Same trusted aggregator as ``fetch_player_news``; only the query is narrowed
    to the fitness/availability angle. Still attributed, still never invented.
    """
    parts = [f'"{player_name}"']
    if team:
        parts.append(team)
    parts.append("(injury OR fitness OR doubt OR return OR fit OR sidelined OR lineup)")
    parts.append("football")
    return " ".join(parts)


def _player_mentioned(headline_norm: str, player_name: str) -> bool:
    """Guard against off-target items: require a real name token (the surname,
    or any name token >=4 chars) to appear in the normalized headline."""
    tokens = [t for t in _normalize(player_name).split() if len(t) >= 3]
    if not tokens:
        return False
    surname = tokens[-1]
    if surname in headline_norm:
        return True
    return any(len(t) >= 4 and t in headline_norm for t in tokens)


def _is_fitness_item(headline_norm: str) -> bool:
    return any(term in headline_norm for term in _FITNESS_TERMS)


def fetch_matchup_news(
    player_name: str,
    team: Optional[str] = None,
    opponent: Optional[str] = None,
    limit: int = 3,
) -> List[Dict]:
    """Return attributed news items targeted at the player's availability/fitness.

    The structured matchup dossier knows form/workload/H2H but not whether the
    player is actually fit to start — that lives in late reporting. This narrows
    the same trusted Google News aggregator to the fitness/availability angle,
    keeps only items that genuinely name the player, and leads with
    matchup-relevant (fitness) items from reputable outlets. Each item gains a
    ``matchup_relevant`` flag. Attribution preserved; empty list on any failure
    so the narrative shows nothing rather than fabricating.
    """
    if not player_name or len(player_name.strip()) < 3:
        return []
    cache_key = f"matchup|{_normalize(player_name)}|{_normalize(team or '')}"
    now = time.time()
    cached = _news_cache.get(cache_key)
    if cached and cached["expires"] > now:
        return cached["items"]

    url = (
        f"{GOOGLE_NEWS_RSS}?q={quote_plus(_matchup_query_for(player_name, team))}"
        "&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        resp = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "YaraSports/1.0 (+https://yaraspeaks.com)"},
        )
        items = _parse_google_news(resp.content) if resp.status_code == 200 else []
    except Exception as e:
        logger.warning("Matchup news fetch failed for %s: %s", player_name, e)
        items = []

    opp_norm = _normalize(opponent or "")
    scored = []
    seen = set()
    for it in items:
        if it.url in seen:
            continue
        headline_norm = _normalize(it.title)
        if not _player_mentioned(headline_norm, player_name):
            continue  # off-target: drop rather than risk a wrong-player item
        seen.add(it.url)
        fitness = _is_fitness_item(headline_norm)
        mentions_opp = bool(opp_norm) and opp_norm in headline_norm
        reputable = _normalize(it.source) in REPUTABLE_SOURCES
        # Sort key (ascending): fitness items first, then opponent mentions, then
        # reputable outlets; feed order (roughly recency) breaks ties.
        sort_key = (0 if fitness else 1, 0 if mentions_opp else 1, 0 if reputable else 1)
        d = it.to_dict()
        d["matchup_relevant"] = fitness
        scored.append((sort_key, d))

    scored.sort(key=lambda x: x[0])
    result = [d for _, d in scored[:limit]]
    _news_cache[cache_key] = {"expires": now + _TTL_SECONDS, "items": result}
    return result
