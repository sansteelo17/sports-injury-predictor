"""Grounded football news retrieval from trusted RSS feeds.

Hybrid-architecture data source. We fetch ONLY from a curated allowlist of
reputable outlets, so every surfaced item carries a verifiable source and a
URL the reader can open. This is the "verified" in "verified rumors": an item
is shown because a trusted outlet reported it, with attribution — not because a
model decided it was true. The narrative layer may summarise these items but it
must never source or invent them. If it is not in a feed here, it does not
appear.

Dependency-light on purpose: RSS/Atom are XML, so we parse with the stdlib
ElementTree rather than adding feedparser. Feeds are fetched once per TTL and
filtered per player in memory, so a player view never makes N HTTP calls.
"""

from __future__ import annotations

import html
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Curated allowlist of reputable outlets. The string is the attribution shown
# in the UI. Add only sources you would be comfortable citing by name.
TRUSTED_FEEDS: List[tuple] = [
    ("BBC Sport", "https://feeds.bbci.co.uk/sport/football/rss.xml"),
    ("The Guardian", "https://www.theguardian.com/football/rss"),
    ("Sky Sports", "https://www.skysports.com/rss/12040"),
    ("ESPN", "https://www.espn.com/espn/rss/soccer/news"),
]

_FEED_TTL_SECONDS = 1800  # 30 min — news moves, but not faster than this matters
_REQUEST_TIMEOUT = 6.0
# url -> {"expires": epoch, "items": List[NewsItem]}
_feed_cache: Dict[str, Dict] = {}

# Common name tokens that should never be the thing we match on alone.
_STOPWORD_TOKENS = {"de", "da", "do", "van", "von", "el", "al", "the", "dos", "di"}


@dataclass
class NewsItem:
    title: str
    summary: str
    source: str
    url: str
    published: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def _normalize(text: str) -> str:
    """Lowercase and strip accents so 'Martín' matches 'martin'."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.lower()


def _strip_html(text: str) -> str:
    if not text:
        return ""
    no_tags = re.sub(r"<[^>]+>", "", text)
    return html.unescape(no_tags).strip()


def _parse_feed(source: str, url: str) -> List[NewsItem]:
    """Fetch and parse one RSS/Atom feed into NewsItems. Never raises."""
    try:
        resp = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "YaraSports/1.0 (+https://yaraspeaks.com)"},
        )
        if resp.status_code != 200:
            logger.warning("News feed %s returned HTTP %s", source, resp.status_code)
            return []
        root = ET.fromstring(resp.content)
    except Exception as e:  # network, XML, anything — degrade to no news
        logger.warning("News feed %s failed: %s", source, e)
        return []

    items: List[NewsItem] = []
    # RSS 2.0: channel/item with title/link/description/pubDate.
    # Atom: feed/entry with title/link(href)/summary/updated. Handle both.
    for node in root.iter():
        tag = node.tag.split("}")[-1].lower()
        if tag not in ("item", "entry"):
            continue
        title = link = summary = published = ""
        for child in node:
            ctag = child.tag.split("}")[-1].lower()
            if ctag == "title":
                title = (child.text or "").strip()
            elif ctag == "link":
                # RSS puts the URL in text; Atom in the href attribute.
                link = (child.text or "").strip() or child.attrib.get("href", "").strip()
            elif ctag in ("description", "summary"):
                summary = _strip_html(child.text or "")
            elif ctag in ("pubdate", "published", "updated"):
                published = (child.text or "").strip()
        if title and link:
            items.append(
                NewsItem(
                    title=_strip_html(title),
                    summary=summary[:300],
                    source=source,
                    url=link,
                    published=published or None,
                )
            )
    return items


def _get_feed_cached(source: str, url: str) -> List[NewsItem]:
    now = time.time()
    cached = _feed_cache.get(url)
    if cached and cached["expires"] > now:
        return cached["items"]
    items = _parse_feed(source, url)
    # Cache even an empty result briefly so a flaky feed doesn't get hammered.
    _feed_cache[url] = {"expires": now + _FEED_TTL_SECONDS, "items": items}
    return items


def _name_tokens(player_name: str) -> List[str]:
    toks = [t for t in re.split(r"[\s\-']+", _normalize(player_name)) if len(t) >= 3 and t not in _STOPWORD_TOKENS]
    return toks


def _matches_player(item: NewsItem, tokens: List[str], team_norm: str) -> bool:
    """Conservative match: require the surname plus a corroborating token.

    Word-overlap rather than single-surname matching — the Zubimendi/Raya
    lesson. A lone surname ('Henderson') in a general football feed is too weak,
    so we require the surname AND (a forename token OR the country/club name).
    """
    if not tokens:
        return False
    haystack = _normalize(item.title + " " + item.summary)
    surname = tokens[-1]
    if surname not in haystack:
        return False
    if len(tokens) == 1:
        return True  # mononym — surname is all we have
    corroborated = any(t in haystack for t in tokens[:-1])
    if team_norm and len(team_norm) >= 4:
        corroborated = corroborated or team_norm in haystack
    return corroborated


def fetch_player_news(player_name: str, team: Optional[str] = None, limit: int = 4) -> List[Dict]:
    """Return up to ``limit`` trusted, attributed news items mentioning the player.

    Returns plain dicts (JSON-ready). Empty list on no match or any failure —
    the card simply shows no news rather than fabricating one.
    """
    tokens = _name_tokens(player_name)
    if not tokens:
        return []
    team_norm = _normalize(team) if team else ""

    matched: List[NewsItem] = []
    seen_urls = set()
    for source, url in TRUSTED_FEEDS:
        for item in _get_feed_cached(source, url):
            if item.url in seen_urls:
                continue
            if _matches_player(item, tokens, team_norm):
                matched.append(item)
                seen_urls.add(item.url)

    return [m.to_dict() for m in matched[:limit]]
