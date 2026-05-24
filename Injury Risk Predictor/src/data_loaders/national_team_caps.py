"""National-team caps + goals from Wikipedia.

Each WC nation's Wikipedia page has a "Current squad" section containing a
wikitable with columns ``No. | Pos. | Player | DOB | Caps | Goals | Club``.
We scrape that table for every WC 2026 nation and produce a lookup keyed by
``(country, normalised_name)`` that the build script joins onto the
international inference rows.

Why Wikipedia over Transfermarkt:
- TM doesn't expose caps on its per-country squad pages — caps live only on
  per-player profile sidebars, which would mean ~1200 fetches at 3s rate
  limit (~2hrs first run). Wikipedia is 48 fetches, no rate limit.
- The Wikipedia tables are well-structured and updated within hours of
  national-team squad announcements.

Resilience: each country is fetched independently; failures log and return
empty rather than failing the whole build. Country name → page title aliases
handle the cases where the WC nation name doesn't match the Wikipedia article
title (e.g. ``Bosnia-Herzegovina`` → ``Bosnia_and_Herzegovina``).
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from ..utils.logger import get_logger

logger = get_logger(__name__)

WIKI_BASE = "https://en.wikipedia.org/wiki/"
USER_AGENT = "YaraSports/1.0 (injury-risk-predictor; gakpovwovwo@gmail.com)"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "wikipedia_national_squads"
CACHE_TTL_HOURS = 24  # squads can change daily during tournament prep
RATE_LIMIT_SEC = 0.5  # Wikipedia is generous but be polite

# Map WC nation name -> Wikipedia article title fragment when they differ.
COUNTRY_PAGE_ALIASES: Dict[str, str] = {
    "United States": "United_States_men%27s_national_soccer_team",
    "Australia": "Australia_men%27s_national_soccer_team",
    "Canada": "Canada_men%27s_national_soccer_team",
    "Bosnia-Herzegovina": "Bosnia_and_Herzegovina_national_football_team",
    "Czechia": "Czech_Republic_national_football_team",
    "Cape Verde Islands": "Cape_Verde_national_football_team",
    "Congo DR": "DR_Congo_national_football_team",
    "Curaçao": "Curaçao_national_football_team",
    "Ivory Coast": "Ivory_Coast_national_football_team",
    "South Korea": "South_Korea_national_football_team",
    "South Africa": "South_Africa_national_football_team",
    "Saudi Arabia": "Saudi_Arabia_national_football_team",
    "New Zealand": "New_Zealand_national_football_team",
}


_last_request_ts = 0.0


def _rate_limit() -> None:
    global _last_request_ts
    elapsed = time.time() - _last_request_ts
    if elapsed < RATE_LIMIT_SEC:
        time.sleep(RATE_LIMIT_SEC - elapsed)
    _last_request_ts = time.time()


def _normalise_name(name: str) -> str:
    stripped = "".join(c for c in unicodedata.normalize("NFKD", name or "") if not unicodedata.combining(c))
    # Drop trailing reference numbers like "[1]" Wikipedia sometimes injects.
    stripped = re.sub(r"\[\d+\]", "", stripped)
    return stripped.lower().strip()


def _candidate_page_titles(country: str) -> List[str]:
    if country in COUNTRY_PAGE_ALIASES:
        return [COUNTRY_PAGE_ALIASES[country]]
    safe = country.replace(" ", "_")
    return [
        f"{safe}_national_football_team",
        f"{safe}_men%27s_national_football_team",
    ]


def _cache_path(title: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9_-]+", "_", title)[:120]
    return CACHE_DIR / f"{key}.html"


def _get_cached(title: str) -> Optional[str]:
    p = _cache_path(title)
    if not p.exists():
        return None
    age_hours = (time.time() - p.stat().st_mtime) / 3600
    if age_hours > CACHE_TTL_HOURS:
        return None
    return p.read_text(encoding="utf-8")


def _fetch_page(title: str) -> Optional[str]:
    cached = _get_cached(title)
    if cached is not None:
        return cached
    _rate_limit()
    url = WIKI_BASE + title
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Wikipedia fetch failed for %s: %s", title, exc)
        return None
    _cache_path(title).write_text(resp.text, encoding="utf-8")
    return resp.text


def _find_squad_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """Return the wikitable under the first heading mentioning 'squad' that
    has Player + Caps columns."""
    candidates = []
    for h in soup.find_all(["h2", "h3"]):
        text = h.get_text(strip=True).lower()
        if "squad" not in text:
            continue
        # "Current squad" > "Latest squad" > "Most recent call-up" preference.
        priority = 0
        if "current" in text:
            priority = 3
        elif "latest" in text or "most recent" in text:
            priority = 2
        else:
            priority = 1
        candidates.append((priority, h))
    candidates.sort(key=lambda x: -x[0])
    for _, h in candidates:
        # Walk forward through siblings to find a wikitable; some pages wrap
        # the table inside intermediate <div> blocks.
        node = h
        for _ in range(20):
            node = node.find_next(["table", "h2"])
            if node is None or node.name == "h2":
                break
            classes = node.get("class") or []
            if "wikitable" in classes:
                headers = [th.get_text(strip=True).lower() for th in node.select("tr:nth-of-type(1) th")]
                if any("caps" in h_ for h_ in headers) and any("player" in h_ for h_ in headers):
                    return node
    return None


def _parse_int(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"-?\d+", text.replace(",", ""))
    return int(m.group(0)) if m else None


def scrape_country_caps(country: str) -> Dict[str, Dict[str, Optional[int]]]:
    """Return ``{normalised_name: {"caps": int, "intl_goals": int}}`` for a country."""
    html: Optional[str] = None
    for title in _candidate_page_titles(country):
        html = _fetch_page(title)
        if html is not None:
            break
    if html is None:
        logger.warning("No Wikipedia page found for %s", country)
        return {}

    soup = BeautifulSoup(html, "html.parser")
    table = _find_squad_table(soup)
    if table is None:
        logger.warning("No 'Current squad' table on Wikipedia page for %s", country)
        return {}

    headers = [th.get_text(strip=True).lower() for th in table.select("tr:nth-of-type(1) th")]
    try:
        player_idx = next(i for i, h in enumerate(headers) if "player" in h)
        caps_idx = next(i for i, h in enumerate(headers) if "caps" in h)
    except StopIteration:
        logger.warning("Expected Player/Caps columns missing for %s; headers=%s", country, headers)
        return {}
    goals_idx = next((i for i, h in enumerate(headers) if "goals" in h), None)

    out: Dict[str, Dict[str, Optional[int]]] = {}
    for row in table.select("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) <= max(player_idx, caps_idx):
            continue
        name = cells[player_idx].get_text(" ", strip=True)
        name = re.sub(r"\(.*?\)", "", name).strip()  # drop suffixes like "(captain)"
        if not name:
            continue
        caps = _parse_int(cells[caps_idx].get_text(strip=True))
        goals = _parse_int(cells[goals_idx].get_text(strip=True)) if goals_idx is not None else None
        out[_normalise_name(name)] = {"caps": caps, "intl_goals": goals}

    logger.info("Wikipedia caps for %s: %d players parsed", country, len(out))
    return out


def build_caps_lookup(countries: List[str]) -> Dict[Tuple[str, str], Dict[str, Optional[int]]]:
    """Aggregate per-country caps into a ``(country, normalised_name)`` map."""
    lookup: Dict[Tuple[str, str], Dict[str, Optional[int]]] = {}
    for country in countries:
        per_country = scrape_country_caps(country)
        for nname, stats in per_country.items():
            lookup[(country, nname)] = stats
    logger.info("Built caps lookup for %d country/player pairs across %d countries",
                len(lookup), len(countries))
    return lookup
