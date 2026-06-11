"""Broad player-photo fallback via Wikipedia.

The API's get_player_image_url covers EPL (FPL CDN) + La Liga / transferred
players (Transfermarkt map) well, but not the rest of the world (Liga MX, MLS,
Saudi, much of Serie A). Wikipedia's pageimages has a photo for almost every
international, keyed by name, free, no key. We only call it for the handful of
players that actually land on a card, and cache per process.
"""
from __future__ import annotations

from typing import Dict, Optional

import requests

_UA = "YaraSports/1.0 (+https://yaraspeaks.com)"
_cache: Dict[str, Optional[str]] = {}


def wikipedia_photo(name: str) -> Optional[str]:
    if not name:
        return None
    if name in _cache:
        return _cache[name]
    url: Optional[str] = None
    try:
        # Scope the search to "footballer" so a common name resolves to the
        # player's page, not a namesake, and grab that page's lead image.
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": f"{name} footballer",
                "gsrlimit": 1,
                "prop": "pageimages",
                "piprop": "thumbnail",
                "pithumbsize": 256,
                "format": "json",
                "redirects": 1,
            },
            headers={"User-Agent": _UA},
            timeout=8,
        )
        pages = (resp.json().get("query", {}) or {}).get("pages", {}) or {}
        for _, pg in pages.items():
            thumb = (pg.get("thumbnail") or {}).get("source")
            if thumb:
                url = thumb
                break
    except Exception:
        url = None
    _cache[name] = url
    return url
