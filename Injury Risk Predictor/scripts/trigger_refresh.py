#!/usr/bin/env python3
"""
Trigger remote prediction refresh endpoint (for Render cron).
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def main() -> int:
    base_url = (os.environ.get("REFRESH_TRIGGER_URL") or "").strip()
    mode = (os.environ.get("REFRESH_MODE") or "api").strip().lower()
    token = (os.environ.get("REFRESH_CRON_TOKEN") or "").strip()

    if not base_url:
        print("REFRESH_TRIGGER_URL is required")
        return 1
    if mode not in {"api", "fbref"}:
        print("REFRESH_MODE must be 'api' or 'fbref'")
        return 1

    url = f"{base_url}?{urllib.parse.urlencode({'mode': mode})}"
    req = urllib.request.Request(url, method="POST")
    req.add_header("Accept", "application/json")
    if token:
        req.add_header("X-Refresh-Token", token)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"HTTP {resp.status}")
            if body:
                try:
                    parsed = json.loads(body)
                    print(json.dumps(parsed, indent=2))
                except Exception:
                    print(body)
            return 0 if 200 <= resp.status < 300 else 1
    except urllib.error.HTTPError as err:
        payload = err.read().decode("utf-8", errors="replace")
        print(f"HTTP {err.code}")
        if payload:
            print(payload)
        return 1
    except Exception as err:
        print(f"Request failed: {err}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
