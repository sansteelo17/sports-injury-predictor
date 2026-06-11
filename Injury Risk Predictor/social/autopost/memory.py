"""SQLite memory for the autopost pipeline.

Local, zero-setup: the file and its tables are created on first use. Powers
week-over-week risk deltas (spike alerts), the 72h per-player/per-format
cooldown, and the accountability ledger. No keys, no network, no rate limits,
which is what a single local cron actually wants.

If you ever need a cloud-viewable copy, add an exporter that reads these tables
and pushes to Airtable/Notion; the engine stays local either way.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from typing import Dict, List, Optional, Tuple

from . import config

DB_PATH = config.ROOT / "data" / "yara_memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS risk_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  player_name TEXT NOT NULL,
  competition TEXT NOT NULL,
  date        TEXT NOT NULL,   -- YYYY-MM-DD of the post
  risk_pct    INTEGER,
  risk_level  TEXT,
  archetype   TEXT,
  post_type   TEXT,
  logged_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_risk_player ON risk_log(player_name, competition, date);

CREATE TABLE IF NOT EXISTS posts (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  player_name TEXT NOT NULL,
  post_type   TEXT NOT NULL,
  posted_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_posts ON posts(player_name, post_type, posted_at);

CREATE TABLE IF NOT EXISTS calls (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  player_name    TEXT NOT NULL,
  competition    TEXT NOT NULL,
  match_date     TEXT,
  predicted_risk INTEGER,
  claim          TEXT,
  outcome        TEXT,         -- started / subbed / injured / fine / unknown
  landed         INTEGER,      -- 1 hit, 0 miss, NULL pending
  created_at     TEXT NOT NULL,
  scored_at      TEXT
);
"""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


# ── risk_log: history for deltas ────────────────────────────────────────────
def log_board(picks: List[Dict], competition: str, date: str, post_type: str) -> None:
    """Snapshot a post's picks so future runs can diff risk week over week."""
    now = _now()
    with _conn() as c:
        c.executemany(
            "INSERT INTO risk_log (player_name,competition,date,risk_pct,risk_level,"
            "archetype,post_type,logged_at) VALUES (?,?,?,?,?,?,?,?)",
            [(p.get("player_name"), competition, date, p.get("risk_score_pct"),
              p.get("risk_level"), p.get("archetype"), post_type, now) for p in picks],
        )


def last_risk(player: str, competition: str, before_date: str) -> Optional[Tuple[int, str]]:
    """Most recent (risk_pct, date) logged for the player strictly before
    ``before_date``. None if we have no prior reading."""
    with _conn() as c:
        row = c.execute(
            "SELECT risk_pct, date FROM risk_log WHERE player_name=? AND competition=? "
            "AND date < ? ORDER BY date DESC, id DESC LIMIT 1",
            (player, competition, before_date),
        ).fetchone()
    return (row["risk_pct"], row["date"]) if row else None


# ── posts: cooldown ─────────────────────────────────────────────────────────
def record_post(player: str, post_type: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO posts (player_name,post_type,posted_at) VALUES (?,?,?)",
                  (player, post_type, _now()))


def on_cooldown(player: str, post_type: str, hours: int = 72) -> bool:
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)).isoformat()
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM posts WHERE player_name=? AND post_type=? AND posted_at > ? LIMIT 1",
            (player, post_type, cutoff),
        ).fetchone()
    return row is not None


# ── calls: accountability ledger ────────────────────────────────────────────
def record_calls(picks: List[Dict], competition: str, match_date: str) -> None:
    now = _now()
    with _conn() as c:
        c.executemany(
            "INSERT INTO calls (player_name,competition,match_date,predicted_risk,claim,"
            "outcome,landed,created_at) VALUES (?,?,?,?,?,?,?,?)",
            [(p.get("player_name"), competition, match_date, p.get("risk_score_pct"),
              p.get("signal_one_liner") or p.get("x_factor"), None, None, now) for p in picks],
        )


def pending_calls(competition: str, match_date: str) -> List[Dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM calls WHERE competition=? AND match_date=? AND landed IS NULL",
            (competition, match_date),
        ).fetchall()
    return [dict(r) for r in rows]


def score_call(call_id: int, outcome: str, landed: bool) -> None:
    with _conn() as c:
        c.execute("UPDATE calls SET outcome=?, landed=?, scored_at=? WHERE id=?",
                  (outcome, 1 if landed else 0, _now(), call_id))
