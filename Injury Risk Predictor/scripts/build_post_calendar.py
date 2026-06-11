"""Generate an .ics calendar of Yara World Cup posting reminders.

One event per matchday (anchored at that day's first kickoff), each carrying
three alarms: one day before, two hours before, one hour before. Import the
output into Google Calendar (Settings > Import & export > Import).

Run:
    python scripts/build_post_calendar.py
Output:
    social/yara_wc_post_schedule.ics
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "processed" / "world_cup_2026_fixtures.pkl"
OUT = ROOT / "social" / "yara_wc_post_schedule.ics"


def _ics_dt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;")


def main() -> int:
    fx = pd.read_pickle(FIXTURES)
    fx = fx.dropna(subset=["utc_date"]).copy()
    fx["utc_date"] = pd.to_datetime(fx["utc_date"], utc=True)
    fx["day"] = fx["utc_date"].dt.strftime("%Y-%m-%d")

    now = _ics_dt(datetime.now(timezone.utc))
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//YaraSports//WC 2026 Post Schedule//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "X-WR-CALNAME:Yara WC 2026 Posts",
    ]

    for day, grp in fx.groupby("day"):
        grp = grp.sort_values("utc_date")
        first = grp.iloc[0]
        kickoff = first["utc_date"].to_pydatetime()
        n = len(grp)
        stage = str(first.get("stage") or "").replace("_", " ").title() or "World Cup"
        headline = f"{first['home_team']} vs {first['away_team']}"
        more = f" +{n - 1} more" if n > 1 else ""
        summary = f"Yara post: WC {stage} ({n} game{'s' if n != 1 else ''})"
        desc = (
            f"Post the matchday risk board for {day}. First up: {headline}{more}. "
            f"Run: python -m social.autopost.run --format matchday_board "
            f"--competition world-cup-2026 --matchday \"{stage}\" --date {day}"
        )
        end = kickoff + timedelta(minutes=30)
        lines += [
            "BEGIN:VEVENT",
            f"UID:yara-wc-{day}@yaraspeaks.com",
            f"DTSTAMP:{now}",
            f"DTSTART:{_ics_dt(kickoff)}",
            f"DTEND:{_ics_dt(end)}",
            f"SUMMARY:{_esc(summary)}",
            f"DESCRIPTION:{_esc(desc)}",
        ]
        # Day-before, then 2h and 1h before first kickoff.
        for trig, label in (("-P1D", "tomorrow"), ("-PT2H", "in 2 hours"), ("-PT1H", "in 1 hour")):
            lines += [
                "BEGIN:VALARM",
                f"TRIGGER:{trig}",
                "ACTION:DISPLAY",
                f"DESCRIPTION:{_esc(f'Yara WC post {label}: {headline}')}",
                "END:VALARM",
            ]
        lines.append("END:VEVENT")

    lines.append("END:VCALENDAR")
    OUT.write_text("\r\n".join(lines) + "\r\n")
    print(f"Wrote {OUT} ({fx['day'].nunique()} matchday events, "
          f"{fx['day'].min()} to {fx['day'].max()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
