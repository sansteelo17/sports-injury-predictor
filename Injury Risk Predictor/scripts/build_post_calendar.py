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

# Big nations: a fixture between two of these gets a battle-card reminder.
_MARQUEE = {
    "Brazil", "Argentina", "France", "England", "Spain", "Germany", "Portugal",
    "Netherlands", "Belgium", "Croatia", "Uruguay", "Mexico", "United States",
    "Italy", "Colombia", "Morocco", "Japan", "Senegal",
}


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

    def _event(uid, start, summary, desc, alarms):
        lines.extend([
            "BEGIN:VEVENT", f"UID:{uid}@yaraspeaks.com", f"DTSTAMP:{now}",
            f"DTSTART:{_ics_dt(start)}", f"DTEND:{_ics_dt(start + timedelta(minutes=30))}",
            f"SUMMARY:{_esc(summary)}", f"DESCRIPTION:{_esc(desc)}",
        ])
        for trig, label in alarms:
            lines.extend(["BEGIN:VALARM", f"TRIGGER:{trig}", "ACTION:DISPLAY",
                          f"DESCRIPTION:{_esc(label)}", "END:VALARM"])
        lines.append("END:VEVENT")

    _RUN = "python -m social.autopost.run --format {fmt} --competition world-cup-2026"
    _BOARD_ALARMS = [("-P1D", "Yara board tomorrow"), ("-PT2H", "Yara board in 2h"),
                     ("-PT1H", "Yara board in 1h")]

    for day, grp in fx.groupby("day"):
        grp = grp.sort_values("utc_date")
        first = grp.iloc[0]
        kickoff = first["utc_date"].to_pydatetime()
        n = len(grp)
        stage = str(first.get("stage") or "").replace("_", " ").title() or "World Cup"
        headline = f"{first['home_team']} vs {first['away_team']}"

        # 1) Matchday board, ~2h before first kickoff, with the day-before + 2h/1h pings.
        _event(f"yara-board-{day}", kickoff,
               f"Yara: WC board, {stage} ({n} game{'s' if n != 1 else ''})",
               f"Post the matchday board for {day}. First up: {headline}. "
               f"Run: {_RUN.format(fmt='matchday_board')} --matchday \"{stage}\" --date {day}",
               _BOARD_ALARMS)

        # 2) Accountability the next morning (score the day's calls).
        acct = (kickoff.replace(hour=9, minute=0, second=0) + timedelta(days=1))
        _event(f"yara-acct-{day}", acct, f"Yara: score {day} (How We Did)",
               f"Score yesterday's calls and post hits/misses. "
               f"Run: {_RUN.format(fmt='accountability')} --date {day}",
               [("-PT15M", "Yara accountability in 15m")])

        # 3) Battle card on marquee days (a fixture between two big nations).
        marquee = grp[grp.apply(lambda r: r["home_team"] in _MARQUEE and r["away_team"] in _MARQUEE, axis=1)]
        if not marquee.empty:
            m = marquee.iloc[0]
            bk = m["utc_date"].to_pydatetime() - timedelta(hours=3)
            _event(f"yara-battle-{day}", bk,
                   f"Yara: battle card, {m['home_team']} vs {m['away_team']}",
                   f"Post the marquee head-to-head. Run: {_RUN.format(fmt='battle_card')}",
                   [("-PT1H", "Yara battle card in 1h")])

    # 4) Weekly Riskiest XI (Mondays) + risk-spike check, across the tournament span.
    days = sorted(fx["day"].unique())
    d0 = datetime.strptime(days[0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    d1 = datetime.strptime(days[-1], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    cur = d0
    while cur <= d1:
        if cur.weekday() == 0:  # Monday
            noon = cur.replace(hour=12)
            _event(f"yara-xi-{cur:%Y%m%d}", noon, "Yara: Riskiest XI of the tournament",
                   f"Post the tournament riskiest XI. Run: {_RUN.format(fmt='riskiest_xi')}",
                   [("-PT30M", "Yara riskiest XI in 30m")])
        cur += timedelta(days=1)

    lines.append("END:VCALENDAR")
    OUT.write_text("\r\n".join(lines) + "\r\n")
    print(f"Wrote {OUT} ({fx['day'].nunique()} matchday events, "
          f"{fx['day'].min()} to {fx['day'].max()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
