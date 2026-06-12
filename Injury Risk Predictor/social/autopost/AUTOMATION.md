# Hands-off automation (local Mac)

Two pieces, both already installed on this machine:

## 1. Keep-alive (launchd) — keeps the API running
`~/Library/LaunchAgents/com.yarasports.api.plist` runs `uvicorn api.main:app
--port 8000` with `RunAtLoad` + `KeepAlive`, so the API survives reboots and
restarts if it crashes. Logs to `social/out/api.log`.

```bash
launchctl list | grep yarasports          # check it's running
launchctl unload ~/Library/LaunchAgents/com.yarasports.api.plist   # stop
launchctl load   ~/Library/LaunchAgents/com.yarasports.api.plist   # start
```

## 2. Daily batch (launchd, catch-up on wake) — emails you a draft per format
The daily formats run via a **launchd** agent, `com.yarasports.daily`
(`StartCalendarInterval` 08:01), which calls `run_daily.sh`. launchd is used
instead of cron because **a run missed while the Mac is asleep/closed fires on
the next wake** — cron just skips it. So if the lid is shut at 08:01, the batch
generates the moment you open it. The script waits for the API to come up first.

`run_daily.sh` runs every day: **matchday board, risk spike, accountability**;
plus **riskiest XI** (Mon), **battle card** (Tue/Fri), **archetype** (Tue).

```bash
launchctl list | grep yarasports.daily        # check it's loaded
launchctl unload ~/Library/LaunchAgents/com.yarasports.daily.plist   # disable
launchctl load   ~/Library/LaunchAgents/com.yarasports.daily.plist   # enable
tail -f social/out/cron.log                   # watch runs
```

**Proactive wake (optional):** with catch-up on wake you do not strictly need
it, but to have the batch run at 08:00 even while the lid is closed (and the Mac
is on AC), schedule a wake:
```bash
sudo pmset repeat wakeorpoweron MTWRFSU 08:00:00   # cancel: sudo pmset repeat cancel
```

## 3. Lineup reaction (cron) — opportunistic, match window
`crontab -l` shows one job: lineup reaction hourly 15:00–23:00. It only posts
when a confirmed XI (API-Football, `API_FOOTBALL_KEY`) starts a flagged player,
and only runs while the Mac is awake — best-effort, no loss if a match is missed.

## The one requirement
Each run **emails a draft**, so `GMAIL_APP_PASSWORD` in `.env` must be a real
16-character Google App Password (myaccount.google.com > Security > App
passwords). Until then the runs render fine but the email step fails. Nothing
auto-posts; you review the draft and post manually.

## Run any format by hand
```bash
./social/autopost/cron_run.sh --format matchday_board --competition world-cup-2026 --matchday "Group Stage"
# add --dry-run to skip the email
```
