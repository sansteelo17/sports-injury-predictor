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

## 2. Schedules (cron) — emails you a draft per format
`crontab -l` shows them; all call `cron_run.sh` (sets cwd/env/python, logs to
`social/out/cron.log`):

| Format | When (local) |
|---|---|
| matchday board | every day 08:07 |
| risk spike | every day 08:37 (only posts if there's a spike) |
| accountability | every day 10:07 (scores yesterday) |
| riskiest XI | Mondays 12:07 |
| battle card | Tue + Fri 09:07 |

```bash
crontab -l            # view
crontab -e            # edit
crontab -r            # remove ALL (careful)
tail -f social/out/cron.log    # watch runs
```

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
