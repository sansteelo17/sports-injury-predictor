# Yara matchday auto-post (email drafts)

Fetches risk data, lets an editorial LLM pick the players, renders the card to a
PNG (Playwright), writes the X + Reddit copy, and **emails you the draft**. It
never auto-posts. One command per format/trigger.

Currently wired: **Card 01, the matchday/round board** (the only data-driven
card). Other formats slot in as new `--format` values + a card under
`social/cards/`.

## One-time setup

```bash
pip install playwright requests
python -m playwright install chromium
```

Add to `.env` (repo root):

```
# Gmail App Password — myaccount.google.com > Security > App passwords
GMAIL_APP_PASSWORD=xxxxxxxxxxxxxxxx
# optional overrides (sensible defaults exist)
# GMAIL_USER=gakpovwovwo@gmail.com
# DRAFT_EMAIL_TO=gakpovwovwo@gmail.com
# YARASPEAKS_API_BASE=http://localhost:8000   # default https://api.yaraspeaks.com
# AUTOPOST_MODEL=gpt-5.4-mini
```

`OPENAI_API_KEY` is already in `.env` and is reused.

> Note: production `/api/players` for the World Cup (~1200 players) is slow.
> For the cron, either run a local API (`uvicorn api.main:app --port 8000`) and
> set `YARASPEAKS_API_BASE=http://localhost:8000`, or bump the production box.

## Run

```bash
# Preview (renders card + prints copy, sends NO email):
python -m social.autopost.run --format matchday_board \
  --competition world-cup-2026 --matchday MD1 --dry-run

# Real (emails the draft to DRAFT_EMAIL_TO):
python -m social.autopost.run --format matchday_board \
  --competition world-cup-2026 --matchday MD1
```

Cards land in `social/out/`. Competitions: `world-cup-2026`, `premier-league`,
`la-liga`, `bundesliga`, `serie-a`, `ligue-1`, `champions-league`.

## Schedule (local cron)

`crontab -e`, e.g. a World Cup board every matchday morning at 09:00:

```
0 9 * * *  cd "/Users/georgeriley/code/Machine-Learning/Injury Risk Predictor" && /usr/bin/env python3 -m social.autopost.run --format matchday_board --competition world-cup-2026 --matchday MD1 >> social/out/cron.log 2>&1
```

(Your Mac must be awake at the scheduled time. Update `--matchday` per round, or
make it a small wrapper that computes the round.)
