#!/bin/bash
# Daily Yara batch, run by a launchd StartCalendarInterval agent so a run missed
# while the Mac was asleep/closed fires on the next wake (cron would skip it).
# Waits for the keep-alive API to come up (it also restarts on wake), then runs
# the day's formats. Drafts are emailed; nothing auto-posts.
REPO="/Users/georgeriley/code/Machine-Learning/Injury Risk Predictor"
cd "$REPO" || exit 1
export YARASPEAKS_API_BASE="http://localhost:8000"
LOG="$REPO/social/out/cron.log"
PY="$REPO/.conda/bin/python"
COMP="world-cup-2026"

echo "[$(date)] daily batch starting" >> "$LOG"
# Wait up to 3 min for the API (launchd restarts it on wake; it loads models).
for _ in $(seq 1 36); do
  curl -sf "http://localhost:8000/api/board-candidates?competition=$COMP&limit=1" >/dev/null 2>&1 && break
  sleep 5
done

run() { "$PY" -m social.autopost.run "$@" >> "$LOG" 2>&1 || echo "[$(date)] FAILED: $*" >> "$LOG"; }

# Every day.
run --format matchday_board  --competition "$COMP"
run --format risk_spike      --competition "$COMP"
run --format accountability  --competition "$COMP"

# Day-of-week extras (1=Mon ... 7=Sun).
dow=$(date +%u)
[ "$dow" = "1" ] && run --format riskiest_xi --competition "$COMP"
{ [ "$dow" = "2" ] || [ "$dow" = "5" ]; } && run --format battle_card --competition "$COMP"
[ "$dow" = "2" ] && run --format archetype --competition "$COMP"

echo "[$(date)] daily batch done" >> "$LOG"
