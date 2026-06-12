#!/bin/bash
# Wrapper so cron can run a Yara autopost format with the right cwd, env, and
# Python, logging to social/out/cron.log. Usage:
#   cron_run.sh --format matchday_board --competition world-cup-2026 --matchday "Group Stage"
set -euo pipefail
REPO="/Users/georgeriley/code/Machine-Learning/Injury Risk Predictor"
cd "$REPO" || exit 1
export YARASPEAKS_API_BASE="http://localhost:8000"
echo "[$(date)] running: $*" >> "$REPO/social/out/cron.log"
"$REPO/.conda/bin/python" -m social.autopost.run "$@" >> "$REPO/social/out/cron.log" 2>&1
