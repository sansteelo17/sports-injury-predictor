#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install python dependencies
pip install -r requirements.txt

# Install Playwright browser ONLY (skipping system deps that require sudo)
export PLAYWRIGHT_BROWSERS_PATH=/opt/render/.cache/ms-playwright
playwright install chromium