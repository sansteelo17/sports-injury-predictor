#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install python dependencies
pip install -r requirements.txt

# Install Playwright browsers and system dependencies
playwright install chromium
playwright install-deps chromium