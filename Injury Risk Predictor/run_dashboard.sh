#!/bin/bash

# Football Injury Risk Predictor - Dashboard Launcher
# This script provides an easy way to start the Streamlit dashboard

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Football Injury Risk Predictor${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}Error: Python $required_version or higher is required${NC}"
    echo -e "${RED}Current version: $python_version${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version detected${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
    echo ""
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Check if dependencies are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
    echo ""
else
    echo -e "${GREEN}✓ Dependencies already installed${NC}"
    echo ""
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo -e "${RED}Error: app.py not found${NC}"
    echo -e "${YELLOW}Make sure you're running this script from the project root directory${NC}"
    exit 1
fi

# Start the dashboard
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Streamlit Dashboard...${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Dashboard will open in your browser at:${NC}"
echo -e "${GREEN}http://localhost:8501${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Run Streamlit
streamlit run app.py
