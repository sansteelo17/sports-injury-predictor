.PHONY: help install run clean test docker-build docker-run deploy-check lint format retrain retrain-accurate retrain-dry-run retrain-cron-install retrain-cron-remove

# Default target
help:
	@echo "Football Injury Risk Predictor - Available Commands"
	@echo "=================================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies"
	@echo "  make venv           Create virtual environment"
	@echo ""
	@echo "Running:"
	@echo "  make run            Start Streamlit dashboard"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run in Docker container"
	@echo ""
	@echo "Retraining:"
	@echo "  make retrain              Run retraining pipeline (fast/API mode)"
	@echo "  make retrain-accurate     Run retraining pipeline (FBref scraping)"
	@echo "  make retrain-dry-run      Preview retraining without saving"
	@echo "  make retrain-cron-install Install Tue/Fri cron schedule"
	@echo "  make retrain-cron-remove  Remove retrain cron jobs"
	@echo ""
	@echo "Development:"
	@echo "  make clean          Remove cache and temporary files"
	@echo "  make lint           Check code style (future)"
	@echo "  make format         Format code (future)"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-check   Verify deployment readiness"
	@echo ""

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	python -m venv venv
	@echo "✓ Virtual environment created"
	@echo "Activate with: source venv/bin/activate (Mac/Linux) or venv\\Scripts\\activate (Windows)"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Run Streamlit dashboard
run:
	@echo "Starting Streamlit dashboard..."
	streamlit run app.py

# Clean cache and temporary files
clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .streamlit/cache 2>/dev/null || true
	@echo "✓ Cache cleaned"

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -t injury-risk-predictor .
	@echo "✓ Docker image built"

# Run Docker container
docker-run:
	@echo "Starting Docker container..."
	docker run -p 8501:8501 injury-risk-predictor

# Check deployment readiness
deploy-check:
	@echo "Checking deployment readiness..."
	@echo ""
	@echo "1. Checking requirements.txt..."
	@test -f requirements.txt && echo "✓ requirements.txt exists" || echo "✗ requirements.txt missing"
	@echo ""
	@echo "2. Checking app.py..."
	@test -f app.py && echo "✓ app.py exists" || echo "✗ app.py missing"
	@echo ""
	@echo "3. Checking .streamlit/config.toml..."
	@test -f .streamlit/config.toml && echo "✓ config.toml exists" || echo "✗ config.toml missing"
	@echo ""
	@echo "4. Checking .gitignore..."
	@test -f .gitignore && echo "✓ .gitignore exists" || echo "✗ .gitignore missing"
	@echo ""
	@echo "5. Checking documentation..."
	@test -f README.md && echo "✓ README.md exists" || echo "✗ README.md missing"
	@test -f DEPLOYMENT.md && echo "✓ DEPLOYMENT.md exists" || echo "✗ DEPLOYMENT.md missing"
	@echo ""
	@echo "6. Testing imports..."
	@python -c "import pandas; import numpy; import sklearn; import lightgbm; import xgboost; import catboost; import shap; import streamlit" && echo "✓ All required packages installed" || echo "✗ Some packages missing"
	@echo ""
	@echo "Deployment readiness check complete!"

# Lint code (placeholder for future)
lint:
	@echo "Linting not yet configured"
	@echo "Consider adding: flake8, pylint, or black"

# Format code (placeholder for future)
format:
	@echo "Code formatting not yet configured"
	@echo "Consider adding: black, autopep8, or isort"

# ── Retraining Pipeline ──────────────────────────────────────

# Run retraining pipeline (fast API mode, ~2 min)
retrain:
	python scripts/retrain.py

# Run retraining with FBref scraping (accurate, ~20 min)
retrain-accurate:
	python scripts/retrain.py --accurate

# Preview retraining without saving anything
retrain-dry-run:
	python scripts/retrain.py --dry-run

# Install cron jobs for Tuesday & Friday 6 AM UTC
retrain-cron-install:
	@PROJECT_DIR="$(shell pwd)"; \
	CRON_MARKER="# YaraSports retrain"; \
	(crontab -l 2>/dev/null | grep -v "$$CRON_MARKER"; \
	 echo "0 6 * * 2 cd $$PROJECT_DIR && python scripts/retrain.py >> logs/retrain.log 2>&1 $$CRON_MARKER"; \
	 echo "0 6 * * 5 cd $$PROJECT_DIR && python scripts/retrain.py >> logs/retrain.log 2>&1 $$CRON_MARKER"; \
	) | crontab -
	@echo "Cron jobs installed (Tue & Fri 6 AM UTC)"
	@echo "Verify with: crontab -l"

# Remove retrain cron jobs
retrain-cron-remove:
	@CRON_MARKER="# YaraSports retrain"; \
	(crontab -l 2>/dev/null | grep -v "$$CRON_MARKER") | crontab -
	@echo "Retrain cron jobs removed"
