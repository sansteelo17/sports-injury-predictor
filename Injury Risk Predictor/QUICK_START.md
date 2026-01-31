# Quick Start Guide

Get the Football Injury Risk Predictor running in 5 minutes!

---

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (optional, for cloning)

---

## Installation

### Option 1: Quick Install (Recommended)

```bash
# 1. Navigate to project directory
cd "Injury Risk Predictor"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Option 2: Virtual Environment (Recommended for Development)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

### Option 3: Docker (For Production)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:8501
```

---

## First Time Using the App

1. **Open the dashboard**: The app should automatically open in your browser
   - If not, navigate to `http://localhost:8501`

2. **Enter player details**:
   - Select a player name (or choose "Other" to enter manually)
   - Set age, position, and team
   - Configure recent match history

3. **View risk assessment**:
   - See injury risk probability (Low/Moderate/High)
   - Check predicted severity (days lost)
   - Review player archetype
   - Explore top risk factors

4. **Get recommendations**:
   - Training load guidance
   - Match minutes recommendations
   - Recovery protocols

---

## Current Mode: Demo vs Production

### Demo Mode (Current Default)

- Uses intelligent mock predictions
- Great for testing and demonstrations
- No trained models required
- Realistic risk calculations based on inputs

### Production Mode (Requires Trained Models)

To enable production mode:

1. Train models using notebooks:
   ```bash
   jupyter notebook notebooks/exploration.ipynb
   ```

2. Save models to `models/` directory:
   - `catboost_classifier.pkl`
   - `lightgbm_classifier.pkl`
   - `xgboost_classifier.pkl`

3. Update `app.py` to load real models (see `DASHBOARD_README.md`)

---

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Kill process on port 8501
lsof -ti:8501 | xargs kill -9

# Or use a different port
streamlit run app.py --server.port=8502
```

#### 2. Module Not Found Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

#### 3. Permission Denied (macOS/Linux)

```bash
# Make launcher executable
chmod +x run_dashboard.sh

# Run launcher
./run_dashboard.sh
```

#### 4. Python Version Too Old

```bash
# Check Python version
python --version

# Should be 3.9 or higher
# If not, install Python 3.9+ from python.org
```

---

## Project Structure Quick Reference

```
Injury Risk Predictor/
│
├── app.py                  # Main Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md              # Full documentation
├── DEPLOYMENT.md          # Deployment guide
│
├── src/                   # Source code
│   ├── models/           # Model training scripts
│   ├── feature_engineering/  # Feature creation
│   └── inference/        # Prediction pipeline
│
├── data/                  # Data files
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned data
│   └── features/         # Feature-engineered data
│
└── notebooks/            # Jupyter notebooks for exploration
```

---

## Next Steps

### For Learning

1. Explore the Streamlit dashboard
2. Review `notebooks/exploration.ipynb` to understand the pipeline
3. Read `README.md` for architecture details
4. Check `CONTRIBUTING.md` to contribute

### For Development

1. Set up virtual environment
2. Install dependencies
3. Run notebooks to train models
4. Test predictions with real models
5. Customize the dashboard

### For Deployment

1. Review `DEPLOYMENT.md`
2. Choose a platform (Streamlit Cloud, Render, Railway, Fly.io)
3. Set up environment variables
4. Deploy and test
5. Monitor performance

---

## Useful Commands

```bash
# Start dashboard
streamlit run app.py

# Start with specific port
streamlit run app.py --server.port=8502

# Clear Streamlit cache
streamlit cache clear

# Run in Docker
docker-compose up

# Stop Docker
docker-compose down

# View logs
docker-compose logs -f

# Install new dependency
pip install package-name
pip freeze > requirements.txt
```

---

## Getting Help

- **Documentation**: See `README.md` for full details
- **Deployment**: See `DEPLOYMENT.md` for deployment guides
- **Contributing**: See `CONTRIBUTING.md` for contribution guidelines
- **Issues**: Open an issue on GitHub

---

## Important Reminders

⚠️ **This is NOT medical advice**
- Educational project only
- Not for professional medical decisions
- Always consult qualified professionals

✅ **Appropriate uses**:
- Learning sports analytics
- Understanding ML in sports medicine
- Amateur team awareness
- Educational exploration

❌ **NOT appropriate for**:
- Professional medical decisions
- Clinical diagnosis
- Elite team management
- Player health screening

---

## Tips for Best Results

1. **Input quality matters**: More accurate inputs = better predictions
2. **Review all factors**: Don't just look at the risk score
3. **Understand limitations**: This is a statistical model, not a diagnosis
4. **Use holistically**: Combine with other injury prevention strategies
5. **Stay updated**: Check for updates and new features

---

**Ready to go? Run `streamlit run app.py` and explore!**
