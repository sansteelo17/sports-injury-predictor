# Injury Risk Predictor Dashboard

A user-friendly Streamlit web application for predicting football player injury risk using machine learning.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## Features

### Input Form
- **Player Name**: Enter any player name
- **Age**: Slider from 16-40 years
- **Position**: Dropdown with all major positions
- **Team**: Premier League teams + custom option
- **Recent Matches**: Workload proxy (0-20 matches in last 4 weeks)
- **FIFA Rating**: Optional quality metric (50-99)

### Prediction Output

1. **Risk Overview**
   - Risk level (High/Moderate/Low)
   - Risk probability percentage
   - Projected severity (days lost)
   - Model confidence level

2. **Detailed Analysis**
   - Color-coded risk panel
   - Ensemble model predictions (CatBoost, LightGBM, XGBoost)
   - Top 5 risk factors with SHAP impact scores
   - Model agreement metrics

3. **Player Archetype**
   - 5 distinct risk profiles:
     - High-Risk Frequent
     - Moderate-Load High-Variance
     - Moderate-Risk Recurrent
     - Low-Severity Stable
     - Catastrophic + Re-aggravation
   - Description and training focus for each archetype

4. **Recommendations**
   - Training load adjustments
   - Match minutes guidance
   - Recovery strategies
   - Fantasy football insights

### Informational Sections

Expandable sections covering:
- **How It Works**: Methodology and ML approach
- **Data Sources**: Training data and coverage
- **Limitations**: Important caveats and scope
- **Use Cases**: Fantasy football, amateur teams, sports science

## Demo Mode vs. Production

### Current State: Demo Mode

The app currently runs in **demo mode** with simulated predictions because:
- Models haven't been trained yet
- No saved model artifacts in `/models` directory
- Uses intelligent heuristics to generate realistic predictions

Demo predictions are based on:
- Age risk factors (normalized 16-40)
- Position-specific injury rates
- Recent match load (acute workload proxy)
- Randomized factors for realism

### Production Mode

To enable real predictions:

1. **Train the models** using the existing notebooks/scripts
2. **Save model artifacts** to `/models` directory:
   - `catboost_classifier.pkl`
   - `lightgbm_classifier.pkl`
   - `xgboost_classifier.pkl`
   - `severity_regressor.pkl`
   - `archetype_model.pkl`
   - `feature_columns.json`

3. **Update `load_models()` function** in `app.py` to load your trained models

4. **Replace `generate_mock_prediction()`** with actual model inference calls

## Use Cases

### Fantasy Football
- Avoid high-risk players before busy periods
- Optimal captain selection
- Smart transfer planning

### Amateur/Semi-Pro Teams
- Training load management
- Squad rotation strategies
- Injury prevention programs

### Sports Analytics
- Pattern analysis
- Research and education
- Performance optimization

### Educational
- Learn about ML in sports
- Understand injury risk factors
- Explore sports science concepts

## Important Disclaimers

⚠️ **NOT MEDICAL ADVICE**
- This is a proof-of-concept demonstration
- For educational and entertainment purposes only
- Should not replace professional medical assessment
- Not suitable for betting or critical decisions

⚠️ **NO LIABILITY**
- Predictions are statistical estimates
- No guarantee of accuracy
- Use at your own discretion
- Professional deployment requires validation

## Technical Architecture

### Machine Learning Stack
- **Models**: CatBoost, LightGBM, XGBoost (ensemble)
- **Clustering**: HDBSCAN + K-Means hybrid
- **Explainability**: SHAP values
- **Framework**: scikit-learn, Streamlit

### Data Pipeline
- Feature engineering from player stats
- Workload metrics (acute/chronic load ratios)
- Historical injury patterns
- Position and age risk factors

### UI/UX Design
- Clean, professional interface
- Color-coded risk levels
- Responsive layout
- Mobile-friendly
- Comprehensive tooltips and help text

## Customization

### Styling
Edit the CSS in `app.py` under the `st.markdown()` custom styles section:
- `.risk-high`, `.risk-medium`, `.risk-low` for risk cards
- `.disclaimer` for warning banners
- `.metric-card` for stat displays

### Risk Thresholds
Modify the risk categorization in `generate_mock_prediction()`:
```python
if risk_prob >= 0.60:  # Adjust threshold
    risk_level = "High"
```

### Archetype Profiles
Update archetype descriptions in `get_archetype_description()` function

### Teams and Positions
Modify `get_default_teams_positions()` to add/remove options

## Troubleshooting

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Dependencies Not Found
```bash
pip install --upgrade -r requirements.txt
```

### Data Loading Issues
The app will fall back to defaults if data files aren't found. Check:
- `/data/raw/player_injuries_impact.csv` exists
- File paths are correct for your OS

### Module Import Errors
Ensure you're running from the project root directory:
```bash
cd /path/to/Injury\ Risk\ Predictor
streamlit run app.py
```

## Development

### File Structure
```
Injury Risk Predictor/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── data/
│   └── raw/
│       ├── player_injuries_impact.csv
│       └── All_Players_1992-2025.csv
├── src/
│   ├── models/              # ML model definitions
│   ├── inference/           # Prediction pipeline
│   ├── dashboard/           # Dashboard utilities
│   └── ...
└── models/                  # Saved model artifacts (create this)
```

### Adding Real Models

Example integration:
```python
@st.cache_resource
def load_models():
    import joblib

    models = {
        'catboost': joblib.load('models/catboost_classifier.pkl'),
        'lightgbm': joblib.load('models/lightgbm_classifier.pkl'),
        'severity': joblib.load('models/severity_regressor.pkl'),
        'archetype': joblib.load('models/archetype_model.pkl'),
    }

    with open('models/feature_columns.json', 'r') as f:
        features = json.load(f)

    return models, features
```

## Future Enhancements

Potential improvements:
- [ ] Historical player tracking (show risk over time)
- [ ] Batch prediction mode (upload CSV of squad)
- [ ] Comparison tool (compare multiple players)
- [ ] Risk heatmap by position
- [ ] Export prediction reports (PDF)
- [ ] API endpoint for programmatic access
- [ ] User authentication and saved profiles
- [ ] Integration with external APIs (football-data.org, etc.)

## Support

For questions or issues:
1. Check this README
2. Review the in-app "How It Works" section
3. Examine the code comments in `app.py`
4. Contact the developer

## License

This is a proof-of-concept demonstration project.

---

**Version**: 1.0
**Last Updated**: January 2026
**Status**: Demo Mode (Mock Predictions)
