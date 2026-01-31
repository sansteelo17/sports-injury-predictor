# Football Injury Risk Predictor

A machine learning system that predicts injury risk for football players based on match data, workload metrics, injury history, and player archetypes.

---

## ⚠️ IMPORTANT DISCLAIMERS

**THIS IS NOT MEDICAL ADVICE**

- This is a **proof-of-concept educational project** only
- **NOT intended for professional medical or clinical decisions**
- Predictions are based on historical data patterns and should not replace professional medical assessment
- **Do not use this tool to make actual player health decisions**
- Always consult qualified sports medicine professionals for injury prevention and management

**Educational Use Only**

This project is designed for:
- Learning about sports analytics and machine learning
- Understanding injury risk factors in football
- Educational exploration of predictive modeling
- Amateur and hobbyist analytics

---

## Overview

This system uses ensemble machine learning (CatBoost, LightGBM, XGBoost) to:
- Predict injury risk probability for players
- Estimate potential injury severity (days lost)
- Classify players into injury risk archetypes
- Provide SHAP-based explanations for predictions
- Generate training recommendations based on risk profiles

### Key Features

- **Multi-model ensemble**: Combines CatBoost, LightGBM, and XGBoost for robust predictions
- **Injury severity prediction**: Estimates potential days lost to injury
- **Player archetypes**: Classifies players into 5 risk profiles
- **Explainable AI**: Uses SHAP values to explain predictions
- **Workload science**: Incorporates ACWR, monotony, strain, and fatigue metrics
- **Interactive dashboard**: Generates detailed player risk cards

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip or conda package manager

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "Injury Risk Predictor"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place raw data files in `data/raw/`
   - Run preprocessing scripts (see Data Pipeline section)

4. Train models:
```bash
# Run the training notebooks in order
jupyter notebook notebooks/exploration.ipynb
```

---

## Running the Application

### Streamlit Dashboard

The interactive web dashboard allows you to:
- Input player details (name, age, position, team, recent matches)
- View comprehensive risk assessments with color-coded risk levels
- Explore ensemble model predictions (CatBoost, LightGBM, XGBoost)
- Analyze top risk factors with impact scores
- View player archetype profiles and recommendations
- Get training load, match minutes, and recovery guidance

**Quick Start:**

```bash
# Option 1: Use the launcher script
./run_dashboard.sh

# Option 2: Run directly
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**Current Status**: Demo mode with intelligent mock predictions. See `DASHBOARD_README.md` for full documentation on enabling real model predictions.

**Features:**
- ✅ User-friendly input form with dropdowns and sliders
- ✅ Risk probability with color-coded displays (red/yellow/green)
- ✅ Severity prediction in days
- ✅ Player archetype classification
- ✅ Top risk factors display
- ✅ Actionable recommendations
- ✅ Comprehensive help sections (How It Works, Data Sources, Limitations, Use Cases)
- ✅ Prominent disclaimers throughout
- ✅ Mobile-responsive design

---

## Project Architecture

```
Injury Risk Predictor/
│
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned and merged data
│   └── features/         # Feature-engineered datasets
│
├── src/
│   ├── data_loaders/     # Data loading utilities
│   ├── preprocessing/    # Data cleaning and merging
│   ├── feature_engineering/  # Feature creation
│   │   ├── injury_history.py
│   │   ├── workload.py
│   │   ├── match_features.py
│   │   └── archetype.py
│   ├── models/           # Model training and evaluation
│   │   ├── classification.py      # Injury risk classifier
│   │   ├── severity.py            # Severity regression
│   │   ├── archetype.py           # Archetype clustering
│   │   ├── classification_ensemble.py
│   │   ├── severity_ensemble.py
│   │   └── thresholding.py
│   ├── inference/        # Prediction pipeline
│   │   ├── inference_pipeline.py
│   │   └── risk_card.py
│   └── dashboard/        # Dashboard components
│       └── player_dashboard.py
│
├── notebooks/            # Jupyter notebooks for exploration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

---

## Data Sources

This project uses publicly available football data:

1. **Player Statistics** (`All_Players_1992-2025.csv`)
   - Historical player performance data
   - Position, age, team information

2. **Injury Records** (`player_injuries_impact.csv`)
   - Injury dates and types
   - Days lost to injury
   - Re-injury patterns

3. **Match Data** (`premier-league-matches.csv`)
   - Match dates and results
   - Team performance metrics
   - Used for workload calculations

**Data Attribution**: Please ensure you have appropriate rights to use any data with this system.

---

## How It Works

### 1. Feature Engineering

The system creates features from:
- **Match frequency**: Matches in last 7/14/30 days
- **Rest patterns**: Days between matches, average rest
- **Team form**: Win/loss streaks, goal differentials
- **Injury history**: Previous injuries, days since last injury
- **Workload metrics**:
  - Acute:Chronic Workload Ratio (ACWR)
  - Training monotony
  - Training strain
  - Fatigue index
  - Workload slope and spikes

### 2. Model Training

Three-model ensemble approach:
- **CatBoost**: Handles categorical features natively
- **LightGBM**: Fast gradient boosting
- **XGBoost**: Robust tree-based model

Models are trained with:
- Optuna hyperparameter tuning
- Cross-validation
- Class imbalance handling (SMOTE, class weights)
- Threshold optimization for precision-recall balance

### 3. Prediction Pipeline

For each player, the system:
1. Computes current feature values
2. Generates ensemble predictions (risk probability)
3. Estimates injury severity (days lost)
4. Assigns player archetype
5. Computes SHAP explanations
6. Generates training recommendations

### 4. Player Archetypes

Players are classified into 5 risk profiles:
- **High-Risk Frequent**: Sensitive to repeated loading
- **Moderate-Load High-Variance**: Inconsistent workload tolerance
- **Moderate-Risk Recurrent**: Recurring injury patterns
- **Low-Severity Stable**: Good tissue adaptation
- **Catastrophic + Re-aggravation**: High relapse vulnerability

---

## Use Cases

### ✅ Appropriate Uses

1. **Football Fans & Analytics Enthusiasts**
   - Explore injury risk patterns
   - Understand workload science in football
   - Learn about predictive modeling

2. **Amateur Teams & Sunday Leagues**
   - General awareness of injury risk factors
   - Educational tool for player development
   - Understanding workload management concepts

3. **Students & Researchers**
   - Learning sports analytics
   - Understanding ML in sports medicine
   - Academic research and education

4. **Data Science Portfolio**
   - Demonstration of ML engineering
   - End-to-end system design
   - Real-world problem solving

### ❌ NOT Appropriate For

1. **Professional Medical Decisions**
   - Do not use for clinical diagnosis
   - Not a replacement for medical professionals
   - Not validated for medical use

2. **Professional Team Management**
   - Not designed for elite-level decision making
   - Predictions are educational estimates only
   - Requires professional sports science integration

3. **Player Health Screening**
   - Not a medical screening tool
   - Cannot replace physical examinations
   - Not suitable for pre-participation screening

---

## Model Performance

Models are evaluated on:
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Precision/Recall**: Trade-off via threshold optimization
- **F1 Score**: Balance of precision and recall
- **MAE/RMSE**: For severity regression
- **Ensemble agreement**: Model consensus confidence

Typical performance (educational dataset):
- ROC-AUC: 0.70-0.75 (injury classification)
- MAE: 5-7 days (severity prediction)

**Note**: These metrics are on historical data and may not generalize to new contexts.

---

## Roadmap

### Current Features
- ✅ Ensemble injury risk classification
- ✅ Severity prediction
- ✅ Player archetype clustering
- ✅ SHAP explainability
- ✅ Training recommendations

### Planned Features
- ✅ Streamlit web dashboard (v1.0 complete - demo mode)
- 🔄 Connect dashboard to trained models (production mode)
- 🔄 Real-time data integration
- 🔄 Advanced visualization (historical tracking, heatmaps)
- 🔄 Batch prediction mode (upload squad CSV)
- 🔄 Multi-sport support
- 🔄 API endpoint for predictions
- 🔄 Docker containerization
- 🔄 Comprehensive test suite

---

## Training Models

To train models from scratch:

1. **Preprocess data**:
```python
# Run preprocessing pipeline
from src.preprocessing import clean_stats, clean_injuries, merge_injury_stats
# Follow notebooks/exploration.ipynb for full pipeline
```

2. **Engineer features**:
```python
from src.feature_engineering import workload, injury_history, match_features
# See notebooks for feature engineering workflow
```

3. **Train models**:
```python
from src.models import classification, severity, archetype
# Training scripts in notebooks
```

4. **Save models**:
```python
import joblib
joblib.dump(model, 'models/catboost_classifier.pkl')
```

---

## Contributing

This is an educational project. Contributions are welcome for:
- Feature improvements
- Documentation enhancements
- Bug fixes
- Additional educational use cases

Please ensure any contributions maintain the educational nature and include appropriate disclaimers.

---

## License

This project is for educational purposes. Please ensure you have appropriate rights to any data used with this system.

---

## Acknowledgments

- Sports science research on injury risk factors
- Open-source ML libraries (scikit-learn, LightGBM, XGBoost, CatBoost)
- SHAP for model interpretability
- Football analytics community

---

## Contact

For questions about this educational project, please open an issue on GitHub.

---

**Remember: This is an educational tool, not medical advice. Always consult qualified professionals for health-related decisions.**
