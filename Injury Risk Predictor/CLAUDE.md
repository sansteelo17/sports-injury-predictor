# Football Injury Risk Predictor - Project Context

## Overview
ML system predicting injury risk for football (soccer) players using ensemble models (CatBoost, LightGBM, XGBoost). Educational/portfolio project, not for medical use.

## Data Sources
- `data/raw/All_Players_1992-2025.csv` - Player stats (49MB, 92K+ rows)
- `data/raw/player_injuries_impact.csv` - Injury records with days lost
- `data/raw/premier-league-matches.csv` - Match data for workload calculations

## Architecture

```
src/
├── data_loaders/load_data.py     # Load raw CSVs
├── preprocessing/
│   ├── clean_injuries.py         # Clean injury data
│   ├── clean_stats.py            # Clean player stats
│   ├── clean_matches.py          # Clean match data
│   ├── team_normalization.py     # Normalize team names
│   ├── merge_injury_stats.py     # Merge injuries + stats
│   └── merge_injury_matches.py   # Merge with match data
├── feature_engineering/
│   ├── workload.py              # ACWR, monotony, strain, fatigue (key file)
│   ├── injury_history.py        # Previous injuries, days since last
│   ├── match_features.py        # Matches in 7/14/30 days, rest days
│   ├── archetype.py             # K-means clustering for player profiles
│   ├── severity.py              # Injury severity features
│   └── negative_sampling.py     # Generate non-injury samples
├── models/
│   ├── classification.py        # Train injury risk classifier
│   ├── severity.py              # Train severity predictor (short/medium/long)
│   ├── classification_ensemble.py   # Weighted LightGBM + XGBoost ensemble
│   ├── stacking_ensemble.py     # Stacking with CatBoost meta-learner
│   ├── thresholding.py          # Optimize classification threshold
│   ├── smote_utils.py           # Handle class imbalance
│   └── temporal_validation.py   # Time-based train/test splits
├── inference/
│   ├── inference_pipeline.py    # Full prediction pipeline (key file)
│   ├── risk_card.py             # Generate player risk cards
│   └── validation.py            # Input validation for inference
└── dashboard/
    └── player_dashboard.py      # Streamlit components
```

## Key Features Computed

### Workload Metrics (src/feature_engineering/workload.py)
- `acute_load` - 7-day rolling match count
- `chronic_load` - 28-day rolling match count
- `acwr` - Acute:Chronic Workload Ratio (injury risk > 1.5)
- `monotony` - Training monotony index
- `strain` - Acute load * monotony
- `fatigue_index` - Acute - chronic load
- `workload_slope` - 5-match trend
- `spike_flag` - ACWR > 1.5 danger indicator

### Match Features
- `matches_last_7`, `matches_last_14`, `matches_last_30`
- `rest_days_before_injury`, `avg_rest_last_5`
- `win_streak`, `loss_streak`, `form_last_5`

### Injury History
- `previous_injuries` - Count of prior injuries
- `days_since_last_injury` - Recovery time

## Model Pipeline

1. **Classification**: Predict if player will get injured (binary)
   - Ensemble of CatBoost + LightGBM + XGBoost
   - Weighted averaging or stacking meta-learner
   - Threshold tuned for F1/precision-recall balance

2. **Severity**: Predict injury duration (short/medium/long)
   - Bins: short (0-21 days), medium (21-60 days), long (60+ days)
   - **CatBoost** with `auto_class_weights='Balanced'` (62% accuracy, 94%+ adjacent accuracy)
   - Key metrics: accuracy, adjacent accuracy (within 1 category)

3. **Archetype**: Cluster players into 5 risk profiles
   - **Injury Prone** - Gets injured often, usually minor
   - **Unpredictable** - Varied injury patterns, hard to manage
   - **Recurring Issues** - Same body parts keep getting injured
   - **Durable** - Rarely gets seriously injured
   - **Fragile** - When injured, it's serious

## Notebooks
- `notebooks/exploration.ipynb` - Main notebook with 106 cells covering:
  - Data loading and inspection
  - Missing value analysis
  - Data cleaning functions
  - Feature engineering
  - Model training experiments

## App
- `app.py` - Streamlit dashboard (demo mode)
- Run with: `streamlit run app.py` or `./run_dashboard.sh`
- Currently uses mock predictions; needs trained models for production

## Current Status
- Data pipeline: Complete
- Feature engineering: Complete
- Classification model: StackingEnsemble (CatBoost + LightGBM + XGBoost)
- Severity model: CatBoost with balanced class weights (62% accuracy, 94%+ adjacent accuracy)
- Dashboard: Demo mode working
- Git: Project reorganized, untracked

## Recent Changes (This Session)
- Fixed adjacent accuracy bug in severity evaluation (CatBoost returns shape (n,1) not (n,))
- Added `auto_class_weights='Balanced'` to CatBoost severity classifier
- Improved severity accuracy from 53.8% to 62.3%, adjacent accuracy to 94.6%
- Tested ordinal regression and ensemble approaches (in severity.py) but rolled back:
  - CatBoost alone performed equal/better (94.6% vs 91.5% adjacent accuracy)
  - Keeping CatBoost for simplicity; ordinal/ensemble code remains available if needed
- Fixed archetype clustering bug: 0-d numpy arrays in features caused `.median()` to fail
  - Added `_unwrap_array_values()` helper in archetype.py to sanitize data
- Fixed monotony/strain explosion bug in workload.py
  - Previously: monotony = 1e9 when std near zero (regular schedule)
  - Now: capped at 5 (realistic sports science range)
- Fixed inference pipeline missing feature engineering
  - Added `apply_all_feature_engineering()` to inference_pipeline.py
  - Now applies temporal, position, and workload interaction features
  - Automatically cleans up merge artifacts (age_x → age)
- Fixed dashboard `panel_severity_projection()` for inference mode
  - Now works with both ground truth (severity_days) and predicted (predicted_severity_class)
- Fixed notebook archetype clustering to include player history features
  - Previously missing: is_injury_prone, player_avg_severity, etc.

## Typical Workflow
```python
# 1. Load data
from src.data_loaders.load_data import load_all_data
data = load_all_data()

# 2. Clean data
from src.preprocessing.clean_injuries import clean_injury_df
injury_df = clean_injury_df(data["injuries"])

# 3. Add features
from src.feature_engineering.workload import add_workload_metrics
df = add_workload_metrics(match_df)

# 4. Train models (in notebooks)
# 5. Run inference
from src.inference.inference_pipeline import build_full_inference_df
```

## Severity Classification Usage

```python
from src.models.severity import (
    get_severity_classification_splits,
    train_severity_classifier,
    evaluate_severity_classifier,
)

# Get temporal splits
X_train, X_val, X_test, y_train, y_val, y_test = get_severity_classification_splits(df)

# Train CatBoost classifier (balanced class weights)
model = train_severity_classifier(X_train, y_train, X_val, y_val, model_type="catboost")
results = evaluate_severity_classifier(model, X_test, y_test)

# Results: 62% accuracy, 94%+ adjacent accuracy
```

Note: OrdinalClassifier and SeverityEnsemble are also implemented in severity.py
but CatBoost alone performs equally well with simpler code.

## Key Functions Reference

### src/models/severity.py
| Function | Purpose |
|----------|---------|
| `get_severity_classification_splits()` | Temporal train/val/test split |
| `train_severity_classifier()` | Train CatBoost (recommended) or LightGBM |
| `evaluate_severity_classifier()` | Evaluate with accuracy + adjacent accuracy |

### src/models/classification_shap.py
| Function | Purpose |
|----------|---------|
| `compute_stacking_ensemble_shap()` | SHAP values for StackingEnsemble |
| `explain_player_ensemble()` | Explain single player prediction |
| `build_temporal_output_df()` | Final output with predictions + SHAP |

## Dependencies
- pandas, numpy, scikit-learn
- catboost, lightgbm, xgboost
- shap (explainability)
- optuna (hyperparameter tuning)
- streamlit (dashboard)
