# Injury Risk Predictor - AI Agent Instructions

## Project Overview
ML project predicting player injury risk in football/soccer using Premier League data (1992-2025). Analyzes injury impact on team performance and player stats.

## Data Architecture

### Three Core Datasets
1. **player_injuries_impact.csv** (658 rows)
   - Injury events with player context (name, team, position, FIFA rating, age, season)
   - Match performance metrics: 3 matches before injury, missed matches, 3 matches after return
   - Tracks player performance deltas (ratings, opposition, results, goal differential)
   
2. **All_Players_1992-2025.csv** (92,172 rows)
   - Comprehensive player statistics across all seasons and leagues
   - 140+ statistical features (goals, assists, expected goals, defense metrics, passing data)
   - Cross-referenced by PlayerID, Player name, Squad, Season
   
3. **premier-league-matches.csv** (12,028 rows)
   - Match records with Season_End_Year, Week, Date, Home/Away teams, goals, result (H/D/A)
   - Baseline for team performance context

### Data Quality Notes
- Missing values in injury data: use variants like "N.A", "N.A.", "NA", "N/A", "n.a", "NaN" → normalize to np.nan
- Name matching requires careful string handling (see exploration.ipynb imports: unidecode, unicodedata)
- Multi-position players encoded as comma-separated strings ("DF,MF")

## Codebase Structure

### Current State
- **src/**: Modular structure (data_loaders/, features/, models/, pipelines/, preprocessing/, utils/) - currently empty, ready for implementation
- **notebooks/exploration.ipynb**: Primary working notebook
  - Initial data loading, inspection, and basic cleaning
  - Foundation for feature engineering and modeling
- **data/**: Directory for processed outputs (features/, processed/, raw/)

### Development Pattern
Build Python modules in src/ directories, import and test in notebooks. Follow the separation:
- `src/data_loaders/`: Load and unify datasets, handle CSV I/O
- `src/preprocessing/`: Handle missing values, name normalization, date parsing
- `src/features/`: Feature engineering (injury impact metrics, player form windows, team context)
- `src/models/`: ML models for injury risk prediction
- `src/pipelines/`: End-to-end workflows

## Key Implementation Patterns

### String Normalization for Player Names
Use `unidecode` + `unicodedata` for accent handling when matching players across datasets. Example: "Fabián Schär" → "Fabian Schar"

### Window-Based Features
Injury data captures 3-match windows (before/after) with:
- Match results (win/draw/loss)
- Goal differential (GD)
- Player performance ratings
- Opposition strength
Use these for calculating performance deltas and recovery patterns.

### Missing Value Strategy
- Injury_df: Standardize NA variants to np.nan before analysis
- Stats_df: Preserve 0 values for counting stats (appears, minutes)
- Forward-fill for match sequences where data is sparse

## Integration Points
1. Match outcomes link injury_df to premier-league-matches.csv via Date/Team/Season
2. Player stats link via Player name (requires normalization) + Season
3. Core analysis: injury → missed matches → performance impact → recovery trajectory

## Testing & Exploration
- Run cells sequentially in exploration.ipynb to validate data loading
- Verify shape consistency after joins (expect loss of rows for injury events with incomplete match records)
- Check FIFA ratings and player performance distributions for outliers before feature engineering
