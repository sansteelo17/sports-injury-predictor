# Football Injury Risk Predictor - Project Context

## Overview

ML system predicting injury risk for football (soccer) players using ensemble models (CatBoost, LightGBM, XGBoost). Features a Next.js frontend, FastAPI backend, narrative generation with LLM enrichment, and FPL (Fantasy Premier League) insights. Educational/portfolio project, not for medical use.

**Competition is a first-class dimension.** Both club leagues (Premier League, La Liga) and international tournaments (FIFA World Cup 2026) are Competitions. They share the same models, payloads, and frontend, but route through `src/competitions/registry.py` capabilities (`has_fpl`, `standings_kind`, `acwr_spike_threshold`, `fixture_label`, `risk_calibration_cohort`) instead of per-league branching.

## Data Sources

- `data/raw/All_Players_1992-2025.csv` - Historical player stats (49MB, 92K+ rows)
- `data/raw/player_injuries_impact.csv` - Injury records with days lost
- `data/raw/premier-league-matches.csv` - Match data for workload calculations
- **FPL API** (live) - Current season stats, fixtures, gameweek data via `src/data_loaders/fpl_api.py`
- **football-data.org API** - Match results, squad data via `src/data_loaders/api_client.py`
- **Odds API** - Bookmaker odds via `src/data_loaders/odds_api.py`
- **football-data.org WC endpoint** - WC 2026 squads, fixtures, groups via `src/data_loaders/international_squads.py` (rate-limited, 6.5s between calls on free tier)
- **Wikipedia per-country squad pages** - National-team caps + intl_goals via `src/data_loaders/national_team_caps.py` (24h disk cache, 0.5s rate limit)
- `data/processed/inference_international_world_cup_2026.pkl` - WC inference frame (1203 rows: 237 risk-featured + 966 baseline identity-only)

## Architecture

```
api/
└── main.py                    # FastAPI backend (3700+ lines, monolithic)

frontend/
├── src/app/page.tsx           # Next.js entry
├── src/components/
│   ├── PlayerCard.tsx         # Main player view (risk, FPL, narrative, injury map)
│   ├── PlayerList.tsx         # Player list/search
│   ├── TeamOverview.tsx       # Team-level risk summary
│   └── ...
├── src/types/api.ts           # TypeScript types for API responses
└── src/lib/api.ts             # API client

src/
├── data_loaders/
│   ├── load_data.py           # Load raw CSVs
│   ├── fpl_api.py             # FPL API client (live data)
│   ├── api_client.py          # football-data.org client
│   └── odds_api.py            # Bookmaker odds client
├── preprocessing/             # Data cleaning pipeline
├── feature_engineering/
│   ├── workload.py            # ACWR, monotony, strain, fatigue
│   ├── injury_history.py      # Previous injuries, days since last
│   ├── match_features.py      # Matches in 7/14/30 days, rest days
│   ├── archetype.py           # K-means clustering for player profiles
│   ├── severity.py            # Injury severity features
│   └── negative_sampling.py   # Generate non-injury samples
├── models/
│   ├── stacking_ensemble.py   # Stacking with CatBoost meta-learner (primary)
│   ├── severity.py            # CatBoost severity classifier
│   └── ...
├── inference/
│   ├── inference_pipeline.py  # Full prediction pipeline
│   ├── story_generator.py     # Narrative generation (Yara voice, FPL insights)
│   ├── context_rag.py         # RAG context chunks for LLM enrichment
│   ├── llm_client.py          # LLM integration (Ollama/OpenAI-compatible)
│   ├── risk_card.py           # Generate player risk cards
│   └── validation.py          # Input validation for inference
├── utils/
│   └── model_io.py            # Model serialization/loading
└── dashboard/
    └── player_dashboard.py    # Streamlit components (legacy)

scripts/
└── refresh_predictions.py     # Rebuild inference_df from live data
```

## Competitions (multi-comp model)

- Registry: `src/competitions/registry.py`. Three competitions live: `premier-league`, `la-liga`, `world-cup-2026`.
- Backend resolves via `for_id(slug)`. `api/main.py` reads `competition_id` off `inference_df` rows; international rows take an early branch in `player_row_to_risk` (`_international_row_to_risk`) that skips ALL FPL/odds/scoring/Yara-betting fetches and instead populates `international_context`.
- `_league_prob_series` cohorts by `competition_id` and drops NaN `ensemble_prob` so baseline rows don't bias percentile thresholds.
- Risk-level percentile cutoffs (0.80 high / 0.40 low) are computed per cohort, so WC has its own calibration that's independent of the club leagues.
- WC `acwr_spike_threshold = 2.0` (wider than club's 1.8) because international windows are longer between matches.
- **Baseline rows** (WC players without club data): `has_risk_features=False`, `ensemble_prob=NaN`. They surface as `risk_level="Unknown"` and the frontend renders them with neutral slate styling + a "Baseline tournament row" notice. Do NOT fabricate a risk score for them.
- **International narrative LLM override**: set `INTL_NARRATIVE_LLM_PROVIDER=openai_compatible` in `.env` to route ONLY WC narratives through OpenAI while keeping club narratives on the global provider (e.g. free Ollama). The override is implemented as a scoped env swap inside `_international_row_to_risk`.

## Key Thresholds (keep in sync across files)

- **ACWR spike_flag**: `> 1.8` (workload.py, story_generator.py, context_rag.py, validation.py, risk_card.py, negative_sampling.py, refresh_predictions.py, api/main.py)
- **ACWR elevated narrative**: `>= 1.5` (story_generator.py, context_rag.py)
- **Demanding schedule**: `matches_last_7 >= 3`, `matches_last_30 >= 8`
- **Probability tiers**: 0.60 (high), 0.40 (elevated), 0.25 (moderate), 0.20 (low)
- **Risk levels** (api/main.py `get_risk_level`): percentile-based with 0.80/0.40 thresholds

## Narrative System (story_generator.py + context_rag.py + llm_client.py)

- **Yara IS the model.** She speaks in first person ("I have", "I see"). Never say "my model says" or "the model reads". Yara is the analyst, not a wrapper around a model.
- **Voice**: Natural, conversational, like a sharp football friend. Stat-driven but human. No template-speak, no dropdown labels ("Start with bench cover"). Say things like "I see why you would want him" or "The numbers just are not there."
- **No em dashes.** Do not use — in any generated text.
- **`_stat_lead(number, unit)`**: Formats stat-first leads like "4 career injuries. " — the sentence that follows must NOT restate the number
- **`_position_group()`**: Returns "goalkeeper", "defender", "attacker", "midfielder", or "other" — goalkeepers are NOT defenders
- **`_safe_float()` / `_safe_int()`**: Handle None, NaN, and Inf — defined in both story_generator.py and context_rag.py
- **Topic dedup**: `covered` set in `generate_player_story()` tracks what the LEAD already said to prevent repetition
- **3-layer match data**: (1) player recent form, (2) opponent defensive record, (3) player H2H vs opponent
- **Role-aware signals**: Goals/assists for attackers, clean sheets for defenders. "Opponent concedes a lot" is a PRO for attackers but IRRELEVANT for defenders (their value comes from clean sheets, not opponent defensive weakness).

## FPL Insight Logic (story_generator.py `get_fpl_insight`)

- **Position-aware matchup signals**:
  - Attackers/mids: form = goal involvements, fixture = weak opponent defense (opp_conceded >= 1.0), H2H = scoring record
  - Defenders/GKs: form = clean sheets, fixture = team dominance in fixture history, H2H = clean sheet record
- **Decision is data-driven**: gather pros (form, fixture, H2H, home, ownership) and cons (risk, cold form, tight opponent), then weigh them
- **"Fan desire" case**: When form + fixture + H2H ALL align for a high-risk player, use language like "I can see why a {team} fan would want {name} in against {opponent}". This ONLY fires when all three signals point the same way — not for partial matches.
- **FPL Insight and Value must be coherent**: If value says Avoid, insight should not say Start unless the matchup fully overrides it. Both sections use the same position-aware matchup logic.
- **No action label prefix**: The insight text is the sentence itself, not "Start. {reason}" — the action is metadata, not prose.

## FPL Value Logic (story_generator.py `get_fpl_value_assessment`)

- Tier thresholds based on `adjusted_value = output_signal * risk_factor`
- **Matchup override**: When form + fixture + H2H all align (position-aware), Avoid bumps to Rotation
- Verdict text is position-aware: defenders get clean sheet language, attackers get scoring language

## Known Issues / Tech Debt

- `api/main.py` is monolithic (3700+ lines) — `player_row_to_risk()` alone is 212 lines
- `_safe_float`/`_safe_int` defined in 3 places (story_generator, context_rag, api/main.py) — should be shared
- Variable extraction duplicated between `generate_player_story()` and `get_fpl_insight()` (~40 lines each)
- `get_risk_level()` accepts unused `row` parameter at all call sites
- `api/main.py` global mutable state (inference_df, caches) accessed from background threads without locks
- FPL API calls not cached per-request — multiple round-trips per player view
- `calculate_scoring_odds()` treats `goals_per_90` as probability (it's a rate) — scoring odds are systematically off
- `SECTION_RAG_PREFIXES` in context_rag.py has identical templates — the variation system does nothing
- `calculate_clean_sheet_odds()` uses hardcoded 1.2 PL average, ignores actual team defense
- Fixture/venue RAG chunks overlap — can produce repetitive LLM output
- `severity_skewed` outlier detection doesn't handle duplicate max values correctly
- ShareCard removed — needs full rebuild with useful data before re-adding

## Running

```bash
# Backend
cd "Injury Risk Predictor" && uvicorn api.main:app --reload

# Frontend
cd "Injury Risk Predictor/frontend" && npm run dev

# Refresh predictions from live data
python scripts/refresh_predictions.py --mode api

# Rebuild World Cup 2026 inference frame (squads + fixtures + caps enrichment)
python scripts/build_international_predictions.py --reuse-cache --enrich-caps

# Yara Studio (content pipeline) — per competition
python yara_studio.py --competition premier-league --top-n 30 --dry-run
python yara_studio.py --competition world-cup-2026 --no-journalist --dry-run
```

## Model Pipeline

1. **Classification**: Predict if player will get injured (binary)
   - Ensemble of CatBoost + LightGBM + XGBoost (stacking)
   - Threshold tuned for F1/precision-recall balance

2. **Severity**: Predict injury duration (short/medium/long)
   - CatBoost with `auto_class_weights='Balanced'` (62% accuracy, 94%+ adjacent accuracy)

3. **Archetype**: Cluster players into risk profiles
   - Injury Prone, Recurring Issues, Fragile, Durable, Currently Vulnerable, Moderate Risk, Clean Record

## Dependencies

- pandas, numpy, scikit-learn, catboost, lightgbm, xgboost
- shap (explainability), optuna (tuning)
- fastapi, uvicorn (API)
- next.js, tailwindcss (frontend)
- Ollama or OpenAI-compatible LLM (narrative enrichment, optional)
