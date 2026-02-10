import pandas as pd
import numpy as np
import shap
import warnings

from ..utils.logger import get_logger
from .validation import (
    validate_inference_dataframe,
    validate_dataframe_and_raise,
    ValidationError
)
from ..feature_engineering.temporal_features import (
    add_temporal_features,
    add_fixture_density_features
)
from ..feature_engineering.position_features import (
    add_position_risk_features,
    add_position_workload_interaction
)
from ..feature_engineering.severity import (
    add_player_injury_history_features,
    build_injury_features
)
from ..preprocessing.rename_finaldf_cols import rename_final_df_columns

logger = get_logger(__name__)


# ============================================================
# 0. COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================
def apply_all_feature_engineering(
    df: pd.DataFrame,
    date_column: str = "injury_datetime",
    position_column: str = "position",
    team_column: str = "player_team"
) -> pd.DataFrame:
    """
    Apply ALL feature engineering transformations to match training pipeline.

    This is the missing link between raw data and model inference. The training
    pipeline applies these transformations manually; this function ensures
    inference data gets the same treatment.

    Automatically cleans up merge artifacts (age_x → age, season_x → season).

    Features added:
        Temporal (from add_temporal_features):
            - month, day_of_week, week_of_year
            - is_preseason, is_midseason, is_endseason
            - is_winter, is_christmas_period, is_season_crunch
            - is_weekend_match, is_midweek_match
            - season_week, season_fatigue_factor

        Fixture density (from add_fixture_density_features):
            - days_since_last_match
            - short_recovery, minimal_recovery, normal_recovery, extended_rest

        Position risk (from add_position_risk_features):
            - position_normalized
            - position_base_risk, position_sprint_risk, position_contact_risk
            - is_forward, is_midfielder, is_defender, is_goalkeeper
            - is_wide_position, is_central_position

        Position-workload interactions (from add_position_workload_interaction):
            - sprint_load_risk, contact_load_risk
            - position_adjusted_acwr
            - wide_player_congestion, defender_congestion
            - age_forward_risk, age_defender_risk

        Player injury history (from add_player_injury_history_features):
            - player_injury_count: Total injuries for this player
            - player_avg_severity: Player's average injury duration
            - player_worst_injury: Longest injury for this player
            - player_severity_std: Variability in injury severity
            - is_injury_prone: 1 if player has 3+ injuries
            - prev_injury_same_area: 1 if previous injury to same body area

    Args:
        df: DataFrame with raw features
        date_column: Column containing datetime for temporal features
        position_column: Column containing player position
        team_column: Column containing team name for fixture density

    Returns:
        DataFrame with all engineered features added

    Example:
        >>> df_enriched = apply_all_feature_engineering(
        ...     final_df,
        ...     date_column="date_of_injury",
        ...     position_column="position",
        ...     team_column="player_team"
        ... )
    """
    df = df.copy()
    logger.info("Applying complete feature engineering pipeline...")

    # 0. Clean up merge artifacts (age_x → age, season_x → season, etc.)
    df = rename_final_df_columns(df)
    logger.debug("Cleaned up merge artifact columns (age_x → age, etc.)")

    # 1. Temporal features
    if date_column in df.columns:
        df = add_temporal_features(df, date_column=date_column)
        logger.debug(f"Added temporal features from '{date_column}'")
    else:
        logger.warning(f"Date column '{date_column}' not found. Skipping temporal features.")

    # 2. Fixture density features
    if date_column in df.columns and team_column in df.columns:
        df = add_fixture_density_features(
            df, date_column=date_column, team_column=team_column
        )
        logger.debug("Added fixture density features")
    else:
        logger.warning(
            f"Columns '{date_column}' or '{team_column}' not found. "
            "Skipping fixture density features."
        )

    # 3. Position risk features
    if position_column in df.columns:
        df = add_position_risk_features(df, position_column=position_column)
        logger.debug(f"Added position risk features from '{position_column}'")
    else:
        logger.warning(f"Position column '{position_column}' not found. Skipping position features.")

    # 4. Position-workload interaction features (requires 'age' column)
    if position_column in df.columns:
        if "age" not in df.columns:
            logger.warning("No 'age' column found. Age-based interactions will be skipped.")
        df = add_position_workload_interaction(df, position_column=position_column)
        logger.debug("Added position-workload interaction features")

    # 5. Player injury history features (requires 'severity_days' and 'name')
    if "severity_days" in df.columns and "name" in df.columns:
        # First ensure injury_type and body_area exist for full history features
        if "injury" in df.columns and "injury_type" not in df.columns:
            df = build_injury_features(df)
            logger.debug("Added injury classification features (body_area, injury_type)")

        df = add_player_injury_history_features(df)
        logger.debug("Added player injury history features (is_injury_prone, player_injury_count, etc.)")
    else:
        missing = []
        if "severity_days" not in df.columns:
            missing.append("severity_days")
        if "name" not in df.columns:
            missing.append("name")
        logger.warning(f"Missing {missing}. Skipping player injury history features.")

    logger.info(f"Feature engineering complete. DataFrame now has {len(df.columns)} columns.")

    return df


# ============================================================
# 1. BUILD INFERENCE FEATURES (same as training)
# ============================================================
def build_inference_features(all_matches, player_metadata, validate_input=True, strict=True):
    """
    Mirrors the feature-engineering used in training.

    Args:
        all_matches: DataFrame with match and player data
        player_metadata: DataFrame with player static information
        validate_input: If True, validate input data before processing
        strict: If True, raise errors on validation failures. If False, log warnings only.

    Returns:
        DataFrame with inference features

    Raises:
        ValidationError: If validate_input=True and strict=True and validation fails
    """

    df = all_matches.copy()

    # ---------------------------------------------------
    # 2. Compute rest_days_before_injury (same as training)
    # ---------------------------------------------------
    if "rest_days" in df.columns:
        df["rest_days_before_injury"] = df["rest_days"]
    else:
        logger.warning("Missing 'rest_days' column, defaulting to 0. This may affect prediction accuracy.")
        df["rest_days_before_injury"] = 0

    # ---------------------------------------------------
    # 3. Ensure numeric injury-history features exist
    # ---------------------------------------------------
    # Fill nulls for injury history (0 injuries and 999 days are reasonable defaults)
    null_count_injuries = df["previous_injuries"].isna().sum()
    null_count_days = df["days_since_last_injury"].isna().sum()

    if null_count_injuries > 0:
        logger.info(f"Filling {null_count_injuries} null 'previous_injuries' values with 0")
    df["previous_injuries"] = df["previous_injuries"].fillna(0)

    if null_count_days > 0:
        logger.info(f"Filling {null_count_days} null 'days_since_last_injury' values with 999")
    df["days_since_last_injury"] = df["days_since_last_injury"].fillna(999)

    # ---------------------------------------------------
    # 4. Ensure all required model features exist
    # ---------------------------------------------------
    required_cols = [
        "matches_last_7", "matches_last_14", "matches_last_30",
        "rest_days_before_injury", "avg_rest_last_5",
        "goals_for_last_5", "goals_against_last_5",
        "goal_diff_last_5", "avg_goal_diff_last_5",
        "form_last_5", "form_avg_last_5",
        "win_ratio_last_5", "win_streak", "loss_streak",
        "previous_injuries", "days_since_last_injury",
        "acute_load", "chronic_load", "acwr",
        "monotony", "strain", "fatigue_index",
        "workload_slope", "spike_flag",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        error_msg = (
            f"Missing {len(missing_cols)} required feature columns: {', '.join(missing_cols)}. "
            f"These will be filled with zeros, which may result in unreliable predictions. "
            f"Please ensure proper feature engineering is applied before inference."
        )

        if strict:
            logger.error(error_msg)
            raise ValidationError(error_msg)
        else:
            logger.warning(error_msg)

        # Add missing columns with zeros
        for c in missing_cols:
            df[c] = 0

    # ---------------------------------------------------
    # 5. Clean NaN values with warnings
    # ---------------------------------------------------
    nan_counts = df[required_cols].isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]

    if len(cols_with_nans) > 0:
        logger.warning(
            f"Found NaN values in {len(cols_with_nans)} columns: "
            f"{dict(cols_with_nans.head(5))}. Filling with 0."
        )

    df = df.fillna(0)

    # ---------------------------------------------------
    # 6. VALIDATION (if enabled)
    # ---------------------------------------------------
    if validate_input:
        logger.info("Running input validation...")
        validation_result = validate_inference_dataframe(df, strict=strict)

        if not validation_result['valid']:
            logger.error(f"Validation failed:\n{validation_result['summary']}")

            if strict:
                raise ValidationError(
                    f"Input validation failed. {validation_result['invalid_rows']} of "
                    f"{validation_result['total_rows']} rows have validation errors.\n"
                    f"{validation_result['summary']}"
                )
        else:
            logger.info(
                f"Validation passed: {validation_result['valid_rows']}/{validation_result['total_rows']} rows valid"
            )

        # Log warnings even if validation passes
        if validation_result['row_warnings']:
            logger.warning(
                f"{len(validation_result['row_warnings'])} rows have warnings. "
                f"Predictions may be less reliable for these records."
            )

    return df


# ============================================================
# 2. RUN ENSEMBLE MODELS
# ============================================================
def add_ensemble_predictions(df, ensemble, feature_cols=None):
    """
    Add predictions from a StackingEnsemble model.

    Args:
        df: DataFrame with features
        ensemble: Trained StackingEnsemble instance
        feature_cols: Optional list of feature columns (uses ensemble's if not provided)

    Returns:
        DataFrame with ensemble predictions added
    """
    if feature_cols is None:
        feature_cols = ensemble.feature_names_

    X = df[feature_cols].copy()

    # Get ensemble probability
    proba = ensemble.predict_proba(X)
    df["ensemble_prob"] = proba[:, 1]

    # Get individual model predictions
    base_preds = ensemble.get_base_predictions(X)
    for col in base_preds.columns:
        df[col] = base_preds[col]

    # Compute agreement from individual model probs
    prob_cols = [c for c in base_preds.columns if c.endswith("_prob")]
    df["agreement"] = sum(
        (base_preds[c] > 0.5).astype(int) for c in prob_cols
    )

    df["confidence"] = df["agreement"].map({
        3: "very-high",
        2: "high",
        1: "medium",
        0: "low"
    })

    return df


def add_model_predictions(df, cat_model, lgb_model, xgb_model, feature_cols):

    # CATBOOST → uses full feature set (including strings)
    X_cat = df[feature_cols]
    df["catboost_prob"] = cat_model.predict_proba(X_cat)[:, 1]

    # Remove non-numeric features for LightGBM/XGB
    feature_cols_numeric = [
        c for c in feature_cols
        if df[c].dtype in ["float64", "int64", "float32", "int32", "bool"]
    ]

    X_lgb = df[feature_cols_numeric]
    X_xgb = df[feature_cols_numeric]

    df["lgb_prob"] = lgb_model.predict_proba(X_lgb)[:, 1]
    df["xgb_prob"] = xgb_model.predict_proba(X_xgb)[:, 1]

    df["agreement"] = (
        (df["catboost_prob"] > 0.5).astype(int) +
        (df["lgb_prob"] > 0.5).astype(int) +
        (df["xgb_prob"] > 0.5).astype(int)
    )

    df["confidence"] = df["agreement"].map({
        3: "very-high",
        2: "high",
        1: "medium",
        0: "low"
    })

    return df


# ============================================================
# 3. SEVERITY PREDICTION + ARCHETYPE
# ============================================================
def predict_severity_class(df, severity_classifier, feature_cols=None):
    """
    Predict severity class (short/medium/long) for each row using trained classifier.

    Args:
        df: DataFrame with features for prediction
        severity_classifier: Trained severity classifier (CatBoost or LightGBM)
        feature_cols: Optional list of feature columns (uses model's if not provided)

    Returns:
        DataFrame with predicted_severity_class column added
    """
    from ..models.severity import SEVERITY_LABELS, _prepare_catboost_data, _sanitize_column_names

    df = df.copy()

    # Get feature columns from model if not provided
    if feature_cols is None:
        if hasattr(severity_classifier, 'feature_names_'):
            feature_cols = severity_classifier.feature_names_
        else:
            raise ValueError("feature_cols must be provided if model doesn't have feature_names_")

    # Check which features are available
    missing_cols = [c for c in feature_cols if c not in df.columns]

    if missing_cols:
        logger.warning(f"Missing {len(missing_cols)} severity features: {missing_cols[:5]}...")
        # Fill missing with defaults
        for col in missing_cols:
            df[col] = 0

    X = df[feature_cols].copy()

    # Detect model type and preprocess accordingly
    model_class = type(severity_classifier).__name__

    if "CatBoost" in model_class:
        X_enc, _ = _prepare_catboost_data(X)
    else:
        # LightGBM/XGBoost: sanitize column names and encode categoricals
        X_enc = _sanitize_column_names(X)
        for col in X_enc.select_dtypes(include=['object', 'category']).columns:
            X_enc[col] = X_enc[col].astype('category').cat.codes

    # Predict
    y_pred_enc = severity_classifier.predict(X_enc)

    # Convert back to labels
    if hasattr(severity_classifier, '_label_encoder'):
        df["predicted_severity_class"] = severity_classifier._label_encoder.inverse_transform(y_pred_enc)
    else:
        # Fallback: use encoded values directly with SEVERITY_LABELS
        df["predicted_severity_class"] = [SEVERITY_LABELS[int(i)] for i in y_pred_enc]

    logger.info(f"Predicted severity classes: {df['predicted_severity_class'].value_counts().to_dict()}")

    return df


def add_archetype(df, archetype_df, default_archetype="Unknown Profile"):
    """
    Merge archetype assignments from clustering.

    Args:
        df: Inference DataFrame
        archetype_df: DataFrame with 'name' and 'archetype' columns
        default_archetype: Fallback for players not in clustering (e.g., insufficient injury history)

    Returns:
        DataFrame with archetype column added
    """
    if "archetype" not in archetype_df.columns:
        logger.warning("No 'archetype' column in archetype_df, skipping archetype merge")
        return df

    merge_cols = ["name", "archetype"]
    if "archetype_description" in archetype_df.columns:
        merge_cols.append("archetype_description")

    df = df.merge(
        archetype_df[merge_cols].drop_duplicates(subset=["name"]),
        on="name",
        how="left"
    )

    # Fill missing archetypes (players not in clustering due to insufficient injury data)
    missing_count = df["archetype"].isna().sum()
    if missing_count > 0:
        df["archetype"] = df["archetype"].fillna(default_archetype)
        logger.info(f"Assigned '{default_archetype}' to {missing_count} players not in clustering")

    return df


def build_player_history_lookup(historical_injury_df):
    """
    Build a player history lookup table from historical injury data.

    This function aggregates injury history statistics per player that can be
    used at inference time. Call this ONCE on your training/historical injury
    data that has severity_days computed.

    Args:
        historical_injury_df: DataFrame with historical injuries containing:
            - name: Player name
            - severity_days: Duration of each injury
            - body_area (optional): For same-area recurrence tracking

    Returns:
        DataFrame with one row per player containing:
            - player_injury_count: Total injuries for this player
            - player_avg_severity: Average injury duration
            - player_worst_injury: Longest injury
            - player_severity_std: Variability in injury duration
            - is_injury_prone: 1 if player has 3+ injuries

    Example:
        >>> player_history = build_player_history_lookup(severity_final)
        >>> inference_df = add_player_history_features(inference_df, player_history)
    """
    df = historical_injury_df.copy()

    if 'name' not in df.columns:
        raise ValueError("historical_injury_df must have 'name' column")

    if 'severity_days' not in df.columns:
        raise ValueError("historical_injury_df must have 'severity_days' column")

    # Aggregate player-level statistics
    player_stats = df.groupby('name').agg({
        'severity_days': ['count', 'mean', 'max', 'std']
    }).reset_index()
    player_stats.columns = ['name', 'player_injury_count', 'player_avg_severity',
                            'player_worst_injury', 'player_severity_std']

    # Fill NaN std (players with only 1 injury)
    player_stats['player_severity_std'] = player_stats['player_severity_std'].fillna(0)

    # Injury-prone flag (3+ injuries)
    player_stats['is_injury_prone'] = (player_stats['player_injury_count'] >= 3).astype(int)

    logger.info(f"Built player history lookup for {len(player_stats)} players")
    logger.info(f"  Injury-prone players (3+ injuries): {player_stats['is_injury_prone'].sum()}")

    return player_stats


def add_player_history_features(df, player_history_lookup):
    """
    Merge pre-computed player history features into inference dataframe.

    Use this at inference time to add player history features that were
    computed from historical injury data.

    Args:
        df: Inference DataFrame with 'name' column
        player_history_lookup: DataFrame from build_player_history_lookup()

    Returns:
        DataFrame with player history features added

    Example:
        >>> # During training/data prep
        >>> player_history = build_player_history_lookup(severity_final)
        >>>
        >>> # At inference time
        >>> inference_df = add_player_history_features(inference_df, player_history)
    """
    if 'name' not in df.columns:
        logger.warning("No 'name' column in inference df, skipping player history merge")
        return df

    history_cols = ['name', 'player_injury_count', 'player_avg_severity',
                    'player_worst_injury', 'player_severity_std', 'is_injury_prone']

    # Only merge columns that exist in the lookup
    available_cols = [c for c in history_cols if c in player_history_lookup.columns]

    # Drop any pre-existing player history columns to avoid _x/_y suffixes
    cols_to_merge = [c for c in available_cols if c != 'name']
    existing = [c for c in cols_to_merge if c in df.columns]
    if existing:
        logger.debug(f"Dropping {len(existing)} existing player history columns to replace with lookup values")
        df = df.drop(columns=existing)

    df = df.merge(
        player_history_lookup[available_cols].drop_duplicates(subset=['name']),
        on='name',
        how='left'
    )

    # Fill missing values for players not in historical data
    fill_values = {
        'player_injury_count': 0,
        'player_avg_severity': 0,
        'player_worst_injury': 0,
        'player_severity_std': 0,
        'is_injury_prone': 0
    }

    for col, default in fill_values.items():
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                df[col] = df[col].fillna(default)
                logger.debug(f"Filled {missing} missing '{col}' values with {default}")

    logger.info(f"Added player history features from lookup")

    return df


def add_severity_and_archetype(df, severity_df):
    """
    DEPRECATED: Merges ground truth severity_days and archetype.

    WARNING: This function uses ACTUAL severity_days which is data leakage
    at inference time. Use predict_severity_class() + add_archetype() instead.

    Kept for backwards compatibility with existing notebooks.
    """
    warnings.warn(
        "add_severity_and_archetype() uses ground truth severity_days which is data leakage. "
        "Use predict_severity_class() and add_archetype() for proper inference.",
        DeprecationWarning
    )

    merge_cols = ["name"]
    if "severity_days" in severity_df.columns:
        merge_cols.append("severity_days")
    if "archetype" in severity_df.columns:
        merge_cols.append("archetype")

    df = df.merge(
        severity_df[merge_cols].drop_duplicates(subset=["name"]),
        on="name",
        how="left"
    )
    return df


# ============================================================
# 4. SHAP EXPLANATIONS (CatBoost)
# ============================================================
def add_shap_values(df, cat_model, feature_cols):
    explainer = shap.TreeExplainer(cat_model)
    shap_vals = explainer.shap_values(df[feature_cols])

    df["shap_sum"] = np.sum(shap_vals, axis=1)
    df["shap_values"] = shap_vals.tolist()
    return df


# ============================================================
# 5. FINAL PIPELINE FUNCTION
# ============================================================
def build_full_inference_df(
    all_matches,
    player_metadata,
    severity_df,
    cat_model,
    lgb_model,
    xgb_model,
    validate_input=True,
    strict=True
):
    """
    MASTER PIPELINE
    Produces final inference_df with:
      - risk probabilities
      - ensemble confidence
      - severity prediction
      - archetype
      - SHAP values

    Args:
        all_matches: DataFrame with match and player data
        player_metadata: DataFrame with player static information
        severity_df: DataFrame with severity predictions and archetypes
        cat_model: Trained CatBoost model
        lgb_model: Trained LightGBM model
        xgb_model: Trained XGBoost model
        validate_input: If True, validate input data before inference (recommended)
        strict: If True, raise errors on validation failures. If False, log warnings only.

    Returns:
        DataFrame with predictions, confidence scores, and SHAP values

    Raises:
        ValidationError: If validate_input=True and strict=True and validation fails

    Example:
        >>> # Production mode (strict validation)
        >>> df = build_full_inference_df(
        ...     all_matches, player_metadata, severity_df,
        ...     cat_model, lgb_model, xgb_model,
        ...     validate_input=True, strict=True
        ... )

        >>> # Development mode (warnings only)
        >>> df = build_full_inference_df(
        ...     all_matches, player_metadata, severity_df,
        ...     cat_model, lgb_model, xgb_model,
        ...     validate_input=True, strict=False
        ... )
    """

    logger.info(f"Starting inference pipeline for {len(all_matches)} matches")
    logger.info(f"Validation: {'enabled (strict)' if validate_input and strict else 'enabled (warnings only)' if validate_input else 'disabled'}")

    try:
        # 1. Build feature table with validation
        df = build_inference_features(
            all_matches,
            player_metadata,
            validate_input=validate_input,
            strict=strict
        )
        logger.debug(f"Built inference features: {df.shape}")

        # 2. Determine feature columns used by model
        feature_cols = cat_model.feature_names_

        # 3. Add CatBoost + LGB + XGB predictions
        df = add_model_predictions(df, cat_model, lgb_model, xgb_model, feature_cols)
        logger.info(f"Generated predictions for {len(df)} records")

        # 4. Merge severity + archetype
        df = add_severity_and_archetype(df, severity_df)
        logger.debug("Merged severity and archetype data")

        # 5. Add interpretability (SHAP)
        df = add_shap_values(df, cat_model, feature_cols)
        logger.debug("Computed SHAP values for interpretability")

        logger.info("Inference pipeline completed successfully")
        return df

    except ValidationError as e:
        logger.error(f"Validation error in inference pipeline: {str(e)}")
        raise

    except Exception as e:
        logger.error(f"Error in inference pipeline: {str(e)}", exc_info=True)
        raise


# ============================================================
# 6. ENSEMBLE-BASED INFERENCE (uses StackingEnsemble)
# ============================================================
def build_inference_df_with_ensemble(
    all_matches,
    player_metadata,
    archetype_df,
    ensemble,
    severity_classifier=None,
    severity_feature_cols=None,
    player_history_lookup=None,
    validate_input=True,
    strict=True,
    date_column="injury_datetime",
    position_column="position",
    team_column="player_team"
):
    """
    Inference pipeline using StackingEnsemble model with proper severity prediction.

    This is the preferred method for production inference. It uses:
    - StackingEnsemble for injury RISK classification (will they get injured?)
    - Severity classifier for injury DURATION prediction (short/medium/long)
    - Archetype clustering for player profiling
    - Player history lookup for historical injury patterns

    Automatically cleans up merge artifacts (age_x → age, season_x → season).

    Args:
        all_matches: DataFrame with match and player data
        player_metadata: DataFrame with player static information
        archetype_df: DataFrame with 'name' and 'archetype' columns from clustering
        ensemble: Trained StackingEnsemble instance for risk classification
        severity_classifier: Optional trained severity classifier for duration prediction
        severity_feature_cols: Feature columns for severity prediction (uses model's if not provided)
        player_history_lookup: Optional DataFrame from build_player_history_lookup() with
            pre-computed player injury history features (player_injury_count, is_injury_prone, etc.)
        validate_input: If True, validate input data before inference
        strict: If True, raise errors on validation failures
        date_column: Column name for datetime (for temporal features)
        position_column: Column name for player position
        team_column: Column name for team (for fixture density)

    Returns:
        DataFrame with:
        - ensemble_prob: Injury risk probability
        - lgb_prob, xgb_prob, catboost_prob: Individual model probabilities
        - agreement, confidence: Model agreement metrics
        - predicted_severity_class: Predicted injury duration (if severity_classifier provided)
        - archetype: Player archetype from clustering
        - player_injury_count, is_injury_prone, etc.: Player history features (if lookup provided)

    Example:
        >>> from src.models import StackingEnsemble, train_stacking_ensemble
        >>> ensemble = train_stacking_ensemble(X_train, y_train, X_test, y_test)
        >>> severity_clf = train_severity_classifier(X_sev_train, y_sev_train, ...)
        >>> player_history = build_player_history_lookup(severity_final)
        >>> inference_df = build_inference_df_with_ensemble(
        ...     all_matches, player_info, archetype_df,
        ...     ensemble, severity_classifier=severity_clf,
        ...     player_history_lookup=player_history,
        ...     date_column="date_of_injury"
        ... )
    """
    logger.info(f"Starting ensemble inference pipeline for {len(all_matches)} matches")

    try:
        # 1. Build basic feature table with validation
        df = build_inference_features(
            all_matches,
            player_metadata,
            validate_input=validate_input,
            strict=strict
        )
        logger.debug(f"Built basic inference features: {df.shape}")

        # 2. Apply ALL feature engineering (temporal, position, interactions)
        #    This also cleans up merge artifacts (age_x → age, etc.)
        df = apply_all_feature_engineering(
            df,
            date_column=date_column,
            position_column=position_column,
            team_column=team_column
        )
        logger.debug(f"Applied feature engineering: {df.shape}")

        # 3. Get feature columns from ensemble and validate
        feature_cols = ensemble.feature_names_
        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            logger.warning(
                f"Missing {len(missing_features)} expected features after engineering: "
                f"{missing_features[:10]}{'...' if len(missing_features) > 10 else ''}"
            )
            # Fill missing with zeros (may affect prediction quality)
            for col in missing_features:
                df[col] = 0

        # 4. Add ensemble predictions (injury RISK)
        df = add_ensemble_predictions(df, ensemble, feature_cols)
        logger.info(f"Generated ensemble risk predictions for {len(df)} records")

        # 5. Add severity predictions (injury DURATION) if classifier provided
        if severity_classifier is not None:
            df = predict_severity_class(df, severity_classifier, severity_feature_cols)
            logger.info("Added severity class predictions")
        else:
            logger.info("No severity classifier provided, skipping severity prediction")

        # 6. Add player history features from lookup (if provided)
        if player_history_lookup is not None:
            df = add_player_history_features(df, player_history_lookup)
            logger.info("Added player history features from lookup")
        else:
            logger.info("No player history lookup provided, skipping player history features")

        # 7. Add archetype from clustering
        df = add_archetype(df, archetype_df)
        logger.debug("Merged archetype data")

        logger.info("Ensemble inference pipeline completed successfully")
        return df

    except ValidationError as e:
        logger.error(f"Validation error in ensemble inference pipeline: {str(e)}")
        raise

    except Exception as e:
        logger.error(f"Error in ensemble inference pipeline: {str(e)}", exc_info=True)
        raise


# ============================================================
# 7. LEGACY INFERENCE (backwards compatibility)
# ============================================================
def build_inference_df_legacy(
    all_matches,
    player_metadata,
    severity_df,
    ensemble,
    validate_input=True,
    strict=True
):
    """
    DEPRECATED: Legacy inference pipeline that uses ground truth severity.

    Use build_inference_df_with_ensemble() with severity_classifier for proper inference.

    This function is kept for backwards compatibility with existing notebooks.
    """
    warnings.warn(
        "build_inference_df_legacy() uses ground truth severity which is data leakage. "
        "Use build_inference_df_with_ensemble() with severity_classifier parameter.",
        DeprecationWarning
    )

    logger.info(f"Starting LEGACY inference pipeline for {len(all_matches)} matches")

    try:
        df = build_inference_features(
            all_matches,
            player_metadata,
            validate_input=validate_input,
            strict=strict
        )

        feature_cols = ensemble.feature_names_
        df = add_ensemble_predictions(df, ensemble, feature_cols)

        # DEPRECATED: merges ground truth severity
        df = add_severity_and_archetype(df, severity_df)

        logger.info("Legacy inference pipeline completed")
        return df

    except Exception as e:
        logger.error(f"Error in legacy inference pipeline: {str(e)}", exc_info=True)
        raise