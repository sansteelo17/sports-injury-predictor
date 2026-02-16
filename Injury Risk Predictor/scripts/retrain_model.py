#!/usr/bin/env python3
"""
Retrain the StackingEnsemble from raw data with full feature engineering.

This script rebuilds the classification model from scratch using:
1. Raw CSVs (player stats, injuries, matches)
2. Full feature engineering pipeline (workload, injury history, temporal, position)
3. StackingEnsemble (CatBoost + LightGBM + XGBoost)
4. Temporal validation splits (no data leakage)

The key improvement: trains on rich injury history features
(player_injury_count, player_avg_severity, player_worst_injury,
player_severity_std, is_injury_prone, total_days_lost) instead of
just previous_injuries and days_since_last_injury.

Usage:
    python scripts/retrain_model.py
    python scripts/retrain_model.py --dry-run    # Don't save artifacts
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env
_env_path = ROOT / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())


def main():
    parser = argparse.ArgumentParser(description="Retrain StackingEnsemble from raw data")
    parser.add_argument("--dry-run", action="store_true", help="Don't save artifacts")
    args = parser.parse_args()

    start = time.time()

    # ================================================================
    # STEP 1: Load raw data
    # ================================================================
    print("=" * 60)
    print("STEP 1: Loading raw data")
    print("=" * 60)

    from src.data_loaders.load_data import load_all
    data = load_all()

    injuries_raw = data["injuries"]
    stats_raw = data["stats"]
    matches_raw = data["matches"]
    print(f"  Injuries: {len(injuries_raw)} records")
    print(f"  Stats: {len(stats_raw)} records")
    print(f"  Matches: {len(matches_raw)} records")

    # ================================================================
    # STEP 2: Clean data
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Cleaning data")
    print("=" * 60)

    from src.preprocessing.clean_injuries import clean_injury_df
    from src.preprocessing.clean_stats import clean_stats_df as clean_stat_df
    from src.preprocessing.clean_matches import clean_match_df

    injury_df = clean_injury_df(injuries_raw)
    stat_df = clean_stat_df(stats_raw)
    match_df = clean_match_df(matches_raw)
    print(f"  Clean injuries: {len(injury_df)}")
    print(f"  Clean stats: {len(stat_df)}")
    print(f"  Clean matches: {len(match_df)}")

    # ================================================================
    # STEP 3: Merge and feature engineering
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Merging data + feature engineering")
    print("=" * 60)

    from src.preprocessing.merge_injury_stats import merge_injury_and_stats
    from src.preprocessing.merge_injury_matches import merge_injuries_with_matches
    from src.preprocessing.rename_finaldf_cols import rename_final_df_columns
    from src.feature_engineering.injury_history import add_injury_history_features
    from src.feature_engineering.match_features import build_team_match_frame, add_match_features
    from src.feature_engineering.workload import add_workload_metrics
    from src.feature_engineering.negative_sampling import generate_negative_samples

    # Merge injuries + stats
    merged_df = merge_injury_and_stats(injury_df, stat_df)
    print(f"  Merged injury+stats: {len(merged_df)}")

    # Build team match frame with workload metrics
    team_matches = build_team_match_frame(match_df)
    team_matches = add_match_features(team_matches)
    team_matches = add_workload_metrics(team_matches)
    print(f"  Team match snapshots: {len(team_matches)}")

    # Merge with match data
    final_df = merge_injuries_with_matches(merged_df, team_matches)
    final_df = rename_final_df_columns(final_df)
    final_df = add_injury_history_features(final_df)
    print(f"  Final injury dataset: {len(final_df)}")

    # Generate negative samples
    negative_samples = generate_negative_samples(
        team_matches, injury_df, sample_frac=0.3, strategy="stratified"
    )
    print(f"  Negative samples: {len(negative_samples)}")

    # ================================================================
    # STEP 4: Build classification dataset (with injury history features)
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Building classification dataset")
    print("=" * 60)

    from src.feature_engineering.classification import build_classification_dataset
    from src.feature_engineering.imputation import impute_features
    from src.feature_engineering.temporal_features import add_temporal_features, add_fixture_density_features
    from src.feature_engineering.position_features import add_position_risk_features, add_position_workload_interaction

    injury_risk_df = build_classification_dataset(final_df, negative_samples, include_date=True)
    print(f"  Classification dataset: {len(injury_risk_df)} samples")
    print(f"  Positive rate: {injury_risk_df['injury_label'].mean():.2%}")
    print(f"  Features: {list(injury_risk_df.columns)}")

    # Impute missing values
    injury_risk_df = impute_features(injury_risk_df)

    # Add temporal features
    injury_risk_df = add_temporal_features(injury_risk_df, date_column="event_date")
    injury_risk_df = add_fixture_density_features(injury_risk_df, date_column="event_date", team_column="player_team")

    # Add position features
    injury_risk_df = add_position_risk_features(injury_risk_df, position_column="position")
    injury_risk_df = add_position_workload_interaction(injury_risk_df, position_column="position")

    # Convert categorical columns
    import pandas as pd
    cat_cols = injury_risk_df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        injury_risk_df[col] = injury_risk_df[col].astype("category")

    print(f"  Final feature count: {len(injury_risk_df.columns)}")

    # ================================================================
    # STEP 5: Temporal splits
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Temporal train/val/test split")
    print("=" * 60)

    from src.models.classification import get_temporal_splits

    X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_splits(
        injury_risk_df, train_ratio=0.6, val_ratio=0.2
    )

    print(f"  Train: {len(X_train)} samples ({y_train.mean():.1%} positive)")
    print(f"  Val:   {len(X_val)} samples ({y_val.mean():.1%} positive)")
    print(f"  Test:  {len(X_test)} samples ({y_test.mean():.1%} positive)")

    # Identify categorical features for the ensemble
    cat_feature_names = X_train.select_dtypes(include=["category"]).columns.tolist()
    print(f"  Categorical features: {cat_feature_names}")

    # ================================================================
    # STEP 6: Train StackingEnsemble
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Training StackingEnsemble")
    print("=" * 60)

    from src.models.stacking_ensemble import StackingEnsemble

    ensemble = StackingEnsemble(n_folds=5, meta_learner="logistic")
    ensemble.fit(X_train, y_train, cat_features=cat_feature_names)

    # Evaluate on test set
    test_metrics = ensemble.evaluate(X_test, y_test)

    print(f"\n  Stacking Ensemble Results (Temporal Validation):")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Features:  {len(ensemble.feature_names_)}")
    print(f"  Feature names: {ensemble.feature_names_}")

    # Check injury history features are present
    injury_features = ["player_injury_count", "player_avg_severity", "player_worst_injury",
                       "player_severity_std", "is_injury_prone", "total_days_lost"]
    present = [f for f in injury_features if f in ensemble.feature_names_]
    missing = [f for f in injury_features if f not in ensemble.feature_names_]
    print(f"\n  Injury history features in model: {present}")
    if missing:
        print(f"  WARNING: Missing injury history features: {missing}")

    # ================================================================
    # STEP 7: Train severity classifier
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Training severity classifier")
    print("=" * 60)

    from src.feature_engineering.severity import (
        build_severity_dataset,
        clean_severity_dataset,
        add_player_injury_history_features,
    )
    from src.models.severity import (
        get_severity_classification_splits,
        train_severity_classifier,
        evaluate_severity_classifier,
    )

    severity_raw = build_severity_dataset(final_df, team_matches)
    severity_df = clean_severity_dataset(severity_raw)
    severity_df = add_player_injury_history_features(severity_df)

    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s = \
        get_severity_classification_splits(severity_df)

    # Drop string columns that aren't features (name, injury descriptions, etc.)
    str_cols = X_train_s.select_dtypes(include=["object", "string"]).columns.tolist()
    if str_cols:
        print(f"  Dropping non-feature string columns: {str_cols}")
        X_train_s = X_train_s.drop(columns=str_cols)
        X_val_s = X_val_s.drop(columns=str_cols)
        X_test_s = X_test_s.drop(columns=str_cols)

    severity_clf = train_severity_classifier(X_train_s, y_train_s, X_val_s, y_val_s, model_type="catboost")
    sev_results = evaluate_severity_classifier(severity_clf, X_test_s, y_test_s)

    print(f"  Severity accuracy: {sev_results['accuracy']:.1%}")
    print(f"  Adjacent accuracy: {sev_results['adjacent_accuracy']:.1%}")

    # ================================================================
    # STEP 8: Build player history lookup
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 8: Building player history lookup")
    print("=" * 60)

    from src.inference.inference_pipeline import build_player_history_lookup

    player_history = build_player_history_lookup(severity_df)
    print(f"  Player history: {len(player_history)} players")
    print(f"  Injury-prone: {player_history['is_injury_prone'].sum()}")

    # ================================================================
    # STEP 9: Build archetype clusters
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 9: Building archetype clusters")
    print("=" * 60)

    from src.feature_engineering.archetype import build_player_archetype_features

    try:
        df_clusters = build_player_archetype_features(severity_df)
        print(f"  Archetype assignments: {len(df_clusters)}")
        if "archetype" in df_clusters.columns:
            print(f"  Distribution: {df_clusters['archetype'].value_counts().to_dict()}")
    except Exception as e:
        print(f"  Archetype clustering failed: {e}")
        print("  Loading existing archetype assignments...")
        import pickle
        arch_path = ROOT / "models" / "archetype_assignments.pkl"
        if arch_path.exists():
            with open(arch_path, "rb") as f:
                df_clusters = pickle.load(f)
            print(f"  Loaded {len(df_clusters)} existing archetype assignments")
        else:
            df_clusters = pd.DataFrame()
            print("  WARNING: No archetype assignments available")

    # ================================================================
    # STEP 10: Save artifacts
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 10: Saving artifacts")
    print("=" * 60)

    if args.dry_run:
        print("  [DRY RUN] Skipping save")
    else:
        from src.utils.model_io import save_artifacts

        save_artifacts(
            ensemble=ensemble,
            severity_clf=severity_clf,
            df_clusters=df_clusters,
            player_history=player_history,
            inference_df=pd.DataFrame(),  # Will be rebuilt by refresh_predictions
        )
        print("  Artifacts saved successfully")

    # ================================================================
    # Summary
    # ================================================================
    duration = time.time() - start
    print("\n" + "=" * 60)
    print("RETRAINING COMPLETE")
    print("=" * 60)
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"  Model features: {len(ensemble.feature_names_)}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Severity accuracy: {sev_results['accuracy']:.1%}")
    print(f"  Player history: {len(player_history)} players")
    print(f"  Archetypes: {len(df_clusters)} assignments")

    if not args.dry_run:
        print("\n  Next step: Run 'python scripts/refresh_predictions.py' to generate inference_df")


if __name__ == "__main__":
    main()
