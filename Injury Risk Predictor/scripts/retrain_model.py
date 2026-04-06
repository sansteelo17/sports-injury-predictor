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
    # STEP 5: Walk-forward cross-validation
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Walk-forward cross-validation (5 folds, 6-month test windows)")
    print("=" * 60)

    from src.models.temporal_validation import walk_forward_validation
    from src.models.stacking_ensemble import StackingEnsemble
    import numpy as np

    # Determine categorical features from the full dataset before splitting
    cat_feature_names = injury_risk_df.drop(
        columns=["injury_label", "event_date"], errors="ignore"
    ).select_dtypes(include=["category"]).columns.tolist()
    print(f"  Categorical features: {cat_feature_names}")

    # Walk-forward CV must stay within the period covered by both classes.
    # Negative samples come from the match CSV which ends before the most recent
    # injuries — recent test windows (2025-2026) end up 100% positive and
    # ROC-AUC is undefined. Cap the CV dataset at the latest negative date.
    neg_dates = injury_risk_df[injury_risk_df["injury_label"] == 0]["event_date"]
    cv_cutoff = pd.to_datetime(neg_dates.max())
    cv_df = injury_risk_df[injury_risk_df["event_date"] <= cv_cutoff].copy()
    n_pos_cv = cv_df["injury_label"].sum()
    n_neg_cv = (cv_df["injury_label"] == 0).sum()
    print(f"  CV dataset: {len(cv_df)} samples up to {cv_cutoff.date()}"
          f" ({n_pos_cv} positive, {n_neg_cv} negative, {cv_df['injury_label'].mean():.1%} positive rate)")

    wf_splits = walk_forward_validation(
        cv_df,
        date_column="event_date",
        target_column="injury_label",
        n_splits=5,
        test_size_months=6,
        gap_months=0,
    )

    fold_metrics = []
    for fold_idx, (X_tr, X_te, y_tr, y_te) in enumerate(wf_splits):
        # event_date was not in the target-dropped X from walk_forward_validation
        # but it may still be present if it wasn't the target column — drop it
        X_tr = X_tr.drop(columns=["event_date"], errors="ignore")
        X_te = X_te.drop(columns=["event_date"], errors="ignore")

        fold_cat_cols = [c for c in cat_feature_names if c in X_tr.columns]

        print(f"\n  --- Fold {fold_idx + 1}/{len(wf_splits)} ---")
        print(f"  Train: {len(X_tr)} samples ({y_tr.mean():.1%} positive)")
        print(f"  Test:  {len(X_te)} samples ({y_te.mean():.1%} positive)")

        fold_ensemble = StackingEnsemble(n_folds=5, meta_learner="logistic")
        fold_ensemble.fit(X_tr, y_tr, cat_features=fold_cat_cols)
        m = fold_ensemble.evaluate(X_te, y_te)
        fold_metrics.append(m)

        print(f"  ROC-AUC: {m['roc_auc']:.4f}  Precision: {m['precision']:.4f}"
              f"  Recall: {m['recall']:.4f}  F1: {m['f1']:.4f}")

    print("\n  Walk-forward CV summary:")
    for metric in ("roc_auc", "precision", "recall", "f1"):
        vals = [m[metric] for m in fold_metrics]
        print(f"  {metric:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}"
              f"  (min {np.min(vals):.4f}, max {np.max(vals):.4f})")

    wf_mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    wf_std_metrics  = {k: float(np.std([m[k]  for m in fold_metrics])) for k in fold_metrics[0]}

    # ================================================================
    # STEP 6: Train final StackingEnsemble on ALL data
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Training final StackingEnsemble on all data")
    print("=" * 60)
    print("  (Walk-forward CV gave honest estimates; final model uses full dataset)")

    X_all = injury_risk_df.drop(columns=["injury_label", "event_date"], errors="ignore")
    y_all = injury_risk_df["injury_label"]

    final_cat_cols = [c for c in cat_feature_names if c in X_all.columns]

    ensemble = StackingEnsemble(n_folds=5, meta_learner="logistic")
    ensemble.fit(X_all, y_all, cat_features=final_cat_cols)

    # Use walk-forward metrics as the reported test_metrics
    test_metrics = wf_mean_metrics

    print(f"\n  Final model trained on {len(X_all)} samples")
    print(f"  Features: {len(ensemble.feature_names_)}")
    print(f"  Walk-forward CV ROC-AUC (mean): {test_metrics['roc_auc']:.4f}")
    print(f"  Walk-forward CV F1     (mean): {test_metrics['f1']:.4f}")

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
    from src.feature_engineering.severity import build_injury_features

    try:
        # Archetype builder needs body_area and injury_type columns which are
        # produced by build_injury_features (not called earlier in the pipeline)
        archetype_input = severity_df.copy()
        if "body_area" not in archetype_input.columns or "injury_type" not in archetype_input.columns:
            if "injury" in archetype_input.columns:
                archetype_input = build_injury_features(archetype_input)
        df_clusters = build_player_archetype_features(archetype_input)
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
        import json

        save_artifacts(
            ensemble=ensemble,
            severity_clf=severity_clf,
            df_clusters=df_clusters,
            player_history=player_history,
            inference_df=pd.DataFrame(),  # Will be rebuilt by refresh_predictions
        )
        print("  Artifacts saved successfully")

        # Write model card with walk-forward CV metrics
        model_card = {
            "model": "StackingEnsemble",
            "description": "Binary injury risk classifier for EPL + La Liga players over a 2-week horizon.",
            "version": "3.0",
            "leagues": ["Premier League", "La Liga"],
            "trained_on": "All available data. Walk-forward CV (5 folds, 6-month test windows) used for honest evaluation.",
            "evaluation": {
                "method": "walk_forward_cv",
                "n_folds": len(fold_metrics),
                "test_window_months": 6,
                "evaluated_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "roc_auc":        round(wf_mean_metrics["roc_auc"], 4),
                "roc_auc_std":    round(wf_std_metrics["roc_auc"], 4),
                "precision":      round(wf_mean_metrics["precision"], 4),
                "precision_std":  round(wf_std_metrics["precision"], 4),
                "recall":         round(wf_mean_metrics["recall"], 4),
                "recall_std":     round(wf_std_metrics["recall"], 4),
                "f1":             round(wf_mean_metrics["f1"], 4),
                "f1_std":         round(wf_std_metrics["f1"], 4),
                "per_fold": [
                    {k: round(v, 4) for k, v in m.items()}
                    for m in fold_metrics
                ],
                "positive_rate":  round(float(y_all.mean()), 4),
                "train_size":     int(len(X_all)),
                "leagues": ["Premier League", "La Liga"],
            },
            "severity_model": {
                "model": "CatBoost",
                "accuracy": round(sev_results["accuracy"], 4),
                "adjacent_accuracy": round(sev_results["adjacent_accuracy"], 4),
            },
            "notes": [
                "league is a categorical feature — model sees EPL vs La Liga.",
                "ROC-AUC is the most reliable metric given class imbalance.",
                "Walk-forward CV trains on past data and tests on future windows — no temporal leakage.",
                "Final model trained on all available data after CV evaluation.",
            ],
        }
        card_path = ROOT / "models" / "model_card.json"
        with open(card_path, "w") as _f:
            json.dump(model_card, _f, indent=2)
        print(f"  Model card written to {card_path}")

    # ================================================================
    # Summary
    # ================================================================
    duration = time.time() - start
    print("\n" + "=" * 60)
    print("RETRAINING COMPLETE")
    print("=" * 60)
    print(f"  Duration:           {duration:.1f}s ({duration/60:.1f} min)")
    print(f"  Model features:     {len(ensemble.feature_names_)}")
    print(f"  Walk-forward folds: {len(fold_metrics)}")
    print(f"  ROC-AUC (mean±std): {wf_mean_metrics['roc_auc']:.4f} ± {wf_std_metrics['roc_auc']:.4f}")
    print(f"  Recall   (mean±std): {wf_mean_metrics['recall']:.4f} ± {wf_std_metrics['recall']:.4f}")
    print(f"  F1       (mean±std): {wf_mean_metrics['f1']:.4f} ± {wf_std_metrics['f1']:.4f}")
    print(f"  Severity accuracy:  {sev_results['accuracy']:.1%}")
    print(f"  Player history:     {len(player_history)} players")
    print(f"  Archetypes:         {len(df_clusters)} assignments")

    if not args.dry_run:
        print("\n  Next step: Run 'python scripts/refresh_predictions.py' to generate inference_df")


if __name__ == "__main__":
    main()
