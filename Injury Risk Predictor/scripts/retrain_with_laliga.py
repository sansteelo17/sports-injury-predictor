#!/usr/bin/env python3
"""
Retrain the StackingEnsemble with both EPL and La Liga data.

Extends retrain_model.py to:
1. Load combined EPL + La Liga match data
2. Merge La Liga player injury history from Transfermarkt scrape
3. Train with `league` as a categorical feature

Prerequisites:
    python scripts/fetch_laliga_matches.py      # fetch La Liga match CSVs
    python scripts/scrape_laliga_injuries.py    # scrape La Liga player injuries

Usage:
    python scripts/retrain_with_laliga.py
    python scripts/retrain_with_laliga.py --dry-run
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_env = ROOT / ".env"
if _env.exists():
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    parser = argparse.ArgumentParser(description="Retrain with EPL + La Liga data")
    parser.add_argument("--dry-run", action="store_true", help="Don't save artifacts")
    args = parser.parse_args()

    start = time.time()

    # ================================================================
    # STEP 1: Load raw data (EPL + La Liga)
    # ================================================================
    print("=" * 60)
    print("STEP 1: Loading raw data (EPL + La Liga)")
    print("=" * 60)

    from src.data_loaders.load_data import load_all_with_laliga
    data = load_all_with_laliga()

    injuries_raw = data["injuries"]
    stats_raw = data["stats"]
    matches_raw = data["matches"]

    leagues_in_matches = matches_raw["league"].value_counts().to_dict() if "league" in matches_raw.columns else {}
    print(f"  Injuries: {len(injuries_raw)} records")
    print(f"  Stats: {len(stats_raw)} records")
    print(f"  Matches: {len(matches_raw)} records — {leagues_in_matches}")

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
    if "league" in match_df.columns:
        print(f"  League distribution: {match_df['league'].value_counts().to_dict()}")

    # ================================================================
    # STEP 3: Feature engineering
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Merging + feature engineering")
    print("=" * 60)

    from src.preprocessing.merge_injury_stats import merge_injury_and_stats
    from src.preprocessing.merge_injury_matches import merge_injuries_with_matches
    from src.preprocessing.rename_finaldf_cols import rename_final_df_columns
    from src.feature_engineering.injury_history import add_injury_history_features
    from src.feature_engineering.match_features import build_team_match_frame, add_match_features
    from src.feature_engineering.workload import add_workload_metrics
    from src.feature_engineering.negative_sampling import generate_negative_samples

    merged_df = merge_injury_and_stats(injury_df, stat_df)
    team_matches = build_team_match_frame(match_df)
    team_matches = add_match_features(team_matches)
    team_matches = add_workload_metrics(team_matches)
    final_df = merge_injuries_with_matches(merged_df, team_matches)
    final_df = rename_final_df_columns(final_df)
    final_df = add_injury_history_features(final_df)
    print(f"  Final injury dataset: {len(final_df)}")

    negative_samples = generate_negative_samples(
        team_matches, injury_df, sample_frac=0.3, strategy="stratified"
    )
    print(f"  Negative samples: {len(negative_samples)}")

    # ================================================================
    # STEP 4: Merge La Liga player injury history
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Merging La Liga player injury history")
    print("=" * 60)

    import pandas as pd
    laliga_history_path = ROOT / "models" / "laliga_player_history.pkl"
    epl_history_path = ROOT / "models" / "player_history.pkl"

    if laliga_history_path.exists():
        laliga_hist = pd.read_pickle(laliga_history_path)
        epl_hist = pd.read_pickle(epl_history_path) if epl_history_path.exists() else pd.DataFrame()
        combined_hist = pd.concat([epl_hist, laliga_hist], ignore_index=True).drop_duplicates(subset=["name"], keep="last")
        print(f"  EPL history: {len(epl_hist)} players")
        print(f"  La Liga history: {len(laliga_hist)} players")
        print(f"  Combined (deduped): {len(combined_hist)} players")
    else:
        print("  No La Liga history found — run scrape_laliga_injuries.py first")
        print("  Continuing with EPL history only...")
        combined_hist = pd.read_pickle(epl_history_path) if epl_history_path.exists() else pd.DataFrame()

    # Patch any La Liga player injury history into final_df
    if not combined_hist.empty and len(combined_hist) > 0:
        hist_cols = [c for c in [
            "player_injury_count", "player_avg_severity", "player_worst_injury",
            "player_severity_std", "is_injury_prone", "total_days_lost",
            "days_since_last_injury"
        ] if c in combined_hist.columns]

        if hist_cols:
            hist_lookup = combined_hist.set_index("name")[hist_cols]
            for col in hist_cols:
                if col in final_df.columns:
                    mask = final_df["name"].isin(hist_lookup.index)
                    final_df.loc[mask, col] = final_df.loc[mask, "name"].map(hist_lookup[col])

    # ================================================================
    # STEP 5: Classification dataset
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Building classification dataset")
    print("=" * 60)

    from src.feature_engineering.classification import build_classification_dataset
    from src.feature_engineering.imputation import impute_features
    from src.feature_engineering.temporal_features import add_temporal_features, add_fixture_density_features
    from src.feature_engineering.position_features import add_position_risk_features, add_position_workload_interaction

    injury_risk_df = build_classification_dataset(final_df, negative_samples, include_date=True)
    injury_risk_df = impute_features(injury_risk_df)
    injury_risk_df = add_temporal_features(injury_risk_df, date_column="event_date")
    injury_risk_df = add_fixture_density_features(injury_risk_df, date_column="event_date", team_column="player_team")
    injury_risk_df = add_position_risk_features(injury_risk_df, position_column="position")
    injury_risk_df = add_position_workload_interaction(injury_risk_df, position_column="position")

    cat_cols = injury_risk_df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        injury_risk_df[col] = injury_risk_df[col].astype("category")

    if "league" in injury_risk_df.columns:
        print(f"  League distribution in dataset: {injury_risk_df['league'].value_counts().to_dict()}")
    print(f"  Final feature count: {len(injury_risk_df.columns)}")
    print(f"  Dataset size: {len(injury_risk_df)} rows")

    # ================================================================
    # STEP 6: Temporal splits
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Temporal train/val/test split")
    print("=" * 60)

    from src.models.classification import get_temporal_splits

    X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_splits(
        injury_risk_df, train_ratio=0.6, val_ratio=0.2
    )
    print(f"  Train: {len(X_train)} ({y_train.mean():.1%} positive)")
    print(f"  Val:   {len(X_val)} ({y_val.mean():.1%} positive)")
    print(f"  Test:  {len(X_test)} ({y_test.mean():.1%} positive)")

    cat_feature_names = X_train.select_dtypes(include=["category"]).columns.tolist()
    print(f"  Categorical features: {cat_feature_names}")

    # ================================================================
    # STEP 7: Train StackingEnsemble
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Training StackingEnsemble (EPL + La Liga)")
    print("=" * 60)

    from src.models.stacking_ensemble import StackingEnsemble
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    import numpy as np

    ensemble = StackingEnsemble(n_folds=5, meta_learner="logistic")
    ensemble.fit(X_train, y_train, cat_features=cat_feature_names)

    proba = ensemble.predict_proba(X_test)[:, 1]
    threshold = getattr(ensemble, "threshold_", 0.5)
    preds = (proba >= threshold).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "avg_precision": round(average_precision_score(y_test, proba), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        "threshold": round(float(threshold), 4),
        "test_size": int(len(X_test)),
        "positive_rate": round(float(y_test.mean()), 4),
        "leagues": list(leagues_in_matches.keys()),
    }

    print(f"\n  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")

    # ================================================================
    # STEP 8: Train severity classifier
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 8: Training severity classifier")
    print("=" * 60)

    from src.feature_engineering.severity import (
        build_severity_dataset, clean_severity_dataset, add_player_injury_history_features,
    )
    from src.models.severity import (
        get_severity_classification_splits, train_severity_classifier, evaluate_severity_classifier,
    )

    severity_raw = build_severity_dataset(final_df, team_matches)
    severity_df = clean_severity_dataset(severity_raw)
    severity_df = add_player_injury_history_features(severity_df)

    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s = \
        get_severity_classification_splits(severity_df)

    str_cols = X_train_s.select_dtypes(include=["object", "string"]).columns.tolist()
    if str_cols:
        X_train_s = X_train_s.drop(columns=str_cols)
        X_val_s = X_val_s.drop(columns=str_cols)
        X_test_s = X_test_s.drop(columns=str_cols)

    severity_clf = train_severity_classifier(X_train_s, y_train_s, X_val_s, y_val_s, model_type="catboost")
    sev_results = evaluate_severity_classifier(severity_clf, X_test_s, y_test_s)
    print(f"  Severity accuracy: {sev_results['accuracy']:.1%}")
    print(f"  Adjacent accuracy: {sev_results['adjacent_accuracy']:.1%}")

    # ================================================================
    # STEP 9: Player history lookup
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 9: Building player history lookup")
    print("=" * 60)

    from src.inference.inference_pipeline import build_player_history_lookup
    player_history = build_player_history_lookup(severity_df)
    print(f"  Player history: {len(player_history)} players")

    # ================================================================
    # STEP 10: Archetypes
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 10: Building archetype clusters")
    print("=" * 60)

    from src.feature_engineering.archetype import build_player_archetype_features
    import pickle

    try:
        df_clusters = build_player_archetype_features(severity_df)
        print(f"  Archetype assignments: {len(df_clusters)}")
    except Exception as e:
        print(f"  Clustering failed: {e} — loading existing")
        arch_path = ROOT / "models" / "archetype_assignments.pkl"
        df_clusters = pd.read_pickle(arch_path) if arch_path.exists() else pd.DataFrame()

    # ================================================================
    # STEP 11: Save
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 11: Saving artifacts")
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
            inference_df=pd.DataFrame(),
        )

        # Write model card
        card = {
            "model": "StackingEnsemble",
            "description": "Binary injury risk classifier for EPL + La Liga players over a 2-week horizon.",
            "version": "2.0",
            "leagues": list(leagues_in_matches.keys()),
            "trained_on": "Historical player stats, injury records, and match data. Temporal train/val/test split (60/20/20).",
            "evaluation": {
                "evaluated_at": datetime.now().strftime("%Y-%m-%d"),
                **metrics,
            },
            "severity_model": {
                "model": "CatBoost",
                "accuracy": round(sev_results["accuracy"], 4),
                "adjacent_accuracy": round(sev_results["adjacent_accuracy"], 4),
            },
            "notes": [
                "league is a categorical feature — model sees EPL vs La Liga.",
                "ROC-AUC is the most reliable metric given class imbalance.",
            ]
        }
        with open(ROOT / "models" / "model_card.json", "w") as f:
            json.dump(card, f, indent=2)

        print(f"  Artifacts saved. Model card updated.")

    duration = time.time() - start
    print(f"\nDone in {duration/60:.1f} minutes.")


if __name__ == "__main__":
    main()
