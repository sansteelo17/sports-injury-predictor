"""Model training and evaluation modules."""

from .classification import (
    get_classification_splits,
    get_temporal_splits,
    get_stratified_temporal_splits,
    get_smote_splits,
    train_lightgbm,
    train_xgboost,
    run_full_catboost_class_tuning,
    evaluate_classifier,
    evaluate_thresholds,
    explain_model_with_shap
)

from .stacking_ensemble import (
    StackingEnsemble,
    train_stacking_ensemble,
    compare_ensemble_vs_individual
)

from .baselines import (
    prepare_training_data,
    evaluate_classifier as evaluate_baseline_classifier,
    train_dummy_classifier,
    train_logistic_regression,
    train_random_forest
)

from .severity import (
    get_severity_splits,
    get_temporal_severity_splits,
    train_lightgbm_severity,
    train_xgboost_severity,
    train_catboost_severity,
    run_full_catboost_severity_tuning,
    evaluate_severity,
    # Classification alternative
    SEVERITY_BINS,
    SEVERITY_LABELS,
    create_severity_bins,
    get_severity_classification_splits,
    train_severity_classifier,
    evaluate_severity_classifier,
    # Outlier filtering
    filter_severity_outliers,
    get_filtered_temporal_severity_splits,
    # High-level training functions
    train_severity_regressors,
    compare_severity_models,
    diagnose_severity_target
)

from .severity_ensemble import (
    build_severity_ensemble,
    evaluate_severity_ensemble
)

from .archetype import (
    # Core clustering functions
    cluster_players,
    prepare_archetype_features,
    get_recommended_clustering_features,
    run_hdbscan,
    run_kmeans,
    # Archetype naming and profiles
    assign_archetype_names,
    get_archetype_profile,
    summarize_archetypes,
    compute_cluster_profiles,
    # Visualization
    plot_pca_clusters,
    plot_archetype_radar,
    plot_cluster_distribution,
    # Definitions
    ARCHETYPE_DEFINITIONS,
)

from .classification_shap import (
    compute_ensemble_shap,
    compute_stacking_ensemble_shap,
    build_final_output_df,
    build_temporal_output_df,
    explain_player_ensemble,
    shap_waterfall_player,
)

__all__ = [
    # Classification
    "get_classification_splits",
    "get_temporal_splits",
    "get_stratified_temporal_splits",
    "get_smote_splits",
    "train_lightgbm",
    "train_xgboost",
    "run_full_catboost_class_tuning",
    "evaluate_classifier",
    "evaluate_thresholds",
    "explain_model_with_shap",
    # Stacking ensemble
    "StackingEnsemble",
    "train_stacking_ensemble",
    "compare_ensemble_vs_individual",
    # Baselines
    "prepare_training_data",
    "evaluate_baseline_classifier",
    "train_dummy_classifier",
    "train_logistic_regression",
    "train_random_forest",
    # Severity regression
    "get_severity_splits",
    "get_temporal_severity_splits",
    "train_lightgbm_severity",
    "train_xgboost_severity",
    "train_catboost_severity",
    "run_full_catboost_severity_tuning",
    "evaluate_severity",
    "build_severity_ensemble",
    "evaluate_severity_ensemble",
    # Severity classification (alternative)
    "SEVERITY_BINS",
    "SEVERITY_LABELS",
    "create_severity_bins",
    "get_severity_classification_splits",
    "train_severity_classifier",
    "evaluate_severity_classifier",
    # Outlier filtering
    "filter_severity_outliers",
    "get_filtered_temporal_severity_splits",
    # High-level training functions
    "train_severity_regressors",
    "compare_severity_models",
    "diagnose_severity_target",
    # Archetype clustering
    "cluster_players",
    "prepare_archetype_features",
    "get_recommended_clustering_features",
    "run_hdbscan",
    "run_kmeans",
    "assign_archetype_names",
    "get_archetype_profile",
    "summarize_archetypes",
    "compute_cluster_profiles",
    "plot_pca_clusters",
    "plot_archetype_radar",
    "plot_cluster_distribution",
    "ARCHETYPE_DEFINITIONS",
    # SHAP explanations
    "compute_ensemble_shap",
    "compute_stacking_ensemble_shap",
    "build_final_output_df",
    "build_temporal_output_df",
    "explain_player_ensemble",
    "shap_waterfall_player",
]
