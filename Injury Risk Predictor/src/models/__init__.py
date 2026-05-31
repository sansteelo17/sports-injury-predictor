"""Model training and evaluation modules.

Lazy-loaded on purpose. Importing this package — which happens automatically
when joblib unpickles a model whose class lives here (e.g. StackingEnsemble) —
must NOT drag in the training/explainability stack (shap, optuna, matplotlib,
hdbscan, imblearn, and their numba/llvmlite tail). That stack is ~150MB+ of RSS
and is never used to call ``.predict()``; loading it at serving time is what
pushed the API past the 512MB instance limit.

Public symbols are still importable exactly as before (``from src.models import
train_stacking_ensemble``) — PEP 562 ``__getattr__`` imports the owning
submodule only on first access. So training keeps working unchanged, while a
serving process that only unpickles the ensemble pays for none of it.
"""

import importlib
from typing import Any

# Exported name -> submodule that defines it (attribute name == export name).
_EXPORTS = {
    # classification
    "get_classification_splits": "classification",
    "get_temporal_splits": "classification",
    "get_stratified_temporal_splits": "classification",
    "get_smote_splits": "classification",
    "train_lightgbm": "classification",
    "train_xgboost": "classification",
    "run_full_catboost_class_tuning": "classification",
    "evaluate_classifier": "classification",
    "evaluate_thresholds": "classification",
    "explain_model_with_shap": "classification",
    # stacking ensemble
    "StackingEnsemble": "stacking_ensemble",
    "train_stacking_ensemble": "stacking_ensemble",
    "compare_ensemble_vs_individual": "stacking_ensemble",
    # baselines
    "prepare_training_data": "baselines",
    "train_dummy_classifier": "baselines",
    "train_logistic_regression": "baselines",
    "train_random_forest": "baselines",
    # severity regression / classification
    "get_severity_splits": "severity",
    "get_temporal_severity_splits": "severity",
    "train_lightgbm_severity": "severity",
    "train_xgboost_severity": "severity",
    "train_catboost_severity": "severity",
    "run_full_catboost_severity_tuning": "severity",
    "evaluate_severity": "severity",
    "SEVERITY_BINS": "severity",
    "SEVERITY_LABELS": "severity",
    "create_severity_bins": "severity",
    "get_severity_classification_splits": "severity",
    "train_severity_classifier": "severity",
    "evaluate_severity_classifier": "severity",
    "filter_severity_outliers": "severity",
    "get_filtered_temporal_severity_splits": "severity",
    "train_severity_regressors": "severity",
    "compare_severity_models": "severity",
    "diagnose_severity_target": "severity",
    # severity ensemble
    "build_severity_ensemble": "severity_ensemble",
    "evaluate_severity_ensemble": "severity_ensemble",
    # archetype clustering
    "cluster_players": "archetype",
    "prepare_archetype_features": "archetype",
    "get_recommended_clustering_features": "archetype",
    "run_hdbscan": "archetype",
    "run_kmeans": "archetype",
    "assign_archetype_names": "archetype",
    "get_archetype_profile": "archetype",
    "summarize_archetypes": "archetype",
    "compute_cluster_profiles": "archetype",
    "plot_pca_clusters": "archetype",
    "plot_archetype_radar": "archetype",
    "plot_cluster_distribution": "archetype",
    "ARCHETYPE_DEFINITIONS": "archetype",
    # SHAP explanations
    "compute_ensemble_shap": "classification_shap",
    "compute_stacking_ensemble_shap": "classification_shap",
    "build_final_output_df": "classification_shap",
    "build_temporal_output_df": "classification_shap",
    "explain_player_ensemble": "classification_shap",
    "shap_waterfall_player": "classification_shap",
}

# Exported name -> (submodule, real attribute name) for renamed re-exports.
_ALIASES = {
    "evaluate_baseline_classifier": ("baselines", "evaluate_classifier"),
}

__all__ = list(_EXPORTS) + list(_ALIASES)


def __getattr__(name: str) -> Any:
    if name in _ALIASES:
        mod_name, attr = _ALIASES[name]
        module = importlib.import_module(f".{mod_name}", __name__)
        return getattr(module, attr)
    if name in _EXPORTS:
        module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
