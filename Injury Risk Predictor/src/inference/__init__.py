"""Inference pipeline for injury risk prediction."""

from .inference_pipeline import (
    # Feature engineering
    apply_all_feature_engineering,
    build_inference_features,
    add_model_predictions,
    add_ensemble_predictions,
    # Severity prediction (recommended)
    predict_severity_class,
    add_archetype,
    # Player history (for inference-time features)
    build_player_history_lookup,
    add_player_history_features,
    # Legacy (deprecated - uses ground truth)
    add_severity_and_archetype,
    add_shap_values,
    build_full_inference_df,
    build_inference_df_with_ensemble,
    build_inference_df_legacy
)

from .risk_card import build_risk_card


def get_latest_snapshot(inference_df, player_name):
    """
    Get the most recent data snapshot for a player from the inference DataFrame.

    This is a convenience re-export from src.dashboard.player_dashboard.
    For the full dashboard functionality, import directly from src.dashboard.

    Args:
        inference_df: DataFrame with inference results
        player_name: Name of the player to look up

    Returns:
        pandas.Series: The most recent row for the player

    Example:
        >>> from src.inference import get_latest_snapshot
        >>> row = get_latest_snapshot(inference_df, "Mohamed Salah")
    """
    # Lazy import to avoid circular dependency
    from ..dashboard.player_dashboard import get_latest_snapshot as _get_latest_snapshot
    return _get_latest_snapshot(inference_df, player_name)


def build_player_dashboard(inference_df, player_name):
    """
    Build a comprehensive player dashboard from the inference DataFrame.

    This is a convenience re-export from src.dashboard.player_dashboard.
    For individual panel functions, import directly from src.dashboard.

    Args:
        inference_df: DataFrame with inference results
        player_name: Name of the player to look up

    Returns:
        dict: Dashboard with overview, risk, severity, archetype, and recommendation panels

    Example:
        >>> from src.inference import build_player_dashboard
        >>> dashboard = build_player_dashboard(inference_df, "Mohamed Salah")
    """
    # Lazy import to avoid circular dependency
    from ..dashboard.player_dashboard import build_player_dashboard as _build_player_dashboard
    return _build_player_dashboard(inference_df, player_name)


__all__ = [
    # Feature engineering
    "apply_all_feature_engineering",
    "build_inference_features",
    # Risk predictions
    "add_model_predictions",
    "add_ensemble_predictions",
    # Severity prediction (recommended)
    "predict_severity_class",
    "add_archetype",
    # Player history (for inference-time features)
    "build_player_history_lookup",
    "add_player_history_features",
    # Legacy (deprecated - uses ground truth)
    "add_severity_and_archetype",
    "add_shap_values",
    # Full pipelines
    "build_full_inference_df",
    "build_inference_df_with_ensemble",
    "build_inference_df_legacy",
    # Risk card
    "build_risk_card",
    # Dashboard utilities (convenience re-exports)
    "get_latest_snapshot",
    "build_player_dashboard",
]
