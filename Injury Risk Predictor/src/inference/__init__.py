"""Inference pipeline for injury risk prediction."""

# All inference_pipeline symbols are lazy-loaded on first access.
# This keeps the API startup footprint small — importing story_generator or
# context_rag no longer pulls in shap, sklearn, hdbscan, numba, optuna,
# matplotlib, and the full training stack (~150-200MB).
# Training scripts that need these symbols import inference_pipeline directly.

_PIPELINE_IMPORT_ERROR = None
_RISK_CARD_IMPORT_ERROR = None

_PIPELINE_SYMBOLS = {
    "apply_all_feature_engineering",
    "build_inference_features",
    "add_model_predictions",
    "add_ensemble_predictions",
    "predict_severity_class",
    "add_archetype",
    "build_player_history_lookup",
    "add_player_history_features",
    "add_severity_and_archetype",
    "add_shap_values",
    "build_full_inference_df",
    "build_inference_df_with_ensemble",
    "build_inference_df_legacy",
}

_pipeline_cache: dict = {}


def __getattr__(name: str):
    if name in _PIPELINE_SYMBOLS:
        if name not in _pipeline_cache:
            try:
                import importlib
                mod = importlib.import_module(".inference_pipeline", package=__name__)
                for sym in _PIPELINE_SYMBOLS:
                    if hasattr(mod, sym):
                        _pipeline_cache[sym] = getattr(mod, sym)
            except Exception as e:
                raise ModuleNotFoundError(
                    f"Failed to import '{name}' from src.inference: {e}. "
                    "Install training extras (shap, imbalanced-learn, etc.) to use the full pipeline."
                ) from e
        if name in _pipeline_cache:
            return _pipeline_cache[name]
        raise AttributeError(name)

    if name == "build_risk_card":
        try:
            from .risk_card import build_risk_card
            return build_risk_card
        except Exception as e:
            raise ModuleNotFoundError(
                f"Failed to import 'build_risk_card': {e}."
            ) from e

    raise AttributeError(name)


def get_latest_snapshot(inference_df, player_name):
    from ..dashboard.player_dashboard import get_latest_snapshot as _fn
    return _fn(inference_df, player_name)


def build_player_dashboard(inference_df, player_name):
    from ..dashboard.player_dashboard import build_player_dashboard as _fn
    return _fn(inference_df, player_name)


__all__ = [
    "apply_all_feature_engineering",
    "build_inference_features",
    "add_model_predictions",
    "add_ensemble_predictions",
    "predict_severity_class",
    "add_archetype",
    "build_player_history_lookup",
    "add_player_history_features",
    "add_severity_and_archetype",
    "add_shap_values",
    "build_full_inference_df",
    "build_inference_df_with_ensemble",
    "build_inference_df_legacy",
    "build_risk_card",
    "get_latest_snapshot",
    "build_player_dashboard",
]
