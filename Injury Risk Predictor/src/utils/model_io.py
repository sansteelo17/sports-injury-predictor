"""
Model persistence utilities for saving and loading trained artifacts.

Usage (in notebook after training):
    from src.utils.model_io import save_artifacts
    save_artifacts(
        ensemble=ensemble,
        severity_clf=severity_clf,
        df_clusters=df_clusters,
        player_history=player_history,
        inference_df=inference_df,
    )

Usage (in app.py):
    from src.utils.model_io import load_artifacts
    artifacts = load_artifacts()  # Returns dict or None
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd

from .logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def _load_pickle_compat(path):
    """Load a pickle file with compatibility fixes for StringDtype.

    When a DataFrame is pickled with newer pandas (StringDtype columns),
    older pandas versions fail to reconstruct the ExtensionArray.
    This patches the unpickler to convert StringDtype → object on the fly.
    """
    import io

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Intercept pandas StringDtype/StringArray reconstruction
            if "StringDtype" in name or "StringArray" in name:
                # Return a factory that just returns the raw values as object array
                return _string_array_compat
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


def _string_array_compat(*args, **kwargs):
    """Replacement constructor for StringArray that returns a plain numpy object array."""
    # StringArray pickle passes (dtype, ndarray) — just return the ndarray
    for arg in args:
        if isinstance(arg, np.ndarray):
            return arg
    # If called differently, return the first arg
    return args[0] if args else None


def _strip_stringdtype(df):
    """Convert any StringDtype/extension columns to plain numpy for pickle compatibility.

    Pandas 3.0 uses StringDtype by default for string columns AND for the
    column Index itself. Both must be converted to plain object dtype to
    ensure pickles can be loaded by pandas 2.x.
    """
    # Temporarily disable pandas 3.0 string inference so pd.DataFrame()
    # doesn't re-infer StringDtype from object arrays.
    old_setting = pd.options.future.infer_string
    try:
        pd.options.future.infer_string = False
        data = {}
        for col in df.columns:
            series = df[col]
            dtype_str = str(series.dtype)
            if dtype_str in ("str", "string", "String", "string[python]") or "String" in dtype_str:
                data[str(col)] = series.astype(object).values
            else:
                data[str(col)] = series.values
        # Rebuild with plain object-dtype column Index (no StringDtype re-inference)
        out = pd.DataFrame(data, index=df.index)
    finally:
        pd.options.future.infer_string = old_setting
    return out

# File names for each artifact
ARTIFACT_FILES = {
    "ensemble": "stacking_ensemble.pkl",
    "severity_clf": "severity_classifier.pkl",
    "df_clusters": "archetype_assignments.pkl",
    "player_history": "player_history.pkl",
    "inference_df": "inference_df.pkl",
}


def save_artifacts(
    ensemble,
    severity_clf,
    df_clusters,
    player_history,
    inference_df,
    models_dir=None,
):
    """
    Save all trained model artifacts to disk.

    Args:
        ensemble: Trained StackingEnsemble instance
        severity_clf: Trained severity classifier (CatBoost)
        df_clusters: DataFrame with player archetype assignments
        player_history: DataFrame from build_player_history_lookup()
        inference_df: Pre-computed inference DataFrame with all predictions
        models_dir: Directory to save to (default: project_root/models/)

    Returns:
        str: Path to the models directory
    """
    if models_dir is None:
        models_dir = os.path.abspath(DEFAULT_MODELS_DIR)

    os.makedirs(models_dir, exist_ok=True)

    artifacts = {
        "ensemble": ensemble,
        "severity_clf": severity_clf,
        "df_clusters": df_clusters,
        "player_history": player_history,
        "inference_df": inference_df,
    }

    for name, obj in artifacts.items():
        path = os.path.join(models_dir, ARTIFACT_FILES[name])
        if isinstance(obj, pd.DataFrame):
            # Strip StringDtype to ensure cross-pandas-version compatibility
            obj = _strip_stringdtype(obj)
            with open(path, "wb") as f:
                pickle.dump(obj, f, protocol=4)
        else:
            joblib.dump(obj, path)
        logger.info(f"Saved {name} → {path}")

    logger.info(f"All artifacts saved to {models_dir}")
    print(f"Saved {len(artifacts)} artifacts to {models_dir}/")
    for name in artifacts:
        print(f"  {ARTIFACT_FILES[name]}")

    return models_dir


def load_artifacts(models_dir=None):
    """
    Load all trained model artifacts from disk.

    Args:
        models_dir: Directory to load from (default: project_root/models/)

    Returns:
        dict with keys: ensemble, severity_clf, df_clusters, player_history, inference_df
        Returns None if models directory doesn't exist or any required file is missing.
    """
    if models_dir is None:
        models_dir = os.path.abspath(DEFAULT_MODELS_DIR)

    if not os.path.exists(models_dir):
        logger.info(f"Models directory not found: {models_dir}")
        return None

    # Check all required files exist
    missing = []
    for name, filename in ARTIFACT_FILES.items():
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            missing.append(filename)

    # Require at least the ensemble model; inference_df can be regenerated
    ensemble_file = ARTIFACT_FILES.get("ensemble", "stacking_ensemble.pkl")
    if ensemble_file in missing:
        logger.warning(f"Ensemble model not found in {models_dir}")
        return None
    if missing:
        logger.warning(f"Some artifact files missing (non-critical): {missing}")

    # Load all artifacts (skip failures for non-essential ones)
    artifacts = {}
    for name, filename in ARTIFACT_FILES.items():
        path = os.path.join(models_dir, filename)
        try:
            if filename.endswith(".pkl"):
                try:
                    artifacts[name] = pd.read_pickle(path)
                except Exception:
                    try:
                        artifacts[name] = joblib.load(path)
                    except Exception:
                        # Last resort: raw pickle load with StringDtype workaround.
                        # Newer pandas saves StringDtype columns as ExtensionArrays
                        # which fail to unpickle on older pandas. Patch it.
                        artifacts[name] = _load_pickle_compat(path)
            logger.info(f"Loaded {name} ← {path}")
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")
            if name == "inference_df":
                logger.warning("inference_df will be regenerated on next refresh")

    # Validate inference_df has expected columns
    idf = artifacts.get("inference_df")
    if idf is not None:
        required = ["name", "ensemble_prob"]
        missing_cols = [c for c in required if c not in idf.columns]
        if missing_cols:
            logger.warning(f"inference_df missing required columns: {missing_cols}")
            logger.warning("inference_df will be regenerated on next refresh")
            artifacts["inference_df"] = None
        else:
            logger.info(
                f"Loaded inference_df: {idf.shape[0]} rows, "
                f"{idf['name'].nunique()} unique players"
            )

    logger.info("All artifacts loaded successfully")
    return artifacts
