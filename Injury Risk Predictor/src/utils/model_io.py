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
import joblib
import pandas as pd

from .logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

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
            obj.to_pickle(path)
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

    if missing:
        logger.warning(f"Missing artifact files in {models_dir}: {missing}")
        return None

    # Load all artifacts
    artifacts = {}
    for name, filename in ARTIFACT_FILES.items():
        path = os.path.join(models_dir, filename)
        if filename.endswith(".pkl"):
            # DataFrames saved with to_pickle, models with joblib
            try:
                artifacts[name] = pd.read_pickle(path)
            except Exception:
                artifacts[name] = joblib.load(path)
        logger.info(f"Loaded {name} ← {path}")

    # Validate inference_df has expected columns
    idf = artifacts.get("inference_df")
    if idf is not None:
        required = ["name", "ensemble_prob"]
        missing_cols = [c for c in required if c not in idf.columns]
        if missing_cols:
            logger.error(f"inference_df missing required columns: {missing_cols}")
            return None
        logger.info(
            f"Loaded inference_df: {idf.shape[0]} rows, "
            f"{idf['name'].nunique()} unique players"
        )

    logger.info("All artifacts loaded successfully")
    return artifacts
