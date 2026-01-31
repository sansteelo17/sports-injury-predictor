import numpy as np
import pandas as pd
import shap
from .thresholding import risk_category

from sklearn.utils.validation import check_is_fitted

# -----------------------------------------------------
# Helper: Collapse all OHE SHAP values back to original feature
# -----------------------------------------------------
def collapse_ohe_shap(shap_matrix, feature_names):
    """
    Collapses one-hot encoded SHAP values back into raw feature SHAP values.
    E.g., position_GK, position_MID → position
    """
    raw_names = [name.split("_")[0] for name in feature_names]

    unique_raw = list(dict.fromkeys(raw_names))  # preserves order
    collapsed = np.zeros((shap_matrix.shape[0], len(unique_raw)))

    for i, raw in enumerate(unique_raw):
        mask = np.array([name.startswith(raw) for name in feature_names])
        collapsed[:, i] = shap_matrix[:, mask].sum(axis=1)

    return collapsed, unique_raw

def compute_ensemble_shap(
    lgbm_model,
    xgb_model,
    X_train,
    X_test,
    preprocessor,
    w_lgbm=0.6,
    w_xgb=0.4
):
    # ------------------------
    # 1. Transform inputs ONCE
    # ------------------------
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans  = preprocessor.transform(X_test)

    # Convert sparse → dense
    if not isinstance(X_train_trans, np.ndarray):
        X_train_trans = X_train_trans.toarray()
    if not isinstance(X_test_trans, np.ndarray):
        X_test_trans = X_test_trans.toarray()

    # ------------------------
    # 2. Get raw classifier objects
    # ------------------------
    lgb_clf = lgbm_model
    xgb_clf = xgb_model

    # ------------------------
    # 3. Get TRUE feature names from models
    # ------------------------
    try:
        feature_names = lgb_clf.feature_name_          # LightGBM feature names
    except:
        feature_names = [f"f{i}" for i in range(X_train_trans.shape[1])]

    # ------------------------
    # 4. Compute SHAP values
    # ------------------------
    shap_lgb = shap.TreeExplainer(lgb_clf)
    shap_xgb = shap.TreeExplainer(xgb_clf)

    shap_train_lgb = shap_lgb.shap_values(X_train_trans)
    shap_test_lgb  = shap_lgb.shap_values(X_test_trans)

    shap_train_xgb = shap_xgb.shap_values(X_train_trans)
    shap_test_xgb  = shap_xgb.shap_values(X_test_trans)

    # ------------------------
    # 5. Weighted ensemble SHAP
    # ------------------------
    shap_train_ens = w_lgbm * shap_train_lgb + w_xgb * shap_train_xgb
    shap_test_ens  = w_lgbm * shap_test_lgb + w_xgb * shap_test_xgb

    return shap_train_ens, shap_test_ens, X_test_trans, feature_names


def explain_player_ensemble(shap_test, X_test, feature_names,
                            ensemble_probs, player_index):
    """
    Explain a single player's risk prediction using SHAP values.

    Args:
        shap_test: SHAP values array for test set
        X_test: Test features (DataFrame or numpy array)
        feature_names: List of feature names
        ensemble_probs: Array of ensemble probabilities
        player_index: Index of player to explain

    Returns:
        DataFrame with top 10 contributing features
    """
    shap_row = shap_test[player_index]

    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_row,
        "abs_shap": np.abs(shap_row)
    }).sort_values("abs_shap", ascending=False)

    prob = float(ensemble_probs[player_index])
    category = risk_category(prob)

    # Get feature values for this player
    if isinstance(X_test, pd.DataFrame):
        player_features = X_test[feature_names].iloc[player_index]
    else:
        player_features = X_test[player_index]

    print(f"\n=== Player {player_index} Explanation ===")
    print(f"Ensemble Probability: {prob:.3f}")
    print(f"Risk Category: {category}")

    # Show top features with their values
    top_features = contributions.head(10).copy()
    if isinstance(X_test, pd.DataFrame):
        top_features["value"] = [player_features[f] for f in top_features["feature"]]

    return top_features


def build_final_output_df(X_test, y_test, ensemble_probs, shap_test, feature_names):
    """
    Builds a final deliverable dataframe for the project.
    """

    # Convert SHAP vectors to Python lists
    shap_list = [shap_test[i].tolist() for i in range(len(shap_test))]

    df = pd.DataFrame({
        "y_true": y_test.values,
        "ensemble_prob": ensemble_probs,
        "risk_category": [risk_category(p) for p in ensemble_probs],
        "shap_sum": shap_test.sum(axis=1),
        "shap_values": shap_list
    })

    # attach raw feature rows (index reset ensures alignment)
    X_clean = X_test.reset_index(drop=True)
    df = pd.concat([X_clean, df], axis=1)

    return df

def shap_waterfall_player(shap_test, X_test, feature_names, player_index):
    """
    Create a SHAP waterfall plot for a single player.

    Args:
        shap_test: SHAP values array for test set
        X_test: Test features (DataFrame or numpy array)
        feature_names: List of feature names
        player_index: Index of player to visualize
    """
    shap_row = shap_test[player_index]

    # Get feature values for this player
    if isinstance(X_test, pd.DataFrame):
        features = X_test[feature_names].iloc[player_index].values
    else:
        features = X_test[player_index]

    # SHAP waterfall requires expected value → use 0 for collapsed SHAP
    expected_value = 0

    shap.plots._waterfall.waterfall_legacy(
        shap_values=shap_row,
        expected_value=expected_value,
        features=features,
        feature_names=feature_names
    )
    
def get_feature_names_from_preprocessor(preprocessor):
    names = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            try:
                ohe = transformer
                names.extend(ohe.get_feature_names_out(cols))
            except:
                names.extend(cols)

    return names


def compute_stacking_ensemble_shap(ensemble, X_train, X_test, use_catboost=True):
    """
    Compute SHAP values for a StackingEnsemble model.

    Uses the CatBoost model from the ensemble by default (best SHAP support).
    For each fold, we average SHAP values to get stable explanations.

    Args:
        ensemble: Trained StackingEnsemble instance
        X_train: Training features (pandas DataFrame)
        X_test: Test features (pandas DataFrame)
        use_catboost: If True, use CatBoost model (index 2). Otherwise use LightGBM (index 0).

    Returns:
        shap_train: SHAP values for training data
        shap_test: SHAP values for test data
        feature_names: List of feature names
    """
    # Get model index (catboost is typically index 2)
    if use_catboost and "catboost" in ensemble.base_learner_names:
        model_idx = ensemble.base_learner_names.index("catboost")
        model_name = "CatBoost"
    else:
        model_idx = 0  # LightGBM
        model_name = "LightGBM"

    print(f"Computing SHAP values using {model_name} from ensemble...")

    # Get all fold models for this learner
    fold_models = ensemble.base_models_[model_idx]

    # Feature names
    feature_names = ensemble.feature_names_

    # Prepare data based on model type
    if use_catboost and "catboost" in ensemble.base_learner_names:
        # CatBoost can use original DataFrame directly
        cat_features = ensemble.cat_features_ if hasattr(ensemble, 'cat_features_') else []

        # Prepare X for CatBoost (fill NaN, convert types)
        def prep_for_catboost(X):
            X = X[feature_names].copy()
            # Convert categoricals to string (CatBoost prefers this)
            # Must convert to str BEFORE fillna to handle pandas Categorical dtype
            for col in cat_features:
                if col in X.columns:
                    X[col] = X[col].astype(str).replace('nan', 'missing').replace('None', 'missing')
            X = X.fillna(0)
            return X

        X_train_prep = prep_for_catboost(X_train)
        X_test_prep = prep_for_catboost(X_test)

        # Use first fold model for SHAP (representative)
        model = fold_models[0]

        explainer = shap.TreeExplainer(model)
        shap_train = explainer.shap_values(X_train_prep)
        shap_test = explainer.shap_values(X_test_prep)

    else:
        # LightGBM: need to encode categoricals
        X_train_prep = ensemble._prepare_data_for_lgb_xgb(
            X_train[feature_names],
            ensemble.cat_features_ if hasattr(ensemble, 'cat_features_') else []
        )
        X_test_prep = ensemble._prepare_data_for_lgb_xgb(
            X_test[feature_names],
            ensemble.cat_features_ if hasattr(ensemble, 'cat_features_') else []
        )

        model = fold_models[0]
        explainer = shap.TreeExplainer(model)
        shap_train = explainer.shap_values(X_train_prep)
        shap_test = explainer.shap_values(X_test_prep)

        # For LightGBM binary classification, shap_values returns [class_0, class_1]
        if isinstance(shap_train, list):
            shap_train = shap_train[1]
            shap_test = shap_test[1]

    print(f"SHAP values computed: train={shap_train.shape}, test={shap_test.shape}")

    return shap_train, shap_test, feature_names


def build_temporal_output_df(X_test, y_test, ensemble, shap_test, feature_names):
    """
    Builds final output dataframe using StackingEnsemble predictions.

    This is the recommended function for temporal validation results.

    Args:
        X_test: Test features (pandas DataFrame)
        y_test: Test labels
        ensemble: Trained StackingEnsemble
        shap_test: SHAP values for test set
        feature_names: Feature names for SHAP

    Returns:
        DataFrame with predictions, risk categories, and SHAP values
    """
    # Get ensemble predictions
    ensemble_probs = ensemble.predict_proba(X_test)[:, 1]

    # Get individual model predictions
    base_preds = ensemble.get_base_predictions(X_test)

    # Convert SHAP vectors to Python lists
    shap_list = [shap_test[i].tolist() for i in range(len(shap_test))]

    df = pd.DataFrame({
        "y_true": y_test.values if hasattr(y_test, 'values') else y_test,
        "ensemble_prob": ensemble_probs,
        "risk_category": [risk_category(p) for p in ensemble_probs],
        "shap_sum": shap_test.sum(axis=1),
        "shap_values": shap_list
    })

    # Add individual model probabilities
    for col in base_preds.columns:
        df[col] = base_preds[col].values

    # Attach raw feature rows
    X_clean = X_test[feature_names].reset_index(drop=True)
    df = pd.concat([X_clean, df], axis=1)

    return df