import numpy as np
import pandas as pd
import shap

def get_true_feature_names(preprocessor):
    """
    Extract true feature names after preprocessing
    (StandardScaler + OneHotEncoder).
    """
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]

    ohe = preprocessor.named_transformers_["cat"]

    try:
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    except:
        # Fallback: use original names if OHE isn't fitted yet
        ohe_names = cat_cols

    return num_cols + ohe_names

def compute_severity_shap(model, X_train, X_test, preprocessor):

    # Transform data
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans  = preprocessor.transform(X_test)

    if not isinstance(X_train_trans, np.ndarray):
        X_train_trans = X_train_trans.toarray()
    if not isinstance(X_test_trans, np.ndarray):
        X_test_trans = X_test_trans.toarray()

    # Extract underlying regressor
    reg = model.named_steps["reg"]

    # Feature names
    feature_names = get_true_feature_names(preprocessor)

    # SHAP explainer
    explainer = shap.TreeExplainer(reg)
    shap_train = explainer.shap_values(X_train_trans)
    shap_test  = explainer.shap_values(X_test_trans)

    expected_value = explainer.expected_value

    return shap_train, shap_test, expected_value, X_test_trans, feature_names


def shap_waterfall_severity(shap_test, expected_value, X_test_trans, feature_names, index):
    """Waterfall plot for a single player severity prediction."""
    shap.plots._waterfall.waterfall_legacy(
        shap_values=shap_test[index],
        expected_value=expected_value,
        features=X_test_trans[index],
        feature_names=feature_names
    )