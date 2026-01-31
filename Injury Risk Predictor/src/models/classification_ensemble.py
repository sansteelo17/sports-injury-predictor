import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.models.thresholding import risk_category, find_best_threshold

def build_ensemble_df(lgbm_model, xgb_model, X_test, y_test, 
                      w_lgbm=0.6, w_xgb=0.4):

    # Model probabilities
    lgbm_prob = lgbm_model.predict_proba(X_test)[:, 1]
    xgb_prob  = xgb_model.predict_proba(X_test)[:, 1]

    # Weighted ensemble
    ensemble_prob = (w_lgbm * lgbm_prob) + (w_xgb * xgb_prob)

    df = pd.DataFrame({
        "y_true": y_test.values,
        "lgbm_prob": lgbm_prob,
        "xgb_prob": xgb_prob,
        "ensemble_prob": ensemble_prob
    })

    return df

def tune_ensemble_threshold(df):
    probs = df["ensemble_prob"].values
    y_true = df["y_true"].values

    table, best = find_best_threshold(y_true, probs, metric="f1")
    return table, float(best.threshold)

def add_risk_category(df):
    df["risk_category"] = df["ensemble_prob"].apply(risk_category)
    return df

def evaluate_ensemble(df, threshold):
    probs = df["ensemble_prob"].values
    y_true = df["y_true"].values

    preds = (probs >= threshold).astype(int)

    return {
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
        "threshold": threshold
    }

def run_full_ensemble(lgbm_smote, xgb_smote, X_test, y_test):

    # 1. Build probability dataframe
    df = build_ensemble_df(lgbm_smote, xgb_smote, X_test, y_test)

    # 2. Find best ensemble threshold
    table, best_threshold = tune_ensemble_threshold(df)

    # 3. Evaluate performance
    metrics = evaluate_ensemble(df, best_threshold)

    # 4. Add risk category
    df = add_risk_category(df)

    return df, metrics, table
    