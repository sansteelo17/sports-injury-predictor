import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from .thresholding import find_best_threshold, ThresholdClassifier

def smote_compare(
    model_no_preprocessor,       # trainer for normal (pipeline) model
    model_smote_trainer,         # trainer for numeric SMOTE model
    X_train, X_test,             # raw training & test data (DataFrames)
    X_train_res, X_test_proc,    # SMOTE-resampled numeric data
    y_train, y_train_res, y_test,
    preprocessor
):
    # ------------------------------------------------------------
    # 1. NORMAL MODEL (NO SMOTE)
    # ------------------------------------------------------------
    model_no = model_no_preprocessor(X_train, y_train, preprocessor)
    probs_no = model_no.predict_proba(X_test)[:, 1]

    table_no, best_no = find_best_threshold(y_test, probs_no, metric="f1")
    thresh_no = float(best_no.threshold)

    clf_no = ThresholdClassifier(model_no, thresh_no)
    preds_no = clf_no.predict(X_test)

    metrics_no = {
        "model": "no_smote",
        "threshold": thresh_no,
        "precision": precision_score(y_test, preds_no, zero_division=0),
        "recall": recall_score(y_test, preds_no, zero_division=0),
        "f1": f1_score(y_test, preds_no, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs_no),
    }

    # ------------------------------------------------------------
    # 2. SMOTE MODEL (NUMERIC INPUT ONLY)
    # ------------------------------------------------------------
    model_sm = model_smote_trainer(X_train_res, y_train_res)
    probs_sm = model_sm.predict_proba(X_test_proc)[:, 1]

    table_sm, best_sm = find_best_threshold(y_test, probs_sm, metric="f1")
    thresh_sm = float(best_sm.threshold)

    clf_sm = ThresholdClassifier(model_sm, thresh_sm)
    preds_sm = clf_sm.predict(X_test_proc)

    metrics_sm = {
        "model": "smote",
        "threshold": thresh_sm,
        "precision": precision_score(y_test, preds_sm, zero_division=0),
        "recall": recall_score(y_test, preds_sm, zero_division=0),
        "f1": f1_score(y_test, preds_sm, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs_sm),
    }

    return pd.DataFrame([metrics_no, metrics_sm]), table_no, table_sm