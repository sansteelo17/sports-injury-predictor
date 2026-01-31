import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def find_best_threshold(y_true, y_prob, metric="f1"):
    thresholds = np.arange(0.05, 0.55, 0.05)
    rows = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)

        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        acc = accuracy_score(y_true, preds)

        rows.append([t, precision, recall, f1, acc])

    df = pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1", "accuracy"])

    if metric == "f1":
        best = df.iloc[df["f1"].idxmax()]
    elif metric == "recall":
        best = df.iloc[df["recall"].idxmax()]
    elif metric == "precision":
        best = df.iloc[df["precision"].idxmax()]
    else:
        df["balanced_score"] = (df["precision"] + df["recall"]) / 2
        best = df.iloc[df["balanced_score"].idxmax()]

    return df, best


class ThresholdClassifier:
    def __init__(self, base_model, threshold=0.5):
        self.model = base_model
        self.threshold = threshold

    def predict(self, X):
        probs = self.model.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def set_threshold(self, val):
        self.threshold = val


def risk_category(prob):
    if prob < 0.20:
        return "low"
    elif prob < 0.35:
        return "medium"
    elif prob < 0.50:
        return "high"
    return "alert"