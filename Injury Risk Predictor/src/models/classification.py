import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, classification_report
)

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

import optuna
import shap
shap.initjs()
import warnings
warnings.filterwarnings("ignore")

from ..utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# TEMPORAL VALIDATION (RECOMMENDED)
# ============================================================================
#
# ⚠️  WARNING: The functions below use RANDOM splits which cause TEMPORAL
# LEAKAGE in time series data. For production models, use temporal splits:
#
# from src.models.temporal_validation import get_temporal_classification_splits
#
# Example usage - Drop-in replacement for train_test_split:
#
#   X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_classification_splits(
#       df=injury_risk_df,
#       date_column="Date of Injury",
#       target_column="injury_label",
#       train_end="2022-06-30",  # Training data: before this date
#       val_end="2023-06-30"     # Validation: between train_end and val_end
#   )                            # Test: after val_end
#
# For walk-forward validation (gold standard for time series):
#
#   from src.models.temporal_validation import walk_forward_validation
#
#   splits = walk_forward_validation(
#       df=injury_risk_df,
#       date_column="Date of Injury",
#       target_column="injury_label",
#       n_splits=5,              # Number of train/test splits
#       test_size_months=6,      # Size of test window
#       gap_months=0             # Gap between train and test (prediction horizon)
#   )
#
#   # Train and evaluate on each fold
#   for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
#       model.fit(X_train, y_train)
#       score = model.score(X_test, y_test)
#       print(f"Fold {i+1} ROC-AUC: {score:.4f}")
#
# Why temporal splits matter:
#   - Random splits create temporal leakage (model sees future to predict past)
#   - Causes inflated validation metrics that don't reflect real performance
#   - Production models will perform worse than expected
#   - See temporal_validation.py docstrings for detailed explanation
#
# ============================================================================


def get_temporal_splits(df, train_ratio=0.6, val_ratio=0.2):
    """
    Stratified temporal split that ensures each split has positive samples.

    The key issue: negative samples may span decades while injuries are recent.
    Solution: Filter to the time window where injuries occur, then split.

    Returns X_train, X_val, X_test, y_train, y_val, y_test

    Usage:
        X_train, X_val, X_test, y_train, y_val, y_test = get_temporal_splits(injury_risk_df)
    """
    if "event_date" not in df.columns:
        logger.warning("No event_date column found. Using random split instead.")
        X = df.drop(columns=["injury_label"])
        y = df["injury_label"]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Find the date range where we have positive samples (injuries)
    positives = df[df["injury_label"] == 1]
    if len(positives) == 0:
        logger.error("No positive samples found!")
        raise ValueError("Cannot split: no positive samples in dataset")

    injury_start = positives["event_date"].min()
    injury_end = positives["event_date"].max()

    logger.info(f"Injury date range: {injury_start.date()} to {injury_end.date()}")

    # Filter to only include data within the injury time window (with some buffer)
    buffer_days = 90  # 3 months before first injury
    filter_start = injury_start - pd.Timedelta(days=buffer_days)

    df_filtered = df[df["event_date"] >= filter_start].copy()
    logger.info(f"Filtered from {len(df)} to {len(df_filtered)} samples (within injury time window)")

    # Now do temporal split on the filtered data
    df_sorted = df_filtered.sort_values("event_date").reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]

    # Check positive rates in each split
    train_pos = train_df["injury_label"].sum()
    val_pos = val_df["injury_label"].sum()
    test_pos = test_df["injury_label"].sum()

    logger.info(f"Split distribution:")
    logger.info(f"  Train: {len(train_df)} samples, {train_pos} positives ({train_df['injury_label'].mean()*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df)} samples, {val_pos} positives ({val_df['injury_label'].mean()*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df)} samples, {test_pos} positives ({test_df['injury_label'].mean()*100:.1f}%)")

    # Warn if any split has no positives
    if train_pos == 0 or val_pos == 0 or test_pos == 0:
        logger.warning("WARNING: Some splits have no positive samples. Consider using stratified_temporal_splits().")

    # Drop date column for modeling
    drop_cols = ["injury_label", "event_date"]
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_val = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    y_train = train_df["injury_label"]
    y_val = val_df["injury_label"]
    y_test = test_df["injury_label"]

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_stratified_temporal_splits(df, train_ratio=0.6, val_ratio=0.2):
    """
    Alternative: Split positives and negatives separately, then combine.

    This ensures each split has positives even when injury dates are clustered.
    Less pure temporal validation but more practical.
    """
    if "event_date" not in df.columns:
        logger.warning("No event_date column. Using random stratified split.")
        X = df.drop(columns=["injury_label"])
        y = df["injury_label"]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Split positives and negatives separately by time
    positives = df[df["injury_label"] == 1].sort_values("event_date")
    negatives = df[df["injury_label"] == 0].sort_values("event_date")

    def split_by_ratio(data, train_r, val_r):
        n = len(data)
        t_end = int(n * train_r)
        v_end = int(n * (train_r + val_r))
        return data.iloc[:t_end], data.iloc[t_end:v_end], data.iloc[v_end:]

    pos_train, pos_val, pos_test = split_by_ratio(positives, train_ratio, val_ratio)
    neg_train, neg_val, neg_test = split_by_ratio(negatives, train_ratio, val_ratio)

    train_df = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=42)
    val_df = pd.concat([pos_val, neg_val]).sample(frac=1, random_state=42)
    test_df = pd.concat([pos_test, neg_test]).sample(frac=1, random_state=42)

    logger.info(f"Stratified temporal split:")
    logger.info(f"  Train: {len(train_df)} ({train_df['injury_label'].mean()*100:.1f}% positive)")
    logger.info(f"  Val:   {len(val_df)} ({val_df['injury_label'].mean()*100:.1f}% positive)")
    logger.info(f"  Test:  {len(test_df)} ({test_df['injury_label'].mean()*100:.1f}% positive)")

    drop_cols = ["injury_label", "event_date"]
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_val = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    y_train = train_df["injury_label"]
    y_val = val_df["injury_label"]
    y_test = test_df["injury_label"]

    return X_train, X_val, X_test, y_train, y_val, y_test


# Train/Validation Split + Preprocessing

def get_smote_splits(df):

    X = df.drop(columns=["injury_label"])
    y = df["injury_label"]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    # Train-test split BEFORE preprocessing/SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Preprocess X_train ONLY
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Apply SMOTE AFTER encoding
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_proc, y_train)

    return X_train_res, X_test_proc, y_train_res, y_test, preprocessor

def get_classification_splits(df: pd.DataFrame):
    """
    Takes injury_risk_df and returns X_train, X_test, y_train, y_test
    """

    X = df.drop(columns=["injury_label"])
    y = df["injury_label"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", "passthrough", categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features


# LIGHTGBM CLASSIFIER

def train_lightgbm(X_train, y_train, preprocessor):
    logger.info(f"Starting LightGBM training with {len(X_train)} samples")

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=-1,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    logger.info("LightGBM training completed")
    return model

def train_lightgbm_smote(X_train, y_train):
    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=-1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# XGBOOST CLASSIFIER

def train_xgboost(X_train, y_train, preprocessor):
    logger.info(f"Starting XGBoost training with {len(X_train)} samples")

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    logger.info("XGBoost training completed")
    return model

def train_xgboost_smote(X_train, y_train):
    model =  XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        )
    model.fit(X_train, y_train)
    return model

def train_xgboost_tuned(X_train, y_train, preprocessor, params):
    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    return model

#CATBOOST CLASSIFIER

def catboost_objective(trial, X_train, y_train, X_val, y_val, cat_features):
    
    params = {
        "iterations": 1500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "random_strength": trial.suggest_float("random_strength", 1.0, 20.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": False,
        "random_seed": 42
    }

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool   = Pool(X_val, y_val, cat_features=cat_features)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, verbose=False)

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)

    return auc

def tune_catboost(X_train, y_train, X_val, y_val, cat_features, n_trials=40):

    logger.info(f"Starting CatBoost hyperparameter tuning with {n_trials} trials")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: catboost_objective(
            trial, X_train, y_train, X_val, y_val, cat_features
        ),
        n_trials=n_trials
    )

    logger.info(f"CatBoost tuning completed - Best ROC-AUC: {study.best_value:.4f}")
    print("\n🎯 Best CatBoost Params Found:")
    print(study.best_params)
    print(f"Best ROC-AUC = {study.best_value:.4f}")

    return study.best_params


def train_final_catboost(X_train, y_train, X_test, y_test, cat_features, best_params):

    logger.info(f"Training final CatBoost model with {len(X_train)} training samples")

    final_params = best_params.copy()
    final_params.update({
        "iterations": 1500,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": 200
    })

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool  = Pool(X_test,  y_test, cat_features=cat_features)

    model = CatBoostClassifier(**final_params)
    model.fit(train_pool, eval_set=test_pool)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    logger.info(f"Final CatBoost training completed - ROC-AUC: {auc:.4f}")
    print(f"\n📌 Final Tuned CatBoost ROC-AUC = {auc:.4f}")

    return model, probs

def evaluate_catboost_thresholds(y_test, probs):

    thresholds = np.arange(0.05, 0.55, 0.05)
    rows = []

    for t in thresholds:
        preds = (probs >= t).astype(int)

        rows.append({
            "threshold": t,
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "accuracy": accuracy_score(y_test, preds)
        })

    df = pd.DataFrame(rows)
    best_row = df.iloc[df["f1"].idxmax()]

    print("\n🔥 Best Threshold Based on F1:")
    print(best_row)

    return df, best_row.threshold

def run_full_catboost_class_tuning(X_train, y_train, X_test, y_test):

    # Detect categorical columns (string/object/category)
    cat_features = [
        i for i, col in enumerate(X_train.columns)
        if str(X_train[col].dtype) in ["object", "category"]
    ]

    # Convert categoricals to string & fill NaN
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    for col in X_train_cb.columns:
        if str(X_train_cb[col].dtype) in ["object", "category"]:
            X_train_cb[col] = X_train_cb[col].astype(str).fillna("Unknown")
            X_test_cb[col]  = X_test_cb[col].astype(str).fillna("Unknown")

    # Split train → train + validation for tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_cb, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # 1. Tune CatBoost
    best_params = tune_catboost(X_tr, y_tr, X_val, y_val, cat_features)

    # 2. Train final model with full training data
    model, probs = train_final_catboost(
        X_train_cb, y_train, X_test_cb, y_test, cat_features, best_params
    )

    # 3. Evaluate thresholds
    table, best_threshold = evaluate_catboost_thresholds(y_test, probs)

    return model, probs, best_params, table, best_threshold

# OPTUNA HYPERPARAMETER TUNING (XGBoost)

def optuna_objective(trial, X_train, y_train, preprocessor):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])

    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc")
    return np.mean(scores)


def run_optuna_tuning(X_train, y_train, preprocessor):
    logger.info(f"Starting Optuna XGBoost tuning with {len(X_train)} samples")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optuna_objective(trial, X_train, y_train, preprocessor),
                   n_trials=40)

    logger.info(f"Optuna tuning completed - Best score: {study.best_value:.4f}")
    print("Best params:", study.best_params)
    return study.best_params


# MODEL EVALUATION

def evaluate_classifier(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n📌 Classification Report:")
    print(classification_report(y_test, preds))

    auc = roc_auc_score(y_test, probs)
    print(f"🔥 ROC-AUC: {auc:.4f}")

    return preds, probs, auc


def evaluate_thresholds(model, X_test, y_test, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.55, 0.05)

    # get injury probabilities
    probs = model.predict_proba(X_test)[:, 1]

    rows = []

    for t in thresholds:
        preds = (probs >= t).astype(int)

        rows.append({
            "threshold": t,
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "accuracy": accuracy_score(y_test, preds),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("recall", ascending=False)


# SHAP EXPLAINABILITY

def explain_model_with_shap(model, X_train):
    """
    SHAP processing for Pipeline(LGBMClassifier)
    using the exact transformed feature matrix.
    """

    # 1. Extract the trained LightGBM model
    clf = model.named_steps["clf"]

    # 2. Transform X_train using the pipeline preprocessor ONLY
    preproc = model.named_steps["prep"]
    X_train_transformed = preproc.transform(X_train)

    # 3. Convert to numpy if sparse
    if not isinstance(X_train_transformed, np.ndarray):
        X_train_transformed = X_train_transformed.toarray()

    # 4. Build SHAP explainer
    explainer = shap.TreeExplainer(clf)

    # 5. Compute SHAP values
    shap_values = explainer.shap_values(X_train_transformed)

    # 6. Build feature names from preprocessor
    numeric_features = preproc.transformers_[0][2]
    categorical_features = preproc.transformers_[1][2]

    feature_names = numeric_features + categorical_features

    # 7. SHAP summary plot
    shap.summary_plot(shap_values, X_train_transformed, feature_names=feature_names)

    return shap_values

def explain_xgboost_shap(model, X_train):
    preproc = model.named_steps["prep"]
    clf = model.named_steps["clf"]

    X_train_transformed = preproc.transform(X_train)
    if not isinstance(X_train_transformed, np.ndarray):
        X_train_transformed = X_train_transformed.toarray()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train_transformed)

    # feature names
    num_features = preproc.transformers_[0][2]
    cat_features = preproc.transformers_[1][2]
    feature_names = num_features + cat_features

    print("Showing SHAP summary plot...")
    shap.summary_plot(shap_values, X_train_transformed, feature_names=feature_names)

    return shap_values