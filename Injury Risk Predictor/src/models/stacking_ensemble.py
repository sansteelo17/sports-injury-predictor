"""
Stacking ensemble for injury prediction.

Combines multiple base learners (LightGBM, XGBoost, CatBoost) with a
meta-learner (Logistic Regression) for improved prediction accuracy.

Stacking helps because:
1. Different models capture different patterns in the data
2. The meta-learner learns optimal weights for combining predictions
3. Reduces variance compared to simple averaging
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from ..utils.logger import get_logger

logger = get_logger(__name__)


class StackingEnsemble:
    """
    Stacking ensemble classifier for injury prediction.

    Uses K-fold cross-validation to generate out-of-fold predictions
    from base learners, then trains a meta-learner on these predictions.

    Usage:
        ensemble = StackingEnsemble(n_folds=5)
        ensemble.fit(X_train, y_train, cat_features=['position', 'team'])
        predictions = ensemble.predict_proba(X_test)
    """

    def __init__(
        self,
        n_folds: int = 5,
        meta_learner: str = "logistic",
        base_learners: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Initialize stacking ensemble.

        Args:
            n_folds: Number of folds for cross-validation
            meta_learner: Type of meta-learner ("logistic", "ridge", "xgboost")
            base_learners: List of base learners to use. Default: ["lgb", "xgb", "catboost"]
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.meta_learner_type = meta_learner
        self.base_learner_names = base_learners or ["lgb", "xgb", "catboost"]
        self.random_state = random_state

        self.base_models_: List[List[Any]] = []  # [learner_idx][fold_idx]
        self.meta_model_: Any = None
        self.scaler_: StandardScaler = None
        self.cat_feature_indices_: List[int] = []
        self.feature_names_: List[str] = []

    def _get_base_learner(self, name: str, cat_features: List[int] = None) -> Any:
        """Create a base learner instance."""
        if name == "lgb":
            return LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                class_weight="balanced",
                random_state=self.random_state,
                verbose=-1
            )
        elif name == "xgb":
            return XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=self.random_state,
                verbosity=0
            )
        elif name == "catboost":
            return CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=self.random_state,
                verbose=False,
                cat_features=cat_features
            )
        else:
            raise ValueError(f"Unknown base learner: {name}")

    def _get_meta_learner(self) -> Any:
        """Create the meta-learner instance."""
        if self.meta_learner_type == "logistic":
            return LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state
            )
        elif self.meta_learner_type == "ridge":
            return LogisticRegression(
                C=0.1,
                penalty="l2",
                max_iter=1000,
                random_state=self.random_state
            )
        elif self.meta_learner_type == "xgboost":
            return XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state,
                verbosity=0
            )
        else:
            raise ValueError(f"Unknown meta learner: {self.meta_learner_type}")

    def _prepare_data_for_lgb_xgb(
        self,
        X: pd.DataFrame,
        cat_features: List[str] = None
    ) -> np.ndarray:
        """
        Prepare data for LightGBM/XGBoost (encode categoricals as integers).
        """
        X = X.copy()

        if cat_features is None:
            cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in cat_features:
            if col in X.columns:
                X[col] = X[col].astype("category").cat.codes

        return X.values.astype(np.float32)

    def _prepare_data_for_catboost(
        self,
        X: pd.DataFrame,
        cat_features: List[str] = None
    ) -> Tuple[pd.DataFrame, List[int]]:
        """
        Prepare data for CatBoost (keep categoricals as strings).
        """
        X = X.copy()

        if cat_features is None:
            cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Get indices of categorical features
        cat_indices = [X.columns.get_loc(c) for c in cat_features if c in X.columns]

        # Convert categoricals to string and fill NaN
        for col in cat_features:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna("Unknown")

        return X, cat_indices

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cat_features: List[str] = None
    ) -> "StackingEnsemble":
        """
        Fit the stacking ensemble.

        Args:
            X: Training features
            y: Training labels
            cat_features: List of categorical feature names

        Returns:
            self
        """
        logger.info(f"Fitting stacking ensemble with {len(self.base_learner_names)} base learners")

        self.feature_names_ = list(X.columns)
        self.cat_features_ = cat_features or X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Prepare data for different model types
        X_lgb_xgb = self._prepare_data_for_lgb_xgb(X, self.cat_features_)
        X_catboost, self.cat_feature_indices_ = self._prepare_data_for_catboost(X, self.cat_features_)

        y_array = y.values
        n_samples = len(y_array)
        n_base_learners = len(self.base_learner_names)

        # Initialize out-of-fold predictions matrix
        oof_predictions = np.zeros((n_samples, n_base_learners))

        # Initialize storage for trained models
        self.base_models_ = [[] for _ in range(n_base_learners)]

        # K-Fold cross-validation
        kfold = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )

        logger.info(f"Training base learners with {self.n_folds}-fold CV...")

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_lgb_xgb, y_array)):
            y_fold_train = y_array[train_idx]
            y_fold_val = y_array[val_idx]

            for learner_idx, learner_name in enumerate(self.base_learner_names):
                # Use appropriate data format for each model type
                if learner_name == "catboost":
                    X_fold_train = X_catboost.iloc[train_idx]
                    X_fold_val = X_catboost.iloc[val_idx]
                    model = self._get_base_learner(learner_name, cat_features=self.cat_feature_indices_)
                else:
                    X_fold_train = X_lgb_xgb[train_idx]
                    X_fold_val = X_lgb_xgb[val_idx]
                    model = self._get_base_learner(learner_name)

                model.fit(X_fold_train, y_fold_train)

                # Store out-of-fold predictions
                oof_predictions[val_idx, learner_idx] = model.predict_proba(X_fold_val)[:, 1]

                # Save trained model
                self.base_models_[learner_idx].append(model)

            logger.debug(f"Completed fold {fold_idx + 1}/{self.n_folds}")

        # Train meta-learner on OOF predictions
        logger.info("Training meta-learner on out-of-fold predictions...")

        self.scaler_ = StandardScaler()
        oof_scaled = self.scaler_.fit_transform(oof_predictions)

        self.meta_model_ = self._get_meta_learner()
        self.meta_model_.fit(oof_scaled, y_array)

        # Log OOF performance
        oof_auc = roc_auc_score(y_array, self.meta_model_.predict_proba(oof_scaled)[:, 1])
        logger.info(f"Stacking ensemble fitted - OOF ROC-AUC: {oof_auc:.4f}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features to predict

        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        # Prepare data for different model types
        X_lgb_xgb = self._prepare_data_for_lgb_xgb(X, self.cat_features_)
        X_catboost, _ = self._prepare_data_for_catboost(X, self.cat_features_)

        n_samples = len(X)
        n_base_learners = len(self.base_learner_names)

        # Get predictions from all base models (average across folds)
        base_predictions = np.zeros((n_samples, n_base_learners))

        for learner_idx, learner_name in enumerate(self.base_learner_names):
            fold_preds = np.zeros((n_samples, self.n_folds))

            for fold_idx, model in enumerate(self.base_models_[learner_idx]):
                if learner_name == "catboost":
                    fold_preds[:, fold_idx] = model.predict_proba(X_catboost)[:, 1]
                else:
                    fold_preds[:, fold_idx] = model.predict_proba(X_lgb_xgb)[:, 1]

            # Average predictions across folds
            base_predictions[:, learner_idx] = fold_preds.mean(axis=1)

        # Scale and pass to meta-learner
        base_scaled = self.scaler_.transform(base_predictions)
        meta_proba = self.meta_model_.predict_proba(base_scaled)

        return meta_proba

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features to predict
            threshold: Classification threshold

        Returns:
            Array of predicted labels (0 or 1)
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def get_base_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get individual predictions from each base learner.

        Useful for analyzing model agreement and confidence.
        """
        # Prepare data for different model types
        X_lgb_xgb = self._prepare_data_for_lgb_xgb(X, self.cat_features_)
        X_catboost, _ = self._prepare_data_for_catboost(X, self.cat_features_)

        predictions = {}
        for learner_idx, learner_name in enumerate(self.base_learner_names):
            fold_preds = []
            for model in self.base_models_[learner_idx]:
                if learner_name == "catboost":
                    fold_preds.append(model.predict_proba(X_catboost)[:, 1])
                else:
                    fold_preds.append(model.predict_proba(X_lgb_xgb)[:, 1])

            predictions[f"{learner_name}_prob"] = np.mean(fold_preds, axis=0)
            predictions[f"{learner_name}_std"] = np.std(fold_preds, axis=0)

        return pd.DataFrame(predictions)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate ensemble on test data.

        Returns dict with ROC-AUC, precision, recall, F1.
        """
        proba = self.predict_proba(X_test)[:, 1]
        preds = (proba >= threshold).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_test, proba),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        }

        logger.info(f"Ensemble evaluation: ROC-AUC={metrics['roc_auc']:.4f}, "
                   f"F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}")

        return metrics


def train_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame = None,
    y_test: pd.Series = None,
    cat_features: List[str] = None,
    n_folds: int = 5
) -> Tuple[StackingEnsemble, Dict[str, float]]:
    """
    Convenience function to train and evaluate a stacking ensemble.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (optional)
        y_test: Test labels (optional)
        cat_features: Categorical feature names
        n_folds: Number of CV folds

    Returns:
        Trained ensemble and evaluation metrics (if test data provided)

    Example:
        ensemble, metrics = train_stacking_ensemble(
            X_train, y_train, X_test, y_test,
            cat_features=['position', 'team']
        )
        print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    """
    ensemble = StackingEnsemble(n_folds=n_folds)
    ensemble.fit(X_train, y_train, cat_features=cat_features)

    metrics = {}
    if X_test is not None and y_test is not None:
        metrics = ensemble.evaluate(X_test, y_test)

    return ensemble, metrics


def compare_ensemble_vs_individual(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cat_features: List[str] = None
) -> pd.DataFrame:
    """
    Compare stacking ensemble against individual models.

    Returns DataFrame with metrics for each approach.
    """
    results = []

    # Train stacking ensemble
    logger.info("Training stacking ensemble...")
    ensemble = StackingEnsemble(n_folds=5)
    ensemble.fit(X_train, y_train, cat_features=cat_features)
    ensemble_metrics = ensemble.evaluate(X_test, y_test)
    ensemble_metrics["model"] = "Stacking Ensemble"
    results.append(ensemble_metrics)

    # Prepare data for individual models (use ensemble's methods)
    X_train_lgb = ensemble._prepare_data_for_lgb_xgb(X_train.copy(), cat_features)
    X_test_lgb = ensemble._prepare_data_for_lgb_xgb(X_test.copy(), cat_features)
    X_train_cat, cat_indices = ensemble._prepare_data_for_catboost(X_train.copy(), cat_features)
    X_test_cat, _ = ensemble._prepare_data_for_catboost(X_test.copy(), cat_features)

    # Train and evaluate LightGBM
    logger.info("Training LightGBM...")
    lgb_model = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        class_weight="balanced", random_state=42, verbose=-1
    )
    lgb_model.fit(X_train_lgb, y_train.values)
    lgb_proba = lgb_model.predict_proba(X_test_lgb)[:, 1]
    lgb_preds = (lgb_proba >= 0.5).astype(int)
    results.append({
        "model": "LightGBM",
        "roc_auc": roc_auc_score(y_test, lgb_proba),
        "precision": precision_score(y_test, lgb_preds, zero_division=0),
        "recall": recall_score(y_test, lgb_preds, zero_division=0),
        "f1": f1_score(y_test, lgb_preds, zero_division=0),
    })

    # Train and evaluate XGBoost
    logger.info("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_train_lgb, y_train.values)
    xgb_proba = xgb_model.predict_proba(X_test_lgb)[:, 1]
    xgb_preds = (xgb_proba >= 0.5).astype(int)
    results.append({
        "model": "XGBoost",
        "roc_auc": roc_auc_score(y_test, xgb_proba),
        "precision": precision_score(y_test, xgb_preds, zero_division=0),
        "recall": recall_score(y_test, xgb_preds, zero_division=0),
        "f1": f1_score(y_test, xgb_preds, zero_division=0),
    })

    # Train and evaluate CatBoost (uses different data format)
    logger.info("Training CatBoost...")
    cat_model = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        random_seed=42, verbose=False, cat_features=cat_indices
    )
    cat_model.fit(X_train_cat, y_train.values)
    cat_proba = cat_model.predict_proba(X_test_cat)[:, 1]
    cat_preds = (cat_proba >= 0.5).astype(int)
    results.append({
        "model": "CatBoost",
        "roc_auc": roc_auc_score(y_test, cat_proba),
        "precision": precision_score(y_test, cat_preds, zero_division=0),
        "recall": recall_score(y_test, cat_preds, zero_division=0),
        "f1": f1_score(y_test, cat_preds, zero_division=0),
    })

    # Simple average ensemble
    logger.info("Evaluating simple average ensemble...")
    base_preds = ensemble.get_base_predictions(X_test)
    avg_proba = base_preds[[c for c in base_preds.columns if c.endswith("_prob")]].mean(axis=1)
    avg_preds = (avg_proba >= 0.5).astype(int)

    results.append({
        "model": "Simple Average",
        "roc_auc": roc_auc_score(y_test, avg_proba),
        "precision": precision_score(y_test, avg_preds, zero_division=0),
        "recall": recall_score(y_test, avg_preds, zero_division=0),
        "f1": f1_score(y_test, avg_preds, zero_division=0),
    })

    return pd.DataFrame(results).set_index("model")
