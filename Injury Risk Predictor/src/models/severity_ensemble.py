import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def build_severity_ensemble(lgb_model, xgb_model, X_test, w_lgb=0.6, w_xgb=0.4):
    lgb_pred = lgb_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    ensemble_pred = w_lgb * lgb_pred + w_xgb * xgb_pred

    return pd.DataFrame({
        "lgb_pred": lgb_pred,
        "xgb_pred": xgb_pred,
        "ensemble_pred": ensemble_pred
    })

def evaluate_severity_ensemble(lgb_pred_log, xgb_pred_log, y_test_log,
                               w_lgb=0.6, w_xgb=0.4):

    # 1. Weighted blend in LOG SPACE
    ens_log = w_lgb * lgb_pred_log + w_xgb * xgb_pred_log

    # 2. Convert predictions BACK to severity days
    ens_pred = np.expm1(ens_log)

    # 3. Convert y_test BACK to severity days
    y_true = np.expm1(y_test_log)

    # 4. Evaluate
    mae = mean_absolute_error(y_true, ens_pred)
    rmse = root_mean_squared_error(y_true, ens_pred)
    r2 = r2_score(y_true, ens_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }, ens_pred