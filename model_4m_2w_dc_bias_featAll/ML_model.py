from pathlib import Path
from typing import List, Tuple

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.api import types as pd_types
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import save_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_error_metrics(data: pd.DataFrame, pred_col: str) -> Tuple[float, float, float, float, float]:
    """
    Compute error metrics between the true 'bias' column and the predictions column.

    Returns:
      (mae, rmse, pct_mae, pct_rmse, mean_error)
    - pct_* are percentages relative to the mean of the bias column (NaN if mean(bias) == 0).
    """
    if "bias" not in data.columns:
        raise KeyError("'bias' column is required in the data to compute error metrics.")
    if pred_col not in data.columns:
        raise KeyError(f"Prediction column '{pred_col}' not found in data.")

    y_true = data["bias"].values
    y_pred = data[pred_col].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mean_bias = np.mean(y_true)

    if mean_bias == 0:
        pct_mae = np.nan
        pct_rmse = np.nan
    else:
        pct_mae = (mae / mean_bias) * 100
        pct_rmse = (rmse / mean_bias) * 100

    mean_error = float(np.mean(y_true - y_pred))

    return float(mae), float(rmse), float(pct_mae), float(pct_rmse), mean_error


def train_xgb_model(
    anal: pd.DataFrame,
    feat_list: List[str],
    feat_name: str,
    out_path: str,
    model_num: int,
    ty: str,
    params: dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train an XGBoost regression model to predict 'bias' and save model + column order.

    Args:
      anal: DataFrame containing training data. Must include 'init_date' (datetime) and 'bias'.
      feat_list: list of columns to use for training, must include 'bias' column.
      feat_name: descriptive name for feature set (used in metrics output).
      out_path: directory where outputs (csv, model file, columns) will be saved.
      model_num: integer used to name the saved model and column file.
      ty: a tag used in filenames.
      params: optional dict of XGBoost training parameters.

    Returns:
      res_train: original DataFrame with an added column 'XGB' containing predictions.
      res_metrics: single-row DataFrame summarizing the error metrics.
    """
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if "init_date" not in anal.columns:
        raise KeyError("'init_date' column missing from input DataFrame.")
    if "bias" not in feat_list:
        raise ValueError("'bias' must be included in feat_list.")
    if "bias" not in anal.columns:
        raise KeyError("'bias' column missing from input DataFrame.")

    # Ensure init_date is datetime

    # If init_date is not already any datetime dtype (including timezone-aware),
    # convert it to datetime. pd.to_datetime will preserve timezone info if present.
    if not pd_types.is_datetime64_any_dtype(anal["init_date"]):
        anal = anal.copy()
        anal["init_date"] = pd.to_datetime(anal["init_date"])
    

    # Compute month_distance and weights (recent months get higher weight)
    T = anal["init_date"].max()
    anal["biweek_distance"] = ((T - anal["init_date"]).dt.days // 14)
    anal["weights"] = np.exp(-0.1 * anal["biweek_distance"])  # Exponential decay
    w_train = anal["weights"].values

    # Save training CSV for records
    csv_path = out_dir / f"ML_train_data_{ty}.csv"
    anal.to_csv(csv_path, index=False)
    logger.info("Saved training data to %s", csv_path)

    # Prepare training DataFrame
    cols = feat_list
    train_data = anal[cols].copy()

    x_train = train_data.drop(columns=["bias"])
    y_train = train_data["bias"].values

    # Save column order (feature names)
    save_columns(out_path, f"Model{model_num}_{ty}_col_order", list(x_train.columns))
    logger.info("Saved column order for model %s_%s", model_num, ty)

    # Default XGBoost params if not provided
    if params is None:
        params = {"objective": "reg:squarederror", "eval_metric": "mae", 'max_depth': 5, 'eta': 0.1,"subsample": 0.8, 
        "colsample_bytree": 0.7}

    # Create DMatrix with weights
    dtrain = xgb.DMatrix(x_train, label=y_train, weight=w_train)
    num_boost_round = 520
    # Train
    logger.info("Training XGBoost model (num_boost_round=%d)...", num_boost_round)
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    logger.info("Training complete.")

    # Predictions on training data
    y_pred_train = model.predict(dtrain)

    # Append predictions to original DataFrame copy
    res_train = pd.concat([anal.reset_index(drop=True), pd.DataFrame({"XGB": y_pred_train})], axis=1)

    # Compute metrics
    mae, rmse, pct_mae, pct_rmse, mean_error = compute_error_metrics(res_train, "XGB")
    metrics_row = {
        "Type": feat_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAE %": pct_mae,
        "RMSE %": pct_rmse,
        "Bias (Actual - Predicted)": mean_error,
    }
    res_metrics = pd.DataFrame([metrics_row])

    # Save model to disk
    model_filename = out_dir / f"model{model_num}_{ty}.ubj"
    model.save_model(str(model_filename))
    logger.info("Saved model to %s", model_filename)

    # Sanity check: load and compare predictions for the first row
    loaded_model = xgb.Booster()
    loaded_model.load_model(str(model_filename))
    loaded_pred = loaded_model.predict(dtrain)
    try:
        equal_first = np.isclose(loaded_pred[0], model.predict(dtrain)[0])
    except Exception:
        equal_first = False
    logger.info("Sanity check - first prediction equal after load: %s", equal_first)

    return res_train, res_metrics