"""
Refactored prediction module.

Replaces the original Model_pred class with a cleaner, better-documented,
and more robust ModelPredictor class.

Assumptions:
- `read_col_order` is provided by `utils` (same as the original).
- `Model_Clean` is provided by `Clean` and has a `.run()` method that returns a pd.DataFrame.
- The original JSON structure for selected columns is { "cols": [...] }.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import xgboost as xgb

# Local imports (explicit, avoid star imports)
from utils import read_col_order
from Clean import ModelClean

# Reduce verbosity for third-party warnings in user code
logging.getLogger("xgboost").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ModelPredictor:
    """
    Encapsulates loading of model artifacts, preparing features and returning
    final prediction DataFrame.

    Example:
        pred = ModelPredictor(dts, model_path, model_year).run()
    """

    def __init__(self, dts: Any, model_path: str | Path, model_year: int) -> None:
        """
        Args:
            dts: input passed to Model_Clean (kept generic to match original behaviour)
            model_path: path to directory containing selected_cols and model files
            model_year: training year used to locate model and column order
        """
        self.dts = dts
        self.model_path = Path(model_path)
        self.model_year = int(model_year)

    def _load_selected_columns(self) -> List[str]:
        """Load list of selected columns from JSON file."""
        cols_file = self.model_path / "selected_cols" / "cols_all.json"
        if not cols_file.exists():
            raise FileNotFoundError(f"Selected columns file not found: {cols_file}")
        try:
            with cols_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in selected columns file: {cols_file}") from exc

        cols = payload.get("cols")
        if not isinstance(cols, list):
            raise ValueError(f"Invalid 'cols' structure in {cols_file}. Expected a list.")
        return cols

    def _load_model(self) -> xgb.Booster:
        """Load the trained XGBoost model from disk."""
        model_file = self.model_path / f"Model1_training_{self.model_year}_all" / f"model1_{self.model_year}.ubj"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        model = xgb.Booster()
        try:
            model.load_model(str(model_file))
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from {model_file}") from exc
        return model

    def _prepare_features(self, data: pd.DataFrame, selected_cols: List[str]) -> pd.DataFrame:
        """
        Select feature columns, apply column ordering expected by the model,
        and cast to float.

        Raises:
            KeyError if required columns are missing.
        """
        # remove "bias" if present (kept behaviour of original)
        feature_cols = [c for c in selected_cols if c != "bias"]

        missing = [c for c in feature_cols if c not in data.columns]
        if missing:
            raise KeyError(f"Missing required feature columns in cleaned data: {missing}")

        pred_data = data[feature_cols].copy()

        # read_col_order is assumed to return a list of column names in correct order
        order_file_relative = f"Model1_training_{self.model_year}_all/Model1_{self.model_year}_col_order.txt"
        col_order = read_col_order(self.model_path, order_file_relative)

        # Validate column order
        if not isinstance(col_order, list) or not col_order:
            raise ValueError(f"Invalid column order returned by read_col_order: {col_order}")

        missing_in_order = [c for c in col_order if c not in pred_data.columns]
        if missing_in_order:
            raise KeyError(f"Column order refers to columns not present in data: {missing_in_order}")

        final_pred_data = pred_data[col_order].astype(float, copy=True)
        return final_pred_data

    def _assemble_output(self, data: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
        """
        Build final output DataFrame with DATE, TIME and POWER OUTPUT (kWh).

        - Renames 'fcst_date' -> 'Date' if present (keeps original behaviour).
        - Index is expected to be a datetime-like index after setting 'Date'.
        - Missing output values are filled with "-999.99" (string), preserving the original behaviour.
        """
        df = data.copy()

        # Rename forecast column if present (original code renamed 'fcst_date' -> 'Date')
        if "fcst_date" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"fcst_date": "Date"})

        if "Date" in df.columns:
            df = df.set_index("Date")
        else:
            # If index is already datetime-like, keep it; otherwise raise
            if not isinstance(df.index, pd.DatetimeIndex):
                raise KeyError("No 'Date' column and index is not a DatetimeIndex. Cannot build output timestamps.")

        # Create DATE and TIME strings
        df["DATE"] = df.index.strftime("%y%m%d")
        df["TIME"] = df.index.strftime("%H%M")

        # Add model outputs
        if predictions.shape[0] != len(df):
            raise ValueError("Number of predictions does not match number of rows in cleaned data.")

        df["pred_bias"] = predictions

        if "DAWF" not in df.columns:
            raise KeyError("Expected column 'DAWF' not found in cleaned data. Required to compute final prediction.")

        df["pred"] = df["DAWF"] - df["pred_bias"]

        df_out = df[["DATE", "TIME", "pred"]].rename(columns={"pred": "POWER OUTPUT (kWh)"})

        # Fill missing values with sentinel used in original code
        df_out = df_out.fillna("-999.99")
        return df_out

    def run(self) -> pd.DataFrame:
        """
        Main entry point:
         - Runs Model_Clean on the provided dts to get cleaned features
         - Loads selected columns and model artifacts
         - Predicts and returns formatted output DataFrame
        """
        logger.info("Starting data cleaning step with Model_Clean.")
        cleaned = ModelClean(self.dts).run()
        if not isinstance(cleaned, pd.DataFrame):
            raise TypeError("Model_Clean.run() must return a pandas DataFrame.")

        logger.info("Loading selected columns.")
        selected_cols = self._load_selected_columns()

        logger.info("Preparing features for prediction.")
        features = self._prepare_features(cleaned, selected_cols)

        logger.info("Loading trained model.")
        model = self._load_model()

        logger.info("Creating DMatrix and predicting.")
        dmatrix = xgb.DMatrix(features)
        predictions = model.predict(dmatrix)

        logger.info("Assembling final output DataFrame.")
        output = self._assemble_output(cleaned, predictions)
        logger.info("Prediction run complete.")
        return output