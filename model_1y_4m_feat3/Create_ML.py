"""
Cleaner, more robust script to prepare training data and run the XGB_model.

Usage:
    python run_xgb.py --model-num 1 --test-dts 2025010100 --name example

Notes:
- Expects utils.create_folder_ml and utils.get_cols to exist.
- Expects ML_model.XGB_model to be importable and return (fcst_data, error_met).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# Import functions from your codebase (explicit imports instead of star import)
from ML_model import train_xgb_model
from utils import create_folder_ml, get_cols

# Keep warnings from being noisy for production runs; only hide FutureWarning as before
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare data and run XGB_model.")
    p.add_argument("--model-num", required=True, help="Model number (string).")
    p.add_argument(
        "--test-dts",
        required=True,
        help="Test timestamp in format YYYYmmddHH (e.g. 2025010100).",
    )
    p.add_argument("--back_year", required=True, help="Back year for training data (e.g. 1,2,3).")
    p.add_argument(
        "--data-path",
        required=False,
        default="../New_ML_data/Wind_NAM_2014_2025.csv",
        help="Path to source CSV with training data.",
    )
    return p.parse_args()


def parse_test_datetime(test_dts_str: str) -> pd.Timestamp:
    """Parse test datetime string to timezone-aware pandas Timestamp (UTC)."""
    try:
        ts = pd.to_datetime(test_dts_str, format="%Y%m%d%H", utc=True)
    except ValueError as exc:
        raise ValueError(f"test_dts must be in format YYYYmmddHH, got '{test_dts_str}'") from exc
    return ts


def load_and_clean_data(csv_path: str | Path) -> pd.DataFrame:
    """Load CSV and clean sentinel values (-999, -999.99)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    logging.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    # Ensure datetime columns are timezone-aware UTC
    for col in ("init_date", "fcst_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)

    # Replace sentinel values with NaN and drop rows with any NaN (matches original logic)
    df = df.replace([-999, -999.0, -999.99], np.nan)
    before = len(df)
    df = df.dropna(how="any")
    after = len(df)
    logging.info("Dropped %d rows with sentinel/missing values (from %d to %d)", before - after, before, after)

    return df


def prepare_training_subset(
    df: pd.DataFrame, test_dts_val: pd.Timestamp, back_year: int
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Filter the input dataframe to the training window:
    - train_start = test_dts_val - 1 year
    - train_end (exclusive) = test_dts_val - 6 hours
    """
    train_start_val = test_dts_val - relativedelta(years=back_year)
    train_end_exclusive = test_dts_val - relativedelta(hours=6)

    if "init_date" not in df.columns:
        raise KeyError("Dataframe must contain 'init_date' column for filtering by date.")

    mask = (df["init_date"] >= train_start_val) & (df["init_date"] < train_end_exclusive)
    train_subset = df.loc[mask].reset_index(drop=True)

    if train_subset.empty:
        raise ValueError(
            f"No training rows found for window {train_start_val} <= init_date < {train_end_exclusive}"
        )

    # Compute bias column (same formula as original)
    if {"DAWF", "MW Prod"}.issubset(train_subset.columns):
        train_subset["bias"] = train_subset["DAWF"] - train_subset["MW Prod"]
    else:
        logging.warning("DAWF or MW Prod column not found; skipping bias computation.")

    logging.info(
        "Prepared training subset with %d rows: init_date min=%s max=%s",
        len(train_subset),
        train_subset["init_date"].min(),
        train_subset["init_date"].max(),
    )

    return train_subset, train_start_val, train_end_exclusive


def save_model_info(out_folder: Path, model_num: str, test_dts_str: str, train_start: pd.Timestamp, train_end: pd.Timestamp):
    meta = {
        "Model Num": model_num,
        "Test Year": test_dts_str,
        "Train Start": train_start.strftime("%Y%m%d%H"),
        "Train End": train_end.strftime("%Y%m%d%H"),
    }
    out_folder.mkdir(parents=True, exist_ok=True)
    meta_file = out_folder / f"Model{model_num}_info_{test_dts_str}.json"
    logging.info("Saving model metadata to %s", meta_file)
    with meta_file.open("w") as fh:
        json.dump(meta, fh)


def main():
    args = parse_args()
    model_num = str(args.model_num)
    test_dts_str = str(args.test_dts)
    back_year = int(args.back_year)
    data_path = args.data_path

    # Parse datetimes
    test_dts_val = parse_test_datetime(test_dts_str)
    logging.info("Test datetime (UTC): %s", test_dts_val.isoformat())

    # Create output folder using the provided helper (keeps original behavior)
    out_folder = Path(create_folder_ml(model_num, test_dts_str, "all"))
    logging.info("Output folder: %s", out_folder)

    # Load columns list (expects JSON with 'cols' key)
    cols_json = get_cols("selected_cols/cols_all.json")
    cols_list = cols_json.get("cols", []) if isinstance(cols_json, dict) else []
    if not cols_list:
        logging.warning("No columns found in selected_cols/cols_all.json; continuing with empty cols list")

    # Load and clean data
    df = load_and_clean_data(data_path)

    # Filter training subset
    train_df, train_start_val, train_end_exclusive = prepare_training_subset(df, test_dts_val, back_year)

    # Save metadata about train window
    save_model_info(out_folder, model_num, test_dts_str, train_start_val, train_end_exclusive)

    # Call the model (keeps the original call signature)
    logging.info("Calling XGB_model...")
    fcst_data, error_met = train_xgb_model(train_df, cols_list, "XGB", str(out_folder), model_num, test_dts_str)

    logging.info("Training Done.")


if __name__ == "__main__":
    main()
