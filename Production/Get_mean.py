"""
Cleaned and improved version of the original Mean_model class.

Changes:
- Renamed class to MeanModel and methods to PEP8-friendly names.
- Added type hints, docstrings and logging.
- Made out_path configurable (either pass explicitly or read from config file).
- Safer file/path handling via pathlib.
- Better error messages for missing files / empty model list.
- Robust TIME handling and missing-value filling.
- Minor performance/readability improvements.

Usage:
    model = MeanModel(dts="2025-10-29", model_list=["m1", "m2"])
    df = model.run()
"""
from __future__ import annotations

import configparser
import logging
from functools import reduce
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MeanModel:
    """
    Read per-model CSVs for a date (dts), merge them on DATE and TIME,
    compute the row-wise mean prediction across the given model columns,
    and return a standardized output DataFrame.

    Parameters
    ----------
    dts : str
        Date string used in the input file path (kept as in original code).
    model_list : list[str]
        List of model names. Each model's CSV is expected under
        {out_path}/{model}/{dts}/{dts}.csv and must contain columns DATE, TIME and
        "POWER OUTPUT (kWh)".
    out_path : str | Path, optional
        Base output path. If not provided, it is read from the config file at
        /mnt/trade05_data/wind_v5/Production/config.ini (same as original).
    config_path : str | Path, optional
        Path to config.ini when out_path is not provided.
    """

    def __init__(
        self,
        dts: str,
        model_list: List[str],
        out_path: Optional[Union[str, Path]] = None,
        config_path: Union[str, Path] = "/mnt/trade05_data/wind_v5_new_code/Production/config.ini",
    ) -> None:
        if not model_list:
            raise ValueError("model_list must contain at least one model name")

        self.dts = dts
        self.model_list = list(model_list)

        if out_path is None:
            cfg = configparser.ConfigParser()
            cfg.read(str(config_path))
            try:
                out_path = cfg["paths"]["out_path"]
            except Exception as e:
                raise RuntimeError(
                    f"Could not read 'paths.out_path' from config {config_path}: {e}"
                )
        self.out_path = Path(out_path)

    def _read_model_csv(self, name: str) -> pd.DataFrame:
        """
        Read a single model CSV and rename the 'POWER OUTPUT (kWh)' column to the model name.

        Returns a DataFrame with columns: DATE, TIME, <model_name>
        """
        file_path = self.out_path / name / self.dts / f"{self.dts}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Expected file for model '{name}' not found: {file_path}")

        # Read TIME as string to preserve leading zeros; we'll normalize later
        df = pd.read_csv(file_path, dtype={"TIME": str})
        if "POWER OUTPUT (kWh)" not in df.columns:
            raise KeyError(f"'POWER OUTPUT (kWh)' column not found in {file_path}")

        # Keep only the necessary columns to avoid accidental duplicates
        df = df[["DATE", "TIME", "POWER OUTPUT (kWh)"]].copy()
        df = df.rename(columns={"POWER OUTPUT (kWh)": name})
        return df

    def calculate_models(self) -> pd.DataFrame:
        """
        Read all model CSVs, merge them on DATE and TIME (inner join),
        and compute the mean prediction across the model columns into 'pred'.
        """
        dfs: List[pd.DataFrame] = []
        for name in self.model_list:
            try:
                df = self._read_model_csv(name)
            except Exception:
                logger.exception("Failed to read model CSV for %s", name)
                raise
            dfs.append(df)

        # Merge all dataframes on DATE and TIME using inner joins
        merged_df = reduce(
            lambda left, right: pd.merge(left, right, on=["DATE", "TIME"], how="inner"),
            dfs,
        )

        # Ensure all expected model columns are present after the merge
        missing_cols = [c for c in self.model_list if c not in merged_df.columns]
        if missing_cols:
            raise RuntimeError(f"Missing model columns after merge: {missing_cols}")

        # Compute mean across model columns (skip NaNs by default)
        merged_df["pred"] = merged_df[self.model_list].mean(axis=1)

        # Optional: round predictions to 3 decimals (uncomment if desired)
        # merged_df["pred"] = merged_df["pred"].round(3)

        # Keep deterministic order
        merged_df = merged_df.sort_values(["DATE", "TIME"]).reset_index(drop=True)
        return merged_df

    def get_final_pred(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build final output DataFrame expected by downstream processes:
        columns: DATE, TIME (zero-padded to 4 chars), POWER OUTPUT (kWh)
        Missing predictions are filled with -999.99 (numeric).
        """
        if "pred" not in data.columns:
            raise KeyError("Input data must contain 'pred' column")

        df_out = data[["DATE", "TIME", "pred"]].rename(columns={"pred": "POWER OUTPUT (kWh)"}).copy()

        # Ensure TIME is a string and pad to 4 characters (e.g., '0030', '1200')
        df_out["TIME"] = df_out["TIME"].astype(str).str.zfill(4)

        # Fill missing numeric predictions with sentinel -999.99
        df_out["POWER OUTPUT (kWh)"] = pd.to_numeric(df_out["POWER OUTPUT (kWh)"], errors="coerce")
        df_out["POWER OUTPUT (kWh)"] = df_out["POWER OUTPUT (kWh)"].fillna(-999.99)

        return df_out

    def run(self) -> pd.DataFrame:
        """
        Convenience method to run the full pipeline and return the final DataFrame.
        """
        merged = self.calculate_models()
        return self.get_final_pred(merged)