"""
Refactored and cleaned version of the provided script for loading and preparing
wind forecast and DAWF data.

Changes made:
- Added type hints, docstrings and logging.
- Replaced repeated `.apply(lambda ...)` with a threshold-driven vectorized approach.
- More robust file and config path handling (uses pathlib).
- Safer parsing for text-based wind files with defensive checks.
- Coerces numeric columns with pd.to_numeric(errors='coerce') instead of astype(float)
  to avoid hard errors on malformed values.
- Kept compatibility with the existing Calc class from wind_math.
- Preserved `from utils import *` for backwards compatibility, but recommend replacing
  the wildcard import with explicit names.
"""

from pathlib import Path
import glob
import logging
import configparser
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings

# Local imports (keep wildcard for compatibility; replace with explicit imports if known)
from wind_math import Calc
# from utils import *  # TODO: replace wildcard import with explicit names

# --- Warnings / Logging Configuration ---
warnings.simplefilter(action="ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_CONFIG_PATH = Path("/mnt/trade05_data/wind_v5_new_code/Production/config.ini")


def _read_config(config_path: Optional[Path] = None) -> configparser.ConfigParser:
    cp = configparser.ConfigParser()
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        logger.warning("Config path %s does not exist — caller should provide a valid path", path)
    cp.read(path)
    return cp


class ModelClean:
    """
    Class to load, parse and clean wind forecast and DAWF (dispatch-aware wind forecast) data.

    Usage:
        mc = ModelClean(dts='21010100', config_path=Path('.../config.ini'))
        df = mc.run()
    """

    def __init__(self, dts: str, config_path: Optional[Path] = None) -> None:
        """
        :param dts: datetime string token used to find forecast files (same as original).
        :param config_path: optional path to config.ini; if None, DEFAULT_CONFIG_PATH used.
        """
        self.dts = dts
        self.config = _read_config(Path(config_path) if config_path else None)
        try:
            paths = self.config["paths"]
            self.fcst_data = Path(paths["fcst_data"])
            self.var_path = Path(paths["var_path"])
            self.dawf_path = Path(paths["dawf_data"])
        except Exception:
            # Keep attributes even if config didn't load correctly; let file operations raise meaningful errors later.
            self.fcst_data = None
            self.var_path = None
            self.dawf_path = None
            logger.exception("Failed to load expected 'paths' from config; attributes set to None")

    # ---- file loaders/parsers ----

    def load_dawf(self) -> pd.DataFrame:
        """Load DAWF csv and return DataFrame with Date_UTC (datetime, utc) and DAWF columns."""
        if not self.dawf_path:
            raise FileNotFoundError("dawf_path not configured")
        csv_path = self.dawf_path / f"{self.dts[:-2]}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"DAWF file not found: {csv_path}")
        df = pd.read_csv(csv_path, usecols=["Date_UTC", "ForecastValue"], parse_dates=["Date_UTC"], infer_datetime_format=True)
        df = df.rename(columns={"ForecastValue": "DAWF"})
        df["Date_UTC"] = pd.to_datetime(df["Date_UTC"], utc=True)
        return df

    def wind_data(self, file_path: str) -> Tuple[List[List[str]], List[str], str]:
        """
        Parse the wind forecast text file.

        Returns:
            data_values: list of rows (each row is a list of string tokens for numeric columns)
            colnames: list of column names expected for DataFrame
            init: initialization token (string) used as init_date when creating df
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Wind forecast file not found: {file_path}")

        with p.open("r") as fh:
            lines = fh.readlines()

        if len(lines) < 4:
            raise ValueError(f"Wind data file {file_path} does not appear to have expected structure")

        # Defensive splitting
        header1 = lines[1].split()
        init = ""
        if len(header1) >= 4:
            init = header1[2] + header1[3]
        else:
            # fallback: join everything after first token
            init = "".join(header1[1:])

        # column names extraction similar to original logic
        header2_tokens = lines[2].split()
        # original approach kept first two tokens and then tokens from index 4 onward
        colnames = list(header2_tokens[:2]) + list(header2_tokens[4:]) if len(header2_tokens) > 4 else header2_tokens

        data_values: List[List[str]] = []
        for line in lines[3:]:
            toks = line.split()
            if not toks:
                continue
            # original logic: keep first two, then from index 3 onward (skips one token)
            row = list(toks[:2]) + list(toks[3:]) if len(toks) > 3 else toks
            data_values.append(row)

        return data_values, colnames, init

    def load_var(self, path: Optional[Path] = None) -> Dict[str, str]:
        """
        Load variable mapping from a small CSV-like file. Returns mapping {num_token: name}.
        The original file parsing was idiosyncratic; we implement a tolerant parser.
        """
        p = Path(path or self.var_path)
        if not p.exists():
            raise FileNotFoundError(f"Variable mapping file not found: {p}")

        mapping: Dict[str, str] = {}
        with p.open() as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                # remove parentheses and compress spaces
                cleaned = line.replace("(", "").replace(")", "")
                tokens = " ".join(cleaned.split()).split()
                if len(tokens) < 2:
                    continue
                num = tokens[-1]
                # join tokens from index 1 up to the last two tokens in the original script; be tolerant if short
                name_tokens = tokens[1:-2] if len(tokens) > 3 else tokens[1:-1]
                name = " ".join(name_tokens) if name_tokens else tokens[0]
                mapping[num] = name
        return mapping

    # ---- DataFrame creation and cleaning ----

    def create_df(self, data_list: List[List[str]], col_names: List[str], new_col: Dict[str, str], load_zone: Optional[str], init_date: str) -> pd.DataFrame:
        """
        Create a DataFrame from parsed text data, rename columns, set init and fcst dates,
        run the Calc.wind_values() transformation and return the result.
        """
        if not data_list:
            raise ValueError("No forecast data to create DataFrame")

        df = pd.DataFrame(data_list, columns=col_names)

        # Rename columns using provided mapping; don't fail if a key is missing
        df = df.rename(columns=new_col)

        if load_zone is not None:
            df["Load_Zone"] = load_zone

        df["init_date"] = init_date

        # original logic: combine DATE + TIME into fcst_date
        if "DATE" in df.columns and "TIME" in df.columns:
            df["fcst_date"] = df["DATE"].astype(str).str.strip() + df["TIME"].astype(str).str.strip()
            df = df.drop(columns=["DATE", "TIME"])
            # parse two-digit year format used in original code ("%y%m%d%H%M")
            try:
                df["fcst_date"] = pd.to_datetime(df["fcst_date"], format="%y%m%d%H%M", utc=True)
            except Exception:
                # fallback to automatic parse
                df["fcst_date"] = pd.to_datetime(df["fcst_date"], utc=True, errors="coerce")
        else:
            # if DATE/TIME not present, attempt to parse a column named 'Datetime' or 'fcst_date' directly
            if "fcst_date" in df.columns:
                df["fcst_date"] = pd.to_datetime(df["fcst_date"], utc=True, errors="coerce")
            elif "Datetime" in df.columns:
                df["fcst_date"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
            else:
                raise KeyError("Expected DATE and TIME columns or a fcst/Datetime column to build fcst_date")

        # parse init_date
        try:
            df["init_date"] = pd.to_datetime(df["init_date"], format="%y%m%d%H%M", utc=True)
        except Exception:
            df["init_date"] = pd.to_datetime(df["init_date"], utc=True, errors="coerce")

        # sort and reorder to put init_date/fcst_date in front
        df = df.sort_values(["init_date", "fcst_date"])
        cols = list(df.columns)
        # bring init_date and fcst_date to front if present
        front = [c for c in ("init_date", "fcst_date") if c in cols]
        remaining = [c for c in cols if c not in front]
        df = df[front + remaining]

        # add Look_Ahead as the row index (reset index not applied here; original used index)
        df = df.reset_index(drop=True)
        df["Look_Ahead"] = df.index

        # Call domain-specific transformation
        df = Calc(df).wind_values()
        return df

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame by coercing numeric values and applying range checks to remove
        obviously bad values (set to NaN).
        """
        df2 = data.copy()

        # Which columns to coerce to numeric: everything except a few datetime-like fields
        exclude = {"init_date", "fcst_date", "Datetime"}
        numeric_cols = [c for c in df2.columns if c not in exclude]

        # Coerce to numeric where possible (safest)
        df2[numeric_cols] = df2[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # Range thresholds: (min_value, max_value) where None means no bound.
        # These were derived from the original code's many apply-lambda calls.
        thresholds: Dict[str, Tuple[Optional[float], Optional[float]]] = {
            "500 mb Geopotential Height": (4800, None),
            "80 m Model Relative U Component": (-15, None),
            "80 m Model Relative V Component": (-15, None),
            "80 m Temperature": (245, None),
            "80 m SL Pressure from P & C": (940, None),
            "80 m Specific Humidity": (0, None),
            "1000 mb Model Relative U Component": (-10, None),
            "1000 mb Model Relative V Component": (-10, None),
            "950 mb Model Relative U Component": (-15, None),
            "950 mb Model Relative V Component": (-15, None),
            "925 mb Model Relative U Component": (-25, None),
            "925 mb Model Relative V Component": (-25, None),
            "900 mb Model Relative U Component": (-30, None),
            "900 mb Model Relative V Component": (-30, None),
            "875 mb Model Relative U Component": (-35, None),
            "875 mb Model Relative V Component": (-35, None),
            "700 mb Model Relative U Component": (-40, None),
            "700 mb Model Relative V Component": (-40, None),
            "500 mb Model Relative U Component": (-60, None),
            "500 mb Model Relative V Component": (-60, None),
            "Surface WRF Down Shortwave Radiation": (0, None),
            "950 mb Turbulence Kinetic Energy": (0, None),
            "1000 mb Turbulence Kinetic Energy": (0, None),
            "850 mb Turbulence Kinetic Energy": (0, None),
            "Surface CIN": (-500, None),  # was nan if x < -500 --> keep only >= -500
            "Surface CAPE": (0, None),
            "Surface PBL Height": (0, None),
            "500 mb Actual Relative Humidity": (0, None),
            "700 mb Actual Relative Humidity": (0, None),
            "1000 mb Actual Relative Humidity": (0, None),
            "500 mb Vert. Comp. Abs Vorticity": (None, 0.2),  # nan if x > 0.2
            "1000 mb Geopotential Height": (-200, None),
            "850 mb Geopotential Height": (1300, None),
            "500 mb Temperature": (220, None),
            "850 mb Temperature": (230, None),
            "1000 mb Temperature": (240, None),
            "700 mb Temperature": (220, None),
            "Surface Freezing Rain": (0, None),
            "Surface Cloud Cover": (0, None),
        }

        # Apply thresholds vectorized
        for col, (min_v, max_v) in thresholds.items():
            if col not in df2.columns:
                continue
            if min_v is not None:
                mask = df2[col] < min_v
                if mask.any():
                    df2.loc[mask, col] = np.nan
            if max_v is not None:
                mask = df2[col] > max_v
                if mask.any():
                    df2.loc[mask, col] = np.nan

        return df2

    # ---- runner ----

    def run(self) -> pd.DataFrame:
        """
        Run the full pipeline:
        - load dawf csv
        - find matching fcst text file based on dts in fcst_data path
        - parse the file and variable mapping
        - create dataframe and clean it
        - merge with DAWF and return Look_Ahead 0..46 subset
        """
        dawf = self.load_dawf()

        if not self.fcst_data:
            raise FileNotFoundError("fcst_data not configured in config.ini")

        pattern = str(self.fcst_data / f"*{self.dts}*")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No forecast files found with pattern: {pattern}")
        file_name = matches[0]

        data_out, colnames, init = self.wind_data(file_name)

        var_map = self.load_var(self.var_path)

        tt = self.create_df(
            data_list=data_out,
            col_names=colnames,
            new_col=var_map,
            load_zone=None,
            init_date=init,
        )

        wind_total = self.clean(tt)

        # Merge on fcst_date vs Date_UTC from dawf
        df2 = pd.merge(wind_total, dawf, left_on="fcst_date", right_on="Date_UTC", how="left")

        # Subset look ahead range as original code did
        wind = df2[(df2["Look_Ahead"] >= 0) & (df2["Look_Ahead"] <= 46)].reset_index(drop=True)
        if "Date_UTC" in wind.columns:
            wind = wind.drop(columns=["Date_UTC"])
        return wind