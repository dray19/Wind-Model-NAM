from pathlib import Path
from typing import List, Optional

import pandas as pd


__all__ = [
    "one_hot_encode_column",
    "one_hot_encode_LZ",
    "one_hot_encode_day_of_week",
    "read_col_order",
]


def one_hot_encode_column(
    data: pd.DataFrame,
    column: str,
    categories: List[str],
    prefix: Optional[str] = None,
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    One-hot encode a single column using the provided categories.

    Ensures the resulting dataframe always contains columns for all categories
    (in the order given) and fills missing categories with zeros.

    Parameters
    - data: pd.DataFrame
    - column: name of the column in `data` to encode
    - categories: list of category values (order will be used for the columns)
    - prefix: optional prefix for dummy column names; if None, uses column name
    - drop_original: if True, drop the original `column` from the returned dataframe

    Returns
    - pd.DataFrame: original dataframe with one-hot columns appended
    """
    if column not in data.columns:
        raise KeyError(f"Column '{column}' not found in the DataFrame.")

    if not isinstance(categories, (list, tuple)):
        raise TypeError("categories must be a list or tuple of category labels.")

    # Use Categorical to guarantee dummy columns for all categories
    cat_series = pd.Categorical(data[column], categories=categories)
    dummies = pd.get_dummies(cat_series, prefix=prefix or column)

    # Ensure columns are present in the specified order and filled with zeros when missing
    expected_cols = [f"{prefix or column}_{c}" for c in categories]
    dummies = dummies.reindex(columns=expected_cols, fill_value=0)

    result = pd.concat([data.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    if drop_original:
        result = result.drop(columns=[column])

    return result


def one_hot_encode_LZ(data: pd.DataFrame, drop_original: bool = False) -> pd.DataFrame:
    """
    One-hot encode the 'LZ' (load zone) column and append columns named
    LZ_<category> for each category in the default list.

    Default categories: ['L89', 'L35', 'L27', 'L06', 'L04', 'L01']

    Parameters
    - data: pd.DataFrame with column 'LZ'
    - drop_original: if True, drop the original 'LZ' column

    Returns
    - pd.DataFrame
    """
    lz = ["L89", "L35", "L27", "L06", "L04", "L01"]
    return one_hot_encode_column(data, column="LZ", categories=lz, prefix="LZ", drop_original=drop_original)


def one_hot_encode_day_of_week(data: pd.DataFrame, drop_original: bool = False) -> pd.DataFrame:
    """
    One-hot encode the 'day_of_week' column and append columns named
    day_of_week_<weekday> for each weekday in the default list.

    Default categories: ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    Parameters
    - data: pd.DataFrame with column 'day_of_week'
    - drop_original: if True, drop the original 'day_of_week' column

    Returns
    - pd.DataFrame
    """
    dw = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return one_hot_encode_column(data, column="day_of_week", categories=dw, prefix="day_of_week", drop_original=drop_original)


def read_col_order(path: str, name: str) -> List[str]:
    """
    Read a file containing column names (one per line) and return a cleaned list.

    - Ignores empty lines and lines that start with '#' (comments).
    - Raises FileNotFoundError if the file does not exist.

    Parameters
    - path: directory path or parent path containing the file
    - name: file name (or relative path under `path`)

    Returns
    - List[str]: cleaned column names
    """
    file_path = Path(path) / name
    if not file_path.exists():
        raise FileNotFoundError(f"Column order file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]

    # Remove blank lines and comment lines
    cols = [ln for ln in lines if ln and not ln.startswith("#")]
    return cols

    