import os
from pathlib import Path

import pandas as pd
import pytest

import Get_mean

from Get_mean import MeanModel


def make_model_csv(base: Path, model_name: str, dts: str, df: pd.DataFrame) -> Path:
    """
    Create directory structure base/model_name/dts/dts.csv and write df as CSV.
    Returns the file path.
    """
    model_dir = base / model_name / dts
    model_dir.mkdir(parents=True, exist_ok=True)
    file_path = model_dir / f"{dts}.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_init_empty_model_list_raises():
    with pytest.raises(ValueError):
        MeanModel(dts="20250101", model_list=[], out_path=".")


def test_calculate_models_and_run(tmp_path):
    dts = "20250101"
    out_path = tmp_path

    # Create two model CSVs with matching DATE and TIME (TIME without zero padding)
    df1 = pd.DataFrame({
        "DATE": ["2025-01-01", "2025-01-01"],
        "TIME": ["30", "1200"],
        "POWER OUTPUT (kWh)": [10.0, 20.0],
    })
    df2 = pd.DataFrame({
        "DATE": ["2025-01-01", "2025-01-01"],
        "TIME": ["30", "1200"],
        "POWER OUTPUT (kWh)": [30.0, 40.0],
    })

    make_model_csv(out_path, "m1", dts, df1)
    make_model_csv(out_path, "m2", dts, df2)

    mm = MeanModel(dts=dts, model_list=["m1", "m2"], out_path=out_path)
    merged = mm.calculate_models()

    # merged should contain columns DATE, TIME, m1, m2, pred
    assert set(["DATE", "TIME", "m1", "m2", "pred"]).issubset(set(merged.columns))

    # Check mean values: (10+30)/2=20, (20+40)/2=30
    assert merged.loc[merged["TIME"] == "30", "pred"].iloc[0] == pytest.approx(20.0)
    assert merged.loc[merged["TIME"] == "1200", "pred"].iloc[0] == pytest.approx(30.0)

    # run() returns final DataFrame with padded TIME and column POWER OUTPUT (kWh)
    final = mm.run()
    assert "POWER OUTPUT (kWh)" in final.columns
    # TIME should be zero-padded to 4 characters
    times = set(final["TIME"].tolist())
    assert "0030" in times and "1200" in times

    # POWER OUTPUT (kWh) should match the pred values
    row_0030 = final[final["TIME"] == "0030"].iloc[0]
    assert row_0030["POWER OUTPUT (kWh)"] == pytest.approx(20.0)
    row_1200 = final[final["TIME"] == "1200"].iloc[0]
    assert row_1200["POWER OUTPUT (kWh)"] == pytest.approx(30.0)



def test_missing_power_column_raises(tmp_path):
    dts = "20250103"
    out_path = tmp_path

    # Create a CSV that does not contain the required POWER OUTPUT column
    df_bad = pd.DataFrame({
        "DATE": ["2025-01-03"],
        "TIME": ["30"],
        "WRONG_COLUMN": [1.0],
    })
    make_model_csv(out_path, "m1", dts, df_bad)

    mm = MeanModel(dts=dts, model_list=["m1"], out_path=out_path)
    with pytest.raises(KeyError):
        mm.calculate_models()


def test_get_final_pred_padding_and_fill():
    # Build a DataFrame that resembles the merged output but with missing pred values
    df = pd.DataFrame({
        "DATE": ["2025-01-04", "2025-01-04"],
        "TIME": ["30", "930"],  # '30' should become '0030', '930' -> '0930'
        "pred": [None, 12.345],  # None should be converted to -999.99
    })

    mm = MeanModel(dts="20250104", model_list=["m1"], out_path=".")  # out_path not used here
    final = mm.get_final_pred(df)

    assert final.loc[final["TIME"] == "0030", "POWER OUTPUT (kWh)"].iloc[0] == pytest.approx(-999.99)
    assert final.loc[final["TIME"] == "0930", "POWER OUTPUT (kWh)"].iloc[0] == pytest.approx(12.345)
