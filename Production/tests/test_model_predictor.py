import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock
import types
import importlib

import Model
from Model import ModelPredictor



def make_selected_cols_file(tmp_path: Path, cols):
    sel_dir = tmp_path / "selected_cols"
    sel_dir.mkdir()
    file_path = sel_dir / "cols_all.json"
    file_path.write_text(json.dumps({"cols": cols}), encoding="utf-8")
    return file_path


def test_load_selected_columns_success(tmp_path):
    cols = ["a", "bias", "b"]
    make_selected_cols_file(tmp_path, cols)

    mp = ModelPredictor(dts=None, model_path=tmp_path, model_year=2020)
    loaded = mp._load_selected_columns()
    assert loaded == cols


def test_load_model_missing_file(tmp_path):
    mp = ModelPredictor(dts=None, model_path=tmp_path, model_year=1999)
    with pytest.raises(FileNotFoundError):
        mp._load_model()


def test_load_model_success(monkeypatch, tmp_path):
    # Create the expected model file path
    model_dir = tmp_path / "Model1_training_2001_all"
    model_dir.mkdir()
    model_file = model_dir / "model1_2001.ubj"
    model_file.write_text("", encoding="utf-8")

    # Create a fake booster and assert load_model is called with the file path
    fake_booster = Mock()
    fake_booster.load_model = Mock()

    # Patch the name `xgb` in the module that contains XGBModel so xgb.Booster() returns our fake_booster
    monkeypatch.setattr(
        Model,
        "xgb",
        types.SimpleNamespace(Booster=lambda: fake_booster),
    )

    mp = ModelPredictor(dts=None, model_path=tmp_path, model_year=2001)
    model = mp._load_model()
    # The method should return the fake booster and call load_model with the expected path
    assert model is fake_booster
    fake_booster.load_model.assert_called_once_with(str(model_file))


def test_prepare_features_success(monkeypatch):
    # Data with columns a,b
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    selected_cols = ["a", "b", "bias"]

    # read_col_order should return the order we expect
    def fake_read_col_order(model_path, relpath):
        assert "Model1_training_2005_all" in relpath
        return ["a", "b"]

    monkeypatch.setattr(Model, "read_col_order", fake_read_col_order)

    mp = ModelPredictor(dts=None, model_path=".", model_year=2005)
    out = mp._prepare_features(df, selected_cols)
    # Check ordering and dtypes
    assert list(out.columns) == ["a", "b"]
    assert out.dtypes["a"] == float or out["a"].dtype == np.float64
    assert out.shape == (2, 2)




def test_prepare_features_invalid_col_order(monkeypatch):
    df = pd.DataFrame({"a": [1], "b": [2]})
    selected_cols = ["a", "b"]

    monkeypatch.setattr(Model, "read_col_order", lambda p, r: "not-a-list")
    mp = ModelPredictor(dts=None, model_path=".", model_year=2010)
    with pytest.raises(ValueError):
        mp._prepare_features(df, selected_cols)


def test_assemble_output_success_and_fillna():
    # create dataframe with fcst_date and DAWF values
    dates = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"])
    df = pd.DataFrame({"fcst_date": dates, "DAWF": [10.0, np.nan]})

    preds = np.array([1.5, 2.0])  # pred_bias

    mp = ModelPredictor(dts=None, model_path=".", model_year=2000)
    out = mp._assemble_output(df, preds)

    # Check columns and index mapping
    assert list(out.columns) == ["DATE", "TIME", "POWER OUTPUT (kWh)"]
    assert out.shape[0] == 2

    # First row: numeric difference
    first_val = out.iloc[0]["POWER OUTPUT (kWh)"]
    assert float(first_val) == pytest.approx(10.0 - 1.5)

    # Second row had NaN result and should be replaced with the sentinel string
    second_val = out.iloc[1]["POWER OUTPUT (kWh)"]
    assert second_val == "-999.99"


def test_run_integration(monkeypatch, tmp_path):
    # Prepare cleaned data returned by ModelClean
    dates = pd.to_datetime(["2021-12-01 00:00", "2021-12-01 01:00"])
    cleaned = pd.DataFrame({"fcst_date": dates, "DAWF": [100.0, 150.0], "a": [1.0, 2.0]})

    class DummyModelClean:
        def __init__(self, dts):
            self.dts = dts

        def run(self):
            return cleaned

    monkeypatch.setattr(Model, "ModelClean", DummyModelClean)

    # Create selected_cols file that references 'a' and 'bias'
    make_selected_cols_file(tmp_path, ["a", "bias"])

    # Make read_col_order return ['a']
    monkeypatch.setattr(Model, "read_col_order", lambda p, r: ["a"])

    # Monkeypatch xgb.DMatrix to be a passthrough (we don't need real DMatrix)
    monkeypatch.setattr(Model, "xgb", types.SimpleNamespace(DMatrix=lambda features: features))

    # Create a dummy model with a predict method
    class DummyBooster:
        def predict(self, dmatrix):
            # return pred_bias for each row (length must match rows)
            return np.array([10.0, 20.0])

    # Monkeypatch ModelPredictor._load_model to return our dummy
    monkeypatch.setattr(ModelPredictor, "_load_model", lambda self: DummyBooster())

    mp = ModelPredictor(dts={"irrelevant": True}, model_path=tmp_path, model_year=2021)
    out = mp.run()

    # Should have two rows and expected columns
    assert out.shape == (2, 3)
    assert list(out.columns) == ["DATE", "TIME", "POWER OUTPUT (kWh)"]

    # Check calculation: POWER OUTPUT (kWh) = DAWF - pred_bias
    expected0 = 100.0 - 10.0
    expected1 = 150.0 - 20.0
    assert float(out.iloc[0]["POWER OUTPUT (kWh)"]) == pytest.approx(expected0)
    assert float(out.iloc[1]["POWER OUTPUT (kWh)"]) == pytest.approx(expected1)