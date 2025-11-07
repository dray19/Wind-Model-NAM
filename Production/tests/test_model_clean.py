import configparser
import os
from pathlib import Path
import glob

import pandas as pd
import pytest

# Adjust this import to the actual module filename where you put the provided class.
# For these tests we assume the code is in a file named `model_clean.py` and defines
# ModelClean and _read_config at module level.
import Clean
from Clean import ModelClean, _read_config


class DummyCalc:
    """Minimal replacement for wind_math.Calc; returns df unchanged."""
    def __init__(self, df):
        self.df = df
    def wind_values(self):
        return self.df


def write_wind_file(path: Path, dts: str):
    """
    Create a minimal valid wind forecast text file with:
    - at least 4 lines
    - header line 1 with >=4 tokens so init is header1[2]+header1[3]
    - header line 2 containing column tokens such that colnames include 'DATE' and 'TIME'
    - one or more data rows with tokens so create_df will build DATE+TIME correctly
    """
    # header lines
    lines = []
    lines.append("LINE0 ignored\n")
    # header1: tokens index 2 and 3 will be combined for init
    # Use '21' '01010000' -> init becomes '2101010000' (yymmddHHMM)
    lines.append("HDR A 210101 0000 extra\n")
    # header2: keep first two tokens and tokens from index 4 onwards (so include DATE and TIME)
    # tokens: [H1, H2, t2, t3, DATE, TIME, COL3]
    lines.append("H1 H2 X X 210101 0000 COL3\n")
    # data rows: create tokens where row = toks[:2] + toks[3:]
    # Provide a row where toks = [r1, r2, skip, DATE, TIME, value]
    lines.append("r1 r2 skip 210101 0000 42\n")
    # Add a second data row
    lines.append("r1b r2b skip 210101 0100 43\n")

    path.write_text("".join(lines), encoding="utf8")
    return str(path)


def write_var_file(path: Path):
    """
    Create a simple var mapping file to exercise load_var parser.
    The parser expects lines with tokens and uses the last token as the numeric key.
    """
    lines = [
        "VAR_A (something) 100\n",         # tokens -> num='100', name tokens -> ['something']
        "SHORT 200\n",                     # tokens -> num='200', name tokens -> [] -> fallback name
        "COMPLEX NAME (info) EXTRA 300\n", # more tokens; tests tolerant parsing
    ]
    path.write_text("".join(lines), encoding="utf8")
    return str(path)


def write_dawf_csv(path: Path, date_str: str, forecast_value: float):
    """
    Write a minimal dawf csv with Date_UTC and ForecastValue. Date_UTC should match
    the fcst_date produced by create_df for correct merging.
    date_str should be an ISO8601 or similar parseable timestamp (UTC).
    """
    df = pd.DataFrame({"Date_UTC": [date_str], "ForecastValue": [forecast_value]})
    df.to_csv(path, index=False)
    return str(path)


def test__read_config_nonexistent(tmp_path, caplog):
    # Provide a non-existent path to _read_config; should return a ConfigParser and log a warning
    missing = tmp_path / "no_such_config.ini"
    cp = _read_config(missing)
    assert isinstance(cp, configparser.ConfigParser)
    # No exception should be raised; a warning logged
    assert "does not exist" in caplog.text


VAR_CONTENT = """1    10 m   Model Relative U Component    m/s          (  1)
1    10 m   Model Relative V Component    m/s          (  2)
1    80 m   Model Relative U Component    m/s          (  3)
1    80 m   Model Relative V Component    m/s          (  4)
1    80 m   Temperature                   F            (  5)
1    80 m   SL Pressure from P & C        Pa           (  6)
1    80 m   Specific Humidity             kg/kg        (  7)
1 1000 mb   Model Relative U Component    m/s          (  8)
1 1000 mb   Model Relative V Component    m/s          (  9)
1  975 mb   Model Relative U Component    m/s          ( 10)
1  975 mb   Model Relative V Component    m/s          ( 11)
1  950 mb   Model Relative U Component    m/s          ( 12)
1  950 mb   Model Relative V Component    m/s          ( 13)
1  925 mb   Model Relative U Component    m/s          ( 14)
1  925 mb   Model Relative V Component    m/s          ( 15)
1  900 mb   Model Relative U Component    m/s          ( 16)
1  900 mb   Model Relative V Component    m/s          ( 17)
1  875 mb   Model Relative U Component    m/s          ( 18)
1  875 mb   Model Relative V Component    m/s          ( 19)
1  850 mb   Model Relative U Component    m/s          ( 20)
1  850 mb   Model Relative V Component    m/s          ( 21)
1  700 mb   Model Relative U Component    m/s          ( 22)
1  700 mb   Model Relative V Component    m/s          ( 23)
1  500 mb   Model Relative U Component    m/s          ( 24)
1  500 mb   Model Relative V Component    m/s          ( 25)
1 Surface   SL Pressure from P & C        Pa           ( 26)
1     2 m   Temperature                   F            ( 27)
1     2 m   Dew Point Temperature         F            ( 28)
1 Surface   WRF Down Shortwave Radiation  W/m**2       ( 29)
1  950 mb   Turbulence Kinetic Energy     J/kg         ( 30)
1 1000 mb   Turbulence Kinetic Energy     J/kg         ( 31)
1  850 mb   Turbulence Kinetic Energy     J/kg         ( 32)
1 Surface   CIN                           J/kg         ( 33)
1 Surface   CAPE                          J/kg         ( 34)
1 Surface   Precipitable Water            kg/m^2       ( 35)
1 Surface   Snow on Ground                m            ( 36)
1 Surface   PBL Height                    m            ( 37)
1 Surface   Model Relative Wind Gust      m/s          ( 38)
1 Surface   Friction Velocity             m/s          ( 39)
1  500 mb   Actual Relative Humidity      %            ( 40)
1  700 mb   Actual Relative Humidity      %            ( 41)
1  850 mb   Actual Relative Humidity      %            ( 42)
1 1000 mb   Actual Relative Humidity      %            ( 43)
1  500 mb   Vert. Comp. Abs Vorticity     1/s          ( 44)
1  500 mb   Geopotential Height           m            ( 45)
1 1000 mb   Geopotential Height           m            ( 46)
1  850 mb   Geopotential Height           m            ( 47)
1  500 mb   Temperature                   F            ( 48)
1  850 mb   Temperature                   F            ( 49)
1 1000 mb   Temperature                   F            ( 50)
1  700 mb   Temperature                   F            ( 51)
1 Surface   Prop Frzn Precip              %            ( 52)
1 Surface   Freezing Rain                 Int          ( 53)
1 Surface   Snow Cover                    %            ( 54)
1 Surface   Cloud Cover                   %            ( 55)
"""


def test_load_var_realistic_format(tmp_path):
    """
    Write a realistic VAR-style file (the user's example) and assert that
    ModelClean.load_var parses the numeric keys and produces reasonable names.
    """
    var_file = tmp_path / "vars.txt"
    var_file.write_text(VAR_CONTENT, encoding="utf8")

    # Instantiate ModelClean with a harmless dts and no config path to avoid touching disk config
    mc = ModelClean(dts="20250101", config_path=None)

    mapping = mc.load_var(var_file)

    # We expect one mapping per non-empty line in the input; the example has 55 lines labeled 1..55
    assert len(mapping) == 55

    # Check several representative keys and their expected parsed human-readable names
    # The parser builds the name from tokens[1:-2] when there are enough tokens.
    assert mapping["1"] == "10 m Model Relative U Component"
    assert mapping["5"] == "80 m Temperature"
    assert mapping["8"] == "1000 mb Model Relative U Component"
    assert mapping["45"] == "500 mb Geopotential Height"
    assert mapping["47"] == "850 mb Geopotential Height"
    assert mapping["53"] == "Surface Freezing Rain"

    # Sanity check: all keys should be strings of integers
    for k in mapping.keys():
        assert k.isdigit()


# Wind file example line as provided by the user (single data row)
WIND_DATA_LINE = (
    "240601 0600 999.99999.99    -0.570774    0.806146   -0.394670    1.946901  "
    "288.836966  963.246972    0.009399   -0.531465    0.890759   -0.329572    1.677762 "
    "   0.383909    3.316849    1.753413    5.219242    3.201013    7.741777    3.880904    "
    "9.100786    4.009482    9.330458    4.605770    5.996069    7.623700    4.444892  972.375194 "
    "288.467101  286.430421    0.000000    0.043994    0.069750    0.021093  -15.381288   34.798274 "
    "-999.000002 -999.000002  184.837314 -999.000002    0.121351   49.693995   73.307795   75.632670 "
    "  82.935561    0.000000 5750.443242  138.513593 1513.850190  259.397701  284.092714  290.903618 "
    "275.762423 -999.000002    0.000000 -999.000002   78.221786"
)

# Build the long header row listing DATE TIME I J ... 1..55 (single-space separated tokens)
HEADER_COLS = "DATE  TIME    I     J " + " ".join(str(i) for i in range(1, 56))

def write_realistic_wind_file(path: Path):
    """
    Create a wind file using:
     - a placeholder first line,
     - a header1 where tokens[2] and tokens[3] join to form init (use 240601 and 0600),
     - a header2 with DATE/TIME/I/J/1..55,
     - one data line matching user sample.
    """
    lines = []
    lines.append("IGNORED LINE 0\n")
    # header1: want header1[2] == '240601', header1[3] == '0600' to create init '2406010600'
    lines.append("HDR A 240601 0600 extra\n")
    lines.append(HEADER_COLS + "\n")
    lines.append(WIND_DATA_LINE + "\n")
    path.write_text("".join(lines), encoding="utf8")
    return str(path)


def test_wind_data_parsing_realistic(tmp_path):
    fcst = tmp_path / "forecast_realistic.txt"
    write_realistic_wind_file(fcst)

    mc = ModelClean(dts="24060106", config_path=None)
    data_values, colnames, init = mc.wind_data(str(fcst))

    # init should be header1[2]+header1[3] -> '2406010600'
    assert init == "2406010600"
    # colnames should include 'DATE' and 'TIME' (HEADER_COLS begins with those)
    assert "DATE" in colnames and "TIME" in colnames
    # data_values should have one row (our single data line)
    assert isinstance(data_values, list)
    assert len(data_values) == 1
    # first row should include the DATE and TIME tokens at the start (split by whitespace)
    first_row = data_values[0]
    assert first_row[0].startswith("240601")
    assert first_row[1].startswith("0600")


def test_create_df_from_realistic_wind_and_merge(tmp_path, monkeypatch):
    # Replace Calc with DummyCalc so create_df doesn't require external implementation
    monkeypatch.setattr(Clean, "Calc", DummyCalc)

    # Write files and config as needed to run a small end-to-end flow of create_df + clean + merge
    fcst = tmp_path / "forecast_realistic.txt"
    write_realistic_wind_file(fcst)

    var_file = tmp_path / "vars.txt"
    write_var_file(var_file)

    # Make a tiny DAWF csv that matches the fcst_date produced by create_df
    dawf_dir = tmp_path / "dawf"
    dawf_dir.mkdir()
    # create a DAWF csv file named by ModelClean.load_dawf expectation: dts[:-2] + '.csv'
    # our ModelClean dts for this test will be '24060106', so load_dawf looks for '240601.csv'
    dawf_csv = dawf_dir / "240601.csv"
    # fcst_date from create_df will be parsed from DATE+TIME '240601' '0600' -> two-digit year 24 -> 2024-06-01T06:00
    dawf_df = pd.DataFrame({
        "Date_UTC": ["2024-06-01T06:00:00Z"],
        "ForecastValue": [999.9]
    })
    dawf_df.to_csv(dawf_csv, index=False)

    # Create a config.ini pointing at our test directories/files
    cfg = tmp_path / "config.ini"
    cfg.write_text(
        f"[paths]\nfcst_data = {str(tmp_path)}\nvar_path = {str(var_file)}\ndawf_data = {str(dawf_dir)}\n",
        encoding="utf8"
    )

    monkeypatch.setattr(Clean, "Calc", DummyCalc)

    mc = ModelClean(dts="24060106", config_path=cfg)
    # Force glob to find our file by naming it with the dts in the filename (forecast_realistic contains no dts,
    # so create another copy with dts embedded to match mc.run() glob pattern)
    fcst_with_dts = tmp_path / f"forecast_{mc.dts}.txt"
    fcst_with_dts.write_text((Path(fcst).read_text()), encoding="utf8")

    result = mc.run()

    # Basic assertions on the resulting DataFrame from run()
    assert isinstance(result, pd.DataFrame)
    assert "DAWF" in result.columns
    assert "Date_UTC" not in result.columns
    assert result["Look_Ahead"].min() >= 0
    assert result["Look_Ahead"].max() <= 46