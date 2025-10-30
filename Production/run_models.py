#!/usr/bin/env python3
"""
clean_predict.py

Refactored entry-point script for running individual models and a mean model.
- Uses argparse for command-line parsing
- Adds logging and basic error handling
- Validates configuration entries before use
- Strips and filters model names from the config
"""

from __future__ import annotations

import argparse
import configparser
import logging
import os
import sys
from pathlib import Path
from typing import List

from Model import ModelPredictor
from Fcst_out import Output
from Get_mean import MeanModel


DEFAULT_CONFIG_PATH = "/mnt/trade05_data/wind_v5_new_code/Production/config.ini"


def setup_logging(level: str | int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str) -> configparser.ConfigParser:
    cfg_path = Path(os.path.expanduser(path))
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    config = configparser.ConfigParser()
    config.read(cfg_path)
    return config


def get_model_list(config: configparser.ConfigParser, mean_type: str) -> List[str]:
    if "Mean_models" not in config:
        raise KeyError("Missing 'Mean_models' section in config file.")
    if mean_type not in config["Mean_models"]:
        raise KeyError(f"Mean type '{mean_type}' not found in 'Mean_models' section.")
    raw = config["Mean_models"][mean_type]
    # Split on comma, strip whitespace, drop empties
    models = [m.strip() for m in raw.split(",") if m.strip()]
    return models


def run_model(dts: str, model_key: str, model_cfg: configparser.SectionProxy) -> None:
    model_path = model_cfg.get("model_path", "")
    model_name = model_cfg.get("model_name", model_key)
    logging.info("Running model '%s' (name=%s path=%s) for date %s", model_key, model_name, model_path, dts)
    try:
        df_pred = ModelPredictor(dts, model_path, model_name).run()
        Output(dts, model_key, model_name, df_pred).Create()
        logging.info("Finished model '%s'", model_key)
    except Exception as exc:  # narrow/adjust exceptions if you know specific ones
        logging.exception("Error running model '%s': %s", model_key, exc)


def run_mean(dts: str, mean_type: str, model_keys: List[str]) -> None:
    if not model_keys:
        logging.warning("No models provided for mean '%s' — skipping MeanModel.", mean_type)
        return
    logging.info("Computing mean '%s' from models: %s", mean_type, ", ".join(model_keys))
    try:
        df_mean = MeanModel(dts, model_keys).run()
        Output(dts, mean_type, mean_type, df_mean).Create()
        logging.info("Mean '%s' created", mean_type)
    except Exception as exc:
        logging.exception("Error computing mean '%s': %s", mean_type, exc)


def main(argv: List[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Run model predictions and generate outputs.")
    parser.add_argument("dts", help="Date string argument passed to the models (e.g. YYYYMMDD or similar).")
    parser.add_argument("mean_type", help="Key in the [Mean_models] section of config indicating the mean model to compute.")
    parser.add_argument(
        "--config",
        "-c",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config.ini (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    args = parser.parse_args(argv)

    log_level = args.log_level.upper()
    setup_logging(getattr(logging, log_level, logging.INFO))

    try:
        config = load_config(args.config)
    except Exception as exc:
        logging.exception("Failed to load config: %s", exc)
        return 2

    try:
        model_keys = get_model_list(config, args.mean_type)
    except Exception as exc:
        logging.exception("Configuration error: %s", exc)
        return 3

    # Run each configured model if its section exists
    valid_model_keys: List[str] = []
    for key in model_keys:
        if key not in config:
            logging.warning("Model key '%s' not present as a section in config — skipping.", key)
            continue
        valid_model_keys.append(key)
        run_model(args.dts, key, config[key])

    # Compute and output the mean model based on the (valid) model keys
    run_mean(args.dts, args.mean_type, valid_model_keys)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())