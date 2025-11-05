# **NAM Wind Forecasting Models**

This repository contains a suite of machine-learning models and production utilities for forecasting wind  generation using NAM (North American Mesoscale) weather model data and historical wind data.  
It includes multiple training pipelines, feature configurations, and production forecasting scripts designed for MLOps workflows.

---

## **📁 Repository Structure**

``` 
Wind-Model-NAM/
├── DAWF/                          # Day-Ahead Wind Forecast (DAWF) input data
│   └── dawf_data/                 # Raw or pre-processed DAWF files

├── fcstout/                       # Forecast output files (model results)

├── model_*/                       # Individual model configurations
│   ├── col_order/                 # Column ordering definitions
│   ├── selected_cols/             # Selected feature JSONs
│   │   └── cols_all.json
│   ├── Create_ML.py               # Data preparation + training pipeline
│   ├── ML_model.py                # Model setup & hyperparameter definitions
│   ├── set_cols.py                # Defines model feature sets
│   ├── utils.py                   # Helper utilities for training
│   └── run.sh                     # Shell script to train/test this model

├── New_ML_data/                   # Updated ML data inputs for retraining
├── pro_data/                      # Forecast data for live predictions 

├── Production/                    # Production forecast execution pipeline
│   ├── Clean.py                   # Cleans & formats model forecast outputs
│   ├── config.ini                 # Paths, credentials, and global settings
│   ├── Fcst_out.py                # Handles forecast output formatting
│   ├── Get_mean.py                # Aggregates model ensemble forecasts
│   ├── Model.py                   # Core production model logic
│   ├── run_fcst.sh                # Runs production forecast only
│   ├── run_models.py              # Runs all production models sequentially
│   ├── utils.py                   # Shared production helper functions
│   └── wind_math.py               # Wind-specific metrics & calculations

├── run_loop.sh                    # Automated monthly training + forecasting loop
├── run_train.sh                   # Manual training execution (single or batch)
└── README.md                      # Project documentation
```


