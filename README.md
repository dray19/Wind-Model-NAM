Wind-Model-NAM Code Structure

├── DAWF/                       # Day-Ahead Wind Forecast data
│   ├── dawf_data/              # Raw or processed DAWF input files
├── fcstout/                    # Directory for forecast output files
├── model_*                     # Multiple model configurations
│   ├── col_order/              # Column ordering definitions
│   ├── Create_ML.py            # Data preparation and training pipeline
│   ├── ML_model.py             # Model architecture and hyperparameter setup
│   ├── run.sh                  # Shell script to train or test this model
│   ├── selected_cols/          # JSON column selections
│   │   └── cols_all.json
│   ├── set_cols.py             # Script to define feature sets
│   └── utils.py                # Utility functions for model operations
├── New_ML_data/                # Updated machine learning data inputs
├── pro_data/                   # Processed/cleaned datasets for modeling
├── Production/                 # Production forecast pipeline
│   ├── Clean.py                # Cleans and formats raw forecast outputs
│   ├── config.ini              # Configuration file for data paths and settings
│   ├── Fcst_out.py             # Forecast output handler
│   ├── Get_mean.py             # Aggregation of forecast model outputs
│   ├── Model.py                # Forecast model execution and logic
│   ├── run_fcst.sh             # Shell script for forecast-only execution
│   ├── run_models.py           # Runs all production models sequentially
│   ├── utils.py                # Shared production utilities
│   └── wind_math.py            # Wind-specific math and metric functions
├── run_loop.sh                 # Automated training/forecast loop
├── run_train.sh                # Script for one-time or monthly training
└── README.md                   # Project documentation


