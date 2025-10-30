import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
from utils import *
import sys
from dateutil.relativedelta import relativedelta

model_num = str(sys.argv[1])
test_dts = int(sys.argv[2])
lz = str(sys.argv[3])
name = str(sys.argv[4])

test_dts_val = pd.to_datetime(test_dts, format = "%Y%m%d%H", utc = True)
train_dts_val = test_dts_val - relativedelta(months = 4)
train_start_val = train_dts_val - relativedelta(years = 1)

def create_folder(mv, ty, loadzone):
    if not os.path.exists(f'Hyp_model{mv}_{ty}_{loadzone}'):
       os.makedirs(f'Hyp_model{mv}_{ty}_{loadzone}')


create_folder(1, test_dts, lz)

train = pd.read_csv(f"../New_ML_data/load_train_{name}.csv")
print(len(train))
train = train[train['OBS_DATA']!= -999.99]
train = train[~(train == -999).any(axis=1)]
train = train[~(train == -999.99).any(axis=1)]
train = train[train['MISO_FORECAST'] != 0]
train = train[train['OBS_DATA'] != 0].reset_index(drop = True)
print(len(train))
train['Date'] = pd.to_datetime(train['Date'])
train['bias'] = train['MISO_FORECAST'] - train['OBS_DATA']

date_counts = train['Date'].value_counts()
valid_dates = date_counts[date_counts.isin([6])].index
filtered_train = train[train['Date'].isin(valid_dates)].reset_index(drop = True)

train2 = filtered_train[(filtered_train['Date'] > train_start_val) & (filtered_train['Date'] < train_dts_val)].reset_index(drop = True)
train2 = train2[train2['LZ'] == lz].reset_index(drop = True)
train2 = train2.drop('MISO_FORECAST', axis = 1)
print(train2['Date'].min(), train2['Date'].max())


fcst = pd.read_csv(f"../New_ML_data/load_fcst_{name}.csv")
fcst['Date'] = pd.to_datetime(fcst['Date'])
fcst['Init_date'] = pd.to_datetime(fcst['Init_date'])
fcst['Year'] = fcst['Init_date'].dt.year
print(fcst.columns[fcst.isnull().sum() > 0])
test = fcst[(fcst['hour_count'] >= 20) & (fcst['hour_count'] <= 46)].reset_index(drop = True)
print(len(test))
test = test[test['MISO_FORECAST'] != 0]
test = test[test['OBS_DATA'] != 0].reset_index(drop = True)
test = test[~(test == -999).any(axis=1)]
test = test[~(test == -999.99).any(axis=1)]
test['bias'] = test['MISO_FORECAST'] - test['OBS_DATA']

date_counts = test['Date'].value_counts()
valid_dates = date_counts[date_counts.isin([6, 12])].index
filtered_test = test[test['Date'].isin(valid_dates)].reset_index(drop = True)

test2 = filtered_test[(filtered_test['Init_date'] >=train_dts_val) & (filtered_test['Init_date'] < test_dts_val)].reset_index(drop = True)
test2 = test2[test2['LZ'] == lz].reset_index(drop = True)
test2 = test2.drop('MISO_FORECAST', axis = 1)
print(test2['Date'].min(), test2['Date'].max())
print(test2['Init_date'].min(), test2['Init_date'].max())


cols_list = get_cols(f'selected_cols/Model{model_num}_{test_dts}_{lz}.json')['cols'] + ['day_of_week','bias' ]

col_order = read_col_order('col_order', f'Model{model_num}_{test_dts}_{lz}_col_order.txt')

train_data = train2[cols_list]
test_data = test2[cols_list]

train_data = pd.get_dummies(train_data,prefix='day_of_week',columns=['day_of_week'], drop_first = True)
test_data = pd.get_dummies(test_data,prefix='day_of_week',columns=['day_of_week'], drop_first = True)

x_train = train_data.drop('bias',axis = 1)
y_train = np.array(train_data['bias'])

x_fcst = test_data.drop('bias',axis = 1)
y_fcst = np.array(test_data['bias'])

x_train = x_train[col_order]
x_fcst = x_fcst[col_order]

def objective(trial,x_train_val, y_train_val, x_test_val, y_test_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5)
    }
    
    model = XGBRegressor(objective="reg:squarederror", random_state=44, **params)
    model.fit(x_train_val, y_train_val)
    y_pred = model.predict(x_test_val)
    
    return mean_absolute_error(y_test_val, y_pred)

def get_trial_results(study_val):
    # Collect results into a list of dictionaries
    trial_data = []
    for trial in study_val.trials:
        trial_data.append({
            'trial_number': trial.number,
            'MAE': trial.value,  # Objective value (e.g., MAE)
            'params': trial.params  # Hyperparameters
        })

    # Convert to a pandas DataFrame for easy analysis
    trials_df = pd.DataFrame(trial_data)
    return trials_df


study = optuna.create_study(direction="minimize",pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
study.optimize(lambda trial: objective(trial, x_train, y_train, x_fcst, y_fcst), n_trials=200)

filename = f'Hyp_model{model_num}_{test_dts}_{lz}/params_model{model_num}_{test_dts}_{lz}.json'
with open(filename, 'w') as f:
    json.dump(study.best_params, f, indent=4)

gtr = get_trial_results(study)
gtr.to_csv(f'Hyp_model{model_num}_{test_dts}_{lz}/results_model{model_num}_{test_dts}_{lz}.csv')