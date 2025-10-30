import pandas as pd
import os
import numpy as np
from utils import XGB_model_check
from dateutil.relativedelta import relativedelta
import warnings
import sys
from utils import *
# Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json

model_num = str(sys.argv[1])
test_dts = int(sys.argv[2])
lz = str(sys.argv[3])
name = str(sys.argv[4])

test_dts_val = pd.to_datetime(test_dts, format = "%Y%m%d%H", utc = True)
train_dts_val = test_dts_val - relativedelta(months = 4)
train_start_val = train_dts_val - relativedelta(years = 1)

ft = pd.read_csv(f'FS_data/Fcst_model{model_num}_{test_dts}_{lz}.csv')
ft['Init_date'] = pd.to_datetime(ft['Init_date'])
fcst_start = min(ft['Init_date']).strftime('%Y%m%d%H')
fcst_end = max(ft['Init_date']).strftime('%Y%m%d%H')

tr = pd.read_csv(f'FS_data/Train_model{model_num}_{test_dts}_{lz}.csv')
tr['Date'] = pd.to_datetime(tr['Date'])
train_start = min(tr['Date']).strftime('%Y%m%d%H')
train_end = max(tr['Date']).strftime('%Y%m%d%H')

bk = pd.read_csv(f'Feature_model{model_num}_{test_dts}_{lz}/BK_removed_features30.csv')
bk_min = bk[bk['MAE'] == bk['MAE'].min()].head(1)
bk_mae = bk_min['MAE'].values[0]

fr = pd.read_csv(f'Feature_model{model_num}_{test_dts}_{lz}/FR_added_features30.csv')
fr_min = fr[fr['MAE'] == fr['MAE'].min()].head(1)
fr_mae = fr_min['MAE'].values[0]
   
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



# if bk_mae < fr_mae:
#     print('BK Less')
#     num = bk[bk['MAE'] == bk['MAE'].min()]['Round'].values[0]
#     if num == 30:
#         mae_val = bk[bk['MAE'] == bk['MAE'].min()].head(1)['MAE'].values[0]
#         df = pd.read_csv(f'Feature_model{model_num}_{test_dts}_{lz}/BK_keep_columns30.csv')
#         feat_val = list(df['Keep_Columns'].values)
#     else: 
#         mae_val = bk[bk['MAE'] == bk['MAE'].min()].head(1)['MAE'].values[0]
#         df = pd.read_csv(f'Feature_model{model_num}_{test_dts}_{lz}/BK_30round{num + 1}.csv')
#         feat_val = list(df['Col_remove'].unique())
#     print(feat_val)
#     fs = "back"
# elif bk_mae > fr_mae:
#     print('FR less')
#     num = fr[fr['MAE'] == fr['MAE'].min()]['Round'].values[0]
#     mae_val = fr[fr['MAE'] == fr['MAE'].min()].head(1)['MAE'].values[0]
#     fr_feat = list(fr[fr['Round'] <= num]['Feature'])
#     feat_val = fr_feat
#     print(feat_val)
#     fs = "front"

if bk_mae < fr_mae:
    print('BK Less')
    num = bk[bk['MAE'] == bk['MAE'].min()]['Round'].values[0]
    mae_val = bk[bk['MAE'] == bk['MAE'].min()].head(1)['MAE'].values[0]
    rm_cols = list(bk[bk['Round'] <= num]['Feature'].values)
    bb = train2.drop(['Categorical_freezing_rain', 'Categorical_rain', 'Categorical_snow','day_of_year',
                            'hour', 'Year','Date','LZ','OBS_DATA'] + rm_cols ,axis = 1)
    feat_val = list(bb.columns)
    print(feat_val)
    fs = "back"
elif bk_mae > fr_mae:
    print('FR less')
    num = fr[fr['MAE'] == fr['MAE'].min()]['Round'].values[0]
    mae_val = fr[fr['MAE'] == fr['MAE'].min()].head(1)['MAE'].values[0]
    new_feat = list(fr[fr['Round'] <= num]['Feature'])
    feat_val = new_feat + ['Downward_short_wave_radiation_flux_pop',
       'Downward_short_wave_radiation_flux_solar', 'cooling_6_housing',
       'cooling_wet_bulb_6', 'heating_6_housing', 'heating_6_electricity','cosHour_Angle','Holiday','cosDOY_Angle','day_of_week','bias' ]
    print(feat_val)
    fs = "front"



res_test, order_col = XGB_model_check(train2, test2,feat_val,"test")

print(round(res_test['MAE'].values[0],2))
print(mae_val)
print(f'Selected Method: {fs} at Round {num}')
match_stat = round(res_test['MAE'].values[0],2) == round(mae_val,2)
print(match_stat)

save_columns('col_order',f'Model{model_num}_{test_dts}_{lz}_col_order', order_col)

columns_to_remove = {'day_of_week', 'bias'}  # Use a set for faster lookup
new_feat = [col for col in feat_val if col not in columns_to_remove]

dic = {"cols":new_feat,
      "Match ML": str(match_stat),
      "Train Start": train_start,
      "Train End":train_end, 
      "Fcst Start": fcst_start,
      "Fcst End": fcst_end,
      "FS selcted": fs}

with open(f'selected_cols/Model{model_num}_{test_dts}_{lz}.json', 'w') as f:
    json.dump(dic, f)
