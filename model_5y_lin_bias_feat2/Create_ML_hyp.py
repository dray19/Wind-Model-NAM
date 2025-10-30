from ML_model import XGB_model_hyp
from utils import *
import sys
from dateutil.relativedelta import relativedelta

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

model_num = str(sys.argv[1])
test_dts = int(sys.argv[2])
lz = str(sys.argv[3])
name = str(sys.argv[4])

test_dts_val = pd.to_datetime(test_dts, format = "%Y%m%d%H", utc = True)
test_end_val = test_dts_val + relativedelta(months = 1)
train_start_val = test_dts_val - relativedelta(years = 1)

out_folder = create_folder_ml(model_num,test_dts,lz)

cols_list = get_cols(f'selected_cols/Model{model_num}_{test_dts}_{lz}.json')['cols']

selected_params = get_hyp(f'Hyp_model{model_num}_{test_dts}_{lz}/params_model{model_num}_{test_dts}_{lz}.json')

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

train2 = filtered_train[(filtered_train['Date'] >= train_start_val) & (filtered_train['Date'] < test_dts_val)].reset_index(drop = True)
train2 = train2[train2['LZ'] == lz].reset_index(drop = True)
train2 = train2.drop('MISO_FORECAST', axis = 1)
print(train2['Date'].min(), train2['Date'].max())
train2.to_csv(f'{out_folder}/Training_data_{test_dts}_hyp.csv', index = False)

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

test2 = filtered_test[(filtered_test['Init_date'] >= test_dts_val) & (filtered_test['Init_date'] < test_end_val)].reset_index(drop = True)
test2 = test2[test2['LZ'] == lz].reset_index(drop = True)
test2 = test2.drop('MISO_FORECAST', axis = 1)
print(test2['Date'].min(), test2['Date'].max())
print(test2['Init_date'].min(), test2['Init_date'].max())


fcst_start = min(test2['Init_date']).strftime('%Y%m%d%H')
fcst_end = max(test2['Init_date']).strftime('%Y%m%d%H')
train_start = min(train2['Date']).strftime('%Y%m%d%H')
train_end = max(train2['Date']).strftime('%Y%m%d%H')
dic = {"Model Num":model_num,
      "Test Year": str(test_dts),
      "Train Start": train_start,
      "Train End":train_end, 
      "Fcst Start": fcst_start,
      "Fcst End": fcst_end}

with open(f'{out_folder}/Model{model_num}_info_{test_dts}_hyp.json', 'w') as f:
    json.dump(dic, f)

col_order = read_col_order('col_order', f'Model{model_num}_{test_dts}_{lz}_col_order.txt')
    
    
fcst_data, error_met  = XGB_model_hyp(train2,test2, cols_list, "XGB", out_folder, model_num,test_dts,col_order,selected_params)


fcst_data.to_csv(f'{out_folder}/fcst_{test_dts}_hyp.csv', index = False)
error_met.to_csv(f'{out_folder}/Error_LZ_{test_dts}_hyp.csv', index = False)
# error_agg.to_csv(f'{out_folder}/Error_agg_{test_dts}.csv', index = False)
