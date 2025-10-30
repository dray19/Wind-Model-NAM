import pandas as pd
import os
import numpy as np
from utils import *
from dateutil.relativedelta import relativedelta
import warnings
import multiprocessing as mp
import sys
# Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

select_val = str(sys.argv[1])
model_num = str(sys.argv[2])
test_dts = int(sys.argv[3])
lz = str(sys.argv[4])
name = str(sys.argv[5])

def create_folder(mv, ty, loadzone):
    if not os.path.exists(f'Feature_model{mv}_{ty}_{loadzone}'):
           os.makedirs(f'Feature_model{mv}_{ty}_{loadzone}')
create_folder(model_num, test_dts,lz)

test_dts_val = pd.to_datetime(test_dts, format = "%Y%m%d%H", utc = True)
train_dts_val = test_dts_val - relativedelta(months = 4)
train_start_val = train_dts_val - relativedelta(years = 1)


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
train2.to_csv(f'FS_data/Train_model{model_num}_{test_dts}_{lz}.csv', index = False)

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
test2.to_csv(f'FS_data/Fcst_model{model_num}_{test_dts}_{lz}.csv', index = False)

cols = ['2m_Temperature', '2m_Dew_point_Temperature', '2m_Wet_Bulb',
       '10_mean_speed', 'Total_cloud_cover', 'Derived_radar_reflectivity_dB',
       'Snow_depth_m', 'Wind_speed_gust', 'Planetary_boundary_layer_height',
       '80_mean_speed', '80m_v_component', '80m_u_component',
       'Surface_SL_Pressure_from_PC', 'Surface_CAPE', 'Surface_CIN',
       'High_cloud_cover', 'Medium_cloud_cover', 'Low_cloud_cover',
       'u_component_850', 'u_component_700', 'u_component_500',
       'v_component_850', 'v_component_700', 'v_component_500',
       'Geopotential_height_850', 'Geopotential_height_700',
       'Geopotential_height_500', 'Geopotential_height_700_minus_850',
       'Geopotential_height_500_minus_850', 'relative_humidity_850',
       'relative_humidity_700', 'relative_humidity_500',
       'relative_humidity_80']



def process_col_bk(train_data, fcst_data, rm_feat1, i):
    rm = rm_feat1 + [i]
    print(f"Processing column: {i}")
    print(f"Features to remove: {rm}")
    gm = feat_selecter_all(train_data, fcst_data, rm, i)
    return gm

def parallel_process_bk(train_data, fcst_data, rm_feat1, cols):
    with mp.Pool(6) as pool:
        # Use pool.starmap to pass multiple arguments
        out_df = pool.starmap(
                process_col_bk, 
                [(train_data, fcst_data, rm_feat1, i) for i in cols]  # Multiple inputs as tuples
            )
        
    # Return or store out_df as needed
    return out_df

def process_col_fr(train_data, fcst_data, rm_feat1, i):
    rm = rm_feat1 + [i]
    print(f"Processing column: {i}")
    print(f"Features to remove: {rm}")
    gm = feat_selecter_all_fr(train_data, fcst_data, rm, i)
    return gm

def parallel_process_fr(train_data, fcst_data, rm_feat1, cols):
    with mp.Pool(6) as pool:
        # Use pool.starmap to pass multiple arguments
        out_df = pool.starmap(
                process_col_fr, 
                [(train_data, fcst_data, rm_feat1, i) for i in cols]  # Multiple inputs as tuples
            )
        
    # Return or store out_df as needed
    return out_df




if select_val == "back":
    out_rm = []
    ff_list = []
    keep_num = 30
    for num in range(keep_num):
        output = parallel_process_bk(train2, test2, ff_list, cols)
        final = pd.concat(output).reset_index(drop = True)
        final.to_csv(f'Feature_model{model_num}_{test_dts}_{lz}/BK_{keep_num}round{num + 1}.csv', index = False)
        rm_feat_name,rm_gain = final.sort_values('MAE').head(1)[['Col_remove', 'MAE']].values[0]
        ff_list.append(rm_feat_name)
        out_rm.append({'Round': num + 1, "Feature":rm_feat_name, "MAE":rm_gain})
        print("Remove:", rm_feat_name)
        cols.remove(rm_feat_name)

    results = pd.DataFrame(out_rm)
    results.to_csv(f'Feature_model{model_num}_{test_dts}_{lz}/BK_removed_features{keep_num}.csv', index = False)
    kc = pd.DataFrame(cols, columns = ['Keep_Columns'])
    kc.to_csv(f'Feature_model{model_num}_{test_dts}_{lz}/BK_keep_columns{keep_num}.csv', index = False)
    
elif select_val == "front":
    out_rm = []
    ff_list = []
    keep_num = 30
    for num in range(keep_num):
        output = parallel_process_fr(train2, test2, ff_list, cols)
        final = pd.concat(output).reset_index(drop = True)
        final.to_csv(f'Feature_model{model_num}_{test_dts}_{lz}/FR_{keep_num}round{num + 1}.csv', index = False)
        rm_feat_name,rm_gain = final.sort_values('MAE').head(1)[['Col_remove', 'MAE']].values[0]
        ff_list.append(rm_feat_name)
        out_rm.append({'Round': num + 1, "Feature":rm_feat_name, "MAE":rm_gain})
        print("Remove:", rm_feat_name)
        cols.remove(rm_feat_name)

    results = pd.DataFrame(out_rm)
    results.to_csv(f'Feature_model{model_num}_{test_dts}_{lz}/FR_added_features{keep_num}.csv', index = False)
    kc = pd.DataFrame(ff_list, columns = ['Keep_Columns'])
    kc.to_csv(f'Feature_model{model_num}_{test_dts}_{lz}/FR_keep_columns{keep_num}.csv', index = False)
