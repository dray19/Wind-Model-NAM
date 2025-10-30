from utils import *
import glob

def monthly_models(name, dts, num):
    local_path = f"/mnt/trade05_data/wind_v5/{name}/Model{num}_training_{dts}_all"
    print(f"** Coming from: {local_path}")
    create_folder_move(local_path,name)

    local_file = f"/mnt/trade05_data/wind_v5/{name}/selected_cols/cols_all.json"
    remote_file = f"/home/wind/data/wind_v5/{name}/selected_cols/cols_all.json"
    basic_move(local_file, remote_file)


dts_val = 2025100106
name_list = [i.split('/')[-1] for i in glob.glob('../model*')]

for nm in name_list:
    print(nm)
    monthly_models(nm, dts_val, 1)
    print('==============')

