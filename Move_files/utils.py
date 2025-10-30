import os
import sys
import numpy as np
import pandas as pd
import datetime
import glob
import paramiko

def upload_folder(sftp, local_path, remote_path):
    # Ensure remote directory exists
    try:
        sftp.mkdir(remote_path)
    except IOError:
        pass  # Ignore if directory already exists
    
    for root, dirs, files in os.walk(local_path):
        for dirname in dirs:
            local_dir = os.path.join(root, dirname)
            relative_dir = os.path.relpath(local_dir, local_path)
            remote_dir = os.path.join(remote_path, relative_dir).replace("\\", "/")
            try:
                sftp.mkdir(remote_dir)
            except IOError:
                pass  # Ignore if directory already exists
        for filename in files:
            local_file = os.path.join(root, filename)
            print(local_file)
            relative_file = os.path.relpath(local_file, local_path)
            remote_file = os.path.join(remote_path, relative_file).replace("\\", "/")
            sftp.put(local_file, remote_file)

def create_remote_folder_if_not_exists(sftp, remote_path):
    """
    Creates a remote folder. If the folder already exists, it does nothing.
    """
    try:
        sftp.mkdir(remote_path)
        print(f"Folder '{remote_path}' created successfully.")
    except IOError as e:
        # Check if the error is due to the folder already existing
        if "File exists" in str(e):
            print(f"Folder '{remote_path}' already exists. No action needed.")
        else:
            # Raise other errors
            raise
            
def create_folder_move(lf, rm_folder):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname="test15.corp.gridpointweather.com", username="wind", password="llasbi!")
    sftp = ssh.open_sftp()
    
    local_folder = lf
    remote_folder = f"/home/wind/data/wind_v5/{rm_folder}"
    new_name = local_folder.split('/')[-1]
    new_folder = remote_folder + "/" + new_name
    
    create_remote_folder_if_not_exists(sftp,new_folder)
    
    upload_folder(sftp, local_folder, new_folder)
    val = local_folder.split('/')[-2] == new_folder.split('/')[-2]
    print(f"** Placed At: {new_folder}")
    print(f"Same Folder:{val}")
    print('=====================================================================')
    sftp.close()
    ssh.close()
    
def basic_move(lf, rf):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname="test15.corp.gridpointweather.com", username="wind", password="llasbi!")
    sftp = ssh.open_sftp()
    
    local_file = lf
    remote_file = rf
    sftp.put(local_file, remote_file)
    
    val = local_file.split('/')[-3] == remote_file.split('/')[-3]
    print(f"** Placed At: {remote_file}")
    print(f"Same Folder:{val}")
    print('=====================================================================')
    sftp.close()
    ssh.close()