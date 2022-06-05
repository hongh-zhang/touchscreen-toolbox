import os
import re
import sys
import shutil
import pandas as pd
from time import (localtime, strftime)
import touchscreen_toolbox.config as cfg


# IO related
# ------------------------

def is_generated(folder_path):
    return os.path.basename(folder_path) in [cfg.DLC_FOLDER, cfg.RST_FOLDER]


def mk_dir(path: str, force: bool = True):
    """(re)Make directory, deletes existing directory"""
    if os.path.exists(path):
        if force:
            print(f"Removing existing {os.path.basename(path)} folder")
            shutil.rmtree(path)
        
        else:
            print(f"Folder already exists, failed to make directory")
            return 1

    os.mkdir(path)


def read_dlc_csv(path: str):
    return pd.read_csv(path, skiprows=[0, 1, 2, 3],
                       names=(['frame'] + cfg.HEADERS)).set_index('frame')


def move_files(files, curr_folder, targ_folder):
    for f in files:
        file_path = os.path.join(curr_folder, f)
        new_path = os.path.join(targ_folder, f)
        os.rename(file_path, new_path)

def find_files(folder_path, extension):
    """
    Find all files under the <folder_path> with the specific <extension>,
    e.g. find_files(/path/to/results, ".csv")
    """
    return list(filter(lambda x: x.endswith(extension), os.listdir(folder_path)))

# logger
# ------------------------
# modified from
# https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file
class Tee(object):
    def __init__(self, file):
        self.file = open(file, 'w')
        self.stdout = sys.stdout
        sys.stdout = self
        
    def __del__(self):
        try:
            self.close()
        except: pass
        
    def write(self, data):
        if data != '\n':
            data = strftime("%H:%M:%S    ", localtime()) + data
        self.file.write(data)
        self.stdout.write(data)
        
    def flush(self): self.file.flush()
        
    def close(self):
        sys.stdout = self.stdout
        self.file.close()