import os
import re
import sys
import shutil
import numpy as np
import pandas as pd
from math import pi
from time import (localtime, strftime)

# configs
# ------------------------

# magic numbers
FPS = 25
TRAY_LENGTH = 26.8  # (cm)

# keypoints
MICE = ('snout',
        'lEar',
        'rEar',
        'spine1',
        'spine2',
        'tail1',
        'tail2',
        'tail3')
REFE = ('food_port',
        'll_corner',
        'lr_corner',
        'l_screen',
        'm_screen',
        'r_screen')

# files
DLC_CONFIG = "touchscreen_toolbox/DLC/config.yaml"  # path of deeplabcut config
DLC_FOLDER = "DLC"      # name of subfolder to put files from DLC (h5 & pickle)
RST_FOLDER = "results"  # name of subfolder to put analyzed results (csv)
STATS_NAME = 'statistics.csv'

# preprocess related
# ------------------------

P_SUFIXX = ['bright.mp4']  # possible preprocessed video suffix

def is_preprocess(name):
    return np.any([name.endswith(suffix) for suffix in P_SUFIXX])


# postprocess related
# ------------------------

# for quick access of columns
HEADERS = [j for i in (MICE + REFE) for j in (i + '_x', i + '_y', i + '_cfd')]
XCOLS = [i for i in HEADERS if '_x' in i]
YCOLS = [i for i in HEADERS if '_y' in i]
CCOLS = [i for i in HEADERS if '_cfd' in i]

# template for statistics.csv
HEAD1 = ['video', 'id', 'chamber', 'date', 'time', 'pre', 'frame'] + [i[:-4] for i in CCOLS for j in '12345']
HEAD2 = ['-' for i in range(7)] + [j for i in CCOLS for j in ('#of0', '%of0', 'cons', '1stQ', '10thQ')]
STATS_TEMPL = pd.DataFrame(np.vstack((HEAD1, HEAD2)))


def create_stats(path: str):
    """Create empty statistics.csv at the given <path>"""
    STATS_TEMPL.to_csv(path, index=False)

# distance
def dist1(point):
    """Euclidean distance from the origin"""
    return np.linalg.norm(point, ord=2)


def dist2(point1, point2):
    """Euclidean distance between 2 points"""
    return np.linalg.norm(point2 - point1, ord=2)



def absmin(x1, x2):
    """Absolute minimum from 2 1D sequences"""
    a1 = np.abs(x1)
    a2 = np.abs(x2)
    idx = (a1 < a2).astype(int)
    return x1*idx + x2*(1-idx)


# time conversion
def frame2sec(frame: int): return frame / FPS
def sec2frame(sec: float): return np.round(FPS * sec).astype(int)

# IO related
# ------------------------

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
                       names=(['frame'] + HEADERS)).set_index('frame')


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


# hardcoded postprocessing
# ------------------------
PATTERN = '^(\d+) - (\S+) - (\d{2}-\d{2}-\d{2}) (\d{2}-\d{2}) (\S+)'
ELEMENTS = ['mouse_id', 'chamber', 'date', 'time', 'suffix']
DEFAULT = ['-' for i in range(5)]
def decode_name(name, pattern=PATTERN):
    try:
        matched = [''.join(i.split('-')) for i in re.match(pattern, name).groups()]
        return True, {i:j for i,j in zip(ELEMENTS, matched)}
    except AttributeError:
        print("Pattern unmatched")
        return False, {}
    
    
def get_time(time_file, mouse_id, date, pre_buffer=1, post_buffer=2, hi_bound=999999):
    col = pd.read_csv(time_file).set_index(['id', 'date']).loc[(int(mouse_id), int(date))]
    start, end = map(sec2frame, (col['vid_start'], col['vid_end']))
    start = max(0,  start - FPS * pre_buffer)
    end   = min(hi_bound, end + FPS * post_buffer)
    return start, end