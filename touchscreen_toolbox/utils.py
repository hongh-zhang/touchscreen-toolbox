import os
import shutil
import numpy as np
import pandas as pd
from math import pi

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
HEAD1 = ['video', 'pre'] + [i[:-4] for i in CCOLS for j in '1234']
HEAD2 = ['-', '-'] + [j for i in CCOLS for j in ('#of0', 'cons', '1stQ', '10thQ')]
STATS_TEMPL = pd.DataFrame(np.vstack((HEAD1, HEAD2)))

# distance
def dist1(point):
    """1 point from origin"""
    return np.linalg.norm(point, ord=2)


def dist2(point1, point2):
    """distance between 2 points"""
    return np.linalg.norm(point2 - point1, ord=2)

# time conversion
def frame2sec(frame: int): return frame / FPS
def sec2frame(sec: float): return int(FPS * sec)


# IO related
# ------------------------

def mk_dir(path: str):
    """(re)Make directory, deletes existing directory"""
    if os.path.exists(path):
        print(f"Removing existing {os.path.basename(path)} folder")
        shutil.rmtree(path)
    os.mkdir(path)

def read_dlc_csv(path: str):
    return pd.read_csv(path, skiprows=[0, 1, 2, 3],
                       names=(['frame'] + HEADERS)).set_index('frame')
# DEPRECATED
# ------------------------
# def path2file(path : str):
#     """Convert a full [path] to file name only"""
#     return path.split('/')[-1]
# ->
# use os.path.basename(path) instead
