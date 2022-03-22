import os
import shutil
import numpy as np
from math import pi

# postprocess related
# ------------------------
# magic numbers
FPS = 25
TRAY_LENGTH = 26.8  # (cm)

# for quick access of columns
MICE = ('snout', 'lEar', 'rEar', 'spine1', 'spine2', 'tail1', 'tail2', 'tail3')
REFE = ('food_port', 'll_corner', 'lr_corner', 'l_screen', 'm_screen', 'r_screen')
HEADERS = [j for i in (MICE + REFE) for j in (i+'_x', i+'_y', i+'_cfd')]
XCOLS = [i for i in HEADERS if '_x' in i]
YCOLS = [i for i in HEADERS if '_y' in i]
CCOLS = [i for i in HEADERS if '_cfd' in i]

# distance
def dist1(point):
    """1 point from origin"""
    return np.linalg.norm(point, ord=2)
def dist2(point1, point2):
    """distance between 2 points"""
    return np.linalg.norm(point2-point1, ord=2)

def frame2sec(frame : int): return frame/FPS
def sec2frame(sec : float): return int(FPS * sec)



# IO related
# ------------------------
def path2file(path : str):
    """Convert a full [path] to file name only"""
    return path.split('/')[-1]

def mk_dir(path : str):
    """(re)Make directory, deletes existing directory"""
    if os.path.exists(path):
        print(f"Removing existing {path2file(path)} folder")
        shutil.rmtree(path)
    os.mkdir(path)
