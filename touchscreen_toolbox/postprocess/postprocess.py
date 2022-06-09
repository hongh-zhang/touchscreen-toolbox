import os
import sys
import numpy as np
import pandas as pd
from itertools import groupby

import touchscreen_toolbox.utils as utils
import touchscreen_toolbox.config as cfg


# prediction refinement
# ----------------------------------------------
def cutoff(data: pd.DataFrame, p_cutoff: float=cfg.P_CUTOFF):
    """Drop low confidence prediction"""
    
    # iterate columns in group of three (x, y, confidence) for each body part
    for i in range(0, data.shape[1], 3):

        # get index of low confidence
        idx = data.index[data.iloc[:, i+2] < p_cutoff]

        # set coordinates to NaN
        data.iloc[idx, i] = pd.NA
        data.iloc[idx, i+1] = pd.NA


def median_filter(data: pd.DataFrame, window_len: int=5):
    """Apply sliding median filter to all columns in <df>"""
    
    # apply filter to each column
    for i in range(data.shape[1]):    
        data.iloc[:,i] = data.iloc[:,i].rolling(window_len, min_periods=1, center=True).median()



# record prediction statistics
# ----------------------------------------------
STATS_TEMPL = pd.DataFrame(np.vstack((cfg.HEAD1, cfg.HEAD2)))

def create_stats(path: str):
    """Create empty statistics.csv at the given <path>"""
    STATS_TEMPL.to_csv(path, index=False)


def statistics(data: pd.DataFrame, p_cutoff: float=cfg.P_CUTOFF):
    """Produce statistics about miss predicted values"""
    frames = len(data)
    value = [frames]
    for col_name in CCOLS:
        col = data[col_name].fillna(0)
        zeros = (col == 0)
        nums = zeros.sum()
        percent = round(nums / frames, 2)
        consecutive = max([len(list(g))
                          for k, g in groupby(zeros) if k]) if nums > 0 else 0
        first = str(round(col.quantile(q=0.01), 2))
        tenth = str(round(col.quantile(q=0.10), 2))
        value += [nums, percent, consecutive, first, tenth]
    return value


def record(data: pd.DataFrame, folder_path: str, video_name: str, 
           mouse_id: str, chamber: str, date: str, time: str, proc_ls: list):
    """Record statistics into csv"""
    stats_path = os.path.join(folder_path, RST_FOLDER, STATS_NAME)
    if os.path.exists(stats_path):
        stats = pd.read_csv(stats_path)
        stats.loc[len(stats)] = ([video_name, mouse_id, chamber, date, time] + [str(proc_ls)] + statistics(data))
        stats.to_csv(stats_path, index=False)