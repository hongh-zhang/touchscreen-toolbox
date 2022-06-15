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