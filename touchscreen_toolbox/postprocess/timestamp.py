# functions for merging timestamp into behaviour data
import numpy as np
import pandas as pd
import os
from touchscreen_toolbox import utils
import touchscreen_toolbox.config as cfg



def merge(vid_info, ...):
    
    states, trials = find_timestamp(vid_info, file)
    
    data = read_data(...)
    data = merge_states
    data = merge_trials


def find_timestamp(vid_info, file):
    ...
    return states, trials


def merge_states(data: pd.DataFrame, timestamp: pd.DataFrame, fps):
    timestamp = timestamp.convert_dtypes()

    # align starting time
    timestamp['time'] += data.time.iloc[0]

    timestamp['frame'] = timestamp['time'] * fps
    increment_dup(timestamp, 'frame')

    merged = data.merge(timestamp.drop('time', axis=1), how='left', on='frame')
    merged['state_'] = merged['state'].fillna(method='ffill')#.fillna(method='bfill')

    return merged


def merge_trial(...):
    ...