# functions for merging timestamp into behaviour data
import numpy as np
import pandas as pd
import os
from touchscreen_toolbox import utils
import touchscreen_toolbox.config as cfg



## TODO:
## merge mice data (KO, male) into vid_info & data

def merge(vid_info, data, timestamp_file):
    
    states, trials = find_timestamp(vid_info, timestamp_file)
    data = merge_states(data, states)
    data = merge_trials(data, trials)
    return data


def search_timestamp(vid_info, timestamp_file):
    
    mouse_id = vid_info['mouse_id']
    exp_date = vid_info['exp_date']
    path = f"{mouse_id}/{exp_date}"
    
    with h5py.File(timestamp_file, 'r') as ts:
        states = pd.DataFrame(ts[path+'/states'], columns=ts[path+'/states'].attrs['headers']).convert_dtypes()
        trials = pd.DataFrame(ts[path+'/trials'], columns=ts[path+'/trials'].attrs['headers']).convert_dtypes()
    
    return states, trials


def merge_states(data: pd.DataFrame, states: pd.DataFrame, fps=25):                ############# <- remember to generalize fps!

    # align starting time
    states['time'] += data['time'].iloc[0]    ### <-is this correct?

    states['frame'] = states['time'] * fps
    increment_duplicates(states, 'frame')

    merged = data.merge(states.drop('time', axis=1), how='left', on='frame')
    merged['state_'] = merged['state'].fillna(method='ffill')#.fillna(method='bfill')

    return merged


def merge_trials(data: pd.DataFrame, trials: pd.DataFrame, fps=25):  
    # align starting time
    trials['time'] += data['time'].iloc[0] - trials['time'].iloc[0]    ### <-is this correct?
    trials['frame'] = (trials['time'] * fps).astype(int)
    increment_duplicates(trials, 'frame')

    merged = data.merge(trials.drop('time', axis=1), how='left', on='frame')
    for col in trials.columns:
        merged[col] = merged[col].fillna(method='ffill')#.fillna(method='bfill')
    
    return merged


# helper
# ------
def increment_duplicates(df: pd.DataFrame, col: str):
    """Remove duplicates by incrementing the latter by 1"""
    for i in range(1, len(df)):
        prev, curr = df.loc[[i-1,i], col]
        if curr<=prev:
            df.loc[i, col] = curr + 1