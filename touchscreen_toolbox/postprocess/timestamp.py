# functions for merging timestamp into behaviour data
import os
import h5py
import numpy as np
import pandas as pd
from touchscreen_toolbox import utils
import touchscreen_toolbox.config as cfg



def merge(vid_info, data, timestamp_file):
    """Merge information from touchscreen to pose estimation data"""
    states, trials, attrs = search_timestamps(vid_info, timestamp_file)
    data = merge_info(data, attrs)
    data = merge_states(data, states)
    data = merge_trials(data, trials)
    return data.convert_dtypes()


def search_timestamps(vid_info, timestamp_file):
    
    mouse_id = vid_info['mouse_id']
    exp_date = vid_info['exp_date']
    path = f"{mouse_id}/{exp_date}"
    
    with h5py.File(timestamp_file, 'r') as ts:
        states = pd.DataFrame(ts[path+'/states'], columns=ts[path+'/states'].attrs['headers']).convert_dtypes()
        trials = pd.DataFrame(ts[path+'/trials'], columns=ts[path+'/trials'].attrs['headers']).convert_dtypes()
        attrs  = dict(ts[mouse_id].attrs)
    
    return states, trials, attrs


def merge_info(data, attrs):
    data = data.copy()
    for a in attrs.keys():
        data[a] = attrs[a]
    return data


def merge_states(data: pd.DataFrame, states: pd.DataFrame, fps=25):                ############# <- remember to generalize fps!
    """Merge state timestamp"""
    # align starting time
    states['time'] += data['time'].iloc[0]

    states['frame'] = (states['time'] * fps).astype(int)
    increment_duplicates(states, 'frame')

    merged = data.merge(states.drop('time', axis=1), how='left', on='frame').set_index('frame')
    merged['state_'] = merged['state'].fillna(method='ffill')
    
    # count trial numbers
    merged = count_trials(merged)

    return merged.convert_dtypes()


def count_trials(data: pd.DataFrame):
    """Count trial number"""
    data = data.copy()
    data['trial'] = np.NaN
    idxs = data.index[data['state'].fillna(0)==1]
    for idx, i in zip(idxs, np.arange(len(idxs))):
        data.loc[idx, 'trial'] = i+1
    data.loc[:,'trial'] = data['trial'].fillna(method='ffill')
    return data


def merge_trials(data: pd.DataFrame, trials: pd.DataFrame, fps=25):  
    """Merge trial information (reward probability etc.)"""
    # use trial timestamp
    ### this is inaccurate
    # trials['time'] += data['time'].iloc[0]
    # trials['frame'] = (trials['time'] * fps).astype(int)
    # increment_duplicates(trials, 'frame')

    merged = data.merge(trials.drop('time', axis=1), how='left', on='trial')
    for col in trials.columns:
        merged[col] = merged[col].fillna(method='ffill')
    
    return merged


# helper
# ------
def increment_duplicates(df: pd.DataFrame, col: str):
    """Remove duplicates by incrementing the latter by 1"""
    for i in range(1, len(df)):
        prev, curr = df.loc[[i-1,i], col]
        if curr<=prev:
            df.loc[i, col] = curr + 1