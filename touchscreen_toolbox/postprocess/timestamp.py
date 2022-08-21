# functions for merging timestamp into behaviour data
import h5py
import numpy as np
import pandas as pd
import touchscreen_toolbox.config as cfg
from .feature import make_multiindex

state_mapping = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 0: 0, 10: 0, 99: 0}


def merge(vid_info, data, timestamp_file):
    """Merge information from touchscreen to pose estimation data"""

    states, trials, attrs = search_timestamps(vid_info, timestamp_file)

    data2 = pd.DataFrame(data.index)
    data2 = merge_info(data2, attrs)
    data2 = merge_states(data2, states, vid_info)
    data2 = merge_trials(data2, trials)

    data2 = make_multiindex(data2, 'task')
    return pd.concat([data, data2], axis=1)


def search_timestamps(vid_info, timestamp_file):
    mouse_id = vid_info['mouse_id']
    exp_date = vid_info['exp_date']
    path = f"{mouse_id}/{exp_date}"

    with h5py.File(timestamp_file, 'r') as ts:
        states = pd.DataFrame(ts[path + '/states'], columns=ts[path + '/states'].attrs['headers']).convert_dtypes()
        trials = pd.DataFrame(ts[path + '/trials'], columns=ts[path + '/trials'].attrs['headers']).convert_dtypes()
        attrs = dict(ts[mouse_id].attrs)

    return states, trials, attrs


def merge_info(data, attrs):
    data = data.copy()
    for a in attrs.keys():
        data[a] = attrs[a]
    return data


def merge_states(data: pd.DataFrame, states: pd.DataFrame, vid_info: dict):
    """Merge state timestamp"""
    # align starting time
    states['time'] += vid_info['time'][0] - cfg.TIME_BUFFER[0]
    states['frame'] = (states['time'] * vid_info['fps']).astype(int)
    states = states.drop('time', axis=1)

    increment_duplicates(states, 'frame')
    states = states.set_index('frame')
    merged = data.reset_index().merge(states, how='left', on='frame').set_index('frame')
    merged['state_'] = merged['state'].fillna(method='ffill')

    # simplify state_ (###harcoded###)
    merged['state_'] = merged['state_'].replace(state_mapping).fillna(0)

    ##############
    # TODO
    # fix this
    try:
        merged.drop('index', axis=1, inplace=True)
    except:
        pass

    # count trial numbers
    merged = count_trials(merged)
    merged = merged.fillna(np.nan)

    return merged.convert_dtypes()


def count_trials(data: pd.DataFrame):
    """Count trial number"""
    data = data.copy()
    data['trial'] = np.nan
    idxs = data.index[data['state'].fillna(0) == 1]
    for idx, i in zip(idxs, np.arange(len(idxs))):
        data.loc[idx, 'trial'] = i + 1
    data.loc[:, 'trial'] = data['trial'].fillna(method='ffill')
    return data


def merge_trials(data: pd.DataFrame, trials: pd.DataFrame):
    """Merge trial information (reward probability etc.)"""
    trials = trials.drop('time', axis=1)
    merged = data.reset_index().merge(trials, how='left', on='trial').set_index('frame')
    for col in trials.columns:
        merged[col] = merged[col].fillna(method='ffill')

    return merged


# helper
# ------
def increment_duplicates(df: pd.DataFrame, col: str):
    """Remove duplicates by incrementing the latter by 1"""
    col = df.columns.get_loc(col)
    for i in range(1, len(df)):
        prev, curr = df.iloc[[i - 1, i], col]
        if curr <= prev:
            df.iloc[i, col] = curr + 1
