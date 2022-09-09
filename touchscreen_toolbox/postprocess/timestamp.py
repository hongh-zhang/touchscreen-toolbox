# functions for merging timestamp into behaviour data
import h5py
import numpy as np
import pandas as pd
import touchscreen_toolbox.config as cfg
from .feature import multiindex_col

state_mapping = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 0: 0, 10: 0, 99: 0}


def merge(vid_info, data, timestamp_file):
    """Merge information from touchscreen to pose estimation data"""

    states, trials, attrs = search_timestamps(vid_info, timestamp_file)

    data2 = pd.DataFrame(data.index)
    data2 = merge_info(data2, attrs)
    data2 = merge_states(data2, states, vid_info)
    data2 = merge_trials(data2, trials)

    data2 = multiindex_col(data2, 'task')
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
    trials = process_trials(trials)
    trials = trials.drop('time', axis=1)
    merged = data.reset_index().merge(trials.convert_dtypes(), how='left', on='trial').set_index('frame')
    for col in trials.columns:
        merged[col] = merged[col].fillna(method='ffill')

    return merged


def process_trials(data: pd.DataFrame) -> pd.DataFrame:
    data = data.astype(float)
    data['P_contrast'] = data['P_left'] - data['P_right']
    data['optimal'] = np.logical_or(((data['P_contrast'] > 0) & data['left_response']),
                                    ((data['P_contrast'] < 0) & data['right_response'])).astype(int)

    # consecutive reward / loss
    cons_reward = 0
    ls_cons_reward = []
    ls_unexpectation = []

    for reward in data['reward']:
        unexpectation = 0
        if cons_reward >= 0 and reward:
            cons_reward += 1
        elif cons_reward <= 0 and not reward:
            cons_reward -= 1
        else:
            cons_reward = 1 if reward else -1
            unexpectation = -ls_cons_reward[-1]
        ls_cons_reward.append(cons_reward)
        ls_unexpectation.append(unexpectation)

    data['cons_reward'] = ls_cons_reward
    data['unexpectation'] = ls_unexpectation

    # win-stay-lose-shift
    # 1: win stay, 2: lose shift, 0: False
    switch = (data['left_response'].diff() != 0).astype(int)
    prev_reward = np.insert(data['reward'].values, 0, 0)[:-1]
    data['switch'] = switch
    data['prev_reward'] = prev_reward

    win_stay = (prev_reward & np.logical_not(switch))
    lose_shift = (switch & np.logical_not(prev_reward))

    data['win_stay'] = 0
    data['win_stay'] += win_stay.astype(int) + lose_shift.astype(int) * 2

    data['rare_reward'] = np.logical_or(
        data['reward'].astype(bool) & (data['P_contrast'] > 0) & ~data['left_response'].astype(bool),
        data['reward'].astype(bool) & (data['P_contrast'] < 0) & data['left_response'].astype(bool)
    ).astype(int)
    data['rare_omission'] = np.logical_or(
        ~data['reward'].astype(bool) & (data['P_contrast'] < 0) & ~data['left_response'].astype(bool),
        ~data['reward'].astype(bool) & (data['P_contrast'] > 0) & data['left_response'].astype(bool)
    ).astype(int)

    # reverse trial number in 1st block
    if len(data['block'].unique()) > 1:
        block1_idx = (data['trial'] == data['trial_'])
        data.loc[block1_idx, 'trial'] = data.loc[block1_idx, 'trial'][::-1]

    data.drop(['right_response', 'P_left', 'P_right', 'prev_response'], axis=1, inplace=True)

    return data


# helper
# ------
def increment_duplicates(df: pd.DataFrame, col: str):
    """Remove duplicates by incrementing the latter by 1"""
    col = df.columns.get_loc(col)
    for i in range(1, len(df)):
        prev, curr = df.iloc[[i - 1, i], col]
        if curr <= prev:
            df.iloc[i, col] = curr + 1
