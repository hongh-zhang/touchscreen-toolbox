# functions for merging timestamp into behaviour data
import h5py
import numpy as np
import pandas as pd
import touchscreen_toolbox.config as cfg
from .feature import multiindex_col

state_mapping = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 0: 0, 10: 0, 99: 0}


def merge(vid_info, data, timestamp_file):
    """Merge information from touchscreen to pose estimation data"""
    
    mouse_id = vid_info['mouse_id']
    exp_date = vid_info['exp_date']
    h5path = f"{mouse_id}/{exp_date}"
    
    data2 = pd.DataFrame(data.index)
    with h5py.File(timestamp_file, 'r') as ts:
        data2 = merge_attrs(data2, ts, mouse_id)
        data2 = merge_states(data2, ts, h5path, vid_info['fps'])
        data2 = merge_trials(data2, ts, h5path)
        data2 = merge_trace(data2, ts, h5path, vid_info['fps'])
        
    data2 = multiindex_col(data2, 'task')
    return pd.concat([data, data2], axis=1)


def merge_attrs(data: pd.DataFrame, ts: h5py._hl.files.File, mouse_id: str):
    """Add mouse attributes (knockout, male) to dataframe"""
    attrs = dict(ts[mouse_id].attrs)
    for a in attrs.keys():
        data[a] = attrs[a][0]
    return data


def merge_states(data: pd.DataFrame, ts: h5py._hl.files.File, path: str, fps: float):
    """Merge state timestamp"""
    
    # retrieve state dataframe
    states = pd.DataFrame(np.transpose(ts[path + '/states']), columns=ts[path + '/states'].attrs['headers']).convert_dtypes()
    
    # align starting time, with buffer
    states['frame'] = (states['time'] * fps).astype(int)
    states = states.drop('time', axis=1)

    increment_duplicates(states, 'frame')
    states = states.set_index('frame')
    merged = data.reset_index().merge(states, how='left', on='frame')
    merged = merged.set_index('frame')
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


def merge_trials(data: pd.DataFrame, ts: h5py._hl.files.File, path: str):
    """Merge trial information (reward probability etc.)"""
    
    # retrieve trials data
    trials = pd.DataFrame(np.transpose(ts[path + '/trials']), columns=ts[path + '/trials'].attrs['headers']).convert_dtypes()
    
    trials = process_trials(trials)
    merged = data.reset_index().merge(trials, how='left', on='trial')
    merged = merged.set_index('frame')
    for col in trials.columns:
        merged[col] = merged[col].fillna(method='ffill')

    return merged


def process_trials(data: pd.DataFrame) -> pd.DataFrame:
    data = data.astype(float)
    data['P_contrast'] = data['P_left'] - data['P_right']
    data['optimal'] = np.logical_or(((data['P_contrast'] > 0) & data['left_response']),
                                    ((data['P_contrast'] < 0) & data['right_response'])).astype(int)
    data['rare'] = np.logical_or(((data['optimal']==1) & (data['reward']==0)),
                                 ((data['optimal']==0) & (data['reward']==1))).astype(int)
    
    data = consecutive_reward(data)
    data = win_stay(data)
    data = format_trial_no(data).astype(float)
    data = get_session_type(data)

    data.drop(['right_response', 'P_left', 'P_right', 'prev_response'], axis=1, inplace=True)

    return data.convert_dtypes()


def consecutive_reward(data: pd.DataFrame) -> pd.DataFrame:
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
    
    return data


def win_stay(data: pd.DataFrame) -> pd.DataFrame:
    # win-stay-lose-shift
    # 1: win stay, 2: lose shift, 0: False
    switch = (data['left_response'].diff() != 0)
    prev_reward = np.insert(data['reward'].values, 0, 0)[:-1].astype(bool)
    prev_rare = np.insert(data['rare'].values, 0, 0)[:-1].astype(bool)
    data['switch'] = switch.astype(int)
    data['prev_reward'] = prev_reward.astype(int)

    win_stay   = (prev_reward & ~prev_rare & ~switch)
    lose_shift = (~prev_reward & ~prev_rare & switch)
    rare_stay  = (prev_reward & prev_rare & ~switch)
    rare_shift = (~prev_reward & prev_rare & switch)
    
    data['win_stay'] = 0
    data['win_stay'] += (win_stay.astype(int)
                         + lose_shift.astype(int) * 2
                         + rare_stay.astype(int) * 3
                         + rare_shift.astype(int) * 4)
    return data


def format_trial_no(data: pd.DataFrame) -> pd.DataFrame:
    
    # reverse trial number in 1st block
    data.loc[:, 'block_trial'] = data['trial']  # [1,n/2] trial within block

    data.loc[:, 'trial_2nd'] = data['block_trial']  # trial after 2nd block
    if len(data['block'].unique()) > 1:  # if session has 2 blocks
        block1_idx = (data['block'] == data['block'].unique()[0])
        data.loc[block1_idx, 'trial_2nd'] = -(data.loc[block1_idx, 'trial_2nd'][::-1].values)

    data.loc[:, 'trial'] = list(range(1, 1 + len(data)))  # [1,n] 'trial within session' (for merging)
    return data


def get_session_type(df: pd.DataFrame) -> pd.DataFrame:
    if len(df['block'].unique()) > 1:
        contrasts = df['P_contrast'].iloc[0], df['P_contrast'].iloc[-1]
        df['session_type'] = '-'.join(map(lambda x: str(int(x*10)), contrasts))
        return df
    else:
        df['session_type'] = str(df['P_contrast'].iloc[0]*10)
        return df
    


# DA trace related
# ------
def merge_trace(data: pd.DataFrame, ts: h5py._hl.files.File, path: str, fps: int):
    """Merge trial information (reward probability etc.)"""
    
    # retrieve trace data
    trace = np.array(ts[path + '/trace'], dtype=float)
    exp_attrs = dict(ts[path].attrs)
    fs = exp_attrs.get('fs')[0]                # photometry sampling rate
    start_cut = exp_attrs.get('DA_start')[0]   # photometry start time (in secs, aligned with start of video)
    
    # prune data to align with video frames
    trace = prune_trace(trace, fs, start_cut, fps=fps)
    
    # merge
    merged = data.reset_index().merge(trace, how='left', on='frame')
    merged = merged.set_index('frame')

    return merged


def prune_trace(trace: np.ndarray, fs: float, start_cut: float, fps: int) -> pd.DataFrame:
    """
    Match sampling rate of photometry recording to video frame per second, 
    by pruning excess data points
    * photometry fs must > video fps for this function to work
    
    Args
    ------
    trace: np.ndarray
        1D photometry signal across time
    
    fs: float
        photometry sampling rate
    
    start_cut: float
        time when photometry starts, relative to video recording start
    
    fps: float
        video frame per second
        
    Returns
    -----
    trace: pd.DataFrame
        dataframe with index representing video frame number, and a single column representing photometry value
    """
    
    assert fps < fs, "Photometry fs must be larger than video fps"
    
    # create index for the trace array
    # each increment represent 1/fs sec in time
    idx = np.arange(0, len(trace))
    
    # convert index to actual time
    idx = idx / fs + start_cut
    
    # convert to frame (dependent on video fps)
    idx = (idx * fps).round().astype(int)

    # remove duplicates
    val, idx = np.unique(idx, return_index=True)
    
    # combine to create dataframe
    # index: frame
    # 'DA': trace value
    trace = pd.DataFrame(trace[idx], index=val, columns=['DA'])
    trace.index.name = 'frame'
    
    return trace.convert_dtypes()



# helper
# ------
def increment_duplicates(df: pd.DataFrame, col: str):
    """Remove duplicates by incrementing the latter by 1"""
    col = df.columns.get_loc(col)
    for i in range(1, len(df)):
        prev, curr = df.iloc[[i - 1, i], col]
        if curr <= prev:
            df.iloc[i, col] = curr + 1
