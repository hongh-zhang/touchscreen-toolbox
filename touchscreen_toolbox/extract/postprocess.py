import os
import sys
import numpy as np
import pandas as pd
from itertools import groupby
from touchscreen_toolbox.utils import *


# prediction refinement
# ----------------------------------------------
def cutoff(df: pd.DataFrame, p_cutoff: float=0.1):
    """Drop low confidence prediction"""
    
    # iterate columns in group of three
    for i in range(0, df.shape[1], 3):

        # get index
        idx = df.index[df.iloc[:, i+2] < p_cutoff]

        # set to NaN
        df.iloc[idx, i] = pd.NA
        df.iloc[idx, i+1] = pd.NA


def median_filter(df: pd.DataFrame, window_len: int=5):
    """Apply sliding median filter to all columns in <df>"""
    for i in range(df.shape[1]):
        df.iloc[:,i] = df.iloc[:,i].rolling(window_len, min_periods=1, center=True).median()


# standardize related
# ----------------------------------------------
def replace_w_median(df: pd.DataFrame, col: str):
    """Replace the values in a <df> column with its median value"""
    df[col + '_x'] = df[col + '_x'].median()
    df[col + '_y'] = df[col + '_y'].median()


def set_origin(df: pd.DataFrame, col: str):
    """Set the [col] as origin for all coordiantes"""
    x_adjustment = df[col + '_x'].iloc[0]
    y_adjustment = df[col + '_y'].iloc[0]

    for col in XCOLS:
        df[col] -= x_adjustment
    for col in YCOLS:
        df[col] -= y_adjustment


class L_transformer():
    """A linear transformer in R2"""

    def __init__(self, cos=1.0, sin=0.0, scale=1.0):
        self.scale = scale
        self.rotation = np.array([[cos, -sin], [sin, cos]])

    def transform(self, X):
        return np.dot(self.rotation, X.T).T * self.scale


def fillna(df: pd.DataFrame):
    """Handle NaNs with an average step method"""

    for col_name in df:
        col = df[col_name]

        # index of missing values
        idx = col[col.isnull()].index

        if idx.any():

            # group consecutive index
            temp = [idx[0]]
            groups = []
            for i in idx[1:]:
                # add to temp list if the element is consecutive
                if (i - temp[-1] == 1):
                    temp.append(i)
                # if the element is not consevutive,
                # complete the current group and reset temp list
                else:
                    groups.append(temp)
                    temp = [i]
            groups.append(temp)

            # fillNaN progressively, using average step of next-prev value
            out_of_bound = False
            for group in groups:
                try:
                    pre = col[(group[0] - 1)]   # previous non-empty value
                    nex = col[(group[-1] + 1)]  # next non-empty value
                    steps = len(group)
                    step = (nex - pre) / steps  # step value

                    for i in group:
                        pre += step
                        col[i] = pre

                # happens when the 1st/last element is NaN so there's no
                # preceding/following value to fill
                except KeyError:
                    out_of_bound = True
                    continue

            # fill remaining value if error occurred
            if out_of_bound:
                col.fillna(method='bfill', inplace=True)
                col.fillna(method='ffill', inplace=True)


def standardize(data: pd.DataFrame):
    """Standardize a csv output from DeepLabCut"""
    
    data.drop(CCOLS, axis=1, inplace=True)
    
    # flip
    data[YCOLS] *= -1

    # remove fluctuation in reference points
    for col in REFE:
        replace_w_median(data, col)

    # make lower left corner the origin
    set_origin(data, 'll_corner')

    # prepare linear transformation
    adj = data['lr_corner_x'].iloc[0]
    opp = - data['lr_corner_y'].iloc[0]
    hyp = dist1((adj, opp))
    transformer = L_transformer(cos=(adj / hyp), sin=(opp / hyp),
                                scale=(TRAY_LENGTH / hyp))

    # apply transformation (rotate + scale)
    for xcol, ycol in zip(XCOLS, YCOLS):
        data[[xcol, ycol]] = transformer.transform(data[[xcol, ycol]].values)
    
    # fill missing values
    fillna(data)
    
    return data.round(decimals=4)

# ----------------------------------------------


def statistics(data: pd.DataFrame):
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
