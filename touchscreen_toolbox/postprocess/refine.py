import os
import numpy as np
import pandas as pd
from itertools import groupby
import touchscreen_toolbox.config as cfg
import touchscreen_toolbox.utils as utils
from touchscreen_toolbox.pose_estimation.dlc import read_dlc_csv


# prediction refinement
# ----------------------------------------------
def refine(data: pd.DataFrame):
    data = cutoff(data)
    data = data.drop(columns=cfg.CCOLS)
    data = median_filter(data)
    data = fillna(data)

    return data


def save_data(vid_info: dict, data: pd.DataFrame):
    data.to_csv(
        os.path.join(vid_info["dir"], cfg.RST_FOLDER, vid_info["vid_name"] + ".csv")
    )


def cutoff(data: pd.DataFrame, p_cutoff: float = cfg.P_CUTOFF):
    """
    Drop low confidence prediction
    """

    data = data.copy()

    # iterate columns in group of three (x, y, confidence) for each body part
    for i in range(0, data.shape[1], 3):
        idx = data.index[data.iloc[:, i + 2] < p_cutoff]
        data.iloc[idx, i] = np.NaN
        data.iloc[idx, i + 1] = np.NaN

    return data


def median_filter(data: pd.DataFrame, window_len: int = 5):
    """
    Apply sliding median filter to all columns in <df>
    """

    data = data.copy()

    # apply filter to each column
    # keep NaN
    for i in range(data.shape[1]):
        idx = np.where(np.isnan(data.iloc[:, i]))
        data.iloc[:, i] = (
            data.iloc[:, i]
            .rolling(window_len, min_periods=1, center=True)
            .median(skipna=False)
        )
        data.iloc[idx, i] = np.NaN

    return data


def fillna(data: pd.DataFrame):
    """
    Replace NaNs with step-average of neighbouring prediction
    """

    data = data.copy()

    for col_name in data:
        col = data[col_name]

        # index of missing values
        idx = col[col.isnull()].index

        if idx.any():

            # group consecutive index
            temp = [idx[0]]
            groups = []
            for i in idx[1:]:
                # add to temp list if the element is consecutive
                if i - temp[-1] == 1:
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
                    pre = col[(group[0] - 1)]  # previous non-empty value
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
                col.fillna(method="bfill", inplace=True)
                col.fillna(method="ffill", inplace=True)

    return data
