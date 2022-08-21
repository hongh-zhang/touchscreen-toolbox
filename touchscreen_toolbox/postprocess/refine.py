import numpy as np
import pandas as pd
import scipy.signal

import touchscreen_toolbox.config as cfg


# prediction refinement
# ----------------------------------------------
def refine_data(data: pd.DataFrame) -> pd.DataFrame:
    """Refine pose estimation data"""
    data = data.copy()
    data = cutoff(data)
    data = data.drop(columns=cfg.CCOLS)
    data = median_filter(data)
    data = savgol_filter(data)
    return data


def cutoff(data: pd.DataFrame, p_cutoff: float = cfg.P_CUTOFF):
    """
    Drop low confidence prediction
    """

    data = data.copy()

    # iterate columns in group of three (x, y, confidence) for each body part
    for i in range(0, data.shape[1], 3):
        idx = np.where(data.iloc[:, i + 2] < p_cutoff)
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


def savgol_filter(data: pd.DataFrame, window_len: int = 5, polyorder: int = 1, deriv=0, delta=1.0) -> pd.DataFrame:
    """
    Smooth trajectory with Savitzky-Golay filter
    """

    data = data.copy()

    for i in range(data.shape[1]):
        data.iloc[:, i] = scipy.signal.savgol_filter(data.iloc[:, i], window_len, polyorder=polyorder, deriv=deriv,
                                                     delta=delta)

    return data
