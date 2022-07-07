import os
import sys
import numpy as np
import pandas as pd
from itertools import groupby
import touchscreen_toolbox.config as cfg
import touchscreen_toolbox.utils as utils



def standardize_data(data: pd.DataFrame) -> None:
    """Standardize pose estimation data"""
    
    data = data.copy()
    
    # flip
    data[cfg.YCOLS] *= -1

    data = replace_w_median(data, cfg.REFE)
    data = set_origin(data, "ll_corner")

    # apply transformation (rotate + scale)
    adj = data["lr_corner_x"].iloc[0]
    opp = -data["lr_corner_y"].iloc[0]
    hyp = utils.dist1((adj, opp))
    transformer = L_transformer(
        cos=(adj / hyp), sin=(opp / hyp), scale=(cfg.TRAY_LENGTH / hyp)
    )
    for xcol, ycol in zip(cfg.XCOLS, cfg.YCOLS):
        data[[xcol, ycol]] = transformer.transform(data[[xcol, ycol]].values)
    
    data = fillna(data)
    
    return data.round(decimals=cfg.DECIMALS)



def replace_w_median(data: pd.DataFrame, columns: list):
    """Replace the values in <columns> with its median value"""
    data = data.copy()
    
    for col in columns:
        data[col + "_x"] = data[col + "_x"].median()
        data[col + "_y"] = data[col + "_y"].median()
    return data


def set_origin(data: pd.DataFrame, col: str):
    """Set the [col] as origin for all coordiantes"""
    
    data = data.copy()
    
    x_adjustment = data[col + "_x"].iloc[0]
    y_adjustment = data[col + "_y"].iloc[0]

    for col in cfg.XCOLS:
        data[col] -= x_adjustment
    for col in cfg.YCOLS:
        data[col] -= y_adjustment
    return data


class L_transformer:
    """A linear transformer in R2"""

    def __init__(self, cos=1.0, sin=0.0, scale=1.0):
        self.scale = scale
        self.rotation = np.array([[cos, -sin], [sin, cos]])

    def transform(self, X):
        return np.dot(self.rotation, X.T).T * self.scale


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