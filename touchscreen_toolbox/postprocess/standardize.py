import os
import sys
import numpy as np
import pandas as pd
from itertools import groupby
import touchscreen_toolbox.config as cfg
import touchscreen_toolbox.utils as utils


# standardize related
# ----------------------------------------------
def replace_w_median(df: pd.DataFrame, col: str):
    """Replace the values in a <df> column with its median value"""
    df[col + "_x"] = df[col + "_x"].median()
    df[col + "_y"] = df[col + "_y"].median()


def set_origin(df: pd.DataFrame, col: str):
    """Set the [col] as origin for all coordiantes"""
    x_adjustment = df[col + "_x"].iloc[0]
    y_adjustment = df[col + "_y"].iloc[0]

    for col in cfg.XCOLS:
        df[col] -= x_adjustment
    for col in cfg.YCOLS:
        df[col] -= y_adjustment


class L_transformer:
    """A linear transformer in R2"""

    def __init__(self, cos=1.0, sin=0.0, scale=1.0):
        self.scale = scale
        self.rotation = np.array([[cos, -sin], [sin, cos]])

    def transform(self, X):
        return np.dot(self.rotation, X.T).T * self.scale


def standardize(data: pd.DataFrame):
    """Standardize a csv output from DeepLabCut"""

    # flip
    data[cfg.YCOLS] *= -1

    # remove fluctuation in reference points
    for col in cfg.REFE:
        replace_w_median(data, col)

    # make lower left corner the origin
    set_origin(data, "ll_corner")

    # prepare linear transformation
    adj = data["lr_corner_x"].iloc[0]
    opp = -data["lr_corner_y"].iloc[0]
    hyp = utils.dist1((adj, opp))
    transformer = L_transformer(
        cos=(adj / hyp), sin=(opp / hyp), scale=(cfg.TRAY_LENGTH / hyp)
    )

    # apply transformation (rotate + scale)
    for xcol, ycol in zip(cfg.XCOLS, cfg.YCOLS):
        data[[xcol, ycol]] = transformer.transform(data[[xcol, ycol]].values)

    return data.round(decimals=cfg.DECIMALS)
