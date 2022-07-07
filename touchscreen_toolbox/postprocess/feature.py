import os
import sys
import numpy as np
import pandas as pd
from math import pi
from touchscreen_toolbox import utils
from touchscreen_toolbox import config as cfg



def engineering(data: pd.DataFrame, fps) -> None:
    """Feature engineering (*hardcoded)"""
    
    data = data.copy()
    
    data['time'] = data['frame'] / fps
    
    # (absolute) orientation
    neck = (select_bodypart(data, 'spine1') + select_bodypart(data, 'lEar') + select_bodypart(data, 'rEar')) / 3
    h_angle = utils.absangle(select_bodypart(data, 'snout'), neck)
    data['head_angle'] = h_angle
    data['v-head_angle'] = get_angle_v(h_angle)
    b_angle = utils.absangle(neck, select_bodypart(data, 'tail1'))
    data['body_angle'] = b_angle
    data['v-body_angle'] = get_angle_v(b_angle)

    
    # body length & angle
    data['snout-tail'] = distance(data, 'snout', 'tail1')
    data['snout-spine1'] = distance(data, 'snout', 'spine1')
    data['spine1-spine2'] = distance(data, 'spine1', 'spine2')
    data['spine2-tail1'] = distance(data, 'spine2', 'tail1')
    
    # (relative) angle
    data['snout-spine1-spine2'] = utils.angle3(select_bodypart(data, 'snout'),
                                               select_bodypart(data, 'spine1'),
                                               select_bodypart(data, 'spine2'))
    data['spine1-spine2-tail1'] = utils.angle3(select_bodypart(data, 'spine1'),
                                               select_bodypart(data, 'spine2'),
                                               select_bodypart(data, 'tail1'))
    

    # snout to key points
    d_cols = []
    for col in ("l_screen", "m_screen", "r_screen", "food_port"):
        d_cols.append("snout-" + col)
        data[d_cols[-1]] = distance(data, "snout", col)

    # velocity
    v_cols = []
    v_cols.append("v-snout")
    data[v_cols[-1]] = velocity2(data, "snout")
    for col in d_cols:
        v_cols.append("v-" + col)
        data[v_cols[-1]] = velocity1(data, col)

    # acceleration
    for col in v_cols:
        data["a-" + col[2:]] = velocity1(data, col)
    return data.round(decimals=cfg.DECIMALS)



# helpers
# -------
def distance(data: pd.DataFrame, pt1: str, pt2: str):
    """Distance between a pair of keypoints"""
    dx = data[pt1 + "_x"] - data[pt2 + "_x"]
    dy = data[pt1 + "_y"] - data[pt2 + "_y"]
    return np.sqrt(dx ** 2 + dy ** 2)


def velocity1(data: pd.DataFrame, col: str):
    """1D Velocity of scalar values (from distance)"""
    return np.diff(data[col], prepend=data[col].iloc[0])


def velocity2(data: pd.DataFrame, col: str):
    """2D Velocity of vector values (from coordinates)"""
    dx = np.diff(data[col + "_x"], prepend=data[col + "_x"].iloc[0])
    dy = np.diff(data[col + "_x"], prepend=data[col + "_x"].iloc[0])
    return np.sqrt(dx ** 2 + dy ** 2)


def select_bodypart(data, bodypart):
    """Index bodypart coordinates as 2d array"""
    return data[[bodypart+'_x', bodypart+'_y']].values


def get_angle_v(angles: pd.DataFrame):
    """Continuous angular velocity"""
    angles2 = (angles<180).astype(int) * 180 + angles  # ~[180, 540] for continuity
    v_angles =  utils.absmin(np.diff(angles, prepend=angles[0]), 
                             np.diff(angles2, prepend=angles2[0]))
    return v_angles