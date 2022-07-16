import os
import sys
import numpy as np
import pandas as pd
from math import pi
from touchscreen_toolbox import utils
from touchscreen_toolbox import config as cfg



def engineering(data: pd.DataFrame) -> None:
    """Feature engineering (*hardcoded)"""
    
    internal = internal_behaviour(data).set_index(data.index)
    external = external_behaviour(data).set_index(data.index)
    
    data = pd.concat([make_multiindex(data, 'coordinate'),
               make_multiindex(internal, 'internal'),
               make_multiindex(external, 'external')], axis=1)
    
    return data.round(decimals=cfg.DECIMALS)



def internal_behaviour(data: pd.DataFrame):
    """Body configuration & movement"""
    
    new = pd.DataFrame()
    
    # body length
    new['snout-tail'] = distance(data, 'snout', 'tail1')
    new['snout-spine1'] = distance(data, 'snout', 'spine1')
    new['spine1-spine2'] = distance(data, 'spine1', 'spine2')
    new['spine2-tail1'] = distance(data, 'spine2', 'tail1')
    
    # body (relative) angle
    new['snout-spine1-spine2'] = utils.angle3(select_bodypart(data, 'snout'),
                                               select_bodypart(data, 'spine1'),
                                               select_bodypart(data, 'spine2'))
    new['spine1-spine2-tail1'] = utils.angle3(select_bodypart(data, 'spine1'),
                                               select_bodypart(data, 'spine2'),
                                               select_bodypart(data, 'tail1'))
    
    # velocity, acceleration
    new['v-snout'] = velocity2(data, "snout")
    new['a-snout'] = velocity1(new, 'v-snout')
    
    return new



def external_behaviour(data: pd.DataFrame):
    """Behaviour relative to task stimuli"""
    
    new = pd.DataFrame()
    
    # orientation
    neck = (select_bodypart(data, 'spine1') + select_bodypart(data, 'lEar') + select_bodypart(data, 'rEar')) / 3
    h_angle = utils.absangle(select_bodypart(data, 'snout'), neck)
    new['head_angle'] = h_angle
    new['v-head_angle'] = get_angle_v(h_angle)
    b_angle = utils.absangle(neck, select_bodypart(data, 'tail1'))
    new['body_angle'] = b_angle
    new['v-body_angle'] = get_angle_v(b_angle)
    
    # distance & velocity to key points
    for col in ("l_screen", "m_screen", "r_screen", "food_port"):
        new_col = "snout-"+col
        dist = distance(data, "snout", col).values
        new["d-"+new_col] = dist
        new["v-"+new_col] = np.diff(dist, prepend=dist[0])
    
    return new



def make_multiindex(df: pd.DataFrame, name: str):
    """Add a top layer index <name> to all columns in <df>"""
    df = df.copy()
    multi_index_level_0 = [name for col in df.columns]
    multi_index = [multi_index_level_0, df.columns.values]
    df.columns = pd.MultiIndex.from_arrays(multi_index)
    return df.convert_dtypes()



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
    dy = np.diff(data[col + "_y"], prepend=data[col + "_y"].iloc[0])
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
