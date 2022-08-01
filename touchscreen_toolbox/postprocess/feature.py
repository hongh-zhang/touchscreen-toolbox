# scripts for feature engineering

# abbreviations:
# 'd': distance
# 'v': velocity
# 'a': acceleration
# 'ang': angle
# 'angv': angular velocity


import os
import sys
import numpy as np
import pandas as pd
from math import pi
from touchscreen_toolbox import utils
from touchscreen_toolbox import config as cfg
from itertools import combinations


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
    from itertools import combinations
    new = pd.DataFrame()
    
    keypoints = ("snout", "spine1", "spine2", "tail1")
    
    # body length
    for pair in list(combinations(keypoints, 2)):
        pair_name = '-'.join(pair)
        new['d-'+pair_name] = get_distance(data, pair[0], pair[1])
        new['v-'+pair_name] = velocity1(new, 'd-'+pair_name)
    
    # body (relative) angle
    for triplet in list(combinations(keypoints, 3)):
        trip_name = '-'.join(triplet)
        new['ang-'+trip_name] = utils.angle3(select_bodypart(data, triplet[0]),
                                              select_bodypart(data, triplet[1]),
                                              select_bodypart(data, triplet[2]))
        new['angv-'+trip_name] = velocity1(new, 'ang-'+trip_name)

    # velocity, acceleration
    for point in keypoints:
        new['v-'+point] = velocity2(data, point)
        new['a-'+point] = velocity1(new, 'v-'+point)
    
    return new.round(decimals=cfg.DECIMALS)



def external_behaviour(data: pd.DataFrame):
    """Behaviour relative to task stimuli"""
    
    new = pd.DataFrame()
    
    # orientation
    neck = (select_bodypart(data, 'spine1') + select_bodypart(data, 'lEar') + select_bodypart(data, 'rEar')) / 3
    h_angle = utils.absangle(select_bodypart(data, 'snout'), neck)
    new['head_ang'] = h_angle
    new['head_angv'] = get_angv(h_angle)
    
    #  relative to key points
    for col in ("l_screen", "m_screen", "r_screen", "food_port"):
        new_col = "snout-"+col

        # distance & velocity
        dist = get_distance(data, "snout", col).values
        new["d-"+new_col] = dist
        new["v-"+new_col] = np.diff(dist, prepend=dist[0])


        # relative angle to the target (snout-neck-screen/port)
        angle = utils.angle3(select_bodypart(data, 'snout'),
                             neck,
                             select_bodypart(data, col))
        new['ang-'+new_col] = angle

        # angular velocity relative to the target
        angv = get_angv(angle)
        
        # reverse the sign if angle < 180
        # so that a positive number means getting closer to 0 degrees (orienting towards the target)
        # negative means getting closer to 180 degrees (orienting away from the target)
        angv = angv * (angle>180).astype(int) + -angv * (angle<180).astype(int)
        new['angv-'+new_col] = angv
    
    return new.round(decimals=cfg.DECIMALS)



def make_multiindex(df: pd.DataFrame, name: str):
    """Add a top layer index <name> to all columns in <df>"""
    df = df.copy()
    multi_index_level_0 = [name for col in df.columns]
    multi_index = [multi_index_level_0, df.columns.values]
    df.columns = pd.MultiIndex.from_arrays(multi_index)
    return df.convert_dtypes()



# helpers
# -------
def get_distance(data: pd.DataFrame, pt1: str, pt2: str):
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


def get_angv(angles: pd.DataFrame):
    """Continuous angular velocity"""
    
    # raw velocity
    # range ~ [-360, 360]
    angv =  np.diff(angles, prepend=angles[0])
    
    # continuouts velocity
    # if angv is outside [-180,180], plus/minus 360 to map into [-180,180]
    angv -= (np.abs(angv) > 180).astype(int) * 360 * ((angv > 0).astype(int)*2 - 1)
    return angv
