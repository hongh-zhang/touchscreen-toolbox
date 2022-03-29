import os
import sys
import numpy as np
import pandas as pd
from touchscreen_toolbox.utils import *


def distance(data: pd.DataFrame, pt1: str, pt2: str):
    """Distance between a pair of keypoints"""
    dx = data[pt1+'_x'] - data[pt2+'_x']
    dy = data[pt1+'_y'] - data[pt2+'_y']
    return np.sqrt(dx**2+dy**2)


def velocity1(data: pd.DataFrame, col: str):
    """1D Velocity of scalar values (from distance)"""
    return np.diff(data[col], prepend=data[col][0])


def velocity2(data: pd.DataFrame, col: str):
    """2D Velocity of vector values (from coordinates)"""
    dx = np.diff(data[col+'_x'], prepend=data[col+'_x'][0])
    dy = np.diff(data[col+'_x'], prepend=data[col+'_x'][0])
    return np.sqrt(dx**2+dy**2)


def orientation(data: pd.DataFrame, pt1: str, pt2: str):
    """Orientation, defined as the angle between pt1, pt2, horizontal axis"""
    # calculate angle
    dx = data[pt1+'_x'] - data[pt2+'_x']
    dy = data[pt1+'_y'] - data[pt2+'_y']
    angle = np.arctan2(dy, dx)
    
    # cast to [0, 2 pi] range
    angle += (angle < 0).astype(int) * 2 * pi + angle
    
    return angle


def engineering(data : pd.DataFrame):
    # orientation
    data['orientation'] = orientation(data, 'snout', 'tail1')

    # snout to key points
    d_cols = []
    for col in ('l_screen','m_screen','r_screen','food_port'):
        d_cols.append('snout-'+col)
        data[d_cols[-1]] = distance(data, 'snout', col)

    # velocity
    v_cols = []
    for col in d_cols:
        v_cols.append('v-'+col)
        data[v_cols[-1]] = velocity1(data, col)

    v_cols.append('v-snout')
    data[v_cols[-1]] = velocity2(data, 'snout')

    # acceleration
    for col in v_cols:
        data['a-'+col[2:]] = velocity1(data, col)
    return data
