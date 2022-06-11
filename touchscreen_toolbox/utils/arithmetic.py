from math import pi
import numpy as np
import touchscreen_toolbox.config as cfg


def dist1(point):
    """Euclidean distance from the origin"""
    return np.linalg.norm(point, ord=2)


def dist2(point1, point2):
    """Euclidean distance between 2 points"""
    return np.linalg.norm(point2 - point1, ord=2)


def absmin(x1, x2):
    """Absolute minimum from 2 1D sequences"""
    a1 = np.abs(x1)
    a2 = np.abs(x2)
    idx = (a1 < a2).astype(int)
    return x1*idx + x2*(1-idx)


# time conversion
def frame2sec(frame: int): return frame / cfg.FPS
def sec2frame(sec: float): return np.round(cfg.FPS * sec).astype(int)