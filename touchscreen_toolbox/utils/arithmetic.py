from math import pi
import numpy as np
import touchscreen_toolbox.config as cfg


def absmin(x1, x2):
    """Absolute minimum from two 1D sequences"""
    a1 = np.abs(x1)
    a2 = np.abs(x2)
    idx = (a1 < a2).astype(int)
    return x1 * idx + x2 * (1 - idx)


def absmax(x1, x2):
    """Absolute maximum"""
    a1 = np.abs(x1)
    a2 = np.abs(x2)
    idx = (a1 > a2).astype(int)
    return x1 * idx + x2 * (1 - idx)


# distance
# ------
def dist1(point):
    """Euclidean distance from the origin"""
    return np.linalg.norm(point, ord=2)


def dist2(point1, point2):
    """Euclidean distance between 2 points"""
    return np.linalg.norm(point2 - point1, ord=2)


# angles
# ------
def convert_angles(angles, radians: bool):
    """Convert range of angles from [-pi, pi] to [0, 2pi] + optional degree conversion"""
    angles += (angles < 0).astype(int) * 2 * pi
    if not radians:
        angles *= 180 / pi
    return angles


def angle1(v, radians=False):
    """Angle between vector <v> and horizontal axis"""
    angles = np.arctan2(v[:, 1], v[:, 0])
    return convert_angles(angles, radians)


def angle2(v, u, radians=False):
    """Relative angle between two vectors <v> & <u> with respect to the origin"""
    angles = np.arctan2(u[:, 1], u[:, 0]) - np.arctan2(v[:, 1], v[:, 0])
    return convert_angles(angles, radians)


def angle3(pts1, pts2, pts3, radians=False):
    """Relative angle between three points, taking <pts2> as vertice"""
    pts1 = pts1 - pts2
    pts3 = pts3 - pts2
    return angle2(pts1, pts3, radians=radians)


def absangle(v, u, radians=False):
    """Absolute angle, defined as the angle between pt1, pt2, horizontal axis"""
    w = v - u
    angles = np.arctan2(w[:, 1], w[:, 0])
    return convert_angles(angles, radians)


# time conversion
# ------
def frame2sec(frame: int):
    return frame / cfg.FPS


def sec2frame(sec: float):
    return np.round(cfg.FPS * sec).astype(int)
