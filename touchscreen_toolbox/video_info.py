import os
import re
import cv2
import logging
import pandas as pd
from moviepy.editor import VideoFileClip

from .utils import io
from . import config as cfg

logger = logging.getLogger(__name__)



def get_vid_info(video_path: str, overwrite: bool = False, time_file: str = False):
    """Get dictionary of video information from <video_path>"""

    vid_info = {
        "path": video_path,
        "target_path": video_path,
        "dir": os.path.dirname(video_path),
        "file_name": os.path.basename(video_path),
        "vid_name": os.path.splitext(os.path.basename(video_path))[0],
        "format": os.path.splitext(os.path.basename(video_path))[1],
    }
    vid_info["save_path"] = (
        os.path.join(vid_info["dir"], cfg.INF_FOLDER, vid_info["vid_name"]) + ".pickle"
    )

    # read saved info
    if os.path.exists(vid_info["save_path"]) and (not overwrite):
        load_info(vid_info)
        logger.info(f"Loaded existing {vid_info['save_path']}")

    else:
        # deconstruct information in video name
        success, name_info = decode_name(vid_info["vid_name"])
        vid_info.update(name_info)
        
        vid_info["length"] = get_vid_len(video_path)
        vid_info["fps"] = get_vid_fps(video_path)
        if time_file:
            get_time(vid_info, time_file)
            vid_info['frames'] = [int(i * cfg.FPS) for i in vid_info['time']]

    return vid_info


def decode_name(video_name: str):
    """
    Extract video information from <video_name> into dictionary,
    pattern matching based on variables in config file

    Returns
    -------
    success: bool
        whether the operation succeded

    vid_info: dict
        dictionary of video information to be accessed by other functions
    """

    success = False
    name_info = {}

    try:
        # decode & save to vid_info
        matched = [
            "".join(i.split("-")) for i in re.match(cfg.PATTERN, video_name).groups()
        ]
        success = True
        name_info = {i: j for i, j in zip(cfg.ELEMENTS, matched)}

    except AttributeError:
        logger.warning(f"Pattern unmatched: {video_name}")

    return success, name_info


def get_vid_len(video_path):
    """Get video duration (sec)"""
    clip = VideoFileClip(video_path)
    return clip.duration


def get_vid_fps(video_path):
    """Get video fps"""

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
#     frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    return fps


def get_time(vid_info: dict, time_file: str, buffer=cfg.TIME_BUFFER) -> None:
    """
    Get time to cut video from <time_file>

    Args
    -------
    vid_info: dict

    time_file: str
        path to time file

    Returns
    ------
    start, end: float
        time (in sec) to cut video
    """

    times = pd.read_csv(time_file).set_index(["id", "date"])
    video_time = times.loc[(int(vid_info["mouse_id"]), int(vid_info["exp_date"]))]

    start = video_time["vid_start"]
    end = video_time["vid_end"] 
    end = (end+buffer) if (end+buffer <= vid_info['length']) else vid_info['length']

    vid_info["time"] = (start, end)


def save_info(vid_info: dict) -> None:
    """Save vid info to the INFO child folder"""

    file_path = os.path.join(vid_info["dir"], cfg.INF_FOLDER, vid_info["vid_name"])
    io.pickle_save(vid_info, file_path + ".pickle")


def load_info(vid_info: dict) -> None:
    file_path = os.path.join(vid_info["dir"], cfg.INF_FOLDER, vid_info["vid_name"])
    vid_info.update(io.pickle_load(file_path + ".pickle"))


def export_info(vid_info: dict) -> list:
    val_ls = [str(vid_info.get(elem, "NA")) for elem in cfg.INFO_LS]
    return val_ls


def save_data(vid_info: dict) -> None:
    pass