import os
import re
import cv2
import json
import logging
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip

from .utils import io
from . import config as cfg

logger = logging.getLogger(__name__)



def get_vid_info(video_path: str, overwrite: bool = False, verbose: bool = True):
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
        os.path.join(vid_info["dir"], cfg.INF_FOLDER, vid_info["vid_name"]) + ".json"
    )

    # read saved info
    if os.path.exists(vid_info["save_path"]) and (not overwrite):
        vid_info = load_info(vid_info)
        if verbose:
            logger.info(f"Loaded existing info from {vid_info['save_path']}")

    else:
        # deconstruct information in video name
        success, name_info = decode_name(vid_info["vid_name"])
        vid_info.update(name_info)
        
        vid_info["length"] = get_vid_len(video_path)
        vid_info["fps"] = get_vid_fps(video_path)

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


def get_time(vid_info: dict, time_file: str, buffer=cfg.TIME_BUFFER) -> bool:
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
    
    try:
        times = pd.read_csv(time_file).set_index(["id", "date"])
        video_time = times.loc[(int(vid_info["mouse_id"]), int(vid_info["exp_date"]))]

        start = max(0, video_time["vid_start"] + buffer[0])
        end = min(vid_info['length'], video_time["vid_end"] + buffer[1])
        assert start < end

        vid_info["time"] = (start, end)
        vid_info['frames'] = (round(start * vid_info['fps']),
                              round(end * vid_info['fps']))

        return True
    
    except KeyError:  # when the id-date pair is not found in timefile
        return False


def save_info(vid_info: dict) -> None:
    """Save vid info to the INFO child folder"""

    file_path = os.path.join(vid_info["dir"], cfg.INF_FOLDER, vid_info["vid_name"]+'.json')
    with open(file_path, 'w') as f:
        json.dump(vid_info, f, indent=4, sort_keys=True)
        
def load_info(vid_info) -> None:
    """Load saved vid_info from json, updates file path if changed"""
    
    if type(vid_info) == dict:
        file_path = os.path.join(vid_info["dir"], cfg.INF_FOLDER, vid_info["vid_name"]+'.json')
    elif type(vid_info) == str:
        file_path = vid_info
    else:
        raise TypeError("Invalid input type - accept either string or dictionary")
        
    with open(file_path, 'r') as f:
        saved = json.load(f)
    
    if type(vid_info) == dict:
        saved.update(vid_info)
    return saved


def export_info(vid_info: dict) -> list:
    val_ls = [str(vid_info.get(elem, "NA")) for elem in cfg.INFO_LS]
    return val_ls


def save_data(vid_info: dict, data: pd.DataFrame, csv: bool=True) -> None:
    """Save data to result folder, default in hd5f format"""
    save_path = os.path.join(cfg.RST_FOLDER, vid_info["vid_name"])
    if csv:
        save_path += ".csv"
        data.to_csv(os.path.join(vid_info["dir"], save_path))
    else:
        save_path += ".h5"
        h5key = vid_info['mouse_id'] + '-' + vid_info['exp_date']
        data.astype(float).replace({pd.NA: np.nan}).to_hdf(os.path.join(vid_info["dir"], save_path), h5key)  #<- TODO: fix this stupid type conversion
    vid_info['post_result'] = save_path