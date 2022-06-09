import os
import re
import sys
import shutil
import numpy as np
import pandas as pd
from time import (localtime, strftime)
from moviepy.editor import VideoFileClip
import touchscreen_toolbox.config as cfg


# vid_info related
# ------------------------
PATTERN = r"^(\d+) - (\S+) - (\d{2}-\d{2}-\d{2}) (\d{2}-\d{2}) (\S+)(\.\S+)"
ELEMENTS = ['mouse_id', 'chamber', 'date', 'time', 'suffix', 'format']


def decode_name(video_name: str):
    """
    Extract video information from <video_name> into dictionary
    
    Returns
    -------
    success: bool
        whether the operation succeded
    
    vid_info: dict
        dictionary of video information to be accessed by other functions
    """
    
    success  = False
    vid_info = {}
    
    try:
        # decode & save to vid_info
        matched = [''.join(i.split('-')) for i in re.match(PATTERN, video_name).groups()]
        sucess = True
        vid_info = {i:j for i,j in zip(ELEMENTS, matched)}
        vid_info['video_name'] = video_name
    
    except AttributeError:
        print(f"Pattern unmatched: {video_name}")
    
    return success, vid_info


def get_vid_info(video_path):
    """Get dictionary of video information from <video_path>"""
    
    # deconstruct information in video name
    success, vid_info = decode_name(os.path.basename(video_path))
    
    # count video length
    vid_info['length'] = get_vid_len(video_path)
    
    return vid_info


def get_vid_len(video_path):
    """Get video duration (sec)"""
    clip = VideoFileClip(video_path)
    return clip.duration


def get_time(vid_info: dict, time_file: str):
    """
    Get time to cut video from <time_file>
    
    Args
    -------
    vid_info: dict
        video information
    time_file: str
        path to time file
        
    Returns
    ------
    start, end: float
        time (in sec) to cut video
    """
    
    times = pd.read_csv(time_file).set_index(['id', 'date'])
    video_time = times.loc[(int(vid_info['mouse_id']), int(vid_info['date']))]
    
    start = video_time['vid_start']
    end   = video_time['vid_end']
    
    return start, end