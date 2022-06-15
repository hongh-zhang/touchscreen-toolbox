import os
import re
import cv2
import sys
import shutil
import numpy as np
import pandas as pd
from time import (localtime, strftime)
from moviepy.editor import VideoFileClip
import touchscreen_toolbox.config as cfg
from . import io



def get_vid_info(video_path: str, overwrite: bool=False):
    """Get dictionary of video information from <video_path>"""
    
    vid_info = {'path': video_path, 
                'target_path': video_path, 
                'dir' : os.path.dirname(video_path),
                'file_name': os.path.basename(video_path),
                'vid_name': os.path.splitext(os.path.basename(video_path))[0],
                'format': os.path.splitext(os.path.basename(video_path))[1]}
    vid_info['save_path'] = os.path.join(vid_info['dir'], cfg.INF_FOLDER, vid_info['vid_name'])+'.pickle'
    
    # read saved info
    if os.path.exists(vid_info['save_path']) and (not overwrite):
        load_info(vid_info)
        print(f"Loaded existing {vid_inf['save_path']}")
        
    else:
        # deconstruct information in video name
        success, name_info = decode_name(vid_info['vid_name'])
        vid_info.update(name_info)

        vid_info['length'] = get_vid_len(video_path)
        vid_info['fps'] = get_vid_fps(video_path)
    
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
    
    success  = False
    name_info = {}
    
    try:
        # decode & save to vid_info
        matched = [''.join(i.split('-')) for i in re.match(cfg.PATTERN, video_name).groups()]
        sucess = True
        name_info = {i:j for i,j in zip(cfg.ELEMENTS, matched)}
    
    except AttributeError:
        print(f"Pattern unmatched: {video_name}")
    
    return success, name_info


def get_vid_len(video_path):
    """Get video duration (sec)"""
    clip = VideoFileClip(video_path)
    return clip.duration

def get_vid_fps(video_path):
    """Get video fps"""

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    
    return fps


def get_time(vid_info: dict, time_file: str):
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
    
    times = pd.read_csv(time_file).set_index(['id', 'date'])
    video_time = times.loc[(int(vid_info['mouse_id']), int(vid_info['exp_date']))]
    
    start = video_time['vid_start']
    end   = video_time['vid_end']
    
    vid_info['time'] = (start, end)
    
    
def save_info(vid_info: dict) -> None:
    """Save vid info to the INFO child folder"""
    
    file_path = os.path.join(vid_info['dir'], cfg.INF_FOLDER, vid_info['vid_name'])
    io.pickle_save(vid_info, file_path+'.pickle')


def load_info(vid_info: dict) -> None:
    file_path = os.path.join(vid_info['dir'], cfg.INF_FOLDER, vid_info['vid_name'])
    vid_info.update(io.pickle_load(file_path+'.pickle'))