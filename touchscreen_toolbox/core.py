import os
from . import utils
from . import extract
from . import postprocess
from . import config as cfg



def analyze_folder(folder_path: str, recursive: bool=True, **kwargs) -> None:
    """
    Analyze a folder (recursively)
    """
    for i in os.listdir(folder_path):
        
        name, ext = os.path.splitext(i)
        targ_path = os.path.join(folder_path, i)
        
        if ext in cfg.FORMATS:
            analyze_video(targ_path, **kwargs)
            
        elif recursive and len(ext)==0:
            analyze_folder(targ_path, recursive=recursive, **kwargs)


def analyze_video(video_path: str, pose: bool=False, post: bool=False, time_file: str=False) -> None:
    """
    Analyze a video
    """
    
    print(f"Analyzing '{video_path}'...")
    
    # initialize result folders + get video info
    vid_info = initialize(video_path)
    if time_file:
        utils.get_time(vid_info, time_file)
    
    # pose estimate
    if pose:
        extract.preprocess_video(vid_info)
        extract.dlc_analyze(vid_info)
        extract.cleanup(vid_info)
        utils.save_info(vid_info)
    
    # postprocess
    if post:
        ...
        
        
def initialize(video_path: str) -> dict:
    """Initialize result folders and get video info"""
    
    vid_info = utils.get_vid_info(video_path)
    utils.initialize_folders(vid_info)
    
    return vid_info