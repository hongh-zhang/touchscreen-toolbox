import os
import logging
from . import utils
from . import extract
from . import postprocess
from . import config as cfg
logger = logging.getLogger(__name__)

def analyze_folder(folder_path: str, recursive: bool=True, **kwargs) -> None:
    """
    Analyze a folder (recursively)
    """
    
    logger.info(f"Start analyzing {folder_path}...")
    
    if recursive and (os.path.basename(folder_path) in cfg.FOLDERS):
        return 1
    
    for i in os.listdir(folder_path):
        
        try:
            name, ext = os.path.splitext(i)
            targ_path = os.path.join(folder_path, i)

            if ext in cfg.FORMATS:
                analyze_video(targ_path, **kwargs)

            elif recursive and len(ext)==0:
                analyze_folder(targ_path, recursive=recursive, **kwargs)
                
        except Exception as e:
            logger.exception(e)
            
    postprocess.record_stats(folder_path)


def analyze_video(video_path: str, pose: bool=False, post: bool=False, time_file: str=False) -> None:
    """
    Analyze a video
    """
    
    logger.info(f"Analyzing '{video_path}'...")
    
    # initialize result folders + get video info
    vid_info = initialize(video_path, time_file=time_file)
    
    # pose estimate
    if pose:
        logger.info("Pose estimating...")
        extract.preprocess_video(vid_info)
        extract.dlc_analyze(vid_info)
        extract.cleanup(vid_info)
        utils.save_info(vid_info)
    
    # postprocess
    if post:
        logger.info("Postprocessing...")
        data = extract.read_dlc_csv(vid_info)
        data = postprocess.refine(data)
        data = postprocess.standardize(data)
#         postprocess.feature(vid_info)
        postprocess.save_data(vid_info, data)
        
        
def initialize(video_path: str, time_file: str=False) -> dict:
    """Initialize result folders and get video info"""
    
    vid_info = utils.get_vid_info(video_path, time_file=time_file)
    utils.initialize_folders(vid_info)
    
    return vid_info