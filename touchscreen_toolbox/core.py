from . import utils
from . import extract
from . import postprocess


def analyze_video(video_path: str, extract: bool=False, post: bool=False) -> None:
    """
    Analyze a video
    """
    
    # initialize result folders + get video info
    vid_info = initialize(video_path)
    
    # pose estimate
    if extract:
        extract.preprocess_video(vid_info)
        extract.dlc_analyze(vid_info)
    # dlc files -> vid_info
    # -> pickle vid_info
    
    # postprocess
    if post:
        ...
        
        
def initialize(video_path: str) -> dict:
    """Initialize result folders and get video info"""
    
    vid_info = utils.get_vid_info(video_path)
    utils.initialize_folders(vid_info)
    
    return vid_info