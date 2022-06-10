from . import utils
from . import extract
from . import postprocess


def analyze_video(video_path, post=False):
    
    # initialize result folders + get video info
    vid_info = utils.initialize(video_path)
    
    # pose estimate
    extract.preprocess_video(vid_info) # target_path->vid_info
    extract.dlc_analyze(vid_info) # <- target_path
    # dlc files -> vid_info
    # -> pickle vid_info
    
    # postprocess
    if post:
        ...