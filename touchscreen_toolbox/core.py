import os
import glob
import logging
from . import utils
from . import video_info
from . import postprocess
from . import config as cfg
from . import pose_estimation as pe

logger = logging.getLogger(__name__)

from joblib import Parallel, delayed
def parallel_postprocessing(root_folder, **kwargs) -> None:
    all_videos = list(filter(lambda x: 'DLC' not in x, 
                      glob.glob(os.path.join(root_folder, "**/*.mp4"), recursive=True)))
    results = Parallel(n_jobs=8)(delayed(analyze_video)(video, **kwargs) for video in all_videos)


def analyze_folder(folder_path: str, recursive: bool = False, **kwargs) -> None:
    """
    Analyze a folder (recursively)
    """

    if recursive and utils.is_generated(folder_path):
        logger.info(f"Skipping folder {folder_path}...")
        return 1
    
    logger.info(f"Start analyzing {folder_path}...")
    
    for i in utils.listdir(folder_path):
        targ_path = os.path.join(folder_path, i)

        if os.path.splitext(i)[1] in cfg.FORMATS:
            analyze_video(targ_path, **kwargs)

        elif recursive and os.path.isdir(targ_path):
            analyze_folder(targ_path, recursive=recursive, **kwargs)

    postprocess.record_stats(folder_path)



def analyze_video(
    video_path: str, pose: bool = False, post: bool = False, time_file: str = False, timestamps: str = False, raise_exception: bool = False
) -> None:
    """
    Analyze a video
    """
    try:
        logger.info(f"Analyzing '{video_path}'...")

        # initialize result folders + get video info
        vid_info = initialize(video_path, time_file=time_file)

        # pose estimate
        if pose:
            logger.info("Pose estimating...")
            pe.preprocess_video(vid_info)
            pe.dlc_analyze(vid_info)
            pe.cleanup(vid_info)
            video_info.save_info(vid_info)

        # postprocess
        if post:
            logger.info("Postprocessing...")
            data = pe.read_dlc_csv(vid_info)
            data = postprocess.refine_data(data)
            data = postprocess.standardize_data(data)
            data = postprocess.engineering(data)
            data = postprocess.merge(vid_info, data, timestamps, vid_info['fps'])
            video_info.save_data(vid_info, data)
            video_info.save_info(vid_info)

        logger.info("Done!")
        
    except Exception as e:
        logger.warning(f"Exception encoutered for {video_path}")
        logger.exception(e)
        logger.exception("-"*20)
        if raise_exception:
            raise e


def initialize(video_path: str, time_file: str = False) -> dict:
    """Initialize result folders and get video info"""

    vid_info = video_info.get_vid_info(video_path, time_file=time_file)
    utils.initialize_folders(vid_info)

    return vid_info
