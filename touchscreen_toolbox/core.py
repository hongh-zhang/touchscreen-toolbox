import os
import logging
from . import utils
from . import postprocess
from . import config as cfg
from . import pose_estimation as pe

logger = logging.getLogger(__name__)



def analyze_folder(folder_path: str, recursive: bool = False, **kwargs) -> None:
    """
    Analyze a folder (recursively)
    """

    logger.info(f"Start analyzing {folder_path}...")

    if recursive and utils.is_generated(folder_path):
        return 1

    for i in utils.listdir(folder_path):
        try:
            targ_path = os.path.join(folder_path, i)

            if os.path.splitext(i)[1] in cfg.FORMATS:
                analyze_video(targ_path, **kwargs)

            elif recursive and os.path.isdir(targ_path):
                analyze_folder(targ_path, recursive=recursive, **kwargs)

        except Exception as e:
            logger.warning(f"Exception encoutered for {i}")
            logger.exception(e)

    postprocess.record_stats(folder_path)



def analyze_video(
    video_path: str, pose: bool = False, post: bool = False, time_file: str = False
) -> None:
    """
    Analyze a video
    """

    logger.info(f"Analyzing '{video_path}'...")

    # initialize result folders + get video info
    vid_info = initialize(video_path, time_file=time_file)

    # pose estimate
    if pose:
        logger.info("Pose estimating...")
        pe.preprocess_video(vid_info)
        pe.dlc_analyze(vid_info)
        pe.cleanup(vid_info)
        utils.save_info(vid_info)

    # postprocess
    if post:
        logger.info("Postprocessing...")
        data = pe.read_dlc_csv(vid_info)
        data = postprocess.refine_data(data)
        data = postprocess.standardize_data(data)
        data = postprocess.engineering(data)
        postprocess.save_data(vid_info, data)
    
    logger.info("Done!")



def initialize(video_path: str, time_file: str = False) -> dict:
    """Initialize result folders and get video info"""

    vid_info = utils.get_vid_info(video_path, time_file=time_file)
    utils.initialize_folders(vid_info)

    return vid_info
