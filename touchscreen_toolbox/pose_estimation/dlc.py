# Scripts to integrate DeepLabCut

import os
import logging
import pandas as pd
import deeplabcut as dlc
from typing import Union
import touchscreen_toolbox.utils as utils
import touchscreen_toolbox.config as cfg

logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def analyze(vid_info: dict) -> None:
    dlc_analyze(vid_info)
    cleanup(vid_info)


def dlc_analyze(vid_info: dict) -> None:
    """Call DLC to analyze video"""

    if "files" in vid_info and "result" in vid_info:
        logger.info("Vid info contain processed files, skipping...")
        return None

    curr_files = utils.find_files(vid_info["dir"])

    dlc.analyze_videos(
        cfg.DLC_CONFIG, vid_info["target_path"], videotype=".mp4", batchsize=32
    )
    dlc.analyze_videos_converth5_to_csv(vid_info["dir"], videotype=".mp4")

    new_files = [f for f in utils.find_files(vid_info["dir"]) if f not in curr_files]
    csv = [f for f in new_files if f.endswith(".csv")][0]
    vid_info["files"] = new_files
    vid_info["dlc_result"] = csv


def cleanup(vid_info: dict) -> None:  # TODO:  enter/exit to cover all files (e.g. _r.mp4)
    """Relocate pose estimation files into the DLC folder"""

    # relocate
    curr_dir = vid_info["dir"]
    targ_dir = os.path.join(vid_info["dir"], cfg.DLC_FOLDER)

    if vid_info["path"] != vid_info["target_path"]:
        vid_info["files"].append(os.path.basename(vid_info["target_path"]))
    utils.move_files(vid_info["files"], curr_dir, targ_dir)

    # rewrite file path
    vid_info["files"] = [os.path.join(cfg.DLC_FOLDER, x) for x in vid_info["files"]]
    vid_info["dlc_result"] = os.path.join(cfg.DLC_FOLDER, vid_info["dlc_result"])


def read_dlc_csv(path: Union[str, dict], frames: tuple = None) -> pd.DataFrame:
    """
    Read pose estimation result produced by DLC
    """
    if type(path) == dict:  # vid_info
        if 'frames' in path:
            return read_dlc_csv(os.path.join(path["dir"], path["dlc_result"]), path['frames'])
        else:
            return read_dlc_csv(os.path.join(path["dir"], path["dlc_result"]))
    
    elif type(path) == str:  # csv path
        data = pd.read_csv(path,
                           skiprows=[0, 1, 2, 3],
                           names=(["frame"] + cfg.HEADERS)
                           ).set_index("frame")
        if frames is not None:
            return data.iloc[frames[0]:frames[1], :]
        else:
            return data
    else:
        raise TypeError("Invalid input type")

        
def dlc_label_video(video_path: str):
    return dlc.create_labeled_video(cfg.DLC_CONFIG, os.path.abspath(video_path),
                                    videotype='mp4', save_frames = False, filtered=False)