import os
import pandas as pd
from itertools import groupby
from touchscreen_toolbox import utils
from touchscreen_toolbox import video_info
import touchscreen_toolbox.config as cfg
from touchscreen_toolbox.pose_estimation.dlc import read_dlc_csv

import logging

logger = logging.getLogger(__name__)


def get_stats(data: pd.DataFrame):
    """Produce statistics about miss predicted values"""
    frames = len(data)
    value = [frames]
    for col_name in cfg.CCOLS:
        col = data[col_name].fillna(0)
        zeros = col == 0
        nums = zeros.sum()
        percent = round(nums / frames, 2)
        consecutive = (
            max([len(list(g)) for k, g in groupby(zeros) if k]) if nums > 0 else 0
        )
        first = str(round(col.quantile(q=0.01), 2))
        tenth = str(round(col.quantile(q=0.10), 2))
        value += [nums, percent, consecutive, first, tenth]
    return value


def record_stats(folder_path: str):
    """Record statistics of an analyzed folder"""
    if not utils.is_valid_folder(folder_path):
        logger.warning(f"Invalid folder for record_stats: {folder_path}")
        return 1
    
    # values of output csv
    # format headers
    values = [cfg.HEAD1, cfg.HEAD2]

    info_folder = os.path.join(folder_path, cfg.INF_FOLDER)
    for file in utils.listdir(info_folder):

        # get analyze output for the video
        vid_info = video_info.load_info(os.path.join(info_folder, file))
        csv_file = os.path.join(folder_path, vid_info["result"])

        # get data
        info = video_info.export_info(vid_info)
        try:
            stats = read_dlc_csv(vid_info)
        except:  # skip files without timestamp
            continue
        stats = get_stats(stats)
        values.append(info + stats)

    # save
    save_path = os.path.join(folder_path, cfg.RST_FOLDER, cfg.STATS_NAME)
    pd.DataFrame(values).to_csv(save_path, index=False, header=False)

    logger.info(f"Recorded statistics for {folder_path}")
