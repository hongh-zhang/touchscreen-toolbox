import os
import json
import shutil
import logging
import pandas as pd
from natsort import os_sorted
import touchscreen_toolbox.config as cfg

logger = logging.getLogger(__name__)


def clear_results(folder_path: str):
    """Remove files in the result folder,
    except for pose estimation statistics"""

    if os.path.basename(folder_path) != cfg.RST_FOLDER:
        clear_results(os.path.join(folder_path, cfg.RST_FOLDER))
    else:
        for file in listdir(folder_path):
            if file != cfg.STATS_NAME:
                os.remove(os.path.join(folder_path, file))


def listdir(path: str):
    if os.path.exists(path):
        return os_sorted(os.listdir(path))
    else:
        logger.warning(f"Invalid input for listdir: {path}")
        return []


def mk_dir(path: str, force: bool = True, verbose: bool = True) -> None:
    """(re)Make directory, deletes existing directory"""
    if os.path.exists(path):
        if force:
            if verbose:
                logger.info(f"Removing existing {os.path.basename(path)} folder")
            shutil.rmtree(path)

        else:
            if verbose:
                logger.info(f"Folder already exists, failed to make directory")
            return None

    os.mkdir(path)


def is_generated(folder_path):
    """Check if folder is generated by toolbox"""
    return os.path.basename(folder_path) in [
        cfg.DLC_FOLDER,
        cfg.RST_FOLDER,
        cfg.INF_FOLDER,
    ]


def move_files(files: list, curr_folder: str, targ_folder: str):
    """Move <files> from <curr_folder> to <targ_folder>"""
    for f in files:
        file_path = os.path.join(curr_folder, f)
        new_path = os.path.join(targ_folder, f)
        if os.path.exists(new_path):
            os.remove(new_path)
        os.rename(file_path, new_path)


def find_files(folder_path: str, prefix: str = "*", suffix: str = "*"):
    return glob.glob(os.path.join(folder_path, prefix+suffix))


def move_dlc_files(vid_info, direction=0):
    """
    Move dlc output files,
    0: DLC folder -> video folder,
    1: video folder -> DLC folder
    """
    dlc_files = map(os.path.basename, vid_info['files'])
    vid_dir = vid_info['dir']
    dlc_dir = os.path.join(vid_dir, cfg.DLC_FOLDER)

    if direction == 0:
        move_files(dlc_files, dlc_dir, vid_dir)
    else:
        move_files(dlc_files, vid_dir, dlc_dir)


# def move_dlc_files(video_path, direction=0):
#     """
#     Move dlc output files,
#     0: DLC folder -> video folder,
#     1: video folder -> DLC folder
#     """
#     video_name = os.path.basename(video_path)
#     video_folder = os.path.dirname(video_path)
#     assert is_valid_folder(video_folder), "Invalid folder"
#     dlc_folder = os.path.join(video_folder, cfg.DLC_FOLDER)

#     if direction == 0:
#         dlc_files = find_files(dlc_folder, prefix=video_name)
#         move_files(dlc_files, dlc_folder, video_folder)
#     else:
#         dlc_files = find_files(video_folder, prefix=video_name)
#         move_files(dlc_files, video_folder, dlc_folder)


def initialize_folders(vid_info: dict) -> None:
    """Make DLC & RST folder if they don't exist"""
    dir_path = vid_info["dir"]
    if not is_valid_folder(dir_path):
        for folder in (cfg.DLC_FOLDER, cfg.RST_FOLDER, cfg.INF_FOLDER):
            mk_dir(os.path.join(dir_path, folder), force=False)


def is_valid_folder(folder_path):
    """Check if folder initialized by toolbox"""
    ls = os.listdir(folder_path)
    return (cfg.DLC_FOLDER in ls) and (cfg.RST_FOLDER in ls) and (cfg.INF_FOLDER in ls)


def read_result(csv_path: str):
    return pd.read_csv(csv_path, index_col=0, header=[0, 1])


def save_json(obj, save_path):
    with open(save_path, 'w') as f:
        json.dump(obj, f, index=4, sort_keys=True)
