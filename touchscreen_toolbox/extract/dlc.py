# Scripts to integrate DeepLabCut

import os
import sys
import pandas as pd
import deeplabcut as dlc
import touchscreen_toolbox.utils as utils
import touchscreen_toolbox.config as cfg



def dlc_analyze(video_path: str, verbosity: bool = False):
    """Call DLC to analyze video"""
    
    video_name = os.path.basename(video_path)
    folder_path = os.path.dirname(video_path)
    
    if verbosity:
        dlc.analyze_videos(cfg.DLC_CONFIG, video_path,
                           videotype='mp4', batchsize=32)

    # shut tf & dlc up
    else:
        # silence tensorflow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # <- NOT WORKING!

        # silence dlc
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        try:
            dlc.analyze_videos(cfg.DLC_CONFIG, video_path,
                               videotype='mp4', batchsize=32)

        # reset stdout before throwing error
        except Exception as err:
            sys.stdout = save_stdout
            raise err
        sys.stdout = save_stdout

    csv = [f for f in utils.find_files(folder_path, '.csv') if f.startswith(video_name[:-4])][0]
    
    return csv


def label_video(video_path):
    
    video_name  = os.path.basename(video_path)
    folder_path = os.path.dirname(video_path)
    dlc_folder  = os.path.join(folder_path, DLC_FOLDER)
    
    # find relevant files & move to video directory
    files = [f for f in os.listdir(dlc_folder) if f.startswith(video_name[:-4])]
    move_files(files, dlc_folder, folder_path)
    
    # label
    # dlc somehow doesnt recognize relative path
    dlc.create_labeled_video(DLC_CONFIG, os.path.abspath(video_path), videotype='mp4', save_frames = False, filtered=True)
    
    # move back files
    move_files(files, folder_path, dlc_folder)


def read_dlc_csv(path: str):
    return pd.read_csv(path, skiprows=[0, 1, 2, 3], names=(['frame'] + cfg.HEADERS)).set_index('frame')



# functions to suppress output
# ---------------------------------------------
# copied from Alex Martelli
# https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto

class DummyFile(object):
    def write(self, *args, **kwargs): pass
    def flush(self, *args, **kwargs): pass


# @contextmanager
# def nostdout():
#     save_stdout = sys.stdout
#     sys.stdout = DummyFile()
#     yield
#     sys.stdout = save_stdout
# ---------------------------------------------
