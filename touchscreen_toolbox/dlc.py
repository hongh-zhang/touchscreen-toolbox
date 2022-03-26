import os
import sys
import deeplabcut as dlc
from touchscreen_toolbox.utils import *


def dlc_analyze(path_cfg: str, video_path: str, verbosity: bool = False):
    """Call DLC to analyze video"""
    
    video_name = os.path.basename(video_path)
    folder_path = os.path.dirname(video_path)
    
    if verbosity:
        dlc.analyze_videos(DLC_CONFIG, video_path,
                           videotype='mp4', batchsize=32)
        dlc.filterpredictions(
            DLC_CONFIG,
            video_path,
            videotype='mp4',
            filtertype='median')

    # shut tf & dlc up
    else:
        # silence tensorflow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # <- NOT WORKING!

        # silence dlc
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        try:
            dlc.analyze_videos(path_cfg, video_path,
                               videotype='mp4', batchsize=32)
            dlc.filterpredictions(
                path_cfg,
                video_path,
                videotype='mp4',
                filtertype='median')

        # reset stdout before throwing error
        except Exception as err:
            sys.stdout = save_stdout
            raise err
        sys.stdout = save_stdout

    csv = [f for f in os.listdir(folder_path) if f.endswith('.csv') \
           and f.startswith(video_name[:-4])][0]
    
    return csv

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


# DEPRECATED
# ------------------------
# def analyze(path_config_file, folder_path, video_path):

#     print(f"Analyzing {video_path}...")

#     # call DLC to analyze video
#     dlc_analyze(path_config_file, video_path)

#     # remove h5 & pickle files from dlc
#     # csv will be renamed to the same as the video
#     cleanup(folder_path, video_path)
# ->
# moved to main.py


# def cleanup(folder_path, file_name, folder2move='DLC'):

#     files = os.listdir(folder_path)
#     file_name = os.path.basename(file_name)[:-4]
#     folder2move = os.path.join(folder_path, folder2move)

#     for f in files:
#         # raw prediction files
#         if (f.endswith('.h5') or f.endswith(
#                 '.pickle')) and f.startswith(file_name):

#             os.rename(os.path.join(folder_path, f),
#                       os.path.join(folder2move, f))

#         # coordinates csv
#         elif f.endswith('.csv') and f.startswith(file_name):
#             os.rename(os.path.join(folder_path, f),
#                       os.path.join(folder_path, file_name + '_raw.csv'))
# ->
# integrated into analyze_video
