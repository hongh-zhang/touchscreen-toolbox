# Scripts to integrate DeepLabCut

import os
import sys
import pandas as pd
import deeplabcut as dlc
import touchscreen_toolbox.utils as utils
import touchscreen_toolbox.config as cfg



def analyze(vid_info: dict, *args, **kwargs) -> None:
    
    dlc_analyze(vid_info, *args, **kwargs)
    cleanup(vid_info)



def dlc_analyze(vid_info: dict, verbose: bool=False) -> None:
    """Call DLC to analyze video"""
    
    if 'files' in vid_info and 'result' in vid_info:
        print("Vid info contain processed files, skipping...")
        return None
    
    curr_files = utils.find_files(vid_info['dir'])
    
    if verbose:
        dlc.analyze_videos(cfg.DLC_CONFIG, vid_info['target_path'], videotype='.mp4', batchsize=32)
        dlc.analyze_videos_converth5_to_csv(vid_info['dir'], videotype='.mp4')

    # shut tf & dlc up
    else:
        # silence tensorflow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # <- NOT WORKING!

        # silence dlc
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        try:
            dlc.analyze_videos(cfg.DLC_CONFIG, vid_info['target_path'], videotype='.mp4', batchsize=32)
            dlc.analyze_videos_converth5_to_csv(vid_info['dir'], videotype='.mp4')

        # reset stdout before throwing error
        except Exception as err:
            sys.stdout = save_stdout
            raise err
        sys.stdout = save_stdout
    
    new_files = [f for f in utils.find_files(vid_info['dir']) if f not in curr_files]
    csv = [f for f in new_files if f.endswith('.csv')][0]
    vid_info['files'] = new_files
    vid_info['result'] = csv
    

def cleanup(vid_info: dict) -> None:
    """Relocate pose estimation files into the DLC folder"""
    
    # relocate
    curr_dir = vid_info['dir']
    targ_dir = os.path.join(vid_info['dir'], cfg.DLC_FOLDER)
    
    if vid_info['path'] != vid_info['target_path']:
        vid_info['files'].append(os.path.basename(vid_info['target_path']))
    utils.move_files(vid_info['files'], curr_dir, targ_dir)
    
    # rewrite file path
    vid_info['files'] = [os.path.join(cfg.DLC_FOLDER, x) for x in vid_info['files']]
    vid_info['result'] = os.path.join(cfg.DLC_FOLDER, vid_info['result'])


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
