import re
import os
import sys
import shutil
import numpy as np
import pandas as pd

from touchscreen_toolbox import utils
from touchscreen_toolbox import feature
import touchscreen_toolbox.preprocess as pre
import touchscreen_toolbox.postprocess as post
from touchscreen_toolbox.dlc import dlc_analyze


def analyze_folders(root: str, time_file: str = None, sort_key=lambda x: int(re.match('\d+', x)[0])):
    """
    Analyze the folder & all sub-folders
    
    Args
    ----
    root : str,
        path to the root folder to start analyzing
    
    sort_key : function, optional
        file name sorting order to be passed to sort() function
        
    """
    tee = utils.Tee('log.txt')  # log stdout
    for (folder_path, _, files) in list(os.walk(root)):
        analyze_folder(folder_path, time_file=time_file, sort_key=sort_key)


def analyze_folder(folder_path: str, time_file: str = None, sort_key=lambda x: x):
    """
    Analyze individual folder
        
    Args
    ----
    folder_path : str,
        path to the folder
    
    sort_key : function, optional
        file name sorting order to be passed to sort() function

    """
    print(f"Analyzing folder {folder_path}...")
    
    to_analyze = initialize(folder_path, sort_key=sort_key)
    for video in to_analyze:
        video_path = os.path.join(folder_path, video)
        analyze_video(video_path, time_file=time_file)


def initialize(folder_path: str, sort_key=lambda x: x):
    """
    Initialize folder/files if the folder is unprocessed,
    otherwise check progress and
    return a to-do list for higher level functions.
    
    Args
    ----
    folder_path : str,
        path to the folder
    
    sort_key : function, optional
        file name sorting order to be passed to sort() function
    
    Returns
    -------
    list
        A list of video (base)names that are not yet analyzed
    
    """
    
    # list all video in the directory
    videos = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
    if not videos:
        print("No videos found")
        return []
    
    # generate relevant folder path
    dlc_folder = os.path.join(folder_path, utils.DLC_FOLDER)
    rst_folder = os.path.join(folder_path, utils.RST_FOLDER)
    stats_path = os.path.join(rst_folder, utils.STATS_NAME)

    # no previous progress
    if not os.path.exists(stats_path):
        utils.mk_dir(dlc_folder)
        utils.mk_dir(rst_folder)
        utils.create_stats(stats_path)
        print(f"Initialized under {folder_path}")
        video_ls = sorted(videos, key=sort_key)

    # read progress and exclude processed videos
    else:
        analyzed = pd.read_csv(stats_path).iloc[2:, 0].values
        if not os.path.exists(dlc_folder):
            utils.mk_dir(dlc_folder)
        video_ls = sorted([v for v in videos if v not in analyzed], 
                      key=sort_key)
    
    print(f"Found videos {video_ls}")
    return video_ls

def analyze_video(video_path: str, time_file: str = None):
    """
    Analyze a video,
    
    Args
    ----
    video_path : str
        path to the video
    
    """
    
    print(f"Analyzing video {video_path}...")

    video_name = os.path.basename(video_path)
    folder_path = os.path.dirname(video_path)
    curr_files = os.listdir(folder_path)

    # preprocess
    # get new video name and a list of process applied (if any)
    print(f"Preprocessing...")
    new_video_path, proc_ls = pre.preprocess(video_path)

    # estimate
    print(f"Calling DeepLabCut...")
    csv = dlc_analyze(utils.DLC_CONFIG, new_video_path)
    csv = os.path.join(folder_path, csv)
    data = utils.read_dlc_csv(csv)
    
    # postprocess
    print(f"Postprocessing...")
    success, vid_info = utils.decode_name(video_name)

    # crop timeline
    if success and time_file:
        start, end = utils.get_time(time_file, vid_info['mouse_id'], vid_info['date'], hi_bound=len(data))
        data = data[start:end]
    
    # add second (of video) to data and reorder
    col_order = ['time']+data.columns.tolist()
    data['time'] = data.index / 25
    data = data[col_order]
    
    post.record(data, folder_path, video_name, vid_info['mouse_id'], vid_info['chamber'], vid_info['date'], vid_info['time'], proc_ls)
    data = post.standardize(data)
    
    # feature engineering
    data = feature.engineering(data)
    
    # save
    rst_folder = os.path.join(folder_path, utils.RST_FOLDER)
    save_path = os.path.join(rst_folder, video_name[:-4]+'.csv') if os.path.exists(rst_folder) \
                else os.path.join(folder_path, video_name[:-4]+'.csv')
    data.to_csv(save_path)
    print(f"Saved results to {save_path}\n\n")
    
    # relocate DLC files
    new_files = [f for f in os.listdir(folder_path) if f not in curr_files+[save_path]]
    cleanup(folder_path, new_files)


def cleanup(folder_path: str, files: list):
    """Move generated files into the DLC folder"""
    dlc_folder = os.path.join(folder_path, utils.DLC_FOLDER)
    if os.path.exists(dlc_folder):
        utils.move_files(files, folder_path, dlc_folder)


def generate_results(folder_path: str, time_file: str = None, sort_key=lambda x: int(re.match('\d+', x)[0])):
    """(re-)Generate results from an analyzed folder"""
    
    dlc_folder = os.path.join(folder_path, utils.DLC_FOLDER)
    rst_folder = os.path.join(folder_path, utils.RST_FOLDER)
    stats_path = os.path.join(rst_folder, utils.STATS_NAME)
    try:
        old_stats = pd.read_csv(stats_path, skiprows=[0]).set_index('video')
    except:
        old_stats = None
    csvs = utils.find_files(dlc_folder, '.csv')
    utils.mk_dir(rst_folder)
    utils.create_stats(stats_path)
    
    videos = sorted(utils.find_files(folder_path, '.mp4'), key=sort_key)
    for video in videos:
        # locate csv
        try:
            csv = [f for f in csvs if f.startswith(video[:-4])][0]
            data = utils.read_dlc_csv(os.path.join(dlc_folder, csv))
        except IndexError:
            raise Exception(f"File for {video} not found")

        # postprocess
        
        success, vid_info = utils.decode_name(video)
        
        if not success:
            print("Skipping")
            continue
        
        # crop timeline
        if time_file:
            start, end = utils.get_time(time_file, vid_info['mouse_id'], vid_info['date'], hi_bound=len(data))
            data = data[start:end]

        # add second (of video) to data and reorder
        col_order = ['time']+data.columns.tolist()
        data['time'] = data.index / 25
        data = data[col_order]
        
        # write in video info
        # ignore preprocess info if stats doesn't exists
        try:
            post.record(data, folder_path, video, vid_info['mouse_id'], vid_info['chamber'], vid_info['date'], vid_info['time'], old_stats.loc[video, 'pre'])
        except:
            post.record(data, folder_path, video, vid_info['mouse_id'], vid_info['chamber'], vid_info['date'], vid_info['time'], [])
            
        data = post.standardize(data)

        # feature engineering
        data = feature.engineering(data)
        
        # save
        data.to_csv(os.path.join(rst_folder, video[:-4]+'.csv'))
        

        
def merge_timestamp(data: pd.DataFrame, timestamp: pd.DataFrame):
    timestamp = timestamp.convert_dtypes()
    
    # align starting time
    timestamp['time'] += data.time.iloc[0]
    
    timestamp['frame'] = utils.sec2frame(timestamp['time'])
    increment_dup(timestamp, 'frame')
    
    merged = data.merge(timestamp.drop('time', axis=1), how='left', on='frame')
    merged['state_'] = merged['state'].fillna(method='ffill')#.fillna(method='bfill')
    
    return merged


def merge_timestamps(rst_folder, timestamp_file):
    
    with h5py.File(timestamp_file, 'r') as timestamps:
        
        for csv_file in utils.find_files(rst_folder, 'csv'):
            
            print(f"Merging {csv_file}...")
            
            csv_path = os.path.join(rst_folder, csv_file)
            
            success, vid_info = utils.decode_name(csv_file)
            
            if success:
                timestamp = pd.DataFrame(timestamps[vid_info['mouse_id']+'/'+vid_info['date']], 
                                         columns=['state','time']).convert_dtypes()
                data = pd.read_csv(csv_path)
                data = merge_timestamp(data, timestamp)
                merged.to_csv(csv_path, index=False)