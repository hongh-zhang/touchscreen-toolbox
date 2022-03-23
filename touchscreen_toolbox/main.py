import os
import sys
import shutil
import numpy as np
import pandas as pd

from touchscreen_toolbox.utils import *
from touchscreen_toolbox.dlc import dlc_analyze
from touchscreen_toolbox.preprocess import preprocess
import touchscreen_toolbox.postprocess as postprocess



def initialize(folder_path):
    """
    Check progress of a folder,
    initialize folder/files if the folder is unprocessed,
    otherwise return a to-do list for higher level functions.
    """
    
    videos = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
    if not videos:
        return []
    
    dlc_folder = os.path.join(folder_path, DLC_FOLDER)
    rst_folder = os.path.join(folder_path, RST_FOLDER)
    stats_path = os.path.join(rst_folder, STATS_NAME)
    
    # no previous progress
    if not os.path.exists(stats_path):
        mk_dir(dlc_folder)
        mk_dir(rst_folder)
        STATS_TEMPL.to_csv(stats_path, index=False, header=False)
        print(f"Initialized under {folder_path}")
        return videos
    
    # read progress and exclude processed videos
    else:
        analyzed = pd.read_csv(stats_path).iloc[2:, 0].values
        if not os.path.exists(dlc_folder):
            mk_dir(dlc_folder)
        return [v for v in videos if v not in analyzed]

    
def analyze_video(video_path):
    
    print(f"Analyzing {video_path}")
    
    video_name  = os.path.basename(video_path)
    folder_path = os.path.dirname(video_path)
    curr_files  = os.listdir(folder_path)
    
    # preprocess
    # get new video name and a list of process applied (if any)
    new_video, proc_ls = preprocess(video_path)

    # analyze
    dlc_analyze(DLC_CONFIG, new_video)
    new_files = [f for f in os.listdir(folder_path) if f not in curr_files]
    csv = os.path.join(folder_path, [f for f in new_files if f.endswith(".csv")][0])
    
    # postprocess
    # record into stats
    data  = read_dlc_csv(csv)
    
    stats_path = os.path.join(folder_path, RST_FOLDER, STATS_NAME)
    if os.path.exists(stats_path):
        stats = pd.read_csv(stats_path)
        stats.loc[len(data)] = ([video_name] + [str(proc_ls)] + postprocess.statistics(data))
        stats.to_csv(stats_path, index=False)
    
    # standardize
    data.drop(CCOLS, axis=1, inplace=True)
    postprocess.standardize(data)
    postprocess.fillna(data)
    data.to_csv(os.path.join(folder_path, RST_FOLDER, video_name[:-4] + '.csv'))
    
    # relcate files if folder exists
    dlc_folder = os.path.join(folder_path, DLC_FOLDER)
    if os.path.exists(dlc_folder):
        for f in new_files:
            file_path = os.path.join(folder_path, f)
            new_path = os.path.join(dlc_folder, f)
            os.rename(file_path, new_path)

            
def analyze_folder(folder_path):
    """Analyze individual folder"""
    
    to_analyze = initialize(folder_path)
    
    for video in to_analyze:
        video_path = os.path.join(folder_path, video)
        analyze_video(video_path)

        
def analyze_folders(root, ):
    """Analyze the folder & all sub-folders"""
    
    for (folder_path, _, files) in list(os.walk(root)):
        analyze_folder(folder_path)