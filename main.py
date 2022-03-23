import os
import sys
import shutil
import numpy as np
import pandas as pd
import touchscreen_toolbox as tb


path_config_file = "touchscreen_toolbox/DLC/config.yaml"
the_folder = 'tests'

for (folder_path, _, files) in list(os.walk(the_folder)):

    if files:
        
        # jump to next folder if no valid videos
        videos = [f for f in files if f.endswith('.mp4') and 
                  (not tb.utils.is_preprocess(f))]
        if not videos:
            continue
        
        print(f"Processing folder {folder_path}...\n")

        # create folder for dlc products
        dlc_folder = os.path.join(folder_path, 'DLC')
        tb.utils.mk_dir(dlc_folder)

        # analyze each video
        for video in videos:
            video = os.path.join(folder_path, video)

            # preprocess if needed
            new_video = tb.preprocess(video)
            
            # analyze
            tb.analyze(path_config_file, folder_path, new_video)

            print("-----------------------------\n")

        # postprocess
        tb.postprocess(folder_path)