import os
import sys
import shutil
import deeplabcut as dlc
from contextlib import contextmanager

def analyze(path_cfg : str, video : str, verbosity : bool = False):
    print(f"Analyzing {video}")

    if verbosity:
        dlc.analyze_videos(path_cfg, video, videotype='mp4', batchsize=32)
        dlc.filterpredictions(path_cfg, video, videotype='mp4', filtertype='median')
    else:
        # silence tensorflow            <- NOT WORKING!
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

        # silence dlc
        with nostdout():
            dlc.analyze_videos(path_cfg, video, videotype='mp4', batchsize=32)
            dlc.filterpredictions(path_cfg, video, videotype='mp4', filtertype='median')
    

def cleanup(folder_path, folder2move, name):

    name = name[:-4].split('/')[-1]
    _, _, files = list(os.walk(folder_path))[0]

    for f in files:
        # raw prediction files
        if (f.endswith('.h5') or f.endswith('.pickle')) and f.startswith(name):
            shutil.move(os.path.join(folder_path, f), 
                  os.path.join(folder2move, f))

        # coordinates csv
        elif f.endswith('.csv') and f.startswith(name):
            os.rename(os.path.join(folder_path, f), os.path.join(folder_path, name+'_raw.csv'))
            print('renamed csv')
    print("Reorganized files")
    return name


# functions to suppress output
# ---------------------------------------------
# copied from Alex Martelli
# https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto

class DummyFile(object):
    def write(self, *args, **kwargs): pass
    def flush(self, *args, **kwargs): pass

@contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
# ---------------------------------------------
