import sys
import deeplabcut as dlc
from contextlib import contextmanager

PATH_CFG = 'touchscreen_toolbox/DLC/config.yaml'

def analyze(video : str):
    with nostdout():  # silence dlc
        dlc.analyze_video(PATH_CFG, video, videotype='mp4', batchsize=32)
        dlc.filterpredictions(PATH_CFG, video, videotype='mp4', filtertype='median')


# functions to suppress output
# ---------------------------------------------
# copied from Alex Martelli
# https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto

class DummyFile(object):
    def write(self, x): pass

@contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
# ---------------------------------------------