import os
import re
import sys
import shutil
import numpy as np
import pandas as pd
from time import (localtime, strftime)
import touchscreen_toolbox.config as cfg



# postprocess related
# ------------------------

# for quick access of columns
XCOLS = [i for i in cfg.HEADERS if '_x' in i]
YCOLS = [i for i in cfg.HEADERS if '_y' in i]
CCOLS = [i for i in cfg.HEADERS if '_cfd' in i]

# template for statistics.csv
HEAD1 = ['video', 'id', 'chamber', 'date', 'time', 'pre', 'frame'] + [i[:-4] for i in CCOLS for j in '12345']
HEAD2 = ['-' for i in range(7)] + [j for i in CCOLS for j in ('#of0', '%of0', 'cons', '1stQ', '10thQ')]
STATS_TEMPL = pd.DataFrame(np.vstack((HEAD1, HEAD2)))


def create_stats(path: str):
    """Create empty statistics.csv at the given <path>"""
    STATS_TEMPL.to_csv(path, index=False)


# hardcoded postprocessing
# ------------------------
PATTERN = r'^(\d+) - (\S+) - (\d{2}-\d{2}-\d{2}) (\d{2}-\d{2}) (\S+)'
ELEMENTS = ['mouse_id', 'chamber', 'date', 'time', 'suffix']
DEFAULT = ['-' for i in range(5)]
def decode_name(name, pattern=PATTERN):
    try:
        matched = [''.join(i.split('-')) for i in re.match(pattern, name).groups()]
        return True, {i:j for i,j in zip(ELEMENTS, matched)}
    except AttributeError:
        print("Pattern unmatched")
        return False, {}
    
    
def get_time(time_file, mouse_id, date, pre_buffer=1, post_buffer=2, hi_bound=999999):
    col = pd.read_csv(time_file).set_index(['id', 'date']).loc[(int(mouse_id), int(date))]
    start, end = map(sec2frame, (col['vid_start'], col['vid_end']))
    start = max(0,  start - cfg.FPS * pre_buffer)
    end   = min(hi_bound, end + cfg.FPS * post_buffer)
    return start, end