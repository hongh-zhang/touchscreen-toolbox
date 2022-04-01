import re
import pandas as pd
from touchscreen_toolbox import utils

PATTERN = '^(\d+) - (\S+) - (\d{2}-\d{2}-\d{2}) (\d{2}-\d{2}) (\S+)'
DEFAULT = ['-' for i in range(5)]
def decode_name(name, pattern=PATTERN):
    try:
        return [''.join(i.split('-')) for i in re.match(pattern, name).groups()]
    except AttributeError:
        print("Pattern unmatched")
        return DEFAULT
    
    
def get_time(time_file, mouse_id, date):
    col = pd.read_csv(time_file).set_index(['id', 'date']).loc[(int(mouse_id), int(date))]
    start, end = map(utils.sec2frame, (col['vid_start'], col['vid_end']))
    return start, end
    