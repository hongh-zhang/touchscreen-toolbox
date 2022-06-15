# configs
# ------------------------

# magic numbers
FPS = 25
TRAY_LENGTH = 26.8  # (cm)
P_CUTOFF = 0.1 # confidence threshold

# keypoints
MICE = ('snout',
        'lEar',
        'rEar',
        'spine1',
        'spine2',
        'tail1',
        'tail2',
        'tail3')
REFE = ('food_port',
        'll_corner',
        'lr_corner',
        'l_screen',
        'm_screen',
        'r_screen')

# files naming
DLC_CONFIG = "touchscreen_toolbox/DLC/config.yaml"  # path of deeplabcut config
DLC_FOLDER = "DLC"      # name of subfolder to put files from DLC (h5 & pickle)
RST_FOLDER = "results"  # name of subfolder to put analyzed results (csv)
INF_FOLDER = "info"
STATS_NAME = 'statistics.csv'
FORMATS = ['.mp4']



# vid info related
# ------
# pattern for decode name in utils/vid_info.py
PATTERN = r"^(\d+) - (\S+) - (\d{2}-\d{2}-\d{2}) (\d{2}-\d{2}) (\S+)"
ELEMENTS = ['mouse_id', 'chamber', 'exp_date', 'exp_time', 'suffix']

INFO_LS = ['file_name', 'mouse_id', 'exp_date', 'time', 'fps', 'prep']  # elements for export



# preprocess
# ------
B_THRESHOLD = 40  # threshold for increasing brightness



# auto generated variables
# ------
# for quick access of columns
HEADERS = [j for i in (MICE + REFE) for j in (i + '_x', i + '_y', i + '_cfd')]
XCOLS = [i for i in HEADERS if '_x' in i]
YCOLS = [i for i in HEADERS if '_y' in i]
CCOLS = [i for i in HEADERS if '_cfd' in i]

# template for statistics.csv
HEAD1 = INFO_LS + ['frame'] + [i[:-4] for i in CCOLS for j in '12345']
HEAD2 = ['-' for i in INFO_LS+['frame']] + [j for i in CCOLS for j in ('#of0', '%of0', 'cons', '1stQ', '10thQ')]