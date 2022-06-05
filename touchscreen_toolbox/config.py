# configs
# ------------------------

# magic numbers
FPS = 25
TRAY_LENGTH = 26.8  # (cm)

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

# files
DLC_CONFIG = "touchscreen_toolbox/extract/DLC/config.yaml"  # path of deeplabcut config
DLC_FOLDER = "DLC"      # name of subfolder to put files from DLC (h5 & pickle)
RST_FOLDER = "results"  # name of subfolder to put analyzed results (csv)
STATS_NAME = 'statistics.csv'




# auto generated variables
# ------
HEADERS = [j for i in (MICE + REFE) for j in (i + '_x', i + '_y', i + '_cfd')]