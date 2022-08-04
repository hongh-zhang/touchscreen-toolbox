import os
import glob
from natsort import natsorted
from collections import defaultdict

from . import utils
from . import video_info


def save_result(dest, mouse, count, result_path, output_format='.csv'):
    output_path = os.path.join(dest, str(mouse), str(count)+output_format)
    result = utils.read_result(result_path)
    if output_format=='.csv':
        result.to_csv(output_path)
    elif output_format=='.h5':
        result.to_hdf(output_path, str(count))


def export(root, dest):
    """
    Organize and export all results from <root> to <dest> folder
    """
    all_videos = list(filter(lambda x: 'DLC' not in x, 
                             glob.glob(os.path.join(root, "**/*.mp4"), recursive=True)))
    
    # list all mouse-exp_date pairs
    # df[mouse id][experiment date] = path to result
    df = defaultdict(dict)
    skipped = []
    for f in all_videos:
        try:
            info = video_info.get_vid_info(f, verbose=False)
            df[info['mouse_id']][info['exp_date']] = os.path.join(info['dir'], info['post_result'])
        except KeyError:
            skipped.append(f)
            
    # reformat dictionary to count experiment number
    df2 = []
    for mouse in natsorted(df.keys()):
        count = 0
        for exp in natsorted(df[mouse].keys()):
            count += 1
            result_path = df[mouse][exp]
            df2.append((mouse, count, result_path))
    
    # save
    for mouse in set(list(zip(*df2))[0]):
        folder_path = os.path.join(dest, mouse)
        os.mkdir(folder_path)
    Parallel(n_jobs=8)(delayed(save_result)(dest, *i) for i in df2)
    
    with open(os.path.join(dest, 'skipped.json'), 'w') as f:
        json.dump(natsorted(skipped), f, indent=4)
    
    df3 = defaultdict(dict)
    for i in df2:
        df3[i[0]][i[1]] = i[2]
    with open(os.path.join(dest, 'index.json'), 'w') as f:
        json.dump(df3, f, indent=4)