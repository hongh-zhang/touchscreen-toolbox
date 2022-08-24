import os
import glob
import pandas as pd
from natsort import natsorted
from collections import defaultdict
from joblib import Parallel, delayed

from . import utils
from . import video_info


def export_results(root: str, dest: str, n_jobs=8) -> None:
    """Export results from <root> to <dest>"""
    results, skipped = list_results(root)

    for animal in results.keys():
        # read all result for the animal into memory
        ls = Parallel(n_jobs=n_jobs)(delayed(_get_result)(f, animal) for f in natsorted(results[animal]))

        # concat, sort, then save
        save_path = os.path.join(dest, str(animal) + '.h5')
        (pd.concat(ls, axis=0)
         .sort_index(level=['ko', 'id', 'block', 'frame'], sort_remaining=False)
         .to_hdf(save_path, str(animal)))

    # record skipped videos
    utils.save_json(natsorted(skipped),
                    os.path.join(dest, 'skipped.json'))


# for multiprocessing
def _get_result(f: str, animal_id: int) -> pd.DataFrame:
    return multiindex_row(utils.read_result(f), int(animal_id))


def list_results(root: str) -> (defaultdict, list):
    """
    Returns
    ------
    results: dict
        dictionary of all processed results
        results[id] = list of result csv
        
    skipped: list
        list of skipped videos (no 'post_result' found)
    """
    all_videos = glob.glob(os.path.join(root, "**/*.mp4"), recursive=True)
    all_videos = list(filter(lambda x: 'DLC' not in x, all_videos))  # filter out videos created by preprocess

    skipped = []
    results = defaultdict(list)
    for f in all_videos:
        try:
            # read saved vid_info to locate postprocessing result
            info = video_info.get_vid_info(f, verbose=False)
            results[info['mouse_id']].append(os.path.join(info['dir'], info['post_result']))
        except KeyError:
            # post_result not found
            skipped.append(f)

    return results, skipped


def multiindex_row(df: pd.DataFrame, mouse_id: int) -> pd.DataFrame:
    # drop buffered frames & last trial
    df.drop(df[df[('task', 'state_')] == 0].index, axis=0, inplace=True)
    df.drop(df[df[('task', 'trial')] == df[('task', 'trial')].max()].index, axis=0, inplace=True)

    # block_ = 0 or 1 to distinguish 1st & 2nd block in the same session 
    block_ = (df[('task', 'block')] - df[('task', 'block')].min()).astype(int)

    # form index: ko, id, block, block_, trial, state_, frame
    multi_index = [df[('task', 'knockout')].astype(int),
                   [int(mouse_id) for _ in df.index],
                   df[('task', 'block')].astype(int),
                   block_,
                   df[('task', 'trial_')].astype(int),
                   df[('task', 'state_')].astype(int),
                   df.index]

    # assign new index to df
    df.drop([('task', 'knockout'), ('task', 'block'), ('task', 'trial'), ('task', 'trial_'), ('task', 'state_'),
             ('task', 'male')], axis=1, inplace=True)
    df.index = pd.MultiIndex.from_arrays(multi_index)
    df.index.names = ['ko', 'id', 'block', 'block_', 'trial', 'state_', 'frame']
    return df
