import os
import sys
import numpy as np
import pandas as pd
from itertools import groupby
from touchscreen_toolbox.utils import *


# standardize related
# ----------------------------------------------
def replace_w_median(df : pd.DataFrame, col : str):
    df[col+'_x'] = df[col+'_x'].median()
    df[col+'_y'] = df[col+'_y'].median()
    
    
def set_origin(df : pd.DataFrame, col : str):
    """Set the [col] as origin for all coordiantes"""
    x_adjustment = df[col+'_x'][0]
    y_adjustment = df[col+'_y'][0]

    for col in XCOLS:
        df[col] -= x_adjustment
    for col in YCOLS:
        df[col] -= y_adjustment
        
        
class L_transformer():
    """A linear transformer in R2"""
    
    def __init__(self, cos=1.0, sin=0.0, scale=1.0):
        self.scale = scale
        self.rotation = np.array([[cos, -sin], [sin, cos]])
    
    def transform(self, X):
        return np.dot(self.rotation, X.T).T * self.scale
                
                
def standardize(data : pd.DataFrame):
    """Standardize a csv output from DeepLabCut"""

    # flip
    data[YCOLS] *= -1
    
    # remove fluctuation in reference points
    for col in REFE:
        replace_w_median(data, col)
    
    # make lower left corner the origin
    set_origin(data, 'll_corner')
    
    # prepare linear transformation
    adj = data['lr_corner_x'][0]
    opp = - data['lr_corner_y'][0]
    hyp = dist1((adj, opp))
    transformer = L_transformer(cos=(adj/hyp), sin=(opp/hyp), 
                                scale=(TRAY_LENGTH/hyp))

    # apply transformation (rotate + scale)
    for xcol, ycol in zip(XCOLS, YCOLS):
        data[[xcol, ycol]] = transformer.transform(data[[xcol, ycol]].values)

# ----------------------------------------------


def fillna(df : pd.DataFrame):
    """Handle NaNs with an average step method"""
    
    for col_name in df:
        col = df[col_name]
        
        # index of missing values
        idx = col[col.isnull()].index
        
        if idx.any():
            
            # group consecutive index
            temp = [idx[0]]
            groups = []
            for i in idx[1:]:
                # add to temp list if the element is consecutive
                if (i - temp[-1] == 1):  
                    temp.append(i)
                # if the element is not consevutive,
                # complete the current group and reset temp list
                else:
                    groups.append(temp)
                    temp = [i]
            groups.append(temp)


            # fillNaN progressively, using average step of next-prev value
            out_of_bound = False
            for group in groups:
                try:
                    pre = col[(group[0] - 1)]   # previous non-empty value
                    nex = col[(group[-1] + 1)]  # next non-empty value
                    steps = len(group)
                    step = (nex - pre) / steps  # step value

                    for i in group:
                        pre += step
                        col[i] = pre
                
                # happens when the 1st/last element is NaN so there's no preceding/following value to fill
                except KeyError:
                    out_of_bound = True
                    continue
            
            # fill remaining value if error occurred
            if out_of_bound:
                col.fillna(method='bfill', inplace=True)
                col.fillna(method='ffill', inplace=True)


def statistics(data : pd.DataFrame):
    value = []
    for col_name in CCOLS:
        col = data[col_name].fillna(0)
        zeros = (col==0)
        nums = zeros.sum()
        consecutive = max([len(list(g)) for k, g in groupby(zeros) if k]) if nums > 0 else 0
        first = str(round(col.quantile(q=0.01), 2))
        tenth = str(round(col.quantile(q=0.10), 2))
        value += [nums, consecutive, first, tenth]
    return value

def postprocess(folder_path, dlcfolder='DLC'):
    
    files = [f for f in os.listdir(folder_path) if f.endswith('_raw.csv')]
    if not files:
        raise ValueError("No valid csv files found")
    
    dlcfolder = os.path.join(folder_path, dlcfolder)
    headers1 = [i[:-4] for i in CCOLS for j in '1234']
    headers2 = [j for i in CCOLS for j in ('#of0', 'cons', '1stQ','10thQ')]
    values = []

    for name in files:
        file = path2file(name)[:-8]
        file_path = os.path.join(folder_path, file+'_raw.csv')
        data = pd.read_csv(file_path, skiprows=[0,1,2,3], names=(['frame'] + HEADERS))\
            .set_index('frame')

        values.append(statistics(data))

        data.drop(CCOLS, axis=1, inplace=True)
        standardize(data)
        fillna(data)

        data.to_csv(os.path.join(folder_path, file+'.csv'))

        # move raw prediction into dlc folder
        os.rename(os.path.join(folder_path, file+'_raw.csv'), 
             os.path.join(dlcfolder, file+'_raw.csv'))

    stats = pd.DataFrame(np.vstack((headers1, headers2, values)))
    stats.to_csv(os.path.join(folder_path, "statistics.csv"))
    print("Statistics saved")



### DEPRECATED FUNCTIONS

# def rotate(df, col):
#     """linear rotation in R2"""
#     for xcol, ycol in zip(x_cols, y_cols):
#         X, Y = (df[xcol], df[ycol])
#         df[xcol] = cos*X - sin*Y
#         df[ycol] = sin*X + cos*Y
