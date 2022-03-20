import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from decorator import decorator
from utils import *

@decorator
def plot_handler(func, *args, **kwargs):
    """
    Decorator for all positional plotting functions,
    1. set range for the axes
    2. draw circle on reference points (food port & screens)
    """

    # initialize graph
    fig, ax = plt.subplots()
    ax.set_xlim(0, TRAY_LENGTH)
    ax.set_ylim(-2, 22)
    
    # plot reference points
    #   find coordinates
    df = args[0]
    fd, _, _, ls, ms, rs = [(df[pt+'_x'].iloc[0], df[pt+'_y'].iloc[0]) for pt in REFE]
    
    #   draw circles
    for pt, clr in zip((fd, ls, ms, rs), ('greenyellow', 'tab:orange', 'orangered', 'tab:red')):
        ax.add_patch(plt.Circle(pt, radius=0.5, color=clr, fill=False))
    
    # pass to function
    func(*args, **kwargs)
    plt.show()

    
@plot_handler
def heatmap(df : pd.DataFrame, col : str, filt_idx=None):
    """Position heatmap using 2d KDE plot"""
    
    # locate data then apply filter (or not)
    if type(filt_idx)!=type(None):
        X = df[col+'_x'].loc[filt_idx]
        Y = df[col+'_y'].loc[filt_idx]
    else:
        X = df[col+'_x']
        Y = df[col+'_y']
    
    # plot
    sns.kdeplot(data=df, x=X, y=Y, fill=True, bw_adjust=5e-1, cut=3, levels=100, cmap='YlOrBr');
    
    
@plot_handler
def traj(df : pd.DataFrame, col : str, start : int, end : int):
    """Plot trajectory of [col] between [start] and [end] (in seconds)"""
    
    x = df[col+'_x'].iloc[range(start, end)]
    y = df[col+'_y'].iloc[range(start, end)]
    
    sns.lineplot(x=x, y=y, sort=False, lw=1, ci=None)