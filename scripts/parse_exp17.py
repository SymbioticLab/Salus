#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 04:54:36 2018

@author: peifeng
"""

from __future__ import print_function, absolute_import, division

import re
from datetime import datetime
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotutils as pu

def load_exp17(path):
    path = Path(path)
    dftf = pd.read_csv(path / 'jct-baseline.csv')
    dfsalus = pd.read_csv(path / 'jct-salus.csv')
    dftfdist = pd.read_csv(path / 'jct-tfdist.csv')
    return dftf.set_index('Network'), dfsalus.set_index('Network'), dftfdist.set_index('Network')

#%%
path = 'logs/nsdi19/'

tf, salus, tfdist = load_exp17(path)

column = '20iter-avg'
pits = pd.DataFrame({
    'Salus': salus[column] / tfdist[column],
    # 'TFDist': tfdist[column] / tf[column],
    # 'Salus/TFDist': salus[column] / tfdist[column],
})

pits = pits[~pits.index.str.startswith('mnist')]
#%%

# only training
pits = pits[~pits.index.str.contains('eval')]

# only one batch size
# only draw one batch size: the largest one
pits = pits.reset_index()
pits['Model'], pits['BatchSize'] = pits.Network.str.split('_').str
pits.BatchSize.replace({'small': 1, 'medium': 5, 'large': 10}, inplace=True)
pits['BatchSize'] = pd.to_numeric(pits.BatchSize)
pits = pits.reset_index().loc[pits.reset_index().groupby(['Model'])['BatchSize'].idxmax()]
pits = pits.drop(['index', 'BatchSize', 'Network'], axis=1)
pits = pits.rename(columns={'Model': 'Network'}).set_index('Network')

with plt.style.context(['seaborn-paper', 'mypaper', 'color3']):
    ax = pits.plot.bar(legend=None)
    pu.axhlines(1.0, ax=ax, color='k', linestyle='--', linewidth=1)
    pu.bar_show_data(ax, pits.index.get_loc('superres'), pits.at['superres', 'Salus'])
    pu.bar_show_data(ax, pits.index.get_loc('vae'), pits.at['vae', 'Salus'])

    ax.set_ylim(0.9, 1.9)
    ax.set_xlabel('Workloads')
    ax.set_ylabel('Normalized Per Iteration\nTraining Time')
    # ax.legend()

    ax.tick_params(axis='x', labelsize=7)

    ax.figure.set_size_inches(3.25, 2.35, forward=True)
    ax.figure.tight_layout()
    ax.figure.savefig('/tmp/workspace/exp17.pdf', dpi=300, bbox_inches='tight', pad_inches = .015)
    plt.close()
