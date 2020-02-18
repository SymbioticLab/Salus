#!/usr/bin/env python3
#
# Copyright 2019 Peifeng Yu <peifeng@umich.edu>
#
# This file is part of Salus
# (see https://github.com/SymbioticLab/Salus).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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

old_vae = pits.at['vae', 'Salus']
pits.loc['vae', 'Salus'] = 1.2
old_superres = pits.at['superres', 'Salus']
pits.loc['superres', 'Salus'] = 1.2

pu.matplotlib_fixes()
with plt.style.context(['seaborn-paper', 'mypaper', 'color3']):
    ax = pits.plot.bar(legend=None)
    pu.axhlines(1.0, ax=ax, color='k', linestyle='--', linewidth=1)
    pu.bar_show_data(ax, pits.index.get_loc('superres'), 1.15, data_y=old_superres, fmt='{:.2f}')
    pu.bar_show_data(ax, pits.index.get_loc('vae'), 1.13, data_y=old_vae, fmt='{:.2f}')

    ax.set_ylim(0.9, 1.15)
    ax.set_xlabel('Workloads')
    ax.set_ylabel('Normalized\nPer Iteration\nTraining Time')
    # ax.legend()

    ax.tick_params(axis='x', labelsize=7)

    ax.figure.set_size_inches(3.25, 1.8, forward=True)
    ax.figure.tight_layout()
    ax.figure.savefig('/tmp/workspace/exp17.pdf', dpi=300, bbox_inches='tight', pad_inches=.015)
    plt.close()
