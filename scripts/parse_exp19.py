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
Created on Mon Apr 30 01:11:53 2018

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
import parse_log as pl


def select_steps(df):
    # count unique numbers
    counts = df.groupby('step').agg({c: 'nunique' for c in ['kernel', 'op']}).reset_index()
    ss = counts.query('step > 10 & kernel > 10 & op > 200')
    
    # so the step list is
    if len(ss) > 1:
        # drop first iteration
        return ss.step.astype(int).tolist()[1:]
    else:
        # nothing we can find programmatically, let the user decide
        for _, s, ker, op in counts.itertuples():
            print('Step {} has {} tasks, with {} kernels, select?'.format(s, op, ker))
        return [int(input('nothing we can find programmatically, let the user decide'))]

def only_step(steps, idx):
    ss = steps.step.sort_values().unique().tolist()
    if idx >= len(ss):
        idx = len(ss) - 1
    return steps[steps.step == ss[idx]]


def load_failure(path):    
    logs = pl.load_file(str(path))
    df = pd.DataFrame(l.__dict__ for l in logs)
    df = df[df.type == 'optracing_evt']
    df = df.drop(['entry_type','level','loc', 'thread', 'type'], axis=1)
    # make sure step is int
    df['step'] = df.step.astype(int)
    
    ss = select_steps(df)
    step25 = df[df.step.isin(ss)]
    
    # discard unneeded event
    step25 = step25[step25.evt == 'done']
    
    # add a name column
    def name(row):
        return '{}[{}]'.format(row['op'], row['kernel'])
    step25['name'] = step25.apply(name, axis=1).values
    
    return step25

def load_exp19(directory):
    directory = Path(directory)
    
    res = {}
    for d in (directory/'salus').iterdir():
        res[d.name] = load_failure(d/'perf.output')
    return res
        
def cdf(X, ax=None, **kws):
    if ax is None:
        _, ax = plt.subplots()
    n = np.arange(1,len(X)+1) / np.float(len(X))
    Xs = np.sort(X)
    ax.step(Xs, n, **kws)
    ax.set_ylim(0, 1)
    return ax

#%%
path = 'logs/osdi18/cc/exp19'

data = load_exp19(path)

plt.style.use(['seaborn-paper', 'mypaper'])

ax = None
for n, failures in data.items():
    ax = cdf(failures, ax=ax, label='{} x resnet50_50'.format(n))
fig = ax.figure
ax.legend()

fig.set_size_inches(3.45, 2.75, forward=True)
fig.tight_layout()
fig.savefig('/tmp/workspace/exp19.pdf', dpi=300)
#plt.close()