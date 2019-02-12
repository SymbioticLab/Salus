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
Created on Sun Apr 29 23:01:43 2018

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

tf_events = ['task_ready', 'task_start', 'task_done']
def load_tf(path):
    logs = pl.load_file(path)
    df = pd.DataFrame(l.__dict__ for l in logs)
    df = df[df.type.isin(tf_events)].drop(['level','loc', 'entry_type'], axis=1)
    # make sure step is int
    df['step'] = df.step.astype(int)

    # discard some internal or async op: _SOURCE, _Recv, _Send
    ignored = ['_Recv', '_Send']
    df = df[~df.kernel.isin(ignored)]
    df = df[df.op != '_SOURCE']
    
    steptf = df.pivot_table(values='timestamp', index=['step', 'op', 'kernel'],
                            columns='type', aggfunc='first').reset_index()
    
    # add a name column
    def name(row):
        return '{}[{}]'.format(row['op'], row['kernel'])
    steptf['name'] = steptf.apply(name, axis=1).values
    
    # reorder
    steptf = steptf[['step', 'name', 'op', 'kernel'] + tf_events]
    
    return steptf.sort_values(by=tf_events).reset_index(drop=True)

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

def load_exp3(path):
    wls = {}
    for d in Path(path).iterdir():
        logfile = next(d.iterdir())
        steptf = load_tf(str(logfile))
        
        # filter steps
        steps = select_steps(steptf)
        steptf = steptf[steptf.step.isin(steps)]
        steptf = only_step(steptf, 0)
        
        # calc length
        wls[d.name] = (steptf.task_done - steptf.task_start) / pd.Timedelta(microseconds=1)
    
    df = pd.DataFrame(wls)
    return df

#%%
    
path = 'logs/osdi18/cc/exp3'

df= load_exp3(path)
    
plt.style.use(['seaborn-paper', 'mypaper'])

fig, axs = plt.subplots(nrows=2, sharex=True)

ax = df.plot.box(ax=axs[1])
ax.set_ylim(0, 160)
ax.set_xlabel('Workloads')
ax.set_ylabel('Execution Time (us)')

ax.tick_params(axis='x', labelsize=7, labelrotation=90)

ax = df.count().plot(kind='bar', ax=axs[0], color='#ed7d31')
ax.set_ylabel('# Unique Tasks')

fig.set_size_inches(3.45, 3.45, forward=True)
fig.tight_layout()
fig.savefig('/tmp/workspace/exp3.pdf', dpi=300)
#plt.close()