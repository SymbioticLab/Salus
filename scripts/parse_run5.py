#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:22:49 2018

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

    
def parse_output_float(outputfile, pattern, group=1):
    """Parse outputfile using pattern"""
    if not outputfile.exists():
        msg = f'File not found: {outputfile}'
        raise ValueError(msg)

    ptn = re.compile(pattern)
    with outputfile.open() as f:
        for line in f:
            line = line.rstrip()
            m = ptn.match(line)
            if m:
                try:
                    return float(m.group(group))
                except (ValueError, IndexError):
                    continue
    raise ValueError(f"Pattern `{pattern}' not found in output file {outputfile}")
    
def parse_jct(outputfile):
    return parse_output_float(outputfile, r'^JCT: ([0-9.]+) .*')

def load_mem(path, return_logs=False, parallel_workers=0):
    try:
        logs = pl.load_file(path, parallel_workers)
    except TypeError:
        logs = path

    df = pd.DataFrame(l.__dict__ for l in logs)
    df = df[df.type.isin(['mem_alloc', 'mem_dealloc'])]
    df = df.drop(['entry_type', 'level', 'loc', 'thread'], axis=1)
    df['ticket'] = df.block.apply(lambda b: b['ticket'])
    df.drop('block', axis=1)

    mapping = {
        'mem_alloc': 1,
        'mem_dealloc': -1
    }

    df.type.replace(mapping, inplace=True)
    df['act'] = df.type * df['size']
    df['act'] = df.act.cumsum() / 1024 / 1024
    # make sure index is consequtive
    df = df.reset_index(drop=True)
    
    if return_logs:
        return df, logs
    else:
        return df

def load_run5(path):
    path = Path(path)
    
    data = pd.DataFrame(columns=['Average JCT', 'Makespan'])
    
    # load salus
    salus_jcts = []
    for f in (path/'salus').iterdir():
        salus_jcts.append(parse_jct(f))
    data.loc['Salus'] = [np.mean(salus_jcts), np.max(salus_jcts)]
    
    # laod fifo
    fifo_jcts = []
    for f in (path/'tf').iterdir():
        fifo_jcts.append(parse_jct(f))
    data.loc['FIFO'] = [np.mean(np.cumsum(fifo_jcts)), np.sum(fifo_jcts)]
    
    # load mem usage
    memdir = path/'salus-alloc'
    if memdir.exists():
        dfmem = load_mem(memdir/'alloc.output')
    else:
        dfmem = None
    return data, dfmem


def plot_cs(cs, **kwargs):
    ax = cs.plot(**kwargs)
    return ax


def plot_df(df, marker=False, offset=None, **kwargs):
    cs = df.set_index('timestamp')
    if marker:
        if offset is None:
            offset = cs.index[0]
        cs.index = (cs.index - offset) / pd.Timedelta(microseconds=1)
    elif offset is not None:
        cs.index = (cs.index - offset) / pd.Timedelta(microseconds=1)

    ax = plot_cs(cs.act, **kwargs)
    if marker:
        css = cs.reset_index()
        ax = css.plot(kind='scatter', x='timestamp', y='act', c='type', cmap=plt.cm.get_cmap('bwr'), ax=ax)
    return ax


def plot_df_persess(df, marker=False, offset=None, fill=False, sessaxs=None, **kwargs):
    cs = df.set_index('timestamp')
    if marker:
        if offset is None:
            offset = cs.index[0]
        cs.index = (cs.index - offset) / pd.Timedelta(microseconds=1)
    elif offset is not None:
        cs.index = (cs.index - offset) / pd.Timedelta(microseconds=1)

    if sessaxs is None:
        _, axs = plt.subplots(nrows=len(cs.sess.unique()), sharex=True,
                              squeeze=False)
        axs = axs.flatten()
        sessaxs = {}
    for idx, (sess, dfsess) in enumerate(cs.groupby('sess')):
        if sess not in sessaxs:
            ax = axs[idx]
            sessaxs[sess] = ax
        else:
            ax = sessaxs[sess]
            
        act = dfsess.type * dfsess['size']
        act = act.cumsum() / 1024 / 1024
        
        if fill:
            # fill nans
            act = pd.DataFrame(act).reset_index()
            nans = np.where(np.empty_like(act.values), np.nan, np.nan)
            data = np.hstack([nans, act.values]).reshape(-1, act.shape[1])
            act = pd.DataFrame(data, columns=act.columns)
            act[0] = act[0].ffill()
            act['timestamp'] = act['timestamp'].bfill()
            act = act.set_index('timestamp').dropna()
        
        plot_cs(act, ax=ax, **kwargs)
        if marker:
            css = dfsess.reset_index()
            ax = css.plot(kind='scatter',
                          x=dfsess.timestamp, y=act, c=dfsess.type,
                          cmap=plt.cm.get_cmap('bwr'), ax=ax)
    return sessaxs

#%% 5 alexnet_25
path = 'logs/osdi18/cc/run5-alexnet'

data, _ = load_run5(path)

print("5 alexnet_25")
print(data)

#%% 5 resnet50_50
path = 'logs/osdi18/cc/run5-resnet50_50'

data, dfmem = load_run5(path)

print("5 resnet50_50")
print(data)

plt.style.use(['seaborn-paper', 'mypaper'])
fig = plt.figure()
ax = plot_df(dfmem)

#ax.set_xlabel('')
ax.set_ylabel('Time (s)')

ax.tick_params(axis='x', labelsize=7, rotation=0)

ax.figure.set_size_inches(3.45, 1.75, forward=True)
ax.figure.tight_layout()
ax.figure.savefig('/tmp/workspace/exp62.pdf', dpi=300)
#plt.close()