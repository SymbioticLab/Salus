#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:59:19 2018

Plot per session memory alloc and computation together

@author: peifeng
"""

import parse_log as pl
import pandas as pd
from datetime import datetime
import json
import multiprocessing as mp
import subprocess as sp
from tqdm import tqdm
from functools import partial
import os
import numpy as np
import matplotlib.pyplot as plt

def _process_line_paral(line, event_set_filter):
    line = line.rstrip('\n')
    
    dt_end = line.find(']')
    thr_end = line.find(']', dt_end + 1)
    channel_end = line.find(']', thr_end + 1)
    lvl_end = line.find(']', channel_end + 1)
    
    dt = datetime.strptime(line[1:dt_end], '%Y-%m-%d %H:%M:%S.%f')
    thr = int(line[dt_end + 3:thr_end])
    channel = line[thr_end + 3:channel_end]
    lvl = line[channel_end + 3:lvl_end]
    
    if not line[lvl_end + 1:].startswith(' event: '):
        return None
    
    event_end = line.find(' ', lvl_end + len(' event: ') + 1)
    event = line[lvl_end + len(' event: ') + 1:event_end]
    
    if event_set_filter is not None:
        if event not in event_set_filter:
            return None

    res = {
        'timestamp': dt,
        'thread': thr,
        'loc': channel,
        'level': lvl,
        'evt': event,
        'type': 'generic_evt',
    }

    # make every prop starts with captial word
    for k, v in json.loads(line[event_end:]).items():
        res[k[0].upper() + k[1:]] = v
    
    return res


def load_generic(path, event_filters=None):
    # count lines first
    numLines = int(sp.check_output(["wc", "-l", path]).split()[0])
    
    # find optimal chunk size
    processes = os.cpu_count()
    chunkSize = numLines // processes
    
    if event_filters is not None:
        event_filters = set(event_filters)
    
    # the process
    logs = []
    with open(path) as f:
        with mp.Pool(processes=processes) as p:
            with tqdm(total=numLines) as pbar:
                def updater(log):
                    pbar.update()
                    return log
                logs = [updater(log) 
                    for log in
                    p.imap_unordered(partial(_process_line_paral,
                                             event_set_filter=event_filters),
                                     f, chunksize=chunkSize) if log is not None]
    
    df = pd.DataFrame.from_records(logs)
    df.sort_values(by=['timestamp'], inplace=True)
    return df


def load_mem(path):
    df = load_generic(path, event_filters=['alloc', 'dealloc'])
    
    df = df.drop(['level', 'loc', 'thread'], axis=1)

    # make sure index is consequtive
    df = df.reset_index(drop=True)
    
    mapping = {
        'alloc': 1,
        'dealloc': -1
    }
    
    df['Sign'] = df.evt.replace(mapping)

    return df


def calc_cumsum(df):
    cs = df.Size * df.Sign
    cs = cs.cumsum() / 1024 / 1024
    return cs

    
def plot_cs(cs, **kwargs):
    ax = cs.plot(**kwargs)
    return ax


def plot_mem(df, marker=False, offset=None, **kwargs):
    df = df.set_index('timestamp')
    if marker:
        if offset is None:
            offset = df.index[0]
        df.index = (df.index - offset) / pd.Timedelta(microseconds=1)
    elif offset is not None:
        df.index = (df.index - offset) / pd.Timedelta(microseconds=1)
    
    cumsum = calc_cumsum(df)
    ax = plot_cs(cumsum, **kwargs)
    if marker:
        css = df.reset_index()
        ax = css.plot(kind='scatter', x='timestamp', y=cumsum, c='evt',
                      cmap=plt.cm.get_cmap('bwr'), ax=ax)
    return ax


def plot_mem_persess(df, marker=False, offset=None, fill=False, sessaxs=None, **kwargs):
    cs = df.set_index('timestamp')
    if marker:
        if offset is None:
            offset = cs.index[0]
        cs.index = (cs.index - offset) / pd.Timedelta(microseconds=1)
    elif offset is not None:
        cs.index = (cs.index - offset) / pd.Timedelta(microseconds=1)

    if sessaxs is None:
        _, axs = plt.subplots(nrows=len(cs.Sess.unique()), sharex=True,
                              squeeze=False)
        axs = axs.flatten()
        sessaxs = {}
    for idx, (sess, dfsess) in enumerate(cs.groupby('Sess')):
        if sess not in sessaxs:
            ax = axs[idx]
            sessaxs[sess] = ax
        else:
            ax = sessaxs[sess]
            
        act = dfsess.evt * dfsess.Size
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
                          x=dfsess.timestamp, y=act, c=dfsess.evt,
                          cmap=plt.cm.get_cmap('bwr'), ax=ax)
    return sessaxs

def plot_compute_timeline(step, ax=None, **kwargs):
        checkpoints = tf_events
        step = step.copy()
        # columns as unix timestamp in us
        columns = [step[c].astype(np.int64) // 10**3 for c in checkpoints]
        # with offset subtracted
        offset = np.min([np.min(col) for col in columns])
        columns = [col - offset for col in columns]
        
        if ax is None:
            _, ax = plt.subplots()
        ax.hlines(y=np.zeros_like(columns[1]), xmin=columns[1], xmax=columns[2], linewidths=100)
        return ax