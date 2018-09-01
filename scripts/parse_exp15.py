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
import compmem as cm


def load_case(path):
    df = pd.read_csv(path, header=None, sep=' ',
                     names=['date', 'time', 'event', 'skip', 'name'],
                     parse_dates=[['date', 'time']])
    df = df[['date_time', 'event', 'name']]
    df['timestamp'] = df['date_time']
    df = df.drop('date_time', axis=1)

    wls = df.pivot_table(values='timestamp', index=['name'],
                         columns='event', aggfunc='first').reset_index()

    for col in ['Started', 'Queued', 'Finished']:
        wls[col] = wls[col].str[:-1]
        wls[col] = pd.to_datetime(wls[col])
    wls['queuing'] = wls.Started - wls.Queued
    wls['JCT'] = wls.Finished - wls.Queued
    return wls


# 2018-09-01 06:22:21.180029: Step 341, loss=3.60 (33.8 examples/sec; 0.740 sec/batch)
ptn_iter = re.compile(r"""(?P<timestamp>.+): \s [sS]tep \s (?P<Step>\d+),\s
                          (loss|perplexity) .* \(
                          (?P<Speed>[\d.]+) \s examples/sec; \s
                          (?P<Duration>[\d.]+) \s sec/batch\)?""", re.VERBOSE)

def parse_iterations(path):
    path = Path(path)
    iterations = []
    with path.open() as f:
        for line in f:
            line = line.rstrip('\n')

            m = ptn_iter.match(line)
            if m:
                iterations.append(m.groupdict())
    df = pd.DataFrame(iterations)
    if len(df) == 0:
        print(f'File {path} is empty??')
    assert len(df) > 0
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['Speed'] = pd.to_numeric(df['Speed'])
    df['Step'] = pd.to_numeric(df.Step)
    df['Duration'] = pd.to_numeric(df.Duration)
    return df


def load_speeds(path):
    path = Path(path)
    
    
    speeds = {}
    for f in path.glob('*.*.*.*.output'):
        model, executor, iterstr, runid, _ = f.name.split('.')
        s = parse_iterations(f)
        
        speeds[model] = s.set_index('timestamp').Speed
    return pd.DataFrame(speeds)


def load_exp15(directory, name=None):
    if name is None:
        name = 'exp15'
    directory = Path(directory)
    salus = load_case(directory/'salus'/name + '.output')
    fifo = load_case(directory/'fifo'/name + '.output')
    sp = load_case(directory/'tf'/name + '.output')

    data = pd.DataFrame({
        'Salus': salus['JCT'],
        'FIFO': fifo['JCT'],
        'SP': sp['JCT']
    })
    queuing = pd.DataFrame({
        'Salus': salus['queuing'],
        'FIFO': fifo['queuing'],
        'SP': sp['queuing']
    })
    return data, queuing

def plot_data(data):
    data = data.copy()
    # convert columns to seconds
    for col in data.columns:
        data[col] = data[col] / pd.Timedelta(seconds=1)
    
    ax = data.plot.bar()
    ax.set_xlabel('Workload ID')
    ax.set_ylabel('JCT (s)')
    return ax

def plot_remaining(times, **kwargs):
    
    times = times.copy()
    times['timestamp'] = (times.timestamp - times.timestamp.min()) / np.timedelta64(1, 's')
    times = times.set_index('timestamp')
    ax = None
    for k, grp in times.groupby('Sess'):
        grp['RemainingTime'] = (grp.TotalRunningTime - grp.UsedRunningTime) / 1000
        ax = grp.plot(y='RemainingTime', ax=ax, label=k, **kwargs)
    ax.legend().remove()
    ax.set_ylabel('Estimated Remaining Time (s)')
    ax.set_xlabel('Time (s)')
    pu.cleanup_axis_timedelta(ax.xaxis)
    
    return ax

def plot_used(times, **kwargs):
    times = times.copy()
    times['timestamp'] = (times.timestamp - times.timestamp.min()) / np.timedelta64(1, 's')
    times = times.set_index('timestamp')
    ax = None
    for k, grp in times.groupby('Sess'):
        grp['UsedRunningTime'] = grp.UsedRunningTime / 1000
        ax = grp.plot(y='UsedRunningTime', ax=ax, label=k, **kwargs)
    ax.legend().remove()
    ax.set_ylabel('Running Time (s)')
    ax.set_xlabel('Time (s)')
    pu.cleanup_axis_timedelta(ax.xaxis)
    
    return ax
#%%


def paper_card234(path):
    path = Path(path)
    times = cm.load_generic(path/'card234'/'salus'/'preempt'/'perf.output',
                            event_filters=['sess_add_time'])
    
    plt.style.use(['seaborn-paper', 'mypaper'])
    ax = plot_remaining(times, linewidth=1)
    pu.axhlines(0.0, ax=ax, color='r', linestyle='--', linewidth=.5)
    
    #ax.set_ylim(0.9, 1.25)
    #ax.set_xlabel('Workloads')
    #ax.set_ylabel('Normalized Per Iteration\nTraining Time')
    
    #ax.tick_params(axis='x', labelsize=7)
    
    ax.figure.set_size_inches(3.25, 2.35, forward=True)
    ax.figure.tight_layout()
    ax.figure.savefig('/tmp/workspace/card234.pdf', dpi=300)
    
def paper_card233(path):
    path = Path(path)
    times = cm.load_generic(path/'card233'/'salus'/'fair'/'perf.output',
                            event_filters=['sess_add_time'])
    
    plt.style.use(['seaborn-paper', 'mypaper'])
    ax = plot_used(times, linewidth=1)
    #pu.axhlines(0.0, ax=ax, color='r', linestyle='--', linewidth=.5)
    
    ax.set_ylim(0, 350)
    #ax.set_xlabel('Workloads')
    #ax.set_ylabel('Normalized Per Iteration\nTraining Time')
    
    #ax.tick_params(axis='x', labelsize=7)
    
    ax.figure.set_size_inches(3.25, 2.35, forward=True)
    ax.figure.tight_layout()
    ax.figure.savefig('/tmp/workspace/card233.pdf', dpi=300)