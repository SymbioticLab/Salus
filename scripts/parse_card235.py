#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 07:38:29 2018

@author: peifeng
"""
from __future__ import print_function, absolute_import, division

import re
from datetime import datetime
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path

import pandas as pd
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import plotutils as pu
import compmem as cm


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
        
        speeds['{}.{}'.format(model, runid)] = s.set_index('timestamp').Speed
    return pd.DataFrame(speeds)


def plot_speeds(df, **kwargs):
    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()

    df = df.interpolate(method='time', limit_area='inside').fillna(0)
    
    df = df.reset_index()
    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / pd.Timedelta(1, 's')
    df = df.set_index('timestamp')
    ax = df.plot(ax=ax, **kwargs)
    
    total = 0
    for col in df.columns:
        total += df[col]
    total.plot(ax=ax, color='k', linestyle='--', **kwargs)
    
    ax.legend().remove()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Batches per second')
    return ax


def prepare_paper(path):
    path = Path(path)
    case2 = load_speeds(path/'card235'/'case2')
    case3 = load_speeds(path/'card235'/'case3')

    fig = plt.figure()
    fig.set_size_inches(3.25, 2.35, forward=True)
    
    ax = plot_speeds(case2, ax=fig.subplots())
    ax.set_title('Space sharing')
    fig.tight_layout()
    fig.savefig('/tmp/workspace/card235-1.pdf', dpi=300)
    
    fig = plt.figure()
    fig.set_size_inches(3.25, 2.35, forward=True)
    
    ax = plot_speeds(case3, ax=fig.subplots())
    ax.set_title('Time sharing')
    fig.tight_layout()
    fig.savefig('/tmp/workspace/card235-2.pdf', dpi=300)
    
