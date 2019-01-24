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
import matplotlib as mpl

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
    assert len(iterations) > 0
    fake = {}
    fake.update(iterations[-1])
    fake['Speed'] = 0
    fake['timestamp'] = (pd.to_datetime(fake['timestamp']) + pd.Timedelta(1, 'us')).strftime('%Y-%m-%d %H:%M:%S.%f')
    iterations.append(fake)

    fake = {}
    fake.update(iterations[0])
    fake['Speed'] = 0
    fake['timestamp'] = (pd.to_datetime(fake['timestamp']) - pd.Timedelta(1, 'us')).strftime('%Y-%m-%d %H:%M:%S.%f')
    iterations[:0] = [fake]

    df = pd.DataFrame(iterations)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['Speed'] = pd.to_numeric(df['Speed'])
    df['Step'] = pd.to_numeric(df.Step)
    df['Duration'] = pd.to_numeric(df.Duration)

    df = df.sort_values('timestamp')
    # calculate a cumulative speed
    # get batch size
    batch_size = int(path.name.partition('.')[0].partition('_')[-1])

    cumspeed = []
    start = df['timestamp'].iloc[0] - pd.Timedelta(df.Duration.iloc[0], 's')
    for idx, row in df.iterrows():
        images = batch_size * (idx + 1)
        dur = (row['timestamp'] - start) / pd.Timedelta(1, 's')
        cumspeed.append(images/dur)
    df['CumSpeed'] = cumspeed
    return df


def load_speeds(path, key='Speed'):
    path = Path(path)

    speeds = {}
    for f in path.glob('*.*.*.*.output'):
        model, executor, iterstr, runid, _ = f.name.split('.')
        s = parse_iterations(f)

        speeds['{}.{}'.format(model, runid)] = s.set_index('timestamp')[key]
    return pd.DataFrame(speeds)


def plot_speeds(df, total_kws=None, **kwargs):
    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()

    #df = df.interpolate(method='time', limit_area='inside')

    #df = df.resample('500ms').mean()

    df = df.reset_index()
    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / pd.Timedelta(1, 's')
    df = df.set_index('timestamp')
    ax = df.plot(ax=ax, **kwargs)

    df = df.fillna(0)
    total = 0
    for col in df.columns:
        total += df[col]
    if total_kws is None:
        total_kws = {}
    total.plot(ax=ax, color='k', linestyle='--', **total_kws)

    ax.legend().remove()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Images per second')
    return ax


path = 'logs/nsdi19'
def prepare_paper(path):
    path = Path(path)
    df = load_speeds(path/'card250'/'case1')

    # smooth data in the range of inception3_50.2
    tmin = df['inception3_50.2'].dropna().index.min()
    tmax = df['inception3_50.2'].dropna().index.max()

    df['inception3_50.2'] = df['inception3_50.2'].interpolate(method='time', limit_area='inside')
    smoothed = df['inception3_50.2'].rolling(15).mean().dropna()
    smoothed.drop(smoothed.tail(5).index,inplace=True)
    df.loc[smoothed.index, 'inception3_50.2'] = smoothed

    # also do this for others
    others = ['inception3_50.1', 'inception3_50.0']
    df[others] = df[others].interpolate(method='time', limit_area='inside')
    smoothed = df[others].rolling(15).mean().dropna()
    smoothed = smoothed.query('index >= @tmin and index <= @tmax')
    df.loc[smoothed.index, others] = smoothed

    with plt.style.context(['seaborn-paper', 'mypaper', 'line12']):
        fig, ax = plt.subplots()
        cycler = (mpl.cycler('color', ['ed7d31', '000000', '244185', '8cb5df'])
                + mpl.cycler('linestyle', ['-', '-.', ':', '--'])
                + mpl.cycler('marker', ['X', '*', '^', 'o'])
                + mpl.cycler('markersize', [4, 5, 3, 2.5]))
        ax.set_prop_cycle(cycler)
        plot_speeds(df, ax=ax,
                    markevery=8,
                    #markersize=3,
                    linestyle='-', linewidth=1,
                    total_kws={'marker': 'None', 'zorder': -1, 'linewidth': 1})

        fig.tight_layout()
        fig.set_size_inches(3.25, 2.35, forward=True)
        fig.savefig('/tmp/workspace/card250.pdf', dpi=300)
        plt.close()
