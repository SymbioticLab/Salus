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


def load_latency(path):
    path = Path(path)

    speeds = {}
    for f in path.glob('*.*.*.*.output'):
        model, executor, iterstr, runid, _ = f.name.split('.')
        s = parse_iterations(f)
        
        speeds['{}.{}.{}'.format(model, executor, runid)] = s.set_index('timestamp').Duration
    return pd.DataFrame(speeds)


def plot_latency(df, **kwargs):
    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()
        
    df = df * 1000  # ms

    ax = df.boxplot(showfliers=False, ax=ax, **kwargs)
    ax.set_ylabel('Latency (ms)')
    return ax


def plot_model_vs_latency(dfs, **kwargs):
    data = [
        {
            "# of models": len(df.columns),
            "Avg. Latency (ms)": df.mean().mean()
        }
        for df in dfs
    ]
    data = pd.DataFrame(data).set_index('# of models')
    
    ax = data['Avg. Latency (ms)'].plot.bar(**kwargs)
    ax.set_ylabel('Avg. Latency (ms)')
    return ax


def prepare_paper(path):
    path = Path(path)
    three = load_latency(path/'card252'/'case2')
    single = load_latency(path/'card252'/'case1')

    with plt.style.context(['seaborn-paper', 'mypaper']):
        fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True,
                                       gridspec_kw={'width_ratios':[3, 3]})
        
        
        ax = plot_latency(three, ax=ax0)
        ax.set_xticklabels(['1', '2', '3'])
        ax.set_title(f'{len(three.columns)} inception3\ninference on Salus')
        ax.xaxis.grid(False)

        ax = plot_latency(single, ax=ax1)
        labels = [
            {
                'tf': 'TF',
                'salus': 'Salus',
                'tfdist': 'TFDist'
            }[label.get_text().split('.')[1]]
            for label in ax.get_xticklabels()
        ]
        ax.set_xticklabels(labels)
        ax.set_ylabel('')
        ax.set_title('Baseline')
        ax.xaxis.grid(False)
        
        fig.tight_layout()
        fig.set_size_inches(3.25, 1.5, forward=True)
        fig.tight_layout()
        fig.savefig('/tmp/workspace/card252.pdf', dpi=300)
    

path = '/tmp/workspace'