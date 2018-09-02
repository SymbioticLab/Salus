#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 04:25:58 2018

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


def load_trace(path):
    df = pd.read_csv(path)
    df = df.sort_values(by='submit_time')
    
    models = defaultdict(dict)
    curr = 0
    for idx, row in df.iterrows():
        if curr < row['submit_time']:
            curr = row['submit_time']
        models[idx]['Queued'] = row['submit_time']
        models[idx]['Started'] = curr
        curr += row['duration']
        models[idx]['Finished'] = curr
    
    data = [
        {
            "name": '{model_name}.tf.{iterations}iter.{job_id}'.format(**df.iloc[idx]),
            "Finished": m['Finished'],
            "Queued": m['Queued'],
            "Started": m['Started'],
            "queuing": m['Started'] - m['Queued'],
            "JCT": m['Finished'] - m['Queued']
        }
        for idx, m in models.items()
    ]
    df = pd.DataFrame(data)
    
    for col in ['Finished', 'Queued', 'Started', 'queuing', 'JCT']:
        df[col] = pd.to_timedelta(df[col], unit='s')
    return df


def plot_timeline(df, colors=None, **kwargs):
    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()
    
    linewidth = kwargs.pop('linewidth', 10)
    
    # sort df by no
    df['No'] = pd.to_numeric(df['name'].str.rpartition('.')[2])
    df = df.sort_values(by='No')    
    
    offset = df.Queued.min()
    qmin = (df.Queued - offset) / pd.Timedelta(1, unit='s')
    xmin = (df.Started - offset) / pd.Timedelta(1, unit='s')
    xmax = (df.Finished - offset) / pd.Timedelta(1, unit='s')
    lcs = []
 
    if colors is None:
        colors = [None] * len(df)
    for idx, q, left, right, name, color in zip(df['No'], qmin, xmin, xmax, df.name, colors):
        model, _, _, no = name.split('.')
        line = ax.plot([q, left], [idx, idx], linewidth=linewidth, color='#b6b6b6')[0]
        line = ax.plot([left, right], [idx, idx],
                       linewidth=linewidth, label='#{}: {}'.format(no, model),
                       color=color,
                       **kwargs)[0]
        lcs.append(line)
    ax.legend()
    
    ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Workload')
    ax.yaxis.set_ticks([])
    return lcs


def prepare_paper(path):
    plt.style.use(['seaborn-paper', 'mypaper'])
    
    path = Path(path)/'card236'/'salus'
    df = load_case(path/'card236.output')
    trace = load_trace(path/'trace.csv')
    
    fig, axs = plt.subplots(nrows=2, sharex=True)
    fig.set_size_inches(3.25, 2.35, forward=True)
    lines = plot_timeline(trace, ax=axs[0], linewidth=2.5)
    axs[0].set_ylabel('FIFO')
    axs[0].legend().remove()
    axs[0].set_xlabel('')
    
    colors = [l.get_color() for l in lines]
    plot_timeline(df, ax=axs[1], linewidth=2.5, colors=colors)
    axs[1].set_ylabel('Salus')
    #fig.subplots_adjust(bottom=0.35)
    axs[1].legend(loc="upper center", frameon=False,
                  bbox_to_anchor=[0.5, -0.8],
                  #bbox_transform=fig.transFigure,
                  fontsize='x-small',
                  ncol=3
                  #mode='expand'
                  )
    
    fig.tight_layout()
    fig.savefig('/tmp/workspace/card236.pdf', dpi=300)
