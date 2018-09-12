#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 21:11:52 2018

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

NVIDIA_SPEED = 30  # GB/s


def load_jct(path):
    df = pd.read_csv(path/'jct-tfdist.csv')
    df = df.set_index('Network').dropna(axis=1)
    return df


def load_transfer_speed(path):
    df = pd.read_csv(path/'mem.csv')

    df['TransferTime'] = df['Persistent Mem (MB)'] / (NVIDIA_SPEED * 1024)  # seconds

    df = df.set_index('Network')
    return df


def load_data(path):
    path = Path(path)

    jct = load_jct(path)
    speeds = load_transfer_speed(path)

    # sort index before merge
    jct = jct.sort_index()
    speeds = speeds.sort_index()
    df = pd.concat([jct, speeds], axis=1)

    df = df.reset_index()

    return df


def plot_eval_pit_vs_speed(df, **kwargs):
    df = df.query('Network.str.contains("eval")').copy()

    # sort
    #df['Model'] = df.Network.str.split('_')[0]
    #df['Batch Size'] = pd.to_numeric(df.Network.str.split('_')[1])
    df['Model'], df['Batch Size'] = df.Network.str.split('_').str
    df['Batch Size'] = pd.to_numeric(df['Batch Size'])
    df = df.sort_values(by=['Model', 'Batch Size'], ascending=[True, False])
    df = df.drop(['Model', 'Batch Size'], axis=1)

    df = df.set_index('Network')

    df['Latency'] = df['20iter-avg']

    ax = df.plot(y=['TransferTime', 'Latency'], kind='barh', **kwargs)
    return ax


path = '/tmp/workspace'
def prepare_paper(path):
    with plt.style.context(['seaborn-paper', 'mypaper']):
        df = load_data(path)

        fig, ax = plt.subplots()
        fig.set_size_inches(3.25, 4, forward=True)

        plot_eval_pit_vs_speed(df, ax=ax)
        ax.set_xlabel('Time (s)')
        ax.yaxis.label.set_size(8)

        fig.tight_layout()
        fig.savefig('/tmp/workspace/card260.pdf', dpi=300, bbox_inches='tight', pad_inches = .005)
