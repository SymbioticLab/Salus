#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 00:39:38 2018

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


def load_memcsv(path):
    df = pd.read_csv(path)
    df = df.query('not Network.str.contains("mnist") and not Network.str.contains("eval")')
    return df


def plot_mem(df, **kwargs):
    df = df.set_index('Network')
    df = df / 1024
    df['Peak'] = df['Peak Mem (MB)']

    ax = df.plot(y=['Average', 'Peak'], kind='barh', **kwargs)
    return ax


def prepare_paper(path):
    path = Path(path)
    with plt.style.context(['seaborn-paper', 'mypaper']):
        df = load_memcsv(path/'mem.csv')

        # draw larger ones first
        ax = plot_mem(df)
        ax.set_xlabel('Memory Usage (GB)')
        ax.set_ylabel('')
        ax.legend(fontsize='xx-small',frameon=False)
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.tick_params(axis='y', which='major', labelsize=8, length=2)
        ax.yaxis.label.set_size(8)
        #ax.xaxis.tick_top()

        #fig.tight_layout()
        #fig.subplots_adjust(top=1, bottom=0, left=0, right=1)

        fig = ax.figure
        fig.set_size_inches(3.25, 4.5, forward=True)
        fig.savefig('/tmp/workspace/mem.pdf', dpi=300, bbox_inches='tight', pad_inches = .005)
        # plt.close()

path = '/tmp/workspace'
prepare_paper(path)