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
import compmem as cm


def load_memcsv(path):
    df = pd.read_csv(path)
    df = df[~df.Network.str.contains("mnist")]
    # df = df.query('not Network.str.contains("mnist")')
    return df


def plot_mem(df, **kwargs):
    df = df[~df.Network.str.contains("eval")]
    # df = df.query('not Network.str.contains("eval")')
    df = df.set_index('Network')
    df = df / 1024
    df['Peak'] = df['Peak Mem (MB)']

    ax = df.plot(y=['Average', 'Peak'], kind='barh', **kwargs)
    return ax

def plot_inferencemem(df, **kwargs):
    df = df[df.Network.str.contains("eval")].copy()

    # sort
    #df['Model'] = df.Network.str.split('_')[0]
    #df['Batch Size'] = pd.to_numeric(df.Network.str.split('_')[1])
    df['Model'], df['Batch Size'] = df.Network.str.split('_').str
    df['Batch Size'] = pd.to_numeric(df['Batch Size'])
    df = df.sort_values(by=['Model', 'Batch Size'], ascending=[True, False])
    df = df.drop(['Model', 'Batch Size'], axis=1)


    df = df.set_index('Network')
    df = df / 1024
    df['Peak Usage'] = df['Peak Mem (MB)']
    df['Model Usage'] = df['Persistent Mem (MB)']


    ax = df.plot(y=['Peak Usage', 'Model Usage'], kind='barh', **kwargs)
    return ax

def do_membar(path):
    with plt.style.context(['seaborn-paper', 'mypaper', 'color3']):
            df = load_memcsv(path/'mem.csv')

            # only draw one batch size: the largest one
            df['Model'], df['BatchSize'] = df.Network.str.split('_').str
            df.BatchSize.replace({'small': 1, 'medium': 5, 'large': 10}, inplace=True)
            df['BatchSize'] = pd.to_numeric(df.BatchSize)
            df = df.reset_index().loc[df.reset_index().groupby(['Model'])['BatchSize'].idxmax()]
            df = df.drop(['index', 'BatchSize', 'Network'], axis=1)
            df = df.rename(columns={'Model': 'Network'})

            # sort values
            df = df.sort_values('Network', ascending=False)

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
            fig.set_size_inches(3.25, 2.5, forward=True)
            fig.savefig('/tmp/workspace/mem.pdf', dpi=300, bbox_inches='tight', pad_inches = .015)
            plt.close()


def do_singlemem(path):
    with plt.style.context(['seaborn-paper', 'mypaper', 'line12']):
        # a single mem
        df = cm.load_mem(path/'exp1'/'alloc.output')
        ax = cm.plot_mem(df, linewidth=.5, markevery=400, color='k')
        pu.cleanup_axis_bytes(ax.yaxis, maxN=4)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('TensorFlow\nMemory Usage')
        ax.set_ylim(bottom=0, top=12 * (1024**3))
        ax.set_xlim(left=1.5, right=7)

        fig = ax.figure
        fig.set_size_inches(3.25, 1.5, forward=True)
        fig.savefig('/tmp/workspace/exp1.pdf', dpi=300, bbox_inches='tight', pad_inches = .015)
        plt.close()


def prepare_paper(path):
    path = Path(path)
    do_membar(path)
    do_singlemem(path)


try:
    path
except NameError:
    path = Path('logs/nsdi19')
# prepare_paper(path)
#df = load_memcsv('/tmp/workspace/mem.csv')
#plot_inferencemem(df)