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
Created on Thu Sep 20 11:55:05 2018

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
import matplotlib.lines as mlines
import plotutils as pu
import itertools
from contextlib import ExitStack
from pathlib import Path

import compmem as cm
import memory as mem


def _process_line_paral(line, updater=None):
    with ExitStack() as stack:
        if updater:
            stack.callback(updater)

        line = line.rstrip('\n')

        if line[0] == '=':
            return None

        if not line[0].isdigit():
            return None

        if line.find(' ') == -1:
            # iteration mark
            return None

        time, act, delta = line.split(' ')
        td = pd.to_timedelta(float(time), unit='ms')
        delta = int(delta)
        act = int(act)

        if delta == 0:
            print('0 delta found!!')
            return None

        if delta > 0:
            event = 'alloc'
        else:
            event = 'dealloc'

        res = {
            'timestamp': td,
            'evt': event,
            'type': 'generic_evt',
            'Size': abs(delta),
            'Sign': delta // abs(delta),
            'act': act / 1024**2  # for compatible with memory.findminmax
        }

        return res


def load_pytorch(path, task_per_cpu=20):
    # check if we already have it
    if path.with_suffix(path.suffix + '.csv'):
        df = pd.read_csv(path.with_suffix(path.suffix + '.csv'))
        df['timestamp'] = pd.to_timedelta(df.timestamp)
        return df

    path, numLines = cm.find_file(path)

    # find optimal chunk size
    numCPU = os.cpu_count()
    chunkSize = numLines // numCPU // task_per_cpu
    if chunkSize == 0:
        chunkSize = 1

    # the process
    with cm.open_file(path) as f, mp.Pool(processes=numCPU) as p,\
            tqdm(total=numLines, desc='Parsing {}'.format(path.name), unit='lines') as pb:
        def updater(log):
            pb.update()
            return log

        ilog = (updater(log) for log in
                p.imap_unordered(_process_line_paral,
                                 f, chunksize=chunkSize))
        df = pd.DataFrame(log for log in ilog if log is not None)

    df.sort_values(by=['timestamp'], inplace=True)
    return df


def parse_to_csv(path):
    path = Path(path)

    csv_file = path.with_suffix('.csv')
    if csv_file.exists():
        return pd.read_csv(csv_file)

    data = []
    for file in path.glob('*_output.txt'):
        df = load_pytorch(file)
        ma, persist, avg, cspartial = mem.find_minmax(df, plot=False)
        name = file.name.rpartition('_')[0]
        data.append((name, persist, ma, avg, cspartial))

    #for n, p, m, a, csp in data:
    #    plt.figure()
    #    mem.plot_cs(csp).set_title(n)
    data = [x[:-1] for x in data]

    # data = [process_mem(item) for item in Path('logs/mem/tf').iterdir()]
    data = pd.DataFrame(data, columns=['Network',
                                       'Persistent Mem (MB)',
                                       'Peak Mem (MB)',
                                       'Average'])
    data['Peak'] = data['Peak Mem (MB)'] - data['Persistent Mem (MB)']
    data.to_csv('/tmp/workspace/mem-pytorch.csv', index=False)

    return data


def plot_mem(df, **kwargs):
    df = df[~df.Network.str.contains("eval")]
    # df = df.query('not Network.str.contains("eval")')
    df = df.set_index('Network')
    df = df / 1024
    df['Peak'] = df['Peak Mem (MB)']

    ax = df.plot(y=['Average', 'Peak'], kind='barh', **kwargs)
    return ax


def do_membar(path):
    df = parse_to_csv(path/'mem-pytorch')

    pu.matplotlib_fixes()
    with plt.style.context(['seaborn-paper', 'mypaper', 'color3']):

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
        fig.savefig('/tmp/workspace/mem-pytorch.pdf',
                    dpi=300, bbox_inches='tight', pad_inches = .015)
        plt.close()


def do_singlemem(path):
    # a single mem
    df = load_pytorch(path/'mem-pytorch'/'resnet101_75_output.txt')

    pu.matplotlib_fixes()
    with plt.style.context(['seaborn-paper', 'mypaper', 'line12']):
        ax = mem.plot_cs(df.set_index('timestamp').act * 1024**2, linewidth=.5,
                         markevery=200, color='k')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('PyTorch\nMemory Usage')
        pu.cleanup_axis_bytes(ax.yaxis, maxN=5)
        ax.set_ylim(bottom=0,
                    top=12 * (1024**3)
                    )
        ax.set_xlim(left=1,
                    right=8
                    )
        ax.legend().remove()

        fig = ax.figure
        fig.set_size_inches(3.25, 1.5, forward=True)
        fig.savefig('/tmp/workspace/exp1-pytorch.pdf',
                    dpi=300, bbox_inches='tight', pad_inches = .015)
        plt.close()

try:
    path
except NameError:
    path = Path('logs/nsdi19')


def prepare_paper(path):
    do_membar(path)
    do_singlemem(path)


if __name__ == '__main__':
    prepare_paper(path)
