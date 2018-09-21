#!/usr/bin/env python3
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
    # count lines first
    print('Counting file lines...')
    numLines = int(sp.check_output(["wc", "-l", path]).split()[0])

    # find optimal chunk size
    numCPU = os.cpu_count()
    chunkSize = numLines // numCPU // task_per_cpu
    if chunkSize == 0:
        chunkSize = 1

    path = Path(path)
    # the process
    with path.open() as f, mp.Pool(processes=numCPU) as p,\
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
    with plt.style.context(['seaborn-paper', 'mypaper', 'color3']):
        df = parse_to_csv(path/'mem-pytorch')

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
    with plt.style.context(['seaborn-paper', 'mypaper', 'line12']):
        # a single mem
        df = load_pytorch(path/'mem-pytorch'/'resnet101_75_output.txt')
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
