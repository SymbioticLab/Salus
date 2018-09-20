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
import subprocess as sp
import tempfile

import pandas as pd
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import plotutils as pu
import compmem as cm
import jctutils as ju


def load_data(path):
    path = Path(path)
    # for all cases
    data = []
    for case in path.iterdir():
        slog = case/'server.output'
        with slog.open() as f:
            st = None
            ed = None
            ptn_exec = re.compile(r"""^\[(?P<timestamp>\d+-\d+-\d+\s\d+:\d+:\d+\.\d{6}) (\d{3})?\]\s
                               \[(?P<thread>\d+)\]\s
                               \[(?P<loc>\w+)\]\s
                               \[(?P<level>\w+)\]\s
                               (?P<content>.*)$""",
                          re.VERBOSE)
            for line in f:
                m = ptn_exec.search(line.rstrip())
                if m is None:
                    continue
                content = m.group('content')
                # Find first "Checking for create lane", use this as whole job start time
                if st is None and content.startswith('Checking to create'):
                    st = pd.to_datetime(m.group('timestamp'))
                # last "Closing session" as whole finish time
                if content.startswith('Closing session'):
                    ed = pd.to_datetime(m.group('timestamp'))

        assert st is not None
        assert ed is not None
        salus_makespan = ed - st

        # find num of salus jobs
        njobs = len(list(case.glob('*.salus.*.*.output')))

        # find jct of tf job
        tfjct = None
        network = None
        for fpath in case.glob('*.tfdist.*.*.output'):
            with fpath.open() as f:
                ptn = re.compile('^JCT: (?P<jct>[\d.]+) s')
                for line in f:
                    m = ptn.search(line)
                    if m:
                        tfjct = float(m.group('jct'))
                        break
            # also use tf's as the network name
            network = fpath.name.split('.')[0]
            break
        assert tfjct is not None
        assert network is not None
        data.append({
            'Network': network,
            'Salus': salus_makespan,
            'TF': pd.to_timedelta(tfjct * njobs, 's'),
        })
    return pd.DataFrame(data)


def plot_makespan(df, **kwargs):
    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()

    # show network on xaxis
    df = df.set_index('Network')

    # use min as unit
    for col in df.columns:
        df[col] = df[col] / pd.Timedelta(1, 'm')
    #df.dt.total_seconds

    df.plot.bar(ax=ax, **kwargs)
    # draw directly using prop cycler
    #cycler = ax._get_lines.prop_cycler
    #for col, prop in zip(df.columns, cycler):
    #    prop.update(kwargs)
    #    ax.bar(df[col].index, df[col], label=col, **prop)

    ax.set_ylabel('Makespan (min)')
    ax.legend()

    return ax


path = 'logs/nsdi19'
def prepare_paper(path):
    path = Path(path)
    with plt.style.context(['seaborn-paper', 'mypaper', 'color3']):
        # fifo = ju.load_trace(path/'card266'/'salus'/'trace.csv')
        df = load_data(path/'card271')

        fig, ax = plt.subplots()
        fig.set_size_inches(3.25, 1.85, forward=True)

        # set col order
        df = df[['Network', 'Salus', 'TF']]

        ax = plot_makespan(df, ax=ax)

        ax.tick_params(axis='x', rotation=0)
        ax.set_xlabel('')

        fig.tight_layout()
        fig.savefig('/tmp/workspace/card271.pdf', dpi=300)
        plt.close()
    return df