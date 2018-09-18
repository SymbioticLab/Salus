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

import plotutils as pu
import compmem as cm
import jctutils as ju


def load_data(path, name):
    df = ju.load_case(path/name)
    try:
        df = ju.refine_time_events(df, ju.load_serverevents(path))
    except ValueError:
        pass
    return df


def load_memory(path, st, ed, offset):
    st = offset + pd.to_timedelta(st, unit='s')
    ed = offset + pd.to_timedelta(ed, unit='s')




def plot_jcts(df, **kwargs):
    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()

    for col in df.columns:
        pu.cdf(df[col].dt.total_seconds(), label=col, ax=ax)
    ax.set_xlabel('JCT (s)')
    ax.set_ylabel('CDF')
    ax.legend()

    return ax

def plot_memory(df, **kwargs):
    pass


path = '/opt/desktop'
def prepare_paper(path):
    path = Path(path)
    with plt.style.context(['seaborn-paper', 'mypaper']):

        # fifo = ju.load_trace(path/'card266'/'salus'/'trace.csv')
        fifo = load_data(path/'card272'/'case0'/'fifo', 'case0.output')

        srtf = load_data(path/'card266'/'salus', 'card266.output')
        srtf_refine = ju.load_refine(path/'card266'/'salus')

        fair = load_data(path/'card272'/'case2'/'salus', 'case2.output')
        pack = load_data(path/'card272'/'case1'/'salus', 'case1.output')

        fig, ax = plt.subplots()
        fig.set_size_inches(3.25, 1.85, forward=True)
        jcts = pd.DataFrame({'FIFO': fifo.JCT, 'SRTF': srtf.JCT, 'PACK': pack.JCT, 'FAIR': fair.JCT})
        plot_jcts(jcts, ax=ax)
        fig.tight_layout()
        fig.savefig('/tmp/workspace/card266-jct.pdf', dpi=300)

        fig, axs = plt.subplots(nrows=4, sharex=True)
        fig.set_size_inches(3.25, 5.35, forward=True)
        _, colors = ju.plot_timeline(fifo, ax=axs[0], linewidth=2.5)
        axs[0].set_ylabel('FIFO')
        axs[0].legend().remove()
        axs[0].set_xlabel('')

        ju.plot_timeline(srtf.drop(['LaneId'], axis=1), ax=axs[1], linewidth=2.5, colors=colors)
        ju.plot_refine(axs[1], srtf, srtf_refine)
        axs[1].set_ylabel('SRTF')

        ju.plot_timeline(pack, ax=axs[2], linewidth=2.5, colors=colors)
        axs[2].set_ylabel('PACK')

        ju.plot_timeline(fair.drop(['LaneId'], axis=1), ax=axs[3], linewidth=2.5, colors=colors)
        axs[3].set_ylabel('FAIR')

        #fig.subplots_adjust(bottom=0.35)
        #axs[-1].legend(loc="upper center", frameon=False,
        #              bbox_to_anchor=[0.5, -0.8],
        #              #bbox_transform=fig.transFigure,
        #              fontsize='x-small',
        #              ncol=3
        #              #mode='expand'
        #              )

        fig.tight_layout()
        fig.savefig('/tmp/workspace/card266.pdf', dpi=300)
    return fifo, srtf, srtf_refine, fair, pack