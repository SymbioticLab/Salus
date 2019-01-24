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
Created on Tue Sep 18 01:04:40 2018

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
import jctutils as ju


def load_data(path):
    path = Path(path)

    df = cm.load_mem(path/'alloc.output')

    # GPU only
    df = df[df.Allocator.str.contains("GPU")]

    iters = cm.load_iters(path/'alloc.output')
    iters = iters[iters.MainIter]

    iters = iters.pivot_table(values='timestamp',
                        index=['StepId', 'GraphId', 'Sess',],
                        columns='evt', aggfunc='first').reset_index()

    # select only a single step
    #st, ed = iters.loc[iters.index[0], ['start_iter', 'end_iter']]
    #df = df[(df.timestamp >= st) & (df.timestamp < ed)]

    # alloc sizes
    return df, iters


def plot_size_cdf(df, cdf_ax, cumsum_ax, cdf_kws=None, cumsum_kws=None, **kwargs):
    sizes = df[df.Sign > 0].Size

    if cdf_kws is None:
        cdf_kws = {}
    cdf_kws.update(kwargs)
    pu.cdf(sizes, ax=cdf_ax, **cdf_kws)

    if cumsum_kws is None:
        cumsum_kws = {}
    cumsum_kws.update(kwargs)
    sizes.sort_values().reset_index(drop=True).cumsum().plot(ax=cumsum_ax, **cumsum_kws)

    pu.cleanup_axis_bytes(cdf_ax.xaxis, format='%(value).0f %(symbol)s')
    cdf_ax.set_ylabel('CDF')
    cdf_ax.set_xlabel('Memory Allocation Size')

    pu.cleanup_axis_bytes(cumsum_ax.yaxis, maxN=5, format='%(value).0f %(symbol)s')
    cumsum_ax.set_ylabel('Cumsum of Sizes')
    cumsum_ax.set_xlabel('Memory Allocations')


path = Path('logs/nsdi19')
def prepare_paper(path=path):
    df, iters = load_data(path/'card275'/'case2')

    fst, fed = iters.loc[iters.index[0], ['start_iter', 'end_iter']]
    sst, sed = iters.loc[iters.index[-2], ['start_iter', 'end_iter']]
    beforeAlloc = df[(df.timestamp < fst)]
    # fAlloc = df[(df.timestamp >= fst) & (df.timestamp < fed)]
    sAlloc = df[(df.timestamp >= sst) & (df.timestamp < sed)]

    KB = 1024
    MB = 1024 * KB
    model = beforeAlloc
    ephemeral = sAlloc[sAlloc.Size >= MB]
    framework = sAlloc[sAlloc.Size < MB]

    with plt.style.context(['seaborn-paper', 'mypaper', 'color3']):
        fig, axs = plt.subplots(ncols=2, nrows=1, squeeze=False, gridspec_kw={'width_ratios':[3,2]})
        fig.set_size_inches(3.25, 1.5, forward=True)

        plot_size_cdf(framework, axs[0][0], axs[0][1], label='Framework',
                      marker=',', markevery=0.1, linestyle=':', linewidth=2)
        plot_size_cdf(model, axs[0][0], axs[0][1], label='Model',
                      marker='.', markevery=0.1, linestyle='-.', linewidth=1,
                      cumsum_kws={'zorder': 10})
        plot_size_cdf(ephemeral, axs[0][0], axs[0][1], label='Ephemeral',
                      marker='^', markevery=0.05, linestyle='-', markersize=3, linewidth=1)

        axs[0][0].set_xlim(left=1)
        axs[0][0].set_xscale('log', basex=2)
        axs[0][0].set_xticks([1, 2**8, 2**16, 2**24, 2**32])
        axs[0][0].tick_params(axis='x',
        #   labelsize='xx-small'
        )
        bytesformatter = pu.FuncFormatter(lambda x, pos: pu.bytes2human(x, format='%(value).0f%(symbol)s'))
        axs[0][0].xaxis.set_major_formatter(bytesformatter)
        #pu.cleanup_axis_bytes(axs[0][0].xaxis, maxN=5, format=)

        axs[0][1].set_ylim(bottom=1)
        axs[0][1].set_yscale('log', basey=2)
        axs[0][1].set_yticks([1, 2**8, 2**16, 2**24, 2**32, 2**37])
        axs[0][1].tick_params(axis='y',
        #   labelsize='xx-small'
        )
        bytesformatter = pu.FuncFormatter(lambda x, pos: pu.bytes2human(x, format='%(value).0f%(symbol)s'))
        axs[0][1].yaxis.set_major_formatter(bytesformatter)
        #pu.cleanup_axis_bytes(axs[0][0].xaxis, maxN=5, format=)

        # legend at the bottom
        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3,
                   bbox_to_anchor = (0,-0.1,1,1),
                   bbox_transform = fig.transFigure
        )

        fig.tight_layout()
        fig.savefig('/tmp/workspace/card275.pdf', dpi=300, bbox_inches='tight', pad_inches = .015)
        plt.close()