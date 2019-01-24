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
Created on Mon Sep 10 22:56:08 2018

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


def load_jcts(path):
    """Load data from jct-salus, jct-tfdist, jct-tfdist-mps"""
    path = Path(path)
    flavors = [
        ('Salus', 'jct-salus.csv'),
        ('TF', 'jct-tfdist.csv'),
        ('MPS', 'jct-tfdist-mps.csv')
    ]
    data = {}
    for case, logfile in flavors:
        df = pd.read_csv(path/logfile)
        df = df.dropna(axis=1).dropna(axis=0)

        df['Network'] = df.Network.str.replace('small', '1')
        df['Network'] = df.Network.str.replace('medium', '5')
        df['Network'] = df.Network.str.replace('large', '10')

        df['BatchSize'] = df.Network.str.rpartition('_')[2]
        df['BatchSize'] = pd.to_numeric(df.BatchSize)
        df = df.sort_values(['Network', 'BatchSize'])
        df = df.set_index('Network')

        data[case] = df['20iter-avg']

    # check each series has the same index
    df = pd.DataFrame(data)

    # drop mnist
    df = df[~df.index.str.contains("mnist")]
    #df = df.query('not index.str.contains("mnist")')

    # only keep eval with bs=1
    #df = df.query('index.str.contains("eval") and index.str.endswith("_1")')
    df = df[df.index.str.contains("eval") & df.index.str.endswith("_1")]

    return df


def load_cardlog(path):
    path = Path(path)
    ptn = re.compile('Average excluding first iteration: (?P<latency>[\d.]+) sec/batch')
    models = []
    latencies = []
    for file in (path/'card270'/'case1').glob('*.*.*.*.output'):
        model = file.name.split('.')[0]
        model = model.replace('small', '1')
        model = model.replace('medium', '5')
        model = model.replace('large', '10')
        latency = None
        with file.open() as f:
            for line in f:
                m = ptn.search(line)
                if m:
                    latency = float(m.group('latency'))
                    break
        assert latency is not None
        models.append(model)
        latencies.append(latency)
    df = pd.DataFrame({'Latency': latencies, 'Network': models})
    df = df.groupby('Network').mean()

    return df


def plot_latency(df, **kwargs):
    # make index label beautiful
    df = df.set_index(df.index.str.rsplit('eval_1').str[0])

    # use ms as unit
    df = df * 1000

    ax = pu.bar(df, **kwargs)
    ax.set_ylabel('Latency (ms)')
    return ax


path = 'logs/nsdi19'
def prepare_paper(path):
    path = Path(path)

    df = load_jcts(path)

    # load in exp data
    exp = load_cardlog(path)
    # Drop anything without exp data
    df = df.assign(Salus=exp).dropna()

    # TODO: fix deep speech, which runs even faster than TF
    #df = df.query('not index.str.contains("speech")')
    df = df[~df.index.str.contains("speech")]

    with plt.style.context(['seaborn-paper', 'mypaper', 'hatchbar']):

        # add color to the cycler
        cycle = mpl.rcParams['axes.prop_cycle'].by_key()
        cycle['color'] = ['ed7d31', '244185', '8cb5df', '000000',   'dcedd0', '006d2c']
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(**cycle)
        # override color cycle
        #plt.rc('axes', prop_cycle=cycler('color', ['ed7d31', '000000', '8cb5df', 'dcedd0']))

        fig, axs = plt.subplots(ncols=2, gridspec_kw={'width_ratios':[4, 1]})
        fig.set_size_inches(6.5, 1.85, forward=True)

        axs[0].set_prop_cycle(None)
        cycle = axs[0]._get_lines.prop_cycler
        for _, prop in zip(range(6), cycle):
            print(prop)

        # set plot bar order
        order = ['Salus','TF', 'MPS']
        df = df[order]

        ax = plot_latency(df, ax=axs[0])
        for _, prop in zip(range(6), cycle):
            print(prop)
        ax.set_ylim([0, 60])
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Draw utilization, current number is hard coded
        # we ran 42 jobs, 3 of each (without speech, which is wierd)
        # so 42 for TF
        ut = pd.DataFrame([42, 6, 1], index=['TF', 'MPS', 'Salus'])
        ax = ut.plot.bar(ax=axs[1], color='w', edgecolor='k', hatch='////', linewidth=1)
        #ax.tick_params(axis='x', labelrotation=0)
        ax.legend().remove()
        ax.set_ylabel('# of GPUs needed')

        fig.tight_layout()
        fig.savefig('/tmp/workspace/card270.pdf', dpi=300, bbox_inches='tight', pad_inches = .005)
        plt.close()