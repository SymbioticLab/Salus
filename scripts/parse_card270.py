#!/usr/bin/env python3
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

        df['BatchSize'] = df.Network.str.rpartition('_')[2].map({'small': 0, 'medium': 1, 'large': 2})
        df['BatchSize'] = pd.to_numeric(df.BatchSize)
        df = df.sort_values(['Network', 'BatchSize'])
        df = df.set_index('Network')

        data[case] = df['20iter-avg']

    # check each series has the same index
    df = pd.DataFrame(data)

    # drop mnist
    df = df.query('not index.str.contains("mnist")')

    # only keep eval with bs=1
    df = df.query('index.str.contains("eval") and index.str.endswith("_1")')

    return df


def plot_latency(df, **kwargs):
    # make index label beautiful
    df = df.set_index(df.index.str.rsplit('eval_1').str[0])

    # use ms as unit
    df = df * 1000

    ax = df.plot.bar(**kwargs)
    ax.set_ylabel('Latency (ms)')
    return ax