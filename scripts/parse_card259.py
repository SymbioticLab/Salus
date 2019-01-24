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


# 2018-09-01 06:22:21.180029: Step 341, loss=3.60 (33.8 examples/sec; 0.740 sec/batch)
ptn_iter = re.compile(r"""(?P<timestamp>.+): \s [sS]tep \s (?P<Step>\d+),\s
                          (loss|perplexity) .* \(
                          (?P<Speed>[\d.]+) \s examples/sec; \s
                          (?P<Duration>[\d.]+) \s sec/batch\)?""", re.VERBOSE)

def parse_iterations(path):
    path = Path(path)
    iterations = []
    with path.open() as f:
        for line in f:
            line = line.rstrip('\n')

            m = ptn_iter.match(line)
            if m:
                iterations.append(m.groupdict())
    df = pd.DataFrame(iterations)
    if len(df) == 0:
        print(f'File {path} is empty??')
    assert len(df) > 0
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['Speed'] = pd.to_numeric(df['Speed'])
    df['Step'] = pd.to_numeric(df.Step)
    df['Duration'] = pd.to_numeric(df.Duration)
    return df


def load_latency(path):
    path = Path(path)

    case2ex = {
        'case1': 'tf',
        'case2': 'salus',
        'case3': 'mps'
    }
    data = {}
    # case1, 2, 3
    for case in path.iterdir():
        ex = case2ex[case.name]

        s = []
        idx = []
        for strrate in case.iterdir():
            rate = float(strrate.name)  # rate req/sec
            for f in strrate.glob('*.*.*.*.output'):
                latencies = parse_iterations(f)
                s.append(latencies.Duration.mean())
                idx.append(rate)
        s = pd.Series(s, idx)
        data[ex] = s
    df = pd.DataFrame(data)

    # sort
    df = df.sort_index()
    return df


def plot_latency_throughtput(df, **kwargs):

    df = df * 1000  # in ms

    ax = df.plot(**kwargs)
    ax.set_xlabel('Request Rate (req/sec)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency vs RequestRate for inception3')

    return ax