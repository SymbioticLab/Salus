#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 01:11:53 2018

@author: peifeng
"""

from __future__ import print_function, absolute_import, division

import re
from datetime import datetime
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotutils as pu


def parse_output_float(outputfile, pattern, group=1):
    """Parse outputfile using pattern"""
    if not outputfile.exists():
        msg = f'File not found: {outputfile}'
        raise ValueError(msg)

    ptn = re.compile(pattern)
    with outputfile.open() as f:
        for line in f:
            line = line.rstrip()
            m = ptn.match(line)
            if m:
                try:
                    return float(m.group(group))
                except (ValueError, IndexError):
                    continue
    raise ValueError(f"Pattern `{pattern}' not found in output file {outputfile}")

def parse_jct(outputfile):
    return parse_output_float(outputfile, r'^JCT: ([0-9.]+) .*')

def load_exp62(path):
    path = Path(path)

    data = pd.DataFrame(columns=['Average JCT', 'Makespan'])
    # load FIFO
    fifo_jct = parse_jct(next((path/'tf').iterdir()))
    data.loc['FIFO'] = [fifo_jct, fifo_jct * 2]

    # load SP
    sp_jcts = []
    for f in (path/'tf-nomps').iterdir():
        sp_jcts.append(parse_jct(f))
    data.loc['SP'] = [np.mean(sp_jcts), np.max(sp_jcts)]

    # load MPS
    mps_jcts = []
    for f in (path/'tf-mps').iterdir():
        mps_jcts.append(parse_jct(f))
    data.loc['SP+MPS'] = [np.mean(mps_jcts), np.max(mps_jcts)]

    # load MPS+OC
    mpsoc_jcts = []
    for f in (path/'tf-mps-oc').iterdir():
        mpsoc_jcts.append(parse_jct(f))
    data.loc['SP+MPS+OC'] = [np.mean(mpsoc_jcts), np.max(mpsoc_jcts)]

    # load Salus
    salus_jcts = []
    for f in (path/'salus/2').iterdir():
        salus_jcts.append(parse_jct(f))
    data.loc['Salus'] = [np.mean(salus_jcts), np.max(salus_jcts)]
    return data

#%%
path = 'logs/nsdi19/exp6_2'

data = load_exp62(path)

data = data[['Average JCT']]
data = data[data.index != 'FIFO']

with plt.style.context(['seaborn-paper', 'mypaper', 'gray']):

    ax = data.plot.bar(hatch='////', color='w', edgecolor='k')
    fig = ax.figure

    ax.tick_params(axis='x', labelsize=7, rotation=0)
    ax.set_ylim(0, 150)
    #ax.set_xlabel('')
    ax.set_ylabel('Time (s)')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1),)
    ax.annotate('{:.2f}'.format(data.loc['SP+MPS+OC']['Average JCT']),
                xy=[data.index.get_loc('SP+MPS+OC'), 150],
                xytext=[0, 7], textcoords='offset points',
                size=7,
                horizontalalignment='center', verticalalignment='top')
    #ax.annotate('{:.2f}'.format(data.loc['SP+MPS+OC']['Makespan']),
    #            xy=[data.index.get_loc('SP+MPS+OC') + 0.1, 145], size=7,
    #            horizontalalignment='left', verticalalignment='top')

    fig.set_size_inches(3.45, 1.25, forward=True)
    fig.tight_layout()
    #fig.subplots_adjust(hspace=0.1)
    fig.savefig('/tmp/workspace/exp62.pdf', dpi=300)
    plt.close()