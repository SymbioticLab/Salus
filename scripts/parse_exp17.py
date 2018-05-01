#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 04:54:36 2018

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

def load_exp17(path):
    path = Path(path)
    dftf = pd.read_csv(path / 'jct-baseline.csv')
    dfsalus = pd.read_csv(path / 'jct-salus.csv')
    return dftf.set_index('Network'), dfsalus.set_index('Network')

#%%
path = 'logs/osdi18/conflux'

tf, salus = load_exp17(path)

column = '20iter-avg'
pits = pd.DataFrame({
    'Normalized Per Iteration Training Time': salus[column] / tf[column]
})
    
pits = pits[~pits.index.str.startswith('mnist')]

plt.style.use(['seaborn-paper', 'mypaper'])
ax = pits.plot.bar(legend=None)
pu.axhlines(1.0, ax=ax, color='r', linestyle='--')

ax.set_ylim(0.9, 1.25)
ax.set_xlabel('Workloads')
ax.set_ylabel('Normalized Per Iteration Training Time')

ax.tick_params(axis='x', labelsize=7)

ax.figure.set_size_inches(3.45, 2.75, forward=True)
ax.figure.tight_layout()
ax.figure.savefig('/tmp/workspace/exp17.pdf', dpi=300)
#plt.close()