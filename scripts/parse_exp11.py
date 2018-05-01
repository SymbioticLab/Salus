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

def load_exp11(path):
    path = Path(path)
    
    data = pd.DataFrame(columns=['alexnet_100', 'inception3_25'])
    # load reference
    ref_jct = []
    for f in (path/'reference').glob('*.tf.*'):
        ref_jct.append(parse_jct(f))
    data.loc['Running alone'] = ref_jct
    
    # load Salus
    salus_jcts = []
    for f in path.iterdir():
        if f.is_dir():
            continue
        salus_jcts.append(parse_jct(f))
    data.loc['With Salus'] = salus_jcts
    
    # load w/o salus
    data.loc['Without Salus'] = [ref_jct[0]+ ref_jct[1] - 60, ref_jct[1]]
    return data

#%%
path = 'logs/osdi18/cc/exp11.2'

data = load_exp11(path)

plt.style.use(['seaborn-paper', 'mypaper'])

ax = data.plot.bar()
ax.legend(loc='lower right')
ax.tick_params(axis='x', rotation=0)
fig = ax.figure

fig.set_size_inches(3.45, 1.75, forward=True)
fig.tight_layout()
fig.savefig('/tmp/workspace/exp11.pdf', dpi=300)
#plt.close()