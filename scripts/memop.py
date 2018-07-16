#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:22:28 2018

@author: peifeng
"""
import numpy as np
import matplotlib.pyplot as plt

from optracing import load_salus, only_steps, unify_names, salus_events, draw_salus
from memory import load_mem, plot_df_withop

path = '/tmp/workspace/memop/salus/inception3_100'

dfop = load_salus(path + '/perf.output', filter_step=False)
dfop = only_steps(dfop, [2,3,4,5,6,7,8,9,10,11,12,13,14,15])
dfop = unify_names(dfop)

dfmem = load_mem(path + '/alloc.output')

# to unix timestamp in us
dfmem['timestamp'] = dfmem.timestamp.astype(np.int64) // 10**3

for c in salus_events:
    dfop[c] = dfop[c].astype(np.int64) // 10**3
    
#%%
fig, axs = plt.subplots(nrows=2)
ax = axs[0]
_, offset = draw_salus(ax, dfop, set_y=False)

ax2 = axs[1]
plot_df_withop(dfmem, ax=ax2, offset=offset)

ax.set_ylabel('op id')
ax.set_xlabel('relative time (us)')
ax2.set_ylabel('mem (MB)')