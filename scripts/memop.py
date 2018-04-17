#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:22:28 2018

@author: peifeng
"""

def only_steps(steps, idxs):
    ss = steps.step.sort_values().unique().tolist()
    return steps[steps.step.isin([ss[idx] for idx in idxs])]


dfop = load_salus('logs/memop/salus/resnet50_25/perf.output')
dfop = only_steps(dfop, [2,3,4,5,6,7,8,9,10,11,12,13,14,15])
dfop = unify_names(dfop)

dfmem = load_mem('logs/memop/salus/resnet50_25/alloc.output')

# to unix timestamp in us
dfmem['timestamp'] = dfmem.timestamp.astype(np.int64) // 10**3

for c in salus_events:
    dfop[c] = dfop[c].astype(np.int64) // 10**3
    

fig, ax = plt.subplots()
_, offset = draw_salus(ax, dfop, set_y=False)

ax2 = ax.twinx()
plot_df_withop(dfmem, ax=ax2, offset=offset)

ax.set_ylabel('op id')
ax.set_xlabel('relative time (us)')
ax2.set_ylabel('mem (MB)')