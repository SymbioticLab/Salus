#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 23:50:04 2018

@author: peifeng
"""
import parse_log as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_mem(path):
    logs = pl.load_file(path)
    df = pd.DataFrame(l.__dict__ for l in logs)
    df = df[df.type.isin(['mem_alloc', 'mem_dealloc'])]
    df = df.drop(['entry_type', 'level', 'loc', 'thread'], axis=1)
    df['ticket'] = df.block.apply(lambda b: b['ticket'])
    df.drop('block', axis=1)
    
    mapping = {
            'mem_alloc': 1,
            'mem_dealloc': -1
    }
    
    df.type.replace(mapping, inplace=True)
    df['act'] = df.type * df['size']
    return df

def load_tfmem(path):
    logs = pl.load_file(path)
    df = pd.DataFrame(l.__dict__ for l in logs)
    df = df[df.type.isin(['tf_alloc', 'tf_dealloc'])]
    df = df[['addr', 'size', 'type', 'timestamp']].set_index('timestamp').sort_index()
    
    # fill na size in tf_dealloc
    def find_size(row):
        addr, size, t = row
        ts = row.name
        if pd.isna(size):
            s = df.query('timestamp < @ts & addr == @addr & type == "tf_alloc"')['size'][-1]
            row['size'] = s
        return row
            
    df = df.apply(find_size, axis=1).reset_index()
    
    # compute act
    mapping = {
            'tf_alloc': 1,
            'tf_dealloc': -1
    }
    df.type.replace(mapping, inplace=True)
    df['act'] = df.type * df['size']
    return df

#dfa = load_mem('/tmp/workspace/alex.csv')
#dfv = load_mem('/tmp/workspace/vgg.csv')
#%%

def plot_cs(cs, **kwargs):
    ax = cs.plot(**kwargs)
    return ax


def plot_df(df, marker=False, offset=None, **kwargs):
    cs = df.set_index('timestamp')
    if marker:
        if offset is None:
            offset = cs.index[0]
        cs.index = (cs.index - offset) / pd.Timedelta(microseconds=1)
    elif offset is not None:
        cs.index = (cs.index - offset) / pd.Timedelta(microseconds=1)
        
    cs['act'] = cs.act.cumsum() / 1024 / 1024
    ax = plot_cs(cs.act, **kwargs)
    if marker:
        css = cs.reset_index()
        ax = css.plot(kind='scatter', x='timestamp', y='act', c='type', cmap=plt.cm.get_cmap('bwr'), ax=ax)
    return ax

def plot_df_withop(df, offset, **kwargs):
    # columns as unix timestamp in us
    df['timestamp'] = df.timestamp.astype(np.int64)
    # with offset subtracted
    if offset is None:
        offset = np.min(df.timestamp)
    df['timestamp'] = df.timestamp - offset
    df = df.set_index('timestamp')
    df['act'] = df.act.cumsum() / 1024 / 1024
    return plot_cs(df.act, **kwargs)
    
#plot_df(dfa)
#plot_df(dfv)

#%%
def find_minmax(df, plot=False):
    cs = df.set_index('timestamp').act.cumsum() / 1024 / 1024
    # first find max
    ma = cs.max()
    
    # start from first non zero
    cspartial = cs[cs > (ma * 0.05)]
    # duration
    dur = cspartial.index[-1] - cspartial.index[0]
    
    # we only want later half of the duration
    st = cspartial.index[-1] - dur * 0.5
    ed = cspartial.index[-1] - dur * 0.1
    
    # select
    cspartial = cspartial[(cspartial.index >= st) & (cspartial.index <= ed)]
    # visual
    ax = None
    if plot:
        ax = plot_cs(cspartial)
    else:
        ax = cspartial
    
    # find min (persistant mem)
    persist = cspartial.min()
    
    # find average
    # convert to relative time
    rcs = cs.reset_index()
    rcs['timestamp'] = rcs.timestamp - rcs.timestamp[0]
    rcs['timestamp'] = rcs.timestamp / pd.Timedelta(microseconds=1)
    durus = dur / pd.Timedelta(microseconds=1)
    # integral
    avg = np.trapz(rcs.act, x=rcs.timestamp) / durus
    
    return ma, persist, avg, ax
#%%
from pathlib import Path
import multiprocessing as mp

def process_mem(item, loader=load_tfmem):
    print(f"{item.name}: Loading log file")
    df = loader(str(item / 'alloc.output'))
    print(f"{item.name}: figure")
    ma, persist, avg, cspartial = find_minmax(df, plot=False)
    print(f"{item.name}: found")
    return (item.name, persist, ma, avg, ma - persist, cspartial)

with mp.Pool() as pool:
    data = pool.map(process_mem, Path('logs/mem/tf').iterdir())
    
for n, p, m, a, _, csp in data:
    plt.figure()
    ax = plot_cs(csp)
    ax.set_title(n)
data = [x[:-1] for x in data]

#data = [process_mem(item) for item in Path('logs/mem/tf').iterdir()]
data = pd.DataFrame(data, columns=['Network',
                             'Persistent Mem (MB)',
                             'Peak Mem (MB)',
                             'Average',
                             'Peak'])
data.to_csv('/tmp/workspace/mem.csv', index=False)

#%%
import functools
with mp.Pool() as pool:
    datasalus = pool.map(functools.partial(process_mem, loader=load_mem),
                         Path('logs/mem/salus').iterdir())
for n, p, m, a, _, csp in datasalus:
    plt.figure()
    ax = plot_cs(csp)
    ax.set_title(n)
datasalus = [x[:-1] for x in datasalus]
datasalus = pd.DataFrame(datasalus, columns=['Network',
                             'Persistent Mem (MB)',
                             'Peak Mem (MB)',
                             'Average',
                             'Peak'])
datasalus.to_csv('/tmp/workspace/mem-salus.csv', index=False)