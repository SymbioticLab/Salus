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

dfa = load_mem('/tmp/workspace/alex.csv')
dfv = load_mem('/tmp/workspace/vgg.csv')
#%%

def plot_cs(cs):
    return cs.plot()


def plot_df(df):
    cs = df.set_index('timestamp')
    cs['act'] = cs.act.cumsum() / 1024 / 1024
    return plot_cs(cs.act)
    
plot_df(dfa)
plot_df(dfv)

#%%
def find_minmax(df):
    cs = df.set_index('timestamp').act.cumsum() / 1024 / 1024
    # first find max
    ma = cs.max()
    
    # duration
    dur = cs.index[-1] - cs.index[0]
    
    # we only want later half of the duration
    st = cs.index[-1] - dur * 0.5
    ed = cs.index[-1] - dur * 0.1
    
    # select
    cspartial = cs[(cs.index >= st) & (cs.index <= ed)]
    # visual
    ax = plot_cs(cspartial)
    
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


data = []
for item in Path('logs/mem/tf').iterdir():
    df = load_tfmem(str(item / 'alloc.output'))
    fig = plt.figure()
    ma, persist, avg, ax = find_minmax(df)
    data.append((item.name, persist, ma, avg, ma - persist))
data = pd.DataFrame(data, columns=['Network',
                             'Persistent Mem (MB)',
                             'Peak Mem (MB)',
                             'Average',
                             'Peak'])
data.to_csv('/tmp/workspace/mem.csv', index=False)