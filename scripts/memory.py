#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 23:50:04 2018

@author: peifeng
"""
import parse_log as pl
import parse_nvvp as pn
import plotutils as pu
import pandas as pd
import functools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
tqdm.pandas()

def load_mem(path, return_logs=False, parallel_workers=0):
    try:
        logs = pl.load_file(path, parallel_workers)
    except TypeError:
        logs = path

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
    df['act'] = df.act.cumsum() / 1024 / 1024
    # make sure index is consequtive
    df = df.reset_index(drop=True)
    
    if return_logs:
        return df, logs
    else:
        return df

def load_holds(path):
    try:
        logs = pl.load_file(path)
    except TypeError:
        logs = path
    
    df = pd.DataFrame(l.__dict__ for l in logs)
    
    # get sess to ticket mapping
    mapping = df[df.type == 'tracker_ticket'][['sess', 'ticket']].set_index('ticket')
    mapping = mapping.to_dict()['sess']
    
    df = df[df.type.isin(['tracker_hold', 'tracker_unhold'])]
    df = df.drop(['entry_type', 'level', 'loc', 'thread'], axis=1)
    
    df.type.replace({
        'tracker_hold': 1,
        'tracker_unhold': -1,
    }, inplace=True)
    
    df['sess'] = df.ticket.replace(mapping)
    df = df.set_index('timestamp').sort_index().reset_index()
    
    return df

def load_tfmem(path):
    try:
        logs = pl.load_file(path)
    except TypeError:
        logs = path

    df = pd.DataFrame(l.__dict__ for l in logs)
    df = df[df.type.isin(['tf_alloc', 'tf_dealloc'])]
    df = df[['addr', 'size', 'type', 'timestamp']].set_index('timestamp').sort_index()

    # fill na size in tf_dealloc
    addrs = {}
    def find_size(row):
        addr, size, t = row
        if pd.isna(size):
            row['size'] = addrs[addr]
        else:
            addrs[addr] = size
        return row

    df = df.progress_apply(find_size, axis=1).reset_index()

    # compute act
    mapping = {
        'tf_alloc': 1,
        'tf_dealloc': -1
    }
    df.type.replace(mapping, inplace=True)
    df['act'] = df.type * df['size']
    df['act'] = df.act.cumsum() / 1024 / 1024
    # make sure index is consequtive
    df = df.reset_index(drop=True)
    return df

def load_iters(path):
    _, iters = pn.parse_iterations(str(path))
    sts, eds = zip(*iters)
    iters = sts[:1] + eds
    return pd.Series(iters)

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

    ax = plot_cs(cs.act, **kwargs)
    if marker:
        css = cs.reset_index()
        ax = css.plot(kind='scatter', x='timestamp', y='act', c='type', cmap=plt.cm.get_cmap('bwr'), ax=ax)
    return ax


def plot_df_persess(df, marker=False, offset=None, fill=False, sessaxs=None, **kwargs):
    cs = df.set_index('timestamp')
    if marker:
        if offset is None:
            offset = cs.index[0]
        cs.index = (cs.index - offset) / pd.Timedelta(microseconds=1)
    elif offset is not None:
        cs.index = (cs.index - offset) / pd.Timedelta(microseconds=1)

    if sessaxs is None:
        _, axs = plt.subplots(nrows=len(cs.sess.unique()), sharex=True,
                              squeeze=False)
        axs = axs.flatten()
        sessaxs = {}
    for idx, (sess, dfsess) in enumerate(cs.groupby('sess')):
        if sess not in sessaxs:
            ax = axs[idx]
            sessaxs[sess] = ax
        else:
            ax = sessaxs[sess]
            
        act = dfsess.type * dfsess['size']
        act = act.cumsum() / 1024 / 1024
        
        if fill:
            # fill nans
            act = pd.DataFrame(act).reset_index()
            nans = np.where(np.empty_like(act.values), np.nan, np.nan)
            data = np.hstack([nans, act.values]).reshape(-1, act.shape[1])
            act = pd.DataFrame(data, columns=act.columns)
            act[0] = act[0].ffill()
            act['timestamp'] = act['timestamp'].bfill()
            act = act.set_index('timestamp').dropna()
        
        plot_cs(act, ax=ax, **kwargs)
        if marker:
            css = dfsess.reset_index()
            ax = css.plot(kind='scatter',
                          x=dfsess.timestamp, y=act, c=dfsess.type,
                          cmap=plt.cm.get_cmap('bwr'), ax=ax)
    return sessaxs


def plot_df_withop(df, offset, **kwargs):
    # columns as unix timestamp in us
    df['timestamp'] = df.timestamp.astype(np.int64)
    # with offset subtracted
    if offset is None:
        offset = np.min(df.timestamp)
    df['timestamp'] = df.timestamp - offset
    df = df.set_index('timestamp')
    return plot_cs(df.act, **kwargs)


def find_minmax(df, plot=False):
    cs = df.set_index('timestamp').act
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


# Simulation of consequtave deallocation

def find_pc1(df, dfpart):
    """Using peak mem and consecutive dealloc"""
    # find iter max
    peak = df[df.step >= 1].act.max()
    threahold = 10
    peakthr = 0.8
    cd = 0
    for _, row in dfpart.iterrows():
        act = row['act']
        if row['type'] < 0:
            cd += 1
        else:
            cd = 0
        if cd >= threahold and act >= peakthr * peak:
            return row['timestamp']
    return None


def find_pc2(df, dfpart):
    """Find using peak mem and slope"""
    peak = df[df.step >= 1].act.max()
    peakthr = 0.9
    window = len(dfpart) // 50

    buf = []
    for idx, row in dfpart.iterrows():
        act = row['act']
        if len(buf) == window:
            buf.pop(0)
        buf.append((row['timestamp'].timestamp(), act))
        if len(buf) < 2:
            continue

        stx, sty = buf[0]
        edx, edy = buf[-1]
        slope = (edy - sty) / (edx - stx)
        if slope < 0 and act >= peakthr * peak:
            return row['timestamp']
    print("Didn't find good pc point, showing slopes")
    return None


def find_pc3(df, dfpart):
    """Find using peak mem and slope with sliding window average"""
    peak = df[df.step >= 1].act.max()

    window = len(dfpart) // 50
    if window == 0:
        window = 2

    peakthrs = [0.9, 0.7]
    slopethrs = [0.1, 1]

    buf = []
    maxslope = 0
    stx, sty = None, None
    for _, row in dfpart.iterrows():
        edx, edy = row['timestamp'].timestamp(), row['act']
        if stx is not None:
            slope = (edy - sty) / (edx - stx)

            buf.append(slope)
            avgslope = sum(buf[-window:]) / len(buf[-window:])

            if avgslope > 0:
                maxslope = max(avgslope, maxslope)
            else:
                for pthr, sthr in zip(peakthrs, slopethrs):
                    if avgslope < (-maxslope * sthr) and row['act'] >= (peak * pthr):
                        return row['timestamp']

        stx, sty = edx, edy

    print("Didn't find good pc point, showing slopes")
    # ax = pd.DataFrame(buf, index=dfpart.timestamp).plot()

    return None


def sim_dealloc(path):
    dirpath = Path(path)
    dfpath = dirpath / 'alloc.output'
    iterpath = dirpath / 'rpc.output'

    df = load_mem(str(dfpath))
    iters = load_iters(str(iterpath))
    # add iter info to df
    df['step'] = df.timestamp.apply(lambda ts: iters.searchsorted(ts)[0])

    print(f'{dirpath.name}: Loaded data')

    # find phase changing point
    phasechanging = []
    for nStep in range(1, iters.index[-1] + 1):
        # focus on one iteration
        dfpart = df[df.step == nStep]

        point = find_pc2(df, dfpart)
        if point is None:
            # be conservative about the point we find
            point = dfpart.iloc[-1]['timestamp']

        phasechanging.append((nStep, point))
        print(f'{dirpath.name}: Found phase changing point for step {nStep}')
    phasechanging = pd.DataFrame(phasechanging, columns=['step', 'timestamp'])
    print(f'{dirpath.name}: Done')
    return dirpath.name, df, iters, phasechanging


def draw_sim(df, iters, phasechanging):
    offset = df.timestamp[0]
    riters = (iters - offset) / pd.Timedelta(microseconds=1)
    if phasechanging is not None:
        phasechanging['timestamp'] = (phasechanging.timestamp - offset) / pd.Timedelta(microseconds=1)

    ax = plot_df(df, marker=True, offset=offset)
    ax.vlines(riters, *ax.get_ylim())
    if phasechanging is not None:
        ax.vlines(phasechanging.timestamp, *ax.get_ylim(), colors='r', linestyles='dashed')
    return ax

#%% def main
def main():
#%% do sim

    with mp.Pool() as pool:
        data = pool.map(sim_dealloc, Path('logs/osdi18/mem/salus').iterdir())
    
    for n, df, iters, pc in data:
        plt.figure()
        ax = draw_sim(df, iters, pc)
        ax.set_title(n)


#%% Produce mem.csv
    def process_mem(item, loader=load_tfmem):
        print(f"{item.name}: Loading log file")
        df = loader(str(item / 'alloc.output'))
        print(f"{item.name}: figure")
        ma, persist, avg, cspartial = find_minmax(df, plot=False)
        print(f"{item.name}: found")
        return item.name, persist, ma, avg, ma - persist, cspartial
    
    
    with mp.Pool() as pool:
        data = pool.map(process_mem, Path('/tmp/workspace/meminfer').iterdir())
    
    for n, p, m, a, _, csp in data:
        plt.figure()
        plot_cs(csp).set_title(n)
    data = [x[:-1] for x in data]
    
    # data = [process_mem(item) for item in Path('logs/mem/tf').iterdir()]
    data = pd.DataFrame(data, columns=['Network',
                                       'Persistent Mem (MB)',
                                       'Peak Mem (MB)',
                                       'Average',
                                       'Peak'])
    data.to_csv('/tmp/workspace/mem.csv', index=False)

#%% Produce mem-salus.csv
    with mp.Pool() as pool:
        datasalus = pool.map(functools.partial(process_mem, loader=load_mem),
                             Path('logs/osdi18/cc/mem/salus').iterdir())
    for n, p, m, a, _, csp in datasalus:
        plt.figure()
        plot_cs(csp).set_title(n)
    datasalus = [x[:-1] for x in datasalus]
    datasalus = pd.DataFrame(datasalus, columns=['Network',
                                                 'Persistent Mem (MB)',
                                                 'Peak Mem (MB)',
                                                 'Average',
                                                 'Peak'])
    datasalus.to_csv('/tmp/workspace/mem-salus.csv', index=False)

#%% Draw session alloc

    dfmem, logs = load_mem('logs/osdi18/cc/exp10/alloc.output', return_logs=True, parallel_workers=5)
    df = load_holds(logs)
    
    axs = plot_df_persess(dfmem)
    axs = plot_df_persess(df, fill=True, sessaxs=axs)

#%% Size CDF
    path = Path('/tmp/workspace/plotrun/resnet50_50')
    mempath = str(path/'alloc.output')
    iterpath = str(next(path.glob('*.salus.*.output')))
    df = load_mem(mempath)
    iters = load_iters(iterpath)
    # add iter info to df
    df['step'] = df.timestamp.apply(lambda ts: iters.searchsorted(ts)[0])
    
    # select only a single step
    df = df[df['step'] == 6]
    
    alloc = df[df.type > 0]
    sizes = alloc['size']
    
    _, axs = plt.subplots(nrows=2)
    
    ax = pu.cdf(sizes, ax=axs[0])
    pu.cleanup_axis_bytes(ax.xaxis)
    ax.set_title('CDF of allocation request sizes')
    ax.set_xlabel('Allcation Request Size')
    
    ax = sizes.sort_values().reset_index(drop=True).cumsum().plot(ax=axs[1])
    ax.set_title('Cumulative sum of request sizes')
    ax.set_xlabel('Allocation Request')
    pu.cleanup_axis_bytes(ax.yaxis)

#%% main
if __name__ == '__main__':
    #main()
    pass