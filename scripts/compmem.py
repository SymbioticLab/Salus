#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:59:19 2018

Plot per session memory alloc and computation together

@author: peifeng
"""
import parse_log as pl
import pandas as pd
from datetime import datetime
import json
import multiprocessing as mp
import subprocess as sp
from tqdm import tqdm
from functools import partial
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import plotutils as pu
import itertools
import memmap
from contextlib import ExitStack
from pathlib import Path


def _process_line_paral(line, event_filters, updater=None):
    with ExitStack() as stack:
        if updater:
            stack.callback(updater)

        line = line.rstrip('\n')

        if line[0] != '[':
            return None

        dt_end = line.find(']')
        thr_end = line.find(']', dt_end + 1)
        channel_end = line.find(']', thr_end + 1)
        lvl_end = line.find(']', channel_end + 1)

        dt = datetime.strptime(line[1:dt_end], '%Y-%m-%d %H:%M:%S.%f')
        thr = int(line[dt_end + 3:thr_end])
        channel = line[thr_end + 3:channel_end]
        lvl = line[channel_end + 3:lvl_end]

        if not line[lvl_end + 1:].startswith(' event: '):
            return None

        event_end = line.find(' ', lvl_end + len(' event: ') + 1)
        event = line[lvl_end + len(' event: ') + 1:event_end]

        if event_filters is not None:
            if event not in event_filters:
                return None

        res = {
            'timestamp': dt,
            'thread': thr,
            'loc': channel,
            'level': lvl,
            'evt': event,
            'type': 'generic_evt',
        }

        # make every prop starts with captial word
        for k, v in json.loads(line[event_end:]).items():
            res[k[0].upper() + k[1:]] = v

        return res


def load_generic(path, event_filters=None, task_per_cpu=20):
    # count lines first
    print('Counting file lines...')
    numLines = int(sp.check_output(["wc", "-l", path]).split()[0])

    # find optimal chunk size
    numCPU = os.cpu_count()
    chunkSize = numLines // numCPU // task_per_cpu
    if chunkSize == 0:
        chunkSize = 1

    if event_filters is not None:
        event_filters = set(event_filters)

    path = Path(path)
    # the process
    with path.open() as f, mp.Pool(processes=numCPU) as p,\
            tqdm(total=numLines, desc='Parsing {}'.format(path.name), unit='lines') as pb:
        def updater(log):
            pb.update()
            return log

        ilog = (updater(log) for log in
                p.imap_unordered(partial(_process_line_paral,
                                         event_filters=event_filters),
                                 f, chunksize=chunkSize))
        df = pd.DataFrame(log for log in ilog if log is not None)

    df.sort_values(by=['timestamp'], inplace=True)
    return df


def load_mem(path, **kwargs):
    df = load_generic(path, event_filters=['alloc', 'dealloc'], **kwargs)

    # sanity check for nullptr in log
    if len(df.query('Ptr == 0')):
        print("There are nullptrs in log, check allocator logging.")
        df = df.query('Ptr != 0')

    df = df.drop(['level', 'loc', 'thread'], axis=1)

    # make sure index is consequtive
    df = df.reset_index(drop=True)

    mapping = {
        'alloc': 1,
        'dealloc': -1
    }

    df['Sign'] = df.evt.replace(mapping)

    return df


def load_iters(path, **kwargs):
    df = load_generic(path, event_filters=['start_iter', 'end_iter'], **kwargs)

    df = df.drop(['level', 'loc', 'entry_type', 'thread', 'type'],
                 axis=1, errors='ignore')
    df = df.dropna(how='all', axis=1)

    return df


def load_comp(path, **kwargs):
    events = ['queued', 'running', 'done']
    df = load_generic(path, event_filters=events, **kwargs)

    df = df.drop(['level', 'loc', 'entry_type', 'thread', 'type'],
                 axis=1, errors='ignore')
    df = df.dropna(how='all', axis=1)

    # fix column names
    df = df.rename(columns={'Session': 'Sess'})

    return df


def load_all(path):
    path = Path(path)
    alloc = load_mem(path/'alloc.output')
    iters = load_iters(path/'perf.output')
    comp = load_comp(path/'perf.output')
    return alloc, iters, comp


def calc_cumsum(df):
    cs = df.Size * df.Sign
    cs = cs.cumsum()
    return cs


def normalize_time(df, offset=None):
    if 'timestamp' not in df:
        return df, offset

    if offset is None:
        offset = df.timestamp[0]
    df['timestamp'] = (df.timestamp - offset).dt.total_seconds()

    df = df.set_index('timestamp')

    return df, offset


def plot_cs(cs, **kwargs):
    ax = cs.plot(**kwargs)
    return ax


def plot_marker(df, cs, **kwargs):
    css = df.reset_index()
    ax = css.plot(kind='scatter', x='timestamp', y=cs, c='Sign',
                  cmap=plt.cm.get_cmap('bwr'), **kwargs)
    return ax


def plot_mem(df, marker=False, offset=None, return_offset=False, **kwargs):
    df, offset = normalize_time(df, offset)

    cumsum = calc_cumsum(df)
    ax = plot_cs(cumsum, **kwargs)
    if isinstance(marker, bool) and marker:
        ax = plot_marker(df, cumsum, ax=ax)
    pu.cleanup_axis_bytes(ax.yaxis)

    if return_offset:
        return ax, offset
    return ax


def plot_mem_persess(df, marker=False, offset=None,
                     sessProps=None, merge=True, **kwargs):
    sesses = df.Sess.unique()

    sessaxs = sessProps.pop('ax', None)
    if sessaxs is None:
        sessaxs = {}
        numAx = 1 if merge else len(sesses)
        _, axs = plt.subplots(nrows=numAx, sharex=True, squeeze=False)
        axs = axs.flatten()
        for sess, ax in zip(sesses, itertools.cycle(axs)):
            sessaxs[sess] = ax

    sesscolors = sessProps.pop('color', None)
    if sesscolors is None:
        sesscolors = {sess: None for sess in sesses}

    # check each sess must have an entry in sessaxs
    for sess in sesses:
        assert sess in sessaxs

    offset = None
    for sess, dfsess in df.groupby('Sess'):
        ax = sessaxs[sess]
        color = sesscolors[sess]

        _, offset = plot_mem(dfsess, marker, offset=offset, return_offset=True,
                             ax=ax, label=sess, color=color, **kwargs)
        ax.legend()
    return sessaxs


def plot_iters(df, y=0.8, mainOnly=True, offset=None, return_offset=False, **kwargs):
    if 'transform' in kwargs:
        raise ValueError("'transform' not allowed as we provide our own transform")

    if mainOnly:
        df = df[df.MainIter]

    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()

    df, offset = normalize_time(df, offset)

    # xmin are start_iter
    xmin = df[df.evt == 'start_iter'].index
    # xmax are end_iter
    xmax = df[df.evt == 'end_iter'].index

    if len(xmin) != len(xmax):
        raise ValueError('Some iteration not stopped')

    # set y as in axes coordinate
    def calc_y(row):
        if mainOnly:
            return y
        base_y = y
        if not row['MainIter']:
            base_y -= 0.2
        return base_y
    y = df.query("evt == 'start_iter'").apply(calc_y, axis=1)
    trans = ax.get_xaxis_transform(which='grid')

    # line
    lc = ax.hlines(y, xmin, xmax, transform=trans, linewidth=2, **kwargs)
    color = lc.get_color()
    # start markers
    sm = ax.scatter(xmin, y, marker='>', transform=trans, color=color)
    # end markers
    em = ax.scatter(xmax, y, marker='o', transform=trans, color=color)
    return lc, sm, em


def plot_comp(df, y=0.2, separateY=True, offset=None, return_offset=False, **kwargs):
    if 'transform' in kwargs:
        raise ValueError("'transform' not allowed as we provide our own transform")

    # prepare data
    df, offset = normalize_time(df, offset)
    df = df.reset_index()
    df = df.pivot_table(values='timestamp',
                        index=['StepId', 'GraphId', 'Name',
                               'Type', 'Device', 'MainIter'],
                        columns='evt', aggfunc='first').reset_index()

    # find start of GPU compute for each iter, using the first GPU compute in main iter
    # done by sort values and drop duplicates
    xmindf = df.query('MainIter and Device.str.contains("GPU")')
    xmin = xmindf.sort_values('running', ascending=True).drop_duplicates(['StepId'])
    xmin = xmin.sort_values('StepId')

    # find end of GPU compute for each iter,
    # using the last _Recv in the same step on CPU
    mainIterSteps = df.query('MainIter').StepId.unique()
    xmaxdf = df.query('not MainIter and Type == "_Recv"'
                      ' and Device.str.contains("CPU")'
                      ' and StepId.isin(@mainIterSteps)')
    xmax = xmaxdf.sort_values('done', ascending=False).drop_duplicates(['StepId'])
    xmax = xmax.sort_values('StepId')

    assert len(xmin) == len(xmax)

    # prepare args
    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()
    linewidths = kwargs.pop('linewidths', 10)

    # set y as in axes coordinate
    def calc_y(row):
        if not separateY:
            return y
        base_y = y
        if 'GPU' in row['Device']:
            base_y += 0.3
        if not row['MainIter']:
            base_y -= 0.15
        return base_y
    y = xmin.apply(calc_y, axis=1)
    trans = ax.get_xaxis_transform(which='grid')

    print(f'{y}, {xmin.running}, {xmax.done}')

    lc = ax.hlines(y=y, xmin=xmin.running, xmax=xmax.done,
                   transform=trans, linewidths=linewidths, **kwargs)

    if return_offset:
        return lc, offset
    return lc


def plot_comp_old(df, y=0.2, separateY=True, offset=None, return_offset=False, **kwargs):
    """This is deprecated, as Compute method doesn't reflect actual computation."""
    if 'transform' in kwargs:
        raise ValueError("'transform' not allowed as we provide our own transform")

    # prepare data
    df, offset = normalize_time(df, offset)
    df = df.reset_index()
    df = df.pivot_table(values='timestamp',
                        index=['StepId', 'GraphId', 'Name',
                               'Type', 'Device', 'MainIter'],
                        columns='evt', aggfunc='first').reset_index()

    # prepare args
    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()
    linewidths = kwargs.pop('linewidths', 10)

    # set y as in axes coordinate
    def calc_y(row):
        if not separateY:
            return y
        base_y = y
        if row['Device'] == '/job:salus/replica:0/task:0/device:GPU:0':
            base_y += 0.3
        if not row['MainIter']:
            base_y -= 0.15
        return base_y
    y = df.apply(calc_y, axis=1)
    trans = ax.get_xaxis_transform(which='grid')

    lc = ax.hlines(y=y, xmin=df.running, xmax=df.done,
                   transform=trans, linewidths=linewidths, **kwargs)

    if return_offset:
        return lc, offset
    return lc


def plot_all(alloc, iters=None, comp=None, offset=None, ax=None, sessProps=None,
             mem_kws=None, iters_kws=None, comp_kws=None,
             **kwargs):
    def groupby(df, col):
        if df is None:
            return None
        return {
            g: df.loc[v].reset_index(drop=True)
            for g, v in df.groupby('Sess').groups.items()
        }

    galloc = groupby(alloc, 'Sess')
    giters = groupby(iters, 'Sess')
    gcomp = groupby(comp, 'Sess')

    def all_same_keys(*args):
        L = [set(d.keys()) for d in args if d is not None]
        return all(x == L[0] for x in L)

    if not all_same_keys(galloc, giters, gcomp):
        raise ValueError('Not the same set of sessions')

    sesses = galloc.keys()
    if sessProps is None:
        #cycle = ax._get_lines.prop_cycler
        cycle = [{} for sess in sesses]
    else:
        cycle = [sessProps[sess] for sess in sesses]

    if mem_kws is None:
        mem_kws = {}
    if iters_kws is None:
        iters_kws = {}
    if comp_kws is None:
        comp_kws = {}

    iters_y = 0.5
    comp_y = 0.4
    for sess, prop in zip(sesses, cycle):
        salloc, offset = normalize_time(galloc[sess], offset)
        print({**prop, **kwargs, **mem_kws})
        plot_mem(salloc, offset=offset, label=sess, ax=ax, **{**prop, **kwargs, **mem_kws})

        if giters is not None:
            siters, offset = normalize_time(giters[sess], offset)
            plot_iters(siters, y=iters_y, offset=offset, ax=ax, **{**prop, **kwargs, **iters_kws})

        if gcomp is not None:
            scomp, offset = normalize_time(gcomp[sess], offset)
            plot_comp(scomp, y=comp_y, offset=offset, ax=ax, **{**prop, **kwargs, **comp_kws})

        iters_y += 0.05
        comp_y += 0.02

    print(f"Used offset value: {offset}")

    if ax is None:
        ax = plt.gca()
    pu.cleanup_axis_timedelta(ax.xaxis)
    ax.xaxis.set_major_locator(pu.MaxNLocator(nbins=30))
    ax.legend()
