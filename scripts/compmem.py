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
            tqdm(total=numLines) as pb:
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


def calc_cumsum(df):
    cs = df.Size * df.Sign
    cs = cs.cumsum()
    return cs


def normalize_time(df, offset=None):
    if 'timestamp' not in df:
        return df, offset

    df = df.set_index('timestamp')

    if offset is None:
        offset = df.index[0]
    df.index = (df.index - offset) / pd.Timedelta(microseconds=1)

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
    if marker:
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
                             ax=ax, label=sess, **kwargs)
        ax.legend()
    return sessaxs


def plot_iters(df, mainOnly=True, offset=None, return_offset=False, **kwargs):
    if mainOnly:
        df = df[df.MainIter]

    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()

    df, offset = normalize_time(df, offset)

    # get all timestamp as xs
    xs = df.index
    # linestyles are set based on it's start or end
    mapping = {
        'start_iter': 'dashed',
        'end_iter': 'solid'
    }
    linestyles = [mlines._get_dash_pattern(ls)
                  for ls in df.evt.replace(mapping)]

    lc = pu.axvlines(xs, linestyles=linestyles, ax=ax, **kwargs)
    if return_offset:
        return lc, offset
    return lc


def plot_iters2(df, y=0.8, mainOnly=True, offset=None, return_offset=False, **kwargs):
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
    y = [y] * len(xmin)
    trans = ax.get_xaxis_transform(which='grid')

    # line
    lc = ax.hlines(y, xmin, xmax, transform=trans, linewidth=2, **kwargs)
    color = lc.get_color()
    # start markers
    sm = ax.scatter(xmin, y, marker='>', transform=trans, color=color)
    # end markers
    em = ax.scatter(xmax, y, marker='o', transform=trans, color=color)
    return lc


def plot_iters_persess(df, offset=None, sessaxs=None, merge=True,
                       mainOnly=True, **kwargs):
    sesses = df.Sess.unique()

    if sessaxs is None:
        sessaxs = {}
        numAx = 1 if merge else len(sesses)
        _, axs = plt.subplots(nrows=numAx, sharex=True, squeeze=False)
        axs = axs.flatten()
        for sess, ax in zip(sesses, itertools.cycle(axs)):
            sessaxs[sess] = ax

    # check each sess must have an entry in sessaxs
    for sess in sesses:
        assert sess in sessaxs

    # seperate data for each session
    for sess, dfsess in df.groupby('Sess'):
        ax = sessaxs[sess]

        lc, offset = plot_iters(dfsess, mainOnly, offset, ax=ax,
                                return_offset=True, **kwargs)

    return sessaxs


def plot_mem_iters(alloc, iters, offset=None):
    def groupby(df, col):
        return {
            g: df.loc[v].reset_index(drop=True)
            for g, v in df.groupby('Sess').groups.items()
        }

    galloc = groupby(alloc, 'Sess')
    giters = groupby(iters, 'Sess')

    if set(galloc.keys()) != set(giters.keys()):
        raise ValueError('Not the same set of sessions')

    sesses = galloc.keys()
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    iters_y = 0.5
    for sess, color in zip(sesses, itertools.cycle(cycle)):
        salloc, offset = normalize_time(galloc[sess], offset)
        siters, offset = normalize_time(giters[sess], offset)

        plot_mem(salloc, offset=offset, color=color)
        plot_iters2(siters, y=iters_y, offset=offset, color=color)
        iters_y += 0.05


def plot_compute_timeline(step, ax=None, **kwargs):
        checkpoints = tf_events
        step = step.copy()
        # columns as unix timestamp in us
        columns = [step[c].astype(np.int64) // 10**3 for c in checkpoints]
        # with offset subtracted
        offset = np.min([np.min(col) for col in columns])
        columns = [col - offset for col in columns]

        if ax is None:
            _, ax = plt.subplots()
        ax.hlines(y=np.zeros_like(columns[1]), xmin=columns[1], xmax=columns[2], linewidths=100)
        return ax
