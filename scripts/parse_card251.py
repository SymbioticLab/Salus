#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 04:25:58 2018

@author: peifeng
"""

from __future__ import print_function, absolute_import, division

import re
from datetime import datetime
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path
import subprocess as sp
import tempfile

import pandas as pd
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import plotutils as pu
import compmem as cm


def load_case(path):
    df = pd.read_csv(path, header=None, sep=' ',
                     names=['date', 'time', 'event', 'skip', 'Model'],
                     parse_dates=[['date', 'time']])
    df = df[['date_time', 'event', 'Model']]
    df['timestamp'] = df['date_time']
    df = df.drop('date_time', axis=1)

    wls = df.pivot_table(values='timestamp', index=['Model'],
                         columns='event', aggfunc='first').reset_index()

    for col in ['Started', 'Queued', 'Finished']:
        wls[col] = wls[col].str[:-1]
        wls[col] = pd.to_datetime(wls[col])
    wls['queuing'] = wls.Started - wls.Queued
    wls['JCT'] = wls.Finished - wls.Queued
    return wls


def load_trace_fifo(path):
    df = pd.read_csv(path)
    df = df.sort_values(by='submit_time')

    models = defaultdict(dict)
    curr = 0
    for idx, row in df.iterrows():
        if curr < row['submit_time']:
            curr = row['submit_time']
        models[idx]['Queued'] = row['submit_time']
        models[idx]['Started'] = curr
        curr += row['duration']
        models[idx]['Finished'] = curr

    data = [
        {
            "Model": '{model_name}.tf.{iterations}iter.{job_id}'.format(**df.iloc[idx]),
            "Finished": m['Finished'],
            "Queued": m['Queued'],
            "Started": m['Started'],
            "queuing": m['Started'] - m['Queued'],
            "JCT": m['Finished'] - m['Queued']
        }
        for idx, m in models.items()
    ]
    df = pd.DataFrame(data)

    for col in ['Finished', 'Queued', 'Started', 'queuing', 'JCT']:
        df[col] = pd.to_timedelta(df[col], unit='s')
    return df


def load_refine(pathdir):
    # load preempt select events
    with tempfile.NamedTemporaryFile() as f:
        server_output = pathdir/'server.output'
        sp.check_call(['grep', 'fifo_select_sess', str(server_output)], stdout=f)
        f.flush()
        df = cm.load_generic(f.name, event_filters=['fifo_select_sess'])
    df = df.drop(['evt', 'level', 'loc', 'thread', 'type'], axis=1)

    # convert UTC from server to local
    df['timestamp'] = df.timestamp.dt.tz_localize('UTC').dt.tz_convert('US/Eastern').dt.tz_localize(None)

    sess2Model = {}
    # model name -> sess handle
    ptn = re.compile('Created session with handle (?P<sess>.+)$')
    for fpath in pathdir.glob('*.*.*.*.output'):
        with fpath.open() as f:
            for line in f:
                m = ptn.search(line)
                if m:
                    sess2Model[m.group('sess')] = fpath.name.rstrip('.output')

    # add model name info to it
    df['Model'] = df.Sess.map(sess2Model)

    # make sure every session is covered
    assert df.Model.isnull().sum() == 0

    # for convinent
    df['No'] = pd.to_numeric(df['Model'].str.rpartition('.')[2])

    return df


def load_serverevents(pathdir):
    # sess handle -> lane id
    with tempfile.NamedTemporaryFile() as f:
        server_output = pathdir/'server.output'
        sp.check_call(['grep', 'lane_assigned', str(server_output)], stdout=f)
        f.flush()
        df = cm.load_generic(f.name, event_filters=['lane_assigned'])
    df = df.drop(['evt', 'level', 'loc', 'thread', 'type'], axis=1)

    # sess handles are unique
    assert len(df.Sess.unique()) == len(df.Sess)

    # make Sess as index so we can lookup
    df = df.set_index('Sess')

    # add a new column
    df['Model'] = None

    # model name -> sess handle
    ptn = re.compile('Created session with handle (?P<sess>.+)$')
    for fpath in pathdir.glob('*.*.*.*.output'):
        with fpath.open() as f:
            for line in f:
                m = ptn.search(line)
                if m:
                    df.loc[m.group('sess'), 'Model'] = fpath.name.rstrip('.output')

    # reset index so we can use that later
    df = df.reset_index()
    return df


def refine_time_events(df, sevts):
    """Return a copy of df"""
    assert df.Model.is_unique
    assert sevts.Model.is_unique

    df = df.set_index('Model').sort_index()
    sevts = sevts.set_index('Model').sort_index()

    # check sevts contains all needed info
    assert sevts.index.equals(df.index)

    # Server logs in UTC, convert to local
    sevts['Started'] = sevts.timestamp.dt.tz_localize('UTC').dt.tz_convert('US/Eastern').dt.tz_localize(None)
    sevts = sevts.drop(['timestamp'], axis=1)

    df['Queued'] = df.Started
    df = df.drop(['Started'], axis=1)

    # set Model as index for both as then and then concat
    df = pd.concat([df, sevts], axis=1)

    # update queuing
    df['queuing'] = df.Started - df.Queued

    return df.reset_index()


def plot_timeline(df, colors=None, **kwargs):
    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()

    # sort df by no
    df['No'] = pd.to_numeric(df['Model'].str.rpartition('.')[2])
    df = df.sort_values(by='No')

    offset = df.Queued.min()
    qmin = (df.Queued - offset) / pd.Timedelta(1, unit='s')
    xmin = (df.Started - offset) / pd.Timedelta(1, unit='s')
    xmax = (df.Finished - offset) / pd.Timedelta(1, unit='s')

    if colors is None:
        color_cycle = ax._get_lines.prop_cycler
        colors = [next(color_cycle)['color'] for _ in qmin]

    for (_, row), q, left, right, color in zip(df.iterrows(), qmin, xmin, xmax, colors):
        barheight = 0.8
        # queuing time
        ax.barh(row.No, left - q, barheight, q, color='#b6b6b6')
        # run time
        bar = ax.barh(row.No, right - left, barheight, left,
                      color=color,
                      label='#{3}: {0}'.format(*row.Model.split('.')))
        if 'LaneId' in row:
            ax.text(right + 2, row.No, f'Lane {row.LaneId}',
                    ha='left', va='center', fontsize=3)

    # ax.legend()
    ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Workload')
    ax.yaxis.set_ticks([])
    return bar, colors


def plot_refine(ax, df, refine_data):
    # so that we can access job using no
    df = df.set_index('No')

    # for every preempt event pair, mask jobs that's not the left event's switch_to job
    offset = df.Queued.min()
    refine_data['Ntime'] = (refine_data['timestamp'] - offset) / pd.Timedelta(1, unit='s')

    # also convert df.Queued to relative time
    df['Started'] = (df.Started - offset) / pd.Timedelta(1, unit='s')
    df['Finished'] = (df.Finished - offset) / pd.Timedelta(1, unit='s')

    bars = []
    # group refine_data by laneId
    for laneId, grp in refine_data.groupby('LaneId'):
        magic = grp.iterrows()
        next(magic)
        for (_, left), (_, right) in zip(grp.iterrows(), magic):
            for no in df.index.unique():
                if no == left.No:
                    continue
                if laneId != df.loc[no].LaneId:
                    continue
                l = max(df.loc[no].Started, left.Ntime)
                r = min(df.loc[no].Finished, right.Ntime)
                if l >= r:
                    continue
                # make sure left and right within job no's started and finished
                # mask from left to right
                bars.append(ax.barh(no, r - l, 0.85, l, color='#ffffff', edgecolor='#ffffff'))

    return bars


def plot_lanes(refined_df, **kwargs):
    lanes = refined_df.groupby(['LaneId', 'LaneSize']).agg({
            'Queued': 'first',
            'Finished': 'last'
    }).rename(columns={'Queued':'Started'}).reset_index()

    tables = []
    for col in ['Started', 'Finished']:
        t = lanes.pivot_table(values='LaneSize', columns='LaneId', index=[col], aggfunc='first')
        tables.append(t)
    lanes2 = pd.concat(tables).sort_index().interpolate(method='linear', limit_area='inside').fillna(0)

    # x
    x = (lanes2.index - lanes2.index.min()) / pd.Timedelta(1, 's')

    # ys
    ys = [lanes2[col].tolist() for col in lanes2.columns]

    plt.stackplot(x, *ys)


path = '/tmp/workspace'
def prepare_paper(path):
    with plt.style.context(['seaborn-paper', 'mypaper']):
        path = Path(path)/'card251'/'salus'
        df = load_case(path/'card251.output')
        fifo = load_trace_fifo(path/'trace.csv')

        # FIXME: refine doesn't work yet
        # refine_data = load_refine(path)

        sevts = load_serverevents(path)
        df = refine_time_events(df, sevts)

        fig, axs = plt.subplots(nrows=2, sharex=True)
        fig.set_size_inches(3.25, 2.35, forward=True)
        _, colors = plot_timeline(fifo, ax=axs[0], linewidth=2.5)
        axs[0].set_ylabel('FIFO')
        axs[0].legend().remove()
        axs[0].set_xlabel('')

        plot_timeline(df, ax=axs[1], linewidth=2.5, colors=colors)
        # plot_refine(axs[1], df, refine_data)
        axs[1].set_ylabel('Salus')
        fig.subplots_adjust(bottom=0.35)
        axs[1].legend(loc="upper center", frameon=False,
                      bbox_to_anchor=[0.5, -0.8],
                      #bbox_transform=fig.transFigure,
                      fontsize='x-small',
                      ncol=3
                      #mode='expand'
                      )

        fig.tight_layout()
        fig.savefig('/tmp/workspace/card251.pdf', dpi=300)
