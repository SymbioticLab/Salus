#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:42:25 2018

Collection of methods for 100 job trace JCT processing

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
import itertools

import pandas as pd
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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

    # for convinent
    wls['No'] = pd.to_numeric(wls['Model'].str.rpartition('.')[2])

    return wls


def load_trace(path, fifo=True):
    df = pd.read_csv(path)
    df = df.sort_values(by='submit_time')

    if fifo:
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
    else:
        data = [
            {
                "Model": f"{row.model_name}.tf.{row.iterations}iter.{row.job_id}",
                "Finished": row.submit_time + row.duration,
                "Queued": row.submit_time,
                "Started": row.submit_time,
                "queuing": 0,
                "JCT": row.duration
            }
            for idx, row in df.iterrows()
        ]
        df = pd.DataFrame(data)

    for col in ['Finished', 'Queued', 'Started', 'queuing', 'JCT']:
            df[col] = pd.to_timedelta(df[col], unit='s')

    df['No'] = pd.to_numeric(df['Model'].str.rpartition('.')[2])
    return df


def load_refine(pathdir, evt='preempt_select_sess'):
    # load preempt select events
    with tempfile.NamedTemporaryFile() as f:
        server_output = pathdir/'server.output'
        sp.check_call(['grep', evt, str(server_output)], stdout=f)
        f.flush()
        df = cm.load_generic(f.name, event_filters=[evt])
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
        try:
            sp.check_call(['grep', 'lane_assigned', str(server_output)], stdout=f)
        except sp.CalledProcessError:
            raise ValueError('No server event found')
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


def update_start_time_using_refine(df, refine_data, offset=None):
    # so that we can access job using no
    df = df.set_index('No')
    refine_data = refine_data.copy()

    # for every preempt event pair, mask jobs that's not the left event's switch_to job
    if offset is None:
        offset = df.Queued.min()
    refine_data['Ntime'] = (refine_data['timestamp'] - offset) / pd.Timedelta(1, unit='s')

    # also convert df.Queued to relative time
    df['Started'] = (df.Started - offset) / pd.Timedelta(1, unit='s')
    df['Finished'] = (df.Finished - offset) / pd.Timedelta(1, unit='s')

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
                # make sure left and right within job no's started and finished
                l = max(df.loc[no].Started, left.Ntime)
                r = min(df.loc[no].Finished, right.Ntime)
                if l >= r:
                    continue
                # use queuing color if we begins from queuing segment
                if l <= df.loc[no].Started:
                    # update df's Started to r
                    df.loc[no, 'Started'] = r

    return df.reset_index()


def plot_timeline(df, props=None, returnSessProps=False, offset=None,
                  new_jnos=None, plot_offset=None, **kwargs):
    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()

    # sort df by no
    df['No'] = pd.to_numeric(df['Model'].str.rpartition('.')[2])
    df = df.sort_values(by='No')

    if offset is None:
        offset = df.Queued.min()
    qmin = (df.Queued - offset) / pd.Timedelta(1, unit='s')
    xmin = (df.Started - offset) / pd.Timedelta(1, unit='s')
    xmax = (df.Finished - offset) / pd.Timedelta(1, unit='s')

    if props is None:
        color_cycle = ax._get_lines.prop_cycler
        props = [next(color_cycle) for _ in qmin]
    else:
        props = itertools.cycle(props)

    sessColors = {}
    if plot_offset is None:
        plot_offset = 0
    for (_, row), q, left, right, prop in zip(df.iterrows(), qmin, xmin, xmax, props):
        barheight = 0.8
        no = row.No if new_jnos is None else new_jnos[row.No]
        # queuing time
        ax.barh(no, left - q, barheight, q + plot_offset, color='#b6b6b6')
        # run time
        bar = ax.barh(no, right - left, barheight, left + plot_offset,
                      label='#{3}: {0}'.format(*row.Model.split('.')), **prop)
        if 'LaneId' in row:
            ax.text(right + 2 + plot_offset, no, f'Lane {row.LaneId}',
                    ha='left', va='center', fontsize=3)
        if returnSessProps:
            sessColors[row.Sess] = prop

    # ax.legend()
    ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Workload')
    ax.yaxis.set_ticks([])

    if returnSessProps:
        return bar, props, sessColors
    else:
        return bar, props


def plot_refine(ax, df, refine_data, offset=None, new_jnos=None, plot_offset=None):
    # so that we can access job using no
    df = df.set_index('No')

    # for every preempt event pair, mask jobs that's not the left event's switch_to job
    if offset is None:
        offset = df.Queued.min()
    refine_data['Ntime'] = (refine_data['timestamp'] - offset) / pd.Timedelta(1, unit='s')

    # also convert df.Queued to relative time
    df['Started'] = (df.Started - offset) / pd.Timedelta(1, unit='s')
    df['Finished'] = (df.Finished - offset) / pd.Timedelta(1, unit='s')

    bars = []
    if plot_offset is None:
        plot_offset = 0
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
                # make sure left and right within job no's started and finished
                l = max(df.loc[no].Started, left.Ntime)
                r = min(df.loc[no].Finished, right.Ntime)
                if l >= r:
                    continue
                # use queuing color if we begins from queuing segment
                if l <= df.loc[no].Started:
                    width = 0.8
                    color = '#b6b6b6'
                    edgecolor = '#b6b6b6'
                    # update df's Started to r
                    df.loc[no, 'Started'] = r
                else:
                    width = 0.85
                    color = '#ffffff'
                    edgecolor = '#ffffff'
                # mask from left to right
                plot_no = no if new_jnos is None else new_jnos[no]
                bars.append(ax.barh(plot_no, r - l, width, l + plot_offset, color=color, edgecolor=edgecolor))
    return bars

