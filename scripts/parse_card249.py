#!/usr/bin/env python3
#
# Copyright 2019 Peifeng Yu <peifeng@umich.edu>
# 
# This file is part of Salus
# (see https://github.com/SymbioticLab/Salus).
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
import tempfile
import subprocess as sp

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


# 2018-09-01 06:22:21.180029: Step 341, loss=3.60 (33.8 examples/sec; 0.740 sec/batch)
ptn_iter = re.compile(r"""(?P<timestamp>.+): \s [sS]tep \s (?P<Step>\d+),\s
                          (loss|perplexity) .* \(
                          (?P<Speed>[\d.]+) \s examples/sec; \s
                          (?P<Duration>[\d.]+) \s sec/batch\)?""", re.VERBOSE)

def parse_iterations(path):
    path = Path(path)
    iterations = []
    with path.open() as f:
        for line in f:
            line = line.rstrip('\n')

            m = ptn_iter.match(line)
            if m:
                iterations.append(m.groupdict())
    assert len(iterations) > 0
    fake = {}
    fake.update(iterations[-1])
    fake['Speed'] = 0
    fake['timestamp'] = (pd.to_datetime(fake['timestamp']) + pd.Timedelta(1, 'us')).strftime('%Y-%m-%d %H:%M:%S.%f')
    iterations.append(fake)

    fake = {}
    fake.update(iterations[0])
    fake['Speed'] = 0
    fake['timestamp'] = (pd.to_datetime(fake['timestamp']) - pd.Timedelta(1, 'us')).strftime('%Y-%m-%d %H:%M:%S.%f')
    iterations[:0] = [fake]

    df = pd.DataFrame(iterations)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['Speed'] = pd.to_numeric(df['Speed'])
    df['Step'] = pd.to_numeric(df.Step)
    df['Duration'] = pd.to_numeric(df.Duration)

    df = df.sort_values('timestamp')
    # calculate a cumulative speed
    # get batch size
    batch_size = int(path.name.partition('.')[0].partition('_')[-1])

    cumspeed = []
    start = df['timestamp'].iloc[0] - pd.Timedelta(df.Duration.iloc[0], 's')
    for idx, row in df.iterrows():
        images = batch_size * (idx + 1)
        dur = (row['timestamp'] - start) / pd.Timedelta(1, 's')
        cumspeed.append(images/dur)
    df['CumSpeed'] = cumspeed
    return df


def load_speeds(path, key='Speed'):
    path = Path(path)

    speeds = {}
    for f in path.glob('*.*.*.*.output'):
        model, executor, iterstr, runid, _ = f.name.split('.')
        s = parse_iterations(f)

        speeds['{}.{}'.format(model, runid)] = s.set_index('timestamp')[key]
    return pd.DataFrame(speeds)


def load_refine(pathdir):
    # load preempt select events
    df = cm.load_generic(pathdir/'server.output', event_filters=['preempt_select_sess'])
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


def plot_speeds(df, **kwargs):
    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()

    df = df.interpolate(method='time', limit_area='inside')

    #df = df.resample('500ms').mean()

    df = df.reset_index()
    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / pd.Timedelta(1, 's')
    df = df.set_index('timestamp')
    ax = df.plot(ax=ax, **kwargs)

    ax.legend().remove()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Images per second')
    return ax


def plot_timeline(df, colors=None, **kwargs):
    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()

    # sort df by no
    df = df.sort_values(by='No')

    offset = df.Queued.min()
    qmin = (df.Queued - offset) / pd.Timedelta(1, unit='s')
    xmin = (df.Started - offset) / pd.Timedelta(1, unit='s')
    xmax = (df.Finished - offset) / pd.Timedelta(1, unit='s')

    if colors is None:
        color_cycle = ax._get_lines.prop_cycler
        colors = [next(color_cycle)['color'] for _ in qmin]

    for no, q, left, right, color, name in zip(df['No'], qmin, xmin, xmax, colors, df.Model):

        # queuing time
        ax.barh(no, left - q, 0.8, q, color='#b6b6b6')
        # run time
        bar = ax.barh(no, right - left, 0.8, left,
                      color=color,
                      label='#{3}: {0}'.format(*name.split('.')))

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
    magic = refine_data.iterrows()
    next(magic)
    for (_, left), (_, right) in zip(refine_data.iterrows(), magic):
        for no in df.index.unique():
            if no == left.No:
                continue
            # make sure left and right within job no's started and finished
            l = max(df.loc[no].Started, left.Ntime)
            r = min(df.loc[no].Finished, right.Ntime)
            # print(f'no = {no}, l={l}, r={r}, left = {left.Ntime}, right={right.Ntime}, Started={df.loc[no].Started}, Finished={df.loc[no].Finished}')
            if l >= r:
                continue
            # use queuing color if we begins from queuing segment
            if l <= df.loc[no].Started:
                width = 0.8
                color = '#b6b6b6'
                edgecolor = None
                # update df's Started to r
                df.loc[no, 'Started'] = r
            else:
                width = 0.85
                color = '#ffffff'
                edgecolor = '#ffffff'
            # mask from left to right
            bars.append(ax.barh(no, r - l, width, l, color=color, edgecolor=edgecolor))

    return bars


path = '/tmp/workspace'
def prepare_paper(path):
    with plt.style.context(['seaborn-paper', 'mypaper']):
        path = Path(path)/'card249'/'salus'
        df = load_case(path/'card249.output')
        trace = load_trace(path/'trace.csv')
        refine_data = load_refine(path)

        sevts = load_serverevents(path)
        df = refine_time_events(df, sevts)

        fig, axs = plt.subplots(nrows=2, sharex=True)
        fig.set_size_inches(6.5, 2, forward=True)
        _, colors = plot_timeline(trace, ax=axs[0])
        axs[0].set_ylabel('FIFO')
        axs[0].legend().remove()
        axs[0].set_xlabel('')

        plot_timeline(df, ax=axs[1], colors=colors)
        plot_refine(axs[1], df, refine_data)
        axs[1].set_ylabel('Salus')
        fig.subplots_adjust(bottom=0.35, hspace=0.05)
        axs[1].legend(loc="upper center", frameon=False,
                      bbox_to_anchor=[0.5, -0.96],
                      #bbox_transform=fig.transFigure,
                      fontsize='x-small',
                      ncol=5
                      #mode='expand'
                      )
        fig.tight_layout()

        fig.savefig('/tmp/workspace/card249.pdf', dpi=300)


if __name__ == '__main__':
    prepare_paper(path)
