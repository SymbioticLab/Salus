from __future__ import print_function, absolute_import, division

import re
from datetime import timedelta
import subprocess as sp
import os
import shutil

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dt

import plotutils as pu


ptn_log = re.compile(r"""^\[(?P<timestamp>\d+-\d+-\d+\s\d+:\d+:\d+\.\d{6}) (\d{3})? \]\s
                           \[(?P<thread>\d+)\]\s
                           \[(?P<loc>\w+)\]\s
                           \[(?P<level>\w+)\]\s
                           (?P<content>.*)$""",
                     re.VERBOSE)

ptn_exec = re.compile(r"""Executed \[(?P<name>.+)\] in \[(?P<count>\d+) (?P<unit>\w+)\]""")
ptn_check = re.compile(r"""Performance \s checkpoint \s \[(?P<name>.+)\]\s
                           for\sblock\s\[(?P<parent>.+)\]\s:\s
                           \[(?P<count>\d+)\s(?P<unit>\w+)\]""",
                       re.VERBOSE)
ptn_mu = re.compile(r"""Mutex \s (?P<name>\w+)@(?P<inst>[\dxa-fA-F]+)
                        .+acquiring \s (?P<acq>\d+)us
                        .+locking \s (?P<lck>\d+)us""",
                    re.VERBOSE)

# OpItem Stat ExecTask(name=_SOURCE, type=NoOp, session=246b928f83c1ecf1, failures=0, inputsize=0) memusage: 1
# queued: 2017-10-24 14:50:08.625489191 scheduled: 2017-10-24 14:50:08.625684743 finished: 2017-10-24 14:50:08.628265822
ptn_opstat = re.compile(r'''OpItem \s Stat .+\bname=(?P<name>[^,]+),
                            .+\btype=(?P<type>[^,]+),
                            .+\bsession=(?P<sess>\w+),
                            .+\bfailures=(?P<failures>\d+),
                            .+\binputsize=(?P<inputsize>\d+)
                            .+\bmemusage:\s (?P<memusage>\d+)
                            .+\bqueued:\s (?P<queued>.+)
                            .+\bscheduled:\s (?P<scheduled>.+)
                            .+\bfinished:\s (?P<finished>.+)''',
                        re.VERBOSE)

# Sched iter 0 session: 246b928f83c1ecf1 pending: 0 scheduled: 2 counter: 0
ptn_sess_iter = re.compile(r'''Sched \s iter \s (?P<iter>\d+) \s
                               \bsession:\s (?P<sess>\w+) \s
                               \bpending:\s (?P<pending>\d+) \s
                               \bscheduled:\s (?P<scheduled>\d+) \s
                               \bcounter:\s (?P<counter>\d+)''',
                           re.VERBOSE)

# Scheduler iter stat: 0 running: 3 noPageRunning: 3
ptn_sched_iter = re.compile(r'''Scheduler \s iter \s stat: \s (?P<iter>\d+) \s
                                running:\s (?P<running>\d+) \s
                                noPageRunning:\s (?P<noPageRunning>\d+)''',
                            re.VERBOSE)

# Paging:  duration: 19995 us released: 160563712 forceevict: ''
ptn_paging = re.compile(r'''Paging: \s+ duration: \s (?P<duration>\d+) \s us
                            \s released: \s (?P<released>\d+)
                            \s forceevict: \s '(?P<forceevicted>\w*)' ''',
                        re.VERBOSE)


def initialize():
    pass


def preprocesse(d):
    perffile = os.path.join(d, 'perf.output')
    tempdir = sp.check_output(['mktemp', '-d', '--tmpdir']).rstrip('\n')
    try:
        path = os.path.join(tempdir, 'sessiter.output')
        with open(path, 'w') as f:
            sp.check_call(['grep', 'Sched iter', perffile], stdout=f)
        shutil.move(path, d)

        path = os.path.join(tempdir, 'schediter.output')
        with open(path, 'w') as f:
            sp.check_call(['grep', 'Scheduler iter', perffile], stdout=f)
        shutil.move(path, d)

        path = os.path.join(tempdir, 'opstat.output')
        with open(path, 'w') as f:
            sp.check_call(['grep', 'OpItem Stat', perffile], stdout=f)
        shutil.move(path, d)

        path = os.path.join(tempdir, 'paging.output')
        with open(path, 'w') as f:
            sp.check_call(['grep', 'Paging', perffile], stdout=f)
        shutil.move(path, d)
    finally:
        sp.call(['rm', '-r', '-f', tempdir])


def load_file(path, reinitialize=True):
    """Load logs"""
    logs = []

    if reinitialize:
        initialize()

    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')

            m = ptn_log.match(line)
            if m:
                ctx = m.groupdict()
                content = ctx['content']
                del ctx['content']
                log = match_exec_content(content, ctx)
                if log:
                    ctx.update(log)
                    logs.append(ctx)
            else:
                print('Unhandled line: ' + line)

    df = pd.DataFrame(logs)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df


def match_exec_content(content, ctx):
    m = ptn_exec.match(content)
    if m:
        return {
            'name': m.group('name'),
            'time': timedelta(microseconds=int(m.group('count'))),
            'parent': None,
            'type': 'func'
        }

    m = ptn_check.match(content)
    if m:
        return {
            'name': m.group('name'),
            'time': timedelta(microseconds=int(m.group('count'))),
            'parent': m.group('parent'),
            'type': 'func'
        }

    m = ptn_mu.match(content)
    if m:
        return {
            'name': m.group('name'),
            'acquire': timedelta(microseconds=int(m.group('acq'))),
            'locked': timedelta(microseconds=int(m.group('lck'))),
            'type': 'mutex'
        }

    m = ptn_opstat.match(content)
    if m:
        d = m.groupdict()
        d['type'] = 'opstat'
        return d

    m = ptn_sess_iter.match(content)
    if m:
        d = m.groupdict()
        d['type'] = 'sess-iter'
        return d

    m = ptn_sched_iter.match(content)
    if m:
        d = m.groupdict()
        d['type'] = 'sched-iter'
        return d

    m = ptn_paging.match(content)
    if m:
        d = m.groupdict()
        d['type'] = 'paging'
        return d

    return None


def perfcalls(df):
    # Change from timedelta to us
    df['time'] = df['time'].astype(int) / 1e3

    func = df[df['type'] == 'func'].drop(['acquire', 'locked', 'type'], axis=1)

    # for func
    grouped = func.groupby('name').agg({
        'time': ['sum', 'mean', 'std', 'min', 'median', 'max']
    })
    # Using ravel, and a string join, we can create better names for the columns:
    grouped.columns = [x[-1] for x in grouped.columns.ravel()]
    grouped.sort_values(by=['sum', 'mean'], ascending=False)

    return grouped, func


def overhead_breakdown(df):
    df = df[df['type'] == 'opstat'].drop('type', axis=1)
    for col in ['queued', 'scheduled', 'finished']:
        df[col] = pd.to_datetime(df[col])
    df['queuing'] = df['scheduled'] - df['queued']
    df['running'] = df['finished'] - df['scheduled']
    df['failures'] = pd.to_numeric(df['failures'])

    operations = ['sum', 'mean', 'min', 'median', 'max']
    grouped = df.groupby('sess').agg({
        'failures': operations,
        'queuing': operations,
        'running': operations
    })
    grouped.columns = ['_'.join(x) for x in grouped.columns.ravel()]
    return grouped, df


def session_counters(df, colnames=None, beginning=None, useFirstRowAsBegining=True, ax=None):
    df = df[df['type'] == 'sess-iter'].drop('type', axis=1)
    for col in ['pending', 'scheduled', 'counter']:
        df[col] = pd.to_numeric(df[col])

    if useFirstRowAsBegining and beginning is None:
        beginning = df.index[0]
    useTimedelta = beginning is not None
    if colnames is None:
        colnames = ['counter']

    if ax is None:
        fig, axs = plt.subplots(nrows=len(colnames), sharex=True, squeeze=False)
        axs = axs.flatten()

        for col, x in zip(colnames, axs):
            sessionsCounter = {}
            for k, gg in df.groupby('sess'):
                sessionsCounter[k] = gg[col]
            ss = pd.DataFrame(sessionsCounter)

            if False:
                # ss = ss.resample('500us').fillna(method='ffill').fillna(0)
                if useTimedelta:
                    ss.index = ss.index - beginning
                    ss.index = ss.index.astype(int)
                # import ipdb; ipdb.set_trace()
                ss.plot.area(ax=x, linewidth=0)
            else:
                for k, s in sessionsCounter.items():
                    if useTimedelta:
                        s.index = s.index - beginning
                        s.index = s.index.astype(int)
                    sz = s[s > 0]
                    if len(sz) > 0:
                        sz.plot(ax=x, kind='line', label=k)
                    else:
                        s.plot(ax=x, kind='line', label=k)
            x.set_title(col)
    else:
        col = colnames[0]
        sessionsCounter = {}
        for k, gg in df.groupby('sess'):
            sessionsCounter[k] = gg[col]
        ss = pd.DataFrame(sessionsCounter)

        if False:
            # ss = ss.resample('500us').fillna(method='ffill').fillna(0)
            if useTimedelta:
                ss.index = ss.index - beginning
                ss.index = ss.index.astype(int)
            # import ipdb; ipdb.set_trace()
            ss.plot.area(ax=ax, linewidth=0)
        else:
            for k, s in sessionsCounter.items():
                if useTimedelta:
                    s.index = s.index - beginning
                    s.index = s.index.astype(int)
                # import ipdb; ipdb.set_trace()
                sz = s[s > 0]
                if len(sz) > 0:
                    sz.plot(ax=ax, kind='line', label=k)
                else:
                    s.plot(ax=ax, kind='line', label=k)
        ax.set_title(col)

    ax = ax if ax is not None else plt.gca()
    if useTimedelta:
        pu.cleanup_axis_timedelta(ax.xaxis)
    else:
        pu.cleanup_axis_datetime(ax.xaxis)

    return df, ax.figure


def paging_stat(df, useFirstRowAsBegining=True, beginning=None):
    if useFirstRowAsBegining and beginning is None:
        beginning = df.index[0]

    df = df[df['type'] == 'paging'].drop('type', axis=1)
    for col in ['duration', 'released']:
        df[col] = pd.to_numeric(df[col])

    df['duration'] = pd.to_timedelta(df['duration'], unit='us')
    df['starttime'] = df.index - df['duration']
    df['endtime'] = df.index
    # convert to reltime
    # df.start = df.start - beginning
    # df.end = df.end - beginning

    ax = plt.hlines(df.index, dt.date2num(df['starttime']), dt.date2num(df['endtime']))

    return df, ax.figure
