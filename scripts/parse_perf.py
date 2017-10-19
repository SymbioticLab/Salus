from __future__ import print_function, absolute_import, division

import re
from datetime import timedelta

import pandas as pd

ptn_log = re.compile(r"""^\[(?P<timestamp>\d+-\d+-\d+\s\d+:\d+:\d+\.\d{6}) (\d{3})? \]\s
                           \[(?P<thread>\d+)\]\s
                           \[(?P<loc>\w+)\]\s
                           \[(?P<level>\w+)\]\s
                           (?P<content>.*)$""",
                     re.VERBOSE)

ptn_exec = re.compile(r"""Executed \[(?P<name>.+)\] in \[(?P<count>\d+) (?P<unit>\w+)\]""")
ptn_check = re.compile(r"""Performance\scheckpoint \[(?P<name>.+)\]\s
                           for\sblock\s\[(?P<parent>.+)\]\s:\s
                           \[(?P<count>\d+)\s(?P<unit>\w+)\]""",
                       re.VERBOSE)
ptn_mu = re.compile(r"""Mutex \s (?P<name>\w+)@(?P<inst>[\dxa-fA-F]+)
                        .+acquiring \s (?P<acq>\d+)us
                        .+locking \s (?P<lck>\d+)us""",
                    re.VERBOSE)
ptn_overhead = re.compile(r'''OpItem .+\bname=(?P<name>[^,]+),
                              .+\bsession=(?P<sess>\w+),
                              .+\bfailures=(?P<failures>\d+)
                              .+\bqueuing \s time \s (?P<queuing>[\d.]+)
                              .+\bpreparation \s time \s (?P<preptime>[\d.]+)
                              .+\brunning \s time \s (?P<running>[\d.]+).+''',
                          re.VERBOSE)


def initialize():
    pass


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
                log = match_exec_content(m.group('content'), m.groupdict())
                if log:
                    logs.append(log)
            else:
                print('Unhandled line: ' + line)

    return logs


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
    
    m = ptn_overhead.match(content)
    if m:
        return {
            'type': 'overhead',
            'op': m.group('name'),
            'sess': m.group('sess'),
            'failures': int(m.group('failures')),
            'queuing': float(m.group('queuing')),
            'preptime': float(m.group('preptime')),
            'running': float(m.group('running')),
        }

    return None


def perfcalls(logs):
    df = pd.DataFrame(logs)

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


def overhead_breakdown(logs):
    data = [l for l in logs if l['type'] == 'overhead']
    df = pd.DataFrame(data).drop('type', axis=1)

    operations = ['sum', 'mean', 'min', 'median', 'max']
    grouped = df.groupby('sess').agg({
        'failures': operations,
        'queuing': operations,
        'preptime': operations,
        'running': operations
    })
    grouped.columns = ['_'.join(x) for x in grouped.columns.ravel()]
    return grouped, df
