#! /bin/env python
from __future__ import print_function, absolute_import, division

import os
import re
import argparse
import pandas as pd
import numpy as np


try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport


ptn_iter = re.compile(r"""(?P<timestamp>.+): \s [sS]tep \s (\d+),\s
                          (loss|perplexity) .*;\s
                          (?P<duration>[\d.]+) \s sec/batch\)?""", re.VERBOSE)
ptn_first = re.compile(r'First iteration: (?P<duration>[\d.]+) sec/batch')
ptn_avg = re.compile(r'Average excluding first iteration: (?P<duration>[\d.]+) sec/batch')
ptn_jct = re.compile(r'JCT: (?P<duration>[\d.]+) s')


def read_iterfile(filepath):
    content = {
        'iters': []
    }
    with open(filepath) as f:
        for line in f:
            line = line.rstrip('\n')
            m = ptn_iter.match(line)
            if m:
                content['iters'].append(float(m.group('duration')))

            m = ptn_first.match(line)
            if m:
                content['first'] = float(m.group('duration'))

            m = ptn_avg.match(line)
            if m:
                content['avg'] = float(m.group('duration'))

            m = ptn_jct.match(line)
            if m:
                content['jct'] = float(m.group('duration'))

    return content


def handle_20iter(target, name, path):
    try:
        content = read_iterfile(path)
        data = {
            'Network': name,
            '20iter-jct': content['jct'],
            '20iter-first': content['first'],
            '20iter-avg': content['avg'],
        }
        target.append(data)
    except OSError as err:
        print('WARNING: file not found: ', err)


def handle_1min(target, name, path):
    try:
        content = read_iterfile(path)
        data = {
            'Network': name,
            '1min-jct': content['jct'],
            '1min-num': len(content['iters']),
            '1min-avg': content['avg'],
        }
        target.append(data)
    except OSError as err:
        print('WARNING: file not found: ', err)


def handle_5min(target, name, path):
    try:
        content = read_iterfile(path)
        data = {
            'Network': name,
            '5min-jct': content['jct'],
            '5min-num': len(content['iters']),
            '5min-avg': content['avg'],
        }
        target.append(data)
    except OSError as err:
        print('WARNING: file not found: ', err)


def handle_10min(target, name, path):
    try:
        content = read_iterfile(path)
        data = {
            'Network': name,
            '10min-jct': content['jct'],
            '10min-num': len(content['iters']),
            '10min-avg': content['avg'],
        }
        target.append(data)
    except OSError as err:
        print('WARNING: file not found: ', err)


def generate_csv(logs_dir, output_dir):
    baseline_data = []
    salus_data = []

    for name in os.listdir(logs_dir):
        path = os.path.join(logs_dir, name)
        if not os.path.isdir(path):
            continue

        if len(name.split('_')) == 2:
            # 20iter
            handle_20iter(baseline_data, name, os.path.join(path, 'gpu.output'))
            handle_20iter(salus_data, name, os.path.join(path, 'rpc.output'))
        elif len(name.split('_')) == 3:
            # more
            name, mode = name.rsplit('_', 1)
            handlers = {
                '1min': handle_1min,
                '5min': handle_5min,
                '10min': handle_10min,
            }
            if mode not in handlers:
                print('Unrecognized directory: ', name)
                continue
            handlers[mode](baseline_data, name, os.path.join(path, 'gpu.output'))
            handlers[mode](salus_data, name, os.path.join(path, 'rpc.output'))
        else:
            print('Unrecognized directory: ', name)

    def reducenan(x):
        return x[x.notnull()] if x.nunique() == 1 else None

    bldf = pd.DataFrame(baseline_data).groupby('Network').agg(reducenan).reset_index()
    sdf = pd.DataFrame(salus_data).groupby('Network').agg(reducenan).reset_index()

    def cleancols(df):
        colnames = ['Network', '20iter-jct', '20iter-first', '20iter-avg',
                    '1min-jct', '1min-num', '1min-avg',
                    '5min-jct', '5min-num', '5min-avg',
                    '10min-jct', '10min-num', '10min-avg']
        # add missing
        for col in colnames:
            if col not in df:
                df[col] = np.nan
        # reorder
        return df[colnames]

    bldf = cleancols(bldf)
    sdf = cleancols(sdf)

    Path(output_dir).mkdir(exist_ok=True)
    bldf.to_csv(os.path.join(output_dir, 'jct-baseline.csv'), index=False)
    sdf.to_csv(os.path.join(output_dir, 'jct-salus.csv'), index=False)


# Expected names:
# alexnet_25
# alexnet_50
# alexnet_100
# alexnet_25_1min
# alexnet_25_5min
# alexnet_25_10min
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='Directory contains jct logs', default='logs/osdi18/jct')
    parser.add_argument('--outputdir', help='Directory for output', default='.')
    config = parser.parse_args()

    generate_csv(config.logdir, config.outputdir)
