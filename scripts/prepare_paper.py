#! /bin/env python
from __future__ import print_function, absolute_import, division

import os
import sys

import parse_log as pl
import parse_nvvp as pn


def process_case(name, save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)

    log_dir = os.path.join('logs', name)

    # Memory usage
    logs = pl.load_file(os.path.join(log_dir, 'exec.output'))
    iter_times = pn.parse_iterations(os.path.join(log_dir, 'mem-iter.log'))
    df, fig = pl.memory_usage(logs, iter_times=iter_times)
    fig.savefig(os.path.join(save_dir, name + '-memory' + '.pdf'), dpi=600)
    with open(os.path.join(save_dir, name + '-memory.csv'), 'w') as f:
        df.to_csv(f)

    # Computation
    reader = pn.load_file(os.path.join(log_dir, 'profile.sqlite'))
    iter_times = pn.parse_iterations(os.path.join(log_dir, 'com-iter.log'))
    df, fig = pn.active_warp_trend(reader, iter_times)
    fig.savefig(os.path.join(save_dir, name + '-compute' + '.pdf'), dpi=600)
    with open(os.path.join(save_dir, name + '-compute.csv'), 'w') as f:
        df.to_csv(f)


if __name__ == '__main__':
    cases = ['vgg16-rpc-gpu', 'mnistconv-rpc-gpu', 'mnistlarge-rpc-gpu']
    if len(sys.argv) > 1:
        cases = sys.argv[1:]
    for name in cases:
        process_case(name)
