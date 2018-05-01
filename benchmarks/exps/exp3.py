# -*- coding: future_fstrings -*-
"""
Collect data for kernel distribution

Run all workloads for 1 iteration

Scheduler: pack
Work conservation: True
Collected data: optracing
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from collections import Iterable

from absl import flags
from typing import Union

from benchmarks.driver.utils import unique
from benchmarks.driver.workload import WTL, Executor
from benchmarks.exps import run_seq, run_tf


FLAGS = flags.FLAGS
TBatchSize = Union[str, int]


def select_workloads(argv):
    # type: (Iterable[str]) -> Iterable[(str, TBatchSize)]
    """Select workloads based on commandline

        Example: alexnet,vgg11
    """
    if not argv:
        names = WTL.known_workloads.keys()
    else:
        names = unique((
            name
            for piece in argv
            for name in piece.split(',')
        ), stable=True)

    def getbs(name):
        if '_' in name:
            name, bs = name.split('_')
            bs = int(bs)
            return [(name, bs)]
        else:
            bss = WTL.from_name(name).available_batch_sizes()
            names = [name] * len(bss)
            return zip(names, bss)
    return [WTL.create(n, batch_size, 1, executor=Executor.TF)
            for name in names
            for n, batch_size in getbs(name)]


def main(argv):
    wls = select_workloads(argv)

    for wl in wls:
        if wl.name in ['speech', 'mnistsf', 'mnistcv', 'mnistlg', 'seq2seq']:
            continue

        if (FLAGS.save_dir / wl.canonical_name).exists():
            continue

        wl.env['TF_CPP_MIN_VLOG_LEVEL'] = '1'
        wl.env['TF_CPP_MIN_LOG_LEVEL'] = ''
        run_tf(FLAGS.save_dir / wl.canonical_name, wl)

