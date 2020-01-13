# -*- coding: future_fstrings -*-
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

