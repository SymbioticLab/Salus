# -*- coding: future_fstrings -*-
"""
Collect data for optracing

Using alexnet_25 for 1 iteration

Scheduler: pack
Work conservation: True
Collected data: JCT
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from itertools import chain

from absl import flags
from typing import Iterable, Union

from benchmarks.driver.server.config import presets
from benchmarks.driver.utils import unique
from benchmarks.driver.workload import WTL, Executor
from benchmarks.exps import run_seq, maybe_forced_preset, run_tf, Pause


FLAGS = flags.FLAGS
TBatchSize = Union[str, int]

def main(argv):
    name, bs, bn = 'vgg11', 25, 10
    if len(argv) > 0:
        name = argv[0]
    if len(argv) > 1:
        bs = int(argv[1])
    if len(argv) > 2:
        bn = int(argv[2])

    def create_wl(ex):
        return WTL.create(name, bs, bn, executor=ex)

    wl = create_wl(Executor.TF)
    wl.env['TF_CPP_MIN_VLOG_LEVEL'] = '1'
    wl.env['TF_CPP_MIN_LOG_LEVEL'] = ''
    run_tf(FLAGS.save_dir / "tf", wl)
