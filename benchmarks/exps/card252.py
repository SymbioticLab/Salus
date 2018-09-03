# -*- coding: future_fstrings -*-
"""
Check inference workloads. See card#241

LaneMgr: enabled
Scheduler: fair
Work conservation: True
Collected data: memory usage over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

import random
from absl import flags

from benchmarks.driver.runner import Executor
from benchmarks.driver.server.config import presets
from benchmarks.driver.utils import execute
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, maybe_forced_preset, Pause, run_tf, run_tfdist

from benchmarks.driver.utils.compatiblity import pathlib


FLAGS = flags.FLAGS


def case1():
    scfg = maybe_forced_preset(presets.MostEfficient)

    run_tf(FLAGS.save_dir/'case1',
           WTL.create("inception3eval", 1, 1000, executor=Executor.TF))

    run_seq(scfg.copy(output_dir=FLAGS.save_dir/'case1'),
            WTL.create("inception3eval", 1, 1000))


def case2():
    scfg = maybe_forced_preset(presets.MostEfficient)
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/'case2'),
            WTL.create("inception3eval", 1, 1000),
            WTL.create("inception3eval", 1, 1000),
            WTL.create("inception3eval", 1, 1000),
            WTL.create("inception3eval", 1, 1000),
            WTL.create("inception3eval", 1, 1000),
            WTL.create("inception3eval", 1, 1000),
            WTL.create("inception3eval", 1, 1000),
            WTL.create("inception3eval", 1, 1000),
            WTL.create("inception3eval", 1, 1000),
            WTL.create("inception3eval", 1, 1000),
            )


def case3():
    run_tfdist(FLAGS.save_dir/'case3', WTL.create("inception3eval", 1, 1000, executor=Executor.TFDist))


def case4():
    scfg = maybe_forced_preset(presets.MostEfficient)

    for model in ['inception3eval', 'vgg19eval']:
        for i in [1, 2, 4, 8]:
            run_seq(scfg.copy(output_dir=FLAGS.save_dir/'case4'/f'{model}-{i}'),
                    *[WTL.create("inception3eval", 1, 1000) for _ in range(i)]
                    )


def main(argv):
    command = argv[0] if argv else "test"

    {
        "case1": case1,
        "case2": case2,
        "case3": case3,
        "case4": case4,
    }[command]()
