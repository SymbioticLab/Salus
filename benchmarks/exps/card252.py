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
           WTL.create("inception3eval", 1, 200, executor=Executor.TF))

    run_seq(scfg.copy(output_dir=FLAGS.save_dir/'case1'),
            WTL.create("inception3eval", 1, 2000))


def case2():
    scfg = maybe_forced_preset(presets.MostEfficient)
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/'case2'),
            WTL.create("inception3eval", 1, 200),
            WTL.create("inception3eval", 1, 200),
            WTL.create("inception3eval", 1, 200),
            )


def main(argv):
    command = argv[0] if argv else "test"

    {
        "case1": case1,
        "case2": case2,
    }[command]()
