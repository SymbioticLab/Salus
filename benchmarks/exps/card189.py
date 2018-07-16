# -*- coding: future_fstrings -*-
"""
For debugging. Reproduces memory outage detailed in card#189.
Initially found by running exp15 with 50 jobs.

Even when run a single job. There are many OOM errors happening

Scheduler: fair
Work conservation: True
Collected data: memory usage over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.runner import Executor
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd, maybe_forced_preset, Pause, run_tf


FLAGS = flags.FLAGS


def case1(scfg):
    """Run one inception3_100 for a few iterations and collect memory map from DBFC"""
    run_seq(scfg.copy(output_dir=FLAGS.save_dir),
            WTL.create("inception3", 100, 100))


def case2(scfg):
    run_tf(scfg.copy(output_dir=FLAGS.save_dir),
           WTL.create("inception3", 100, 10, executor=Executor.TF))


def tfop(_):
    wl = WTL.create("inception3", 100, 10, executor=Executor.TF)
    wl.env['TF_CPP_MIN_VLOG_LEVEL'] = '2'
    wl.env['TF_CPP_MIN_LOG_LEVEL'] = ''
    run_tf(FLAGS.save_dir / 'tfop', wl)


def test(scfg):
    run_seq(scfg.copy(output_dir=FLAGS.save_dir),
            WTL.create("inception3", 100, 165),
            WTL.create("resnet50", 50, 798),
            WTL.create("resnet152", 75, 19),
            Pause.Wait,
            WTL.create("inception3", 100, 165),
            WTL.create("resnet50", 50, 798),
            WTL.create("resnet152", 75, 19))


def main(argv):
    scfg = maybe_forced_preset(presets.Debugging)

    command = argv[0] if argv else "test"

    {
        "case1": case1,
        "case2": case2,
        "tfop": tfop,
        "test": test,
    }[command](scfg)
