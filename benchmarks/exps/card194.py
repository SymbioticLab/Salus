# -*- coding: future_fstrings -*-
"""
For debugging. Salus uses more memory than TF, as detailed in card#194.

Even when run a single job. Thus there are more failed alloc and retires.
Possible reason is the different order of tasks

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
    """Run one alexnet_100 for a few iterations and collect memory map from DBFC"""
    run_seq(scfg.copy(output_dir=FLAGS.save_dir),
            WTL.create("alexnet", 100, 10))


def case2(scfg):
    run_tf(scfg.copy(output_dir=FLAGS.save_dir),
           WTL.create("alexnet", 100, 10, executor=Executor.TF))


def tfop(_):
    wl = WTL.create("alexnet", 100, 10, executor=Executor.TF)
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
    scfg = maybe_forced_preset(presets.AllocProf)
    scfg.logconf = "memop"

    command = argv[0] if argv else "test"

    {
        "case1": case1,
        "case2": case2,
        "tfop": tfop,
        "test": test,
    }[command](scfg)
