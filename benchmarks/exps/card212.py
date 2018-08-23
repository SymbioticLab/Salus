# -*- coding: future_fstrings -*-
"""
Test the benchmark code for newly added workloads. See card#212

Scheduler: fair
Work conservation: True
Collected data: memory usage over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.runner import Executor
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, maybe_forced_preset, Pause


FLAGS = flags.FLAGS


def case1():
    scfg = maybe_forced_preset(presets.MostEfficient)

    # BUG: seems we must run a single job first otherwise it will hang
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/'case1'),
            WTL.create("vae", 128, 20, executor=Executor.TF),
            WTL.create("vae", 128, 20))


def case2():
    scfg = maybe_forced_preset(presets.MostEfficient)

    # BUG: seems we must run a single job first otherwise it will hang
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/'case2'),
            WTL.create("super_res", 128, 20, executor=Executor.TF),
            WTL.create("super_res", 128, 20))


def main(argv):
    command = argv[0] if argv else "case1"

    {
        "case1": case1,
        "case2": case2,
    }[command]()
