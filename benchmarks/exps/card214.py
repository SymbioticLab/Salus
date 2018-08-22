# -*- coding: future_fstrings -*-
"""
Determine the best method used for GPU sync. See card#214

Case 1 uses BaseGPUDevice::Sync

Case 2 uses sync main compute

Case 3 uses sync all four

Plot allocation and compute timeline, with iteration data

Scheduler: fair
Work conservation: True
Collected data: memory usage over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, maybe_forced_preset, Pause


FLAGS = flags.FLAGS


def case1():
    scfg = maybe_forced_preset(presets.MostEfficient)

    # BUG: seems we must run a single job first otherwise it will hang
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/'case1'),
            WTL.create("inception3", 100, 20),
            Pause.Wait,
            WTL.create("resnet50", 50, 20),
            Pause.Wait,
            WTL.create("alexnet", 25, 20),
            Pause.Wait,
            WTL.create("alexnet", 50, 100),
            WTL.create("alexnet", 50, 100))


def case2():
    scfg = maybe_forced_preset(presets.MostEfficient)

    # BUG: seems we must run a single job first otherwise it will hang
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/'case2'),
            WTL.create("inception3", 100, 20),
            Pause.Wait,
            WTL.create("resnet50", 50, 20),
            Pause.Wait,
            WTL.create("alexnet", 25, 20),
            Pause.Wait,
            WTL.create("alexnet", 50, 100),
            WTL.create("alexnet", 50, 100))


def case3():
    scfg = maybe_forced_preset(presets.MostEfficient)

    # BUG: seems we must run a single job first otherwise it will hang
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/'case3'),
            WTL.create("inception3", 100, 20),
            Pause.Wait,
            WTL.create("resnet50", 50, 20),
            Pause.Wait,
            WTL.create("alexnet", 25, 20),
            Pause.Wait,
            WTL.create("alexnet", 50, 100),
            WTL.create("alexnet", 50, 100))


def main(argv):
    command = argv[0] if argv else "case1"

    {
        "case1": case1,
        "case2": case2,
        "case3": case3,
    }[command]()
