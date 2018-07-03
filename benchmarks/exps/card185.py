# -*- coding: future_fstrings -*-
"""
For debugging. Reproduces memory outage detailed in card#185.
Initially found by running exp15 with 50 jobs.

Scheduler: fair
Work conservation: True
Collected data: memory usage over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd, maybe_forced_preset, Pause


FLAGS = flags.FLAGS


def case1(scfg):
    run_seq(scfg.copy(output_dir=FLAGS.save_dir),
            WTL.create("inception3", 100, 165),
            WTL.create("resnet50", 50, 798),
            WTL.create("resnet152", 75, 19),
            Pause.Wait,
            Pause.Manual,
            WTL.create("inception3", 100, 165),
            WTL.create("resnet50", 50, 798),
            WTL.create("resnet152", 75, 19))


def case2(scfg):
    run_seq(scfg.copy(output_dir=FLAGS.save_dir),
            WTL.create("resnet50", 50, 798),
            Pause.Wait,
            Pause.Manual,
            WTL.create("resnet50", 50, 798))


def main(argv):
    scfg = maybe_forced_preset(presets.AllocProf)
    if argv:
        run_seq(scfg.copy(output_dir=FLAGS.save_dir),
                *parse_actions_from_cmd(argv))
        return

    # case1(scfg)
    case2(scfg)
