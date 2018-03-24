# -*- coding: future_fstrings -*-
"""
OSDI Experiment 11

Run 2 jobs together.

Scheduler: preempt
Work conservation: True
Collected data: Numer of scheduled tasks over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd, Pause, maybe_forced_preset

FLAGS = flags.FLAGS


def main(argv):
    scfg = maybe_forced_preset(presets.Profiling)
    scfg.scheduler = 'preempt'
    scfg.disable_wc = False

    if argv:
        run_seq(scfg.copy(output_dir=FLAGS.save_dir),
                *parse_actions_from_cmd(argv))
        return

    run_seq(scfg.copy(output_dir=FLAGS.save_dir),
            WTL.create("inception3", 25, 303),
            Pause(60),
            WTL.create("alexnet", 100, 506),
            )
