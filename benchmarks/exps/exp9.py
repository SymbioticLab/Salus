# -*- coding: future_fstrings -*-
"""
OSDI Experiment 9

Run 2 jobs together.

Scheduler: fair
Work conservation: True
Collected data: fairness counter over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd, maybe_forced_preset

FLAGS = flags.FLAGS


def main(argv):
    scfg = maybe_forced_preset(presets.Profiling)
    if argv:
        run_seq(scfg.copy(output_dir=FLAGS.save_dir),
                *parse_actions_from_cmd(argv))
        return

    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "exp9"),
            WTL.create("resnet50", 50, 265),
            WTL.create("resnet50", 50, 265),
            )
