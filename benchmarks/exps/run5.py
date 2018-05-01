# -*- coding: future_fstrings -*-
"""
OSDI Experiment 10

Have jobs of 5 different JCTs, let them enter the system in order, creating stair shaped memory usage figure.

Scheduler: fair
Work conservation: True
Collected data: memory usage over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.runner import Executor
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd, Pause, maybe_forced_preset, run_tf


FLAGS = flags.FLAGS


def main(argv):
    scfg = maybe_forced_preset(presets.AllocProf)
    if argv:
        run_seq(scfg.copy(output_dir=FLAGS.save_dir),
                *parse_actions_from_cmd(argv))
        return

    run_seq(scfg.copy(output_dir=FLAGS.save_dir),
            WTL.create("alexnet", 25, 200),
            WTL.create("alexnet", 25, 200),
            WTL.create("alexnet", 25, 200),
            WTL.create("alexnet", 25, 200),
            WTL.create("alexnet", 25, 200),
            )

    run_tf(FLAGS.save_dir,
           WTL.create("alexnet", 25, 200, executor=Executor.TF),
           Pause.Wait,
           WTL.create("alexnet", 25, 200, executor=Executor.TF),
           Pause.Wait,
           WTL.create("alexnet", 25, 200, executor=Executor.TF),
           Pause.Wait,
           WTL.create("alexnet", 25, 200, executor=Executor.TF),
           Pause.Wait,
           WTL.create("alexnet", 25, 200, executor=Executor.TF),
           )
