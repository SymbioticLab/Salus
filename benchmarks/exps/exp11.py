# -*- coding: future_fstrings -*-
"""
OSDI Experiment 11

Run 2 jobs together. One of them is really long running and the other one is short

- inception3_25 for 5min
- alexnet_100 for 0.5min

Scheduler: preempt
Work conservation: True
Collected data: Numer of scheduled tasks over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.runner import Executor
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd, Pause, maybe_forced_preset


FLAGS = flags.FLAGS
flags.DEFINE_boolean('with_ref', True, "Also do reference runs")


def main(argv):
    scfg = maybe_forced_preset(presets.Profiling)
    scfg.scheduler = 'preempt'
    scfg.disable_wc = False

    if argv:
        run_seq(scfg.copy(output_dir=FLAGS.save_dir),
                *parse_actions_from_cmd(argv))
        return

    run_seq(scfg.copy(output_dir=FLAGS.save_dir),
            WTL.create("inception3", 25, 1298),
            Pause(60),
            WTL.create("alexnet", 100, 508),
            )

    if not FLAGS.with_ref:
        return

    # we also need reference data
    run_seq(presets.MostEfficient(output_dir=FLAGS.save_dir / 'reference'),
            WTL.create("alexnet", 100, 508),
            Pause.Wait,
            WTL.create("alexnet", 100, 508, executor=Executor.TF),
            Pause.Wait,
            WTL.create("inception3", 25, 1298),
            Pause.Wait,
            WTL.create("inception3", 25, 1298, executor=Executor.TF),
            )
