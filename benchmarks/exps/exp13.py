# -*- coding: future_fstrings -*-
"""
OSDI Experiment 13

Tasks start at begining
For TF-Salus: JCT of running 2 jobs together, using packing scheduling
For TF: run the same jobs sequentially.

Show packing can improve JCT

Scheduler: pack
Work conservation: True
Collected data: JCT
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL, Executor
from benchmarks.exps import run_seq, parse_actions_from_cmd, Pause, maybe_forced_preset


FLAGS = flags.FLAGS


def main(argv):
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = 'pack'

    # Firstly run concurrently on salus
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "salus"),
            WTL.create("alexnet", 50, 1714),
            WTL.create("alexnet", 50, 1714),
            )

    # Then run on tf
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "tf"),
            WTL.create("alexnet", 50, 1714, executor=Executor.TF),
            Pause.Wait,
            WTL.create("alexnet", 50, 1714, executor=Executor.TF),
            )
