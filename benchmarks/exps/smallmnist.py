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

    if argv:
        run_seq(scfg.copy(output_dir=FLAGS.save_dir),
                *parse_actions_from_cmd(argv))
        return

    logdir = FLAGS.save_dir / "smallmnist"

    # Firstly run concurrently on salus
    run_seq(scfg.copy(output_dir=logdir / "salus"),
            WTL.create("mnistlg", 50, 5048),  # 1min
            WTL.create("mnistlg", 50, 27529),  # 5min
            )

    # Then run on tf
    run_seq(scfg.copy(output_dir=logdir / "tf"),
            WTL.create("mnistlg", 50, 5048, executor=Executor.TF),  # 1min
            Pause.Wait,
            WTL.create("mnistlg", 50, 27529, executor=Executor.TF),  # 5min
            )
