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
    scfg = maybe_forced_preset(presets.Nvprof)
    scfg.scheduler = 'pack'
    scfg.logconf = 'disable'

    if argv:
        run_seq(scfg.copy(output_dir=FLAGS.save_dir),
                *parse_actions_from_cmd(argv))
        return

    logdir = FLAGS.save_dir / "multistream"
    run_seq(scfg.copy(output_dir=logdir / "salus" / "2perf500"),
            WTL.create("mnistsf", 25, 500),
            WTL.create("mnistsf", 25, 500),
            )
    return

    run_seq(scfg.copy(output_dir=logdir / "salus" / "1"),
            WTL.create("mnistsf", 25, 100),
            Pause.Wait,
            WTL.create("mnistsf", 25, 200),
            Pause.Wait,
            WTL.create("mnistsf", 25, 300),
            )
    run_seq(scfg.copy(output_dir=logdir / "tf"),
            WTL.create("mnistsf", 25, 100, executor=Executor.TF),  # 1min
            Pause.Wait,
            WTL.create("mnistsf", 25, 200, executor=Executor.TF),  # 1min
            Pause.Wait,
            WTL.create("mnistsf", 25, 300, executor=Executor.TF),  # 1min
            )

    for conc in range(2, 10):
        actions = [WTL.create("mnistsf", 25, 100) for _ in range(conc)]
        run_seq(scfg.copy(output_dir=logdir / "salus" / str(conc)), *actions)

