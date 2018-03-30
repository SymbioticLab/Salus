# -*- coding: future_fstrings -*-
"""
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
    scfg.logconf = 'disable'

    if argv:
        run_seq(scfg.copy(output_dir=FLAGS.save_dir),
                *parse_actions_from_cmd(argv))
        return

    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "salus" / "1"),
            WTL.create("mnistsf", 25, 100),
            Pause.Wait,
            WTL.create("mnistsf", 25, 200),
            Pause.Wait,
            WTL.create("mnistsf", 25, 300),
            )
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "tf"),
            WTL.create("mnistsf", 25, 100, executor=Executor.TF),  # 1min
            Pause.Wait,
            WTL.create("mnistsf", 25, 200, executor=Executor.TF),  # 1min
            Pause.Wait,
            WTL.create("mnistsf", 25, 300, executor=Executor.TF),  # 1min
            )

    for conc in range(2, 10):
        actions = [WTL.create("mnistsf", 25, 100) for _ in range(conc)]
        run_seq(scfg.copy(output_dir=FLAGS.save_dir / "salus" / str(conc)), *actions)

