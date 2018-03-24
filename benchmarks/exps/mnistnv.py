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

    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "5perf500"),
            WTL.create("mnistsf", 25, 500),
            WTL.create("mnistsf", 25, 500),
            WTL.create("mnistsf", 25, 500),
            WTL.create("mnistsf", 25, 500),
            WTL.create("mnistsf", 25, 500),
            )
