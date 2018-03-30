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
    scfg = maybe_forced_preset(presets.Nvprof)
    scfg.scheduler = 'pack'
    scfg.logconf = 'disable'

    if argv:
        run_seq(scfg.copy(output_dir=FLAGS.save_dir),
                *parse_actions_from_cmd(argv))
        return

    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "5mnistsf25"),
            WTL.create("mnistsf", 25, 500),
            WTL.create("mnistsf", 25, 500),
            WTL.create("mnistsf", 25, 500),
            WTL.create("mnistsf", 25, 500),
            WTL.create("mnistsf", 25, 500),
            )
