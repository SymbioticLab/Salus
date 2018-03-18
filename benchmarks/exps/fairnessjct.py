# -*- coding: future_fstrings -*-
"""
JCT for fairness scheduler
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd, maybe_forced_preset


FLAGS = flags.FLAGS


def main(argv):
    scfg = maybe_forced_preset(presets.MostEfficient)

    if argv:
        run_seq(scfg.copy(output_dir=FLAGS.save_dir),
                *parse_actions_from_cmd(argv))
        return

    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "makespan_3of"),
            WTL.create("overfeat", 50, 424),
            WTL.create("overfeat", 50, 424),
            WTL.create("overfeat", 50, 424),
            )

    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "makespan_3res"),
            WTL.create("resnet50", 50, 265),
            WTL.create("resnet50", 50, 265),
            WTL.create("resnet50", 50, 265),
            )
