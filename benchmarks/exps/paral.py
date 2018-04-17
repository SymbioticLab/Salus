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

    # Firstly run concurrently on salus
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "conc"),
            WTL.create("resnet50", 25, 271),
            WTL.create("alexnet", 25, 2307),
            )
    return
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "seq"),
            WTL.create("alexnet", 25, 2307),
            )

    # Then run on tf
    run_tf(FLAGS.save_dir / "tf",
           WTL.create("alexnet", 50, 1714, executor=Executor.TF),
           Pause.Wait,
           WTL.create("alexnet", 50, 1714, executor=Executor.TF),
           )
