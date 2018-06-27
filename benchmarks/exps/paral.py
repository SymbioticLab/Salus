# -*- coding: future_fstrings -*-
"""
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL, Executor
from benchmarks.exps import run_seq, run_tf, Pause, maybe_forced_preset


FLAGS = flags.FLAGS


def main(argv):
    scfg = maybe_forced_preset(presets.OpTracing)
    scfg.logconf = 'disable'

    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "resnet101_50-alone"),
            WTL.create("resnet101", 50, 31),
            )
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "resnet101_50"),
            WTL.create("resnet101", 50, 31),
            WTL.create("resnet101", 50, 31),
            )
    return
    # Firstly run concurrently on salus
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "mix2"),
            WTL.create("resnet50", 25, 10),
            WTL.create("alexnet", 25, 10),
            )
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "alexnet_25"),
            WTL.create("alexnet", 25, 10),
            WTL.create("alexnet", 25, 10),
            )
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "resnet50_25"),
            WTL.create("resnet50", 25, 10),
            WTL.create("resnet50", 25, 10),
            )
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "seq"),
            WTL.create("alexnet", 25, 10),
            Pause.Wait,
            WTL.create("resnet50", 25, 10),
            )
    return

    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "alexnet_25-alone"),
            WTL.create("alexnet", 25, 10),
            )
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "resnet50_25-alone"),
            WTL.create("resnet50", 25, 10),
            )
