# -*- coding: future_fstrings -*-
"""
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL, Executor
from benchmarks.exps import run_tf, parse_actions_from_cmd, Pause, maybe_forced_preset


FLAGS = flags.FLAGS


def main(argv):
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = 'pack'

    # Then run on tf
    run_tf(FLAGS.save_dir / "resnet152_75-alone",
           WTL.create("resnet152", 75, 11, executor=Executor.TF),
           )
    return
    run_tf(FLAGS.save_dir / "resnet101_50-alone",
           WTL.create("resnet101", 50, 31, executor=Executor.TF),
           )
    run_tf(FLAGS.save_dir / "resnet101_50",
           WTL.create("resnet101", 50, 31, executor=Executor.TF),
           WTL.create("resnet101", 50, 31, executor=Executor.TF),
           )

    run_tf(FLAGS.save_dir / "resnet152_75",
           WTL.create("resnet152", 75, 22, executor=Executor.TF),
           WTL.create("resnet152", 75, 22, executor=Executor.TF),
           )

