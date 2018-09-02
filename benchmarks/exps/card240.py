# -*- coding: future_fstrings -*-
"""
Test if tf dist server works. See card#240
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.runner import Executor
from benchmarks.driver.workload import WTL
from benchmarks.exps import Pause, run_tfdist


FLAGS = flags.FLAGS


def test():
    run_tfdist(FLAGS.save_dir,
               WTL.create("inception4", 25, 1, executor=Executor.TFDist),
               Pause.Wait,
               WTL.create("inception3", 50, 1, executor=Executor.TFDist))


def main(argv):
    command = argv[0] if argv else "test"

    {
        "test": test,
    }[command]()
