# -*- coding: future_fstrings -*-
"""
LaneMgr seems to have a data race when processRequests in removingLane and when
requestingLanes. Try to reproduce by run lots of short job that creates and closes
lane. See card#241

LaneMgr: enabled
Scheduler: fair
Work conservation: True
Collected data: memory usage over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

import random
from absl import flags

from benchmarks.driver.runner import Executor
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, maybe_forced_preset, Pause, run_tf


FLAGS = flags.FLAGS


def test():
    scfg = maybe_forced_preset(presets.Debugging)

    run_seq(scfg.copy(output_dir=FLAGS.save_dir),
            WTL.create("inception4", 25, 10),
            WTL.create("inception3", 50, 10))


def main(argv):
    command = argv[0] if argv else "test"

    {
        "test": test,
    }[command]()
