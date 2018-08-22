# -*- coding: future_fstrings -*-
"""
TF allocator cannot be used in multiple streams. See card#210.

Plot allocation and compute timeline, with iteration data

Scheduler: fair
Work conservation: True
Collected data: memory usage over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, maybe_forced_preset, Pause


FLAGS = flags.FLAGS


def case1():
    scfg = maybe_forced_preset(presets.OpTracing)
    scfg.logconf = 'memop'

    # BUG: seems we must run a single job first otherwise it will hang
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/'case1'),
            WTL.create("inception3", 100, 10),
            Pause.Wait,
            WTL.create("resnet50", 50, 10),
            Pause.Wait,
            WTL.create("inception3", 100, 20),
            WTL.create("resnet50", 50, 20))


def case2():
    """Inception3_100 is missing some dealloc log entry"""
    scfg = maybe_forced_preset(presets.Debugging)
    run_seq(scfg.copy(output_dir=FLAGS.save_dir),
            WTL.create("inception3", 100, 10),
            Pause.Wait,
            WTL.create("inception3", 50, 10),
            Pause.Wait,
            WTL.create("inception3", 25, 10),
            Pause.Wait
            )


def main(argv):
    command = argv[0] if argv else "case1"

    {
        "case1": case1,
        "case2": case2,
    }[command]()
