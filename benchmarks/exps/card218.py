# -*- coding: future_fstrings -*-
"""
VAE and SuperRes runs very slow. See card#218

Scheduler: fair
Work conservation: True
Collected data: memory usage over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.runner import Executor
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, maybe_forced_preset, Pause, run_tf


FLAGS = flags.FLAGS


def case1():
    scfg = maybe_forced_preset(presets.MostEfficient)

    # BUG: seems we must run a single job first otherwise it will hang
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/'case1'),
            WTL.create("vae", 128, 20, executor=Executor.TF),
            Pause.Wait,
            WTL.create("vae", 128, 20))


def case2():
    # Run on TF
    wl = WTL.create("super_res", 128, 20, executor=Executor.TF)
    wl.env['TF_CPP_MIN_VLOG_LEVEL'] = '2'
    wl.env['TF_CPP_MIN_LOG_LEVEL'] = ''
    run_tf(FLAGS.save_dir/'case2'/'tf', wl)

    scfg = maybe_forced_preset(presets.OpTracing)
    scfg.logconf = 'memop'
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/'case2'/'salus'),
            WTL.create("super_res", 128, 20))


def test():
    scfg = maybe_forced_preset(presets.MostEfficient)

    # BUG: seems we must run a single job first otherwise it will hang
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/'test'),
            WTL.create("alexnet", 25, 20, executor=Executor.TF),
            Pause.Wait,
            WTL.create("alexnet", 25, 20))


def main(argv):
    command = argv[0] if argv else "case1"

    {
        "case1": case1,
        "case2": case2,
        "test": test,
    }[command]()
