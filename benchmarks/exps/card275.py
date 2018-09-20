# -*- coding: future_fstrings -*-
"""
Get mem allocation for one iteration, to plot CDF. See card#275

LaneMgr: enabled
InLane Scheduler: pack
Collected data: allocation
"""
from __future__ import absolute_import, print_function, division, unicode_literals

import inspect
from absl import flags

from benchmarks.driver.runner import Executor
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, maybe_forced_preset, run_tfdist, case_switch_main


FLAGS = flags.FLAGS


def case1(argv):
    model, bs, bn = 'inception3', 50, 10
    name = inspect.currentframe().f_code.co_name

    scfg = maybe_forced_preset(presets.AllocProf)
    scfg.scheduler = 'pack'

    wl = WTL.create(model, bs, bn)
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/name), wl)


def case2(argv):
    model, bs, bn = 'inception3', 50, 10
    name = inspect.currentframe().f_code.co_name

    scfg = maybe_forced_preset(presets.OpTracing)
    scfg.logconf = 'memop'
    scfg.scheduler = 'pack'

    wl = WTL.create(model, bs, bn)
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/name), wl)


@case_switch_main
def main():
    return case1, case2
