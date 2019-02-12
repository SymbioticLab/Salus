# -*- coding: future_fstrings -*-
#
# Copyright 2019 Peifeng Yu <peifeng@umich.edu>
# 
# This file is part of Salus
# (see https://github.com/SymbioticLab/Salus).
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
