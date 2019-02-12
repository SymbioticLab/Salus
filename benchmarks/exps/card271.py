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
Pack AutoML workloads together as possible on a GPU. See card#271

Use about 300 small jobs.

LaneMgr: enabled
InLane Scheduler: pack
Collected data: per iteration time (latency)
"""
from __future__ import absolute_import, print_function, division, unicode_literals

import inspect
from absl import flags

from benchmarks.driver.runner import Executor
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, maybe_forced_preset, run_tfdist, case_switch_main


FLAGS = flags.FLAGS


def test(argv):
    model, bs, bn = 'vae', 64, 500
    name = inspect.currentframe().f_code.co_name

    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = 'pack'

    # create 300 vae
    wls = [WTL.create(model, bs, bn) for _ in range(1)]
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/name), *wls)


def case1(argv):
    model, bs, bn = 'vae', 64, 500
    name = inspect.currentframe().f_code.co_name

    # first run one along to get JCT
    run_tfdist(FLAGS.save_dir/name, WTL.create(model, bs, bn, executor=Executor.TFDist))

    # create 300 vae
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = 'pack'

    wls = [WTL.create(model, bs, bn) for _ in range(300)]
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/name), *wls)


def case2(argv):
    model, bs, bn = 'superres', 128, 500
    name = inspect.currentframe().f_code.co_name

    # first run one along to get JCT
    run_tfdist(FLAGS.save_dir/name, WTL.create(model, bs, bn, executor=Executor.TFDist))

    # create 300 vae
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = 'pack'

    wls = [WTL.create(model, bs, bn) for _ in range(300)]
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/name), *wls)


def case3(argv):
    model, bs, bn = 'resnet50', 50, 500
    name = inspect.currentframe().f_code.co_name

    # first run one along to get JCT
    run_tfdist(FLAGS.save_dir/name, WTL.create(model, bs, bn, executor=Executor.TFDist))

    # create 300 vae
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = 'pack'

    wls = [WTL.create(model, bs, bn) for _ in range(300)]
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/name), *wls)


@case_switch_main
def main():
    return case1, case2, case3, test
