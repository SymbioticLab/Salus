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
Collect data for optracing

Using alexnet_25 for 1 iteration

Scheduler: pack
Work conservation: True
Collected data: JCT
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL, Executor
from benchmarks.exps import run_seq, maybe_forced_preset, run_tf


FLAGS = flags.FLAGS


def main(argv):
    scfg = maybe_forced_preset(presets.OpTracing)

    name, bs, bn = 'vgg11', 25, 10
    if len(argv) > 0:
        name = argv[0]
    if len(argv) > 1:
        bs = int(argv[1])
    if len(argv) > 2:
        bn = int(argv[2])

    def create_wl(ex):
        return WTL.create(name, bs, bn, executor=ex)

    # Run 2 alexnet_25 on Salus
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "salus" / '2'),
            create_wl(Executor.Salus),
            create_wl(Executor.Salus),
            )
