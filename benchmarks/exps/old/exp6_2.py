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
Collect data for together JCT run

Run two alexnet_25

Run all workloads for 1 iteration

Scheduler: pack
Work conservation: True
Collected data: optracing
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from collections import Iterable

from absl import flags
from typing import Union

from benchmarks.driver.server.config import presets
from benchmarks.driver.utils import unique
from benchmarks.driver.workload import WTL, Executor
from benchmarks.exps import run_seq, run_tf, maybe_forced_preset


FLAGS = flags.FLAGS

flags.DEFINE_boolean('using_mps', False, 'If MPS server is started')
flags.DEFINE_boolean('use_oc', False, 'If use UMA memory')


def main(argv):
    scfg = maybe_forced_preset(presets.MostEfficient)

    name, bs, bn = 'alexnet', 25, 2307
    if len(argv) > 0:
        name = argv[0]
    if len(argv) > 1:
        bs = int(argv[1])
    if len(argv) > 2:
        bn = int(argv[2])

    def create_wl(ex):
        wl = WTL.create(name, bs, bn, executor=ex)
        if FLAGS.use_oc:
            wl.env['TF_GPU_ALLOCATOR'] = 'cuda_managed'
        return wl

    if FLAGS.using_mps:
        # Run 2 alexnet_25 on TF
        logdirname = 'tf-mps-oc' if FLAGS.use_oc else 'tf-mps'
        run_tf(FLAGS.save_dir / logdirname,
               create_wl(Executor.TF),
               create_wl(Executor.TF),
               )
        return

    # Run alexnet_25 on Salus
    wl = create_wl(Executor.Salus)
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "salus" / '1'), wl)

    # Run 2 alexnet_25 on Salus
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "salus" / '2'),
            create_wl(Executor.Salus),
            create_wl(Executor.Salus),
            )

    # Run alexnet_25 on TF
    wl = create_wl(Executor.TF)
    run_tf(FLAGS.save_dir / 'tf', wl)

    # Run 2 alexnet_25 on TF
    run_tf(FLAGS.save_dir / 'tf-nomps',
           create_wl(Executor.TF),
           create_wl(Executor.TF),
           )
