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
from benchmarks.driver.utils import execute, try_with_default
from benchmarks.exps import run_seq, maybe_forced_preset, run_tf


FLAGS = flags.FLAGS

flags.DEFINE_boolean('also_tf', False, "Also run on TF")


def main(argv):
    scfg = maybe_forced_preset(presets.OpTracing)

    name, bs, bn = 'vgg11', 25, 10
    if len(argv) > 0:
        name = argv[0]
    if len(argv) > 1:
        bs = argv[1]
        bs = try_with_default(int, bs, ValueError)(bs)
    if len(argv) > 2:
        bn = int(argv[2])

    def create_wl(ex):
        return WTL.create(name, bs, bn, executor=ex)

    # Run on Salus
    wl = create_wl(Executor.Salus)
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "salus" / '1'), wl)

    if FLAGS.also_tf:
        wl = create_wl(Executor.TF)
        wl.env['TF_CPP_MIN_VLOG_LEVEL'] = '1'
        wl.env['TF_CPP_MIN_LOG_LEVEL'] = ''
        run_tf(FLAGS.save_dir / "tf", wl)
        # filter and move file to a more convinent name
        for f in (FLAGS.save_dir / "tf").iterdir():
            f.rename(f.with_name('perf.output'))
            break
