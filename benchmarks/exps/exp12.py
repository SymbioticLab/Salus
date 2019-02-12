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
OSDI Experiment 12

Run 2 jobs together.

Scheduler: preempt
Work conservation: False
Collected data: Numer of scheduled tasks over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd, Pause, maybe_forced_preset

FLAGS = flags.FLAGS


def main(argv):
    scfg = maybe_forced_preset(presets.Profiling)
    scfg.scheduler = 'preempt'
    scfg.disable_wc = True

    if argv:
        run_seq(scfg.copy(output_dir=FLAGS.save_dir),
                *parse_actions_from_cmd(argv))
        return

    run_seq(scfg.copy(output_dir=FLAGS.save_dir),
            WTL.create("inception3", 25, 1298),
            Pause(60),
            WTL.create("alexnet", 100, 508),
            )

    if not FLAGS.with_ref:
        return

    # we also need reference data
    run_seq(presets.MostEfficient(output_dir=FLAGS.save_dir / 'reference'),
            WTL.create("alexnet", 100, 508),
            Pause.Wait,
            WTL.create("alexnet", 100, 508, executor=Executor.TF),
            )
