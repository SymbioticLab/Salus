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
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL, Executor
from benchmarks.exps import run_tf, parse_actions_from_cmd, Pause, maybe_forced_preset


FLAGS = flags.FLAGS


def main(argv):
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = 'pack'

    # Then run on tf
    run_tf(FLAGS.save_dir / "resnet152_75-alone",
           WTL.create("resnet152", 75, 11, executor=Executor.TF),
           )
    return
    run_tf(FLAGS.save_dir / "resnet101_50-alone",
           WTL.create("resnet101", 50, 31, executor=Executor.TF),
           )
    run_tf(FLAGS.save_dir / "resnet101_50",
           WTL.create("resnet101", 50, 31, executor=Executor.TF),
           WTL.create("resnet101", 50, 31, executor=Executor.TF),
           )

    run_tf(FLAGS.save_dir / "resnet152_75",
           WTL.create("resnet152", 75, 22, executor=Executor.TF),
           WTL.create("resnet152", 75, 22, executor=Executor.TF),
           )

