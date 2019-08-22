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
Mixed training and inference workloads.

LaneMgr: enabled
Scheduler: fair
Work conservation: True
Collected data: speed over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

import os
from absl import flags

from benchmarks.driver.runner import Executor
from benchmarks.driver.server.config import presets
from benchmarks.driver.utils import execute
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, maybe_forced_preset, Pause, run_tf, run_tfdist


FLAGS = flags.FLAGS

def case(policy):
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = policy
    if policy == "mix":
        os.environ["SALUS_TIMEOUT"] = "666666"
    else:
        os.environ["SALUS_TIMEOUT"] = "0"

    folder_name = policy
    workload_list = [
        WTL.create("resnet50", 25, 500), # training
        Pause(3),
        WTL.create("resnet50eval", 1, 1000)
    ]
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/folder_name),
            *workload_list
            )



def main(argv):
    policy = argv[0]
    case(policy)
