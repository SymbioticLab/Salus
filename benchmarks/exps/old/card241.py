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
LaneMgr seems to have a data race when processRequests in removingLane and when
requestingLanes. Try to reproduce by run lots of short job that creates and closes
lane. See card#241

LaneMgr: enabled
Scheduler: fair
Work conservation: True
Collected data: memory usage over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

import random
from absl import flags

from benchmarks.driver.runner import Executor
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, maybe_forced_preset, Pause, run_tf


FLAGS = flags.FLAGS


def test():
    scfg = maybe_forced_preset(presets.Debugging)

    run_seq(scfg.copy(output_dir=FLAGS.save_dir),
            WTL.create("inception4", 25, 10),
            WTL.create("inception3", 50, 10))


def main(argv):
    command = argv[0] if argv else "test"

    {
        "test": test,
    }[command]()
