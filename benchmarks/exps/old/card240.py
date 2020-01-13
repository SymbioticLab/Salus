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
Test if tf dist server works. See card#240
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.runner import Executor
from benchmarks.driver.workload import WTL
from benchmarks.exps import Pause, run_tfdist


FLAGS = flags.FLAGS


def test():
    run_tfdist(FLAGS.save_dir,
               WTL.create("inception4", 25, 1, executor=Executor.TFDist),
               Pause.Wait,
               WTL.create("inception3", 50, 1, executor=Executor.TFDist))


def main(argv):
    command = argv[0] if argv else "test"

    {
        "test": test,
    }[command]()
