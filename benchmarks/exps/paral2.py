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
from benchmarks.exps import run_seq, run_tf, Pause, maybe_forced_preset


FLAGS = flags.FLAGS


def main(argv):
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = 'pack'

    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "resnet101_50-alone"),
            WTL.create("resnet101", 50, 30),
            )
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "resnet101_50"),
            WTL.create("resnet101", 50, 30),
            WTL.create("resnet101", 50, 30),
            )
    return
    # Firstly run concurrently on salus
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "mix2"),
            WTL.create("resnet50", 25, 10),
            WTL.create("alexnet", 25, 10),
            )
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "alexnet_25"),
            WTL.create("alexnet", 25, 10),
            WTL.create("alexnet", 25, 10),
            )
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "resnet50_25"),
            WTL.create("resnet50", 25, 10),
            WTL.create("resnet50", 25, 10),
            )
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "seq"),
            WTL.create("alexnet", 25, 10),
            Pause.Wait,
            WTL.create("resnet50", 25, 10),
            )
    return

    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "alexnet_25-alone"),
            WTL.create("alexnet", 25, 10),
            )
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "resnet50_25-alone"),
            WTL.create("resnet50", 25, 10),
            )
