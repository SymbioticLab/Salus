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
from benchmarks.exps import run_seq, parse_actions_from_cmd, Pause, maybe_forced_preset


FLAGS = flags.FLAGS


def main(argv):
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = 'pack'
    scfg.logconf = 'disable'

    if argv:
        run_seq(scfg.copy(output_dir=FLAGS.save_dir),
                *parse_actions_from_cmd(argv))
        return

    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "salus" / "1"),
            WTL.create("mnistsf", 25, 100),
            Pause.Wait,
            WTL.create("mnistsf", 25, 200),
            Pause.Wait,
            WTL.create("mnistsf", 25, 300),
            )
    run_seq(scfg.copy(output_dir=FLAGS.save_dir / "tf"),
            WTL.create("mnistsf", 25, 100, executor=Executor.TF),  # 1min
            Pause.Wait,
            WTL.create("mnistsf", 25, 200, executor=Executor.TF),  # 1min
            Pause.Wait,
            WTL.create("mnistsf", 25, 300, executor=Executor.TF),  # 1min
            )

    for conc in range(2, 10):
        actions = [WTL.create("mnistsf", 25, 100) for _ in range(conc)]
        run_seq(scfg.copy(output_dir=FLAGS.save_dir / "salus" / str(conc)), *actions)

