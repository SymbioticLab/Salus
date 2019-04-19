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
Measure the throughtput and latency of each batch size.

Collected data: model, batch_size, latency, throughtput (in 2 min)
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags
from typing import Sequence
import logging

from benchmarks.driver.runner import TFBenchmarkRunner
from benchmarks.driver.server.config import presets
from benchmarks.driver.utils.compatiblity import pathlib
from benchmarks.driver.workload import Executor, WTL
from benchmarks.exps import run_tf, select_workloads, run_seq, maybe_forced_preset


FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)


def set_env(wl):
    wl.env['SALUS_TFBENCH_EVAL_INTERVAL'] = '0'
    wl.env['SALUS_TFBENCH_EVAL_RAND_FACTOR'] = '0'
    wl.env['SALUS_TFBENCH_EVAL_BLOCK'] = 'true'

    model_dir = pathlib.Path('~/../symbiotic/peifeng/tf_cnn_benchmarks_models/legacy_checkpoint_models')
    model_dir = model_dir.expanduser().resolve()
    wl.env['SALUS_TFBENCH_EVAL_MODEL_DIR'] = model_dir


def do_measure(scfg, name, batch_sizes):
    batch_num = 100
    # batch_sizes = [1, 2, 4, 8, 16, 32]
    # batch_sizes = [1024, 1536, 2048, 4096]
    for bs in batch_sizes:
        wl = WTL.create(name, bs, batch_num, executor=Executor.Salus)
        set_env(wl)
        run_seq(scfg.copy(output_dir=FLAGS.save_dir / "salus"), wl)

        wl = WTL.create(name, bs, batch_num, executor=Executor.TF)
        set_env(wl)
        run_seq(scfg.copy(output_dir=FLAGS.save_dir / "tf"), wl)


def main(argv):
    # type: (Sequence[str]) -> None
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.logconf = 'disable'

    name = "alexnet"
    if len(argv) > 1:
        name = argv[0]
    batch_sizes = [int(v) for v in argv[1:]]

    if not batch_sizes:
        batch_sizes = [1, 2, 4, 8]

    do_measure(scfg, name, batch_sizes)

