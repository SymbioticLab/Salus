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
Card 308: Make inference job accepts requests using tfweb

Export SavedModel for existing tf_cnn_benchmark models

Collected data: SavedModel
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags
from typing import Sequence
import logging

from benchmarks.driver.runner import TFBenchmarkRunner
from benchmarks.driver.utils.compatiblity import pathlib
from benchmarks.driver.workload import Executor
from benchmarks.exps import run_tf, select_workloads

FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)


def main(argv):
    # type: (Sequence[str]) -> None

    model_dir = pathlib.Path('~/../symbiotic/peifeng/tf_cnn_benchmarks_models/legacy_checkpoint_models')
    model_dir = model_dir.expanduser().resolve()

    saved_model_dir = pathlib.Path('~/../symbiotic/peifeng/tf_cnn_benchmarks_models/saved_models')
    saved_model_dir = saved_model_dir.expanduser().resolve()

    for wl in select_workloads(argv, batch_size=1, batch_num=1, executor=Executor.TF):
        if wl.wtl.runnerCls is not TFBenchmarkRunner:
            logger.info(f'Skipping {wl.name}')
            continue
        if not wl.name.endswith('eval'):
            logger.info(f'Skipping {wl.name}')
            continue

        logger.info(f"**** Saving SavedModel: {wl.canonical_name}")
        logger.info(f"**** Location: {FLAGS.save_dir}")

        wl.env['SALUS_TFBENCH_EVAL_MODEL_DIR'] = str(model_dir)
        wl.env['SALUS_TFBENCH_EVAL_SAVED_MODEL_DIR'] = str(saved_model_dir)
        run_tf(FLAGS.save_dir, wl)
