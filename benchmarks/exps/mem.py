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
Measure memory usage
"""
from __future__ import absolute_import, print_function, division, unicode_literals
from builtins import str

import logging
from absl import flags
from typing import Iterable, Sequence, Union

from benchmarks.driver.runner import RunConfig
from benchmarks.driver.server import SalusServer
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL, Executor
from benchmarks.driver.utils import atomic_directory, unique, execute
from benchmarks.driver.utils.compatiblity import pathlib
from benchmarks.exps import maybe_forced_preset, run_tf


logger = logging.getLogger(__name__)
FLAGS = flags.FLAGS
TBatchSize = Union[str, int]

flags.DEFINE_boolean('use_salus', False, 'Use salus to measure rather than TF')
flags.DEFINE_boolean('resume', False, 'Reuse existing data')


def select_workloads(argv):
    # type: (Iterable[str]) -> Iterable[(str, TBatchSize)]
    """Select workloads based on commandline

        Example: alexnet,vgg11
    """
    if not argv:
        names = WTL.known_workloads.keys()
    else:
        names = unique((
            name
            for piece in argv
            for name in piece.split(',')
        ), stable=True)

    def getbs(name):
        if '_' in name:
            name, bs = name.split('_')
            bs = int(bs)
            return [(name, bs)]
        else:
            bss = WTL.from_name(name).available_batch_sizes()
            names = [name] * len(bss)
            return zip(names, bss)
    # TODO: return directly WTL instances
    return [(n, batch_size)
            for name in names
            for n, batch_size in getbs(name)]


def do_mem(logdir, network, batch_size):
    """Do basic JCT on workload"""
    batch_num = 20
    if network == "speech":
        batch_num = 10

    logger.info(f'Measuring memory for {network}_{batch_size} for {batch_num} iter')

    ex = "salus" if FLAGS.use_salus else "tf"
    final_dst = logdir / ex / WTL.from_name(network).canonical_name(RunConfig(batch_size, batch_num, None))

    if FLAGS.resume:
        filename = {
            'salus': 'rpc.output',
            'tf': 'alloc.output'
        }[ex]
        if (final_dst / filename).exists():
            return final_dst

    with atomic_directory(final_dst) as outputdir:
        if not FLAGS.use_salus:
            logger.info('    Running on TF')
            wl = WTL.create(network, batch_size, batch_num, Executor.TF)
            wl.env['TF_CPP_MIN_VLOG_LEVEL'] = '1'
            wl.env['TF_CPP_MIN_LOG_LEVEL'] = ''
            run_tf(outputdir, wl)
            # filter and move file to a more convinent name
            for f in pathlib.Path(outputdir).iterdir():
                with f.with_name('alloc.output').open('w') as file:
                    grep = execute(['egrep', r"] (\+|-)", f.name], stdout=file, cwd=str(f.parent))
                    grep.wait()
                f.unlink()
                break
        else:
            scfg = maybe_forced_preset(presets.AllocProf)
            scfg.output_dir = outputdir
            server = SalusServer(scfg)
            with server.run():
                logger.info('    Running on Salus')
                WTL.block_run(network, batch_size, batch_num, Executor.Salus, outputdir / 'rpc.output')

    return final_dst


def main(argv):
    # type: (Sequence[str]) -> None
    for network, batch_size in select_workloads(argv):
        print(f"\n**** Memory: {network}_{batch_size} *****\n\n")
        do_mem(FLAGS.save_dir, network, batch_size)
