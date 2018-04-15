# -*- coding: future_fstrings -*-
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
from benchmarks.driver.utils import atomic_directory, unique
from benchmarks.driver.utils.compatiblity import pathlib
from benchmarks.exps import maybe_forced_preset, run_tf


logger = logging.getLogger(__name__)
FLAGS = flags.FLAGS
TBatchSize = Union[str, int]

flags.DEFINE_boolean('use_salus', False, 'Use salus to measure rather than TF')


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

    # TODO: return directly WTL instances
    return [(name, batch_size)
            for name in names
            for batch_size in WTL.from_name(name).available_batch_sizes()]


def do_mem(logdir, network, batch_size):
    """Do basic JCT on workload"""
    batch_num = 20

    logger.info(f'Measuring memory for {network}_{batch_size} for {batch_num} iter')

    ex = "salus" if FLAGS.use_salus else "tf"
    final_dst = logdir / ex / WTL.from_name(network).canonical_name(RunConfig(batch_size, batch_num, None))
    with atomic_directory(final_dst) as outputdir:
        if not FLAGS.use_salus:
            logger.info('    Running on TF')
            wl = WTL.create(network, batch_size, batch_num, Executor.TF)
            wl.env['TF_CPP_MIN_VLOG_LEVEL'] = '2'
            wl.env['TF_CPP_MIN_LOG_LEVEL'] = ''
            run_tf(outputdir, wl)
            # move file to a more convinent name
            for f in pathlib.Path(outputdir).iterdir():
                f.rename(f.with_name('alloc.output'))
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
