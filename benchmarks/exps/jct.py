# -*- coding: future_fstrings -*-
from __future__ import absolute_import, print_function, division, unicode_literals
from builtins import str

import logging
import re
from absl import flags
from typing import Iterable, Sequence, Union

from benchmarks.driver.runner import RunConfig
from benchmarks.driver.server import SalusServer
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL, Executor
from benchmarks.driver.utils import atomic_directory, unique
from benchmarks.exps import parse_output_float, maybe_forced_preset


logger = logging.getLogger(__name__)
FLAGS = flags.FLAGS
TBatchSize = Union[str, int]

flags.DEFINE_boolean('basic_only', False, 'Only run basic 20 iterations JCT only')
flags.DEFINE_float('threshold', 0.1, 'What ratio is actual time allowed to have within target time')
flags.DEFINE_integer('max_chance', 10, 'How many times to try')
flags.DEFINE_boolean('resume', False, 'Check and skip existing configurations')
flags.DEFINE_multi_integer('extra_mins', [1, 5, 10], 'Extra lengths')


def select_workloads(argv):
    # type: (Iterable[str]) -> Iterable[(str, TBatchSize)]
    """Select workloads based on commandline"""
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


def do_jct(logdir, network, batch_size):
    """Do basic JCT on workload"""
    batch_num = 20

    final_dst = logdir / WTL.from_name(network).canonical_name(RunConfig(batch_size, batch_num, None))
    with atomic_directory(final_dst) as outputdir:
        logger.info(f'Measuring basic JCT for {batch_num} iterations')
        if not (final_dst / 'gpu.output').exists() or not FLAGS.resume:
            logger.info('    Running on TF')
            WTL.block_run(network, batch_size, batch_num, Executor.TF, outputdir / 'gpu.output')

        if not (final_dst / 'rpc.output').exists() or not FLAGS.resume:
            scfg = maybe_forced_preset(presets.MostEfficient)
            scfg.output_dir = outputdir
            server = SalusServer(scfg)
            with server.run():
                logger.info('    Warming up Salus')
                # always use 20 batch num when warming up
                WTL.block_run(network, batch_size, 20, Executor.Salus, outputdir / 'rpc-warm.output')

                logger.info('    Running on Salus')
                WTL.block_run(network, batch_size, batch_num, Executor.Salus, outputdir / 'rpc.output')

    return final_dst


def calc_periter(outputfile):
    """Calc per iteration time in seconds"""
    if not outputfile.exists():
        msg = f'File not found after running: {outputfile}'
        logger.fatal(msg)
        raise ValueError(msg)

    ptn = re.compile('^Average excluding ')
    with outputfile.open() as f:
        for line in f.readline():
            if ptn.match(line):
                try:
                    return float(re.split('[^0-9.]+', line)[1])
                except (ValueError, IndexError):
                    continue


def do_jct_hint(logdir, network, batch_size, per_iter, target, tag):
    """Calculate JCT for target time"""
    final_dst = logdir / f"{network}_{batch_size}_{tag}"
    with atomic_directory(final_dst) as outputdir:
        if (final_dst / 'rpc.output').exists() and FLAGS.resume:
            per_iter = parse_output_float(final_dst / 'rpc.output', r'^Average excluding[^0-9.]+([0-9.]+).*')
            return per_iter

        logger.info(f"Finding suitable batch_num for {tag}")
        actual = 0
        chance = FLAGS.max_chance
        batch_num = int(target / per_iter)
        while chance > 0 and abs(actual - target) >= target * FLAGS.threshold:
            chance -= 1
            batch_num = int(target / per_iter)

            logger.info(f'    Trying batch_num={batch_num}')
            file = outputdir / 'gpu.output'
            WTL.block_run(network, batch_size, batch_num, Executor.TF, file)

            actual = parse_output_float(file, r'^JCT[^0-9.]+([0-9.]+).*')
            # assume linear time distribution
            per_iter = actual / batch_num
            logger.info(f"   actual_time={actual}, per_iter={per_iter}")

        # use the batch_num to run salus
        logger.info(f"Using batch_num={batch_num} for {tag}")
        scfg = maybe_forced_preset(presets.MostEfficient)
        scfg.output_dir = outputdir
        server = SalusServer(scfg)
        with server.run():
            logger.info('    Warming up Salus')
            WTL.block_run(network, batch_size, 20, Executor.Salus, outputdir / 'rpc-warm.output')

            logger.info('    Running on Salus')
            WTL.block_run(network, batch_size, batch_num, Executor.Salus, outputdir / 'rpc.output')
    return per_iter


def main(argv):
    # type: (Sequence[str]) -> None
    for network, batch_size in select_workloads(argv):
        print(f"\n**** JCT: {network}_{batch_size} *****\n\n")
        # Do base jct first
        outputdir = do_jct(FLAGS.save_dir, network, batch_size)

        if FLAGS.basic_only:
            continue

        # Get per iter time
        per_iter = parse_output_float(outputdir / 'gpu.output', r'^Average excluding[^0-9.]+([0-9.]+).*')

        # for m in [1, 5, 10]:
        for m in FLAGS.extra_mins:
            per_iter = do_jct_hint(FLAGS.save_dir, network, batch_size, per_iter, 60 * m, f'{m}min')
