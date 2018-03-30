# -*- coding: future_fstrings -*-
"""
OSDI Experiment 18

Run 2-20 alexnet networks, until the JCT begins to increase.
At which time show paging activity statistics.

Scheduler: pack
Admission control: False
Work conservation: True
Collected data: JCT
"""
from __future__ import absolute_import, print_function, division, unicode_literals

import numpy as np
import logging
from absl import flags
from typing import Sequence

from benchmarks.driver.runner import Executor, RunConfig
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd, update_jct, maybe_forced_preset


flags.DEFINE_float('break_when', 1.1, 'When JCT is high than this ratio compared to original value, stop')
flags.DEFINE_integer('uplimit', 20, 'The upper bound of how many concurrent workload we try')
FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)


def report(concurrent, jct, ratio):
    # type: (int, float, float) -> None
    logger.info(f'Parallel=f{concurrent} JCT={jct:.2f} {ratio:.2%}')


def main(argv):
    # type: (Sequence[str]) -> None
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = 'pack'
    scfg.disable_adc = True

    if argv:
        run_seq(scfg.copy(output_dir=FLAGS.save_dir),
                *parse_actions_from_cmd(argv))
        return

    wtl = WTL.from_name('alexnet')
    rcfg = RunConfig(25, 2726, None)

    # check if we have reference JCT
    reference_jct = wtl.geometry(rcfg, Executor.Salus).jct

    if reference_jct is None:
        start_from = 1
        logger.warning(f"No reference JCT data available for `{wtl.canonical_name(rcfg)}'")
    else:
        start_from = 2
        report(1, reference_jct, 1)

    logger.info(f'Will stop when JCT degratation larger than {FLAGS.break_when}')
    for concurrent in range(start_from, FLAGS.uplimit):
        # run them at once
        logger.info(f'Runing {concurrent} workloads together')
        workloads = [wtl.create_from_rcfg(rcfg) for _ in range(concurrent)]
        run_seq(scfg.copy(output_dir=FLAGS.save_dir / f"{concurrent}"), *workloads)

        # calculate average jct
        for w in workloads:
            update_jct(w)
        jcts = [w.geometry.jct for w in workloads]
        avgjct = np.mean(jcts)  # type: float
        ratio = avgjct / reference_jct
        report(concurrent, avgjct, ratio)
        if ratio > FLAGS.break_when:
            break

