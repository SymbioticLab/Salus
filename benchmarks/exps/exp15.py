# -*- coding: future_fstrings -*-
"""
OSDI Experiment 15

Have 99 jobs entering the system over the time, using packing scheduling.

Scheduler: pack
Work conservation: True
Collected data: JCT
"""
from __future__ import absolute_import, print_function, division, unicode_literals

import logging
from enum import Enum
from itertools import chain
import time
from itertools import islice
from absl import flags
from typing import Sequence, Iterable, Union, Tuple

from benchmarks.driver.utils import UsageError, unique, try_with_default
from benchmarks.driver.server import SalusServer
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL, Executor, RunConfig, Workload
from benchmarks.exps import run_seq, RunFn, maybe_forced_preset


FLAGS = flags.FLAGS
TBatchSize = Union[str, int]
logger = logging.getLogger(__name__)

flags.DEFINE_integer('concurrent_jobs', 2, 'Maximum concurrent running jobs', lower_bound=1)
flags.DEFINE_integer('total_num', 0, 'Only run this number of workloads. If 0, means no limit', lower_bound=0)
flags.DEFINE_string('select_wl', '', 'Select only to run workloads from the list of canonical names given')


class Cases(Enum):
    Shortest = ('jct', False)
    Longest = ('jct', True)
    Smallest = ('persistmem', False)
    Largest = ('persistmem', True)


def gen_workload_list(selection):
    # type: (str) -> Iterable[Tuple[WTL, RunConfig]]
    """Select workloads based on commandline"""
    if not selection:
        blacklist = ['speech', 'seq2seq', 'mnistlg', 'mnistsf', 'mnistcv']
        names = (
            (v, bs)
            for k, v in WTL.known_workloads.items()
            for bs in v.available_batch_sizes()
            if k not in blacklist
        )
    else:
        names = []
        for cname in unique((cname for cname in selection.split(',')), stable=True):
            if '_' not in cname:
                raise UsageError(f"Not a canonical name: {cname}")
            name, bs = cname.split('_', 1)
            bs = try_with_default(int, bs, ValueError)(bs)
            names.append((WTL.from_name(name), bs))

    # Find all available batch_num with JCT and mem data
    return (
        (wtl, RunConfig(bs, bn, None))
        for wtl, bs in names
        for bn in wtl.available_batch_nums(bs)
    )


def main(argv):
    # type: (Sequence[str]) -> None
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = 'pack'

    cases = (Cases[c] for c in argv) if argv else Cases
    templates = list(gen_workload_list(FLAGS.select_wl))
    if FLAGS.total_num > 0:
        templates = templates[:FLAGS.total_num]

    logger.info("Selected the following list of workloads")
    for wtl, rcfg in templates:
        logger.info(f"    {wtl.canonical_name(rcfg)} of {rcfg.batch_num} iters")

    # Check if workloads have the info we need
    for wtl, rcfg in templates:
        for field in ['jct', 'persistmem']:
            if wtl.geometry(rcfg, Executor.Salus)[field] is None:
                raise ValueError(f'Missing {field} data for workload {wtl.canonical_name(rcfg)} of {rcfg.batch_num} iters, available geometries: {wtl._geometries}')

    for case in cases:
        logdir = FLAGS.save_dir / case.name

        # create workload instances
        workloads = (wtl._create_from_rcfg(rcfg, Executor.Salus) for wtl, rcfg in templates)
        # sort workload according to case
        key, desc = case.value
        workloads = sorted(workloads, key=lambda w: w.geometry[key], reverse=desc)

        def limit_concurrent(wls):
            # type: (Iterable[Workload]) -> None
            """Wait for something to finish"""
            gone, alive = SalusServer.wait_workloads(wls, timeout=0)
            while len(alive) >= FLAGS.concurrent_jobs:
                gone, alive = SalusServer.wait_workloads(wls, timeout=0)
                time.sleep(.25)

        actions = chain(*(
            [w, RunFn(limit_concurrent)]
            for w in workloads
        ))

        run_seq(scfg.copy(output_dir=logdir), *actions)
