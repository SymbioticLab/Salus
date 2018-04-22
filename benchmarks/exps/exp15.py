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
import csv
from absl import flags
from typing import Sequence, Iterable, Union, Tuple

from benchmarks.driver.utils import UsageError, unique, try_with_default
from benchmarks.driver.utils.compatiblity import pathlib
from benchmarks.driver.server import SalusServer
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL, Executor, RunConfig, Workload
from benchmarks.exps import run_seq, Pause, maybe_forced_preset


FLAGS = flags.FLAGS
TBatchSize = Union[str, int]
logger = logging.getLogger(__name__)

flags.DEFINE_boolean('use_salus', True, 'Run on Salus or TF')


def load_trace(path, ex):
    path = pathlib.Path(path)
    with path.open() as f:
        reader = csv.DictReader(f)

        def create_from_row(row):
            name, bs = row['name'].split('_')
            bs = try_with_default(int, bs, ValueError)(bs)
            bn = int(row['iterations'])
            interval = int(row['interval'])
            return WTL.create(name, bs, bn, ex), interval
        return [create_from_row(row) for row in reader]


def main(argv):
    # type: (Sequence[str]) -> None
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = 'pack'

    ex = Executor.Salus if FLAGS.use_salus else Executor.TF
    logdir = FLAGS.save_dir / ex.value
    # create workload instances
    workloads = load_trace(argv[0], ex)

    actions = chain(*(
        [w, Pause(interval)]
        for w, interval in workloads
    ))

    run_seq(scfg.copy(output_dir=logdir), *actions)
