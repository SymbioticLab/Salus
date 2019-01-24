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
OSDI Experiment 15

Have 99 jobs entering the system over the time, using packing scheduling.

Scheduler: pack
Work conservation: True
Collected data: JCT
"""
from __future__ import absolute_import, print_function, division, unicode_literals

import csv
import logging
import time
import shutil
from datetime import datetime
from itertools import chain
from timeit import default_timer
from typing import Sequence, Union

from absl import flags

from benchmarks.driver.server import SalusServer
from benchmarks.driver.server.config import presets
from benchmarks.driver.utils import try_with_default, atomic_directory, kill_tree, prompt
from benchmarks.driver.utils.compatiblity import pathlib
from benchmarks.driver.workload import WTL, Executor, RunConfig, Workload
from benchmarks.exps import run_seq, Pause, maybe_forced_preset


FLAGS = flags.FLAGS
TBatchSize = Union[str, int]
logger = logging.getLogger(__name__)

flags.DEFINE_boolean('use_salus', False, 'Run on Salus or TF')
flags.DEFINE_integer('concurrent', 3, 'Concurrent workload allowed on Salus')
flags.DEFINE_boolean('fifo', False, 'Use FIFO for TF')
flags.DEFINE_float('overcommit', 1, 'Factor of amount of total memory in TF for over commit')
flags.DEFINE_float('phymem', 14 * (1024 ** 3), 'Amount of physical memory in GPU, in bytes')
flags.DEFINE_integer('scale_down', 1, 'Scale down iterations')


def load_trace(path, ex):
    path = pathlib.Path(path)
    with path.open() as f:
        reader = csv.DictReader(f)

        def create_from_row(row):
            name, bs = row['model_name'].split('_')
            bs = try_with_default(int, bs, ValueError)(bs)
            bn = int(row['iterations'])
            submit_time = int(row['submit_time'])
            if FLAGS.scale_down > 1:
                bn = bn // FLAGS.scale_down
                submit_time = submit_time / FLAGS.scale_down
            return WTL.create(name, bs, bn, ex), submit_time, row
        return [create_from_row(row) for row in reader]


def find_geometry(w, field):
    """
    :type w: Workload
    :type field: str
    """
    if w.geometry[field] is not None:
        return w.geometry[field]

    # check for another bn
    for bn in w.wtl.available_batch_nums(w.batch_size):
        g = WTL.from_name(w.name).geometry(RunConfig(w.batch_size, bn, None), w.executor)
        if g[field] is not None:
            w.geometry[field] = g[field]
            return g[field]

    return None


def main(argv):
    # type: (Sequence[str]) -> None
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = 'pack'

    ex = Executor.Salus if FLAGS.use_salus else Executor.TF
    if FLAGS.fifo:
        logdir = FLAGS.save_dir / 'fifo'
    else:
        logdir = FLAGS.save_dir / ex.value
    # create workload instances
    workloads = load_trace(argv[0], ex)

    # Check and update if workloads have the info we need
    if ex == Executor.TF and not FLAGS.fifo:
        for w, _, _ in workloads:
            for field in ['peakmem']:
                if find_geometry(w, field) is None:
                    raise ValueError(f'Missing {field} data for workload {w.canonical_name} of {w.batch_num} iters'
                                     f', available geometries: {w.wtl._geometries}')

    # enable overcommit
    if FLAGS.overcommit > 1:
        for w, _, _ in workloads:
            w.env['TF_GPU_ALLOCATOR'] = 'cuda_managed'

    def accept_workload(w, alive):
        if FLAGS.fifo:
            return len(alive) == 0
        elif FLAGS.use_salus:
            return len(alive) < FLAGS.concurrent
        else:
            currmem = sum(wl.geometry.peakmem for wl in alive)
            return w.geometry.peakmem + currmem < FLAGS.overcommit * FLAGS.phymem

    try:
        try:
            with atomic_directory(logdir) as tmp:
                # copy trace file
                shutil.copy2(argv[0], str(tmp/'trace.csv'))

                with (tmp / 'exp15.output').open('w') as f:
                    started = []
                    pending = []
                    alive = []

                    def workload_done(proc):
                        w = proc.workload
                        logger.info(f'Finished workload {w.output_name}.{w.batch_num}iter.{w.job_id}')
                        print(f'{datetime.now()}: Finished workload '
                              f'{w.output_name}.{w.batch_num}iter.{w.job_id}',
                              file=f)

                    def do_stuff(rel_time):
                        if workloads:
                            w, submit_time, row = workloads[0]
                            if rel_time >= submit_time:
                                workloads.pop(0)
                                w.job_id = row["job_id"]
                                logger.info(f'Queued workload {w.output_name}.{w.batch_num}iter.{w.job_id}')
                                print(f'{datetime.now()}: Queued workload '
                                      f'{w.output_name}.{w.batch_num}iter.{w.job_id}',
                                      file=f)
                                pending.append(w)

                        _, alive[:] = SalusServer.wait_workloads(alive, timeout=0, callback=workload_done)

                        while pending and accept_workload(pending[0], alive):
                            w = pending.pop(0)

                            logger.info(f'Started workload {w.output_name}.{w.batch_num}iter.{w.job_id}')
                            print(f'{datetime.now()}: Started workload '
                                  f'{w.output_name}.{w.batch_num}iter.{w.job_id}',
                                  file=f)

                            output_file = tmp / f'{w.output_name}.{w.batch_num}iter.{w.job_id}.output'
                            w.run(output_file)
                            started.append(w)
                            alive.append(w)

                            _, alive[:] = SalusServer.wait_workloads(alive, timeout=0, callback=workload_done)

                        if not workloads and not pending:
                            _, alive[:] = SalusServer.wait_workloads(alive, callback=workload_done)
                            return False

                        return True

                    def event_loop():
                        # check every 0.1 second
                        interval = 0.1
                        origin = default_timer()
                        while True:
                            st = default_timer()
                            should_continue = do_stuff(st - origin)
                            if not should_continue:
                                break
                            ed = default_timer()

                            elispped = ed - st
                            time.sleep(interval - (elispped % interval))

                    if FLAGS.use_salus:
                        ss = SalusServer(scfg.copy(output_dir=logdir))
                        with ss.run():
                            event_loop()
                    else:
                        event_loop()

        except Exception as ex:
            logger.exception("Got exception when running workloads")
    finally:
        # if there's alive, we are doing cleanup
        for w, _, _ in workloads:
            if w.proc is not None and w.proc.poll() is None:
                logger.warning(f'Killing workload that is not stopped yet: {w.canonical_name}')
                kill_tree(w.proc, hard=True)

        # check each workloads and fix workload output_file path
        for w, _, _ in workloads:
            if not FLAGS.ignore_error and w.proc is not None and w.proc.returncode != 0:
                prompt.pause()
                raise RuntimeError(f'Workload {w.canonical_name} did not finish cleanly: {w.proc.returncode}')
            if w.output_file is not None:
                w.output_file = logdir / w.output_file.name
