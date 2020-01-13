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
Tune pending: Experiment one inference job with one training job

Almost the same as Card 304, but try to tune the pending parameter.

The propurse of this experiment is to tune and debug the SM tracing pipeline.

- reduce the inference latency, and see if the tail latency for training reduces

Record inference latency. Compare inference job latency running along vs. running with a training job.

The latency should be measured with increasing throughput (qps) for the inference job.

Collected data: inference per iteration speed (latency), training throughput (derived from per iteration speed)
"""
from __future__ import absolute_import, print_function, division, unicode_literals

import tempfile
from typing import Sequence

from absl import flags
import logging
import os

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL, Executor
from benchmarks.driver.utils.compatiblity import pathlib
from benchmarks.exps import (
    run_seq, maybe_forced_preset, RunFn, Pause, wait_on_pipe, release_on_pipe,
    case_switch_main,
    run_tfdist, run_tf,
    random_id,
)


FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)


def set_env(wl):
    wl.env['SALUS_TFBENCH_EVAL_BLOCK'] = 'true'

    model_dir = pathlib.Path('~/../symbiotic/peifeng/tf_cnn_benchmarks_models/legacy_checkpoint_models')
    model_dir = model_dir.expanduser().resolve()
    wl.env['SALUS_TFBENCH_EVAL_MODEL_DIR'] = model_dir


def create_train(executor, idx, td=None):
    # the batch number has no effect here, only used to distinguish different runs
    train_wl = WTL.create('inception4', 50, 100 + idx, executor=executor)
    # make sure it runs long enough
    train_wl.env['SALUS_ITER_SECONDS'] = '300'

    if td is not None:
        # create a pipe to signal train_wl
        pipetrain = str(pathlib.Path(td) / f'{train_wl.canonical_name}-{random_id()}-fifo')
        os.mkfifo(pipetrain)
        train_wl.env['SALUS_WAIT_FOR_SIGNAL'] = pipetrain
        return train_wl, pipetrain
    return train_wl


def create_infer(executor, bs, td=None):
    wl = WTL.create('vgg11eval', bs, 300, executor=executor)
    set_env(wl)
    wl.env['SALUS_ITER_SECONDS'] = '150'
    wl.extra_args += [
        # '--eval_interval_secs=0.02',
        # '--eval_interval_random_factor=5'
    ]

    if td is not None:
        pipe = str(pathlib.Path(td) / f'{wl.canonical_name}-{random_id()}-fifo')
        os.mkfifo(pipe)
        wl.env['SALUS_WAIT_FOR_SIGNAL'] = pipe
        return wl, pipe

    return wl


def alone_tf(_argv):
    # run tf
    # the foreground inference job
    wl = create_infer(Executor.TF, 10)
    wl.extra_args += ['--min_mem']
    run_tf(FLAGS.save_dir / "alone", wl)

    # the background training job
    wl = create_train(Executor.TF, 0)
    wl.extra_args += ['--min_mem']
    run_tf(FLAGS.save_dir / "alone", wl)


def alone(argv):
    """Run each workload alone for reference"""
    sm_factors = [float(v) for v in argv]
    if not sm_factors:
        sm_factors = [1.0, 1.5, 2.0, 2.5, 3.0]

    logger.info(f"Running Salus with sm factors: {sm_factors}")

    # run salus
    for factor in sm_factors:
        scfg = maybe_forced_preset(presets.MostEfficient)
        scfg.extra_args += [
            '--sm-factor', f'{factor:.2f}'
        ]
        logger.info(f"Running Salus with sm factor: {factor}")
        wl = create_infer(Executor.Salus, 10)
        run_seq(scfg.copy(output_dir=FLAGS.save_dir / "alone" / f"{factor:.2f}"), wl)

        # the background training job
        wl = create_train(Executor.Salus, 0)
        run_seq(scfg.copy(output_dir=FLAGS.save_dir / "alone" / f"{factor:.2f}"), wl)


def salus(argv):
    # type: (Sequence[str]) -> None
    base_cfg = maybe_forced_preset(presets.MostEfficient)

    sm_factors = [float(v) for v in argv]
    if not sm_factors:
        sm_factors = [1.0, 1.5, 2.0, 2.5, 3.0]

    for idx, factor in enumerate(sm_factors):
        scfg = base_cfg.copy(output_dir=FLAGS.save_dir / "salus" / f"{factor:.2f}")
        scfg.extra_args += [
            '--sm-factor', f'{factor:.2f}'
        ]
        with tempfile.TemporaryDirectory() as td:
            # create a background training job
            train_wl, pipetrain = create_train(Executor.Salus, 0, td)

            # create the foreground inference job
            wl, pipe = create_infer(Executor.Salus, 10, td)

            run_seq(scfg,
                    train_wl,  # start the background job
                    wl,  # start the foreground job
                    # wait for both jobs to be ready
                    RunFn(lambda *args, **kwargs: wait_on_pipe(pipetrain)),
                    RunFn(lambda *args, **kwargs: wait_on_pipe(pipe)),
                    # start train job
                    RunFn(lambda *args, **kwargs: release_on_pipe(pipetrain)),
                    # wait 10 seconds
                    Pause(10),
                    # release inference job
                    RunFn(lambda *args, **kwargs: release_on_pipe(pipe)),
                    # run_seq automatically join all jobs at the end of the sequence
                    )


def inverse_salus(argv):
    # type: (Sequence[str]) -> None
    """Inversed priority for training and inference"""
    base_cfg = maybe_forced_preset(presets.MostEfficient)

    sm_factors = [float(v) for v in argv]
    if not sm_factors:
        sm_factors = [1.0, 1.5, 2.0, 2.5, 3.0]

    for idx, factor in enumerate(sm_factors):
        scfg = base_cfg.copy(output_dir=FLAGS.save_dir / "inverse" / f"{factor:.2f}")
        scfg.extra_args += [
            '--sm-factor', f'{factor:.2f}'
        ]
        with tempfile.TemporaryDirectory() as td:
            # create a background training job
            train_wl, pipetrain = create_train(Executor.Salus, 0, td)

            # create the foreground inference job
            wl, pipe = create_infer(Executor.Salus, 10, td)
            wl.extra_args += [
                '--eval_sched_priority', '40'
            ]

            run_seq(scfg,
                    train_wl,  # start the background job
                    wl,  # start the foreground job
                    # wait for both jobs to be ready
                    RunFn(lambda *args, **kwargs: wait_on_pipe(pipetrain)),
                    RunFn(lambda *args, **kwargs: wait_on_pipe(pipe)),
                    # start train job
                    RunFn(lambda *args, **kwargs: release_on_pipe(pipetrain)),
                    # wait 10 seconds
                    Pause(10),
                    # release inference job
                    RunFn(lambda *args, **kwargs: release_on_pipe(pipe)),
                    # run_seq automatically join all jobs at the end of the sequence
                    )


def same_pri_salus(argv):
    # type: (Sequence[str]) -> None
    """Inversed priority for training and inference"""
    base_cfg = maybe_forced_preset(presets.MostEfficient)

    sm_factors = [float(v) for v in argv]
    if not sm_factors:
        sm_factors = [1.0, 1.5, 2.0, 2.5, 3.0]

    for idx, factor in enumerate(sm_factors):
        scfg = base_cfg.copy(output_dir=FLAGS.save_dir / "same_pri" / f"{factor:.2f}")
        scfg.extra_args += [
            '--sm-factor', f'{factor:.2f}'
        ]
        with tempfile.TemporaryDirectory() as td:
            # create a background training job
            train_wl, pipetrain = create_train(Executor.Salus, 0, td)

            # create the foreground inference job
            wl, pipe = create_infer(Executor.Salus, 10, td)
            wl.extra_args += [
                '--eval_sched_priority', '20'
            ]

            run_seq(scfg,
                    train_wl,  # start the background job
                    wl,  # start the foreground job
                    # wait for both jobs to be ready
                    RunFn(lambda *args, **kwargs: wait_on_pipe(pipetrain)),
                    RunFn(lambda *args, **kwargs: wait_on_pipe(pipe)),
                    # start train job
                    RunFn(lambda *args, **kwargs: release_on_pipe(pipetrain)),
                    # wait 10 seconds
                    Pause(10),
                    # release inference job
                    RunFn(lambda *args, **kwargs: release_on_pipe(pipe)),
                    # run_seq automatically join all jobs at the end of the sequence
                    )


def tfmps(argv):
    # type: (Sequence[str]) -> None
    batch_sizes = [int(v) for v in argv[1:]]

    if not batch_sizes:
        batch_sizes = [1, 2, 4, 8]

    for idx, bs in enumerate(batch_sizes):
        with tempfile.TemporaryDirectory() as td:
            # create a background training job
            train_wl, pipetrain = create_train(Executor.TF, idx, td)
            train_wl.extra_args += ['--min_mem']

            # create the foreground inference job
            wl, pipe = create_infer(Executor.TF, bs, td)
            wl.extra_args += ['--min_mem']

            run_tf(FLAGS.save_dir / "tfmps",
                   train_wl,  # start the background job
                   wl,  # start the foreground job
                   # wait for both jobs to be ready
                   RunFn(lambda *args, **kwargs: wait_on_pipe(pipetrain)),
                   RunFn(lambda *args, **kwargs: wait_on_pipe(pipe)),
                   # start train job
                   RunFn(lambda *args, **kwargs: release_on_pipe(pipetrain)),
                   # wait 10 seconds
                   Pause(10),
                   # release inference job
                   RunFn(lambda *args, **kwargs: release_on_pipe(pipe)),
                   # run_seq automatically join all jobs at the end of the sequence
                   )


def twoinfer_tfmps(argv):
    # type: (Sequence[str]) -> None
    batch_sizes = [int(v) for v in argv]

    if not batch_sizes:
        batch_sizes = [1, 2, 4, 8]

    for idx, bs in enumerate(batch_sizes):
        with tempfile.TemporaryDirectory() as td:
            # create the foreground inference job
            wl1, pipe1 = create_infer(Executor.TF, bs, td)
            wl1.extra_args += ['--min_mem']
            # create the foreground inference job
            wl2, pipe2 = create_infer(Executor.TF, bs, td)
            wl2.extra_args += ['--min_mem']

            run_tf(FLAGS.save_dir / "twoinfer" / "tfmps",
                   wl1,  # start the background job
                   wl2,  # start the foreground job
                   # wait for both jobs to be ready
                   RunFn(lambda *args, **kwargs: wait_on_pipe(pipe1)),
                   RunFn(lambda *args, **kwargs: wait_on_pipe(pipe2)),
                   # start train job
                   RunFn(lambda *args, **kwargs: release_on_pipe(pipe1)),
                   # release inference job
                   RunFn(lambda *args, **kwargs: release_on_pipe(pipe2)),
                   # run_seq automatically join all jobs at the end of the sequence
                   )


def twoinfer(argv):
    # type: (Sequence[str]) -> None
    base_cfg = maybe_forced_preset(presets.MostEfficient)

    sm_factors = [float(v) for v in argv]
    if not sm_factors:
        sm_factors = [1.0, 1.5, 2.0, 2.5, 3.0]

    for idx, factor in enumerate(sm_factors):
        scfg = base_cfg.copy(output_dir=FLAGS.save_dir / "twoinfer" / "salus" / f"{factor:.2f}")
        scfg.extra_args += [
            '--sm-factor', f'{factor:.2f}'
        ]
        with tempfile.TemporaryDirectory() as td:
            # create the foreground inference job
            wl1, pipe1 = create_infer(Executor.Salus, 10, td)

            # create the foreground inference job
            wl2, pipe2 = create_infer(Executor.Salus, 10, td)

            run_seq(scfg,
                    wl1,  # start the first job
                    wl2,  # start the second job
                    # wait for both jobs to be ready
                    RunFn(lambda *args, **kwargs: wait_on_pipe(pipe1)),
                    RunFn(lambda *args, **kwargs: wait_on_pipe(pipe2)),
                    # start 1st job
                    RunFn(lambda *args, **kwargs: release_on_pipe(pipe1)),
                    # release 2nd job
                    RunFn(lambda *args, **kwargs: release_on_pipe(pipe2)),
                    # run_seq automatically join all jobs at the end of the sequence
                    )


def twoinfer_pri(argv):
    # type: (Sequence[str]) -> None
    """Two inferences with difference priority"""
    base_cfg = maybe_forced_preset(presets.MostEfficient)

    sm_factors = [float(v) for v in argv]
    if not sm_factors:
        sm_factors = [1.0, 1.5, 2.0, 2.5, 3.0]

    for idx, factor in enumerate(sm_factors):
        scfg = base_cfg.copy(output_dir=FLAGS.save_dir / "twoinfer_pri" / "salus" / f"{factor:.2f}")
        scfg.extra_args += [
            '--sm-factor', f'{factor:.2f}'
        ]
        with tempfile.TemporaryDirectory() as td:
            # create the foreground inference job
            wl1, pipe1 = create_infer(Executor.Salus, 10, td)

            # create the background inference job
            wl2, pipe2 = create_infer(Executor.Salus, 10, td)
            wl2.extra_args += [
                '--eval_sched_priority', '20'
            ]

            run_seq(scfg,
                    wl1,  # start the first job
                    wl2,  # start the second job
                    # wait for both jobs to be ready
                    RunFn(lambda *args, **kwargs: wait_on_pipe(pipe1)),
                    RunFn(lambda *args, **kwargs: wait_on_pipe(pipe2)),
                    # start 1st job
                    RunFn(lambda *args, **kwargs: release_on_pipe(pipe1)),
                    # release 2nd job
                    RunFn(lambda *args, **kwargs: release_on_pipe(pipe2)),
                    # run_seq automatically join all jobs at the end of the sequence
                    )


@case_switch_main
def main():
    return alone, salus, tfmps, twoinfer, twoinfer_tfmps, inverse_salus, same_pri_salus, twoinfer_pri
