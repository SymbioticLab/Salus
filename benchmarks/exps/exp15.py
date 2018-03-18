# -*- coding: future_fstrings -*-
"""
OSDI Experiment 15

Have 99 jobs entering the system over the time, using packing scheduling.

Scheduler: packing
Work conservation: True
Collected data: JCT
"""
from __future__ import absolute_import, print_function, division

from benchmarks.driver.server import SalusConfig
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL, Executor
from benchmarks.exps import run_seq, parse_actions_from_cmd, Pause


def main(argv):
    scfg = presets.MostEfficient.copy()  # type: SalusConfig
    scfg.scheduler = 'packing'

    if argv:
        run_seq(scfg.copy(output_dir="templogs"),
                *parse_actions_from_cmd(argv))
        return

    # Firstly run concurrently on salus
    run_seq(scfg.copy(output_dir="templogs/exp15/salus"),
            WTL.create("resnet50", 50, 530),
            WTL.create("resnet50", 50, 265),
            )

    # Then run on tf
    run_seq(scfg.copy(output_dir="templogs/exp15/tf"),
            WTL.create("resnet50", 50, 530, executor=Executor.TF),
            Pause.Wait,
            WTL.create("resnet50", 50, 265, executor=Executor.TF),
            )
