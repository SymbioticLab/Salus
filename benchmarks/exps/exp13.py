# -*- coding: future_fstrings -*-
"""
OSDI Experiment 13

Tasks start at begining
For TF-Salus: JCT of running 2 jobs together, using packing scheduling
For TF: run the same jobs sequentially.

Show packing can improve JCT

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
    run_seq(scfg.copy(output_dir="templogs/exp13/salus"),
            WTL.create("resnet50", 50, 530),
            WTL.create("resnet50", 50, 265),
            )

    # Then run on tf
    run_seq(scfg.copy(output_dir="templogs/exp13/tf"),
            WTL.create("resnet50", 50, 530, executor=Executor.TF),
            Pause.Wait,
            WTL.create("resnet50", 50, 265, executor=Executor.TF),
            )
