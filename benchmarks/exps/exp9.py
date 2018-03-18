# -*- coding: future_fstrings -*-
"""
OSDI Experiment 9

Run 2 jobs together.

Scheduler: fair
Work conservation: True
Collected data: fairness counter over time
"""
from __future__ import absolute_import, print_function, division

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd


def main(argv):
    scfg = presets.Profiling
    if argv:
        run_seq(scfg.copy(output_dir="templogs"),
                *parse_actions_from_cmd(argv))
        return

    run_seq(scfg.copy(output_dir="templogs/exp9"),
            WTL.create("resnet50", 50, 265),
            WTL.create("resnet50", 50, 265),
            )
