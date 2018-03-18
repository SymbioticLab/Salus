# -*- coding: future_fstrings -*-
"""
OSDI Experiment 10

Have jobs of 5 different JCTs, let them enter the system in order, creating stair shaped memory usage figure.

Scheduler: fair
Work conservation: True
Collected data: memory usage over time
"""
from __future__ import absolute_import, print_function, division

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd, Pause


def main(argv):
    scfg = presets.AllocProf
    if argv:
        run_seq(scfg.copy(output_dir="templogs"),
                *parse_actions_from_cmd(argv))
        return

    run_seq(scfg.copy(output_dir="templogs/exp10"),
            WTL.create("resnet50", 50, 265),
            Pause(10),
            WTL.create("resnet50", 50, 180),
            Pause(10),
            WTL.create("resnet50", 50, 170),
            Pause(10),
            WTL.create("resnet50", 50, 100),
            Pause(10),
            WTL.create("resnet50", 50, 80),
            )
