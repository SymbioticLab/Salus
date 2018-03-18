# -*- coding: future_fstrings -*-
"""
OSDI Experiment 11

Run 2 jobs together.

Scheduler: preempt
Work conservation: True
Collected data: Numer of scheduled tasks over time
"""
from __future__ import absolute_import, print_function, division

from benchmarks.driver.server import SalusConfig
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd, Pause


def main(argv):
    scfg = presets.Profiling.copy()  # type: SalusConfig
    scfg.scheduler = 'preempt'
    scfg.disable_wc = False

    if argv:
        run_seq(scfg.copy(output_dir="templogs"),
                *parse_actions_from_cmd(argv))
        return

    run_seq(scfg.copy(output_dir="templogs/exp11"),
            WTL.create("inception3", 25, 303),
            Pause(60),
            WTL.create("alexnet", 100, 506),
            )
