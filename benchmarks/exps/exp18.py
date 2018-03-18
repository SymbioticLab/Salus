# -*- coding: future_fstrings -*-
"""
OSDI Experiment 18

Run 2-20 alexnet networks, until the JCT begins to increase.
At which time show paging activity statistics.

Scheduler: packing
Admission control: False
Work conservation: True
Collected data: JCT
"""
from __future__ import absolute_import, print_function, division

import numpy as np
from absl import flags

from benchmarks.driver.server import SalusConfig
from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd, update_jct

flags.DEFINE_float('break_when', 1.1, 'When JCT is high than this ratio compared to original value, stop')
FLAGS = flags.FLAGS


def report(concurrent, jct, ratio):
    # type: (int, float, float) -> None
    print(f'Parallel=f{concurrent} JCT={jct:.2f} {ratio:.2%}')


def main(argv):
    scfg = presets.MostEfficient.copy()  # type: SalusConfig
    scfg.scheduler = 'packing'
    scfg.disable_adc = True

    if argv:
        run_seq(scfg.copy(output_dir="templogs"),
                *parse_actions_from_cmd(argv))
        return

    def create_wl():
        return WTL.create('alexnet', 25, 2726)

    # check if we have reference JCT
    reference_jct = create_wl().geometry.jct

    if reference_jct is None:
        start_from = 1
    else:
        start_from = 2
        report(1, reference_jct, 1)

    for concurrent in range(start_from, 20):
        # run them at once
        workloads = [create_wl() for _ in range(concurrent)]
        run_seq(scfg.copy(output_dir=f"templogs/exp18/{concurrent}"), *workloads)

        # calculate average jct
        for w in workloads:
            update_jct(w)
        jcts = [w.geometry.jct for w in workloads]
        avgjct = np.mean(jcts)  # type: float
        ratio = avgjct / reference_jct
        report(concurrent, avgjct, ratio)
        if ratio > FLAGS.break_when:
            break

