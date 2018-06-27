# -*- coding: future_fstrings -*-
"""
Collect data for optracing

Using alexnet_25 for 1 iteration

Scheduler: pack
Work conservation: True
Collected data: JCT
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL, Executor
from benchmarks.exps import run_seq, maybe_forced_preset, run_tf


FLAGS = flags.FLAGS


def main(argv):
    scfg = maybe_forced_preset(presets.OpTracing)

    name, bs, bn = 'resnet50', 50, 10
    if len(argv) > 0:
        name = argv[0]
    if len(argv) > 1:
        bs = int(argv[1])
    if len(argv) > 2:
        bn = int(argv[2])

    def create_wl(ex):
        return WTL.create(name, bs, bn, executor=ex)

    for con in range(1, 6):
        run_seq(scfg.copy(output_dir=FLAGS.save_dir/"salus"/str(con)),
                *[create_wl(Executor.Salus) for _ in range(con)])

