# -*- coding: future_fstrings -*-
"""
For debugging. Collect memory requests for card#188.
The data will then be used to determine suitable ratio for two portion allocator.

Scheduler: fair
Work conservation: True
Collected data: memory usage over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd, maybe_forced_preset, Pause


FLAGS = flags.FLAGS


def plotrun(scfg):
    wls = [
        ('inception3', 100, 5),
        ("resnet50", 50, 5),
        ("resnet152", 75, 5),
    ]
    for name, bs, bn in wls:
        run_seq(scfg.copy(output_dir=FLAGS.save_dir / "plotrun" / "{}_{}".format(name, bs)),
                WTL.create(name, bs, bn))


def single(scfg):
    run_seq(scfg.copy(output_dir=FLAGS.save_dir),
            WTL.create("inception3", 100, 165),
            Pause.Manual,
            WTL.create("resnet50", 50, 798),
            Pause.Manual,
            WTL.create("resnet152", 75, 19))


def all3(scfg):
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/"all3"),
            WTL.create("inception3", 100, 165),
            WTL.create("resnet50", 50, 798),
            WTL.create("resnet152", 75, 19))


def main(argv):
    scfg = maybe_forced_preset(presets.AllocProf)
    command = argv[0] if argv else "single"

    {
        "plotrun": plotrun,
        "single": single,
        "all3": all3,
    }[command](scfg)

