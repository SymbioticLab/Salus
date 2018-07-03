# -*- coding: future_fstrings -*-
"""
For debugging. Run one large job repeatly, to see if the GPU memory can be completely freed between sessions.

Scheduler: fair
Work conservation: True
Collected data: memory usage over time
"""
from __future__ import absolute_import, print_function, division, unicode_literals

from absl import flags

from benchmarks.driver.server.config import presets
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd, maybe_forced_preset, Pause

from itertools import chain


FLAGS = flags.FLAGS


def main(argv):
    scfg = maybe_forced_preset(presets.AllocProf)
    if argv:
        run_seq(scfg.copy(output_dir=FLAGS.save_dir),
                *parse_actions_from_cmd(argv))
        return

    def create_wl():
        return WTL.create("inception4", 25, 200)

    seq = [
        create_wl(),
        create_wl(),
        create_wl(),
        Pause.Wait,
        create_wl(),
        Pause.Wait,
        create_wl(),
        create_wl(),
        Pause.Wait,
        create_wl(),
        create_wl(),
        create_wl(),
        Pause.Wait,
    ]

    run_seq(scfg.copy(output_dir=FLAGS.save_dir), *seq)
