# -*- coding: future_fstrings -*-
from __future__ import absolute_import, print_function, division

from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, parse_actions_from_cmd


def main(argv):
    if argv:
        run_seq("templogs", *parse_actions_from_cmd(argv))
        return

    run_seq("templogs/area_3res_sleep5_1min_nocpu_pef",
            WTL.create("resnet50", 50, 265),
            WTL.create("resnet50", 50, 265),
            WTL.create("resnet50", 50, 265),
            )

    run_seq("templogs/area_3of_sleep5_rand",
            WTL.create("overfeat", 50, 424),
            WTL.create("overfeat", 50, 424),
            WTL.create("overfeat", 50, 424),
            )

    run_seq("templogs/area_mix_inception",
            WTL.create("inception3", 25, 1537),
            WTL.create("inception3", 50, 836),
            WTL.create("inception3", 100, 436),
            WTL.create("inception4", 25, 743),
            )

    run_seq("templogs/area_vgg",
            WTL.create("vgg11", 25, 20),
            WTL.create("vgg11", 50, 20),
            WTL.create("vgg11", 100, 20),
            WTL.create("vgg19", 25, 20),
            WTL.create("vgg19", 50, 20),
            WTL.create("vgg19", 100, 20),
            )
