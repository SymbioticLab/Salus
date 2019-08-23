'''
Created Date: Friday, August 23rd 2019, 9:18:04 pm
Author: Yuchen Zhong
Email: yczhong@hku.hk
'''

from __future__ import absolute_import, print_function, division, unicode_literals

import os
from absl import flags

from benchmarks.driver.runner import Executor
from benchmarks.driver.server.config import presets
from benchmarks.driver.utils import execute
from benchmarks.driver.workload import WTL
from benchmarks.exps import run_seq, maybe_forced_preset, Pause, run_tf, run_tfdist


FLAGS = flags.FLAGS

def case(policy):
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = policy
    if policy == "mix":
        os.environ["SALUS_TIMEOUT"] = "666666"
    else:
        os.environ["SALUS_TIMEOUT"] = "0"

    folder_name = policy
    workload_list = [
        WTL.create("alexnet", 25, 250), # training
        WTL.create("resnet50eval", 1, 2000),
        WTL.create("resnet50eval", 1, 2000),
        WTL.create("resnet50eval", 1, 2000)
    ]
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/folder_name),
            *workload_list
            )



def main(argv):
    policy_list = [
        "pack",
        "fair",
        "preempt",
        "mix"
    ]
    for policy in policy_list:
        case(policy)
