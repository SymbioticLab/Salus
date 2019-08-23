'''
Created Date: Thursday, August 22nd 2019, 11:02:32 am
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

def case1():
    scfg = maybe_forced_preset(presets.Debugging)
    scfg.scheduler = 'mix'
    # scfg.env['SALUS_DISABLE_SHARED_LANE'] = '1'
    os.environ["SALUS_TIMEOUT"] = "666666"
    scfg.env["TF_CPP_MIN_LOG_LEVEL"] = ''
    folder_name = "case1"
    workload_list = [
        WTL.create("resnet50eval", 1, 1000),
        Pause(3),
        WTL.create("resnet50eval", 1, 1000)
    ]
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/folder_name),
            *workload_list
            )


def case2():
    scfg = maybe_forced_preset(presets.MostEfficient)
    scfg.scheduler = 'pack'
    scfg.env['SALUS_DISABLE_SHARED_LANE'] = '1'
    os.environ["SALUS_TIMEOUT"] = "0"

    folder_name = "case2"
    workload_list = [
        WTL.create("resnet50eval", 1, 1000),
        Pause(3),
        WTL.create("resnet50eval", 1, 1000)
    ]
    run_seq(scfg.copy(output_dir=FLAGS.save_dir/folder_name),
            *workload_list
            )


def main(argv):
    command = argv[0] if argv else "case1"

    {
        "case1": case1,
        "case2": case2
    }[command]()

