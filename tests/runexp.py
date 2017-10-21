#! /usr/bin/env python
import os
from subprocess import Popen
import csv
import argparse
from operator import attrgetter
import shlex

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # pythontry:

try:
    from subprocess import DEVNULL  # py3k
except ImportError:
    DEVNULL = open(os.devnull, 'wb')  # 2 backport


class Workload(object):
    """Workload"""
    def __init__(self, d, config):
        self.name = d['name']
        self.jct = float(d['jct'])
        self.mem = float(d['mem'])
        self.cmd = shlex.split(d['cmd'])

        self.env = os.environ.copy()
        self.env['EXEC_ITER_NUMBER'] = d['env']
        self.env['TF_CPP_MIN_LOG_LEVEL'] = '4'
        self.env['CUDA_VISIBLE_DEVICES'] = '0,1'

        self.outputpath = os.path.join(config.save_dir, self.name)
        self.outputfile = None
        self.proc = None

    def runAsync(self):
        self.outputfile = open(self.outputpath, 'w')
        self.proc = Popen(self.cmd, env=self.env, stdout=self.outputfile,
                          stdin=DEVNULL, stderr=DEVNULL)
        return self.proc

    def wait(self):
        if self.proc:
            self.proc.wait()
        if self.outputfile:
            self.outputfile.close()


def load_workloads(config):
    workloads = []
    with open(config.workloads, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            workloads.append(Workload({
                'name': row[0],
                'jct': row[1],
                'mem': row[2],
                'env': row[3],
                'cmd': row[4]
            }, config))
    if config.workload_limit > 0 and len(workloads) > config.workload_limit:
        workloads = workloads[:config.workload_limit]
    return workloads


casekey = {
    'shortest': ('jct', False),
    'longest': ('jct', True),
    'smallest': ('mem', False),
    'largest': ('mem', True),
}


def runServer(config):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '2,3'
    env['TF_CPP_MIN_LOG_LEVEL'] = '4'

    stdout = DEVNULL if config.hide_server_output else None
    stderr = DEVNULL if config.hide_server_output else None
    build_dir = os.path.abspath(config.build_dir)

    serverP = Popen([
        os.path.join(build_dir, 'Release', 'src', 'executor'),
        '--logconf',
        os.path.join(build_dir, config.server_log_config)
    ], env=env, stdin=DEVNULL, stdout=stdout, stderr=stderr)
    return serverP


def run(workloads, config):
    if config.case not in casekey:
        raise ValueError('Case should be one of ' + str(casekey.keys()))

    key, desc = casekey[config.case]
    key = attrgetter(key)
    torun = sorted(workloads, key=key, reverse=desc)

    serverP = runServer(config)

    for w in torun:
        w.runAsync()

    for w in torun:
        w.wait()

    serverP.terminate()
    serverP.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hide_server_output', help='Hide server output', default=False, action='store_true')
    parser.add_argument('--server_log_config', help='Log configuration to use for executor server', default='disable.config')
    parser.add_argument('--build_dir', help='Build directory', default='../build')
    parser.add_argument('--save_dir', help='Output directory, default to the same name as case')
    parser.add_argument('--workload_limit', help='Only run this number of workloads. If 0, means no limit', type=int, default=0)
    parser.add_argument('workloads', help='Path to the CSV containing workload info')
    parser.add_argument('case', help='Which case to run', choices=casekey.keys())

    config = parser.parse_args()
    if config.save_dir is None:
        config.save_dir = config.case

    Path(config.save_dir).mkdir(exist_ok=True)
    workloads = load_workloads(config)

    run(workloads, config)


if __name__ == '__main__':
    main()
