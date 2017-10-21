#! /usr/bin/env python
import os
from subprocess import Popen
import csv
import argparse
from operator import attrgetter

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport


class Workload(object):
    """Workload"""
    def __init__(self, d, config):
        self.name = d['name']
        self.jct = float(d['jct'])
        self.mem = float(d['mem'])
        self.cmd = d['cmd']

        self.env = os.environ.copy()
        self.env['EXEC_ITER_NUMBER'] = d['ENV']
        self.env['TF_CPP_MIN_LOG_LEVEL'] = '4'
        self.env['CUDA_VISIBLE_DEVICES'] = '0,1'

        self.outputpath = os.path.join(config.save_dir, self.name)
        self.outputfile = None
        self.proc = None

    def runAsync(self):
        self.outputfile = open(self.outputpath, 'w')
        self.proc = Popen(self.cmd, env=self.env, stdout=self.outputfile,
                          stdin=os.devnull, stderr=os.devnull)
        return self.proc

    def wait(self):
        if self.proc:
            self.proc.wait()
        if self.outputfile:
            self.outputfile.close()


def load_workloads(config):
    workloads = []
    with open(config.workloads, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)
            workloads.append(Workload({
                'name': row[0],
                'jct': row[1],
                'mem': row[2],
                'env': row[3],
                'cmd': row[4]
            }, config))
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
    env['TF_CPP_MIN_LOG_LEVEL'] = 4
    serverP = Popen([
        'src/executor',
        '--logconf', '../disable.config'
    ], env=env, stderr=os.devnull, stdin=os.devnull, stdout=os.devnull)
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
    parser.add_argument('--save_dir', help='Output directory', default='output')
    parser.add_argument('workloads', help='Path to the CSV containing workload info')
    parser.add_argument('case', help='Which case to run', choices=casekey.keys())

    config = parser.parse_args()

    Path(config.save_dir).mkdir(exist_ok=True)
    workloads = load_workloads(config)

    run(workloads, config.case)


if __name__ == '__main__':
    main()
