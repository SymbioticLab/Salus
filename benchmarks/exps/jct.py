from __future__ import absolute_import, print_function, division
from builtins import str

from typing import Iterable, Sequence
from benchmarks.driver.workload import WorkloadTemplate


def select_workloads(argv):
    # type: (Iterable[str]) -> Iterable[WorkloadTemplate]
    """Select workloads based on commandline"""
    if not argv:
        return WorkloadTemplate.known_workloads.values()
    else:
        return [
            WorkloadTemplate.known_workloads[name]
            for piece in argv
            for name in piece.split(',')
        ]


def main(argv):
    # type: (Sequence[str]) -> None
    for wl in select_workloads(argv):
        print('Running', wl.name)
    print('Extra argvs:', argv)
