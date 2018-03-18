# -*- coding: future_fstrings -*-
from __future__ import absolute_import, print_function, division

import time
import psutil
import re
from typing import Union, Iterable, List
from absl import app

from benchmarks.driver.server import SalusServer, SalusConfig
from benchmarks.driver.utils import atomic_directory, try_with_default
from benchmarks.driver.utils.compatiblity import Path
from benchmarks.driver.workload import Workload, WTL, ResourceGeometry


class Pause(int):
    """Represent a pause in an action sequence"""

    def run(self, workloads):
        if self == Pause.Manual:
            try_with_default(input, ignore=SyntaxError)('Press enter to continue...')
        elif self == Pause.Wait:
            psutil.wait_procs(w.proc for w in workloads)
        else:
            time.sleep(self)

    """Pause execution and wait for human input"""
    Manual = -1

    """Wait until previous workloads finishes"""
    Wait = -2


TAction = Union[Pause, Workload]


def run_seq(scfg, *actions):
    # type: (SalusConfig, *TAction) -> List[Workload]
    """Run a sequence of actions"""
    workloads = []  # type: List[Workload]

    with atomic_directory(scfg.output_dir) as temp_dir:  # type: Path
        # start server
        ss = SalusServer(scfg.copy(output_dir=temp_dir))
        with ss.run():
            # Do action specified in seq
            for act in actions:
                ss.check()

                if isinstance(act, Workload):
                    output_file = temp_dir / f'{act.output_name}.{act.batch_num}iter.{len(workloads)}.output'
                    act.run(output_file)
                    workloads.append(act)
                elif isinstance(act, Pause):
                    act.run(workloads)
                else:
                    raise ValueError(f"Unexpected value `{act}' passed to run_seq")

    psutil.wait_procs(w.proc for w in workloads)
    # fix workload output_file path
    for w in workloads:
        w.output_file = scfg.output_dir / w.output_file.name
    return workloads


def parse_actions_from_cmd(argv):
    # type: (Iterable[str]) -> List[TAction]
    """Parse actions from command line
    E.g.
    manual inception3 75(batch_size) 128(batch_num) 5(wait) manual
    """
    actions = []
    while argv:
        curr = argv[0]
        if curr == 'manual':
            actions.append(Pause.Manual)
            argv.pop(0)
            continue
        try:
            name, batch_size, batch_num, pause, *argv = argv
            actions.append(WTL.create(name, batch_size, batch_num))
        except ValueError:
            raise app.UsageError(f'Unexpected sequence of arguments: {argv}')

    return actions


def update_jct(workload, update_global=False):
    # type: (Workload, bool) -> None
    """Parse and update JCT value of a completed workload"""
    if workload.proc is None or workload.proc.returncode != 0:
        raise ValueError(f'Workload {workload.name} not started or terminated in error')

    ptn = re.compile('^JCT: ')
    with open(workload.output_file) as f:
        for line in f.readline():
            if ptn.match(line):
                try:
                    jct = float(line.split(' ')[1])
                    workload.geometry.jct = jct
                    if update_global:
                        WTL.from_name(workload.name).add_geometry(workload.rcfg, workload.executor,
                                                                  ResourceGeometry(jct=jct))
                    return
                except (ValueError, IndexError):
                    continue

    raise ValueError(f'No JCT info found in output file {workload.output_file}')
