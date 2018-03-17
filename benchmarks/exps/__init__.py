# -*- coding: future_fstrings -*-
from __future__ import absolute_import, print_function, division

import time
import psutil
from typing import Union, Iterable, List
from absl import app

from benchmarks.driver.server import SalusServer
from benchmarks.driver.server.config import presets as presets
from benchmarks.driver.utils import atomic_directory, try_with_default
from benchmarks.driver.utils.compatiblity import Path
from benchmarks.driver.workload import Workload, WTL


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


def run_seq(output_dir, *actions):
    # type: (str, *TAction) -> None
    """Run a sequence of actions"""
    workloads = []

    with atomic_directory(output_dir) as output_dir:  # type: Path
        # start server
        scfg = presets.Verbose.copy(output_dir=output_dir)
        ss = SalusServer(scfg)
        with ss.run():
            # Do action specified in seq
            for act in actions:
                ss.check()

                if isinstance(act, Workload):
                    output_file = output_dir / f'{act.output_name}.{act.batch_num}iter.{len(workloads)}.output'
                    act.run(output_file)
                    workloads.append(act)
                elif isinstance(act, Pause):
                    act.run(workloads)
                else:
                    raise ValueError(f"Unexpected value `{act}' passed to run_seq")


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


