# -*- coding: future_fstrings -*-
from __future__ import absolute_import, print_function, division, unicode_literals
from builtins import super, str

import os
import logging
from collections import deque
import time
import psutil
import textwrap
from datetime import datetime
from absl import flags
from contextlib import contextmanager
from typing import List, Deque, Dict, Union

from benchmarks.driver.utils import prompt, remove_prefix
from benchmarks.driver.utils.prompt import pause
from .utils.compatiblity import pathlib, subprocess as sp
from .utils import Popen, execute, ServerError, kill_tree, kill_hard


Path = pathlib.Path
FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)
flags.DEFINE_string('tfserver_endpoint', 'grpc://localhost:2345', 'TF server endpoint to listen on')


class TFDistServer(object):

    def __init__(self, env=None, outputdir=None):
        # type: (Dict, Union[Path, str]) -> TFDistServer
        super().__init__()

        self.env = os.environ.copy()
        if env is not None:
            self.env.update(env)
        if 'CUDA_VISIBLE_DEVICES' not in self.env:
            self.env['CUDA_VISIBLE_DEVICES'] = '0'
        if 'TF_CPP_MIN_LOG_LEVEL' not in self.env:
            self.env['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.endpoint = FLAGS.tfserver_endpoint  # type: str

        self.output = None
        if outputdir is not None:
            self.output = Path(outputdir)

        self._build_cmd()

        self.proc = None  # type: Popen

    def _build_cmd(self):
        # type: () -> List[str]
        """Build commandline using 'config' information"""
        self.args = []

        self.args += [
            'python',
        ]

        return self.args

    @contextmanager
    def run(self):
        # type: () -> None
        """Run server"""
        if self.output:
            captured_stdout_path = self.output/'tfdist.stdout'
            captured_stderr_path = self.output/'tfdist.stderr'

            captured_stdout_path.parent.mkdir(exist_ok=True)
            captured_stderr_path.parent.mkdir(exist_ok=True)

            stdout, stderr = captured_stdout_path.open('w'), captured_stderr_path.open('w')
        else:
            stdout, stderr = None, None

        # noinspection PyBroadException
        try:
            pyscript = textwrap.dedent(f"""
            import tensorflow as tf
            cluster = tf.train.ClusterSpec({{"tfworker": ["{remove_prefix(self.endpoint, "grpc://")}"]}})
            tf.train.Server(cluster, job_name="tfworker", task_index=0,
                            config=tf.ConfigProto(isolate_session_state=True)).join()
            """)
            # start
            self.proc = execute(self.args, env=self.env, stdin=sp.PIPE, stdout=stdout, stderr=stderr)
            self.proc.stdin.write(pyscript + "\n")
            self.proc.stdin.close()  # Ensures the process knows nothing else is coming

            time.sleep(2)

            logger.info(f'Started tf server with pid: {self.proc.pid}')

            # make self the current server
            with self.as_current():
                yield
        except Exception as ex:
            logger.error(f'Got exception while running the tf server: {ex!s}')
        finally:
            self.kill()

            if self.output:
                stdout.close()
                stderr.close()

    _current = deque()  # type: Deque[TFDistServer]

    @contextmanager
    def as_current(self):
        TFDistServer._current.append(self)
        yield self
        TFDistServer._current.pop()

    @classmethod
    def has_current(cls):
        # type: () -> bool
        return len(cls._current) > 0

    @classmethod
    def current_server(cls):
        # type: () -> TFDistServer
        try:
            return cls._current[-1]
        except IndexError:
            raise ServerError('No current running tf server')

    def check(self):
        # type: () -> None
        """Check that the server is healthy and running"""
        if self.proc is None:
            raise ServerError('TF Server is not yet started')
        if self.proc.poll() is not None:
            out, err = self.proc.communicate()
            msg = [f'TF Server died unexpectedly with return code: {self.proc.returncode}']
            if out is not None:
                msg.append(f'\nStandard output:\n{out}')
            if err is not None:
                msg.append(f'\nStandard error:\n{err}')
            raise ServerError('\n'.join(msg))

    def kill(self):
        # type: () -> None
        """Kill the server"""
        if FLAGS.no_server:
            return

        if self.proc is None or self.proc.poll() is not None:
            logger.warning('TF Server already died or is not yet started')
            self.proc = None
            return

        logger.info(f'Killing TF server with pid: {self.proc.pid}')
        _, alive = kill_tree(self.proc, timeout=2)
        if alive:
            prompt.confirm('TF Server did not respond in time, do you want to kill hard?')
            logger.info(f'Force killing server with pid: {self.proc.pid}')
            kill_hard(alive)

        self.proc = None

    @classmethod
    def wait_workloads(cls, workloads, timeout=None, callback=None):
        """Wait workloads, raise if server died"""
        if callback is None:
            def done(proc):
                logger.info(f'Workload {proc.workload.canonical_name} exited with {proc.returncode}')

            callback = done

        gone = []
        alive = [w.proc for w in workloads]
        enter = datetime.now()
        while alive:
            if TFDistServer.has_current():
                TFDistServer.current_server().check()

            g, alive = psutil.wait_procs(alive, timeout=.25, callback=callback)
            gone += g

            if timeout is not None and (datetime.now() - enter).total_seconds() >= timeout:
                break

        return [p.workload for p in gone], [p.workload for p in alive]

