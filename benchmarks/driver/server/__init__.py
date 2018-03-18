# -*- coding: future_fstrings -*-
from __future__ import absolute_import, print_function, division
from builtins import super, str

import os
import time
from absl import flags
from typing import List, Deque
from contextlib import contextmanager
from collections import deque

from .config import SalusConfig
from ..utils.compatiblity import DEVNULL, Path
from ..utils import ServerError, Popen, execute, kill_tree, kill_hard, remove_prefix

flags.DEFINE_string('server_endpoint', 'zrpc://tcp://localhost:5501', 'Salus server endpoint to listen on')
FLAGS = flags.FLAGS


class SalusServer(object):

    def __init__(self, cfg):
        # type: (SalusConfig) -> None
        super().__init__()

        self.config = cfg
        self.env = os.environ.copy()
        self.env['CUDA_VISIBLE_DEVICES'] = '2,3'
        self.env['TF_CPP_MIN_LOG_LEVEL'] = '4'

        self.endpoint = FLAGS.server_endpoint  # type: str

        # normalize build_dir before doing any path finding
        self.config.build_dir = cfg.build_dir.resolve()
        self._build_cmd()

        self.proc = None  # type: Popen

    def _find_executable(self):
        # type: () -> str
        """Find the absolute path to server executable, according to 'config.build_type'"""
        candidates = [
            self.config.build_dir / self.config.build_type / 'src' / 'executor',
            self.config.build_dir / self.config.build_type / 'bin' / 'executor',
            self.config.build_dir / self.config.build_type / 'bin' / 'salus-server',
            self.config.build_dir / self.config.build_type.lower() / 'src' / 'executor',
            self.config.build_dir / self.config.build_type.lower() / 'bin' / 'executor',
            self.config.build_dir / self.config.build_type.lower() / 'bin' / 'salus-server',
        ]
        for path in candidates:
            if os.access(str(path), os.X_OK):
                return str(path)
        raise ServerError(f'Cannot find server executable, examined candidates are: {candidates}')

    def _find_logconf(self):
        # type: () -> str
        """Find the absolute path to the logconf file specified in 'config.logconf'.

        First try to use 'config.logconf_dir' is specified.
        Second try walk up and find project dir
        """
        logconf_dir = self.config.logconf_dir
        if logconf_dir is None:
            for p in self.config.build_dir.parents:
                if not (p / 'README.md').exists():
                    continue
                logconf_dir = p / 'scripts' / 'logconf'
            if logconf_dir is None:
                raise ServerError('Cannot find logconf dir')

        if not logconf_dir.is_dir():
            raise ServerError(f'Logconf dir does not exist: {logconf_dir}')

        logconf = logconf_dir / self.config.logconf + '.config'  # type: Path
        if not logconf.exists():
            raise ServerError(f"Requested logconf `{self.config.logconf}'does not exist in logconf_dir: {logconf_dir}")
        return str(logconf)

    def _build_cmd(self):
        # type: () -> List[str]
        """Build commandline using 'config' information"""
        self.args = []

        if self.config.use_nvprof:
            self.args += [
                'nvprof',
                '--export-profile', str(self.config.output_dir / 'profile.sqlite'),
                '-f',
                '--metrics', 'executed_ipc',
                '--'
            ]

        self.args += [
            self._find_executable(),
            '--listen', remove_prefix(self.endpoint, 'zrpc://'),
            '--logconf', self._find_logconf(),
            '--sched', self.config.scheduler,
        ]
        self.args += self.config.extra_args

        if self.config.disable_adc:
            self.args.append('--disable-adc')
        if self.config.disable_wc:
            self.args.append('--disable-wc')

        return self.args

    @contextmanager
    def run(self):
        # type: () -> Popen
        """Run server"""
        outputfiles = [Path(p) for p in ['/tmp/server.output', '/tmp/perf.output', '/tmp/alloc.output']]
        stdout = DEVNULL if self.config.hide_output else None
        stderr = DEVNULL if self.config.hide_output else None
        try:
            # remove any existing output
            for f in outputfiles:
                f.unlink()

            # start
            self.proc = execute(self.args, env=self.env, stdin=DEVNULL, stdout=stdout, stderr=stderr)
            # wait for a while for the server to be ready
            # FUTURE: make the server write a pid file when it's ready
            time.sleep(5)

            # make self the current server
            with self.as_current():
                yield self.proc

            # move back server log files
            for f in outputfiles:
                if f.exists():
                    f.rename(self.config.output_dir / f.name)
        finally:
            self.kill()

    _current = deque()  # type: Deque[SalusServer]

    @contextmanager
    def as_current(self):
        SalusServer._current.append(self)
        yield self
        SalusServer._current.pop()

    @classmethod
    def current_server(cls):
        # type: () -> SalusServer
        try:
            return cls._current[-1]
        except IndexError:
            raise ServerError('No current running server')

    def check(self):
        # type: () -> None
        """Check that the server is healthy and running"""
        if self.proc is None:
            raise ServerError('Server not yet started')
        if self.proc.poll() is not None:
            raise ServerError('Server died unexpectedly with return code: {}', self.proc.returncode)

    def kill(self):
        # type: () -> None
        """Kill the server"""
        self.check()

        _, alive = kill_tree(self.proc)
        if alive:
            kill_hard(alive)

        self.proc = None
