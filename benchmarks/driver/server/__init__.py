from __future__ import absolute_import, print_function, division
from builtins import super, str

import os
import time
import shutil
from absl import flags
from typing import List
from contextlib import contextmanager

from .config import SalusConfig
from ..utils.compatiblity import DEVNULL
from ..utils import Popen, remove_prefix
from benchmarks.driver.utils import ServerError, execute, kill_tree, kill_hard

flags.DEFINE_string('server_endpoint', 'zrpc://tcp://localhost:5501', 'Salus server endpoint to listen on')
FLAGS = flags.FLAGS


class SalusServer(object):
    default_endpoint = FLAGS.server_endpoint

    def __init__(self, cfg):
        # type: (SalusConfig) -> None
        super().__init__()

        self.config = cfg
        self.env = os.environ.copy()
        self.env['CUDA_VISIBLE_DEVICES'] = '2,3'
        self.env['TF_CPP_MIN_LOG_LEVEL'] = '4'

        # normalize build_dir before doing any path finding
        self.config.build_dir = os.path.abspath(cfg.build_dir)
        self._build_cmd()

        self.proc = None  # type: Popen

    def _find_executable(self):
        # type: () -> str
        """Find the absolute path to server executable, according to 'config.build_type'"""
        candidates = [
            os.path.join(self.config.build_dir, self.config.build_type, 'src', 'executor'),
            os.path.join(self.config.build_dir, self.config.build_type, 'bin', 'executor'),
            os.path.join(self.config.build_dir, self.config.build_type, 'bin', 'salus-server'),
            os.path.join(self.config.build_dir, self.config.build_type.lower(), 'src', 'executor'),
            os.path.join(self.config.build_dir, self.config.build_type.lower(), 'bin', 'executor'),
            os.path.join(self.config.build_dir, self.config.build_type.lower(), 'bin', 'salus-server'),
        ]
        for path in candidates:
            if os.access(path, os.X_OK):
                return path

    def _find_logconf(self):
        # type: () -> str
        """Find the absolute path to the logconf file specified in 'config.logconf'.

        First try to use 'config.logconf_dir' is specified.
        Second try walk up and find project dir
        """
        logconf_dir = self.config.logconf_dir
        if logconf_dir is None:
            # HACK: walk up and find project root dir and then guess log config dir
            project_dir = self.config.build_dir
            while not os.path.exists(os.path.join(project_dir, 'README.md')):
                pardir = os.path.dirname(project_dir)
                if pardir == project_dir:
                    raise ServerError('Cannot find logconf dir')
                project_dir = os.path.dirname(project_dir)
            logconf_dir = os.path.join(project_dir, 'scripts', 'logconf')

        if not os.path.isdir(logconf_dir):
            raise ServerError('Logconf dir does not exist: {}', logconf_dir)

        logconf = os.path.join(logconf_dir, self.config.logconf) + '.config'
        if not os.path.exists(logconf):
            raise ServerError("Requested logconf `{}'does not exist in logconf_dir: {}", self.config.logconf,
                              self.config.logconf_dir)
        return logconf

    def _build_cmd(self):
        # type: () -> List[str]
        """Build commandline using 'config' information"""
        self.args = []

        if self.config.use_nvprof:
            self.args += [
                'nvprof',
                '--export-profile', os.path.join(self.config.output_dir, 'profile.sqlite'),
                '-f',
                '--metrics', 'executed_ipc',
                '--'
            ]

        self.args += [
            self._find_executable(),
            '--listen', remove_prefix(SalusServer.default_endpoint, 'zrpc://'),
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
        outputfiles = ['/tmp/server.output', '/tmp/perf.output', '/tmp/alloc.output']
        stdout = DEVNULL if self.config.hide_output else None
        stderr = DEVNULL if self.config.hide_output else None
        try:
            # remove any existing output
            for f in outputfiles:
                os.unlink(f)

            # start
            self.proc = execute(self.args, env=self.env, stdin=DEVNULL, stdout=stdout, stderr=stderr)
            # wait for a while for the server to be ready
            # FUTURE: make the server write a pid file when it's ready
            time.sleep(5)

            yield self.proc

            # move back server log files
            for f in ['/tmp/server.output', '/tmp/perf.output', '/tmp/alloc.output']:
                if os.path.exists(f):
                    shutil.move(f, self.config.output_dir)
        finally:
            self.kill()

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
