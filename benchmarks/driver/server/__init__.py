# -*- coding: future_fstrings -*-
#
# Copyright 2019 Peifeng Yu <peifeng@umich.edu>
#
# This file is part of Salus
# (see https://github.com/SymbioticLab/Salus).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import absolute_import, print_function, division, unicode_literals
from builtins import super, str

import os
import time
import shutil
import shlex
import logging
import psutil
from datetime import datetime
from absl import flags
from typing import List, Deque
from contextlib import contextmanager
from collections import deque

from .config import SalusConfig
from ..utils.compatiblity import pathlib, subprocess as sp
from ..utils import ServerError, Popen, execute, kill_tree, kill_hard, remove_prefix, try_with_default
from ..utils import prompt


Path = pathlib.Path
FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)
flags.DEFINE_string('server_endpoint', 'zrpc://tcp://127.0.0.1:5501', 'Salus server endpoint to listen on')
flags.DEFINE_boolean('no_server', False, "Don't start Salus server, just print out the command and wait for the user")
flags.DEFINE_string('server_args', '', 'Extra arguments to Salus server')
flags.DEFINE_string('server_save_output', '', 'Capture and save server output to the given path')


class SalusServer(object):

    def __init__(self, cfg):
        # type: (SalusConfig) -> None
        super().__init__()

        self.config = cfg
        self.env = os.environ.copy()
        self.env.update(cfg.env)
        if 'CUDA_VISIBLE_DEVICES' not in self.env:
            self.env['CUDA_VISIBLE_DEVICES'] = '0'
        if 'TF_CPP_MIN_LOG_LEVEL' not in self.env:
            self.env['TF_CPP_MIN_LOG_LEVEL'] = '1'

        if cfg.save_outerr:
            FLAGS.server_save_output = 'server'

        self.endpoint = FLAGS.server_endpoint  # type: str

        # normalize build_dir before doing any path finding
        self.config.build_dir = cfg.build_dir.resolve(strict=True)
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

        logconf = (logconf_dir / self.config.logconf).with_suffix('.config')
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
                '--export-profile', str(self.config.output_dir / 'profile.prof'),
            ]

        self.args += [
            self._find_executable(),
            '--listen', remove_prefix(self.endpoint, 'zrpc://'),
            '--logconf', self._find_logconf(),
            '--sched', self.config.scheduler,
            '--verbose', "9"
        ]
        self.args += self.config.extra_args

        # handle extra args from cmd
        self.args += shlex.split(FLAGS.server_args)

        if self.config.use_gperf:
            self.env['SALUS_PROFILE'] = '/tmp/gperf.out'
            self.args += ['--gperf']
        else:
            self.env['SALUS_PROFILE'] = ''

        if self.config.disable_adc:
            self.args.append('--disable-adc')
        if self.config.disable_wc:
            self.args.append('--disable-wc')

        return self.args

    @contextmanager
    def run(self):
        # type: () -> None
        """Run server"""
        outputfiles = [Path(p) for p in ['/tmp/server.output',
                                         '/tmp/perf.output',
                                         '/tmp/alloc.output',
                                         '/tmp/gperf.out',
                                         'verbose.log']]
        stdout, stderr = None, None
        if self.config.hide_output:
            stdout, stderr = sp.DEVNULL, sp.DEVNULL
        # remove any existing output
        for f in outputfiles:
            if f.exists():
                f.unlink()

        # assert output_dir exists
        assert(self.config.output_dir.is_dir())

        if FLAGS.server_save_output:
            captured_stdout_path = self.config.output_dir / Path(FLAGS.server_save_output).with_suffix('.stdout')
            captured_stderr_path = self.config.output_dir / Path(FLAGS.server_save_output).with_suffix('.stderr')
            stdout, stderr = captured_stdout_path.open('w'), captured_stderr_path.open('w')

        # noinspection PyBroadException
        try:
            if FLAGS.no_server:
                print('Start server with the following command:')
                prefixs = ['SALUS', 'TF', 'CUDA']
                special_envs = ' '.join(f'{k}="{v}"' for k, v in self.env.items() if any(k.startswith(p) for p in prefixs))
                print('env ' + special_envs + ' ' + ' '.join(self.args))
                prompt.pause()
            else:
                logger.info(f'Starting server with cmd: {self.args}')

                print("self.args: {}".format(self.args))
                print("self.env: {}".format(self.env))
                # start
                self.proc = execute(self.args, env=self.env, stdin=sp.DEVNULL, stdout=stdout, stderr=stderr)

                # wait for a while for the server to be ready
                # FUTURE: make the server write a pid file when it's ready
                time.sleep(5)

                logger.info(f'Started server with pid: {self.proc.pid}')

            # make self the current server
            with self.as_current():
                yield
        except Exception as ex:
            logger.error(f'Got exception while running the server: {ex!s}')
        finally:
            self.kill()

            if FLAGS.server_save_output:
                stdout.close()
                stderr.close()

            # move back server log files
            for f in outputfiles:
                if f.exists():
                    if f.stat().st_size == 0:
                        f.unlink()
                    else:
                        target = self.config.output_dir/f.name
                        if target.exists():
                            target.unlink()
                        shutil.move(str(f), str(target))

    _current = deque()  # type: Deque[SalusServer]

    @contextmanager
    def as_current(self):
        SalusServer._current.append(self)
        yield self
        SalusServer._current.pop()

    @classmethod
    def has_current(cls):
        # type: () -> bool
        return len(cls._current) > 0

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
        if FLAGS.no_server:
            return

        if self.proc is None:
            raise ServerError('Server is not yet started')
        if self.proc.poll() is not None:
            out, err = self.proc.communicate()
            msg = [f'Server died unexpectedly with return code: {self.proc.returncode}']
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
            logger.warning('Server already died or is not yet started')
            self.proc = None
            return

        logger.info(f'Killing server with pid: {self.proc.pid}')
        _, alive = kill_tree(self.proc, timeout=self.config.kill_timeout)
        if alive:
            prompt.confirm('Server did not respond in time, do you want to kill hard?')
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
            if SalusServer.has_current():
                SalusServer.current_server().check()

            g, alive = psutil.wait_procs(alive, timeout=.25, callback=callback)
            gone += g

            if timeout is not None and (datetime.now() - enter).total_seconds() >= timeout:
                break

        return [p.workload for p in gone], [p.workload for p in alive]
