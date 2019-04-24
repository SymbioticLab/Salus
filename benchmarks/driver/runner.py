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
from future.utils import with_metaclass

import logging
from absl import flags
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from enum import Enum
from typing import Iterable, Tuple, Union, Any, Dict

from .server import SalusServer
from .tfserver import TFDistServer
from .utils import Popen, execute, snake_to_pascal, str2bool, remove_suffix
from .utils.compatiblity import pathlib, subprocess as sp

Path = pathlib.Path
FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)

flags.DEFINE_string('tfbench_base', '../tf_benchmarks', 'Base dir of TFBenchmark based workloads')
flags.DEFINE_string('unit_base', 'tests', 'Base dir of unittest based workloads')
flags.DEFINE_string('fathom_base', '../fathom', 'Base dir of Fathom based workloads')
flags.DEFINE_string('tfweb_base', '../tfweb', 'Base dir of TFWeb based workloads')
flags.DEFINE_string('tfweb_saved_model_dir', '~/../symbiotic/peifeng/tf_cnn_benchmarks_models/saved_models',
                    'SavedModel dir of TFWeb based workloads')
flags.DEFINE_string('tfweb_request_body_dir', '~/../symbiotic/peifeng/tf_cnn_benchmarks_models/reqeusts',
                    'Predefined request body dir for TFWeb based workloads')
flags.DEFINE_boolean('no_capture', False, 'Do not capture workload outputs')


RunConfig = namedtuple('RunConfig', [
    'batch_size',
    'batch_num',
    'cfgname',
])


class Executor(Enum):
    Salus = "salus"
    TF = "tf"
    TFDist = "tfdist"


def enumerate_rcfgs(batch_sizes, batch_nums):
    # type: (Iterable[Union[int, str]], Iterable[int]) -> Iterable[RunConfig]
    """Convenient method to generate a list of RunConfig"""
    return [
        RunConfig(batch_size, batch_num, None)
        for batch_size in batch_sizes
        for batch_num in batch_nums
    ]


class Runner(with_metaclass(ABCMeta, object)):
    """A runner knows how to run a given workload"""
    def __init__(self, wl):
        # type: (Any) -> None
        super().__init__()
        self.wl = wl
        self.env = wl.env.copy()  # type: Dict[str, str]

        def set_default(d, key, defval):
            if key not in d:
                d[key] = defval
            else:
                logger.info(f'Using custom value {key}={d[key]}')

        set_default(self.env, 'CUDA_VISIBLE_DEVICES', '0')
        set_default(self.env, 'TF_CPP_MIN_LOG_LEVEL', '2')

    @abstractmethod
    def __call__(self, executor, output_file):
        # type: (Executor, Path) -> Popen
        pass


class TFBenchmarkRunner(Runner):
    """Run a tf benchmark job"""

    def __init__(self, wl, base_dir=None):
        # type: (Any, Path) -> None
        super().__init__(wl)
        self.base_dir = base_dir
        if self.base_dir is None:
            self.base_dir = Path(FLAGS.tfbench_base)

    def __call__(self, executor, output_file):
        # type: (Executor, Path) -> Popen
        cwd = self.base_dir / 'scripts' / 'tf_cnn_benchmarks'
        cmd = [
            'stdbuf', '-o0', '-e0', '--',
            'python', 'tf_cnn_benchmarks.py',
            '--display_every=1',
            '--num_gpus=1',
            '--variable_update=parameter_server',
            '--nodistortions',
            '--executor={}'.format(executor.value),
            '--num_batches={}'.format(self.wl.batch_num),
            '--batch_size={}'.format(self.wl.batch_size),
        ]
        eval_interval = self.wl.env.pop('SALUS_TFBENCH_EVAL_INTERVAL', '0.1')
        eval_rand_factor = self.wl.env.pop('SALUS_TFBENCH_EVAL_RAND_FACTOR', '5')
        eval_block = self.wl.env.pop('SALUS_TFBENCH_EVAL_BLOCK', 'true')

        eval_model_dir = self.wl.env.pop('SALUS_TFBENCH_EVAL_MODEL_DIR', 'models')
        eval_model_dir = str(Path(eval_model_dir).joinpath(remove_suffix(self.wl.name, 'eval')))

        eval_saved_model_dir = self.wl.env.pop('SALUS_TFBENCH_EVAL_SAVED_MODEL_DIR', None)
        if eval_saved_model_dir is not None:
            eval_saved_model_dir = str(Path(eval_saved_model_dir).joinpath(remove_suffix(self.wl.name, 'eval')))

        num_seconds = self.wl.env.pop('SALUS_ITER_SECONDS', None)
        if num_seconds is not None:
            cmd += [
                '--num_seconds={}'.format(num_seconds)
            ]

        wait_for_signal = self.wl.env.pop('SALUS_WAIT_FOR_SIGNAL', None)
        if wait_for_signal is not None:
            cmd += [
                '--wait_for_signal={}'.format(wait_for_signal)
            ]

        if self.wl.name.endswith('eval'):
            model_name = remove_suffix(self.wl.name, 'eval')
            cmd += [
                '--model_dir=' + eval_model_dir,
                '--model={}'.format(model_name),
                '--eval_interval_secs={}'.format(eval_interval),
                '--eval_interval_random_factor={}'.format(eval_rand_factor),
                '--eval_block={}'.format(eval_block),
                '--eval'
            ]
            if eval_saved_model_dir is not None:
                cmd += [
                    '--saved_model_dir=' + eval_saved_model_dir
                ]
        else:
            cmd += [
                '--model={}'.format(self.wl.name),
            ]
            if str2bool(self.wl.env.pop('SALUS_SAVE_MODEL', '')):
                cmd += [
                    '--model_dir=' + eval_model_dir,
                ]

        cmd += self.wl.extra_args

        if FLAGS.no_capture:
            return execute(cmd, cwd=str(cwd), env=self.env)
        else:
            output_file.parent.mkdir(exist_ok=True, parents=True)
            with output_file.open('w') as f:
                return execute(cmd, cwd=str(cwd), env=self.env, stdout=f, stderr=sp.STDOUT)


class UnittestRunner(Runner):
    """Run a unittest job"""

    def __init__(self, wl, base_dir=None):
        # type: (Any, Path) -> None
        super().__init__(wl)
        self.base_dir = base_dir
        if self.base_dir is None:
            self.base_dir = Path(FLAGS.unit_base)

    def __call__(self, executor, output_file):
        # type: (Executor, Path) -> Popen
        env = self.env.copy()
        env['EXEC_ITER_NUMBER'] = str(self.wl.batch_num)
        env['SALUS_BATCH_SIZE'] = str(self.wl.batch_size)
        if executor == Executor.TFDist:
            env['SALUS_TFDIST_ENDPOINT'] = TFDistServer.current_server().endpoint

        cwd = self.base_dir
        pkg, method = self._construct_test_name(executor)
        cmd = [
            'stdbuf', '-o0', '-e0', '--',
            'python', '-m', pkg, method,
        ]
        cmd += self.wl.extra_args
        if FLAGS.no_capture:
            return execute(cmd, cwd=str(cwd), env=self.env)
        else:
            output_file.parent.mkdir(exist_ok=True, parents=True)
            with output_file.open('w') as f:
                return execute(cmd, cwd=str(cwd), env=env, stdout=f, stderr=sp.STDOUT)

    def _construct_test_name(self, executor):
        # type: (Executor) -> Tuple[str, str]
        """Construct test class and name from RunConfig"""
        supported_model = {
            'seq2seq': ('test_tf.test_seq', 'TestSeqPtb', {
                'small': '0_small',
                'medium': '1_medium',
                'large': '2_large',
            }),
            'mnistsf': ('test_tf.test_mnist_tf', 'TestMnistSoftmax', {
                25: '0', 50: '1', 100: '2'
            }),
            'mnistcv': ('test_tf.test_mnist_tf', 'TestMnistConv', {
                25: '0', 50: '1', 100: '2'
            }),
            'mnistlg': ('test_tf.test_mnist_tf', 'TestMnistLarge', {
                25: '0', 50: '1', 100: '2'
            }),
            'superres': ('test_tf.test_super_res', 'TestSuperRes', {
                32: '0', 64: '1', 128: '2',
                1: '0', 5: '1', 10: '2',
            })
        }

        variable_batch_size_models = {'vae', 'superres'}
        if remove_suffix(self.wl.name, 'eval') not in variable_batch_size_models:
            if self.wl.batch_size not in self.wl.wtl.available_batch_sizes():
                raise ValueError(f"Batch size `{self.wl.batch_size}' is not supported for {self.wl.name},"
                                 f" available ones: {self.wl.wtl.available_batch_sizes()}")

        if executor == Executor.Salus:
            prefix = 'test_rpc_'
        elif executor == Executor.TF:
            prefix = 'test_gpu_'
        elif executor == Executor.TFDist:
            prefix = 'test_distributed_'
        else:
            raise ValueError(f'Unknown executor: {executor}')

        if self.wl.name.endswith('eval'):
            prefix += 'eval_'

        model_name = remove_suffix(self.wl.name, 'eval')

        if model_name in supported_model:
            pkg, cls, names = supported_model[model_name]
        else:
            # fallback to guessing
            pkg = f'test_tf.test_{model_name}'
            cls = f'Test{snake_to_pascal(model_name)}'

            # get method name
            names = {
                s: str(idx)
                for idx, s in enumerate(self.wl.wtl.available_batch_sizes())
            }

        postfix = names.get(self.wl.batch_size, '0')

        method = f'{cls}.{prefix}{postfix}'
        return pkg, method


class FathomRunner(Runner):
    """Run a fathom job"""

    def __init__(self, wl, base_dir=None):
        super().__init__(wl)
        self.base_dir = base_dir
        if self.base_dir is None:
            self.base_dir = FLAGS.fathom_base

    def __call__(self, executor, output_file):
        # type: (Executor, Path) -> Popen
        cwd = self.base_dir
        cmd = [
            'stdbuf', '-o0', '-e0', '--',
            'python', '-m', 'fathom.cli',
            '--workload', remove_suffix(self.wl.name, 'eval'),
            '--action', 'test' if self.wl.name.endswith('eval') else 'train',
            '--num_iters', str(self.wl.batch_num),
            '--batch_size', str(self.wl.batch_size),
        ]
        if executor == Executor.Salus:
            cmd += [
                '--target', SalusServer.current_server().endpoint,
                '--dev', '/gpu:0',
            ]
        elif executor == Executor.TF:
            cmd += [
                '--dev', '/gpu:0',
            ]
        elif executor == Executor.TFDist:
            cmd += [
                '--target', TFDistServer.current_server().endpoint,
                '--dev', '/job:tfworker/gpu:0',
            ]
        else:
            raise ValueError(f'Unknown executor: {executor}')

        cmd += self.wl.extra_args

        if FLAGS.no_capture:
            return execute(cmd, cwd=str(cwd), env=self.env)
        else:
            output_file.parent.mkdir(exist_ok=True, parents=True)
            with output_file.open('w') as f:
                return execute(cmd, cwd=str(cwd), env=self.env, stdout=f, stderr=sp.STDOUT)


class TFWebDirectRunner(Runner):
    """Using TFWeb's load infrastructure to directly run"""

    def __init__(self, wl, base_dir=None):
        super().__init__(wl)
        self.base_dir = base_dir
        if self.base_dir is None:
            self.base_dir = FLAGS.tfweb_base

    def __call__(self, executor, output_file):
        model_name = remove_suffix(self.wl.name, 'eval')
        cwd = self.base_dir
        cmd = [
            'stdbuf', '-o0', '-e0', '--',
            'examples/direct/client',
            '--model="{}"'.format(str(Path(FLAGS.tfweb_saved_model_dir).joinpath(model_name))),
            '--batch_size={}'.format(self.wl.batch_size),
            '--batch_num={}'.format(self.wl.batch_num),
        ]

        if executor == Executor.Salus:
            cmd += [
                '--sess_target', SalusServer.current_server().endpoint,
            ]
        elif executor == Executor.TF:
            cmd += [
                '--sess_target', '""',
            ]
        elif executor == Executor.TFDist:
            cmd += [
                '--sess_target', TFDistServer.current_server().endpoint,
            ]
        else:
            raise ValueError(f'Unknown executor: {executor}')
        cmd += self.wl.extra_args

        if FLAGS.no_capture:
            return execute(cmd, cwd=str(cwd), env=self.env)
        else:
            output_file.parent.mkdir(exist_ok=True, parents=True)
            with output_file.open('w') as f:
                return execute(cmd, cwd=str(cwd), env=self.env, stdout=f, stderr=sp.STDOUT)


class TFWebRunner(Runner):
    """
    Run a TFWeb based inference job

    We start several servers and a balancer on the same node.
    The server commandline: tfweb --model=path/to/saved_model/network --sess_target=...
    The client commandline: gobetween from-file xxx.toml
    """

    def __init__(self, wl, base_dir=None):
        super().__init__(wl)
        self.base_dir = base_dir
        if self.base_dir is None:
            self.base_dir = FLAGS.tfweb_base

    def __call__(self, executor, output_file):
        # type: (Executor, Path) -> Popen
        model_name = remove_suffix(self.wl.name, 'web')
        cwd = self.base_dir
        cmd = [
            'stdbuf', '-o0', '-e0', '--',
            'examples/cluster/start_cluster',
            '--model="{}"'.format(str(Path(FLAGS.tfweb_saved_model_dir).joinpath(model_name))),
        ]

        if executor == Executor.Salus:
            cmd += [
                '--sess_target', SalusServer.current_server().endpoint,
            ]
        elif executor == Executor.TF:
            cmd += [
                '--sess_target', '""',
            ]
        elif executor == Executor.TFDist:
            cmd += [
                '--sess_target', TFDistServer.current_server().endpoint,
            ]
        else:
            raise ValueError(f'Unknown executor: {executor}')

        num_replicas = self.wl.env.pop('SALUS_TFWEB_REPLICAS', '1')
        cmd += [
            '--num_replicas', num_replicas
        ]
        cmd += self.wl.extra_args

        if FLAGS.no_capture:
            return execute(cmd, cwd=str(cwd), env=self.env)
        else:
            output_file.parent.mkdir(exist_ok=True, parents=True)
            with output_file.open('w') as f:
                return execute(cmd, cwd=str(cwd), env=self.env, stdout=f, stderr=sp.STDOUT)


class TFWebClientRunner(Runner):
    """
    Run a tfweb client attacker.
    Command: examples/cluster/tfweb-client TARGET REQ_BODY PLANTXT
    """

    def __init__(self, wl, base_dir=None):
        super().__init__(wl)
        self.base_dir = base_dir
        if self.base_dir is None:
            self.base_dir = FLAGS.tfweb_base

    def __call__(self, executor, output_file):
        # type: (Executor, Path) -> Popen

        model_name = remove_suffix(self.wl.name, 'client')

        cwd = self.base_dir
        cmd = [
            'stdbuf', '-o0', '-e0', '--',
            'examples/tfweb-client',
            '-output', str(output_file),
            self.wl.target,
            # request body
            str(Path(FLAGS.tfweb_request_body_dir).joinpath(model_name).with_suffix('.txt')),
            # always write plan to stdin
            '-',
        ]
        cmd += self.wl.extra_args

        proc = execute(cmd, cwd=str(cwd), env=self.env, stdin=sp.PIPE)
        proc.stdin.write(self._plan_to_bytes())
        proc.stdin.close()
        return proc

    def _plan_to_bytes(self):
        return ' '.join(self.wl.plan).encode('utf-8')

