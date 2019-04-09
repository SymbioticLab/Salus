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

import os
from builtins import super, str

import csv
import logging
from collections import defaultdict
from typing import Dict, Iterable, Type, Union

from .runner import Runner, RunConfig, Popen, Executor
from .runner import TFBenchmarkRunner, UnittestRunner, FathomRunner, TFWebRunner, TFWebClientRunner
from .utils import try_with_default, kill_tree, unique
from .utils.compatiblity import pathlib

Path = pathlib.Path
logger = logging.getLogger(__name__)


class ResourceGeometry(object):
    """Resource geometry of a workload"""
    __slots__ = (
        'jct',
        'peakmem',
        'persistmem',
    )

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.jct = None  # type: float
        self.peakmem = None  # type: int
        self.persistmem = None  # type: int

        for f, v in zip(ResourceGeometry.__slots__, args):
            if f not in kwargs:
                kwargs[f] = v

        for f, v in kwargs.items():
            setattr(self, f, v)

    def __repr__(self):
        content = ', '.join([f'{f}={getattr(self, f)!r}' for f in ResourceGeometry.__slots__])
        return f'ResourceGeometry({content})'

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def copy(self):
        # type: () -> ResourceGeometry
        """Make a shadow copy"""
        return ResourceGeometry().merge(self, True)

    def merge(self, other, overwrite=False):
        # type: (ResourceGeometry) -> ResourceGeometry
        if other is None:
            return self

        for f in ResourceGeometry.__slots__:
            if getattr(other, f) is None:
                continue
            if getattr(self, f) is not None and not overwrite:
                raise ValueError(f'Attempt to overwrite field {f}: {getattr(self, f)} -> {getattr(other, f)}')

            setattr(self, f, getattr(other, f))

        return self

    @classmethod
    def default_geometries(cls):
        # type: () -> Dict[RunConfig, Dict[Executor, ResourceGeometry]]
        return defaultdict(lambda: {
            ex: ResourceGeometry() for ex in Executor
        })


TGeometries = Dict[RunConfig, Dict[Executor, ResourceGeometry]]


class WorkloadTemplate(object):
    """An instance of Workload contains metadata of the workload, and a concrete runner"""

    known_workloads = {}  # type: Dict[str, WorkloadTemplate]

    def __init__(self, name, batch_sizes, runnercls):
        # type: (str, Iterable[Union[int, str]], Type[Runner]) -> None
        super().__init__()
        self.name = name
        self._batch_sizes = set(batch_sizes)
        self.runnerCls = runnercls
        self._geometries = ResourceGeometry.default_geometries()  # type: TGeometries

        self.create = self._create
        self.create_from_rcfg = self._create_from_rcfg

    def add_geometry(self, rcfg, executor, geometry, overwrite=False):
        # type: (RunConfig, Executor, ResourceGeometry, bool) -> None
        self._geometries[rcfg][executor].merge(geometry, overwrite)

    def geometry(self, rcfg, executor):
        # type: (RunConfig, Executor) -> ResourceGeometry
        return self._geometries[rcfg][executor]

    def canonical_name(self, rcfg):
        # type: (RunConfig) -> str
        """Format a canonical name for specific run config"""
        return f'{self.name}_{rcfg.batch_size}'

    def available_batch_sizes(self):
        # type: () -> Iterable[Union[int, str]]
        """Return all known batch sizes under this template"""
        return self._batch_sizes

    def available_batch_nums(self, batch_size):
        # type: (Union[int, str]) -> Iterable[int]
        """Return all known batch nums for a given batch_size under this template"""
        return unique(
            k.batch_num
            for k in self._geometries.keys()
            if k.batch_size == batch_size
        )

    def _create(self, batch_size, batch_num, executor=Executor.Salus):
        # type: (Union[str, int], Union[str, int], Executor) -> Workload
        """Create a concrete workload from template"""
        batch_size = try_with_default(int, batch_size, ValueError)(batch_size)
        batch_num = int(batch_num)
        return self._create_from_rcfg(RunConfig(batch_size, batch_num, None), executor)

    def _create_from_rcfg(self, rcfg, executor=Executor.Salus):
        # type: (RunConfig, Executor) -> Workload
        if rcfg.batch_size not in self.available_batch_sizes():
            raise ValueError(f"Batch size `{rcfg.batch_size}' is not supported for {self.name},"
                             f" available ones: {self.available_batch_sizes()}")
        return Workload(self, rcfg, executor, self.geometry(rcfg, executor).copy())

    @classmethod
    def from_name(cls, name):
        return cls.known_workloads[name]

    @classmethod
    def create_from_rcfg(cls, name, rcfg, executor=Executor.Salus):
        # type: (str, RunConfig, Executor) -> Workload
        """Create a concrete workload from template"""
        return cls.from_name(name)._create_from_rcfg(rcfg, executor)

    @classmethod
    def create(cls, name, batch_size, batch_num, executor=Executor.Salus):
        # type: (str, Union[str, int], Union[str, int], Executor) -> Workload
        """Create a concrete workload from template"""
        return cls.from_name(name)._create(batch_size, batch_num, executor)

    @classmethod
    def block_run(cls, name, bs_or_rcfg, *args, **kwargs):
        # type: () -> Workload
        if isinstance(bs_or_rcfg, RunConfig):
            return cls._block_run_rcfg(name, bs_or_rcfg, *args, **kwargs)
        else:
            if 'batch_num' in kwargs:
                batch_num = kwargs['batch_num']
            else:
                batch_num = args[0]
                args = args[1:]
            return cls._block_run_rcfg(name, RunConfig(bs_or_rcfg, batch_num, None), *args, **kwargs)

    @classmethod
    def _block_run_rcfg(cls, name, rcfg, executor, outputfile):
        # type: (str, RunConfig, Executor, Path) -> Workload
        w = cls.create_from_rcfg(name, rcfg, executor)
        try:
            w.run(outputfile)
            w.proc.wait()
        except Exception as ex:
            logger.exception(f'Got exception while running the experiment:')
        finally:
            if w.proc is not None and w.proc.poll() is None:
                logger.warning(f'Killing workload that is not stopped yet: {w.canonical_name}')
                kill_tree(w.proc, hard=True)
            if w.proc.returncode != 0:
                raise RuntimeError(f'Workload {w.canonical_name} did not finish cleanly: {w.proc.returncode}')
        return w

    @classmethod
    def define(cls, *args, **kwargs):
        wtl = WorkloadTemplate(*args, **kwargs)

        cls.known_workloads[wtl.name] = wtl

    @classmethod
    def load_extra(cls, csvfile):
        """Load extra infomation from csv file"""
        with open(csvfile, 'rb') as f:
            logger.info(f'Loading extra info from: {csvfile}')

            count = 0
            reader = csv.reader(f)
            for name, ex, batch_size, batch_num, cfgname, jct, peakmem, persistmem in reader:
                if name not in cls.known_workloads:
                    logger.warning(f"Ignored unknown workload: {name}")
                    continue

                cfgname = None if not cfgname else cfgname
                ex = Executor[ex]
                batch_size = try_with_default(int, batch_size, ValueError)(batch_size)
                batch_num = try_with_default(int, batch_num, ValueError)(batch_num)
                if not isinstance(batch_num, int):
                    logger.warning(f"Ignored invlid batch num: {batch_num}")
                    continue

                jct = try_with_default(float, None, ValueError)(jct)
                peakmem = try_with_default(int, None, ValueError)(peakmem)
                persistmem = try_with_default(int, None, ValueError)(persistmem)

                wtl = cls.known_workloads[name]
                # first add batch_size if not present
                wtl._batch_sizes.add(batch_size)
                # update geometry of the rcfg
                rcfg = RunConfig(batch_size, batch_num, cfgname)
                wtl.add_geometry(rcfg, ex, ResourceGeometry(jct, peakmem, persistmem))
                count += 1

            logger.info(f'Loaded {count} records')

    @classmethod
    def load_jctcsv(cls, csvfile):
        csvfile = Path(csvfile)
        count = 0
        with csvfile.open() as f:
            logger.info(f'Loading extra JCT info from: {csvfile!s}')

            if 'salus' in csvfile.name:
                ex = Executor.Salus
            elif 'baseline' in csvfile.name:
                ex = Executor.TF
            else:
                ex = Executor.TF

            reader = csv.DictReader(f)
            for row in reader:
                name, bs = row['Network'].split('_')
                bs = try_with_default(int, bs, ValueError)(bs)

                # first add batch_size if not present
                wtl = cls.known_workloads[name]
                wtl._batch_sizes.add(bs)

                # 20iter
                jct = try_with_default(float, None, ValueError)(row['20iter-jct'])
                if jct is not None:
                    # update geometry of the rcfg
                    rcfg = RunConfig(bs, 20, None)
                    wtl.add_geometry(rcfg, ex, ResourceGeometry(jct, None, None))

                # 1min
                jct = try_with_default(float, None, ValueError)(row['1min-jct'])
                bn = try_with_default(float, None, ValueError)(row['1min-num'])
                if jct is not None and bn is not None:
                    bn = int(bn)
                    # update geometry of the rcfg
                    rcfg = RunConfig(bs, bn, None)
                    wtl.add_geometry(rcfg, ex, ResourceGeometry(jct, None, None))

                # 5min
                jct = try_with_default(float, None, ValueError)(row['5min-jct'])
                bn = try_with_default(float, None, ValueError)(row['5min-num'])
                if jct is not None and bn is not None:
                    bn = int(bn)
                    # update geometry of the rcfg
                    rcfg = RunConfig(bs, bn, None)
                    wtl.add_geometry(rcfg, ex, ResourceGeometry(jct, None, None))

                # 10min
                jct = try_with_default(float, None, ValueError)(row['10min-jct'])
                bn = try_with_default(float, None, ValueError)(row['10min-num'])
                if jct is not None and bn is not None:
                    bn = int(bn)
                    # update geometry of the rcfg
                    rcfg = RunConfig(bs, bn, None)
                    wtl.add_geometry(rcfg, ex, ResourceGeometry(jct, None, None))

                count += 1
        logger.info(f'Loaded {count} records')

    @classmethod
    def load_memcsv(cls, csvfile):
        csvfile = Path(csvfile)
        count = 0
        with csvfile.open() as f:
            logger.info(f'Loading extra mem info from: {csvfile!s}')

            if 'salus' in csvfile.name:
                ex = Executor.Salus
            elif 'baseline' in csvfile.name:
                ex = Executor.TF
            else:
                ex = Executor.TF

            MB = 1024 * 1024
            reader = csv.DictReader(f)
            for row in reader:
                name, bs = row['Network'].split('_')
                bs = try_with_default(int, bs, ValueError)(bs)

                peakmem = try_with_default(float, None, ValueError)(row['Peak Mem (MB)'])
                if peakmem is not None:
                    peakmem = peakmem * MB
                persistmem = try_with_default(float, None, ValueError)(row['Persistent Mem (MB)'])
                if persistmem is not None:
                    persistmem = persistmem * MB

                wtl = cls.known_workloads[name]
                # first add batch_size if not present
                wtl._batch_sizes.add(bs)
                # for all batch num
                abns = wtl.available_batch_nums(bs)
                if not abns:
                    # no batch num available, add a default one
                    abns = [20]
                for bn in abns:
                    # update geometry of the rcfg
                    rcfg = RunConfig(bs, bn, None)
                    wtl.add_geometry(rcfg, ex, ResourceGeometry(None, peakmem, persistmem))
                count += 1
        logger.info(f'Loaded {count} records')


# An alias
WTL = WorkloadTemplate


class Workload(object):
    """A concrete workload with specific run config"""
    def __init__(self, wtl, rcfg, executor, geo):
        # type: (WorkloadTemplate, RunConfig, Executor, ResourceGeometry) -> None
        super().__init__()
        self.wtl = wtl
        self.env = os.environ.copy()
        self.rcfg = rcfg
        self.executor = executor
        self._geo = geo
        self.proc = None  # type: Popen
        self.output_file = None  # type: Path

    @property
    def name(self):
        # type: () -> str
        return self.wtl.name

    @property
    def batch_size(self):
        # type: () -> Union[int, str]
        return self.rcfg.batch_size

    @property
    def batch_num(self):
        # type: () -> int
        return self.rcfg.batch_num

    @property
    def canonical_name(self):
        # type: () -> str
        """Format a canonical name for specific run config"""
        return self.wtl.canonical_name(self.rcfg)

    @property
    def output_name(self):
        # type: () -> str
        """Format a name for specific run config suitable for use as unique filename"""
        d = self.canonical_name
        if self.batch_num != 20 and self.rcfg.cfgname is not None:
            d += f'_{self.rcfg.cfgname}'
        d += f'.{self.executor.value}'
        return d

    @property
    def geometry(self):
        # type: () -> ResourceGeometry
        """Return the resource geometry for the workload"""
        return self._geo

    def run(self, output_file):
        # type: (Path) -> Popen
        """Run workload on executor"""
        _runner = self.wtl.runnerCls(self)

        if self.proc is not None:
            raise RuntimeError("This workload is already started")

        logger.info(f"Starting workload `{self.canonical_name}' on {self.executor.name}"
                    f" with output file: {output_file!s}")
        self.output_file = output_file
        self.proc = _runner(self.executor, output_file)
        self.proc.workload = self
        return self.proc


WorkloadTemplate.define('vgg11', [25, 50, 100], TFBenchmarkRunner)
WorkloadTemplate.define('vgg16', [25, 50, 100], TFBenchmarkRunner)
WorkloadTemplate.define('vgg19', [25, 50, 100], TFBenchmarkRunner)
WorkloadTemplate.define('resnet50', [25, 50, 75], TFBenchmarkRunner)
WorkloadTemplate.define('resnet101', [25, 50, 75], TFBenchmarkRunner)
WorkloadTemplate.define('resnet152', [25, 50, 75], TFBenchmarkRunner)
WorkloadTemplate.define('googlenet', [25, 50, 100], TFBenchmarkRunner)
WorkloadTemplate.define('alexnet', [25, 50, 100], TFBenchmarkRunner)
WorkloadTemplate.define('overfeat', [25, 50, 100], TFBenchmarkRunner)
WorkloadTemplate.define('inception3', [25, 50, 100], TFBenchmarkRunner)
WorkloadTemplate.define('inception4', [25, 50, 75], TFBenchmarkRunner)
WorkloadTemplate.define('speech', [25, 50, 75], FathomRunner)
WorkloadTemplate.define('seq2seq', ['small', 'medium', 'large'], UnittestRunner)
WorkloadTemplate.define('mnistsf', [25, 50, 100], UnittestRunner)
WorkloadTemplate.define('mnistcv', [25, 50, 100], UnittestRunner)
WorkloadTemplate.define('mnistlg', [25, 50, 100], UnittestRunner)
WorkloadTemplate.define('vae', [64, 128, 256], UnittestRunner)
WorkloadTemplate.define('superres', [32, 64, 128], UnittestRunner)
WorkloadTemplate.define('vgg11eval', [1, 5, 10], TFBenchmarkRunner)
WorkloadTemplate.define('vgg16eval', [1, 5, 10], TFBenchmarkRunner)
WorkloadTemplate.define('vgg19eval', [1, 5, 10], TFBenchmarkRunner)
WorkloadTemplate.define('resnet50eval', [1, 5, 10], TFBenchmarkRunner)
WorkloadTemplate.define('resnet101eval', [1, 5, 10], TFBenchmarkRunner)
WorkloadTemplate.define('resnet152eval', [1, 5, 10], TFBenchmarkRunner)
WorkloadTemplate.define('googleneteval', [1, 5, 10], TFBenchmarkRunner)
WorkloadTemplate.define('alexneteval', [1, 5, 10], TFBenchmarkRunner)
WorkloadTemplate.define('overfeateval', [1, 5, 10], TFBenchmarkRunner)
WorkloadTemplate.define('inception3eval', [1, 5, 10], TFBenchmarkRunner)
WorkloadTemplate.define('inception4eval', [1, 5, 10], TFBenchmarkRunner)
WorkloadTemplate.define('speecheval', [1, 5, 10], FathomRunner)
WorkloadTemplate.define('vaeeval', [1, 5, 10], UnittestRunner)
WorkloadTemplate.define('superreseval', [1, 5, 10], UnittestRunner)
WorkloadTemplate.define('seq2seqeval', ['small', 'medium', 'large'], UnittestRunner)

WorkloadTemplate.define('vgg11web', [1], TFWebRunner)
WorkloadTemplate.define('vgg16web', [1], TFWebRunner)
WorkloadTemplate.define('vgg19web', [1], TFWebRunner)
WorkloadTemplate.define('resnet50web', [1], TFWebRunner)
WorkloadTemplate.define('resnet101web', [1], TFWebRunner)
WorkloadTemplate.define('resnet152web', [1], TFWebRunner)
WorkloadTemplate.define('googlenetweb', [1], TFWebRunner)
WorkloadTemplate.define('alexnetweb', [1], TFWebRunner)
WorkloadTemplate.define('overfeatweb', [1], TFWebRunner)
WorkloadTemplate.define('inception3web', [1], TFWebRunner)
WorkloadTemplate.define('inception4web', [1], TFWebRunner)

WorkloadTemplate.define('vgg11client', [1], TFWebClientRunner)
WorkloadTemplate.define('vgg16client', [1], TFWebClientRunner)
WorkloadTemplate.define('vgg19client', [1], TFWebClientRunner)
WorkloadTemplate.define('resnet50client', [1], TFWebClientRunner)
WorkloadTemplate.define('resnet101client', [1], TFWebClientRunner)
WorkloadTemplate.define('resnet152client', [1], TFWebClientRunner)
WorkloadTemplate.define('googlenetclient', [1], TFWebClientRunner)
WorkloadTemplate.define('alexnetclient', [1], TFWebClientRunner)
WorkloadTemplate.define('overfeatclient', [1], TFWebClientRunner)
WorkloadTemplate.define('inception3client', [1], TFWebClientRunner)
WorkloadTemplate.define('inception4client', [1], TFWebClientRunner)


# noinspection PyUnusedLocal
def _disable_init(self):
    raise RuntimeError('Cannot create new instance of WorkloadTemplate')


WorkloadTemplate.__init__ = _disable_init
del _disable_init
