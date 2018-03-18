# -*- coding: future_fstrings -*-
from __future__ import absolute_import, print_function, division
from builtins import super, str

import csv
from collections import defaultdict
from typing import Dict, Iterable, Type, Union

from .runner import Runner, RunConfig, enumerate_rcfgs, Popen, Executor
from .runner import TFBenchmarkRunner, UnittestRunner, FathomRunner
from .utils import eprint, try_with_default
from .utils.compatiblity import Path


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

        for f in ResourceGeometry.__slots__[len(args):]:
            if f not in kwargs:
                kwargs[f] = None

    def __repr__(self):
        content = ', '.join([f'{f}={getattr(self, f)!r}' for f in ResourceGeometry.__slots__])
        return f'ResourceGeometry({content})'

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

    def __init__(self, name, rcfgs, runnercls):
        # type: (str, Iterable[RunConfig], Type[Runner]) -> None
        super().__init__()
        self.name = name
        self.rcfgs = set(rcfgs)
        self.runnerCls = runnercls
        self._geometries = ResourceGeometry.default_geometries()  # type: TGeometries

    def add_geometry(self, rcfg, executor, geometry, overwrite=False):
        # type: (RunConfig, Executor, ResourceGeometry, bool) -> None
        self._geometries[rcfg][executor].merge(geometry, overwrite)

    known_workloads = {}  # type: Dict[str, WorkloadTemplate]

    @classmethod
    def from_name(cls, name):
        return cls.known_workloads[name]

    @classmethod
    def create_from_rcfg(cls, name, rcfg, executor=Executor.Salus):
        # type: (str, RunConfig, Executor) -> Workload
        """Create a concrete workload from template"""
        wtl = cls.from_name(name)
        return Workload(wtl, rcfg, executor, wtl._geometries[rcfg][executor].copy())

    @classmethod
    def create(cls, name, batch_size, batch_num, executor=Executor.Salus):
        # type: (str, Union[str, int], Union[str, int], Executor) -> Workload
        """Create a concrete workload from template"""
        batch_size = int(batch_size)
        batch_num = int(batch_num)
        return cls.create_from_rcfg(name, RunConfig(batch_size, batch_num, None), executor)

    @classmethod
    def define(cls, *args, **kwargs):
        wtl = WorkloadTemplate(*args, **kwargs)

        cls.known_workloads[wtl.name] = wtl

    @classmethod
    def load_extra(cls, csvfile):
        """Load extra infomation from csv file"""
        with open(csvfile, 'rb') as f:
            reader = csv.reader(f)
            for name, ex, batch_size, batch_num, cfgname, jct, peakmem, persistmem in reader:
                if name not in cls.known_workloads:
                    eprint(f"WARNING: unknown workload `{name}'")

                cfgname = None if not cfgname else cfgname
                ex = Executor[ex]
                jct = try_with_default(float, None, ValueError)(jct)
                peakmem = try_with_default(int, None, ValueError)(peakmem)
                persistmem = try_with_default(int, None, ValueError)(persistmem)

                wtl = cls.known_workloads[name]
                rcfg = RunConfig(batch_size, batch_num, cfgname)
                # first add rcfg if not present
                wtl.rcfgs.add(rcfg)
                # update geometry of the rcfg
                wtl.add_geometry(rcfg, ex, ResourceGeometry(jct, peakmem, persistmem))


# An alias
WTL = WorkloadTemplate


class Workload(object):
    """A concrete workload with specific run config"""
    def __init__(self, wtl, rcfg, executor, geo):
        # type: (WorkloadTemplate, RunConfig, Executor, ResourceGeometry) -> None
        super().__init__()
        self._wtl = wtl
        self.rcfg = rcfg
        self.executor = executor
        self._geo = geo
        self._runner = self._wtl.runnerCls(self)
        self.proc = None  # type: Popen
        self.output_file = None  # type: Path

    @property
    def name(self):
        # type: () -> str
        return self._wtl.name

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
        return f'{self.name}_{self.batch_size}'

    @property
    def output_name(self):
        # type: () -> str
        """Format a name for specific run config suitable for use as unique filename"""
        d = self.canonical_name
        if self.batch_num != 20:
            d += f'_{self.rcfg.cfgname}'
        d += f'.{self.executor.value}'
        return d

    @property
    def geometry(self):
        # type: () -> ResourceGeometry
        """Return the resource geometry for the workload"""
        return self._geo

    def run(self, output_file):
        """Run workload on executor"""
        if self.proc is not None:
            raise RuntimeError("This workload is already started")

        self.output_file = output_file
        self.proc = self._runner(self.executor, output_file)
        return self.proc


WorkloadTemplate.define('vgg11', enumerate_rcfgs([25, 50, 100], [20]), TFBenchmarkRunner)
WorkloadTemplate.define('vgg16', enumerate_rcfgs([25, 50, 100], [20]), TFBenchmarkRunner)
WorkloadTemplate.define('vgg19', enumerate_rcfgs([25, 50, 100], [20]), TFBenchmarkRunner)
WorkloadTemplate.define('resnet50', enumerate_rcfgs([25, 50, 75], [20]), TFBenchmarkRunner)
WorkloadTemplate.define('resnet101', enumerate_rcfgs([25, 50, 75], [20]), TFBenchmarkRunner)
WorkloadTemplate.define('resnet152', enumerate_rcfgs([25, 50, 75], [20]), TFBenchmarkRunner)
WorkloadTemplate.define('googlenet', enumerate_rcfgs([25, 50, 100], [20]), TFBenchmarkRunner)
WorkloadTemplate.define('alexnet', enumerate_rcfgs([25, 50, 100], [20]), TFBenchmarkRunner)
WorkloadTemplate.define('overfeat', enumerate_rcfgs([25, 50, 100], [20]), TFBenchmarkRunner)
WorkloadTemplate.define('inception3', enumerate_rcfgs([25, 50, 100], [20]), TFBenchmarkRunner)
WorkloadTemplate.define('inception4', enumerate_rcfgs([25, 50, 75], [20]), TFBenchmarkRunner)
WorkloadTemplate.define('speech', enumerate_rcfgs([25, 50, 75], [20]), FathomRunner)
WorkloadTemplate.define('seq2seq', enumerate_rcfgs(['small', 'medium', 'large'], [20]), UnittestRunner)


def _disable_init(self):
    raise RuntimeError('Cannot create new instance of WorkloadTemplate')


WorkloadTemplate.__init__ = _disable_init
del _disable_init
