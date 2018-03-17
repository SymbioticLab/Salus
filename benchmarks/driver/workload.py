from __future__ import absolute_import, print_function, division
from builtins import super, str

import csv
from collections import namedtuple
from typing import Dict, Iterable, Type, Union

from .runner import Runner, RunConfig, enumerate_rcfgs, TFBenchmarkRunner, UnittestRunner, FathomRunner, Popen
from .utils import eprint, try_with_default


class ResourceGeometry(namedtuple('ResourceGeometry', 'jct peakmem persistmem')):
    """Resource geometry of a workload"""
    def __init__(self, *args, **kwargs):
        for f in ResourceGeometry._fields[len(args):]:
            if f not in kwargs:
                kwargs[f] = None

        super().__init__(*args, **kwargs)

    def merge(self, other):
        # type: (ResourceGeometry) -> ResourceGeometry
        if other is None:
            return self

        for f in ResourceGeometry._fields:
            if getattr(self, f) is None:
                setattr(self, f, getattr(other, f))

        return self


class WorkloadTemplate(object):
    """An instance of Workload contains metadata of the workload, and a concrete runner"""

    def __init__(self, name, rcfgs, runnercls):
        # type: (str, Iterable[RunConfig], Type[Runner]) -> None
        super().__init__()
        self.name = name
        self.rcfgs = set(rcfgs)
        self.runnerCls = runnercls
        self._geometries = {}  # type: Dict[RunConfig, ResourceGeometry]

    def add_geometry(self, rcfg, geometry):
        # type: (RunConfig, ResourceGeometry) -> None
        self._geometries[rcfg] = geometry.merge(self._geometries.get(rcfg))

    known_workloads = {}  # type: Dict[str, WorkloadTemplate]

    @classmethod
    def from_rcfg(cls, name, rcfg, executor='salus'):
        # type: (str, RunConfig, str) -> Workload
        """Create a concrete workload from template"""
        wtl = cls.known_workloads[name]
        return Workload(wtl, rcfg, executor, wtl._geometries[rcfg])

    @classmethod
    def create(cls, name, batch_size, batch_num, executor='salus'):
        # type: (str, Union[str, int], Union[str, int], str) -> Workload
        """Create a concrete workload from template"""
        batch_size = int(batch_size)
        batch_num = int(batch_num)
        return cls.from_rcfg(name, RunConfig(batch_size, batch_num, None), executor)

    @classmethod
    def define(cls, *args, **kwargs):
        wtl = WorkloadTemplate(*args, **kwargs)

        cls.known_workloads[wtl.name] = wtl

    @classmethod
    def load_extra(cls, csvfile):
        """Load extra infomation from csv file"""
        with open(csvfile, 'rb') as f:
            reader = csv.reader(f)
            for name, batch_size, batch_num, cfgname, jct, peakmem, persistmem in reader:
                if name not in cls.known_workloads:
                    eprint("WARNING: unknown workload `{}'".format(name))

                cfgname = None if not cfgname else cfgname
                jct = try_with_default(float, None, ValueError)(jct)
                peakmem = try_with_default(int, None, ValueError)(peakmem)
                persistmem = try_with_default(int, None, ValueError)(persistmem)

                wtl = cls.known_workloads[name]
                rcfg = RunConfig(batch_size, batch_num, cfgname)
                # first add rcfg if not present
                wtl.rcfgs.add(rcfg)
                # update geometry of the rcfg
                wtl.add_geometry(rcfg, ResourceGeometry(jct, peakmem, persistmem))


# An alias
WTL = WorkloadTemplate


class Workload(object):
    """A concrete workload with specific run config"""
    def __init__(self, wtl, rcfg, executor, geo):
        # type: (WorkloadTemplate, RunConfig, str, ResourceGeometry) -> None
        super().__init__()
        self._wtl = wtl
        self._rcfg = rcfg
        self._executor = executor
        self._geo = geo
        self._runner = self._wtl.runnerCls(self)
        self.proc = None  # type: Popen

    @property
    def name(self):
        # type: () -> str
        return self._wtl.name

    @property
    def batch_size(self):
        # type: () -> int
        return self._rcfg.batch_size

    @property
    def batch_num(self):
        # type: () -> int
        return self._rcfg.batch_num

    @property
    def canonical_name(self):
        # type: () -> str
        """Format a canonical name for specific run config"""
        return '{}_{}'.format(self.name, self.batch_size)

    @property
    def output_name(self):
        # type: () -> str
        """Format a name for specific run config suitable for use as unique filename"""
        d = '{}_{}'.format(self.name, self.batch_num)
        if self.batch_num != 20:
            d += '_' + self._rcfg.cfgname
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
        self.proc = self._runner(self._executor, output_file)
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
