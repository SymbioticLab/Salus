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

from . import SalusConfig


# noinspection PyPep8Naming
def MostEfficient(**kwargs):
    # type: (...) -> SalusConfig
    return SalusConfig(
        build_type='Release',
        logconf='disable',
        hide_output=True
    ).update(**kwargs)


# noinspection PyPep8Naming
def Debugging(**kwargs):
    # type: (...) -> SalusConfig
    return SalusConfig(
        build_type='Debug',
        logconf='log',
        hide_output=False
    ).update(**kwargs)


# noinspection PyPep8Naming
def OpTracing(**kwargs):
    # type: (...) -> SalusConfig
    return SalusConfig(
        build_type='OpTracing',
        logconf='optracing',
        extra_args=['-v1'],
        hide_output=False,
    ).update(**kwargs)


# noinspection PyPep8Naming
def Profiling(**kwargs):
    # type: (...) -> SalusConfig
    return SalusConfig(
        build_type='OpTracing',
        logconf='perf',
        hide_output=False,
        extra_args=['--perflog', '/tmp/perf.output'],
    ).update(**kwargs)


# noinspection PyPep8Naming
def AllocProf(**kwargs):
    # type: (...) -> SalusConfig
    return SalusConfig(
        build_type='OpTracing',
        logconf='alloc',
        hide_output=False
    ).update(**kwargs)


# noinspection PyPep8Naming
def Verbose(**kwargs):
    # type: (...) -> SalusConfig
    return SalusConfig(
        build_type='Release',
        logconf='both',
        hide_output=False
    ).update(**kwargs)


# noinspection PyPep8Naming
def Nvprof(**kwargs):
    # type: (...) -> SalusConfig
    return SalusConfig(
        build_type='Release',
        logconf='disable',
        hide_output=False,
        use_nvprof=True,
    ).update(**kwargs)


# noinspection PyPep8Naming
def Gperf(**kwargs):
    # type: (...) -> SalusConfig
    return SalusConfig(
        build_type='Gperf',
        logconf='disable',
        hide_output=False,
        use_gperf=True,
        kill_timeout=10,
    ).update(**kwargs)


__all__ = [
    'MostEfficient',
    'Debugging',
    'Profiling',
    'AllocProf',
    'Verbose',
    'Nvprof',
    'Gperf'
]
