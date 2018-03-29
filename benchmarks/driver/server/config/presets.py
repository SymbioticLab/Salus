# -*- coding: future_fstrings -*-
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
        logconf='debug',
        hide_output=False
    ).update(**kwargs)


# noinspection PyPep8Naming
def Profiling(**kwargs):
    # type: (...) -> SalusConfig
    return SalusConfig(
        build_type='Release',
        logconf='perf',
        hide_output=False,
        extra_args=['--perflog', '/tmp/perf.output'],
    ).update(**kwargs)


# noinspection PyPep8Naming
def AllocProf(**kwargs):
    # type: (...) -> SalusConfig
    return SalusConfig(
        build_type='Release',
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


__all__ = [
    'MostEfficient',
    'Debugging',
    'Profiling',
    'AllocProf',
    'Verbose',
    'Nvprof'
]
