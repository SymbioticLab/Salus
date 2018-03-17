from __future__ import absolute_import, print_function, division

from . import SalusConfig


MostEfficient = SalusConfig(
    build_type='Release',
    logconf='disable',
    hide_output=True
)

Debugging = SalusConfig(
    build_type='Debug',
    logconf='debug',
    hide_output=False
)

Profiling = SalusConfig(
    build_type='Release',
    logconf='perf',
    hide_output=False
)

AllocProf = SalusConfig(
    build_type='Release',
    logconf='alloc',
    hide_output=False
)

Verbose = SalusConfig(
    build_type='Release',
    logconf='both',
    hide_output=False
)

Nvprof = SalusConfig(
    build_type='Release',
    logconf='both',
    hide_output=False,
    use_nvprof=True,
)

__all__ = [
    'MostEfficient',
    'Debugging',
    'Profiling',
    'AllocProf',
    'Verbose',
    'Nvprof'
]
