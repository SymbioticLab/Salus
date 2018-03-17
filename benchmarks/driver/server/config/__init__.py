from __future__ import absolute_import, print_function, division
from builtins import super

from collections import namedtuple
from ...utils import remove_none


class SalusConfig(namedtuple('SalusConfig', [
    'build_dir',
    'build_type',
    'logconf',
    'logconf_dir',
    'use_nvprof',
    'hide_output',
    'scheduler',
    'disable_adc',
    'disable_wc',
    'extra_args',
    'output_dir',
])):
    """Configuration to SalusServer"""

    def __init__(self, **kwargs):
        """Initialize a config

        @kwparam build_dir
        @type str
        """
        defaults = {
            'build_dir': '../build',
            'build_type': 'Release',
            'logconf': 'disable',
            'logconf_dir': None,
            'use_nvprof': False,
            'hide_output': True,
            'scheduler': 'fair',
            'disable_adc': False,
            'disable_wc': False,
            'extra_args': [],
            'output_dir': 'templogs'
        }
        defaults.update(remove_none(kwargs))
        super().__init__(**defaults)

    def copy(self, **kwargs):
        # type: (...) -> SalusConfig
        """Return a new copy of the tuple"""
        return SalusConfig(**self._asdict()).update(kwargs)

    def update(self, d=None, **kwargs):
        # type: (...) -> SalusConfig
        """Update this tuple"""
        if d is not None:
            self._replace(**remove_none(d))

        self._replace(**remove_none(kwargs))
        return self
