from __future__ import absolute_import, print_function, division
from builtins import super

from copy import copy

from ...utils.compatiblity import Path


class SalusConfig(object):
    """Configuration to SalusServer"""
    def __init__(self, **kwargs):
        """Initialize a config
        """
        super().__init__()
        self.build_dir = Path('../build')
        self.build_type = 'Release'
        self.logconf = 'disable'
        self.logconf_dir = None  # type: Path
        self.use_nvprof = False
        self.hide_output = True
        self.scheduler = 'fair'
        self.disable_adc = False
        self.disable_wc = False
        self.extra_args = []
        self.output_dir = Path('templogs')
        self.update(kwargs)

    def __repr__(self):
        content = ', '.join([f'{f}={getattr(self, f)!r}' for f in SalusConfig.__slots__])
        return f'ResourceGeometry({content})'

    def copy(self, **kwargs):
        # type: (...) -> SalusConfig
        """Return a new copy of the tuple"""
        return copy(self).update(**kwargs)

    def update(self, d=None, **kwargs):
        # type: (...) -> SalusConfig
        """Update this tuple"""
        if d is not None:
            self.update(**d)

        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)
        return self
