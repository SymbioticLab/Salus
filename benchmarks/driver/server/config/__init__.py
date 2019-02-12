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
from builtins import super

from absl import flags
from copy import copy

from ...utils import maybe_path
from ...utils.compatiblity import pathlib

Path = pathlib.Path
FLAGS = flags.FLAGS
flags.DEFINE_string('build_dir', 'build', 'Default build directory')
flags.DEFINE_string('logconf_dir', None, 'Default logconf directory')
flags.DEFINE_boolean('force_show_output', False, 'Force show server stdout')


class SalusConfig(object):
    """Configuration to SalusServer"""
    _pathlike_keys = (
        'build_dir',
        'logconf_dir',
        'output_dir',
    )

    def __init__(self, **kwargs):
        """Initialize a config
        """
        super().__init__()
        self.build_dir = Path(FLAGS.build_dir)  # type: Path
        self.build_type = 'Release'
        self.logconf = 'disable'
        self.logconf_dir = None if FLAGS.logconf_dir is None else Path(FLAGS.logconf_dir)  # type: Path
        self.use_nvprof = False
        self.use_gperf = False
        self.hide_output = True
        self.scheduler = 'fair'
        self.disable_adc = False
        self.disable_wc = False
        self.extra_args = []
        self.env = {}
        self.save_outerr = False
        self.output_dir = Path('templogs')
        self.kill_timeout = 3
        self.update(kwargs)

    def __repr__(self):
        content = ', '.join([f'{f}={getattr(self, f)!r}' for f in SalusConfig.__slots__])
        return f'ResourceGeometry({content})'

    def __setattr__(self, key, value):
        if key in self._pathlike_keys:
            value = maybe_path(value)
        if FLAGS.force_show_output and key == 'hide_output':
            value = False

        object.__setattr__(self, key, value)

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
