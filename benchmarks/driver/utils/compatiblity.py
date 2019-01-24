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

import sys
from .utils import is_unix

# Useful for very coarse version differentiation.
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

# Backport subprocess from python 3.2, with timeout parameter from python 3.3
if is_unix and sys.version_info[0:2] < (3, 3):
    # noinspection PyUnresolvedReferences
    import subprocess32 as subprocess
    import os
    subprocess.DEVNULL = open(os.devnull, 'wb')
    del os
else:
    import subprocess

# Backport pathlib from python 3
if sys.version_info[0:2] < (3, 6):
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    import pathlib2 as pathlib
else:
    # noinspection PyUnresolvedReferences
    import pathlib


# Backport TemporaryDirectory from python 3
# noinspection PyPep8
import tempfile
if not hasattr(tempfile, 'TemporaryDirectory'):
    # noinspection PyUnresolvedReferences
    from backports.tempfile import TemporaryDirectory
    tempfile.TemporaryDirectory = TemporaryDirectory
    del TemporaryDirectory


__all__ = [
    'PY2',
    'PY3',
    'subprocess',
    'pathlib',
    'tempfile',
]
