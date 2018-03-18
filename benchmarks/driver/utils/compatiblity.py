# -*- coding: future_fstrings -*-
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
