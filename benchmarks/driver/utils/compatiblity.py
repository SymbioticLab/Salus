from __future__ import absolute_import, print_function, division

try:
    from subprocess import DEVNULL  # py3k
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')  # 2 backport
    del os

try:
    from pathlib import Path
except ImportError:
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    from pathlib2 import Path  # noqa: F401

try:
    from tempfile import TemporaryDirectory
except ImportError:
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    from backports.tempfile import TemporaryDirectory


__all__ = [
    'DEVNULL',
    'Path',
    'TemporaryDirectory',
]
