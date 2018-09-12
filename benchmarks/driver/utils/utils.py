# -*- coding: future_fstrings -*-
from __future__ import absolute_import, print_function, division, unicode_literals
from builtins import str

import os
import platform
import signal
import logging
import sys
import shutil
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Union, TypeVar, Mapping

import psutil
from psutil import Popen

TOnTerminate = Callable[[psutil.Process], None]
TKey = TypeVar('TKey')
TValue = TypeVar('TValue')
T = TypeVar('T')
logger = logging.getLogger(__name__)

# Whether we are on unix
is_unix = platform.system() != "Windows"


class SalusError(Exception):
    """Base class for all exceptions"""
    pass


class ServerError(SalusError):
    """Exception related to salus server"""
    pass


class UsageError(SalusError):
    """Exception raised when the arguments supplied by the user are invalid.
    Raise this when the arguments supplied are invalid from the point of
    view of the application. For example when two mutually exclusive
    flags have been supplied or when there are not enough non-flag
    arguments. It is distinct from flags.Error which covers the lower
    level of parsing and validating individual flags.
    """

    def __init__(self, message, exitcode=1):
        super().__init__(message)
        self.exitcode = exitcode


def eprint(*args, **kwargs):
    # type: (*Any, **Any) -> None
    """Print to stderr"""
    print(*args, file=sys.stderr, **kwargs)


def remove_none(d):
    # type: (Mapping[TKey, TValue]) -> Mapping[TKey, TValue]
    """Remove None value from dict"""
    return {k: v for k, v in d.items() if v is not None}


def remove_prefix(text, prefix):
    # type: (str, str) -> str
    """Remove prefix from text if any"""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever


def try_with_default(func, default=None, ignore=Exception):
    """ A wrapper that ignores exception from a function.
    """
    def _dec(*args, **kwargs):
        # noinspection PyBroadException
        try:
            return func(*args, **kwargs)
        except ignore:
            return default
    return _dec


def execute(*args, **kwargs):
    # type: (*Any, **Any) -> psutil.Popen
    """A thin wrapper around 'psutil.Popen'"""
    if is_unix:
        # Call os.setpgrp to make sure the process starts in a new group
        orig = kwargs.get('preexec_fn', None)

        def wrapper():
            os.setpgrp()
            if orig:
                orig()

        kwargs['preexec_fn'] = wrapper
    kwargs['universal_newlines'] = True

    return psutil.Popen(*args, **kwargs)


# @formatter:off
def kill_tree(leader,               # type: Union[int, psutil.Process]
              sig=signal.SIGTERM,   # type: int
              include_leader=True,  # type: bool
              timeout=3,            # type: int
              hard=False,           # type: bool
              on_terminate=None     # type: TOnTerminate
              ):                    # @formatter:on
    # type: (...) -> (Iterable[psutil.Process], Iterable[psutil.Process])
    """Kill a process tree (including grandchildren) with signal
    "sig" and return a (gone, still_alive) tuple.
    "on_terminate", if specified, is a callabck function which is
    called as soon as a child terminates.
    """
    if not isinstance(leader, psutil.Process):
        leader = psutil.Process(leader)

    if leader.pid == os.getpid():
        raise RuntimeError("I refuse to kill myself")

    children = leader.children(recursive=True)
    if include_leader:
        children.append(leader)

    if hard:
        gone, alive = kill_hard(children, timeout=timeout, on_terminate=on_terminate)
    else:
        for p in children:
            p.send_signal(sig)
        gone, alive = psutil.wait_procs(children, timeout=timeout, callback=on_terminate)
    return gone, alive


def kill_hard(procs, timeout=3, on_terminate=None):
    # type: (Iterable[psutil.Process], int, TOnTerminate) -> (Iterable[psutil.Process], Iterable[psutil.Process])
    """Tries hard to terminate and ultimately kill the processes."""
    # send SIGTERM
    for p in procs:
        p.terminate()
    gone, alive = psutil.wait_procs(procs, timeout=timeout, callback=on_terminate)
    if alive:
        # send SIGKILL
        for p in alive:
            logger.warning(f"process {p} survived SIGTERM, trying SIGKILL")
            p.kill()
        gone, alive = psutil.wait_procs(alive, timeout=timeout, callback=on_terminate)
        if alive:
            # give up
            for p in alive:
                logger.error(f"process {p} survived SIGKILL, giving up")
    return gone, alive


def reap_children(timeout=3, on_terminate=None):
    # type: (int, TOnTerminate) -> None
    """Tries hard to terminate and ultimately kill all the children of this process."""
    if on_terminate is None:
        def on_terminate(proc):
            print(f"Killed child process {proc} with exit code {proc.returncode}")

    procs = psutil.Process().children(recursive=True)
    kill_hard(procs, timeout=timeout, on_terminate=on_terminate)


def merge_directory(src, dst, delete_src=False):
    """Merge src directory to dst.
    """
    if not src.is_dir():
        raise FileNotFoundError(f'Source directory not found or is not a directory: {src}')
    if dst.exists() and not dst.is_dir():
        raise FileExistsError(f'Destination exists and is not a directory: {dst}')

    if not dst.exists():
        shutil.move(str(src), str(dst))
        return

    for item in src.iterdir():
        dstitem = dst / item.name
        if item.is_dir():
            merge_directory(item, dstitem)
        else:
            shutil.move(str(item), str(dstitem))

    if delete_src:
        src.rmdir()


@contextmanager
def atomic_directory(final_dest):
    # type: (Union[str, 'Path']) -> 'Path'
    """Create a temporary directory, when finishing, move it to final dest.
    When raises exception, final dest not affected.

    :param final_dest The final destination to move to. It is always assumed as a directory. Raises exception
    if 'final_dest' exists and is file.
    :type final_dest Union[str, Path]
    """
    from .compatiblity import tempfile, pathlib
    Path = pathlib.Path

    final_dest = Path(final_dest)

    final_dest.mkdir(exist_ok=True, parents=True)

    with tempfile.TemporaryDirectory(dir='/dev/shm') as name:
        name = Path(name)

        logger.info(f'Using temporary directory: {name!s}')
        try:
            yield name
        except Exception:
            logger.exception("Caught exception:")
        finally:
            # move content to final dest
            if final_dest.exists() and not final_dest.is_dir():
                raise FileExistsError(f"Destination exists and is not a directory: {final_dest}")
            merge_directory(name, final_dest)


def format_secs(sec):
    # type: (Union[float, int]) -> str
    """Format seconds"""
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return f'{d:02}:{h:02}:{m:02}:{s:02.6f}'


def maybe_path(str_path):
    # type: (Union[str, 'Path']) -> 'Path'
    """Convert str to path, noop if is already path. None safe."""
    if str_path is None:
        return None
    from .compatiblity import pathlib
    Path = pathlib.Path
    return Path(str_path)


def stable_unique(array):
    # type: (Iterable[T]) -> Iterable[T]
    """Make elements in array unique, keeping order"""
    from collections import OrderedDict

    uniq = OrderedDict()
    for elem in array:
        uniq[elem] = 1
    # convert odict_keys object to an generator
    return (el for el in uniq.keys())


def unique(array, stable=False):
    # type: (Iterable[T], bool) -> Iterable[T]
    if stable:
        return stable_unique(array)
    return list(set(array))


def format_timespan(num_seconds, threhold=1):
    """Format timespan in seconds, with maximum presicion"""
    units = ['s', 'ms', 'us', 'ns']
    for unit in units:
        if num_seconds > threhold:
            return f'{num_seconds:.2f}{unit}'
        num_seconds *= 1000
    return f'{num_seconds / 1000:.2f}{unit}'


def snake_to_pascal(snake_str):
    parts = snake_str.split('_')
    return ''.join(map(str.capitalize, parts))


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


__all__ = [
    'is_unix',
    'Popen',
    'SalusError',
    'ServerError',
    'UsageError',
    'eprint',
    'remove_none',
    'remove_prefix',
    'try_with_default',
    'execute',
    'kill_tree',
    'kill_hard',
    'reap_children',
    'atomic_directory',
    'merge_directory',
    'format_secs',
    'maybe_path',
    'unique',
    'format_timespan',
    'snake_to_pascal',
    'str2bool',
]
