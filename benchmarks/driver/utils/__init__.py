# -*- coding: future_fstrings -*-
from __future__ import absolute_import, print_function, division
from builtins import str

import os
import platform
import signal
import sys
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Union, TypeVar, Mapping

import psutil
from psutil import Popen

from .compatiblity import TemporaryDirectory, Path

TOnTerminate = Callable[[psutil.Process], None]
TKey = TypeVar('TKey')
TValue = TypeVar('TValue')
T = TypeVar('T')

# Whether we are on unix
is_unix = platform.system() != "Windows"


class SalusError(Exception):
    """Base class for all exceptions"""

    def __init__(self, *args, **kwargs):
        # type: (*Any, **Any) -> None
        message = args[0] if len(args) > 0 else ""
        if not isinstance(message, str):
            raise ValueError("First argument to SalusError must be a str")

        message.format(*args[1:], **kwargs)
        super().__init__(message)


class ServerError(SalusError):
    """Exception related to salus server"""
    pass


def eprint(*args, **kwargs):
    # type: (*Any, **Any) -> None
    """Print to stderr"""
    print(*args, file=sys.stderr, **kwargs)


def remove_none(d):
    # type: (Mapping[TKey, TValue]) -> Mapping[TKey, TValue]
    """Remove None value from dict"""
    return {k: v for k, v in d if v is not None}


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

    return psutil.Popen(*args, **kwargs)


# @formatter:off
def kill_tree(leader,               # type: Union[int, psutil.Process]
              sig=signal.SIGTERM,   # type: int
              include_leader=True,  # type: bool
              timeout=None,         # type: int
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
    for p in children:
        p.send_signal(sig)
    gone, alive = psutil.wait_procs(children, timeout=timeout, callback=on_terminate)
    return gone, alive


def kill_hard(procs, timeout=3, on_terminate=None):
    # type: (Iterable[psutil.Process], int, TOnTerminate) -> None
    """Tries hard to terminate and ultimately kill the processes."""
    # send SIGTERM
    for p in procs:
        p.terminate()
    gone, alive = psutil.wait_procs(procs, timeout=timeout, callback=on_terminate)
    if alive:
        # send SIGKILL
        for p in alive:
            eprint("process {} survived SIGTERM, trying SIGKILL".format(p))
            p.kill()
        gone, alive = psutil.wait_procs(alive, timeout=timeout, callback=on_terminate)
        if alive:
            # give up
            for p in alive:
                eprint("process {} survived SIGKILL, giving up".format(p))


def reap_children(timeout=3, on_terminate=None):
    # type: (int, TOnTerminate) -> None
    """Tries hard to terminate and ultimately kill all the children of this process."""
    if on_terminate is None:
        def on_terminate(proc):
            print("Killed child process {} with exit code {}".format(proc, proc.returncode))

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
        src.rename(dst)
        return

    for item in src.iterdir():
        dstitem = dst / item.name
        if item.is_dir():
            merge_directory(item, dstitem)
        else:
            item.rename(dstitem)

    if delete_src:
        src.rmdir()


@contextmanager
def atomic_directory(final_dest, merge=True):
    # type: (Union[str, Path], bool) -> Path
    """Create a temporary directory, when finishing, move it to final dest.
    When raises exception, final dest not affected.

    :param final_dest The final destination to move to. It is always assumed as a directory. Raises exception
    if 'final_dest' exists and is file.
    :type final_dest Union[str, Path]
    :param merge When 'final_dest' exists, move contents in the temporary directory into existing 'final_dest'.
    Otherwise raise exception if 'final_dest' is directory and not exmpty.
    :type merge bool
    """
    final_dest = Path(final_dest)

    final_dest.mkdir(exist_ok=True, parents=True)

    with TemporaryDirectory as name:
        name = Path(name)

        yield name

        # move content to final dest
        if final_dest.exists() and not final_dest.is_dir():
            raise FileExistsError(f"Destination exists and is not a directory: {final_dest}")
        if merge:
            merge_directory(name, final_dest)
        else:
            name.rename(final_dest)


__all__ = [
    'Popen',
    'SalusError',
    'ServerError',
    'eprint',
    'remove_none',
    'remove_prefix',
    'try_with_default',
    'execute',
    'kill_tree',
    'kill_hard',
    'reap_children',
    'atomic_directory',
]
