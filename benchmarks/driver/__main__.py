# -*- coding: future_fstrings -*-
from __future__ import absolute_import, print_function, division, unicode_literals
from builtins import super

import errno
import sys
import logging
import shutil
from timeit import default_timer
from importlib import import_module
from absl import flags, command_name
from typing import Sequence

from .workload import WorkloadTemplate
from .utils import UsageError, eprint, format_secs, reap_children
from .utils.compatiblity import pathlib
from .utils import prompt

Path = pathlib.Path
FLAGS = flags.FLAGS
logger = logging.getLogger('cli')


class HelpFlag(flags.BooleanFlag):
    """Special boolean flag that displays usage and raises SystemExit."""
    NAME = 'help'
    SHORT_NAME = 'h'

    def __init__(self):
        super().__init__(self.NAME, False, 'show this help', short_name=self.SHORT_NAME, allow_hide_cpp=True)

    def parse(self, arg):
        if arg:
            usage(shorthelp=True, writeto_stdout=True)
            # Advertise --helpfull on stdout, since usage() was on stdout.
            print()
            print('Try --helpfull to get a list of all flags.')
        sys.exit(1)


class HelpfullFlag(flags.BooleanFlag):
    """Display help for flags in the main module and all dependent modules."""

    def __init__(self):
        super().__init__('helpfull', False, 'show full help', allow_hide_cpp=True)

    def parse(self, arg):
        if arg:
            usage(writeto_stdout=True)
        sys.exit(1)


flags.DEFINE_string('save_dir', 'scripts/templogs', 'Output direcotry')
flags.DEFINE_boolean('clear_save', False, 'Remove anything previously in the output directory')
flags.DEFINE_multi_string('extra_wl', [], 'Path to the CSV containing extra workload info')
flags.DEFINE_multi_string('extra_mem', [], 'Path to the CSV containing extra mem info')
flags.DEFINE_string('force_preset', None, 'Force to use specific server config preset')
flags.DEFINE_flag(HelpFlag())
flags.DEFINE_flag(HelpfullFlag())


def usage(shorthelp=False, writeto_stdout=False, detailed_error=None,
          exitcode=None):
    """Writes __main__'s docstring to stderr with some help text.
    Args:
      shorthelp: bool, if True, prints only flags from the main module,
          rather than all flags.
      writeto_stdout: bool, if True, writes help message to stdout,
          rather than to stderr.
      detailed_error: str, additional detail about why usage info was presented.
      exitcode: optional integer, if set, exits with this status code after
          writing help.
    """
    if writeto_stdout:
        stdfile = sys.stdout
    else:
        stdfile = sys.stderr

    doc = sys.modules['__main__'].__doc__
    if not doc:
        doc = f'USAGE: python -m benchmarks.driver <exp> [flags]\n\n'
        doc = flags.text_wrap(doc, indent='       ', firstline_indent='')
    if shorthelp:
        flag_str = FLAGS.main_module_help()
    else:
        flag_str = str(FLAGS)
    try:
        stdfile.write(doc)
        if flag_str:
            stdfile.write('\nflags:\n')
            stdfile.write(flag_str)
        stdfile.write('\n')
        if detailed_error is not None:
            stdfile.write('\n%s\n' % detailed_error)
    except IOError as e:
        # We avoid printing a huge backtrace if we get EPIPE, because
        # "foo.par --help | less" is a frequent use case.
        if e.errno != errno.EPIPE:
            raise
    if exitcode is not None:
        sys.exit(exitcode)


def parse_flags_with_usage(args, known_only=False):
    """Tries to parse the flags, print usage, and exit if unparseable.
    Args:
      args: [str], a non-empty list of the command line arguments including
          program name.
      known_only: bool, whether ignore unknown arguments.
    Returns:
      [str, str], a tuple of program name and a list of any remaining command line arguments after parsing
      flags.
    """
    try:
        parsed = FLAGS(args, known_only=known_only)
        return parsed[0], parsed[1:]
    except flags.Error as ex:
        eprint(f'FATAL Flags parsing error: {ex}')
        eprint('Pass --help or --helpfull to see help on flags.')
        sys.exit(1)


def parse_expname(args):
    if not args:
        raise ValueError('Must be non-empty list')

    exp = None
    # skip program name at 0
    remaining = [args[0]]
    for s in args[1:]:
        if exp is None and not s.startswith('-'):
            exp = s
        else:
            remaining.append(s)
    if exp is None:
        raise UsageError('Too few command line arguments.')
    return exp, remaining


def main():
    # type: (Sequence[str]) -> None
    # find first argument not starting with dash
    exp, argv = parse_expname(sys.argv)

    expm = import_module('benchmarks.exps.' + exp)
    logger.info(f'Running experiment: {expm.__name__}')

    # Parse FLAGS now in case any module also defines some,
    progname, argv = parse_flags_with_usage(argv)

    # process any variables and sanity check
    for wl in FLAGS.extra_wl:
        WorkloadTemplate.load_jctcsv(wl)

    for mem in FLAGS.extra_mem:
        WorkloadTemplate.load_memcsv(mem)

    save_dir = (Path(FLAGS.save_dir) / exp).resolve(strict=False)
    if save_dir.is_dir() and FLAGS.clear_save:
            print(f"The following paths will be removed:")
            print(f'  {save_dir!s}')
            for p in sorted(save_dir.rglob('*')):
                print(f'  {p!s}')
            ch = prompt.choose('What would you like to do?',
                               choices=[
                                   ('r', 'Remove'),
                                   ('a', 'Abort'),
                                   ('s', 'Skip without remove'),
                               ])
            if ch == 'r':
                shutil.rmtree(str(save_dir))
            elif ch == 'a':
                return

    save_dir.mkdir(exist_ok=True, parents=True)
    FLAGS.save_dir = save_dir
    logger.info(f'Saving log files to: {FLAGS.save_dir!s}')

    start = default_timer()
    expm.main(argv)
    dur = default_timer() - start
    logger.info(f'Experiment finished in {format_secs(dur)}')


def initialize_logging():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')


if __name__ == '__main__':
    command_name.make_process_name_useful()
    try:
        initialize_logging()
        sys.exit(main())
    except UsageError as error:
        usage(shorthelp=True, detailed_error=error, exitcode=error.exitcode)
    finally:
        reap_children()

