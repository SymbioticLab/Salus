from __future__ import absolute_import, print_function, division

from importlib import import_module
from absl import app, flags
from typing import Sequence

from .workload import WorkloadTemplate
from .utils.compatiblity import Path

FLAGS = flags.FLAGS

flags.DEFINE_string('build_dir', '../build', 'Build directory')
flags.DEFINE_string('save_dir', 'templogs', 'Output direcotry')
flags.DEFINE_string('extra_wl', None, 'Path to the CSV containing extra workload info')


def main(argv):
    # type: (Sequence[str]) -> None
    # ignore the first module name
    argv = argv[1:]

    if len(argv) < 1:
        raise app.UsageError('Too few command line arguments.')
    action = argv[0]

    # process any variables and sanity check
    if FLAGS.extra_wl is not None:
        WorkloadTemplate.load_extra(FLAGS.extra_wl)

    Path(FLAGS.save_dir).mkdir(exist_ok=True, parents=True)

    exp = import_module('benchmarks.exps.' + action)
    exp.main(argv[1:])


if __name__ == '__main__':
    app.run(main)
