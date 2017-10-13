#! /bin/env python
from __future__ import print_function, absolute_import, division

import os
import sys
from collections import namedtuple, defaultdict
from functools import wraps

try:
  from pathlib import Path
except ImportError:
  from pathlib2 import Path  # python 2 backport

import matplotlib.pyplot as plt

import parse_log as pl
import parse_nvvp as pn


ConfigT = namedtuple('ConfigT', 'save_dir, log_dir')

cases = defaultdict(list)


def plotter(name):
    def plotter_decorator(func):
        filename = func.__name__.lstrip('plot_').replace('_', '-')

        @wraps(func)
        def wrapped(config):
            local_dir = os.path.join(config.log_dir, name)
            log_file = os.path.join(local_dir, 'alloc.output')
            iter_file = os.path.join(local_dir, 'mem-iter.output')
            logs = pl.load_file(log_file)
            iters = pn.parse_iterations(iter_file)
            return func(config, local_dir, logs, iters)

        cases[name].append((wrapped, filename))

        return wrapped

    return plotter_decorator


@plotter('conv25')
def plot_mem_conv25(config, local_dir, logs, iters):
    def smoother(ss):
        return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[1:11],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    fig.axes[-1].set_title('Memory usage of CONV2 with batch size 25')
    return fig


@plotter('conv50')
def plot_mem_conv50(config, local_dir, logs, iters):
    def smoother(ss):
        return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[1:11],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    fig.axes[-1].set_title('Memory usage of CONV2 with batch size 50')
    return fig


@plotter('conv100')
def plot_mem_conv100(config, local_dir, logs, iters):
    def smoother(ss):
        return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[1:11],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    fig.axes[-1].set_title('Memory usage of CONV2 with batch size 100')
    return fig


@plotter('mnist25')
def plot_mem_mnist25(config, local_dir, logs, iters):
    def smoother(ss):
        return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[1:11],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    fig.axes[-1].set_title('Memory usage of CONV4 with batch size 25')
    return fig


@plotter('mnist50')
def plot_mem_mnist50(config, local_dir, logs, iters):
    def smoother(ss):
        return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[1:11],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    fig.axes[-1].set_title('Memory usage of CONV4 with batch size 50')
    return fig


@plotter('mnist100')
def plot_mem_mnist100(config, local_dir, logs, iters):
    def smoother(ss):
        return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[1:11],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    fig.axes[-1].set_title('Memory usage of CONV4 with batch size 100')
    return fig


@plotter('vgg25')
def plot_mem_vgg25(config, local_dir, logs, iters):
    def smoother(ss):
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[1:11],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    fig.axes[-1].set_title('Memory usage of VGG16 with batch size 25')
    return fig


@plotter('vgg50')
def plot_mem_vgg50(config, local_dir, logs, iters):
    def smoother(ss):
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[1:11],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    fig.axes[-1].set_title('Memory usage of VGG16 with batch size 50')
    return fig


@plotter('vgg100')
def plot_mem_vgg100(config, local_dir, logs, iters):
    def smoother(ss):
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[1:11],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    fig.axes[-1].set_title('Memory usage of VGG16 with batch size 100')
    return fig


@plotter('res25')
def plot_mem_res25(config, local_dir, logs, iters):
    def smoother(ss):
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[1:11],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    fig.axes[-1].set_title('Memory usage of ResNet with batch size 25')
    return fig


@plotter('res50')
def plot_mem_res50(config, local_dir, logs, iters):
    def smoother(ss):
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[1:11],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    fig.axes[-1].set_title('Memory usage of ResNet with batch size 50')
    return fig


@plotter('res75')
def plot_mem_res75(config, local_dir, logs, iters):
    def smoother(ss):
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[1:11],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    fig.axes[-1].set_title('Memory usage of ResNet with batch size 75')
    return fig


def main():
    config = ConfigT(save_dir='figures', log_dir='logs')

    Path(config.save_dir).mkdir(exist_ok=True)

    args = sys.argv[1:]
    showFigure = False
    if len(args) >= 1 and args[0] == 'show':
        showFigure = True
        args = args[1:]

    names = args if len(args) > 0 else cases.keys()
    for name in names:
        for f, filename in cases[name]:
            print("Generating " + filename)
            fig = f(config)
            if showFigure:
                plt.show()
            else:
                fig.savefig(os.path.join(config.save_dir, filename + '.pdf'), transparent=True)


if __name__ == '__main__':
    main()
