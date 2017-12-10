#! /bin/env python
from __future__ import print_function, absolute_import, division

import os
import argparse
from collections import defaultdict
from functools import wraps

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import parse_perf as pf
import parse_log as pl
import parse_nvvp as pn
import plotutils as pu


cases = defaultdict(list)


def plotter(name, mem=True):
    def plotter_decorator(func):
        filename = func.__name__.lstrip('plot_').replace('_', '-')

        @wraps(func)
        def wrapped(config):
            local_dir = os.path.join(config.log_dir, name)
            if not os.path.isdir(local_dir):
                return None
            log_file = os.path.join(local_dir, 'alloc.output')
            iter_file = os.path.join(local_dir, 'mem-iter.output')

            logs = None
            iters = None

            if mem:
                try:
                    logs = pl.load_file(log_file)
                except IOError:
                    pass

                if os.path.isfile(iter_file):
                    sessstart, iters = pn.parse_iterations(iter_file)

            fig = func(config, local_dir, logs, iters)
            # fig.set_size_inches(2.35, 2.35, forward=True)
            return fig

        cases[name].append((wrapped, filename))

        return wrapped

    return plotter_decorator


# @plotter('conv25')
def plot_mem_conv25(config, local_dir, logs, iters):
    def smoother(ss):
        return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


# @plotter('conv50')
def plot_mem_conv50(config, local_dir, logs, iters):
    def smoother(ss):
        return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


# @plotter('conv100')
def plot_mem_conv100(config, local_dir, logs, iters):
    def smoother(ss):
        return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


# @plotter('mnist25')
def plot_mem_mnist25(config, local_dir, logs, iters):
    def smoother(ss):
        return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


# @plotter('mnist50')
def plot_mem_mnist50(config, local_dir, logs, iters):
    def smoother(ss):
        return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


# @plotter('mnist100')
def plot_mem_mnist100(config, local_dir, logs, iters):
    def smoother(ss):
        return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('vgg11_25')
def plot_mem_vgg11_25(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('vgg11_50')
def plot_mem_vgg11_50(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('vgg11_100')
def plot_mem_vgg11_100(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('vgg16_25')
def plot_mem_vgg16_25(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('vgg16_50')
def plot_mem_vgg16_50(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('vgg16_100')
def plot_mem_vgg16_100(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('vgg19_25')
def plot_mem_vgg19_25(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('vgg19_50')
def plot_mem_vgg19_50(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('vgg19_100')
def plot_mem_vgg19_100(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('inception4_75')
def plot_mem_inception4_75(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('inception4_50')
def plot_mem_inception4_50(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('inception4_25')
def plot_mem_inception4_25(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('inception3_100')
def plot_mem_inception3_100(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('inception3_50')
def plot_mem_inception3_50(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('inception3_25')
def plot_mem_inception3_25(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('overfeat_100')
def plot_mem_overfeat_100(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('overfeat_50')
def plot_mem_overfeat_50(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('overfeat_25')
def plot_mem_overfeat_25(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('alexnet_100')
def plot_mem_alexnet_100(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('alexnet_50')
def plot_mem_alexnet_50(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('alexnet_25')
def plot_mem_alexnet_25(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('googlenet_100')
def plot_mem_googlenet_100(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('googlenet_50')
def plot_mem_googlenet_50(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('googlenet_25')
def plot_mem_googlenet_25(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('resnet152_75')
def plot_mem_resnet152_75(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, ends=iters[0][1], show_avg=False, mem_type='GPU_0_bfc')
    fig.set_size_inches(3.45, 1.75, forward=True)
    return fig


@plotter('resnet152_50')
def plot_mem_resnet152_50(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('resnet152_25')
def plot_mem_resnet152_25(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('resnet101_75')
def plot_mem_resnet101_75(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('resnet101_50')
def plot_mem_resnet101_50(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('resnet101_25')
def plot_mem_resnet101_25(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('resnet50_75')
def plot_mem_resnet50_75(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('resnet50_50')
def plot_mem_resnet50_50(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('resnet50_25')
def plot_mem_resnet50_25(config, local_dir, logs, iters):
    def smoother(ss, ss2):
        print('{}, {}, {}'.format(ss.min(), ss.max(), ss2.mean()))
        return ss
        # return ss.ewm(span=15).mean()

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


# @plotter('ptbT')
def plot_mem_ptbT(config, local_dir, logs, iters):
    def smoother(ss):
        return ss

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


# @plotter('ptbS')
def plot_mem_ptbS(config, local_dir, logs, iters):
    def smoother(ss):
        return ss

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


# @plotter('ptbM')
def plot_mem_ptbM(config, local_dir, logs, iters):
    def smoother(ss):
        return ss

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


# @plotter('ptbL')
def plot_mem_ptbL(config, local_dir, logs, iters):
    def smoother(ss):
        return ss

    df, _, fig = pl.memory_usage(logs, iter_times=iters[0:10],
                                 mem_type='GPU_0_bfc', smoother=smoother)
    return fig


@plotter('case_preemption')
def plot_case_preemption(config, local_dir, logs, iters):
    pf.preprocess(local_dir)
    perfdf = pf.load_file(os.path.join(local_dir, 'sessiter.output'))

    fig = plt.figure()
    spec = mpl.gridspec.GridSpec(2, 1, height_ratios=[1, 3])
    spec.update(hspace=0.2, left=0.16, right=.98, top=.8, bottom=0.18)
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1], sharex=ax0)
    plt.setp(ax0.get_xticklabels(), visible=False)

    df, _ = pf.session_counters(perfdf, colnames=['scheduled'], ax=ax0)

    def smoother(ss):
        sampled = ss.resample('500us').interpolate(method='time')
        print("previous len: {} now: {}".format(len(ss), len(sampled)))
        return sampled

    df, _, _ = pl.memory_usage(logs, mem_type='GPU_0_bfc', per_sess=True,
                               ax=ax1, show_avg=False, smoother=smoother)

    ax0.legend().set_visible(False)
    ax0.set_title('Scheduled Tasks')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=['alexnet_100', 'inception3_25'], ncol=2,
               loc='upper center', bbox_to_anchor=(0.5, -0.21), frameon=False)
    ax0.set_ylabel('# of Tasks')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Memory Usage')
    ax1.set_xlim([0, 115 * 1e9])
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)
    ax0.tick_params(axis='y', labelsize=8)
    return fig


@plotter('case_bigsmall_wc')
def plot_case_bigsmall_wc(config, local_dir, logs, iters):
    pf.preprocess(local_dir)
    perfdf = pf.load_file(os.path.join(local_dir, 'sessiter.output'))

    fig = plt.figure()
    spec = mpl.gridspec.GridSpec(2, 1, height_ratios=[1, 3])
    spec.update(hspace=0.2, left=0.16, right=.98, top=.8, bottom=0.18)
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1], sharex=ax0)
    plt.setp(ax0.get_xticklabels(), visible=False)

    df, _ = pf.session_counters(perfdf, colnames=['scheduled'], ax=ax0)

    def smoother(ss):
        sampled = ss.resample('500us').interpolate(method='time')
        print("previous len: {} now: {}".format(len(ss), len(sampled)))
        return sampled

    df, _, _ = pl.memory_usage(logs, mem_type='GPU_0_bfc', per_sess=True,
                               ax=ax1, show_avg=False, smoother=smoother)

    ax0.set_title('Scheduled Tasks')
    ax0.legend().set_visible(False)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=['alexnet_100', 'inception3_25'], ncol=2,
               loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax0.set_ylabel('# of Tasks')
    ax1.set_title('Memory Usage')
    ax1.set_xlabel('Time (s)')
    ax1.set_xlim([0, 115 * 1e9])
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)
    ax0.tick_params(axis='y', labelsize=8)
    return fig


@plotter('case_study1_diff', mem=False)
def plot_case_study1_diff(config, local_dir, logs, iters):
    with mpl.style.context(('color3')):
        # mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['ed7d31', 'dcedd0', '244185'])
        pf.preprocess(local_dir)
        perfdf = pf.load_file(os.path.join(local_dir, 'sessiter.output'))

        df, fig = pf.session_counters(perfdf, colnames=['counter', 'scheduled', 'pending'], zorders={
            '16262ffc6e3ddddf': 5,
            'cfe7d613e94bcc28': 10,
            '318dbb0290ba058d': 1
        })

        fig.axes[0].legend().remove()
        fig.axes[0].set_title('Aggregate Memory Usage')
        fig.axes[0].set_ylabel('(byte * s)')
        # fig.axes[0].set_yscale('symlog')
        # fig.axes[0].yaxis.set_major_locator(pu.MaxNLocator(nbins=4, min_n_ticks=2))

        fig.axes[1].legend().remove()
        fig.axes[1].set_title('Scheduled Tasks')
        fig.axes[1].set_ylabel('# of Tasks')

        fig.axes[2].set_title('Pending Tasks')
        fig.axes[2].set_ylabel('# of Tasks')

        ax = fig.axes[-1]
        ax.set_xlabel('Time (s)')
        ax.set_xlim(left=0)
        fig.tight_layout(rect=[0, 0.18, 1, 1])
        # fig.subplots_adjust(bottom=.18, left=.16, right=.98, top=.98)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles,
                  labels=['googlenet_100', 'overfeat_50', 'resnet50_50'],
                  # labels=labels,
                  ncol=3,
                  loc='upper center', bbox_to_anchor=(0.42, -0.55), frameon=False)
        fig.set_size_inches(3.45, 3.6, forward=True)
        return fig


@plotter('nested_doll_4res')
def plot_nested_doll_4res(config, local_dir, logs, iters):
    with mpl.style.context(('color4')):
        def smoother(ss):
            sampled = ss.resample('500us').interpolate(method='time')
            print("previous len: {} now: {}".format(len(ss), len(sampled)))
            return sampled

        df, _, fig = pl.memory_usage(logs, mem_type='GPU_0_bfc', per_sess=True,
                                     show_avg=False, smoother=smoother)

        ax = fig.axes[-1]
        ax.legend().remove()
        ax.set_title('Memory Usage')
        # ax.set_title('resnet50_50 of 265,180,170,100 iterations')
        fig.tight_layout(pad=0)
        # fig.adjust(right=.98, top=.85, left=.2, bottom=.2)
        return fig


@plotter('paging_inception_of_vgg')
def plot_paging_inception_of_vgg(config, local_dir, logs, iters):
    with mpl.style.context(('color3')):
        pf.preprocess(local_dir)
        perfdf = pf.load_file(os.path.join(local_dir, 'sessiter.output'))

        fig = plt.figure()
        spec = mpl.gridspec.GridSpec(2, 1, height_ratios=[1, 3])
        spec.update(hspace=0.2, left=0.2, right=.98, top=.8, bottom=0.18)
        ax0 = fig.add_subplot(spec[0])
        ax1 = fig.add_subplot(spec[1], sharex=ax0)
        plt.setp(ax0.get_xticklabels(), visible=False)

        df, _ = pf.session_counters(perfdf, colnames=['scheduled'], ax=ax0)

        def smoother(ss):
            sampled = ss.resample('500us').interpolate(method='time')
            print("previous len: {} now: {}".format(len(ss), len(sampled)))
            return sampled

        df, _, _ = pl.memory_usage(logs, mem_type='GPU_0_bfc', per_sess=True,
                                   ax=ax1, show_avg=False, smoother=smoother)

        ax0.set_title('Scheduled Tasks')
        ax0.legend().set_visible(False)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles, labels=['overfeat_100', 'vgg16_25', 'inception3_100'], ncol=3,
                   loc='upper center', bbox_to_anchor=(0.4, -0.2), frameon=False)
        ax0.set_ylabel('# of Tasks')
        ax1.set_title('Memory Usage')
        ax1.set_xlabel('Time (s)')
        ax1.tick_params(axis='x', labelsize=8)
        ax1.tick_params(axis='y', labelsize=8)
        ax0.tick_params(axis='y', labelsize=8)
        return fig


@plotter('nested_doll_mix5_samelength')
def plot_nested_doll_mix5_samelength(config, local_dir, logs, iters):
    with mpl.style.context(('color5')):
        pf.preprocess(local_dir)
        perfdf = pf.load_file(os.path.join(local_dir, 'sessiter.output'))

        fig = plt.figure()
        spec = mpl.gridspec.GridSpec(2, 1, height_ratios=[1, 3])
        spec.update(hspace=0.2, left=0.18, right=.98, top=.8, bottom=0.22)
        ax0 = fig.add_subplot(spec[0])
        ax1 = fig.add_subplot(spec[1], sharex=ax0)
        plt.setp(ax0.get_xticklabels(), visible=False)

        df, _ = pf.session_counters(perfdf, colnames=['scheduled'], ax=ax0)

        def smoother(ss):
            sampled = ss.resample('50ms').interpolate(method='time')
            print("previous len: {} now: {}".format(len(ss), len(sampled)))
            return sampled

        df, _, _ = pl.memory_usage(logs, mem_type='GPU_0_bfc', per_sess=True,
                                   ax=ax1, show_avg=False, smoother=smoother)

        ax0.set_title('Scheduled Tasks')
        ax0.legend().set_visible(False)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles, labels=['alexnet_100', 'overfeat_50', 'googlenet_50', "inception3_25",
                                            "resnet50_50"], ncol=3,
                   loc='upper center', bbox_to_anchor=(0.48, -0.19), frameon=False)
        ax0.set_ylabel('# of Tasks')
        ax1.set_xlabel('Time (s)')
        ax1.set_title('Memory Usage')
        ax1.tick_params(axis='x', labelsize=8)
        ax1.tick_params(axis='y', labelsize=8)
        ax0.tick_params(axis='y', labelsize=8)
        return fig


@plotter('jctratio')
def plot_jctratio(config, local_dir, logs, iters):
    df = pd.read_csv(os.path.join(local_dir, 'jctratio.csv'))
    # sort columns
    cols = df.columns.tolist()
    cols.sort()
    df = df[cols]

    ax = df.boxplot(showfliers=False)
    ax.set_ylabel('JCT Ratio')
    ax.grid(b=False)
    plt.xticks(rotation=60)
    ax.figure.set_size_inches(3.45, 2)
    ax.figure.tight_layout()
    plt.subplots_adjust(left=.16, right=.98, bottom=.18, top=.98)
    return ax.figure


@plotter('fair20')
def plot_fair20(config, local_dir, logs, iters):
    with mpl.style.context(('grayscale20')):
        def smoother(ss):
            sampled = ss.resample('500us').interpolate(method='time')
            print("previous len: {} now: {}".format(len(ss), len(sampled)))
            return sampled

        df, _, fig = pl.memory_usage(logs, mem_type='GPU_0_bfc', per_sess=True,
                                     show_avg=False)

        ax = fig.axes[-1]
        ax.legend().remove()
        ax.set_title('Memory Usage')
        # ax.set_title('resnet50_50 of 265,180,170,100 iterations')
        fig.tight_layout(pad=0)
        # fig.adjust(right=.98, top=.85, left=.2, bottom=.2)
        return fig


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('cases', nargs='*', metavar='CASE',
                        help='Case names to generate figures',
                        default=cases.keys())
    parser.add_argument('--log_dir',
                        help='Base directory containing logs',
                        default='logs')
    parser.add_argument('--save_dir',
                        help='Directory to generate outputs',
                        default='figures')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show figure instead of generate images')
    config = parser.parse_args(argv)

    Path(config.save_dir).mkdir(exist_ok=True)

    # configure matplotlib style
    if config.show:
        plt.style.use('seaborn')
    else:
        plt.style.use(['seaborn-paper', 'mypaper'])

    for name in config.cases:
        for f, filename in cases[name]:
            print("Generating " + filename)
            fig = f(config)
            if not fig:
                print('Not found')
                continue
            if config.show:
                fig.set_size_inches(6, 6, forward=True)
                plt.show()
            else:
                fig.savefig(os.path.join(config.save_dir, filename + '.pdf'), transparent=True)
                plt.close(fig)


if __name__ == '__main__':
    main()
