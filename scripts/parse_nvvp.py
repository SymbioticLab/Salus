from __future__ import print_function, absolute_import, division

import re
from datetime import datetime, timedelta
from nvvpreader import NvvpReader

import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt


def axhlines(ys, **plot_kwargs):
    """
    Draw horizontal lines across plot
    :param ys: A scalar, list, or 1D array of vertical offsets
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    if 'ax' in plot_kwargs:
        ax = plot_kwargs['ax']
        del plot_kwargs['ax']
    else:
        ax = plt.gca()
    ys = np.array((ys, ) if np.isscalar(ys) else ys, copy=False)
    lims = ax.get_xlim()
    y_points = np.repeat(ys[:, None], repeats=3, axis=1).flatten()
    x_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(ys), axis=0).flatten()
    plot = ax.plot(x_points, y_points, scalex=False, **plot_kwargs)
    return plot


def axvlines(xs, **plot_kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    print(plot_kwargs)
    if 'ax' in plot_kwargs:
        ax = plot_kwargs['ax']
        del plot_kwargs['ax']
    else:
        ax = plt.gca()
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    lims = ax.get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(xs), axis=0).flatten()
    plot = ax.plot(x_points, y_points, scaley=False, **plot_kwargs)
    return plot


def load_file(path):
    return NvvpReader(path, True)


ptn_iter = re.compile(r"""(?P<timestamp>.+): step (\d+), loss .*; (?P<duration>[\d.]+) sec/batch""")


def parse_iterations(path):
    iterations = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')

            m = ptn_iter.match(line)
            if m:
                timestamp = datetime.strptime(m.group('timestamp'), '%Y-%m-%d %H:%M:%S.%f')
                start = timestamp - timedelta(seconds=float(m.group('duration')))
                iterations.append((start, timestamp))
    return iterations


def active_wrap_trend(reader, iter_times):
    data = []
    df = reader.kernels
    df = df[df['event'] == 'active_warps']
    for start, end, dur, val in zip(df['start'], df['end'], df['duration'], df['event_val']):
        avg = val / (dur / 1e6)
        data.append({
            'timestamp': start,
            'active_warps': avg
        })
        data.append({
            'timestamp': end,
            'active_warps': -avg
        })

    df = pd.DataFrame(data).set_index('timestamp').sort_index()

    ddf = df.cumsum()

    # drop first iteration
    starts, ends = zip(*iter_times[1:])
    ddf = ddf.loc[starts[0]:ends[-1]]

    ax = ddf.plot()
    ax.grid('on')
    ax.set_ylabel('(Estimated) Warp Activation Per Millisecond')
    ax.set_xlabel('Time')
    ax.set_title('Estimated Warp Activation (Dropped first iteration)')

    axvlines(starts, ax=ax, linestyle='--', color='lightgreen', label='Iteration Begin')
    axvlines(ends, ax=ax, linestyle='--', color='r', label='Iteration End')

    ax.autoscale(axis='x')
    ax.set_ylim(bottom=-10)
    ax.legend()
    ax.figure.tight_layout()

    return df, ax
