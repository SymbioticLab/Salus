from __future__ import division, print_function, absolute_import

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.dates import SECONDLY, rrulewrapper, RRuleLocator, DateFormatter
from matplotlib.ticker import MaxNLocator, FuncFormatter


def cleanup_axis_timedelta(axis, formatter=None):
    if formatter is None:
        def formatter(x, pos):
            return '{:.1f}'.format(x / 1e6)
    axis.set_major_formatter(FuncFormatter(formatter))
    return axis


def cleanup_axis_datetime(axis):
    rule = rrulewrapper(SECONDLY, interval=1)
    loc = RRuleLocator(rule)
    fmt = DateFormatter('%M:%S')
    axis.set_major_locator(loc)
    axis.set_major_formatter(fmt)
    return axis


def human_readable(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def cleanup_axis_bytes(axis):
    axis.set_major_locator(MaxNLocator(steps=[1, 2, 4, 6, 8, 10]))
    axis.set_major_formatter(FuncFormatter(lambda x, pos: human_readable(x)))
    return axis


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
