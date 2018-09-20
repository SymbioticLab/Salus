from __future__ import division, print_function, absolute_import

import numpy as np

import matplotlib as mpl
from matplotlib import transforms as mtransforms
import matplotlib.pyplot as plt
from matplotlib.dates import SECONDLY, rrulewrapper, RRuleLocator, DateFormatter
from matplotlib.ticker import MultipleLocator, MaxNLocator, FuncFormatter, AutoMinorLocator, Locator, Base
from matplotlib.collections import LineCollection

from contextlib import contextmanager
from os.path import getsize, basename

from tqdm import tqdm

SYMBOLS = {
    'customary': ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'),
    'customary_ext': ('byte', 'kilo', 'mega', 'giga', 'tera', 'peta', 'exa',
                      'zetta', 'iotta'),
    'iec': ('Bi', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi'),
    'iec_ext': ('byte', 'kibi', 'mebi', 'gibi', 'tebi', 'pebi', 'exbi',
                'zebi', 'yobi'),
}


def bytes2human(n, format='%(value).1f %(symbol)s', symbols='customary'):
    """
    Convert n bytes into a human readable string based on format.
    symbols can be either "customary", "customary_ext", "iec" or "iec_ext",
    see: http://goo.gl/kTQMs

      >>> bytes2human(0)
      '0.0 B'
      >>> bytes2human(0.9)
      '0.0 B'
      >>> bytes2human(1)
      '1.0 B'
      >>> bytes2human(1.9)
      '1.0 B'
      >>> bytes2human(1024)
      '1.0 K'
      >>> bytes2human(1048576)
      '1.0 M'
      >>> bytes2human(1099511627776127398123789121)
      '909.5 Y'

      >>> bytes2human(9856, symbols="customary")
      '9.6 K'
      >>> bytes2human(9856, symbols="customary_ext")
      '9.6 kilo'
      >>> bytes2human(9856, symbols="iec")
      '9.6 Ki'
      >>> bytes2human(9856, symbols="iec_ext")
      '9.6 kibi'

      >>> bytes2human(10000, "%(value).1f %(symbol)s/sec")
      '9.8 K/sec'

      >>> # precision can be adjusted by playing with %f operator
      >>> bytes2human(10000, format="%(value).5f %(symbol)s")
      '9.76562 K'
    """
    n = int(n)
    sign = ''
    if n < 0:
        sign = '-'
        n = -n
    symbols = SYMBOLS[symbols]
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i + 1) * 10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return sign + format % locals()
    return sign + format % dict(symbol=symbols[0], value=n)


def human2bytes(s):
    """
    Attempts to guess the string format based on default symbols
    set and return the corresponding bytes as an integer.
    When unable to recognize the format ValueError is raised.

      >>> human2bytes('0 B')
      0
      >>> human2bytes('1 K')
      1024
      >>> human2bytes('1 M')
      1048576
      >>> human2bytes('1 Gi')
      1073741824
      >>> human2bytes('1 tera')
      1099511627776

      >>> human2bytes('0.5kilo')
      512
      >>> human2bytes('0.1  byte')
      0
      >>> human2bytes('1 k')  # k is an alias for K
      1024
      >>> human2bytes('12 foo')
      Traceback (most recent call last):
          ...
      ValueError: can't interpret '12 foo'
    """
    init = s
    num = ""
    while s and s[0:1].isdigit() or s[0:1] == '.':
        num += s[0]
        s = s[1:]
    num = float(num)
    letter = s.strip()
    for name, sset in SYMBOLS.items():
        if letter in sset:
            break
    else:
        if letter == 'k':
            # treat 'k' as an alias for 'K' as per: http://goo.gl/kTQMs
            sset = SYMBOLS['customary']
            letter = letter.upper()
        else:
            raise ValueError("can't interpret %r" % init)
    prefix = {sset[0]: 1}
    for i, s in enumerate(sset[1:]):
        prefix[s] = 1 << (i + 1) * 10
    return int(num * prefix[letter])


class BytesLocator(Locator):
    """
    Set a tick on whole values of B or KB or MB, etc.
    """
    def __init__(self, maxn=15, minn=3):
        self._maxn = maxn
        self._minn = minn

    def set_params(self, maxn=None, minn=None):
        """Set parameters within this locator."""
        if maxn is not None:
            self._maxn = maxn
        if minn is not None:
            self._minn = minn

    def __call__(self):
        'Return the locations of the ticks'
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def _select_base(self, vdmin, vdmax):
        interval = abs(vdmax - vdmin)
        scale = 0
        while interval > (1024 ** scale):
            scale += 1
        scale -= 1
        base = 1024 ** scale

        # scale down one level for cases like interval = 1.1TB, and base is 1TB
        if interval // base < self._minn:
            scale -= 1
            base = 1024 ** scale

        # use a similar logic as MaxNLocator to determin the factor on base
        good_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        for factor in good_factors:
            assert factor < 1024
            if interval // (factor * base) <= self._maxn:
                return Base(factor * base)

        # should not happen
        return Base(base * 1024)

    def tick_values(self, vmin, vmax):
        if vmax < vmin:
            vmin, vmax = vmax, vmin

        base = self._select_base(vmin, vmax)

        vmin = base.ge(vmin)
        base = base.get_base()
        n = (vmax - vmin + 0.001 * base) // base
        locs = vmin - base + np.arange(n + 3) * base
        return self.raise_if_exceeds(locs)

    def view_limits(self, dmin, dmax):
        """
        Set the view limits to the nearest multiples of base that
        contain the data
        """
        base = self._select_base(dmin, dmax)
        if mpl.rcParams['axes.autolimit_mode'] == 'round_numbers':
            vmin = base.le(dmin)
            vmax = base.ge(dmax)
            if vmin == vmax:
                vmin -= 1
                vmax += 1
        else:
            vmin = dmin
            vmax = dmax

        return mtransforms.nonsingular(vmin, vmax)


def cleanup_axis_bytes(axis, maxN=10, minorN=2, format='%(value).1f %(symbol)s'):
    # axis.set_major_locator(MaxNLocator(steps=[1, 2, 4, 6, 8, 10]))
    #dmin, dmax = axis.get_data_interval()
    #interval = dmax - dmin
    #scale = 0
    #while interval > (1024 ** scale):
    #    scale += 1
    #scale -= 1
    #base = 1024 ** scale
    #while interval // base > 5:
    #    base += 1024 ** scale
    #locator = MultipleLocator(base=base)

    locator = BytesLocator(maxn=maxN)

    axis.set_major_locator(locator)
    axis.set_minor_locator(AutoMinorLocator(minorN))
    axis.set_major_formatter(FuncFormatter(lambda x, pos: bytes2human(x, format=format)))
    return axis


def cleanup_axis_timedelta(axis, formatter=None):
    if formatter is None:
        def formatter(x, pos):
            return '{:.1f}'.format(x)
    axis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=2))
    axis.set_major_formatter(FuncFormatter(formatter))
    return axis


def cleanup_axis_datetime(axis, rule=None):
    if rule is None:
        rule = rrulewrapper(SECONDLY, interval=1)
    loc = RRuleLocator(rule)
    fmt = DateFormatter('%M:%S')
    axis.set_major_locator(loc)
    axis.set_major_formatter(fmt)
    return axis


def axhlines(ys, xmin=0, xmax=1, **kwargs):
    """
    Draw horizontal lines across plot
    :param ys: A scalar, list, or 1D array of y position in data coordinates
        of the horizontal line.
    :param xmin: A scalar, list, or 1D array. Should be between 0 and 1,
        0 being the far left of the plot, 1 the far right of the plot.
    :param xmax: A scalar, list, or 1D array. Should be between 0 and 1,
        0 being the far left of the plot, 1 the far right of the plot.
    :param kwargs: Valid kwargs are
        :class:`~matplotlib.collections.LineCollection` properties, with the
        exception of 'transform'.
    :return: The LineCollection object corresponding to the lines.
    """
    if "transform" in kwargs:
        raise ValueError("'transform' is not allowed as a kwarg;"
                         + "axhlines generates its own transform.")

    # prepare data
    ys = np.array((ys, ) if np.isscalar(ys) else ys, copy=False)
    xmins = np.array((xmin, ) if np.isscalar(xmin) else xmin, copy=False)
    xmaxs = np.array((xmax, ) if np.isscalar(xmax) else xmax, copy=False)

    if len(ys) > 1:
        if len(xmins) == 1:
            xmins = np.repeat(xmins, len(ys))
        if len(xmaxs) == 1:
            xmaxs = np.repeat(xmaxs, len(ys))

    if len(xmins) != len(xmaxs) or len(xmins) != len(ys):
        raise ValueError("Incompatible data")

    # prepare the ax
    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()

    # prepare colors
    colors = kwargs.pop('colors', None)
    if colors is None:
        cycle_props = next(ax._get_lines.prop_cycler)
        colors = cycle_props.pop('color', None)

    # prepare trans
    trans = ax.get_yaxis_transform(which='grid')
    # prepare lines
    lines = [
        ([xmin, y], [xmax, y])
        for xmin, xmax, y in zip(xmins, xmaxs, ys)
    ]
    lc = LineCollection(lines, transform=trans, colors=colors, **kwargs)
    ax.add_collection(lc)
    ax.autoscale_view(scalex=False, scaley=True)

    return lc


def axvlines(xs, ymin=0, ymax=1, **kwargs):
    """
    Draw vertical lines across plot
    :param xs: A scalar, list, or 1D array of x position in data coordinates
        of the horizontal line.
    :param ymin: A scalar, list, or 1D array. Should be between 0 and 1,
        0 being the bottom of the plot, 1 the top of the plot.
    :param ymax: A scalar, list, or 1D array. Should be between 0 and 1,
        0 being the bottom of the plot, 1 the top of the plot.
    :param kwargs: Valid kwargs are
        :class:`~matplotlib.collections.LineCollection` properties, with the
        exception of 'transform'.
    :return: The LineCollection object corresponding to the lines.
    """
    if "transform" in kwargs:
        raise ValueError("'transform' is not allowed as a kwarg;"
                         + "axvlines generates its own transform.")

    # prepare data
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    ymins = np.array((ymin, ) if np.isscalar(ymin) else ymin, copy=False)
    ymaxs = np.array((ymax, ) if np.isscalar(ymax) else ymax, copy=False)

    if len(xs) > 1:
        if len(ymins) == 1:
            ymins = np.repeat(ymins, len(xs))
        if len(ymaxs) == 1:
            ymaxs = np.repeat(ymaxs, len(xs))

    if len(ymins) != len(ymaxs) or len(ymins) != len(xs):
        raise ValueError("Incompatible data")

    # prepare the ax
    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()

    # prepare colors
    colors = kwargs.pop('colors', None)
    if colors is None:
        cycle_props = next(ax._get_lines.prop_cycler)
        colors = cycle_props.pop('color', None)

    # prepare trans
    trans = ax.get_xaxis_transform(which='grid')
    # prepare lines
    lines = [
        ([x, ymin], [x, ymax])
        for x, ymin, ymax in zip(xs, ymins, ymaxs)
    ]
    lc = LineCollection(lines, transform=trans, colors=colors, **kwargs)
    ax.add_collection(lc)
    ax.autoscale_view(scalex=True, scaley=False)

    return lc


@contextmanager
def pbopen(filename):
    total = getsize(filename)
    pb = tqdm(total=total, unit="B", unit_scale=True,
              desc=basename(filename), miniters=1,
              ncols=80, ascii=True)

    def wrapped_line_iterator(fd):
        processed_bytes = 0
        for line in fd:
            processed_bytes += len(line)
            # update progress every MB.
            if processed_bytes >= 1024 * 1024:
                pb.update(processed_bytes)
                processed_bytes = 0

            yield line

        # finally
        pb.update(processed_bytes)
        pb.close()

    with open(filename) as fd:
        yield wrapped_line_iterator(fd)

def cdf(X, ax=None, **kws):
    if ax is None:
        _, ax = plt.subplots()
    n = np.arange(1,len(X)+1) / np.float(len(X))
    Xs = np.sort(X)
    ax.step(Xs, n, **kws)
    ax.set_ylim(0, 1)
    return ax


class DraggableLine:
    def __init__(self, ax, orientation, XorY, **kwargs):
        self.ax = ax
        self.c = ax.get_figure().canvas
        self.o = orientation
        self.XorY = XorY

        lineArgs = {
            'picker': 5
        }
        lineArgs.update(kwargs)

        if orientation == "h":
            self.line = self.ax.axhline(self.XorY, **lineArgs)
        elif orientation == "v":
            self.line = self.ax.axvline(self.XorY, **lineArgs)
        else:
            assert False

        self.c.draw_idle()
        self.sid = self.c.mpl_connect('pick_event', self.onClick)

    def onClick(self, event):
        if event.artist == self.line:
            print("line selected ", event.artist)
            self.follower = self.c.mpl_connect("motion_notify_event", self.onMotion)
            self.releaser = self.c.mpl_connect("button_press_event", self.onRelease)

    def onMotion(self, event):
        if self.o == "h":
            self.XorY = self.line.get_ydata()[0]
            self.line.set_ydata([event.ydata, event.ydata])
        else:
            self.XorY = self.line.get_xdata()[0]
            self.line.set_xdata([event.xdata, event.xdata])
        self.c.draw_idle()

    def onRelease(self, event):
        if self.o == "h":
            self.XorY = self.line.get_ydata()[0]
        else:
            self.XorY = self.line.get_xdata()[0]

        print (self.XorY)

        self.c.mpl_disconnect(self.releaser)
        self.c.mpl_disconnect(self.follower)


def draggable_line(data, orientation='h', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    dline = DraggableLine(ax, orientation, data)
    return dline
