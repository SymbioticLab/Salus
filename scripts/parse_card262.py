#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:45:07 2018

@author: peifeng
"""

import itertools
from functools import partial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec

from tqdm import tqdm
import os
import multiprocessing as mp

import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import plotutils as pu
from compmem import load_generic


def load_allocmemmap(path):
    df = load_generic(path, event_filters=['alloc', 'dealloc'])
    df = df.drop(['level', 'loc', 'thread', 'type'], axis=1)

    if 'Memmap' in df:
        df['MemMap'] = df['Memmap']
    return df


def parse_mapstr(mapstr, noColor=False):
    """Parse mapstr, which is like:
        0x7f2a76000000, 2211840, 1, 256;0x7f2a7621c000, 8200192, 0, 1024;

       returns the dataframe and colors
    """
    # split to chunks of different allocator
    chunkdfs = []
    for chunk in mapstr.split('&'):
        if not chunk:
            continue
        if '\t' in chunk:
            name, memmap = chunk.split('\t')
        else:
            name = 'default'
            memmap = chunk

        # change ';' to '\n' and read directly as csv
        mapcsv = memmap.replace(';', '\n')
        df = pd.read_csv(StringIO(mapcsv), header=None, index_col=False,
                         names=['Origin', 'Size', 'InUse', 'Bin'])
        df['Name'] = name

        # convert each column to proper type
        df['Origin'] = df.Origin.apply(int, base=0)
        df['Size'] = pd.to_numeric(df.Size)
        df['InUse'] = pd.to_numeric(df.InUse).astype('bool')
        df['End'] = df.Origin + df.Size
        df['Bin'] = pd.to_numeric(df.Bin)

        st = df.Origin.min()
        df['NOrigin'] = df.Origin - st
        df['NEnd'] = df.End - st

        if not noColor:
            df['Color'] = df.InUse.apply(lambda inuse: 'r' if inuse else 'g')

        chunkdfs.append(df)

    # concat all together
    df = pd.concat(chunkdfs)
    return df


def mem_segments(mapdf):
    # use normalized address space, i.e. starting from zero
    return [
        np.column_stack([col, np.zeros_like(col)])
        for _, col in mapdf[['NOrigin', 'NEnd']].T.iteritems()
    ]


def _preprocess_memmap_row(idxrow, ptr2sess, sessColors):
    idx, row = idxrow
    mapdf = parse_mapstr(row.MemMap, noColor=True)

    def findSess(ptr):
        try:
            evts = ptr2sess.loc[ptr]
        except KeyError:
            return ''
        idx = evts.index.searchsorted(row.timestamp) - 1
        if idx == -1:
            return ''
        if evts.iat[idx, 1] == 'alloc':
            return evts.iat[idx, 0]
        #res = ptr2sess.query('Ptr == @thePtr and timestamp <= @theRow.timestamp')
        #if len(res) and res.iloc[-1].evt == 'alloc':
        #    return res.iloc[-1].Sess
        return ''
    mapdf['Sess'] = mapdf.Origin.map(findSess)

    # set colors
    mapdf['Color'] = mapdf.apply(lambda row: sessColors.get(row.Sess, 'r') if row.InUse else 'g',
                                 axis=1)
    return row.timestamp, mapdf


def preprocess_memmap2(df, sessColors = None, ptr2sess=None, task_per_cpu=1000):
    # only use GPU data
    df = df[df.Allocator.str.contains("GPU")]

    if ptr2sess is None:
        # build a index from (Ptr, timestamp) to Sess
        keys, grps = zip(*[(ptr, grp.set_index('timestamp').sort_index().drop(['Ptr'], axis=1))
                            for ptr, grp in df[['Ptr', 'Sess', 'timestamp', 'evt']].groupby('Ptr')])
        ptr2sess = pd.concat(grps, keys=keys, names=['Ptr'])

    # sessColors
    if sessColors is None:
        sessColors = {}

    # find optimal chunk size
    numCPU = os.cpu_count()
    chunkSize = len(df) // numCPU // task_per_cpu
    if chunkSize == 0:
        chunkSize = 1

    # the process
    with mp.Pool(processes=numCPU) as p,\
            tqdm(total=len(df), desc='Parsing memmaps in parallel', unit='rows') as pb:
        def updater(log):
            pb.update()
            return log

        ilog = (updater(log) for log in
                p.imap_unordered(partial(_preprocess_memmap_row,
                                         ptr2sess=ptr2sess,
                                         sessColors=sessColors),
                                 df.iterrows(),
                                 chunksize=chunkSize)
                )
        maps = list(ilog)
        maps.sort(key=lambda x: x[0])

    return maps


def preprocess_memmap3(df, sessColors = None, ptr2sess=None):
    # only use GPU data
    df = df[df.Allocator.str.contains("GPU")]

    if ptr2sess is None:
        # build a index from (Ptr, timestamp) to Sess
        keys, grps = zip(*[(ptr, grp.set_index('timestamp').sort_index().drop(['Ptr'], axis=1))
                            for ptr, grp in df[['Ptr', 'Sess', 'timestamp', 'evt']].groupby('Ptr')])
        ptr2sess = pd.concat(grps, keys=keys, names=['Ptr'])

    # sessColors
    if sessColors is None:
        sessColors = {}

    # the process
    maps = []
    for idxrow in tqdm(df.iterrows(), total=len(df), desc='Parsing memmaps', unit='rows'):
        maps.append(_preprocess_memmap_row(idxrow, ptr2sess, sessColors))
    return maps


def preprocess_memmap(df, sessColors=None, hdf=None):
    """"Parse all memmap and associate session info to it. Return a list of memmaps"""
    # only use GPU data
    # df = df.query('Allocator.str.contains("GPU")')
    df = df[df.Allocator.str.contains("GPU")]

    maps = []
    ptr2sess = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing memmaps", unit="rows"):
        # first update ptr2sess
        if row.evt == 'alloc':
            ptr2sess[row.Ptr] = row.Sess
        elif row.evt == 'dealloc':
            del ptr2sess[row.Ptr]

        mapdf = parse_mapstr(row.MemMap, noColor=True)
        mapdf['Sess'] = mapdf.Origin.map(ptr2sess).fillna('')

        # set colors
        if sessColors is None:
            sessColors = {}
        mapdf['Color'] = mapdf.apply(lambda row: sessColors.get(row.Sess, 'r') if row.InUse else 'g',
                                     axis=1)

        if hdf is not None:
            maps.append(row.timestamp)
            mapdf.to_hdf(hdf, 'data', format='t', append=True)
        else:
            maps.append((row.timestamp, mapdf))

    if hdf is not None:
        pd.Series(maps).to_hdf(hdf, 'timestamps')

    return maps


def preprocess_memmap_hdf(df, sessColors=None, task_per_cpu=20):
    """"Parse all memmap and associate session info to it. Return a list of memmaps"""
    # only use GPU data
    # df = df.query('Allocator.str.contains("GPU")')
    df = df[df.Allocator.str.contains("GPU")]

    with pd.HDFStore('/opt/desktop/card262.data.h5', complib='bzip2', complevel=9, mode='w') as store:
        ptr2sess = {}
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing memmaps", unit="rows"):
            # first update ptr2sess
            if row.evt == 'alloc':
                ptr2sess[row.Ptr] = row.Sess
            elif row.evt == 'dealloc':
                del ptr2sess[row.Ptr]

            mapdf = parse_mapstr(row.MemMap, noColor=True)
            mapdf['Sess'] = mapdf.Origin.map(ptr2sess).fillna('')

            # set colors
            if sessColors is None:
                sessColors = {}
            mapdf['Color'] = mapdf.apply(lambda row: sessColors.get(row.Sess, 'r') if row.InUse else 'g',
                                         axis=1)

            mapdf['timestamp'] = row.timestamp
            store.append('data', mapdf)


def load_preprocessed_memmap(path):
    with pd.HDFStore(path) as store:
        return [(k, grp.drop('timestamp', axis=1).reset_index(drop=True))
                for k, grp in store['data'].groupby('timestamp')]


def draw_on_lc(mapdf, lc):
    lc.set_segments(mem_segments(mapdf))
    lc.set_color(mapdf.Color)
    return mapdf, lc


def draw_memmap(mapstr, lc=None, ax=None, **kwargs):
    """Draw memmap str using matplotlib
    mapstr is like:
        0x7f2a76000000, 2211840, 1, 256;0x7f2a7621c000, 8200192, 0, 1024;
    """
    if ax is None:
        _, ax = plt.subplots()
        ax.set_xlim([0, 17179869184])
        ax.set_ylim([-0.2, 0.2])
    args = {
        'linewidths': 100
    }
    args.update(kwargs)
    newlc = False
    if lc is None:
        lc = LineCollection([], **args)
        newlc = True

    mapdf = parse_mapstr(mapstr)
    mapdf, lc = draw_on_lc(mapdf, lc)

    ax.set_xlim([mapdf.NOrigin.min(), mapdf.NEnd.max()])

    if newlc:
        ax.add_collection(lc, autolim=False)
        pu.cleanup_axis_bytes(ax.xaxis)
    return lc


class MemmapViewer(object):
    def __init__(self, df, colormap, ptr2sess, label=None, doPreprocess=False, hdf=None):
        self.label = label
        if self.label is None:
            self.label = 'id = {index} timestamp = {timestamp}'

        self.colormap = colormap
        self.ptr2sess = ptr2sess
        self._do_preprocess = doPreprocess
        self._hdf = hdf
        if doPreprocess:
            self.data = preprocess_memmap2(df, self.colormap, self.ptr2sess)
        else:
            self.data = df

        self.fig = plt.gcf()
        self.fig.set_size_inches([25, 15], forward=True)
        gs = gridspec.GridSpec(2, 1, height_ratios=[20,1])


        self._ax = self.fig.add_subplot(gs[0, 0])
        self._ax.set_ylim([-0.2, 0.2])

        # create artists
        self._lc = draw_memmap('0x0, 17179869184, 0, 0;', ax=self._ax)
        self._text = self._ax.text(0.5, 0.8, "",
                             ha='center', va='baseline', fontsize=25,
                             transform=self._ax.transAxes)
        # lc and texts for bins
        self._lcalloc = LineCollection([[(0,0.1), (17179869184, 0.1)]], linewidths=10)
        self._ax.add_collection(self._lcalloc)
        self._textalloc = []

        pu.cleanup_axis_bytes(self._ax.xaxis)

        init_frame = 0
        self._step(init_frame)

        # session legend
        self._ax.legend(handles=[
            mpatches.Patch(color=c, label=sess) for sess, c in self.colormap.items()
        ])

        # add a slider
        axstep = self.fig.add_subplot(gs[1, 0], facecolor='lightgoldenrodyellow')
        self._stepSlider = Slider(axstep, 'ID', 0, len(self.data) - 1,
                                  valinit=init_frame, valfmt='%d', valstep=1,
                                  dragging=True)
        def update(val):
            self._step(int(val))
            self.fig.canvas.draw_idle()
        self._stepSlider.on_changed(update)

        # timer
        self._timer = None

        # listen key press
        self.fig.canvas.mpl_connect('key_press_event', lambda evt: self._keydown(evt))

    def _data(self, i):
        if self._do_preprocess:
            return self.data[i]
        else:
            if self._hdf is None:
                try:
                    row = self.data.iloc[i]
                except AttributeError:
                    row = self.data[i]
                return _preprocess_memmap_row((i, row), self.ptr2sess, self.colormap)
            else:
                ts = self.data[i]
                mapdf = pd.from_hdf(self._hdf, str(ts))
                return ts, mapdf

    def _step(self, i):
        """draw step"""
        ts, mapdf = self._data(i)
        try:
            mapdf, _ = draw_on_lc(mapdf, self._lc)
        except ValueError:
            print('Error when parsing memmap for index at {}'.format(i))
            return

        self._text.set_text(self.label.format(index=i, timestamp=ts))

        # draw some lines showing each allocator...
        nAlloc = len(mapdf.Bin.unique())
        while len(self._textalloc) < nAlloc:
            self._textalloc.append(self._ax.text(0, 0, "", rotation='vertical',
                                                 ha='center', va='baseline'))
        # ... and clear any extra ones
        for t in self._textalloc[nAlloc:]:
            t.set_text('')

        allocSeg = []
        for (bsize, grp), t in zip(mapdf.groupby('Bin'), self._textalloc):
            rgmin, rgmax = grp.NOrigin.min(), grp.NEnd.max()
            yval = -.1
            allocSeg.append([[rgmin, yval], [rgmax, yval]])

            t.set_text('Bin({})'.format(pu.bytes2human(bsize)))
            t.set_position([(rgmin+rgmax) / 2, yval * 1.2])
        self._lcalloc.set_segments(allocSeg)

        self._ax.set_xlim([mapdf.NOrigin.min(), mapdf.NEnd.max()])

    def _keydown(self, evt):
        if evt.key == 'left':
            self.prev()
        elif evt.key == 'right':
            self.next()
        elif evt.key == ' ':
            if self._timer is None:
                self._timer = self.fig.canvas.new_timer(interval=100)
                def autonext():
                    self.next()
                    if self._stepSlider.val == self._stepSlider.valmax:
                        # stop timer
                        self._timer.stop()
                        self._timer = None
                self._timer.add_callback(autonext)
                self._timer.start()
            else:
                self._timer.stop()
                self._timer = None
        else:
            print(evt.key)

    def next(self):
        newval = min(self._stepSlider.val + 1, self._stepSlider.valmax)
        self._stepSlider.set_val(newval)
        self.fig.canvas.draw_idle()

    def prev(self):
        newval = max(self._stepSlider.val - 1, self._stepSlider.valmin)
        self._stepSlider.set_val(newval)
        self.fig.canvas.draw_idle()

    def goto(self, val):
        self._stepSlider.set_val(val)
        self.fig.canvas.draw_idle()


def render_video(df, st=0, length=None):
    colors = [
        #'#e6194b',
        #'#3cb44b',
        '#ffe119',
        '#4363d8',
        '#f58231',
        '#911eb4',
        '#46f0f0',
        '#f032e6',
        '#bcf60c',
        '#fabebe',
        '#008080',
        '#e6beff',
        '#9a6324',
        '#fffac8',
        '#800000',
        '#aaffc3',
        '#808000',
        '#ffd8b1',
        '#000075',
        '#808080',
        '#ffffff',
        '#000000'
    ]
    colormap = {
        sess: c
        for sess, c in zip(df.Sess.unique(), itertools.cycle(colors))
    }
    # build a index from (Ptr, timestamp) to Sess
    print("Building index...")
    keys, grps = zip(*[(ptr, grp.set_index('timestamp').sort_index().drop(['Ptr'], axis=1))
                        for ptr, grp in df[['Ptr', 'Sess', 'timestamp', 'evt']].groupby('Ptr')])
    ptr2sess = pd.concat(grps, keys=keys, names=['Ptr'])


    import matplotlib.animation as manimation
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Memory Allocation', artist='Aetf')
    writer = FFMpegWriter(fps=15, metadata=metadata, bitrate=-1)

    m = MemmapViewer(df, colormap, ptr2sess, doPreprocess=False)
    m.goto(st)
    if length is None:
        length = len(df) - st
    length = min(length, len(df) - st)
    with writer.saving(m.fig, "/opt/desktop/memalloc.mp4", dpi=100):
        for i in tqdm(range(length), desc='Writing frame', unit='frames'):
            m.next()
            writer.grab_frame()


def main():
    df = load_allocmemmap('/opt/desktop/card262/case3/part.output')
    df = df[df.Allocator.str.contains("GPU")]
    colors = [
        #'#e6194b',
        #'#3cb44b',
        '#ffe119',
        '#4363d8',
        '#f58231',
        '#911eb4',
        '#46f0f0',
        '#f032e6',
        '#bcf60c',
        '#fabebe',
        '#008080',
        '#e6beff',
        '#9a6324',
        '#fffac8',
        '#800000',
        '#aaffc3',
        '#808000',
        '#ffd8b1',
        '#000075',
        '#808080',
        '#ffffff',
        '#000000'
    ]
    colormap = {
        sess: c
        for sess, c in zip(df.Sess.unique(), itertools.cycle(colors))
    }
    # build a index from (Ptr, timestamp) to Sess
    print("Building index...")
    keys, grps = zip(*[(ptr, grp.set_index('timestamp').sort_index().drop(['Ptr'], axis=1))
                        for ptr, grp in df[['Ptr', 'Sess', 'timestamp', 'evt']].groupby('Ptr')])
    ptr2sess = pd.concat(grps, keys=keys, names=['Ptr'])

    print("Building memmap...")
    #hdf = '/opt/desktop/card262.memmap.h5'
    hdf = None
    doPreprocess = False
    tss = preprocess_memmap(df, colormap, hdf=hdf)

    m = MemmapViewer(tss, colormap, ptr2sess, hdf=hdf, doPreprocess=doPreprocess)

    return m
