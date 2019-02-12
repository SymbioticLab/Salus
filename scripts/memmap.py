#!/usr/bin/env python3
#
# Copyright 2019 Peifeng Yu <peifeng@umich.edu>
# 
# This file is part of Salus
# (see https://github.com/SymbioticLab/Salus).
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:45:07 2018

@author: peifeng
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec

import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import parse_log as pl
import plotutils as pu

def load_memmap(path):
    logs = pl.load_file(path)
    df = pd.DataFrame(l.__dict__ for l in logs)
    df = df[df.type == 'memmap']
    df = df.drop(['level', 'loc', 'entry_type', 'thread', 'type'], axis=1)
    # make sure size is int
    df['Size'] = df.Size.astype(int)

    return df


def load_enditermap(path):
    df = pd.DataFrame(l.__dict__ for l in pl.load_file(path))
    #df = df[df.type == 'generic_evt']
    df = df.query('type == "generic_evt" & (evt == "start_iter" | evt == "end_iter")')
    df = df.drop(['level', 'loc', 'entry_type', 'thread', 'type'], axis=1)
    df = df.dropna(how='all', axis=1)

    # get only end_iter events for memmory map
    return df[df.evt == 'end_iter']

def parse_mapstr(mapstr):
    """Parse mapstr, which is like:
        0x7f2a76000000, 2211840, 1;0x7f2a7621c000, 8200192, 0;

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
        df = pd.read_csv(StringIO(mapcsv), header=None, names=['Origin', 'Size', 'InUse'],
                         index_col=False)
        df['Name'] = name

        # convert each column to proper type
        df['Origin'] = df.Origin.apply(int, base=0)
        df['Size'] = pd.to_numeric(df.Size)
        df['InUse'] = pd.to_numeric(df.InUse).astype('bool')
        df['End'] = df.Origin + df.Size
        chunkdfs.append(df)

    # concat all together
    df = pd.concat(chunkdfs)
    # set colors
    df['Color'] = df.InUse.apply(lambda inuse: 'r' if inuse else 'g')

    return df


def mem_segments(mapdf):
    # use normalized address space, i.e. starting from zero
    st = mapdf.Origin.min()
    mapdf['NOrigin'] = mapdf.Origin - st
    mapdf['NEnd'] = mapdf.End - st
    return [
        np.column_stack([col, np.zeros_like(col)])
        for _, col in mapdf[['NOrigin', 'NEnd']].T.iteritems()
    ]


def draw_on_lc(mapstr, lc):
    mapdf = parse_mapstr(mapstr)
    lc.set_segments(mem_segments(mapdf))
    lc.set_color(mapdf.Color)
    return mapdf, lc


def draw_memmap(mapstr, lc=None, ax=None, **kwargs):
    """Draw memmap str using matplotlib
    mapstr is like:
        0x7f2a76000000, 2211840, 1;0x7f2a7621c000, 8200192, 0;
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

    mapdf, lc = draw_on_lc(mapstr, lc)

    ax.set_xlim([mapdf.NOrigin.min(), mapdf.NEnd.max()])

    if newlc:
        ax.add_collection(lc, autolim=False)
        pu.cleanup_axis_bytes(ax.xaxis)
    return lc


class MemmapViewer(object):
    def __init__(self, df, label=None):
        self.df = df
        self.label = label
        if self.label is None:
            self.label = 'id = {index} size = {Size} timestamp = {timestamp}'

        self.fig = plt.gcf()
        self.fig.set_size_inches([25, 15], forward=True)
        gs = gridspec.GridSpec(2, 1, height_ratios=[20,1])


        self._ax = self.fig.add_subplot(gs[0, 0])
        self._ax.set_ylim([-0.2, 0.2])

        # create artists
        self._lc = draw_memmap('0x0, 17179869184, 0;', ax=self._ax)
        self._text = self._ax.text(0.5, 0.8, "",
                             ha='center', va='baseline', fontsize=25,
                             transform=self._ax.transAxes)
        # lc and texts for allocators
        self._lcalloc = LineCollection([[(0,0.1), (17179869184, 0.1)]], linewidths=10)
        self._ax.add_collection(self._lcalloc)
        self._textalloc = []

        pu.cleanup_axis_bytes(self._ax.xaxis)

        init_frame = 0
        self._step(init_frame)

        # add a slider
        axstep = self.fig.add_subplot(gs[1, 0], facecolor='lightgoldenrodyellow')
        self._stepSlider = Slider(axstep, 'ID', 0, len(df), closedmax=False,
                                  valinit=init_frame, valfmt='%d', valstep=1,
                                  dragging=True)

        def update(val):
            self._step(int(val))
            self.fig.canvas.draw_idle()
        self._stepSlider.on_changed(update)

        # listen key press
        self.fig.canvas.mpl_connect('key_press_event', lambda evt: self._keydown(evt))

    def _step(self, i):
        """draw step"""
        row = self.df.iloc[i]
        mapdf, _ = draw_on_lc(row.MemMap, self._lc)
        self._text.set_text(self.label.format(index=i, **dict(row.items())))

        # draw some lines showing each allocator
        nAlloc = len(mapdf.Name.unique())
        while len(self._textalloc) < nAlloc:
            self._textalloc.append(self._ax.text(0, 0, "",
                                                 ha='center', va='baseline'))

        allocSeg = []
        for (name, grp), t in zip(mapdf.groupby('Name'), self._textalloc):
            rgmin, rgmax = grp.NOrigin.min(), grp.NEnd.max()
            yval = -.1
            allocSeg.append([[rgmin, yval], [rgmax, yval]])

            t.set_text(name)
            t.set_position([(rgmin+rgmax) / 2, yval * 1.2])
        self._lcalloc.set_segments(allocSeg)

        self._ax.set_xlim([mapdf.NOrigin.min(), mapdf.NEnd.max()])

    def _keydown(self, evt):
        if evt.key == 'left':
            newval = max(self._stepSlider.val - 1, self._stepSlider.valmin)
            self._stepSlider.set_val(newval)
        elif evt.key == 'right':
            newval = min(self._stepSlider.val + 1, self._stepSlider.valmax - 1)
            self._stepSlider.set_val(newval)
        self.fig.canvas.draw_idle()

def main():
    df = load_memmap('/tmp/workspace/card189/server.output')
    return MemmapViewer(df)
