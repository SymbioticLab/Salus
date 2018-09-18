#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 07:38:29 2018

@author: peifeng
"""
from __future__ import print_function, absolute_import, division

import re
from datetime import datetime
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path

import pandas as pd
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import plotutils as pu
import compmem as cm
import jctutils as ju


def load_data(path, name):
    df = ju.load_case(path/name)
    try:
        df = ju.refine_time_events(df, ju.load_serverevents(path))
    except ValueError:
        pass
    return df


def do_srtf(path):
    srtf = load_data(path/'card272'/'case5'/'salus', 'case5.output')
    srtf_refine = ju.load_refine(path/'card272'/'case5'/'salus')
    df = cm.load_mem(path/'card274/srtf/alloc.output')

    offset_local = srtf.Queued.min()
    # timeline logs in US/Eastern, but server logs in UTC
    # convert offset from US/Eastern to UTC
    offset_server = offset_local.tz_localize('US/Eastern').tz_convert('UTC').tz_localize(None)

    # select sess name after calc the offset
    # use jno = [12, 13, 14, 15]
    jnos = [13, 14, 15]
    #srtf = srtf[srtf.No.isin(jnos)]
    #srtf_refine = srtf_refine[srtf_refine.No.isin(jnos)]

    # select data in range
    st_sec = 400
    ed_sec = 515
    st = offset_server + pd.to_timedelta(st_sec, 's')
    ed = offset_server + pd.to_timedelta(ed_sec, 's')
    df = df[(df.timestamp >= st) & (df.timestamp <= ed)]
    df = df[df.Sess.isin(srtf[srtf.No.isin(jnos)].Sess.unique())]

    sess2Model = srtf[srtf.No.isin(jnos)].set_index('Sess').Model.str.split('.', expand=True)
    sess2Model['Label'] = sess2Model.apply(lambda x : '#{}: {}'.format(x[3],x[0]), axis=1)
    df['Sess'] = df.Sess.map(sess2Model.Label)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    with plt.style.context(['seaborn-paper', 'mypaper']):
        fig, axs = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios':[1, 4]})

        _, _, sessColors = ju.plot_timeline(srtf.drop(['LaneId'], axis=1), ax=axs[0], linewidth=2.5,
                                            colors=colors, returnSessColors=True, offset=offset_local)
        ju.plot_refine(axs[0], srtf, srtf_refine)
        #axs[0].set_ylabel('Job No.')
        # srtf is selected by simply set axis limit, because they are not easy to cut in data
        axs[0].set_xlim([st_sec, ed_sec])
        axs[0].set_ylim([12.45, 15.5])
        #axs[0].set_ylabel('Job', rotation=0, labelpad=20)

        sessColors = {
            row.Label: sessColors[sess]
            for sess, row in sess2Model.iterrows()
        }

        cm.plot_all(df, offset=offset_server, ax=axs[1], sessColors=sessColors)
        axs[1].set_ylim(bottom=0)

        #axs[1].legend().remove()
        # Use axs[0]'s legend, but put it at axs[1]'s legend's place
        fig.subplots_adjust(bottom=0.15, hspace=0.05)
        axs[1].legend(
                      #*axs[0].get_legend_handles_labels(),
                      loc="lower center", frameon=False,
                      bbox_to_anchor=[0.5, 1],
                      #bbox_transform=fig.transFigure,
                      fontsize='x-small',
                      ncol=4
                      #mode='expand'
                      )

        axs[1].xaxis.set_major_locator(pu.MaxNLocator(nbins=5))
        axs[1].minorticks_off()
        pu.cleanup_axis_bytes(axs[1].yaxis, maxN=5)
        axs[1].set_ylabel('Memory Usage')
        axs[1].set_xlabel('Time (s)')

        #fig.tight_layout()
        fig.set_size_inches(6.5, 2, forward=True)
        fig.savefig('/tmp/workspace/card274-srtf.pdf', dpi=300, bbox_inches='tight', pad_inches = .015)
        #fig.close()


def do_fair(path):
    fair = load_data(path/'card272'/'case4'/'salus', 'case4.output')
    df = cm.load_mem(path/'card274/fair/alloc.output')

    offset_local = fair.Queued.min()
    # timeline logs in US/Eastern, but server logs in UTC
    # convert offset from US/Eastern to UTC
    offset_server = offset_local.tz_localize('US/Eastern').tz_convert('UTC').tz_localize(None)

    # select sess name after calc the offset
    # use jno
    jnos = [1, 2, 4, 5]
    #srtf = srtf[srtf.No.isin(jnos)]
    #srtf_refine = srtf_refine[srtf_refine.No.isin(jnos)]

    # select data in range
    st_sec = 158
    ed_sec = 164
    st = offset_server + pd.to_timedelta(st_sec, 's')
    ed = offset_server + pd.to_timedelta(ed_sec, 's')
    df = df[(df.timestamp >= st) & (df.timestamp <= ed)]
    df = df[df.Sess.isin(fair[fair.No.isin(jnos)].Sess.unique())]

    sess2Model = fair[fair.No.isin(jnos)].set_index('Sess').Model.str.split('.', expand=True)
    sess2Model['Label'] = sess2Model.apply(lambda x : '#{}: {}'.format(x[3],x[0]), axis=1)
    df['Sess'] = df.Sess.map(sess2Model.Label)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    with plt.style.context(['seaborn-paper', 'mypaper']):
        fig, ax = plt.subplots()

        sessColors = {
            l: c
            for l, c in zip(sess2Model.Label, colors)
        }

        cm.plot_all(df, offset=offset_server, ax=ax, sessColors=sessColors)
        ax.set_ylim(bottom=0)

        #axs[1].legend().remove()
        # Use axs[0]'s legend, but put it at axs[1]'s legend's place
        fig.subplots_adjust(bottom=0.15, hspace=0.05)
        ax.legend(
                      #*axs[0].get_legend_handles_labels(),
                      loc="lower center", frameon=False,
                      bbox_to_anchor=[0.5, 1],
                      #bbox_transform=fig.transFigure,
                      fontsize='x-small',
                      ncol=4
                      #mode='expand'
                      )

        ax.xaxis.set_major_locator(pu.MaxNLocator(nbins=5))
        ax.minorticks_off()
        pu.cleanup_axis_bytes(ax.yaxis, maxN=5)
        ax.set_ylabel('Memory Usage')
        ax.set_xlabel('Time (s)')

        #fig.tight_layout()
        fig.set_size_inches(6.5, 1.6, forward=True)
        fig.savefig('/tmp/workspace/card274-fair.pdf', dpi=300, bbox_inches='tight', pad_inches = .015)
        #fig.close()


path = '/opt/desktop'
def prepare_paper(logpath=path):
    logpath = Path(logpath)

    #do_srtf(Path('/opt/desktop'))
    do_fair(logpath)
