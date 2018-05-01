#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 04:54:36 2018

@author: peifeng
"""

from __future__ import print_function, absolute_import, division

import re
from datetime import datetime
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import plotutils as pu


ptn_exec = re.compile(r"""^\[(?P<timestamp>\d+-\d+-\d+\s\d+:\d+:\d+\.\d{6}) (\d{3})?\]\s
                           \[(?P<thread>\d+)\]\s
                           \[(?P<loc>\w+)\]\s
                           \[(?P<level>\w+)\]\s
                           (?P<content>.*)$""",
                      re.VERBOSE)

ptn_tf = re.compile(r"""^.*(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{6}):\s  # time
                         (?P<level>\w)\s
                         (?P<loc>.+)\]\s
                         (\[(?P<thread>\d+)\]\s)?
                         (?P<content>.*)$""", re.VERBOSE)


def load_case(path):
    df = pd.read_csv(path, header=None, sep=' ',
                     names=['date', 'time', 'event', 'skip', 'name'],
                     parse_dates=[['date', 'time']])
    df = df[['date_time', 'event', 'name']]
    df['timestamp'] = df['date_time']
    df = df.drop('date_time', axis=1)

    wls = df.pivot_table(values='timestamp', index=['name'],
                         columns='event', aggfunc='first').reset_index()

    for col in ['Started', 'Queued', 'Finished']:
        wls[col] = wls[col].str[:-1]
        wls[col] = pd.to_datetime(wls[col])
    wls['queuing'] = wls.Started - wls.Queued
    wls['JCT'] = wls.Finished - wls.Queued
    return wls


def load_exp15(directory):
    directory = Path(directory)
    salus = load_case(directory/'salus'/'exp15.output')
    fifo = load_case(directory/'fifo'/'exp15.output')
    sp = load_case(directory/'tf'/'exp15.output')

    data = pd.DataFrame({
        'Salus': salus['JCT'],
        'FIFO': fifo['JCT'],
        'SP': sp['JCT']
    })
    queuing = pd.DataFrame({
        'Salus': salus['queuing'],
        'FIFO': fifo['queuing'],
        'SP': sp['queuing']
    })
    return data, queuing

def plot_data(data):
    data = data.copy()
    # convert columns to seconds
    for col in data.columns:
        data[col] = data[col] / pd.Timedelta(seconds=1)
    
    ax = data.plot.bar()
    ax.set_xlabel('Workload ID')
    ax.set_ylabel('JCT (s)')
    return ax
#%%
