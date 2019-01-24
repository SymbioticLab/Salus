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
Created on Mon Jul 16 21:50:25 2018

@author: peifeng
"""

import parse_log as pl
import pandas as pd

def sqldf(q):
    import pandasql as ps
    return ps.sqldf(q, globals())

def load_sessiter(path, return_logs=False, parallel_workers=0):
    try:
        logs = pl.load_file(path, parallel_workers)
    except TypeError:
        logs = path

    df = pd.DataFrame(l.__dict__ for l in logs)
    df = df[df.type == 'generic_evt']
    df = df[df.evt.isin(['start_iter', 'end_iter'])]
    df = df.drop(['entry_type', 'level', 'loc', 'thread'], axis=1).dropna(axis=1)
    
    
    enditers = df[df.evt == 'end_iter']
    mainiters = df[df.GraphId.isin(enditers.GraphId)]
    return mainiters
