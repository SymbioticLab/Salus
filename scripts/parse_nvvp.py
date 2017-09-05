from __future__ import print_function, absolute_import, division

from nvvpreader import NvvpReader

import pandas as pd
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt


def load_file(path):
    return NvvpReader(path, True)


def active_wrap_trend(reader):
    data = []
    df = reader.kernels
    df = df[df['event'] == 'active_warps']
    for start, end, val in zip(df['start'], df['end'], df['event_val']):
        avg = val / (end - start)
        data.append({
            'nanotime': start,
            'active_wraps': avg
        })
        data.append({
            'nanotime': end,
            'active_wraps': -avg
        })

    df = pd.DataFrame(data).set_index('nanotime').sort_index()

    ax = df.cumsum().plot()
    ax.grid('on')
    ax.set_ylabel('Number of active warps')
    ax.set_xlabel('Time (ns)')
    ax.figure.tight_layout()

    return df
