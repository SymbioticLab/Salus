from __future__ import print_function, absolute_import, division

from nvvpreader import NvvpReader

import pandas as pd
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt


def load_file(path):
    return NvvpReader(path, True)


def active_wrap_trend(reader, iter_end_times):
    data = []
    df = reader.kernels
    df = df[df['event'] == 'active_warps']
    for start, end, dur, val in zip(df['start'], df['end'], df['duration'], df['event_val']):
        avg = val / dur
        data.append({
            'timestamp': start,
            'active_warps': avg
        })
        data.append({
            'timestamp': end,
            'active_warps': -avg
        })

    df = pd.DataFrame(data).set_index('timestamp').sort_index()

    ax = df.cumsum().plot()
    ax.grid('on')
    ax.set_ylabel('Number of active warps')
    ax.set_xlabel('Time')

    for t in iter_end_times:
        ax.axvline(x=t, linestyle='--', color='r')

    ax.figure.tight_layout()

    return df
