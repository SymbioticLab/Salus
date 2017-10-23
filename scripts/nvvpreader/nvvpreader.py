from __future__ import absolute_import, print_function, division

import sqlite3 as sql
import struct
import pandas as pd
from datetime import datetime
from tqdm import tqdm

metric_name = {
    19923062: 'achieved_occupancy',
    19923058: 'executed_ipc'
}
event_name = {
    83886184: 'active_warps',
    83886182: 'active_cycles',
    83886183: 'elapsed_cycles_sm',
}


def parseMetricValue(binstr):
    return struct.unpack('d', binstr)[0]


def parseEventValue(s):
    return int(s)


def parseNanoTime(nanotime):
    return datetime.utcfromtimestamp(nanotime / 1e9)


class Metric(object):
    def __init__(self, row):
        super(Metric, self).__init__()


class Kernel(object):
    def __init__(self, row, refpoint):
        super(Kernel, self).__init__()
        self.id = row['correlationId']
        self.start = row['start'] - refpoint
        self.end = row['end'] - refpoint
        self.duration = self.end - self.start
        self.name = row['name']


class NvvpReader(object):
    def __init__(self, filepath, progress=False):
        super(NvvpReader, self).__init__()
        self.filepath = filepath
        self.dbLoaded = False
        self.loadDB(progress)

    def loadDB(self, progress):
        if self.dbLoaded:
            return
        self.dbLoaded = True
        self.conn = sql.connect(self.filepath)
        self.conn.row_factory = sql.Row

        prog_wrapper = tqdm if progress else lambda x, *args, **kwargs: x

        cursor = self.conn.cursor()
        # get timeline reference point (start time of the first overhead event is 0ns)
        cursor.execute("""select start from CUPTI_ACTIVITY_KIND_OVERHEAD order by start""")
        (self.refpoint, ) = cursor.fetchone()
        self.refpoint = parseNanoTime(self.refpoint)
        # get all kernels
        total_amount = 0
        if progress:
            cursor.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL')
            (total_amount, ) = cursor.fetchone()
        cursor.execute("""select strings.value as strname, kernel.*
                          from CUPTI_ACTIVITY_KIND_KERNEL as kernel, StringTable as strings
                           where kernel.name = strings._id_""")
        # create dataset
        data = []
        cursor2 = self.conn.cursor()
        for row in prog_wrapper(cursor, total=total_amount):
            correlationId = row['correlationId']
            kernel = {
                'id': correlationId,
                'start': parseNanoTime(row['start']),
                'end': parseNanoTime(row['end']),
                'duration': row['end'] - row['start'],
                'name': row['strname'],
            }
            # fetch all instances metric on this kernel
            for ins, val, metric_id in cursor2.execute("""select instance, value, id
                                                          from CUPTI_ACTIVITY_KIND_METRIC_INSTANCE
                                                          where correlationId=?""",
                                                       [correlationId]):
                val = parseMetricValue(val)
                observation = {
                    'metric': metric_name[metric_id],
                    'sm': ins,
                    'metric_val': val
                }
                observation.update(kernel)
                data.append(observation)
            # fetch all aggregated metric
            for val, metric_id in cursor2.execute("""select value, id
                                                     from CUPTI_ACTIVITY_KIND_METRIC
                                                     where correlationId=?""",
                                                  [correlationId]):
                val = parseMetricValue(val)
                observation = {
                    'metric': metric_name[metric_id],
                    'sm': -1,
                    'metric_val': val
                }
                observation.update(kernel)
                data.append(observation)
            # fetch all instances events on this kernel
            for ins, val, event_id in cursor2.execute("""select instance, value, id
                                                         from CUPTI_ACTIVITY_KIND_EVENT_INSTANCE
                                                         where correlationId=?""",
                                                      [correlationId]):
                val = parseEventValue(val)
                observation = {
                    'event': event_name[event_id],
                    'sm': ins,
                    'event_val': val
                }
                observation.update(kernel)
                data.append(observation)
            # fetch all aggregated events on this kernel
            for val, event_id in cursor2.execute("""select value, id
                                                    from CUPTI_ACTIVITY_KIND_EVENT
                                                    where correlationId=?""",
                                                 [correlationId]):
                val = parseEventValue(val)
                observation = {
                    'event': event_name[event_id],
                    'sm': -1,
                    'event_val': val
                }
                observation.update(kernel)
                data.append(observation)
        self.kernels = pd.DataFrame(data)
