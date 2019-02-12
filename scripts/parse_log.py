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
from __future__ import print_function, absolute_import, division

import json
import re
from datetime import datetime
from collections import defaultdict
import multiprocessing as mp

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


class Log(object):
    """Collection of logs"""
    def __init__(self, arg):
        super(Log, self).__init__()
        self.arg = arg


class Entry(object):
    """Base class for log entry"""
    def __init__(self, g, entry_type):
        super(Entry, self).__init__()

        self.entry_type = entry_type

        self.timestamp = datetime.strptime(g['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
        if g['thread'] is not None:
            self.thread = int(g['thread'])
        self.loc = g['loc']
        self.level = g['level']
        self.raw_content = g['content']

    def __repr__(self):
        return 'Entry{}'.format(self.__dict__.__repr__())

    def update(self, content):
        """append content"""
        self.raw_content += '\n' + content

    def finalize(self):
        if self.entry_type == 'exec':
            d = match_exec_content(self.raw_content, self)
        elif self.entry_type == 'tf':
            d = match_tf_content(self.raw_content, self)
        else:
            raise ValueError('Unknown entry type: "{}"'.format(self.entry_type))

        if 'type' not in d:
            d['type'] = 'unknown'

        del self.raw_content
        self.__dict__.update(d)
        return self


thread_seq_map = {}
tf_thread_seq_map = {}
seq_info = defaultdict(dict)
blocks = {}
thread_alloc_type_map = {}
last_paging_start = []


def initialize():
    global thread_seq_map, tf_thread_seq_map, seq_info, blocks, thread_alloc_type_map, last_paging_start
    thread_seq_map = {}
    tf_thread_seq_map = {}
    seq_info = defaultdict(dict)
    blocks = {}
    thread_alloc_type_map = {}
    last_paging_start = []


initialize()


def _process_line_paral(line):
    line = line.rstrip('\n')
    m = ptn_exec.match(line)
    if m:
        # executor line
        entry = Entry(m.groupdict(), 'exec')
        return entry.finalize()

    m = ptn_tf.match(line)
    if m:
        # tf line
        entry = Entry(m.groupdict(), 'tf')
        return entry.finalize()

    print('Unhandled line: ' + line)
    return None


def load_file(path, reinitialize=True, parallel_workers=0):
    """Load logs"""

    if reinitialize:
        initialize()

    if parallel_workers == 0:
        logs = []
        with pu.pbopen(path) as f:
            entry = None
            for line in f:
                line = line.rstrip('\n')

                m = ptn_exec.match(line)
                if m:
                    if entry:
                        logs.append(entry.finalize())
                        entry = None
                    # executor line
                    entry = Entry(m.groupdict(), 'exec')
                    continue

                m = ptn_tf.match(line)
                if m:
                    if entry:
                        logs.append(entry.finalize())
                        entry = None
                    # tf line
                    entry = Entry(m.groupdict(), 'tf')
                    continue

                # assume it belongs to previous line
                if entry:
                    entry.update(line)
                else:
                    print('Unhandled line: ' + line)
        return logs
    else:
        try:
            pool = mp.Pool(processes=parallel_workers, maxtasksperchild=10)

            with open(path) as f:
                res = pool.map_async(_process_line_paral, f, chunksize=20000)
                while True:
                    try:
                        return res.get(999)
                    except mp.TimeoutError:
                        pass
        finally:
            pool.terminate()
            pool.join()


def load_both(exec_file, tf_file):
    print('Loading ', exec_file)
    log1 = load_file(exec_file)
    print('Loaded {} entries'.format(len(log1)))

    print('Loading ', tf_file)
    log2 = load_file(tf_file, reinitialize=False)
    print('Loaded {} entries'.format(len(log2)))

    print('Merging...')
    logs = log1 + log2
    for seq, info in seq_info.items():
        if 'recv_evenlop' in info and 'tf_send_evenlop' in info:
            sending = (info['recv_evenlop'].timestamp - info['tf_send_evenlop'].timestamp).total_seconds()
            info['recv_evenlop'].travel_time = sending
        else:
            print('Either recv_evenlop or tf_send_evenlop missing in info:', info)

        if 'tf_rpc_return' in info and 'resp_sent' in info:
            replying = (info['tf_rpc_return'].timestamp - info['resp_sent'].timestamp).total_seconds()
            info['tf_rpc_return'].travel_time = replying
        else:
            print('Either tf_rpc_return or resp_sent missing in info:', info)

    return logs


ptn_recv_frame = re.compile(r"""Received \w+ frame( \d+)?: zmq::message_t\(len=(?P<size>\d+),.*""")
ptn_recv_evenlop = re.compile(r"""Received \s request \s evenlop: \s
                                  .+type='executor.(?P<req_type>\w+)',\s
                                  seq=(?P<seq>\d+),.*""",
                              re.VERBOSE)
ptn_disp_custom = re.compile(r"""Dispatching custom task (?P<req>[\w.]+) of seq (?P<seq>\d+)""")
ptn_sess_create = re.compile(r"""Session (?P<sess>\w+) created with recvId (?P<recvid>.+)""")
ptn_create_opkernel = re.compile(r"""Created OpKernel for seq (?P<seq>\d+)""")
ptn_running = re.compile(r"""running(?P<async> async)? in thread \d+""")
ptn_compute_done = re.compile(r"""OpKernel->Compute finished with status.*""")
ptn_compute_async_done = re.compile(r"""OpKernel->ComputeAsync for seq (?P<seq>\d+) finished with status.*""")
ptn_resp_sent = re.compile(r"""Response\sproto\sobject\shave\ssize\s\d+\swith\sevenlop\s.+
                               type='executor.(?P<req_type>\w+)',\s
                               seq=(?P<seq>\d+),.*""",
                           re.VERBOSE)
ptn_fwd_msg = re.compile(r"""Forwarding message part: zmq::message_t\(len=(?P<size>\d+),.*""")

ptn_mem_pre = re.compile(r"""Pre \s allocated \s AllocationTicket\((?P<ticket>\d+),
                             .+session=(?P<sess>\w+).*$""",
                         re.VERBOSE)
ptn_mem_alloc = re.compile(r"""TFAllocator\s.+\s(?P<size>\d+)\sbytes\sof\smemory\sat\s
                               (?P<addr>\w+)\s.*\susing\sallocator\s
                               (?P<mem_type>\w+)@(?P<alloc_inst>\w+)\s
                               with \s AllocationTicket\((?P<ticket>\d+).+
                               sess=(?P<sess>\w+).+""",
                           re.VERBOSE)
ptn_mem_dealloc = re.compile(r"""TFAllocator\sdeallocating\smemory\sat\s(?P<addr>\w+)\s
                                 size\s(?P<size>\d+)\s
                                 using\sallocator\s(?P<mem_type>\w+)@(?P<alloc_inst>\w+)\s
                                 with \s AllocationTicket\((?P<ticket>\d+).+
                                 sess=(?P<sess>\w+).+""",
                             re.VERBOSE)
ptn_progcnt = re.compile(r"""Progress counter for session (?P<sess>\w+): (?P<cnt>\d+)""")

# OpItem Stat ExecTask(name=v/affine2/biases/Initializer/Const, type=Const, session=59089c8c075bcd3e, failures=0, inputsize=0) queued: 2018-03-29 23:33:53.088091647 scheduled: 2018-03-29 23:33:53.093002944 running: 2018-03-29 23:33:53.093008399 finished: 2018-03-29 23:33:53.093745447
ptn_optracing = re.compile(r"""OpItem\sStat.*name=(?P<op>[^,]+),.*
                               type=(?P<kernel>[^,]+),.*
                               session=(?P<sess>[^,]+).*
                               step_id=(?P<step>[^,()]+).*
                               queued:\s(?P<task_ready>.+)\s
                               scheduled:\s(?P<task_sched>.+)\s
                               running:\s(?P<task_start>.+)\s
                               finished: (?P<task_done>.+)$""",
                           re.VERBOSE)

# OpItem Event ExecTask(name=v/affine2/biases/Initializer/Const, type=Const, session=59089c8c075bcd3e, failures=0, inputsize=0) event: queued
ptn_optracing_evt = re.compile(r"""OpItem\sEvent.*name=(?P<op>[^,]+),.*
                                   type=(?P<kernel>[^,]+),.*
                                   session=(?P<sess>[^,]+).*
                                   step_id=(?P<step>[^,()]+).*?
                                   (failed=(?P<failed>\d+))?\s
                                   event:\s(?P<evt>[^,]+)$""",
                               re.VERBOSE)

# event: end_iter sess=
ptn_optracing_evt = re.compile(r"""OpItem\sEvent.*name=(?P<op>[^,]+),.*
                                   type=(?P<kernel>[^,]+),.*
                                   session=(?P<sess>[^,]+).*
                                   step_id=(?P<step>[^,()]+).*?
                                   (failed=(?P<failed>\d+))?\s
                                   event:\s(?P<evt>[^,]+)$""",
                               re.VERBOSE)
# event: type JSON
ptn_generic_evt = re.compile(r"""event: (?P<evt>[^ ]+) (?P<props>.+)$""")

# Session abcdefg has tracker ticket 123
ptn_tracker_ticket = re.compile(r"""Session\s(?P<sess>\w+)\shas\s
                                    tracker\sticket\s(?P<ticket>\d+)$
                                """, re.VERBOSE)

# Start session allocation hold: ticket=123, res=1213.5
ptn_tracker_hold = re.compile(r"""Start\ssession\sallocation.+ticket=(?P<ticket>\d+)
                                  ,.+res=(?P<res>[\d.]+)$
                              """, re.VERBOSE)
# End session allocation hold: ticket=123, res=1213.5
ptn_tracker_unhold = re.compile(r"""End\ssession\sallocation.+ticket=(?P<ticket>\d+)
                                  ,.+res=(?P<res>[\d.]+)$
                              """, re.VERBOSE)

# MismatchedResRequest of size 1234 mem map: 0x1234, 123, 1; 0x567, 456, 0;
ptn_memmap = re.compile(r"""MismatchedResRequest \s of \s size \s (?P<Size>\d+)
                            \s mem \s map:\s(?P<memmap>.+)""", re.VERBOSE)

def seq_from_entry(entry):
    map_to_use = thread_seq_map
    if entry.entry_type == 'tf':
        map_to_use = tf_thread_seq_map
    if entry.thread not in map_to_use:
        raise ValueError('Thread {} not found in thread_seq_map for entry {}'.format(entry.thread, entry))
    seq = map_to_use[entry.thread]
    return seq, seq_info[seq]


def match_exec_content(content, entry):
    m = ptn_recv_frame.match(content)
    if m:
        return {
            'type': 'recv_msg',
            'size': int(m.group('size'))
        }

    m = ptn_recv_evenlop.match(content)
    if m:
        seq = int(m.group('seq'))
        seq_info[seq]['recv_evenlop'] = entry
        return {
            'type': 'recv_evenlop',
            'seq': seq,
            'req_type': m.group('req_type')
        }

    m = ptn_disp_custom.match(content)
    if m:
        seq = int(m.group('seq'))
        seq_info[seq]['disp_custom'] = entry
        return {
            'type': 'disp_custom',
            'seq': seq,
            'req': m.group('req')
        }

    m = ptn_create_opkernel.match(content)
    if m:
        seq = int(m.group('seq'))
        thread_seq_map[entry.thread] = seq
        seq_info[seq]['create_kernel'] = entry
        return {
            'type': 'create_kernel',
            'seq': seq
        }

    m = ptn_running.match(content)
    if m:
        seq, info = seq_from_entry(entry)
        info['start_running'] = entry
        if 'recv_evenlop' not in info:
            raise ValueError('Seq {} info does not contain expected event recv_evenlop: {}'.format(seq, info))
        got_evenlop_time = info['recv_evenlop'].timestamp
        return {
            'type': 'start_running',
            'seq': seq,
            'async': m.group('async') is not None,
            'prep_time': (entry.timestamp - got_evenlop_time).total_seconds()
        }

    m = ptn_compute_done.match(content)
    if m:
        seq, info = seq_from_entry(entry)
        info['compute_done'] = entry
        if 'start_running' not in info:
            raise ValueError(
                'Seq {} info does not contain expected event start_running: {}'.format(seq, info))
        start_running_stamp = info['start_running'].timestamp
        return {
            'type': 'compute_done',
            'seq': seq,
            'compute_time': (entry.timestamp - start_running_stamp).total_seconds()
        }

    m = ptn_compute_async_done.match(content)
    if m:
        seq = int(m.group('seq'))
        info = seq_info[seq]
        info['compute_done'] = entry
        if 'start_running' not in info:
            raise ValueError(
                'Seq {} info does not contain expected event start_running: {}'.format(seq, info))
        start_running_stamp = info['start_running'].timestamp
        return {
            'type': 'compute_done',
            'seq': seq,
            'compute_time': (entry.timestamp - start_running_stamp).total_seconds()
        }

    m = ptn_resp_sent.match(content)
    if m:
        seq = int(m.group('seq'))
        info = seq_info[seq]
        info['resp_sent'] = entry

        d = {
            'type': 'resp_sent',
            'seq': seq,
            'resp_type': m.group('req_type')
        }
        if m.group('req_type') != 'RunResponse':
            return d

        if 'compute_done' not in info:
            raise ValueError('Seq {} info does not contain expected event compute_done: {}'.format(seq, info))
        compute_done_stamp = info['compute_done'].timestamp
        d['process_time'] = (entry.timestamp - compute_done_stamp).total_seconds()
        return d

    m = ptn_fwd_msg.match(content)
    if m:
        return {
            'type': 'fwd_msg',
            'size': int(m.group('size'))
        }

    m = ptn_mem_alloc.match(content)
    if m:
        addr = m.group('addr')
        size = int(m.group('size'))
        mem_type = m.group('mem_type')
        alloc_inst = m.group('alloc_inst')
        ticket = int(m.group('ticket'))
        block = {
            'size': size,
            'addr': addr,
            'mem_type': mem_type,
            'alloc_inst': alloc_inst,
            'ticket': ticket
        }
        return {
            'type': 'mem_alloc',
            'size': size,
            'addr': addr,
            'sess': m.group('sess'),
            'block': block
        }

    m = ptn_mem_dealloc.match(content)
    if m:
        addr = m.group('addr')
        size = int(m.group('size'))
        mem_type = m.group('mem_type')
        alloc_inst = m.group('alloc_inst')
        ticket = int(m.group('ticket'))
        block = {
            'size': size,
            'addr': addr,
            'mem_type': mem_type,
            'alloc_inst': alloc_inst,
            'ticket': ticket
        }
        return {
            'type': 'mem_dealloc',
            'addr': addr,
            'size': size,
            'sess': m.group('sess'),
            'block': block
        }

    m = ptn_mem_pre.match(content)
    if m:
        return {
            'type': 'mem_pre',
            'ticket': int(m.group('ticket')),
            'sess': m.group('sess'),
        }

    m = ptn_progcnt.match(content)
    if m:
        sess = m.group('sess')
        cnt = int(m.group('cnt'))
        return {
            'type': 'prog_cnt',
            'sess': sess,
            'cnt': cnt
        }

    m = ptn_sess_create.match(content)
    if m:
        sess = m.group('sess')
        return {
            'type': 'sess_create',
            'sess': sess
        }
        
    m = ptn_optracing.match(content)
    if m:
        sess = m.group('sess')
        return {
            'type': 'optracing',
            'sess': sess,
            'op': m.group('op'),
            'kernel': m.group('kernel'),
            'step': int(m.group('step')),
            'task_start': m.group('task_start'),
            'task_ready': m.group('task_ready'),
            'task_sched': m.group('task_sched'),
            'task_done': m.group('task_done')
        }

    m = ptn_optracing_evt.match(content)
    if m:
        sess = m.group('sess')
        failed = m.group('failed')
        failed = None if failed is None else int(failed)
        return {
            'type': 'optracing_evt',
            'sess': sess,
            'op': m.group('op'),
            'kernel': m.group('kernel'),
            'step': int(m.group('step')),
            'evt': m.group('evt'),
            'failed': failed,
        }
        
    m = ptn_tracker_ticket.match(content)
    if m:
        sess = m.group('sess')
        ticket = m.group('ticket')
        return {
            'type': 'tracker_ticket',
            'sess': sess,
            'ticket': ticket
        }
    
    m = ptn_tracker_hold.match(content)
    if m:
        ticket = m.group('ticket')
        res = float(m.group('res'))
        return {
            'type': 'tracker_hold',
            'ticket': ticket,
            'size': res
        }
    
    m = ptn_tracker_unhold.match(content)
    if m:
        ticket = m.group('ticket')
        res = float(m.group('res'))
        return {
            'type': 'tracker_unhold',
            'ticket': ticket,
            'size': res
        }
    
    m = ptn_memmap.match(content)
    if m:
        size = int(m.group('Size'))
        memmap = m.group('memmap')
        return {
            'type': 'memmap',
            'Size': size,
            'memmap': memmap
        }
        
    m = ptn_generic_evt.match(content)
    if m:
        # make every prop starts with captial word
        props = {}
        for k, v in json.loads(m.group('props')).items():
            props[k[0].upper() + k[1:]] = v
        ret = {
        'type': 'generic_evt',
        'evt': m.group('evt')
        }
        ret.update(props)
        return ret

    return {}


ptn_tf_vanilla_start = re.compile(r"""\w+ Kernel Compute start: seq=(?P<seq>\d+)""")
ptn_tf_vanilla_done = re.compile(r"""\w+ Kernel Compute done: seq=(?P<seq>\d+)""")
ptn_tf_vanilla_start_async = re.compile(r"""\w+ ComputeAsync start: seq=(?P<seq>\d+)""")
ptn_tf_vanilla_done_async = re.compile(r"""\w+ ComputeAsync done: seq=(?P<seq>\d+)""")
ptn_rpc_run = re.compile(r"""RpcClient::run(Async)?\s+calling rpc using rpc stub""")
ptn_send_evenlop = re.compile(r"""Sending evenlop message_t: executor\.(?P<req_type>\w+) seq (?P<seq>\d+)""")
ptn_evenlop_sent = re.compile(r"""Message sent for seq: (?P<seq>\d+)""")
ptn_recv_resp = re.compile(r"""Received evenlop: seq=(?P<seq>\d+) type=executor\.(?P<req_type>\w+)""")
ptn_rpc_return = re.compile(r"""RpcClient::run(Async)?\s+rpc returned with status:.*""")
# Process node: 0 step -1 _SOURCE = NoOp[]() is dead: 0
ptn_tf_process_node = re.compile(
    r"""Process node: (?P<nid>\d+) step (?P<step>[-\d]+) (?P<op>[\w/]+) = (?P<kernel>[^[]+).*""")
# NodeDone: 11 step 30 _retval_AssignAdd_0_0 = _Retval[T=DT_INT64, index=0, _device="/job:localhost/replica:0/task:0/device:CPU:0"](AssignAdd)
ptn_tf_node_done = re.compile(
    r"""NodeDone: (?P<nid>\d+) step (?P<step>[-\d]+) (?P<op>[\w/]+) = (?P<kernel>[^[]+).*""")
# NodeReady: 11 step 30 _retval_AssignAdd_0_0 = _Retval[T=DT_INT64, index=0, _device="/job:localhost/replica:0/task:0/device:CPU:0"](AssignAdd)
ptn_tf_node_ready = re.compile(
    r"""NodeReady: (?P<nid>\d+) step (?P<step>[-\d]+) (?P<op>[\w/]+) = (?P<kernel>[^[]+).*""")
# +12345, 0xabcde
ptn_tf_alloc = re.compile(r"""^\+(?P<size>\d+), (?P<addr>.+)$""")
# -, 0xabcde
ptn_tf_dealloc = re.compile(r"""^-(?P<size>\d+), (?P<addr>.+)$""")

def match_tf_content(content, entry):
    m = ptn_rpc_run.match(content)
    if m:
        return {}

    m = ptn_send_evenlop.match(content)
    if m:
        seq = int(m.group('seq'))
        tf_thread_seq_map[entry.thread] = seq
        seq_info[seq]['tf_send_evenlop'] = entry
        return {
            'type': 'tf_send_evenlop',
            'seq': seq,
            'req_type': m.group('req_type')
        }

    m = ptn_evenlop_sent.match(content)
    if m:
        seq = int(m.group('seq'))
        tf_thread_seq_map[entry.thread] = seq
        seq_info[seq]['tf_evenlop_sent'] = entry
        return {
            'type': 'tf_evenlop_sent',
            'seq': seq,
        }

    m = ptn_recv_resp.match(content)
    if m:
        seq = int(m.group('seq'))
        tf_thread_seq_map[entry.thread] = seq
        seq_info[seq]['tf_recv_resp'] = entry
        return {
            'type': 'tf_recv_resp',
            'seq': seq,
            'resp_type': m.group('req_type')
        }

    m = ptn_rpc_return.match(content)
    if m:
        seq, info = seq_from_entry(entry)
        info['tf_rpc_return'] = entry
        return {
            'type': 'tf_rpc_return',
            'seq': seq,
            'roundtrip': (entry.timestamp - info['tf_send_evenlop'].timestamp).total_seconds()
        }

    m = ptn_tf_vanilla_start.match(content)
    if m:
        seq = int(m.group('seq'))
        info = seq_info[seq]
        info['tf_vanilla_start'] = entry
        return {
            'type': 'tf_vanilla_start',
            'seq': seq
        }

    m = ptn_tf_vanilla_done.match(content)
    if m:
        seq = int(m.group('seq'))
        info = seq_info[seq]
        info['compute_done'] = entry
        return {
            'type': 'compute_done',
            'seq': seq,
            'compute_time': (entry.timestamp - info['tf_vanilla_start'].timestamp).total_seconds()
        }

    m = ptn_tf_vanilla_start_async.match(content)
    if m:
        seq = int(m.group('seq'))
        info = seq_info[seq]
        info['tf_vanilla_start'] = entry
        return {
            'type': 'tf_vanilla_start',
            'seq': seq
        }

    m = ptn_tf_vanilla_done_async.match(content)
    if m:
        seq = int(m.group('seq'))
        info = seq_info[seq]
        info['compute_done'] = entry
        return {
            'type': 'compute_done',
            'seq': seq,
            'compute_time': (entry.timestamp - info['tf_vanilla_start'].timestamp).total_seconds()
        }

    m = ptn_tf_process_node.match(content)
    if m:
        nid = int(m.group('nid'))
        return {
            'type': 'task_start',
            'nid': nid,
            'step': int(m.group('step')),
            'op': m.group('op'),
            'kernel': m.group('kernel'),
        }

    m = ptn_tf_node_ready.match(content)
    if m:
        nid = int(m.group('nid'))
        return {
            'type': 'task_ready',
            'nid': nid,
            'step': int(m.group('step')),
            'op': m.group('op'),
            'kernel': m.group('kernel'),
        }

    m = ptn_tf_node_done.match(content)
    if m:
        nid = int(m.group('nid'))
        return {
            'type': 'task_done',
            'nid': nid,
            'step': int(m.group('step')),
            'op': m.group('op'),
            'kernel': m.group('kernel'),
        }
        
    m = ptn_tf_alloc.match(content)
    if m:
        return {
            'type': 'tf_alloc',
            'size': int(m.group('size')),
            'addr': m.group('addr')
        }
    
    m = ptn_tf_dealloc.match(content)
    if m:
        return {
            'type': 'tf_dealloc',
            'addr': m.group('addr')
        }

    return {}


def get_beginning(logs):
    for l in logs:
        if l.type == 'disp_custom' and l.req == 'tensorflow.CreateSessionRequest':
                return l.timestamp


def session_beginnings(logs):
    sessstarts = {}
    for l in logs:
        if l.type == 'sess_create':
            sessstarts[l.sess] = l.timestamp
    return sessstarts


def message_size(logs):
    recv_sizes = [l.size for l in logs if l.type == 'recv_msg']
    rs = pd.Series(recv_sizes)

    send_sizes = [l.size for l in logs if l.type == 'fwd_msg']
    ss = pd.Series(send_sizes)

    print('Received {} messages'.format(len(rs)))
    print('Cumulative count at each point: ', np.array([.25, .75, .90, .999, .9999]) * len(rs))
    print(rs.quantile([.25, .75, .90, .999, .9999]))

    print('Sent {} messages'.format(len(ss)))
    print('Cumulative count at each point: ', np.array([.25, .75, .90, .999, .9999]) * len(ss))
    print(ss.quantile([.25, .75, .90, .999, .9999]))

    fig, axs = plt.subplots(ncols=2)

    ax = sns.distplot(rs, hist=True, kde=False, ax=axs[0])
    ax.set_xlabel('Size (byte)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_title('Message size of {} received messages'.format(len(rs)))

    ax = sns.distplot(ss, hist=True, kde=False, ax=axs[1])
    ax.set_xlabel('Size (byte)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_title('Message size of {} sent messages'.format(len(ss)))

    fig.tight_layout()

    return rs, ss, fig


def scheduling_time(logs):
    times = [l.prep_time for l in logs if l.type == 'start_running']
    ts = pd.Series(times)

    if len(ts) == 0:
        return ts, None

    print('Preparation times for {} RunRequests'.format(len(ts)))
    print('Cumulative count at each point: ', np.array([.25, .75, .90, .999, .9999]) * len(ts))
    print(ts.quantile([.25, .75, .90, .999, .9999]))

    ax = sns.distplot(ts, hist=True, kde=False)
    ax.set_xlabel('Preparation time (s)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_title('Preparation time for {} RunRequests'.format(len(ts)))
    ax.figure.tight_layout()

    return ts, ax.figure


def compute_time(logs):
    times = [l.compute_time for l in logs if l.type == 'compute_done']
    ts = pd.Series(times)

    if len(ts) == 0:
        return ts, None

    print('Compute time for {} requests'.format(len(ts)))
    print('Cumulative count at each point: ', np.array([.25, .75, .90, .999, .9999]) * len(ts))
    print(ts.quantile([.25, .75, .90, .999, .9999]))

    ax = sns.distplot(ts, hist=True, kde=False)
    ax.set_xlabel('Compute time (s)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_title('Compute time for {} messages'.format(len(ts)))
    ax.figure.tight_layout()

    return ts, ax.figure


def process_time(logs):
    times = [l.process_time for l in logs if l.type == 'resp_sent' and hasattr(l, 'process_time')]
    ts = pd.Series(times)

    if len(ts) == 0:
        return ts, None

    print('Post process time for {} requests'.format(len(ts)))
    print('Cumulative count at each point: ', np.array([.25, .75, .90, .999, .9999]) * len(ts))
    print(ts.quantile([.25, .75, .90, .999, .9999]))

    ax = sns.distplot(ts, hist=True, kde=False)
    ax.set_xlabel('Post process time (s)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_title('Post process time for {} messages'.format(len(ts)))
    ax.figure.tight_layout()

    return ts, ax.figure


def roundtrip_time(logs):
    times = [l.roundtrip for l in logs if l.type == 'tf_rpc_return']
    ts = pd.Series(times)

    if len(ts) == 0:
        return ts, None

    print('Round-trip time for {} messages'.format(len(ts)))
    print('Cumulative count at each point: ', np.array([.25, .75, .90, .999, .9999]) * len(ts))
    print(ts.quantile([.25, .75, .90, .999, .9999]))

    ax = sns.distplot(ts, hist=True, kde=False)
    ax.set_xlabel('Round-trip time (s)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_title('Round-trip for {} messages'.format(len(ts)))
    ax.figure.tight_layout()

    return ts, ax.figure


def req_on_wire_time(logs):
    times = [l.travel_time for l in logs if l.type == 'recv_evenlop' and hasattr(l, 'travel_time')]
    ts = pd.Series(times)
    if len(ts) == 0:
        return ts, None

    print('Transmission time for {} requests'.format(len(ts)))
    print('Cumulative count at each point: ', np.array([.25, .75, .90, .999, .9999]) * len(ts))
    print(ts.quantile([.25, .75, .90, .999, .9999]))

    ax = sns.distplot(ts, hist=True, kde=False)
    ax.set_xlabel('Transmission time (s)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_title('Transmission for {} requests'.format(len(ts)))
    ax.figure.tight_layout()

    return ts, ax.figure


def resp_on_wire_time(logs):
    times = [l.travel_time for l in logs if l.type == 'tf_rpc_return' and hasattr(l, 'travel_time')]
    ts = pd.Series(times)

    if len(ts) == 0:
        return ts, None

    print('Transmission time for {} responses'.format(len(ts)))
    print('Cumulative count at each point: ', np.array([.25, .75, .90, .999, .9999]) * len(ts))
    print(ts.quantile([.25, .75, .90, .999, .9999]))

    ax = sns.distplot(ts, hist=True, kde=False)
    ax.set_xlabel('Transmission time (s)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_title('Transmission for {} responses'.format(len(ts)))
    ax.figure.tight_layout()

    return ts, ax.figure


def memory_usage(logs, iter_times=None, beginning=None, starts=None, ends=None, mem_type=None,
                 unified_ylabel=False, smoother=None, xformatter=None,
                 per_sess=False, show_avg=None, ax=None):
    if beginning is None:
        beginning = get_beginning(logs)

    if show_avg is None:
        show_avg = not per_sess

    # Prepare ticket -> session map
    ticket2sess = {}
    for l in logs:
        if l.type != 'mem_pre':
            continue
        if l.ticket not in ticket2sess:
            ticket2sess[l.ticket] = l.sess
        elif l.sess != ticket2sess[l.ticket]:
            print('WARNING: ticket {} reused: previous {}, now: {}'.format(l.ticket,
                                                                           ticket2sess[l.ticket], l.sess))

    mem_usages = [l for l in logs if l.type == 'mem_alloc' or l.type == 'mem_dealloc']

    mem_activities = []
    for m in mem_usages:
        if m.addr == '0x0' or m.addr == '0':
            continue

        if m.type == 'mem_alloc':
            mem_activities.append({
                'timestamp': m.timestamp,
                'size': m.size,
                'mem_type': m.block['mem_type'],
                'alloc_inst': m.block['alloc_inst'],
                'session': ticket2sess[m.block['ticket']]
            })
        elif m.type == 'mem_dealloc':
            mem_activities.append({
                'timestamp': m.timestamp,
                'size': -m.size,
                'mem_type': m.block['mem_type'],
                'alloc_inst': m.block['alloc_inst'],
                'session': ticket2sess[m.block['ticket']]
            })
        else:
            raise ValueError("Unexpected value: ", m)

    df = pd.DataFrame(mem_activities)

    if len(df) == 0:
        return df, None

    if mem_type is not None:
        df = df[df['mem_type'] == mem_type]

    df = df.set_index('timestamp').sort_index()

    # Save beginning
    if beginning is None:
        beginning = df.index[0]

    nrows = len(df['mem_type'].unique())
    if ax is None:
        fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True, squeeze=False)
        # Make axes into a 1D array
        axs = axs.reshape(-1)
    else:
        axs = [ax]

    series = []
    pending_avg = []
    for (name, group), ax in zip(df.groupby('mem_type'), axs):
        if per_sess:
            sessionUsages = {}
            for k, gg in group.groupby('session'):
                sessionUsages[k] = gg['size'].cumsum()

            ss = pd.DataFrame(sessionUsages).fillna(method='ffill').fillna(0)
        else:
            ss = group['size'].cumsum()

        # Do range restriction
        rstarts = None
        rends = None
        # Restrict x axis to iteration timaxes, must be done after cumsum, otherwise there
        # will be negative number
        if iter_times is not None:
            rstarts = iter_times[0][0]
            rends = iter_times[-1][1]
        rstarts = starts if starts is not None else rstarts
        rends = ends if ends is not None else rends
        if rstarts is not None and rends is not None:
            ss = ss.loc[rstarts:rends]
        elif rstarts is not None:
            ss = ss.loc[rstarts:]
        elif rends is not None:
            ss = ss.loc[:rends]

        ss.index = ss.index - beginning

        if show_avg:
            ss2 = ss.resample('100us').interpolate(method='time')
            pending_avg.append((ss2.mean(), ax))

        series.append(ss)
        if smoother:
            if show_avg:
                ss = smoother(ss, ss2)
            else:
                ss = smoother(ss)

        # Change to timedelta after iteration restriction.
        # for some reason slicing doesn't work on Timedeltas
        # Pandas doesn't support irregular Timedelta intervals on xaxis
        # see https://stackoverflow.com/questions/40621710/axis-interval-spacing-when-plotting-with-pandas-timedelta
        # and https://github.com/pandas-dev/pandas/issues/8711
        ss.index = ss.index.astype(int)

        if per_sess:
            ss.plot.area(ax=ax, linewidth=0)
        else:
            ss.plot(ax=ax)
            ax.legend().remove()

        pu.cleanup_axis_bytes(ax.yaxis)
        if not unified_ylabel:
            ax.set_ylabel('Memory Usage')

    # Adjust x axis
    if unified_ylabel:
        axs[-1].xaxis.label.set_visible(False)
    else:
        axs[-1].set_xlabel('Time (s)')
    axs[-1].autoscale(axis='x')
    # xlim = axs[-1].get_xlim()
    axs[-1].set_xlim(left=0)
    pu.cleanup_axis_timedelta(axs[-1].xaxis, xformatter)

    # Draw avg line after adjust xaxis
    if show_avg:
        for d, ax in pending_avg:
            pu.axhlines(d, linestyle='--', ax=ax)

    def format_coord(x, y):
        return 'x={:.4f}, y={:.1f} MB'.format(x, y / 1024 / 1024)
    axs[-1].format_coord = format_coord

    fig = axs[-1].figure
    #fig.tight_layout()

    if unified_ylabel:
        fig.text(0.5, 0.02, 'Time (s)', ha='center')
        fig.text(0.02, 0.5, 'Memory Usage (bytes)', va='center', rotation='vertical')
        fig.subplots_adjust(left=0.1, bottom=0.13)
    return df, beginning, fig

