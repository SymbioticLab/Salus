from collections import defaultdict
import re


def check_threadpool(path):
    pat = re.compile(r'Threadpool (?P<evt>start|end) to run seq (?P<seq>\d+)')
    with open(path) as f:
        lines = f.readlines()
    evts = [pat.search(line).groups() for line in lines if pat.search(line)]
    r = set()
    for evt, seq in evts:
        if evt == 'start':
            r.add(seq)
        else:
            r.remove(seq)
    return r


def check_pending_ops(path):
    kernels = defaultdict(int)
    ptn_st = re.compile(r'''Process node: (?P<node>[^ \[]+) ''')
    ptn_ed = re.compile("Propagate outputs for node: (?P<node>.+)")
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            m = ptn_st.search(line)
            if m:
                kernels[m.group('node')] += 1

            m = ptn_ed.search(line)
            if m:
                if kernels[m.group('node')] == 0:
                    raise ValueError("Unknown kernel name: ", m.group('node'), line)
                kernels[m.group('node')] -= 1
    remaining = [(k, v) for k, v in kernels.items() if v != 0]
    print(remaining)
    return remaining


def check_kernel_create(path):
    kernels = {}
    ptn_create = re.compile(r'''Created kernel: (?P<kernel>\w+) (?P<op>.+)''')
    ptn_find = re.compile(r'''Found cached kernel: (?P<kernel>\w+) (?P<op>.+)''')
    ptn_delete = re.compile(r'''Deleted kernel: (?P<kernel>\w+) (?P<op>.+)''')
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')

            m = ptn_create.search(line)
            if m:
                kernels[m.group('kernel')] = m.group('op')
            
            m = ptn_find.search(line)
            if m:
                addr = m.group('kernel')
                if addr not in kernels:
                    raise ValueError('Found nonexist kernel: ', addr, m.group('op'))
                if kernels[addr] != m.group('op'):
                    raise ValueError('Found kernel changed op: ', addr, kernels[addr], m.group('op'))
            
            m = ptn_delete.search(line)
            if m:
                addr = m.group('kernel')
                if addr not in kernels:
                    raise ValueError('Delete nonexist kernel: ', addr, m.group('op'))
                if kernels[addr] != m.group('op'):
                    raise ValueError('Delete kernel changed op: ', addr, kernels[addr], m.group('op'))
                del kernels[addr]
    return kernels