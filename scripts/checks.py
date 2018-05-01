from collections import defaultdict
import re
import pandas


def check_threadpool(path, name='Threadpool'):
    pat = re.compile(name + r' (?P<evt>start|end) to run seq (?P<seq>\d+)')
    with open(path) as f:
        lines = f.readlines()
    evts = [pat.search(line).groups() for line in lines if pat.search(line)]
    print('evts num: {}'.format(len(evts)))
    r = set()
    for evt, seq in evts:
        if evt == 'start':
            r.add(seq)
        else:
            r.remove(seq)
    return r


def check_pending_ops(path):
    kernels = defaultdict(int)
    lines = defaultdict(list)
    kernel_type = {}
    ptn_st = re.compile(r'''Process node: (?P<node>[^ \[]+) = (?P<kernel>[^\[]+)''')
    ptn_ed = re.compile("Propagate outputs for node: (?P<node>.+)")
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            m = ptn_st.search(line)
            if m:
                kernels[m.group('node')] += 1
                lines[m.group('node')].append(line)
                kernel_type[m.group('node')] = m.group('kernel')
            m = ptn_ed.search(line)
            if m:
                if kernels[m.group('node')] == 0:
                    raise ValueError("Unknown kernel name: ", m.group('node'), line)
                kernels[m.group('node')] -= 1
    remaining = [('{}[{}]'.format(k, kernel_type[k]), v) for k, v in kernels.items() if v != 0]
    print(remaining)
    return remaining, kernel_type, lines


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


def check_iter_create(path):
    iters = defaultdict(list)
    ptn_create = re.compile(r'''Created iteration (?P<graphId>\d+) for graph (?P<gh>\w+) in session (?P<sess>\w+)''')
    ptn_running = re.compile(r'''Running iteration (?P<sess>\w+):(?P<graphId>\d+)''')
    ptn_finish = re.compile(r'''(?P<sess>\w+):(?P<gh>\w+):(?P<graphId>\d+) finish iteration''')
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')

            m = ptn_create.search(line)
            if m:
                l = iters[m.group('graphId')]
                if l and l[-1] != 2:
                    print('Iteration {} created while it is running'.format(m.group('graphId')))
                l.append(0)

            m = ptn_running.search(line)
            if m:
                l = iters[m.group('graphId')]
                if l and l[-1] != 0:
                    print('Iteration {} running while it is not created'.format(m.group('graphId')))
                l.append(1)

            m = ptn_finish.search(line)
            if m:
                l = iters[m.group('graphId')]
                if l and l[-1] != 1:
                    print('Iteration {} stopped while it is not running'.format(m.group('graphId')))
                l.append(2)

    return iters


def check_part_nodes(path):
    nodes = defaultdict(list)
    # Node 1 in graphHandle=0000000000000001, graphId=1: _SINK = NoOp[]
    ptn_node = re.compile(r'''Node (?P<nid>\d+) in graphHandle=(?P<gh>\w+), graphId=(?P<graphId>\d+): (?P<name>[\w/_]+) = (?P<kernel>[^[]+)''')
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')

            m = ptn_node.search(line)
            if m:
                nodes[m.group('graphId')].append({
                    'nid': int(m.group('nid')),
                    'name': m.group('name'),
                    'kernel': m.group('kernel'),
                    'gh': m.group('gh')
                })

    return nodes
