import re


def check():
    pat = re.compile(r'Threadpool (?P<evt>start|end) to run seq (?P<seq>\d+)')
    with open('tf.output') as f:
        lines = f.readlines()
    evts = [pat.search(line).groups() for line in lines if pat.search(line)]
    r = set()
    for evt, seq in evts:
        if evt == 'start':
            r.add(seq)
        else:
            r.remove(seq)
    return r
