from collections import defaultdict
import re


def main(path='/tmp/workspacev.output'):
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


if __name__ == '__main__':
    main()