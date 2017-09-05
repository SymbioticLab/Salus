import re
from collections import defaultdict

with open('tf.output') as f:
    lines = f.readlines()

requests = defaultdict(set)
rrequests = defaultdict(set)
pat_send = re.compile(r"Sending evenlop message_t: (?P<type>[a-zA-Z.]+) seq (?P<seq>\d+)")
pat_recv = re.compile(r"Received evenlop: seq=(?P<seq>\d+) type=(?P<type>[a-zA-Z.]+)")

for line in lines:
    if pat_send.search(line):
        res = pat_send.search(line)
        reqtype = res.group('type')

        s = requests[res.group('type')]
        if res.group('seq') in s:
            print('Request sent twice: ', line)
            import ipdb; ipdb.set_trace()
            continue
        s.add(res.group('seq'))
    elif pat_recv.search(line):
        res = pat_recv.search(line)

        reqtype = res.group('type')
        if reqtype.endswith('Response'):
            reqtype = reqtype.replace('Response', 'Request')

        if reqtype == 'executor.TFRendezRecvRequests':
            s = rrequests[reqtype]
            if res.group('seq') in s:
                print('Sending out twice requests: ', line)
                continue
            s.add(res.group('seq'))
            continue

        if reqtype not in requests:
            print('Response for non-exist request: ', line)
            import ipdb; ipdb.set_trace()
            continue
        s = requests[reqtype]
        if res.group('seq') not in s:
            print('Response for non-exist request seq: ', line)
            import ipdb; ipdb.set_trace()
            continue
        s.remove(res.group('seq'))

print('===========================================')
print('Remaining')

for k, v in requests.items():
    print(k)
    for seq in v:
        print('    ', seq)

for k, v in rrequests.items():
    print(k)
    for seq in v:
        print('    ', seq)
