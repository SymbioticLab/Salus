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
import re
from collections import defaultdict

with open('exec.output') as f:
    lines = f.readlines()

requests = defaultdict(set)
rrequests = defaultdict(set)
pat_evenlop = re.compile(r"Received request evenlop: EvenlopDef\(type='(?P<type>[a-zA-Z.]+)', seq=(?P<seq>\d+)")
pat_resp = re.compile(r"Response proto object have size \d+ with evenlop EvenlopDef\(type='(?P<type>[a-zA-Z.]+)', seq=(?P<seq>\d+)")
pat_dispatch = re.compile(r"Dispatching custom task executor.TFRendezRecvUpdate of seq (?P<seq>\d+)")
pat_recvupd = re.compile(r"executor.TFRendezRecvUpdate for seq (?P<seq>\d+)")

for line in lines:
    if pat_evenlop.search(line):
        res = pat_evenlop.search(line)
        reqtype = res.group('type')

        s = requests[res.group('type')]
        if res.group('seq') in s:
            print('Request got twice: ', line)
            import ipdb; ipdb.set_trace()
            continue
        s.add(res.group('seq'))
    elif pat_resp.search(line):
        res = pat_resp.search(line)

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
    elif pat_dispatch.search(line):
        res = pat_dispatch.search(line)
        s = requests['executor.CustomRequest']
        if res.group('seq') not in s:
            print('CustomRequest not found for TFRendezRecvUpdate')
            import ipdb; ipdb.set_trace()
            continue
        s.remove(res.group('seq'))
    elif pat_recvupd.search(line):
        res = pat_recvupd.search(line)
        s = rrequests['executor.TFRendezRecvRequests']
        if res.group('seq') not in s:
            print('Response for non-exist request seq: ', line)
            import ipdb; ipdb.set_trace()
            continue
        s.remove(res.group('seq'))
        continue

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
