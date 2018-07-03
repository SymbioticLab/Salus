
def main():
    blocks = []
    with open('/tmp/workspace/gpu.map') as f:
        for line in f:
            line = line.strip('\n')
            if line.startswith('Name'):
                continue
                
            words = line.split()
            if len(words) < 3:
                break
            
            st = int(words[1], 16)
            l = int(words[2], 16)
            ed = st + l
            name = words[0]
            
            blocks.append((st, ed, l, name))
    
    # sort blocks by st
    blocks.sort()
    
    # assuming no overlap
    
    # merge continuous blocks if they are the same name
    newblocks = []
    prev = None
    for b in blocks:
        if prev is None:
            prev = b
            continue
        
        st, ed, l, name = b
        pst, ped, pl, pname = prev
        
        assert st >= ped
        
        if name == pname and st == ped:
            assert ed - pst == pl + l
            prev = (pst, ed, pl + l, name)
        else:
            newblocks.append(prev)
            prev = b
    
    # print
    for st, ed, l, name in newblocks:
        print('{name}\t0x{st:x}\t0x{l:x} ({l} B)'.format(st=st, ed=ed, l=l, name=name))