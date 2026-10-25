"""Port of jellyfish Rust Metaphone rules for CPU-only reproducibility.
Original: https://github.com/jamesturk/jellyfish/blob/main/src/metaphone.rs
Check against existing 190-term tier assignments before scientific use.
"""
import unicodedata

def metaphone(s):
    if not s:
        return ''
    raw=str(s).upper()
    v=list(unicodedata.normalize('NFKD',raw))
    if raw.startswith(('KN','GN','PN','WR','AE')):
        v.pop(0)
    ret=[];i=0
    vowel='AEIOU'
    while i<len(v):
        c=v[i]; nxt=v[i+1] if i+1<len(v) else '*'; nn=v[i+2] if i+2<len(v) else '*'
        if c==nxt and c!='C':
            i+=1;continue
        if c in vowel:
            if i==0 or v[i-1]==' ':ret.append(c)
        elif c=='B':
            if (i==0 or v[i-1]!='M') or nxt!='*':ret.append('B')
        elif c=='C':
            if (nxt=='I' and nn=='A') or nxt=='H':
                i+=1;ret.append('X')
            elif nxt in 'IEY':i+=1;ret.append('S')
            else:ret.append('K')
        elif c=='D':
            if nxt=='G' and nn in 'IEY':i+=2;ret.append('J')
            else:ret.append('T')
        elif c in 'FJLMNR':ret.append(c)
        elif c=='G':
            if nxt in 'IEY':ret.append('J')
            elif (nxt=='H' and nn not in vowel) or (nxt=='N' and nn=='*'):i+=1
            else:ret.append('K')
        elif c=='H':
            if i==0 or nxt in vowel or v[i-1] not in vowel:ret.append('H')
        elif c=='K':
            if i==0 or v[i-1]!='C':ret.append('K')
        elif c=='P':
            if nxt=='H':i+=1;ret.append('F')
            else:ret.append('P')
        elif c=='Q':ret.append('K')
        elif c=='S':
            if nxt=='H':i+=1;ret.append('X')
            elif nxt=='I' and nn in 'OA':i+=2;ret.append('X')
            else:ret.append('S')
        elif c=='T':
            if nxt=='I' and nn in 'OA':ret.append('X')
            elif nxt=='H':i+=1;ret.append('0')
            elif nxt!='C' or nn!='H':ret.append('T')
        elif c=='V':ret.append('F')
        elif c=='W':
            if i==0 and nxt=='H':i+=1;ret.append('W')
            elif nxt in vowel:ret.append('W')
        elif c=='X':
            if i==0:
                ret.append('X' if nxt=='H' or (nxt=='I' and nn in 'OA') else 'S')
            else:ret.extend(('K','S'))
        elif c=='Y':
            if nxt in vowel:ret.append('Y')
        elif c=='Z':ret.append('S')
        elif c==' ':
            if ret and ret[-1]!=' ':ret.append(' ')
        i+=1
    return ''.join(ret)
