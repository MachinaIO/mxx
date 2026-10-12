"""Write `Part_<c_j>_PreSums.lean`: the member sums of the preprocessing's PRF keys (four key
blocks), the counterpart of `gen_sums.py`.

Usage: `python3 gen_pre_sums.py <parameter directory>`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_part import parts_of
from gen_sums import lemmas, sums

d = sys.argv[1]
for part in parts_of(d):
    path = f'{d}/generated/Stage_preprocess_{part}.lean'
    ns = f'Stage_preprocess_{part}'
    found = sums(path)
    out = [f'''import TopLemmas

/-! The member sums of the preprocessing's PRF keys (written by `generator/gen_pre_sums.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

''']
    for args in found:
        out.append(lemmas(path, ns, *args, nf=4, tag='pre'))
    out.append(f'end RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_PreSums.lean', 'w').write(''.join(out))
    print(part, [a[0] for a in found])
