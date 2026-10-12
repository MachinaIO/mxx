"""Write `Part_I_Local.lean`: the lemmas of the shared `Top*` files that mention the decoding ring
`Q₁` of chain `0` (or its constants), at the decoding ring `R i0` of instance `i0` of the merged
stages, in the instance's namespace, where they shadow the shared ones (as `gen_chain3.py` does
for chain `3`). The proofs come from `template/Local.lean` once ported by hand; this script writes
the mechanical substitution when there is none.

Usage: `python3 gen_local.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_chain3 import TOP, MARK, SKIP, chunks, name_of  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))


def subst(text):
    text = re.sub(r'\bhalfQ1\b', '(halfZ (i0 : Int))', text)
    text = re.sub(r'\b618968813111010066723674601\b', '(halfZ (i0 : Int))', text)
    text = re.sub(r'\bQ1\b', '(R i0)', text)
    text = re.sub(r'Backend\.layout4\s+layoutQ1\s+rfl\b', 'L hL hLx', text)
    return text


def main(d):
    port = f'{here}/template/Local.lean'
    if os.path.exists(port):
        open(f'{d}/Part_I_Local.lean', 'w').write(open(port).read())
        return
    out = ['''import TopChain
import TopLinkC
import TopFinal
import TopInst

/-! The lemmas of the decoding ring of instance `i0`: those of the shared `Top*` files that
mention `Q₁`, at `R i0` (written by `generator/gen_local.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof
''']
    for f in TOP:
        for c in chunks(open(f'{d}/{f}.lean').read()):
            n = name_of(c)
            if n is None or n in SKIP or n == 'halfQ1' or not MARK.search(c):
                continue
            out.append(subst(c).rstrip() + '\n')
    out.append('end RefreshTopProof.Part_I\n')
    open(f'{d}/Part_I_Local.lean', 'w').write('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1])
