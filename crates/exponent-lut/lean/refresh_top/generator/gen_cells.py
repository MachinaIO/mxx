"""Write `Part_<c_j>_Cells.lean`: the cell lemmas of both buckets of every PRF child of the online
evaluation (`templates/cells.fmt`, the mask cells with their levels as parameters).

Usage: `python3 gen_cells.py <parameter directory>`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import gen_part
from gen_part import instantiate, parts_of
from layouts import layouts

d = sys.argv[1]
L = layouts(d)
gen_part.d = d
here = os.path.dirname(os.path.abspath(__file__))
fmt = open(f'{here}/templates/cells.fmt').read()

TOP = dict(QA='Q7', QB='Q6', QC='Q5', PA='p6', PB='p5', PC='p4', MA=14, MB=12, MC=10, MD=8,
           LA=L['Q7'], LB=L['Q6'], LC=L['Q5'],
           HA='layoutQ7', HB='layoutQ6', HC='layoutQ5', EA='q7_eq', EB='q6_eq')
LOW = dict(QA='Q6', QB='Q5', QC='Q0', PA='p5', PB='p4', PC='q3', MA=12, MB=10, MC=8, MD=6,
           LA=L['Q6'], LB=L['Q5'], LC=L['Q0'],
           HA='layoutQ6', HB='layoutQ5', HC='layoutQ0', EA='q6_eq', EB='q5_eq')
# (name, bucket-0 cell def, bucket-1 cell def, cells, members, levels)
CHILDREN = [('mask', 115, 119, 40, 8, TOP), ('fresh', 386, 390, 240, 48, TOP)] + [
    (f'smudge{c}', 554 + 54 * c + (1 if c >= 1 else 0) * 0, 0, 800, 160, LOW) for c in range(0)]


def smudge_children(path):
    """The cell definitions of the smudge PRFs, found by their cell count."""
    import re
    text = open(path).read()
    out = []
    for k in sorted(set(int(x) for x in re.findall(r'^def parallel_generatedRoot_(\d+) ', text, re.M))):
        body = text[text.index(f'def parallel_generatedRoot_{k} '):]
        head = body[:body.index('\n')]
        if 'Fin 5 → Mxx.Primitives.ExactMatrix' in head and '(w_1_0 < 800)' in body[:200]:
            out.append(k)
    return out


for part in parts_of(d):
    import re
    path = f'{d}/generated/Stage_online_eval_{part}.lean'
    text = open(path).read()
    # The cell definitions: the root parallels over the cells of a bucket.
    defs = [int(k) for c, k in re.findall(r'\(∀ i : Fin (\d+), parallel_generatedRoot_(\d+) ', text)
            if int(c) in (40, 240, 800)]
    defs = sorted(set(defs))
    # Pair them: bucket 0 then bucket 1 of each child, in order.
    pairs = [(defs[2 * c], defs[2 * c + 1]) for c in range(len(defs) // 2)]
    names = ['mask', 'fresh'] + [f'smudge{c}' for c in range(len(pairs) - 2)]
    out = [f'''import TopLemmas

/-!
# The PRF cells (written by `generator/gen_cells.py`)

Cell `i = 5 o + k` of bucket `b` of a PRF child multiplies member `o`'s state by member `k` of the
label `a` (a public multiplication, after aligning `a` to the state's modulus in bucket `1`) and by
its selector bit (a private multiplication that drops one prime).
-/

namespace RefreshTopProof.Part_@P@

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

''']
    for (d0, d1), name in zip(pairs, names):
        cells = 40 if name == 'mask' else 240 if name == 'fresh' else 800
        members = cells // 5
        lv = TOP if name in ('mask', 'fresh') else LOW
        # The RHS tags: the prefix and the instance's digests (instance `I`, see `rhsTop_tag`).
        binder = ' (i0 : Nat)' if part == 'I' else ''
        out.append(f'abbrev {name}Rhs0{binder} : Blob := @TAGX(online_eval, parallel_generatedRoot_{d0}, h46_0)@\n')
        out.append(f'abbrev {name}Rhs1{binder} : Blob := @TAGX(online_eval, parallel_generatedRoot_{d1}, h76_0)@\n\n')
        out.append(fmt.format(NAME=name, D0=d0, D1=d1, CELLS=cells, MEMBERS=members, **lv))
    out.append('\nend RefreshTopProof.Part_@P@\n')
    open(f'{d}/Part_{part}_Cells.lean', 'w').write(instantiate(''.join(out), part))
    print(part, pairs)
