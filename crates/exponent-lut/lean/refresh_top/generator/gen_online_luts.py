"""Write `Part_<c_j>_Readout.lean` and `Part_<c_j>_Smudge.lean`: the member LUTs of the online
evaluation (the PRF readout and sampling LUTs of the mask and the fresh noise, and every decoder's
smudge LUT).

Usage: `python3 gen_online_luts.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_luts import member_lut
from gen_part import parts_of
from layouts import layouts

d = sys.argv[1]
L = layouts(d)
for part in parts_of(d):
    # Instance `I` of the merged stages decodes at `R i0`, dropping `qd i0` from `Q0`.
    inst = part == 'I'
    R, qd, q0eq = ('(R i0)', '(qd i0)', '(q0_eqR i0 hi0)') if inst else ('Q1', 'q3', 'q0_eq')
    Q3 = f'((({qd} : Nat) : Int) : ErrorPoly N)'
    path = f'{d}/generated/Stage_online_eval_{part}.lean'
    ns = f'Stage_online_eval_{part}'
    text = open(path).read()
    # The member LUTs: root parallels over members whose body runs a LUT loop.
    luts = []
    lut_a = '[114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]'
    for c, k in re.findall(r'\(∀ i : Fin (\d+), parallel_generatedRoot_(\d+) backend', text):
        m = re.search(rf'^abbrev (sequential_parallel_generatedRoot_{k}_\d+)\.constraints_0 ', text, re.M)
        if m and lut_a in text[m.start():m.start() + 40000]:
            luts.append(int(k))
    luts = sorted(set(luts))
    readout = [luts[0], luts[2]]  # mask, fresh readout
    sample = [luts[1], luts[3]]   # mask (with offset), fresh sampling
    smudges = luts[4:]
    body = ''
    body += member_lut(path, ns, readout[0], 'round', 'Q5', 'Q0', 'p4', 8, 6, 'Backend.' + L['Q0'],
                       'layoutQ0', 'q5_eq', 0, scale=Q3, table='readoutTable')
    body += member_lut(path, ns, sample[0], 'branches', 'Q0', R, qd, 6, 6, 'Backend.' + L['Q1'],
                       'layoutQ1', q0eq, 0, offset=True, inst=inst)
    body += member_lut(path, ns, readout[1], 'round', 'Q5', 'Q0', 'p4', 8, 6, 'Backend.' + L['Q0'],
                       'layoutQ0', 'q5_eq', 0, scale=Q3, table='readoutTable')
    body += member_lut(path, ns, sample[1], 'branches', 'Q0', R, qd, 6, 6, 'Backend.' + L['Q1'],
                       'layoutQ1', q0eq, 0, inst=inst)
    open(f'{d}/Part_{part}_Readout.lean', 'w').write(f'''import TopLut
{'import TopInst' if inst else ''}

/-! The PRF readout and sampling LUTs of the mask and the fresh noise (written by
`generator/gen_online_luts.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

/-- The readout table of the PRF: `X^(2 (2 f(k) - 1))`, `f = (1, 1, 2, 2)`. -/
noncomputable def readoutTable (k : Fin 4) : ErrorPoly N := Xe (2 * ![1, 1, 3, 3] k)
{body}
end RefreshTopProof.Part_{part}
''')
    sm = ''.join(member_lut(path, ns, k, 'smudge', 'Q0', R, qd, 6, 6, 'Backend.' + L['Q1'],
                            'layoutQ1', q0eq, 0, scale='(1 : ErrorPoly N)', table='smudgeTable',
                            inst=inst)
                 for k in smudges)
    open(f'{d}/Part_{part}_Smudge.lean', 'w').write(f'''import TopLut
import TopPlace
{'import TopInst' if inst else ''}

/-! The smudge LUTs of the decoders (written by `generator/gen_online_luts.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

/-- The readout table of a decoder's smudge PRF: the coefficients `2 f(k) - 1`. -/
noncomputable def smudgeTable (k : Fin 4) : ErrorPoly N := ((![1, 1, 3, 3] k : Int) : ErrorPoly N)
{sm}
end RefreshTopProof.Part_{part}
''')
    print(part, readout, sample, smudges)
    break
