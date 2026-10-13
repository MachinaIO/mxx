"""Write `Part_<c_j>_RootLuts.lean`: the loop lemmas of the LUT loops of the online root (the
width-`2` source LUT and the two LUTs of the mask's binary offset).

Usage: `python3 gen_root_luts.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_luts import loop_lemmas
from gen_part import parts_of
from layouts import layouts

d = sys.argv[1]
L = layouts(d)
for part in parts_of(d):
    path = f'{d}/generated/Stage_online_eval_{part}.lean'
    ns = f'Stage_online_eval_{part}'
    text = open(path).read()
    lut_a = '[114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]'
    def is_lut(k):
        body = text[text.index(f'abbrev sequential_generatedRoot_{k}.constraints_0 '):]
        return lut_a in body[:body.index('\n\n')]
    loops = sorted({int(k) for k in re.findall(r'^def sequential_generatedRoot_(\d+) ', text, re.M)})
    loops = [k for k in loops if is_lut(k)]
    assert len(loops) == 3, loops
    def level(k):
        body = text[text.index(f'abbrev sequential_generatedRoot_{k}.constraints_0 '):]
        return int(re.search(r'\[\.u64Le \((\d+)\), \.integer', body).group(1))
    source, readout, sample = loops
    # Instance `I` of the merged stages decodes at `R i0` (`Q0 = qd i0 * R i0`).
    inst = part == 'I'
    R, qd, q0eq = ('(R i0)', '(qd i0)', '(q0_eqR i0 hi0)') if inst else ('Q1', 'q3', 'q0_eq')
    body = loop_lemmas(path, ns, f'sequential_generatedRoot_{source}', str(source), 'Q0', R, qd,
                       6, 4, 'Backend.' + L['Q1'], 'layoutQ1', q0eq, level(source), 16384, width=2,
                       inst=inst)
    body += loop_lemmas(path, ns, f'sequential_generatedRoot_{readout}', str(readout), 'Q5', 'Q0',
                        'p4', 8, 6, 'Backend.' + L['Q0'], 'layoutQ0', 'q5_eq', level(readout), 16384)
    body += loop_lemmas(path, ns, f'sequential_generatedRoot_{sample}', str(sample), 'Q0', R,
                        qd, 6, 6, 'Backend.' + L['Q1'], 'layoutQ1', q0eq, level(sample), 16384,
                        inst=inst)
    open(f'{d}/Part_{part}_RootLuts.lean', 'w').write(f'''import TopLut2
{'import TopInst' if inst else ''}

/-! The LUT loops of the online root: the source LUT and the two LUTs of the mask's binary offset
(written by `generator/gen_root_luts.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
{body}
end RefreshTopProof.Part_{part}
''')
    print(part, loops, [level(k) for k in loops])
    break
