"""Write `Part_<c_j>_SmudgeMembers.lean`: the member sums of every decoder's smudge PRF, from the
mask PRF's member lemmas (`Part_<c_j>_Mask.lean`) one level down (label and one encoding at
`Q6`, buckets at `Q5` and `Q0`).

Usage: `python3 gen_smudge_members.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_part import parts_of

d = sys.argv[1]
for part in parts_of(d):
    mask = open(f'{d}/Part_{part}_Mask.lean').read()
    start = mask.index('/-- The bound of a bucket-`0` cell. -/')
    stop = mask.index('/-- The readout exponent:')
    base = mask[start:stop]
    text = open(f'{d}/generated/Stage_online_eval_{part}.lean').read()
    cells = sorted({int(k) for k in re.findall(r'∀ i : Fin 800, parallel_generatedRoot_(\d+) ', text)})
    assert len(cells) == 12, cells
    out = [f'''import Part_{part}_Mask

/-! The member sums of every decoder's smudge PRF (written by `generator/gen_smudge_members.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
''']
    for dec in range(6):
        c0, c1 = cells[2 * dec], cells[2 * dec + 1]
        s0, s1 = c0 + 1, c1 + 1
        body = base
        # Fin sizes first (members and cells), through placeholders.
        body = body.replace('Fin 40', '@CELLS@').replace('Fin 8', '@MEMBERS@')
        # One level down: moduli, gadget widths and scales, simultaneously.
        mods = {'Q7': 'Q6', 'Q6': 'Q5', 'Q5': 'Q0', 'p6': 'p5', 'p5': 'p4', 'p4': 'q3',
                '14': '12', '12': '10', '10': '8', '8': '6'}
        body = re.sub(r'\b(Q7|Q6|Q5|p6|p5|p4|14|12|10|8)\b', lambda m: mods[m.group(1)], body)
        body = body.replace('@CELLS@', 'Fin 800').replace('@MEMBERS@', 'Fin 160')
        names = {'maskCell0Bound': f'smudge{dec}Cell0Bound', 'maskCell1Bound': f'smudge{dec}Cell1Bound',
                 'maskMember0': f'smudge{dec}Member0', 'maskMember1': f'smudge{dec}Member1',
                 'mask_cell0': f'smudge{dec}_cell0', 'mask_cell1': f'smudge{dec}_cell1',
                 'maskRhs0': f'smudge{dec}Rhs0', 'maskRhs1': f'smudge{dec}Rhs1',
                 'parallel_generatedRoot_115': f'parallel_generatedRoot_{c0}',
                 'parallel_generatedRoot_116': f'parallel_generatedRoot_{s0}',
                 'parallel_generatedRoot_119': f'parallel_generatedRoot_{c1}',
                 'parallel_generatedRoot_120': f'parallel_generatedRoot_{s1}',
                 'members116': f'members{s0}', 'members120': f'members{s1}',
                 'h115': f'h{c0}', 'h116': f'h{s0}', 'h119': f'h{c1}', 'h120': f'h{s1}',
                 'mask PRF': f'smudge PRF of decoder {dec}'}
        body = re.sub(r'\b(' + '|'.join(re.escape(k) for k in names) + r')\b',
                      lambda m: names[m.group(1)], body)
        out.append(body)
    out.append(f'end RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_SmudgeMembers.lean', 'w').write('\n'.join(out))
    print(part, cells)
    break
