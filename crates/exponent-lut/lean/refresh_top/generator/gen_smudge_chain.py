"""Write `Part_<c_j>_SmudgeChain.lean`: per decoder, the smudge carrier over the root wires (the
decoder's `smudgeTotal` with its member sum `rootSum`).

Usage: `python3 gen_smudge_chain.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from chain import WIRE, chain_binders, wrap
from gen_part import parts_of
from patterns import parse
from segment import abbreviations

d = sys.argv[1]
ab = abbreviations(open(f'{d}/TopSetup.lean').read())


def tuple_args(atom):
    """The input tuple wires of a parallel atom (a family member `w i` gives `w`)."""
    i = atom.index('params i (')
    j = atom.index(')) ((w_', i) if ')) ((w_' in atom[i:] else len(atom)
    return WIRE.findall(atom[i:j])


for part in parts_of(d):
    ns = f'Stage_online_eval_{part}'
    path = f'{d}/generated/{ns}.lean'
    _, names, atoms = parse(path, 'generatedRoot')
    A = dict(zip(names, atoms))
    total = open(f'{d}/Part_{part}_SmudgeTotal.lean').read()
    luts = [int(x) for x in re.findall(r'\(hlut : ∀ o : Fin 160, \S+\.parallel_generatedRoot_(\d+) ', total)]
    out = [f'''import Part_{part}_SmudgeTotal
import Part_{part}_RootSums

/-! The smudge carriers of part `{part}` over the root wires (written by
`generator/gen_smudge_chain.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
''']
    for dec, L in enumerate(luts):
        C, S = L - 6, L + 19
        cell = tuple_args(A[f'h{C}_5'])
        assert len(cell) == 16, cell
        names0 = ['ca0', 'aA00', 'aA10', 'ca1', 'aA01', 'aA11', 'one00', 'one10', 'onec0', 'apay',
                  'one01', 'one11', 'onec1', 'key', 'bot0', 'bot1']
        mapping = dict(zip(names0, cell))
        mapping.update({"bot0'": tuple_args(A[f'h{C + 4}_5'])[14], "bot1'": tuple_args(A[f'h{C + 4}_5'])[15]})
        lut = tuple_args(A[f'h{L}_5'])
        mapping.update(dict(zip(['fam00', 'fam01', 'fam10', 'fam11', 'hl0', 'key', 'hl1'], lut[:7])))
        for p, ids in (('f', C), ('m', C + 1), ('g', C + 4), ('n', C + 5), ('u', L)):
            mapping.update({f'{p}{k}': f'w_{ids}_{k}' for k in range(6)})
        atom_args = {f'h{C}': f'h{C}_5', f'h{C + 1}': f'h{C + 1}_5', f'h{C + 4}': f'h{C + 4}_5',
                     f'h{C + 5}': f'h{C + 5}_5', 'hlut': f'h{L}_5'}
        kept, concl, call = wrap(total, f'smudgeTotal{dec}', mapping, atom_args)
        # The generic cell hypothesis names of `smudgeTotal` are those of decoder 0.
        fams = [n for n in names if re.match(rf'MxxRuntime\.familyGetStatic w_{L}_\d \(\d\) ', A[n])]
        inits = [x for x in re.findall(r'\(w_(\d+)_0, \(w_', A[f'h{S}_5'])]
        init_lets = re.search(r'\(Int\.toNat \(\d+\)\) \((.*?)\) \(w_' + str(S), A[f'h{S}_5']).group(1)
        init_lets = ' '.join(f'e_{w[2:]}' for w in WIRE.findall(init_lets))
        binders = chain_binders(path, ns, [], ab, list(atom_args.values()) + fams + [f'h{S}_5'])
        for k in range(6):
            concl = concl.replace(f'∑ o, w_{L}_{k} o', f'w_{S}_{k}')
        out.append(f'''
set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The smudge carrier of decoder `{dec}`. -/
theorem smudgeChain{dec} {{model : HashModel}} {{params : {ns}.Params}}
''' + '\n'.join('    ' + b for b in binders) + '\n' + '\n'.join('    ' + ' '.join(g.split()) for g in kept) +
                   f''' :
    {concl} := by
  subst {init_lets}
  have hs := rootSum{S} {' '.join(fams)} h{S}_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  rw [s0, s1, s2, s3, s4, s5]
  exact {call}
''')
    out.append(f'\nend RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_SmudgeChain.lean', 'w').write('\n'.join(out))
    break
