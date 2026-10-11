"""Write `Part_<c_j>_Labels.lean`: the label payloads `±X^e` the online stage bakes in (parallels
`(∀ i : Fin 5, parallel_generatedRoot_k … i () (w i))`) are `X^(2 a_i)` for the label exponents
`a = labelExp`.

Usage: `python3 gen_labels.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_part import parts_of
from patterns import parse
from segment import abbreviations, short

d = sys.argv[1]
ab = abbreviations(open(f'{d}/TopSetup.lean').read())
for part in parts_of(d):
    ns = f'Stage_online_eval_{part}'
    path = f'{d}/generated/{ns}.lean'
    text = open(path).read()
    _, names, atoms = parse(path, 'generatedRoot')
    out = [f'''import TopLabel
import {ns}

/-! The label payloads of the online stage of part `{part}` (written by `generator/gen_labels.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
''']
    for n, a in zip(names, atoms):
        m = re.match(r'\(∀ i : Fin 5, parallel_generatedRoot_(\d+) backend hashModel params i \(\) \((w_\d+_0) i\)\)', a)
        if not m:
            continue
        k = m.group(1)
        q = short(re.search(rf'def parallel_generatedRoot_{k} .*?\(outputs : Mxx\.Primitives\.ExactMatrix (\d+) ', text).group(1), ab)
        out.append(f'''
theorem labelPay{k} {{model : HashModel}} {{params : {ns}.Params}} (i : Fin 5)
    {{out : ExactMatrix {q} N 1 1}}
    (h : {ns}.parallel_generatedRoot_{k} Backend.backend model params i () out) :
    out = fun _ _ ↦ reducePoly {q} N (Xe (n := N) (2 * labelExp i)) := by
  unfold {ns}.parallel_generatedRoot_{k} at h
  simp only [{ns}.parallel_generatedRoot_{k}.constraints_0] at h
  rw [h.2.2.2.2]
  exact label_value i
''')
    out.append(f'\nend RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_Labels.lean', 'w').write('\n'.join(out))
    break
