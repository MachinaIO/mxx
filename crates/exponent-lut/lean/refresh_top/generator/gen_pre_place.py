"""Write `Part_<c_j>_PrePlace.lean`: the preprocessing's monomial placements of PRF members (the
key counterparts of the online placements).

Usage: `python3 gen_pre_place.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_part import parts_of

d = sys.argv[1]
for part in parts_of(d):
    ns = f'Stage_preprocess_{part}'
    text = open(f'{d}/generated/{ns}.lean').read()
    ks = []
    for m in re.finditer(r'^def (parallel_generatedRoot_(\d+)) ', text, re.M):
        i = m.start()
        body = text[i:text.index('\n\n', i)]
        if body.count('MxxRuntime.multiplyMonomial') == 4 and '∃' not in body:
            ks.append(int(m.group(2)))
    out = [f'''import TopLemmas

/-! The preprocessing's monomial placements (written by `generator/gen_pre_place.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
''']
    for k in ks:
        q = re.search(rf'def parallel_generatedRoot_{k} [^\n]*?inputs : ExactMatrix (\S+) 8 1 (\d+)', text)
        q = re.search(rf'def parallel_generatedRoot_{k} [^\n]*?Mxx\.Primitives\.ExactMatrix (\d+) 8 1 (\d+)', text)
        out.append(f'''
theorem prePlace{k} {{model : HashModel}} {{params : {ns}.Params}} {{o : Nat}}
    {{x y : Keys {q.group(1)} {q.group(2)}}}
    (h : {ns}.parallel_generatedRoot_{k} Backend.backend model params o
      (x.1, x.2.1, x.2.2.1, x.2.2.2.1, ()) y) :
    y = (multiplyMonomial x.1 (o : Int), multiplyMonomial x.2.1 (o : Int),
      multiplyMonomial x.2.2.1 (o : Int), multiplyMonomial x.2.2.2.1 (o : Int), ()) := by
  unfold {ns}.parallel_generatedRoot_{k} at h
  simp only [{ns}.parallel_generatedRoot_{k}.constraints_0] at h
  rw [h]
  rfl
''')
    out.append(f'\nend RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_PrePlace.lean', 'w').write(''.join(out))
    print(part, ks)
    break
