"""Write `Part_<c_j>_KeyFacts.lean`: the label keys of the encodings are hash samples.

`preHash<N>`: the preprocessing's `parallel_generatedRoot_<N>` that hashes a key with tag
`(member i, component c)` outputs `model.sample`. `encKeys<N>`: an encryption label encoding
outputs its key blocks unchanged.

Usage: `python3 gen_key_facts.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_part import parts_of
from patterns import destruct_inline

d = sys.argv[1]
MODULI = {'1645491229252209031554101362188915932898173126614280606821001729': 'Q7',
          '1532485821333670884548742795052537878078547363014104417': 'Q6'}
LABEL = '114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 49, 41, 47, 48'
for part in parts_of(d):
    pre = f'Stage_preprocess_{part}'
    enc = f'Stage_encrypt_{part}'
    out = [f'''import {pre}
import {enc}
import Part_{part}_KeysCells

/-! The label keys of the encodings of part `{part}` are hash samples (written by
`generator/gen_key_facts.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
''']
    text = open(f'{d}/generated/{pre}.lean').read()
    pat = (r'abbrev (parallel_generatedRoot_(\d+))\.constraints_0 \(hashModel : MxxRuntime\.HashModel\) '
           r'\(i_0 : Nat\) \(w_0_0 : ByteArray\) \(w_1_0 : Mxx\.Primitives\.ExactMatrix (\d+) 8 1 (\d+)\)[^\n]*\n'
           r'  MxxRuntime\.hashSample \(hashModel\) \(\[' + LABEL + r'\]\) \(\[[^\n]*\.u64Le \((\d)\)\]\) '
           r'\(w_0_0\) w_1_0 ∧\n  outputs = w_1_0\n')
    for m in re.finditer(pat, text):
        q = MODULI[m.group(3)]
        out.append(f'''
theorem preHash{m.group(2)} {{model : HashModel}} {{params : {pre}.Params}} {{i : Nat}} {{key : ByteArray}}
    {{out : ExactMatrix {q} N 1 {m.group(4)}}}
    (h : {pre}.{m.group(1)} Backend.backend model params i key out) :
    out = model.sample {q} N 1 {m.group(4)} key (completeHashTag labelPrefix
      [.bytes memberBytes, .u64Le (i : Int), .bytes componentBytes, .u64Le ({m.group(5)} : Int)]) := by
  obtain ⟨w, hs, rfl⟩ := h
  exact hashSample_eq hs
''')
    etext = open(f'{d}/generated/{enc}.lean').read()
    for n in re.findall(r'∀ i : Fin 5, (parallel_generatedRoot_\d+) backend hashModel params i \(\(w_\d+_0 i\), '
                        r'\(\(w_\d+_0 i\), \(\(w_\d+_0 i\), \(\(w_\d+_0 i\), \(w_\d+_0, \(w_\d+_0, \(\(w_\d+_0 i\)', etext):
        q = 'Q7' if re.search(rf'def {n} [^\n]*ExactMatrix 1645491229252209031554101362188915932898173126614280606821001729 8 1 14', etext) else 'Q6'
        cols = 14 if q == 'Q7' else 12
        num = n.rsplit('_', 1)[1]
        out.append(f'''
set_option maxHeartbeats 4000000 in
theorem encKeys{num} {{model : HashModel}} {{params : {enc}.Params}} {{o : Nat}}
    {{A00 A01 A10 A11 : ExactMatrix {q} N 1 {cols}}} {{smM oneM : ExactMatrix {q} N 1 1}} {{c e : Int}} {{outputs}}
    (h : {enc}.{n} Backend.backend model params o (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs) :
    outputs.1 = A00 ∧ outputs.2.1 = A01 ∧ outputs.2.2.1 = A10 ∧ outputs.2.2.2.1 = A11 := by
{destruct_inline(f'{d}/generated/{enc}.lean', enc, n, 'h')}  subst hout
  exact ⟨rfl, rfl, rfl, rfl⟩
''')
    out.append(f'\nend RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_KeyFacts.lean', 'w').write('\n'.join(out))
    break
