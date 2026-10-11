"""Write `Part_<c_j>_EncFacts.lean`: the label encodings of every PRF child encode `X^(2 a_k)` for the
label exponents `a = labelExp` (from the requirement's label check and the externals' bounds).

Usage: `python3 gen_enc_facts.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_part import parts_of

d = sys.argv[1]
for part in parts_of(d):
    enc = open(f'{d}/Part_{part}_Enc.lean').read()
    out = [f'''import Part_{part}_Enc
import TopLabel

/-! The label encodings of part `{part}` (written by `generator/gen_enc_facts.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
''']
    for P in re.findall(r'theorem encLabel(\d+) ', enc):
        head = enc[enc.index(f'theorem encLabel{P} '):]
        head = head[:head.index(':= by')]
        q, m = re.search(r'\{A00 A01 A10 A11 : ExactMatrix (\S+) N 1 (\d+)\}', head).groups()
        out.append(f'''
/-- The label encodings of the encryption parallel `{P}`. -/
theorem encLabelRel{P} {{model : HashModel}} {{params : Stage_encrypt_{part}.Params}}
    {{A00 A01 A10 A11 : Fin 5 → ExactMatrix {q} N 1 {m}}} {{smM oneM : ExactMatrix {q} N 1 1}}
    {{c e : Fin 5 → Int}} {{o0 o1 o2 o3 o4 o5 : Fin 5 → ExactMatrix {q} N 1 {m}}} {{o6 : Fin 5 → ExactMatrix {q} N 1 1}}
    (h : ∀ i : Fin 5, Stage_encrypt_{part}.parallel_generatedRoot_{P} Backend.backend model params i
      (A00 i, A01 i, A10 i, A11 i, smM, oneM, c i, e i, ()) (o0 i, o1 i, o2 i, o3 i, o4 i, o5 i, o6 i, ()))
    {{sm : ErrorPoly N}} (hsm : smM = fun _ _ ↦ reducePoly {q} N sm) (hone : oneM = 1)
    {{G : ExactMatrix {q} N 1 {m}}} (hG : gadgetMatrixRuns Backend.backend 32768 {m} G)
    (hc : ∀ k, c k * c k = 1) (hr : ∀ k, e k * 2 + (1 - c k) * 8 = (labelExp k : Int) * 4)
    (he : ∀ k, 0 ≤ e k ∧ e k ≤ 7) :
    ∀ k : Fin 5, Rel sm ![![o0 k, o1 k], ![o2 k, o3 k]]
      (fun j ↦ reducePoly {q} N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • G) ![o4 k, o5 k] 26 := by
  intro k
  obtain ⟨hk, hrel, -⟩ := encLabel{P} (h := h k) hsm hone hG
  simp only [Prod.mk.injEq] at hk
  obtain ⟨k0, k1, k2, k3⟩ := hk
  rw [label_message (hc k) (hr k) (he k).1 (he k).2, Int.toNat_natCast] at hrel
  rw [k0, k1, k2, k3]
  exact hrel
''')
    out.append(f'\nend RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_EncFacts.lean', 'w').write('\n'.join(out))
    break
