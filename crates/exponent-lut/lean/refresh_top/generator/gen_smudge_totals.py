"""Write `Part_<c_j>_SmudgeTotal.lean`: every decoder's smudge, the sum over its 160 members of
the member's LUT output (a signed odd digit times the digit's weight, placed at its coefficient).

Usage: `python3 gen_smudge_totals.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_part import parts_of

d = sys.argv[1]
for part in parts_of(d):
    members = open(f'{d}/Part_{part}_SmudgeMembers.lean').read()
    smudge = open(f'{d}/Part_{part}_Smudge.lean').read()
    luts = re.findall(r'theorem lut(\d+) ', smudge)
    out = [f'''import Part_{part}_SmudgeMembers
import Part_{part}_Smudge

/-! Every decoder's smudge (written by `generator/gen_smudge_totals.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
''']
    for dec, L in enumerate(luts):
        level = re.search(rf'theorem lut{L} [\s\S]*?lutA model key Q0 6 (\d+)', smudge).group(1)
        i = members.index(f'theorem smudge{dec}Member1')
        j = members.index(' (o : Fin 160) :', i)
        hyps = members[members.index('{model', i):j]
        mb = f'(5 * smudge{dec}Cell1Bound Ba (5 * smudge{dec}Cell0Bound Ba Bs BB) BB\')'
        out.append(f'''
/-- The bound of one member's LUT output of decoder `{dec}`. -/
abbrev smudge{dec}LutBound (Ba Bs BB BB' Bh BS : Nat) : Nat :=
  (6 * ((3 * lutBranchBound{L} {mb} Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * {mb} * 16384 +
    (N * 1 * (q3 / 2) + q3 / 2)) + (N + 2) * (q3 / 2)) / q3 + 4 * (N * 1 * BS)) * (32768 / 2))

/-- The smudge of decoder `{dec}`: `Σ_o X^⌊o/20⌋ w_(o mod 20) T'(z_o)`. -/
noncomputable def smudgeValue{dec} (z : Fin 160 → Nat) : ErrorPoly N :=
  ∑ o : Fin 160, Xe (n := N) (o / 20) *
    (((smudgeWeight{L} ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
      ((1 : ErrorPoly N) * ((((-1 : Int) ^ (z o / 4) : Int) : ErrorPoly N) *
        smudgeTable ⟨z o % 4, Nat.mod_lt _ (by norm_num)⟩)))

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem smudgeTotal{dec} {hyps}
    {{fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}} {{hl0 hl1 : Fin 4 → ExactMatrix Q0 N 1 6}}
    {{u0 u1 u2 u3 u4 u5 : Fin 160 → ExactMatrix Q1 N 1 1}}
    (hlut : ∀ o : Fin 160, Stage_online_eval_{part}.parallel_generatedRoot_{L} Backend.backend model
      params o (fam00, fam01, fam10, fam11, hl0, key, hl1, n4 o, n0 o, n2 o, n5 o, n1 o, n3 o, ())
        (u0 o, u1 o, u2 o, u3 o, u4 o, u5 o, ()))
    {{g1' : ExactMatrix Q1 N 1 6}} (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1')
    (hh5 : centeredRebaseRuns g1' h5)
    {{Bh : Nat}} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 {level} (sigma i))
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![hl0 i, hl1 i] Bh)
    {{gq : ExactMatrix Q1 N 1 6}} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {{inverse : Nat}}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {{BS : Nat}}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (smudgeTable k)) (sigma i).toNat *
          reducePoly Q1 N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    Rel sm ![![∑ o, u0 o, ∑ o, u1 o], ![∑ o, u2 o, ∑ o, u3 o]]
      (fun j ↦ reducePoly Q1 N (smudgeValue{dec} (fun o ↦ a (sel' o) + a (sel o)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![∑ o, u4 o, ∑ o, u5 o] (160 * smudge{dec}LutBound Ba Bs BB BB' Bh BS) := by
  have hmem : ∀ o ∈ (Finset.univ : Finset (Fin 160)), Rel sm ![![u0 o, u1 o], ![u2 o, u3 o]]
      (fun j ↦ reducePoly Q1 N (Xe (n := N) (o / 20)) • reducePoly Q1 N
        (((smudgeWeight{L} ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
          ((1 : ErrorPoly N) * ((((-1 : Int) ^ ((a (sel' o) + a (sel o)) / 4) : Int) : ErrorPoly N) *
            smudgeTable ⟨(a (sel' o) + a (sel o)) % 4, Nat.mod_lt _ (by norm_num)⟩)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![u4 o, u5 o] (smudge{dec}LutBound Ba Bs BB BB' Bh BS) := by
    intro o _
    have hm := smudge{dec}Member1 {' '.join(re.findall(r'\((h\d+) : ∀', hyps)[:4])} hsm hG hg6' hg7 hg5' hh6 ha hpay hone hrhs heB hrhs'
      heB' hsel hsel' o
    exact lut{L} (hlut o) hsm hg1' hh5 hm hhelp o.isLt hgq hinverse hswitch
  have hsum := Rel.sum Finset.univ hmem
  rw [blocks_sum, pair_sum] at hsum
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  rw [Finset.sum_apply, smudgeValue{dec}, Finset.sum_mul, map_sum, Finset.sum_smul]
  refine Finset.sum_congr rfl fun o _ ↦ ?_
  rw [smul_smul, ← map_mul]
  congr 2
  ring
''')
    out.append(f'\nend RefreshTopProof.Part_{part}\n')
    txt = ''.join(out)
    # the member lemma's parallel hypothesis names
    open(f'{d}/Part_{part}_SmudgeTotal.lean', 'w').write(txt)
    print(part, luts)
    break
