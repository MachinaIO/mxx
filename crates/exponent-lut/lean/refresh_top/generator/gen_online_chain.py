"""Write `Part_<c_j>_OnlineChain.lean`: chain theorems over the online root, each applying several
segment theorems in sequence (see `chain.py`).

- `offsetChain`: the encoding of one moved down by two private multiplications (`rootOffset`),
  read out at `X^2` (`rootLut186`) and expanded into its LUT branches (`rootLut252`): the offset
  of the mask PRF's binary sampling.

Usage: `python3 gen_online_chain.py <parameter directory>`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from chain import chain_binders, conclusion_bound, seg_call, subst
from gen_part import parts_of
from segment import abbreviations

d = sys.argv[1]
ab = abbreviations(open(f'{d}/TopSetup.lean').read())


def indent(binders):
    return '\n'.join('    ' + b for b in binders)


for part in parts_of(d):
    ns = f'Stage_online_eval_{part}'
    path = f'{d}/generated/{ns}.lean'
    online = open(f'{d}/Part_{part}_Online.lean').read()
    texts = [(online, 'rootOffset'), (online, 'rootLut186'), (online, 'rootLut252')]
    binders = chain_binders(path, ns, texts, ab)
    b0 = subst(conclusion_bound(online, 'rootOffset'), {'B': 'Bone'})
    b1 = subst(conclusion_bound(online, 'rootLut186'), {'B': f'({b0})', 'Bh': 'Bh5', 'BB': 'BS'})
    b2 = subst(conclusion_bound(online, 'rootLut252'), {'B': f'({b1})', 'Bh': 'Bh0'})
    out = [f'''import Part_{part}_Online
import Part_{part}_Readout
import Part_{part}_MaskCarrier
import Part_{part}_RootSums

/-! Chain theorems of the online root of part `{part}` (written by `generator/gen_online_chain.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

/-- The bound of the offset of the mask PRF's binary sampling. -/
abbrev offsetBound (Bone BB BB' Bh5 BS Bh0 : Nat) : Nat :=
  {b2}

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The offset of the mask PRF's binary sampling: the encoding of one, moved down, read out at
`X^2` and expanded into the LUT branches of `X^2`. -/
theorem offsetChain {{model : HashModel}} {{params : {ns}.Params}}
{indent(binders)}
    {{sm sp : ErrorPoly N}} (hsm : polyNorm sm ≤ 1)
    {{g6' : ExactMatrix Q6 N 1 12}} {{g7 : ExactMatrix Q7 N 1 12}}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 12 g6') (hg7 : centeredRebaseRuns g6' g7)
    {{Bone : Nat}}
    (hone : Rel sm ![![w_106_0, w_110_0], ![w_107_0, w_111_0]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * 1 * unitPair sp j) • g7) ![w_108_0, w_112_0] Bone)
    {{g5' : ExactMatrix Q5 N 1 10}} {{g6 : ExactMatrix Q6 N 1 10}}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 10 g5') (hg6 : centeredRebaseRuns g5' g6)
    {{eB : Fin 2 → ErrorMatrix N 1 10}} {{BB : Nat}}
    (hrhs : ∀ j, reducePoly Q6 N sp • ![w_135_0, w_148_0] j + ![w_136_0, w_149_0] j =
      reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g6 + reduceMatrix Q6 N 1 10 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    {{g0' : ExactMatrix Q0 N 1 8}} {{h5 : ExactMatrix Q5 N 1 8}}
    (hg0' : gadgetMatrixRuns Backend.backend 32768 8 g0') (hh5 : centeredRebaseRuns g0' h5)
    {{eB' : Fin 2 → ErrorMatrix N 1 8}} {{BB' : Nat}}
    (hrhs' : ∀ j, reducePoly Q5 N sp • ![w_162_0, w_174_0] j + ![w_163_0, w_175_0] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * 0) * unitPair sp j) • h5 +
        reduceMatrix Q5 N 1 8 (eB' j))
    (heB' : ∀ j, CoeffBound (eB' j) BB')
    {{Bh5 : Nat}} (hhelp5 : ∀ i : Fin 4, Rel sm (lutA model w_92_0 Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![w_98_0 i, w_99_0 i] Bh5)
    {{g1' : ExactMatrix Q1 N 1 6}} {{g0 : ExactMatrix Q0 N 1 6}}
    (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1') (hg0 : centeredRebaseRuns g1' g0)
    {{inverse : Nat}} (hinverse : (inverse : ExactPoly Q0 N) * ((4 : Nat) : ExactPoly Q0 N) = 1)
    {{BS : Nat}}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_94_0, w_95_0], ![w_96_0, w_97_0]] 0 j i + ![![w_94_0, w_95_0], ![w_96_0, w_97_0]] 1 j i =
        (lutCoefficient Q0 inverse (fun k ↦ reducePoly Q0 N (readoutTable k)) (sigma i).toNat *
          reducePoly Q0 N ((((q3 : Nat) : Int) : ErrorPoly N) * unitPair sp j)) • g0 +
            reduceMatrix Q0 N 1 6 (e j))
    {{Bh0 : Nat}} (hhelp0 : ∀ i : Fin 4, Rel sm (lutA model w_92_0 Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g0) ![w_91_0 i, w_93_0 i] Bh0) :
    Rel sm ![![w_285_0, w_299_0], ![w_301_0, w_303_0]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 q3
        (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * 1))) •
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![w_87_0, w_88_0], ![w_89_0, w_90_0]] 0 j i + ![![w_87_0, w_88_0], ![w_89_0, w_90_0]] 1 j i)))
      ![w_307_0, w_311_0] (offsetBound Bone BB BB' Bh5 BS Bh0) := by
  have h1 := {seg_call(online, 'rootOffset')} (hsm := hsm) (hg6' := hg6') (hg7 := hg7)
    (hinput := hone) (hg5' := hg5') (hg6 := hg6) (hrhs := hrhs) (heB := heB) (hrhs' := hrhs')
    (heB' := heB')
  have h2 := {seg_call(online, 'rootLut186')} (z := 0) (hsm := hsm) (hg' := hg0') (hg := hh5)
    (hinput := h1) (hhelp := hhelp5) (hinverse := hinverse) (hswitch := hswitch)
  have e : ((((-1 : Int) ^ (0 / 4) : Int) : ErrorPoly N) *
      readoutTable ⟨0 % 4, Nat.mod_lt _ (by norm_num)⟩) = Xe (n := N) (2 * 1) := by
    simp [readoutTable]
  rw [e] at h2
  exact {seg_call(online, 'rootLut252')} (z := 1) (hsm := hsm) (hg' := hg1') (hg := hg0)
    (hinput := h2) (hhelp := hhelp0)

''']
    chain_text = out[0]
    carrier = open(f'{d}/Part_{part}_MaskCarrier.lean').read()
    off_args = ' '.join(f'({b} := {b})' for b in __import__('chain').seg_args(chain_text, 'offsetChain'))
    sums = open(f'{d}/Part_{part}_RootSums.lean').read()
    mb = subst(conclusion_bound(carrier, 'maskCarrier'), {
        'Br': '(maskReadoutBound Ba Bs BBm BBm' + "'" + ' Bh5 BS)', 'Bh': 'Bh0',
        'Boff': '(offsetBound Bs BB BB' + "'" + ' Bh5 BS Bh0)', 'BS': 'BS1'})
    mbinders = chain_binders(path, ns, [(chain_text, 'offsetChain')], ab,
                             ['h115_5', 'h116_5', 'h119_5', 'h120_5', 'h121_5', 'h312_5', 'h313_5',
                              'h314_0', 'h315_0', 'h317_0', 'h318_0', 'h320_0', 'h321_0', 'h323_0',
                              'h324_0', 'h326_0', 'h327_0', 'h329_0', 'h330_0', 'h332_5', 'h333_0'])
    out.append(f'''
/-- The bound of the mask carrier. -/
abbrev maskBound (Ba Bs BBm BBm' BB BB' Bh5 BS Bh0 BS1 : Nat) : Nat :=
  {mb}

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The mask carrier: the mask PRF's members read out, sampled with the offset, placed, summed and
moved down to `Q₁`. -/
theorem maskChain {{model : HashModel}} {{params : {ns}.Params}}
{indent(mbinders)}
    {{sm sp : ErrorPoly N}} (hsm : polyNorm sm ≤ 1)
    {{g6' : ExactMatrix Q6 N 1 12}} {{g7 : ExactMatrix Q7 N 1 12}}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 12 g6') (hg7 : centeredRebaseRuns g6' g7)
    {{Bs : Nat}}
    (hone : Rel sm ![![w_106_0, w_110_0], ![w_107_0, w_111_0]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * 1 * unitPair sp j) • g7) ![w_108_0, w_112_0] Bs)
    {{g5' : ExactMatrix Q5 N 1 10}} {{g6 : ExactMatrix Q6 N 1 10}}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 10 g5') (hg6 : centeredRebaseRuns g5' g6)
    {{eB : Fin 2 → ErrorMatrix N 1 10}} {{BB : Nat}}
    (hrhs : ∀ j, reducePoly Q6 N sp • ![w_135_0, w_148_0] j + ![w_136_0, w_149_0] j =
      reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g6 + reduceMatrix Q6 N 1 10 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    {{g0' : ExactMatrix Q0 N 1 8}} {{h5 : ExactMatrix Q5 N 1 8}}
    (hg0' : gadgetMatrixRuns Backend.backend 32768 8 g0') (hh5 : centeredRebaseRuns g0' h5)
    {{eB' : Fin 2 → ErrorMatrix N 1 8}} {{BB' : Nat}}
    (hrhs' : ∀ j, reducePoly Q5 N sp • ![w_162_0, w_174_0] j + ![w_163_0, w_175_0] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * 0) * unitPair sp j) • h5 +
        reduceMatrix Q5 N 1 8 (eB' j))
    (heB' : ∀ j, CoeffBound (eB' j) BB')
    {{Bh5 : Nat}} (hhelp5 : ∀ i : Fin 4, Rel sm (lutA model w_92_0 Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![w_98_0 i, w_99_0 i] Bh5)
    {{g1' : ExactMatrix Q1 N 1 6}} {{g0 : ExactMatrix Q0 N 1 6}}
    (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1') (hg0 : centeredRebaseRuns g1' g0)
    {{inverse : Nat}} (hinverse : (inverse : ExactPoly Q0 N) * ((4 : Nat) : ExactPoly Q0 N) = 1)
    {{BS : Nat}}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_94_0, w_95_0], ![w_96_0, w_97_0]] 0 j i + ![![w_94_0, w_95_0], ![w_96_0, w_97_0]] 1 j i =
        (lutCoefficient Q0 inverse (fun k ↦ reducePoly Q0 N (readoutTable k)) (sigma i).toNat *
          reducePoly Q0 N ((((q3 : Nat) : Int) : ErrorPoly N) * unitPair sp j)) • g0 +
            reduceMatrix Q0 N 1 6 (e j))
    {{Bh0 : Nat}} (hhelp0 : ∀ i : Fin 4, Rel sm (lutA model w_92_0 Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g0) ![w_91_0 i, w_93_0 i] Bh0)
    {{a : Fin 5 → Nat}} {{G : ExactMatrix Q7 N 1 14}} (hG : gadgetMatrixRuns Backend.backend 32768 14 G)
    {{Ba : Nat}}
    (ha : ∀ k : Fin 5, Rel sm ![![w_101_0 k, w_104_0 k], ![w_102_0 k, w_105_0 k]]
      (fun j ↦ reducePoly Q7 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![w_100_0 k, w_103_0 k] Ba)
    (hpay : ∀ k : Fin 5, w_109_0 k = fun _ _ ↦ reducePoly Q7 N (Xe (n := N) (2 * a k)))
    {{h6 : ExactMatrix Q6 N 1 10}} (hh6 : centeredRebaseRuns g5' h6)
    {{s : Fin 40 → Int}} {{eBm : Fin 40 → Fin 2 → ErrorMatrix N 1 10}} {{BBm : Nat}}
    (hrhsm : ∀ (i : Fin 40) j, reducePoly Q6 N sp • rhsTop model w_92_0 Q6 10 maskRhs0 i j +
        ![w_113_0 i, w_114_0 i] j =
      reducePoly Q6 N (((p5 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q6 N 1 10 (eBm i j))
    (heBm : ∀ i j, CoeffBound (eBm i j) BBm)
    {{s' : Fin 40 → Int}} {{eBm' : Fin 40 → Fin 2 → ErrorMatrix N 1 8}} {{BBm' : Nat}}
    (hrhsm' : ∀ (i : Fin 40) j, reducePoly Q5 N sp • rhsTop model w_92_0 Q5 8 maskRhs1 i j +
        ![w_117_0 i, w_118_0 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q5 N 1 8 (eBm' i j))
    (heBm' : ∀ i j, CoeffBound (eBm' i j) BBm')
    {{sel sel' : Fin 8 → Fin 5}}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {{T : Fin 4 → ErrorPoly N}} {{Δ' : ErrorPoly N}} {{gq : ExactMatrix Q1 N 1 6}} {{inverse' : Nat}}
    (hinverse' : (inverse' : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {{BS1 : Nat}}
    (hswitch1 : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS1) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_87_0, w_88_0], ![w_89_0, w_90_0]] 0 j i + ![![w_87_0, w_88_0], ![w_89_0, w_90_0]] 1 j i =
        (lutCoefficient Q1 inverse' (fun k ↦ reducePoly Q1 N (T k)) (sigma i).toNat *
          reducePoly Q1 N (Δ' * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    Rel sm (fun k j ↦ blockModSwitch [q0, q1, q2, q3] 1 (![![w_332_0, w_332_1], ![w_332_2, w_332_3]] k j))
      (fun j ↦ reducePoly Q1 N (Δ' * (∑ o : Fin 8, Xe (n := N) o *
        (sampled T (readoutExp (a (sel' o) + a (sel o))) + sampled T 1)) * unitPair sp j) • gq)
      ![blockModSwitch [q0, q1, q2, q3] 1 w_332_4, w_333_0]
      (maskBound Ba Bs BBm BBm' BB BB' Bh5 BS Bh0 BS1) := by
  have hoff := offsetChain (model := model) {off_args} (hsm := hsm) (hg6' := hg6') (hg7 := hg7)
    (hone := hone) (hg5' := hg5') (hg6 := hg6) (hrhs := hrhs) (heB := heB) (hg0' := hg0')
    (hh5 := hh5) (hrhs' := hrhs') (heB' := heB') (hhelp5 := hhelp5) (hg1' := hg1') (hg0 := hg0)
    (hinverse := hinverse) (hswitch := hswitch) (hhelp0 := hhelp0)
  have hread := fun o ↦ maskReadout (model := model) (h115 := h115_5) (h116 := h116_5)
    (h119 := h119_5) (h120 := h120_5) (h121 := h121_5) hsm hG hg6' hg7 hg5' hh6 ha hpay hone hrhsm
    heBm hrhsm' heBm' hsel hsel' hg0' hh5 hhelp5 hinverse hswitch o
  subst e_316_0 e_319_0 e_322_0 e_325_0 e_328_0 e_331_0
  have hs := rootSum332 h314_0 h315_0 h317_0 h318_0 h320_0 h321_0 h323_0 h324_0 h326_0 h327_0
    h329_0 h330_0 h332_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  rw [s0, s1, s2, s3, s4]
  exact maskCarrier (model := model) (off := (w_285_0, w_299_0, w_301_0, w_303_0, w_307_0, w_311_0, ()))
    (h312 := h312_5) (h313 := h313_5) hsm hg1' hg0 (ζ := fun o ↦ readoutExp (a (sel' o) + a (sel o)))
    hread hhelp0 hoff hinverse' hswitch1 (by rw [← s5]; exact h333_0)

end RefreshTopProof.Part_{part}
''')
    open(f'{d}/Part_{part}_OnlineChain.lean', 'w').write('\n'.join(out))
    break
