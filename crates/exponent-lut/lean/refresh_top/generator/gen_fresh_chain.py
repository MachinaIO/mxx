"""Write `Part_<c_j>_FreshChain.lean`: chain theorems of the fresh noise of the online root.

- `freshSamples`: every member of the fresh PRF read out (`freshReadout`) and sampled (`lut393`).
- `freshCol<k>`: column `k`: its members placed (`freshColumn<k>`), summed (`rootSum`), moved down
  to `Q₁` and multiplied by the digits of `Q_q`.

Usage: `python3 gen_fresh_chain.py <parameter directory>`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from chain import chain_binders, conclusion_bound, subst
from gen_part import parts_of
from chains import chain_of
from segment import abbreviations

d = sys.argv[1]
ab = abbreviations(open(f'{d}/TopSetup.lean').read())


def indent(binders):
    return '\n'.join('    ' + b for b in binders)


COMMON = '''    {{sm sp : ErrorPoly N}} (hsm : polyNorm sm ≤ 1)'''

for part in parts_of(d):
    Qq = ab.get(str(chain_of(part)['Qq']), chain_of(part)['Qq'])
    ns = f'Stage_online_eval_{part}'
    path = f'{d}/generated/{ns}.lean'
    readout = open(f'{d}/Part_{part}_Readout.lean').read()
    b393 = subst(conclusion_bound(readout, 'lut393'),
                 {'B': '(freshReadoutBound Ba Bs BB BB\' Bh5 BS)', 'Bh': 'Bh0'})
    sbinders = chain_binders(path, ns, [], ab, ['h386_5', 'h387_5', 'h390_5', 'h391_5', 'h392_5',
                                               'h393_5'])
    out = [f'''import Part_{part}_Fresh
import Part_{part}_FreshCarrier
import Part_{part}_RootSums
import TopAction
import TopChain

/-! Chain theorems of the fresh noise of part `{part}` (written by `generator/gen_fresh_chain.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof Mxx.ExponentLut.Refresh

/-- The bound of a sampled member of the fresh PRF. -/
abbrev freshSampleBound (Ba Bs BB BB' Bh5 BS Bh0 : Nat) : Nat :=
  {b393}

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Every member of the fresh PRF, read out and sampled into its LUT branches. -/
theorem freshSamples {{model : HashModel}} {{params : {ns}.Params}}
{indent(sbinders)}
    {{sm sp : ErrorPoly N}} (hsm : polyNorm sm ≤ 1)
    {{g6' : ExactMatrix Q6 N 1 12}} {{g7 : ExactMatrix Q7 N 1 12}}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 12 g6') (hg7 : centeredRebaseRuns g6' g7)
    {{Bs : Nat}}
    (hone : Rel sm ![![w_377_0, w_381_0], ![w_378_0, w_382_0]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * 1 * unitPair sp j) • g7) ![w_379_0, w_383_0] Bs)
    {{g5' : ExactMatrix Q5 N 1 10}} {{h6 : ExactMatrix Q6 N 1 10}}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 10 g5') (hh6 : centeredRebaseRuns g5' h6)
    {{g0' : ExactMatrix Q0 N 1 8}} {{h5 : ExactMatrix Q5 N 1 8}}
    (hg0' : gadgetMatrixRuns Backend.backend 32768 8 g0') (hh5 : centeredRebaseRuns g0' h5)
    {{a : Fin 5 → Nat}} {{G : ExactMatrix Q7 N 1 14}} (hG : gadgetMatrixRuns Backend.backend 32768 14 G)
    {{Ba : Nat}}
    (ha : ∀ k : Fin 5, Rel sm ![![w_372_0 k, w_375_0 k], ![w_373_0 k, w_376_0 k]]
      (fun j ↦ reducePoly Q7 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![w_371_0 k, w_374_0 k] Ba)
    (hpay : ∀ k : Fin 5, w_380_0 k = fun _ _ ↦ reducePoly Q7 N (Xe (n := N) (2 * a k)))
    {{s : Fin 240 → Int}} {{eB : Fin 240 → Fin 2 → ErrorMatrix N 1 10}} {{BB : Nat}}
    (hrhs : ∀ (i : Fin 240) j, reducePoly Q6 N sp • rhsTop model w_363_0 Q6 10 freshRhs0 i j +
        ![w_384_0 i, w_385_0 i] j =
      reducePoly Q6 N (((p5 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q6 N 1 10 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {{s' : Fin 240 → Int}} {{eB' : Fin 240 → Fin 2 → ErrorMatrix N 1 8}} {{BB' : Nat}}
    (hrhs' : ∀ (i : Fin 240) j, reducePoly Q5 N sp • rhsTop model w_363_0 Q5 8 freshRhs1 i j +
        ![w_388_0 i, w_389_0 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q5 N 1 8 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {{sel sel' : Fin 48 → Fin 5}}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {{Bh5 : Nat}} (hhelp5 : ∀ i : Fin 4, Rel sm (lutA model w_363_0 Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![w_369_0 i, w_370_0 i] Bh5)
    {{g1' : ExactMatrix Q1 N 1 6}} {{g0 : ExactMatrix Q0 N 1 6}}
    (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1') (hg0 : centeredRebaseRuns g1' g0)
    {{inverse : Nat}} (hinverse : (inverse : ExactPoly Q0 N) * ((4 : Nat) : ExactPoly Q0 N) = 1)
    {{BS : Nat}}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_365_0, w_366_0], ![w_367_0, w_368_0]] 0 j i + ![![w_365_0, w_366_0], ![w_367_0, w_368_0]] 1 j i =
        (lutCoefficient Q0 inverse (fun k ↦ reducePoly Q0 N (readoutTable k)) (sigma i).toNat *
          reducePoly Q0 N ((((q3 : Nat) : Int) : ErrorPoly N) * unitPair sp j)) • g0 +
            reduceMatrix Q0 N 1 6 (e j))
    {{Bh0 : Nat}} (hhelp0 : ∀ i : Fin 4, Rel sm (lutA model w_363_0 Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g0) ![w_362_0 i, w_364_0 i] Bh0)
    (o : Fin 48) :
    Rel sm ![![w_393_0 o, w_393_1 o], ![w_393_2 o, w_393_3 o]]
      (fun j ↦ (∑ i : Fin 4, scaleUpMatrix Q0 q3
        (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i)
          (Xe (2 * readoutExp (a (sel' o) + a (sel o))))) •
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
              ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i))))
      ![w_393_4 o, w_393_5 o] (freshSampleBound Ba Bs BB BB' Bh5 BS Bh0) := by
  have hread := freshReadout (model := model) (h386 := h386_5) (h387 := h387_5) (h390 := h390_5)
    (h391 := h391_5) (h392 := h392_5) (hsm := hsm) (hG := hG) (hg6' := hg6') (hg7 := hg7)
    (hg5' := hg5') (hh6 := hh6) (ha := ha) (hpay := hpay) (hone := hone) (hrhs := hrhs) (heB := heB)
    (hrhs' := hrhs') (heB' := heB') (hsel := hsel) (hsel' := hsel') (hg0' := hg0') (hh5 := hh5)
    (hhelp := hhelp5) (hinverse := hinverse) (hswitch := hswitch) (o := o)
  exact lut393 (h := h393_5 o) hsm hg1' hg0 hread hhelp0
''']
    carrier = open(f'{d}/Part_{part}_FreshCarrier.lean').read()
    sums = open(f'{d}/Part_{part}_RootSums.lean').read()
    places = [394, 419, 441, 463, 485, 507]
    for k, P in enumerate(places):
        S, M = P + 19, P + 20
        prod = 418 if k == 0 else M + 1
        fams = [f'h{P + 1 + 3 * i}_0' for i in range(6)] + [f'h{P + 2 + 3 * i}_0' for i in range(6)]
        fams = [x for i in range(6) for x in (f'h{P + 1 + 3 * i}_0', f'h{P + 2 + 3 * i}_0')]
        inits = ' '.join(f'e_{P + 3 + 3 * i}_0' for i in range(6))
        atoms = [f'h{P}_5'] + fams + [f'h{S}_5', f'h{M}_0', 'h416_0', 'h417_0']
        binders = chain_binders(path, ns, [], ab, atoms, [f'w_{prod}_0', 'w_358_0', 'w_359_0', 'w_360_0', 'w_361_0'])
        idx = '⟨i.val, by omega⟩' if k == 0 else f'⟨i.val + {8 * k}, by omega⟩'
        out.append(f'''
set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Column `{k}` of the fresh noise times the digits of `Q_q`. -/
theorem freshCol{k} {{model : HashModel}} {{params : {ns}.Params}}
{indent(binders)}
    {{sm sp : ErrorPoly N}} (hsm : polyNorm sm ≤ 1) {{ζ : Fin 48 → Nat}} {{Bs : Nat}}
    (hsamp : ∀ o : Fin 48, Rel sm ![![w_393_0 o, w_393_1 o], ![w_393_2 o, w_393_3 o]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 q3
        (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ o))) •
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
              ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i)))
      ![w_393_4 o, w_393_5 o] Bs)
    {{T : Fin 4 → ErrorPoly N}} {{Δ' : ErrorPoly N}} {{gq : ExactMatrix Q1 N 1 6}}
    (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {{inverse : Nat}}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {{BS : Nat}}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i =
        (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (T k)) (sigma i).toNat *
          reducePoly Q1 N (Δ' * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    ∃ e : ErrorMatrix N 1 1,
      CoeffBound e (6 * N * ((8 * Bs + (N + 2) * (q3 / 2)) / q3 + 32 * (N * 1 * BS)) * 16384) ∧
      w_{prod}_0 = maskPart (reducePoly Q1 N sm) (fun k ↦
          blockModSwitch [q0, q1, q2, q3] 1 (![![w_{S}_0, w_{S}_1], ![w_{S}_2, w_{S}_3]] k 1) * w_417_0) -
        reducePoly Q1 N (Δ' * (∑ i : Fin 8, Xe (n := N) i * sampled T (ζ {idx})) *
          unitPair sp 1) • (matrixPolynomial [{Qq}] : ExactMatrix Q1 N 1 1) +
        reduceMatrix Q1 N 1 1 e := by
  subst {inits}
  have hs := rootSum{S} {' '.join(fams)} h{S}_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  have hc := freshColumn{k} (model := model) (hplace := h{P}_5) hsm (ζ := ζ) hsamp hinverse hswitch
    (by rw [← s5]; exact h{M}_0)
  rw [← s0, ← s1, ← s2, ← s3] at hc
  obtain ⟨e, he, hrow⟩ := Rel.row hc 1
  simp only [Matrix.cons_val_one, Matrix.head_cons] at hrow
  rw [e_{prod}_0]
  exact row_mul_qqQ hgq he hrow (by rw [← e_415_0]; exact h416_0) h417_0
''')
    out.append(f'\nend RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_FreshChain.lean', 'w').write('\n'.join(out))
    break
