import TopInst
import Part_I_Mask
import TopFinish
import TopPlace

/-!
# The mask carrier of part `I`

Every member's readout `X^(2 ζ_o)` goes through the sampling LUT (table `(0, h, 0, h)`), plus the
binary offset (the same LUT on the readout of the encoding of one); the members are placed at
`X^o`, summed, and moved down to `(R i0)`: an encoding of `Σ_o X^o (T'(ζ_o) + T'(1))` with the sampled
values `T'(z) = (-1)^⌊z/4⌋ T(z mod 4)`.
-/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

theorem place313 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {o : Nat}
    {x : Blocks Q0 6} {y : Blocks Q0 6}
    (h : W_Stage_online_eval.parallel_parallel_generatedRoot_290_313 Backend.backend model params i0 o
      (x.1, x.2.1, x.2.2.1, x.2.2.2.1, x.2.2.2.2.1, x.2.2.2.2.2.1, ()) y) :
    y = (multiplyMonomial x.1 (o : Int), multiplyMonomial x.2.1 (o : Int),
      multiplyMonomial x.2.2.1 (o : Int), multiplyMonomial x.2.2.2.1 (o : Int),
      multiplyMonomial x.2.2.2.2.1 (o : Int), multiplyMonomial x.2.2.2.2.2.1 (o : Int), ()) := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_313 at h
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_313.constraints_0] at h
  rw [h]
  rfl

/-- A sampled value of the LUT with table `T`: `(-1)^⌊z/4⌋ T(z mod 4)`. -/
noncomputable def sampled (T : Fin 4 → ErrorPoly N) (z : Nat) : ErrorPoly N :=
  (((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) * T ⟨z % 4, Nat.mod_lt _ (by norm_num)⟩

/-- The branch multiplier of member `o`, part `b` (the member's sample or the offset) and branch
`i`. -/
noncomputable def maskMul (ζ : Fin 8 → Nat) (x : Fin 8 × Fin 2 × Fin 4) : ErrorPoly N :=
  Xe (n := N) x.1 * substitution (sigma x.2.2).toNat (sigma_odd x.2.2)
    (Xe (n := N) (2 * ![ζ x.1, 1] x.2.1))

theorem polyNorm_Xe_mul_subst {ζ : Fin 8 → Nat} (x : Fin 8 × Fin 2 × Fin 4) :
    polyNorm (maskMul ζ x) ≤ 1 := by
  rw [maskMul, substitution_root_pow]
  exact polyNorm_Xe_mul _ _

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem maskCarrier {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {key : ByteArray}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix (R i0) N 1 6} {hl0 hl1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {r0 r1 r2 r3 r4 r5 : Fin 8 → ExactMatrix Q0 N 1 6} {off : Blocks Q0 6}
    {s0 s1 s2 s3 s4 s5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h312 : ∀ o : Fin 8, W_Stage_online_eval.parallel_parallel_generatedRoot_290_312 Backend.backend model
      params i0 o (fam00, fam01, fam10, fam11, hl0, key, hl1, r4 o, r0 o, r2 o, r5 o, r1 o, r3 o,
        off.1, off.2.1, off.2.2.1, off.2.2.2.1, off.2.2.2.2.1, off.2.2.2.2.2.1, ())
        (s0 o, s1 o, s2 o, s3 o, s4 o, s5 o, ()))
    {t0 t1 t2 t3 t4 t5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h313 : ∀ o : Fin 8, W_Stage_online_eval.parallel_parallel_generatedRoot_290_313 Backend.backend model
      params i0 o (s0 o, s1 o, s2 o, s3 o, s4 o, s5 o, ()) (t0 o, t1 o, t2 o, t3 o, t4 o, t5 o, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {ζ : Fin 8 → Nat} {Br : Nat}
    (hread : ∀ o : Fin 8, Rel sm ![![r0 o, r1 o], ![r2 o, r3 o]]
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) * Xe (n := N) (2 * ζ o) * unitPair sp j) • g)
      ![r4 o, r5 o] Br)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model key Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![hl0 i, hl1 i] Bh)
    {Boff : Nat}
    (hoff : Rel sm ![![off.1, off.2.1], ![off.2.2.1, off.2.2.2.1]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 (qd i0)
        (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * 1))) •
          (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i)))
      ![off.2.2.2.2.1, off.2.2.2.2.2.1] Boff)
    {T : Fin 4 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix (R i0) N 1 6} {inverse : Nat}
    (hinverse : (inverse : ExactPoly (R i0) N) * ((4 : Nat) : ExactPoly (R i0) N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient (R i0) inverse (fun k ↦ reducePoly (R i0) N (T k)) (sigma i).toNat *
          reducePoly (R i0) N (Δ' * unitPair sp j)) • gq + reduceMatrix (R i0) N 1 6 (e j))
    {basis : List Nat} {c : ExactMatrix (R i0) N 1 6}
    (h333 : blockModSwitchRuns basis 1 (∑ o, t5 o) c) :
    Rel sm (fun k j ↦ blockModSwitch basis 1 (![![∑ o, t0 o, ∑ o, t1 o], ![∑ o, t2 o, ∑ o, t3 o]] k j))
      (fun j ↦ reducePoly (R i0) N (Δ' * (∑ o : Fin 8, Xe (n := N) o * (sampled T (ζ o) + sampled T 1)) *
        unitPair sp j) • gq)
      ![blockModSwitch basis 1 (∑ o, t4 o), c]
      ((8 * ((3 * (lutBranchBound312 i0) Br Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * Br * 16384 +
        (N * 1 * ((qd i0) / 2) + (qd i0) / 2))) + Boff) + (N + 2) * ((qd i0) / 2)) / (qd i0) + 64 * (N * 1 * BS)) := by
  -- Each member: the sampling LUT plus the offset, placed at `X^o`.
  have hmem : ∀ o : Fin 8, Rel sm (fun a j ↦ multiplyMonomial (![![s0 o, s1 o], ![s2 o, s3 o]] a j) (o : Int))
      (fun j ↦ reducePoly Q0 N (Xe (n := N) o) •
        ((∑ i : Fin 4, scaleUpMatrix Q0 (qd i0)
          (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ o))) •
            (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
              ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i))) +
          ∑ i : Fin 4, scaleUpMatrix Q0 (qd i0)
            (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * 1))) •
              (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
                ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i))))
      (fun j ↦ multiplyMonomial (![s4 o, s5 o] j) (o : Int))
      ((3 * (lutBranchBound312 i0) Br Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * Br * 16384 +
        (N * 1 * ((qd i0) / 2) + (qd i0) / 2))) + Boff) := by
    intro o
    have h := lut312 (off := off) (h312 o) hsm hg' hg (hread o) hhelp hoff
    exact rel_monomial h (k := o) (by have := o.isLt; simp only [N]; omega)
  -- The placed members are the outputs of `313`.
  have ht : ∀ o : Fin 8, (t0 o, t1 o, t2 o, t3 o, t4 o, t5 o, ()) =
      (multiplyMonomial (s0 o) (o : Int), multiplyMonomial (s1 o) (o : Int),
        multiplyMonomial (s2 o) (o : Int), multiplyMonomial (s3 o) (o : Int),
        multiplyMonomial (s4 o) (o : Int), multiplyMonomial (s5 o) (o : Int), ()) :=
    fun o ↦ place313 (x := (s0 o, s1 o, s2 o, s3 o, s4 o, s5 o, ())) (h313 o)
  have hmem' : ∀ o ∈ (Finset.univ : Finset (Fin 8)), Rel sm ![![t0 o, t1 o], ![t2 o, t3 o]]
      (fun j ↦ reducePoly Q0 N (Xe (n := N) o) •
        ((∑ i : Fin 4, scaleUpMatrix Q0 (qd i0)
          (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ o))) •
            (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
              ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i))) +
          ∑ i : Fin 4, scaleUpMatrix Q0 (qd i0)
            (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * 1))) •
              (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
                ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i))))
      ![t4 o, t5 o]
      ((3 * (lutBranchBound312 i0) Br Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * Br * 16384 +
        (N * 1 * ((qd i0) / 2) + (qd i0) / 2))) + Boff) := by
    intro o _
    have h := hmem o
    simp only [Prod.mk.injEq] at ht
    obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := ht o
    rw [e0, e1, e2, e3, e4, e5]
    convert h using 1
    · funext a j
      fin_cases a <;> fin_cases j <;> rfl
    · funext j
      fin_cases j <;> rfl
  have hsum := Rel.sum Finset.univ hmem'
  rw [blocks_sum, pair_sum] at hsum
  -- The switching keys and the interpolation of every branch.
  choose eS heS hS using hswitch
  have hsum' := hsum.target_eq (t' := fun j ↦ ∑ x : Fin 8 × Fin 2 × Fin 4, scaleUpMatrix Q0 (qd i0)
      (reducePoly (R i0) N (maskMul ζ x) •
        (reducePoly (R i0) N (substitution (sigma x.2.2).toNat (sigma_odd x.2.2) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j x.2.2 +
            ![![fam00, fam01], ![fam10, fam11]] 1 j x.2.2))) (by
    funext j
    simp only [Finset.sum_apply, Fintype.sum_prod_type, Fin.sum_univ_two, maskMul]
    refine Finset.sum_congr rfl fun o _ ↦ ?_
    rw [smul_add, Finset.smul_sum, Finset.smul_sum]
    simp only [smul_scaleUpMatrix (q0_eqR i0 hi0) (by inst_num), smul_smul, ← map_mul]
    rfl)
  have hfin := lut_finish_sum (ι := Fin 8 × Fin 2 × Fin 4) (q0_eqR i0 hi0) (by inst_num) (by inst_num)
    (by inst_num) hsm (maskMul ζ) (fun x ↦ substitution (sigma x.2.2).toNat (sigma_odd x.2.2) sp)
    (fun x ↦ polyNorm_Xe_mul_subst x)
    (bsw := fun x k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j x.2.2) hsum' (g' := gq)
    (C := fun x ↦ lutCoefficient (R i0) inverse (fun k ↦ reducePoly (R i0) N (T k)) (sigma x.2.2).toNat)
    (Δ' := Δ') (fun x j ↦ hS x.2.2 j) (fun x j ↦ heS x.2.2 j)
    (ν := ∑ o : Fin 8, Xe (n := N) o * (sampled T (ζ o) + sampled T 1)) (by
      simp only [Fintype.sum_prod_type, Fin.sum_univ_two, maskMul, map_sum, map_mul, map_add]
      refine Finset.sum_congr rfl fun o _ ↦ ?_
      have hw := fun z ↦ lut_walsh_value (q := (R i0)) T hinverse z
      simp only [mul_assoc, ← Finset.mul_sum, ← mul_add]
      congr 1
      rw [hw, hw]
      rfl)
    (K' := fun k j ↦ blockModSwitch basis 1 (![![∑ o, t0 o, ∑ o, t1 o], ![∑ o, t2 o, ∑ o, t3 o]] k j))
    (R' := ![blockModSwitch basis 1 (∑ o, t4 o), c])
    (fun k j ↦ blockModSwitchRuns_fn h333 _)
    (fun j ↦ by fin_cases j; exacts [blockModSwitchRuns_fn h333 _, h333])
  refine hfin.mono (le_of_eq ?_)
  simp [Finset.sum_const, Finset.card_univ]

end RefreshTopProof.Part_I
