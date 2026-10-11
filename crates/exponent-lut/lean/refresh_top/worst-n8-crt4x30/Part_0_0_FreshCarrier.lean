import Part_0_0_MaskCarrier
import Part_0_0_Fresh

/-!
# The fresh-noise columns of part `0_0`

Column `c` of the fresh noise samples members `8c, …, 8c + 7`, places member `8c + i` at `X^i`,
sums and moves the sum down to `Q1`: an encoding of `Σ_i X^i T'(ζ_(8c+i))`.
-/

namespace RefreshTopProof.Part_0_0

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

/-- The branch multiplier of member `i` of a column and branch `k`. -/
noncomputable def freshMul (ζ : Fin 8 → Nat) (x : Fin 8 × Fin 4) : ErrorPoly N :=
  Xe (n := N) x.1 * substitution (sigma x.2).toNat (sigma_odd x.2) (Xe (n := N) (2 * ζ x.1))

theorem polyNorm_freshMul {ζ : Fin 8 → Nat} (x : Fin 8 × Fin 4) : polyNorm (freshMul ζ x) ≤ 1 := by
  rw [freshMul, substitution_root_pow]
  exact polyNorm_Xe_mul _ _

theorem place394 {model : HashModel} {params : Stage_online_eval_0_0.Params} {o : Nat}
    {x : Blocks Q0 6} {y : Blocks Q0 6}
    (h : Stage_online_eval_0_0.parallel_generatedRoot_394 Backend.backend model params o
      (x.1, x.2.1, x.2.2.1, x.2.2.2.1, x.2.2.2.2.1, x.2.2.2.2.2.1, ()) y) :
    y = (multiplyMonomial x.1 (o : Int), multiplyMonomial x.2.1 (o : Int),
      multiplyMonomial x.2.2.1 (o : Int), multiplyMonomial x.2.2.2.1 (o : Int),
      multiplyMonomial x.2.2.2.2.1 (o : Int), multiplyMonomial x.2.2.2.2.2.1 (o : Int), ()) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_394 at h
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_394.constraints_0] at h
  rw [h]
  rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Column `0` of the fresh noise. -/
theorem freshColumn0 {model : HashModel} {params : Stage_online_eval_0_0.Params}
    {s0 s1 s2 s3 s4 s5 : Fin 48 → ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {t0 t1 t2 t3 t4 t5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (hplace : ∀ i : Fin 8, Stage_online_eval_0_0.parallel_generatedRoot_394 Backend.backend model params i
      (s0 ⟨i.val, by omega⟩, s1 ⟨i.val, by omega⟩, s2 ⟨i.val, by omega⟩, s3 ⟨i.val, by omega⟩, s4 ⟨i.val, by omega⟩, s5 ⟨i.val, by omega⟩, ())
      (t0 i, t1 i, t2 i, t3 i, t4 i, t5 i, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    {ζ : Fin 48 → Nat} {Bs : Nat}
    (hsamp : ∀ o : Fin 48, Rel sm ![![s0 o, s1 o], ![s2 o, s3 o]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 q3
        (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ o))) •
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i)))
      ![s4 o, s5 o] Bs)
    {T : Fin 4 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix Q1 N 1 6} {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (T k)) (sigma i).toNat *
          reducePoly Q1 N (Δ' * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j))
    {basis : List Nat} {c : ExactMatrix Q1 N 1 6}
    (hdown : blockModSwitchRuns basis 1 (∑ i, t5 i) c) :
    Rel sm (fun k j ↦ blockModSwitch basis 1 (![![∑ i, t0 i, ∑ i, t1 i], ![∑ i, t2 i, ∑ i, t3 i]] k j))
      (fun j ↦ reducePoly Q1 N (Δ' * (∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val, by omega⟩)) *
        unitPair sp j) • gq)
      ![blockModSwitch basis 1 (∑ i, t4 i), c]
      ((8 * Bs + (N + 2) * (q3 / 2)) / q3 + 32 * (N * 1 * BS)) := by
  have ht : ∀ o : Fin 8, (t0 o, t1 o, t2 o, t3 o, t4 o, t5 o, ()) =
      (multiplyMonomial (s0 ⟨o.val, by omega⟩) (o : Int), multiplyMonomial (s1 ⟨o.val, by omega⟩) (o : Int),
        multiplyMonomial (s2 ⟨o.val, by omega⟩) (o : Int), multiplyMonomial (s3 ⟨o.val, by omega⟩) (o : Int),
        multiplyMonomial (s4 ⟨o.val, by omega⟩) (o : Int), multiplyMonomial (s5 ⟨o.val, by omega⟩) (o : Int), ()) :=
    fun o ↦ place394 (x := (s0 ⟨o.val, by omega⟩, s1 ⟨o.val, by omega⟩, s2 ⟨o.val, by omega⟩, s3 ⟨o.val, by omega⟩, s4 ⟨o.val, by omega⟩, s5 ⟨o.val, by omega⟩, ())) (hplace o)
  have hmem : ∀ o ∈ (Finset.univ : Finset (Fin 8)), Rel sm ![![t0 o, t1 o], ![t2 o, t3 o]]
      (fun j ↦ reducePoly Q0 N (Xe (n := N) o) •
        ∑ i : Fin 4, scaleUpMatrix Q0 q3
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ ⟨o.val, by omega⟩))) •
            (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
              ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i)))
      ![t4 o, t5 o] Bs := by
    intro o _
    have h := rel_monomial (hsamp ⟨o.val, by omega⟩) (k := o) (by have := o.isLt; simp only [N]; omega)
    simp only [Prod.mk.injEq] at ht
    obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := ht o
    rw [e0, e1, e2, e3, e4, e5]
    convert h using 1
    · funext a j
      fin_cases a <;> fin_cases j <;> rfl
    · funext j
      fin_cases j <;> rfl
  have hsum := Rel.sum Finset.univ hmem
  rw [blocks_sum, pair_sum] at hsum
  choose eS heS hS using hswitch
  have hsum' := hsum.target_eq (t' := fun j ↦ ∑ x : Fin 8 × Fin 4, scaleUpMatrix Q0 q3
      (reducePoly Q1 N (freshMul (fun i ↦ ζ ⟨i.val, by omega⟩) x) •
        (reducePoly Q1 N (substitution (sigma x.2).toNat (sigma_odd x.2) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j x.2 +
            ![![fam00, fam01], ![fam10, fam11]] 1 j x.2))) (by
    funext j
    simp only [Finset.sum_apply, Fintype.sum_prod_type, freshMul]
    refine Finset.sum_congr rfl fun o _ ↦ ?_
    rw [Finset.smul_sum]
    simp only [smul_scaleUpMatrix q0_eq (by norm_num), smul_smul, ← map_mul])
  have hfin := lut_finish_sum (ι := Fin 8 × Fin 4) q0_eq (by norm_num) (by norm_num)
    (by norm_num) hsm (freshMul (fun i ↦ ζ ⟨i.val, by omega⟩)) (fun x ↦ substitution (sigma x.2).toNat (sigma_odd x.2) sp)
    (fun x ↦ polyNorm_freshMul x)
    (bsw := fun x k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j x.2) hsum' (g' := gq)
    (C := fun x ↦ lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (T k)) (sigma x.2).toNat)
    (Δ' := Δ') (fun x j ↦ hS x.2 j) (fun x j ↦ heS x.2 j)
    (ν := ∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val, by omega⟩)) (by
      simp only [Fintype.sum_prod_type, freshMul, map_sum, map_mul]
      refine Finset.sum_congr rfl fun o _ ↦ ?_
      simp only [mul_assoc, ← Finset.mul_sum]
      congr 1
      exact lut_walsh_value (q := Q1) T hinverse _)
    (K' := fun k j ↦ blockModSwitch basis 1 (![![∑ i, t0 i, ∑ i, t1 i], ![∑ i, t2 i, ∑ i, t3 i]] k j))
    (R' := ![blockModSwitch basis 1 (∑ i, t4 i), c])
    (fun k j ↦ blockModSwitchRuns_fn hdown _)
    (fun j ↦ by fin_cases j; exacts [blockModSwitchRuns_fn hdown _, hdown])
  refine hfin.mono (le_of_eq ?_)
  simp [Finset.sum_const, Finset.card_univ]

theorem place419 {model : HashModel} {params : Stage_online_eval_0_0.Params} {o : Nat}
    {x : Blocks Q0 6} {y : Blocks Q0 6}
    (h : Stage_online_eval_0_0.parallel_generatedRoot_419 Backend.backend model params o
      (x.1, x.2.1, x.2.2.1, x.2.2.2.1, x.2.2.2.2.1, x.2.2.2.2.2.1, ()) y) :
    y = (multiplyMonomial x.1 (o : Int), multiplyMonomial x.2.1 (o : Int),
      multiplyMonomial x.2.2.1 (o : Int), multiplyMonomial x.2.2.2.1 (o : Int),
      multiplyMonomial x.2.2.2.2.1 (o : Int), multiplyMonomial x.2.2.2.2.2.1 (o : Int), ()) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_419 at h
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_419.constraints_0] at h
  rw [h]
  rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Column `1` of the fresh noise. -/
theorem freshColumn1 {model : HashModel} {params : Stage_online_eval_0_0.Params}
    {s0 s1 s2 s3 s4 s5 : Fin 48 → ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {t0 t1 t2 t3 t4 t5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (hplace : ∀ i : Fin 8, Stage_online_eval_0_0.parallel_generatedRoot_419 Backend.backend model params i
      (s0 ⟨i.val + 8, by omega⟩, s1 ⟨i.val + 8, by omega⟩, s2 ⟨i.val + 8, by omega⟩, s3 ⟨i.val + 8, by omega⟩, s4 ⟨i.val + 8, by omega⟩, s5 ⟨i.val + 8, by omega⟩, ())
      (t0 i, t1 i, t2 i, t3 i, t4 i, t5 i, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    {ζ : Fin 48 → Nat} {Bs : Nat}
    (hsamp : ∀ o : Fin 48, Rel sm ![![s0 o, s1 o], ![s2 o, s3 o]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 q3
        (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ o))) •
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i)))
      ![s4 o, s5 o] Bs)
    {T : Fin 4 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix Q1 N 1 6} {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (T k)) (sigma i).toNat *
          reducePoly Q1 N (Δ' * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j))
    {basis : List Nat} {c : ExactMatrix Q1 N 1 6}
    (hdown : blockModSwitchRuns basis 1 (∑ i, t5 i) c) :
    Rel sm (fun k j ↦ blockModSwitch basis 1 (![![∑ i, t0 i, ∑ i, t1 i], ![∑ i, t2 i, ∑ i, t3 i]] k j))
      (fun j ↦ reducePoly Q1 N (Δ' * (∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val + 8, by omega⟩)) *
        unitPair sp j) • gq)
      ![blockModSwitch basis 1 (∑ i, t4 i), c]
      ((8 * Bs + (N + 2) * (q3 / 2)) / q3 + 32 * (N * 1 * BS)) := by
  have ht : ∀ o : Fin 8, (t0 o, t1 o, t2 o, t3 o, t4 o, t5 o, ()) =
      (multiplyMonomial (s0 ⟨o.val + 8, by omega⟩) (o : Int), multiplyMonomial (s1 ⟨o.val + 8, by omega⟩) (o : Int),
        multiplyMonomial (s2 ⟨o.val + 8, by omega⟩) (o : Int), multiplyMonomial (s3 ⟨o.val + 8, by omega⟩) (o : Int),
        multiplyMonomial (s4 ⟨o.val + 8, by omega⟩) (o : Int), multiplyMonomial (s5 ⟨o.val + 8, by omega⟩) (o : Int), ()) :=
    fun o ↦ place419 (x := (s0 ⟨o.val + 8, by omega⟩, s1 ⟨o.val + 8, by omega⟩, s2 ⟨o.val + 8, by omega⟩, s3 ⟨o.val + 8, by omega⟩, s4 ⟨o.val + 8, by omega⟩, s5 ⟨o.val + 8, by omega⟩, ())) (hplace o)
  have hmem : ∀ o ∈ (Finset.univ : Finset (Fin 8)), Rel sm ![![t0 o, t1 o], ![t2 o, t3 o]]
      (fun j ↦ reducePoly Q0 N (Xe (n := N) o) •
        ∑ i : Fin 4, scaleUpMatrix Q0 q3
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ ⟨o.val + 8, by omega⟩))) •
            (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
              ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i)))
      ![t4 o, t5 o] Bs := by
    intro o _
    have h := rel_monomial (hsamp ⟨o.val + 8, by omega⟩) (k := o) (by have := o.isLt; simp only [N]; omega)
    simp only [Prod.mk.injEq] at ht
    obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := ht o
    rw [e0, e1, e2, e3, e4, e5]
    convert h using 1
    · funext a j
      fin_cases a <;> fin_cases j <;> rfl
    · funext j
      fin_cases j <;> rfl
  have hsum := Rel.sum Finset.univ hmem
  rw [blocks_sum, pair_sum] at hsum
  choose eS heS hS using hswitch
  have hsum' := hsum.target_eq (t' := fun j ↦ ∑ x : Fin 8 × Fin 4, scaleUpMatrix Q0 q3
      (reducePoly Q1 N (freshMul (fun i ↦ ζ ⟨i.val + 8, by omega⟩) x) •
        (reducePoly Q1 N (substitution (sigma x.2).toNat (sigma_odd x.2) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j x.2 +
            ![![fam00, fam01], ![fam10, fam11]] 1 j x.2))) (by
    funext j
    simp only [Finset.sum_apply, Fintype.sum_prod_type, freshMul]
    refine Finset.sum_congr rfl fun o _ ↦ ?_
    rw [Finset.smul_sum]
    simp only [smul_scaleUpMatrix q0_eq (by norm_num), smul_smul, ← map_mul])
  have hfin := lut_finish_sum (ι := Fin 8 × Fin 4) q0_eq (by norm_num) (by norm_num)
    (by norm_num) hsm (freshMul (fun i ↦ ζ ⟨i.val + 8, by omega⟩)) (fun x ↦ substitution (sigma x.2).toNat (sigma_odd x.2) sp)
    (fun x ↦ polyNorm_freshMul x)
    (bsw := fun x k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j x.2) hsum' (g' := gq)
    (C := fun x ↦ lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (T k)) (sigma x.2).toNat)
    (Δ' := Δ') (fun x j ↦ hS x.2 j) (fun x j ↦ heS x.2 j)
    (ν := ∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val + 8, by omega⟩)) (by
      simp only [Fintype.sum_prod_type, freshMul, map_sum, map_mul]
      refine Finset.sum_congr rfl fun o _ ↦ ?_
      simp only [mul_assoc, ← Finset.mul_sum]
      congr 1
      exact lut_walsh_value (q := Q1) T hinverse _)
    (K' := fun k j ↦ blockModSwitch basis 1 (![![∑ i, t0 i, ∑ i, t1 i], ![∑ i, t2 i, ∑ i, t3 i]] k j))
    (R' := ![blockModSwitch basis 1 (∑ i, t4 i), c])
    (fun k j ↦ blockModSwitchRuns_fn hdown _)
    (fun j ↦ by fin_cases j; exacts [blockModSwitchRuns_fn hdown _, hdown])
  refine hfin.mono (le_of_eq ?_)
  simp [Finset.sum_const, Finset.card_univ]

theorem place441 {model : HashModel} {params : Stage_online_eval_0_0.Params} {o : Nat}
    {x : Blocks Q0 6} {y : Blocks Q0 6}
    (h : Stage_online_eval_0_0.parallel_generatedRoot_441 Backend.backend model params o
      (x.1, x.2.1, x.2.2.1, x.2.2.2.1, x.2.2.2.2.1, x.2.2.2.2.2.1, ()) y) :
    y = (multiplyMonomial x.1 (o : Int), multiplyMonomial x.2.1 (o : Int),
      multiplyMonomial x.2.2.1 (o : Int), multiplyMonomial x.2.2.2.1 (o : Int),
      multiplyMonomial x.2.2.2.2.1 (o : Int), multiplyMonomial x.2.2.2.2.2.1 (o : Int), ()) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_441 at h
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_441.constraints_0] at h
  rw [h]
  rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Column `2` of the fresh noise. -/
theorem freshColumn2 {model : HashModel} {params : Stage_online_eval_0_0.Params}
    {s0 s1 s2 s3 s4 s5 : Fin 48 → ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {t0 t1 t2 t3 t4 t5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (hplace : ∀ i : Fin 8, Stage_online_eval_0_0.parallel_generatedRoot_441 Backend.backend model params i
      (s0 ⟨i.val + 16, by omega⟩, s1 ⟨i.val + 16, by omega⟩, s2 ⟨i.val + 16, by omega⟩, s3 ⟨i.val + 16, by omega⟩, s4 ⟨i.val + 16, by omega⟩, s5 ⟨i.val + 16, by omega⟩, ())
      (t0 i, t1 i, t2 i, t3 i, t4 i, t5 i, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    {ζ : Fin 48 → Nat} {Bs : Nat}
    (hsamp : ∀ o : Fin 48, Rel sm ![![s0 o, s1 o], ![s2 o, s3 o]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 q3
        (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ o))) •
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i)))
      ![s4 o, s5 o] Bs)
    {T : Fin 4 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix Q1 N 1 6} {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (T k)) (sigma i).toNat *
          reducePoly Q1 N (Δ' * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j))
    {basis : List Nat} {c : ExactMatrix Q1 N 1 6}
    (hdown : blockModSwitchRuns basis 1 (∑ i, t5 i) c) :
    Rel sm (fun k j ↦ blockModSwitch basis 1 (![![∑ i, t0 i, ∑ i, t1 i], ![∑ i, t2 i, ∑ i, t3 i]] k j))
      (fun j ↦ reducePoly Q1 N (Δ' * (∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val + 16, by omega⟩)) *
        unitPair sp j) • gq)
      ![blockModSwitch basis 1 (∑ i, t4 i), c]
      ((8 * Bs + (N + 2) * (q3 / 2)) / q3 + 32 * (N * 1 * BS)) := by
  have ht : ∀ o : Fin 8, (t0 o, t1 o, t2 o, t3 o, t4 o, t5 o, ()) =
      (multiplyMonomial (s0 ⟨o.val + 16, by omega⟩) (o : Int), multiplyMonomial (s1 ⟨o.val + 16, by omega⟩) (o : Int),
        multiplyMonomial (s2 ⟨o.val + 16, by omega⟩) (o : Int), multiplyMonomial (s3 ⟨o.val + 16, by omega⟩) (o : Int),
        multiplyMonomial (s4 ⟨o.val + 16, by omega⟩) (o : Int), multiplyMonomial (s5 ⟨o.val + 16, by omega⟩) (o : Int), ()) :=
    fun o ↦ place441 (x := (s0 ⟨o.val + 16, by omega⟩, s1 ⟨o.val + 16, by omega⟩, s2 ⟨o.val + 16, by omega⟩, s3 ⟨o.val + 16, by omega⟩, s4 ⟨o.val + 16, by omega⟩, s5 ⟨o.val + 16, by omega⟩, ())) (hplace o)
  have hmem : ∀ o ∈ (Finset.univ : Finset (Fin 8)), Rel sm ![![t0 o, t1 o], ![t2 o, t3 o]]
      (fun j ↦ reducePoly Q0 N (Xe (n := N) o) •
        ∑ i : Fin 4, scaleUpMatrix Q0 q3
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ ⟨o.val + 16, by omega⟩))) •
            (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
              ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i)))
      ![t4 o, t5 o] Bs := by
    intro o _
    have h := rel_monomial (hsamp ⟨o.val + 16, by omega⟩) (k := o) (by have := o.isLt; simp only [N]; omega)
    simp only [Prod.mk.injEq] at ht
    obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := ht o
    rw [e0, e1, e2, e3, e4, e5]
    convert h using 1
    · funext a j
      fin_cases a <;> fin_cases j <;> rfl
    · funext j
      fin_cases j <;> rfl
  have hsum := Rel.sum Finset.univ hmem
  rw [blocks_sum, pair_sum] at hsum
  choose eS heS hS using hswitch
  have hsum' := hsum.target_eq (t' := fun j ↦ ∑ x : Fin 8 × Fin 4, scaleUpMatrix Q0 q3
      (reducePoly Q1 N (freshMul (fun i ↦ ζ ⟨i.val + 16, by omega⟩) x) •
        (reducePoly Q1 N (substitution (sigma x.2).toNat (sigma_odd x.2) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j x.2 +
            ![![fam00, fam01], ![fam10, fam11]] 1 j x.2))) (by
    funext j
    simp only [Finset.sum_apply, Fintype.sum_prod_type, freshMul]
    refine Finset.sum_congr rfl fun o _ ↦ ?_
    rw [Finset.smul_sum]
    simp only [smul_scaleUpMatrix q0_eq (by norm_num), smul_smul, ← map_mul])
  have hfin := lut_finish_sum (ι := Fin 8 × Fin 4) q0_eq (by norm_num) (by norm_num)
    (by norm_num) hsm (freshMul (fun i ↦ ζ ⟨i.val + 16, by omega⟩)) (fun x ↦ substitution (sigma x.2).toNat (sigma_odd x.2) sp)
    (fun x ↦ polyNorm_freshMul x)
    (bsw := fun x k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j x.2) hsum' (g' := gq)
    (C := fun x ↦ lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (T k)) (sigma x.2).toNat)
    (Δ' := Δ') (fun x j ↦ hS x.2 j) (fun x j ↦ heS x.2 j)
    (ν := ∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val + 16, by omega⟩)) (by
      simp only [Fintype.sum_prod_type, freshMul, map_sum, map_mul]
      refine Finset.sum_congr rfl fun o _ ↦ ?_
      simp only [mul_assoc, ← Finset.mul_sum]
      congr 1
      exact lut_walsh_value (q := Q1) T hinverse _)
    (K' := fun k j ↦ blockModSwitch basis 1 (![![∑ i, t0 i, ∑ i, t1 i], ![∑ i, t2 i, ∑ i, t3 i]] k j))
    (R' := ![blockModSwitch basis 1 (∑ i, t4 i), c])
    (fun k j ↦ blockModSwitchRuns_fn hdown _)
    (fun j ↦ by fin_cases j; exacts [blockModSwitchRuns_fn hdown _, hdown])
  refine hfin.mono (le_of_eq ?_)
  simp [Finset.sum_const, Finset.card_univ]

theorem place463 {model : HashModel} {params : Stage_online_eval_0_0.Params} {o : Nat}
    {x : Blocks Q0 6} {y : Blocks Q0 6}
    (h : Stage_online_eval_0_0.parallel_generatedRoot_463 Backend.backend model params o
      (x.1, x.2.1, x.2.2.1, x.2.2.2.1, x.2.2.2.2.1, x.2.2.2.2.2.1, ()) y) :
    y = (multiplyMonomial x.1 (o : Int), multiplyMonomial x.2.1 (o : Int),
      multiplyMonomial x.2.2.1 (o : Int), multiplyMonomial x.2.2.2.1 (o : Int),
      multiplyMonomial x.2.2.2.2.1 (o : Int), multiplyMonomial x.2.2.2.2.2.1 (o : Int), ()) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_463 at h
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_463.constraints_0] at h
  rw [h]
  rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Column `3` of the fresh noise. -/
theorem freshColumn3 {model : HashModel} {params : Stage_online_eval_0_0.Params}
    {s0 s1 s2 s3 s4 s5 : Fin 48 → ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {t0 t1 t2 t3 t4 t5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (hplace : ∀ i : Fin 8, Stage_online_eval_0_0.parallel_generatedRoot_463 Backend.backend model params i
      (s0 ⟨i.val + 24, by omega⟩, s1 ⟨i.val + 24, by omega⟩, s2 ⟨i.val + 24, by omega⟩, s3 ⟨i.val + 24, by omega⟩, s4 ⟨i.val + 24, by omega⟩, s5 ⟨i.val + 24, by omega⟩, ())
      (t0 i, t1 i, t2 i, t3 i, t4 i, t5 i, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    {ζ : Fin 48 → Nat} {Bs : Nat}
    (hsamp : ∀ o : Fin 48, Rel sm ![![s0 o, s1 o], ![s2 o, s3 o]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 q3
        (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ o))) •
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i)))
      ![s4 o, s5 o] Bs)
    {T : Fin 4 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix Q1 N 1 6} {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (T k)) (sigma i).toNat *
          reducePoly Q1 N (Δ' * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j))
    {basis : List Nat} {c : ExactMatrix Q1 N 1 6}
    (hdown : blockModSwitchRuns basis 1 (∑ i, t5 i) c) :
    Rel sm (fun k j ↦ blockModSwitch basis 1 (![![∑ i, t0 i, ∑ i, t1 i], ![∑ i, t2 i, ∑ i, t3 i]] k j))
      (fun j ↦ reducePoly Q1 N (Δ' * (∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val + 24, by omega⟩)) *
        unitPair sp j) • gq)
      ![blockModSwitch basis 1 (∑ i, t4 i), c]
      ((8 * Bs + (N + 2) * (q3 / 2)) / q3 + 32 * (N * 1 * BS)) := by
  have ht : ∀ o : Fin 8, (t0 o, t1 o, t2 o, t3 o, t4 o, t5 o, ()) =
      (multiplyMonomial (s0 ⟨o.val + 24, by omega⟩) (o : Int), multiplyMonomial (s1 ⟨o.val + 24, by omega⟩) (o : Int),
        multiplyMonomial (s2 ⟨o.val + 24, by omega⟩) (o : Int), multiplyMonomial (s3 ⟨o.val + 24, by omega⟩) (o : Int),
        multiplyMonomial (s4 ⟨o.val + 24, by omega⟩) (o : Int), multiplyMonomial (s5 ⟨o.val + 24, by omega⟩) (o : Int), ()) :=
    fun o ↦ place463 (x := (s0 ⟨o.val + 24, by omega⟩, s1 ⟨o.val + 24, by omega⟩, s2 ⟨o.val + 24, by omega⟩, s3 ⟨o.val + 24, by omega⟩, s4 ⟨o.val + 24, by omega⟩, s5 ⟨o.val + 24, by omega⟩, ())) (hplace o)
  have hmem : ∀ o ∈ (Finset.univ : Finset (Fin 8)), Rel sm ![![t0 o, t1 o], ![t2 o, t3 o]]
      (fun j ↦ reducePoly Q0 N (Xe (n := N) o) •
        ∑ i : Fin 4, scaleUpMatrix Q0 q3
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ ⟨o.val + 24, by omega⟩))) •
            (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
              ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i)))
      ![t4 o, t5 o] Bs := by
    intro o _
    have h := rel_monomial (hsamp ⟨o.val + 24, by omega⟩) (k := o) (by have := o.isLt; simp only [N]; omega)
    simp only [Prod.mk.injEq] at ht
    obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := ht o
    rw [e0, e1, e2, e3, e4, e5]
    convert h using 1
    · funext a j
      fin_cases a <;> fin_cases j <;> rfl
    · funext j
      fin_cases j <;> rfl
  have hsum := Rel.sum Finset.univ hmem
  rw [blocks_sum, pair_sum] at hsum
  choose eS heS hS using hswitch
  have hsum' := hsum.target_eq (t' := fun j ↦ ∑ x : Fin 8 × Fin 4, scaleUpMatrix Q0 q3
      (reducePoly Q1 N (freshMul (fun i ↦ ζ ⟨i.val + 24, by omega⟩) x) •
        (reducePoly Q1 N (substitution (sigma x.2).toNat (sigma_odd x.2) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j x.2 +
            ![![fam00, fam01], ![fam10, fam11]] 1 j x.2))) (by
    funext j
    simp only [Finset.sum_apply, Fintype.sum_prod_type, freshMul]
    refine Finset.sum_congr rfl fun o _ ↦ ?_
    rw [Finset.smul_sum]
    simp only [smul_scaleUpMatrix q0_eq (by norm_num), smul_smul, ← map_mul])
  have hfin := lut_finish_sum (ι := Fin 8 × Fin 4) q0_eq (by norm_num) (by norm_num)
    (by norm_num) hsm (freshMul (fun i ↦ ζ ⟨i.val + 24, by omega⟩)) (fun x ↦ substitution (sigma x.2).toNat (sigma_odd x.2) sp)
    (fun x ↦ polyNorm_freshMul x)
    (bsw := fun x k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j x.2) hsum' (g' := gq)
    (C := fun x ↦ lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (T k)) (sigma x.2).toNat)
    (Δ' := Δ') (fun x j ↦ hS x.2 j) (fun x j ↦ heS x.2 j)
    (ν := ∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val + 24, by omega⟩)) (by
      simp only [Fintype.sum_prod_type, freshMul, map_sum, map_mul]
      refine Finset.sum_congr rfl fun o _ ↦ ?_
      simp only [mul_assoc, ← Finset.mul_sum]
      congr 1
      exact lut_walsh_value (q := Q1) T hinverse _)
    (K' := fun k j ↦ blockModSwitch basis 1 (![![∑ i, t0 i, ∑ i, t1 i], ![∑ i, t2 i, ∑ i, t3 i]] k j))
    (R' := ![blockModSwitch basis 1 (∑ i, t4 i), c])
    (fun k j ↦ blockModSwitchRuns_fn hdown _)
    (fun j ↦ by fin_cases j; exacts [blockModSwitchRuns_fn hdown _, hdown])
  refine hfin.mono (le_of_eq ?_)
  simp [Finset.sum_const, Finset.card_univ]

theorem place485 {model : HashModel} {params : Stage_online_eval_0_0.Params} {o : Nat}
    {x : Blocks Q0 6} {y : Blocks Q0 6}
    (h : Stage_online_eval_0_0.parallel_generatedRoot_485 Backend.backend model params o
      (x.1, x.2.1, x.2.2.1, x.2.2.2.1, x.2.2.2.2.1, x.2.2.2.2.2.1, ()) y) :
    y = (multiplyMonomial x.1 (o : Int), multiplyMonomial x.2.1 (o : Int),
      multiplyMonomial x.2.2.1 (o : Int), multiplyMonomial x.2.2.2.1 (o : Int),
      multiplyMonomial x.2.2.2.2.1 (o : Int), multiplyMonomial x.2.2.2.2.2.1 (o : Int), ()) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_485 at h
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_485.constraints_0] at h
  rw [h]
  rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Column `4` of the fresh noise. -/
theorem freshColumn4 {model : HashModel} {params : Stage_online_eval_0_0.Params}
    {s0 s1 s2 s3 s4 s5 : Fin 48 → ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {t0 t1 t2 t3 t4 t5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (hplace : ∀ i : Fin 8, Stage_online_eval_0_0.parallel_generatedRoot_485 Backend.backend model params i
      (s0 ⟨i.val + 32, by omega⟩, s1 ⟨i.val + 32, by omega⟩, s2 ⟨i.val + 32, by omega⟩, s3 ⟨i.val + 32, by omega⟩, s4 ⟨i.val + 32, by omega⟩, s5 ⟨i.val + 32, by omega⟩, ())
      (t0 i, t1 i, t2 i, t3 i, t4 i, t5 i, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    {ζ : Fin 48 → Nat} {Bs : Nat}
    (hsamp : ∀ o : Fin 48, Rel sm ![![s0 o, s1 o], ![s2 o, s3 o]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 q3
        (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ o))) •
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i)))
      ![s4 o, s5 o] Bs)
    {T : Fin 4 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix Q1 N 1 6} {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (T k)) (sigma i).toNat *
          reducePoly Q1 N (Δ' * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j))
    {basis : List Nat} {c : ExactMatrix Q1 N 1 6}
    (hdown : blockModSwitchRuns basis 1 (∑ i, t5 i) c) :
    Rel sm (fun k j ↦ blockModSwitch basis 1 (![![∑ i, t0 i, ∑ i, t1 i], ![∑ i, t2 i, ∑ i, t3 i]] k j))
      (fun j ↦ reducePoly Q1 N (Δ' * (∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val + 32, by omega⟩)) *
        unitPair sp j) • gq)
      ![blockModSwitch basis 1 (∑ i, t4 i), c]
      ((8 * Bs + (N + 2) * (q3 / 2)) / q3 + 32 * (N * 1 * BS)) := by
  have ht : ∀ o : Fin 8, (t0 o, t1 o, t2 o, t3 o, t4 o, t5 o, ()) =
      (multiplyMonomial (s0 ⟨o.val + 32, by omega⟩) (o : Int), multiplyMonomial (s1 ⟨o.val + 32, by omega⟩) (o : Int),
        multiplyMonomial (s2 ⟨o.val + 32, by omega⟩) (o : Int), multiplyMonomial (s3 ⟨o.val + 32, by omega⟩) (o : Int),
        multiplyMonomial (s4 ⟨o.val + 32, by omega⟩) (o : Int), multiplyMonomial (s5 ⟨o.val + 32, by omega⟩) (o : Int), ()) :=
    fun o ↦ place485 (x := (s0 ⟨o.val + 32, by omega⟩, s1 ⟨o.val + 32, by omega⟩, s2 ⟨o.val + 32, by omega⟩, s3 ⟨o.val + 32, by omega⟩, s4 ⟨o.val + 32, by omega⟩, s5 ⟨o.val + 32, by omega⟩, ())) (hplace o)
  have hmem : ∀ o ∈ (Finset.univ : Finset (Fin 8)), Rel sm ![![t0 o, t1 o], ![t2 o, t3 o]]
      (fun j ↦ reducePoly Q0 N (Xe (n := N) o) •
        ∑ i : Fin 4, scaleUpMatrix Q0 q3
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ ⟨o.val + 32, by omega⟩))) •
            (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
              ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i)))
      ![t4 o, t5 o] Bs := by
    intro o _
    have h := rel_monomial (hsamp ⟨o.val + 32, by omega⟩) (k := o) (by have := o.isLt; simp only [N]; omega)
    simp only [Prod.mk.injEq] at ht
    obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := ht o
    rw [e0, e1, e2, e3, e4, e5]
    convert h using 1
    · funext a j
      fin_cases a <;> fin_cases j <;> rfl
    · funext j
      fin_cases j <;> rfl
  have hsum := Rel.sum Finset.univ hmem
  rw [blocks_sum, pair_sum] at hsum
  choose eS heS hS using hswitch
  have hsum' := hsum.target_eq (t' := fun j ↦ ∑ x : Fin 8 × Fin 4, scaleUpMatrix Q0 q3
      (reducePoly Q1 N (freshMul (fun i ↦ ζ ⟨i.val + 32, by omega⟩) x) •
        (reducePoly Q1 N (substitution (sigma x.2).toNat (sigma_odd x.2) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j x.2 +
            ![![fam00, fam01], ![fam10, fam11]] 1 j x.2))) (by
    funext j
    simp only [Finset.sum_apply, Fintype.sum_prod_type, freshMul]
    refine Finset.sum_congr rfl fun o _ ↦ ?_
    rw [Finset.smul_sum]
    simp only [smul_scaleUpMatrix q0_eq (by norm_num), smul_smul, ← map_mul])
  have hfin := lut_finish_sum (ι := Fin 8 × Fin 4) q0_eq (by norm_num) (by norm_num)
    (by norm_num) hsm (freshMul (fun i ↦ ζ ⟨i.val + 32, by omega⟩)) (fun x ↦ substitution (sigma x.2).toNat (sigma_odd x.2) sp)
    (fun x ↦ polyNorm_freshMul x)
    (bsw := fun x k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j x.2) hsum' (g' := gq)
    (C := fun x ↦ lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (T k)) (sigma x.2).toNat)
    (Δ' := Δ') (fun x j ↦ hS x.2 j) (fun x j ↦ heS x.2 j)
    (ν := ∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val + 32, by omega⟩)) (by
      simp only [Fintype.sum_prod_type, freshMul, map_sum, map_mul]
      refine Finset.sum_congr rfl fun o _ ↦ ?_
      simp only [mul_assoc, ← Finset.mul_sum]
      congr 1
      exact lut_walsh_value (q := Q1) T hinverse _)
    (K' := fun k j ↦ blockModSwitch basis 1 (![![∑ i, t0 i, ∑ i, t1 i], ![∑ i, t2 i, ∑ i, t3 i]] k j))
    (R' := ![blockModSwitch basis 1 (∑ i, t4 i), c])
    (fun k j ↦ blockModSwitchRuns_fn hdown _)
    (fun j ↦ by fin_cases j; exacts [blockModSwitchRuns_fn hdown _, hdown])
  refine hfin.mono (le_of_eq ?_)
  simp [Finset.sum_const, Finset.card_univ]

theorem place507 {model : HashModel} {params : Stage_online_eval_0_0.Params} {o : Nat}
    {x : Blocks Q0 6} {y : Blocks Q0 6}
    (h : Stage_online_eval_0_0.parallel_generatedRoot_507 Backend.backend model params o
      (x.1, x.2.1, x.2.2.1, x.2.2.2.1, x.2.2.2.2.1, x.2.2.2.2.2.1, ()) y) :
    y = (multiplyMonomial x.1 (o : Int), multiplyMonomial x.2.1 (o : Int),
      multiplyMonomial x.2.2.1 (o : Int), multiplyMonomial x.2.2.2.1 (o : Int),
      multiplyMonomial x.2.2.2.2.1 (o : Int), multiplyMonomial x.2.2.2.2.2.1 (o : Int), ()) := by
  unfold Stage_online_eval_0_0.parallel_generatedRoot_507 at h
  simp only [Stage_online_eval_0_0.parallel_generatedRoot_507.constraints_0] at h
  rw [h]
  rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Column `5` of the fresh noise. -/
theorem freshColumn5 {model : HashModel} {params : Stage_online_eval_0_0.Params}
    {s0 s1 s2 s3 s4 s5 : Fin 48 → ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {t0 t1 t2 t3 t4 t5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (hplace : ∀ i : Fin 8, Stage_online_eval_0_0.parallel_generatedRoot_507 Backend.backend model params i
      (s0 ⟨i.val + 40, by omega⟩, s1 ⟨i.val + 40, by omega⟩, s2 ⟨i.val + 40, by omega⟩, s3 ⟨i.val + 40, by omega⟩, s4 ⟨i.val + 40, by omega⟩, s5 ⟨i.val + 40, by omega⟩, ())
      (t0 i, t1 i, t2 i, t3 i, t4 i, t5 i, ()))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    {ζ : Fin 48 → Nat} {Bs : Nat}
    (hsamp : ∀ o : Fin 48, Rel sm ![![s0 o, s1 o], ![s2 o, s3 o]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 q3
        (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ o))) •
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i)))
      ![s4 o, s5 o] Bs)
    {T : Fin 4 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix Q1 N 1 6} {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i =
        (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (T k)) (sigma i).toNat *
          reducePoly Q1 N (Δ' * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j))
    {basis : List Nat} {c : ExactMatrix Q1 N 1 6}
    (hdown : blockModSwitchRuns basis 1 (∑ i, t5 i) c) :
    Rel sm (fun k j ↦ blockModSwitch basis 1 (![![∑ i, t0 i, ∑ i, t1 i], ![∑ i, t2 i, ∑ i, t3 i]] k j))
      (fun j ↦ reducePoly Q1 N (Δ' * (∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val + 40, by omega⟩)) *
        unitPair sp j) • gq)
      ![blockModSwitch basis 1 (∑ i, t4 i), c]
      ((8 * Bs + (N + 2) * (q3 / 2)) / q3 + 32 * (N * 1 * BS)) := by
  have ht : ∀ o : Fin 8, (t0 o, t1 o, t2 o, t3 o, t4 o, t5 o, ()) =
      (multiplyMonomial (s0 ⟨o.val + 40, by omega⟩) (o : Int), multiplyMonomial (s1 ⟨o.val + 40, by omega⟩) (o : Int),
        multiplyMonomial (s2 ⟨o.val + 40, by omega⟩) (o : Int), multiplyMonomial (s3 ⟨o.val + 40, by omega⟩) (o : Int),
        multiplyMonomial (s4 ⟨o.val + 40, by omega⟩) (o : Int), multiplyMonomial (s5 ⟨o.val + 40, by omega⟩) (o : Int), ()) :=
    fun o ↦ place507 (x := (s0 ⟨o.val + 40, by omega⟩, s1 ⟨o.val + 40, by omega⟩, s2 ⟨o.val + 40, by omega⟩, s3 ⟨o.val + 40, by omega⟩, s4 ⟨o.val + 40, by omega⟩, s5 ⟨o.val + 40, by omega⟩, ())) (hplace o)
  have hmem : ∀ o ∈ (Finset.univ : Finset (Fin 8)), Rel sm ![![t0 o, t1 o], ![t2 o, t3 o]]
      (fun j ↦ reducePoly Q0 N (Xe (n := N) o) •
        ∑ i : Fin 4, scaleUpMatrix Q0 q3
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ ⟨o.val + 40, by omega⟩))) •
            (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
              ![![fam00, fam01], ![fam10, fam11]] 0 j i + ![![fam00, fam01], ![fam10, fam11]] 1 j i)))
      ![t4 o, t5 o] Bs := by
    intro o _
    have h := rel_monomial (hsamp ⟨o.val + 40, by omega⟩) (k := o) (by have := o.isLt; simp only [N]; omega)
    simp only [Prod.mk.injEq] at ht
    obtain ⟨e0, e1, e2, e3, e4, e5, -⟩ := ht o
    rw [e0, e1, e2, e3, e4, e5]
    convert h using 1
    · funext a j
      fin_cases a <;> fin_cases j <;> rfl
    · funext j
      fin_cases j <;> rfl
  have hsum := Rel.sum Finset.univ hmem
  rw [blocks_sum, pair_sum] at hsum
  choose eS heS hS using hswitch
  have hsum' := hsum.target_eq (t' := fun j ↦ ∑ x : Fin 8 × Fin 4, scaleUpMatrix Q0 q3
      (reducePoly Q1 N (freshMul (fun i ↦ ζ ⟨i.val + 40, by omega⟩) x) •
        (reducePoly Q1 N (substitution (sigma x.2).toNat (sigma_odd x.2) sp) •
          ![![fam00, fam01], ![fam10, fam11]] 0 j x.2 +
            ![![fam00, fam01], ![fam10, fam11]] 1 j x.2))) (by
    funext j
    simp only [Finset.sum_apply, Fintype.sum_prod_type, freshMul]
    refine Finset.sum_congr rfl fun o _ ↦ ?_
    rw [Finset.smul_sum]
    simp only [smul_scaleUpMatrix q0_eq (by norm_num), smul_smul, ← map_mul])
  have hfin := lut_finish_sum (ι := Fin 8 × Fin 4) q0_eq (by norm_num) (by norm_num)
    (by norm_num) hsm (freshMul (fun i ↦ ζ ⟨i.val + 40, by omega⟩)) (fun x ↦ substitution (sigma x.2).toNat (sigma_odd x.2) sp)
    (fun x ↦ polyNorm_freshMul x)
    (bsw := fun x k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j x.2) hsum' (g' := gq)
    (C := fun x ↦ lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (T k)) (sigma x.2).toNat)
    (Δ' := Δ') (fun x j ↦ hS x.2 j) (fun x j ↦ heS x.2 j)
    (ν := ∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val + 40, by omega⟩)) (by
      simp only [Fintype.sum_prod_type, freshMul, map_sum, map_mul]
      refine Finset.sum_congr rfl fun o _ ↦ ?_
      simp only [mul_assoc, ← Finset.mul_sum]
      congr 1
      exact lut_walsh_value (q := Q1) T hinverse _)
    (K' := fun k j ↦ blockModSwitch basis 1 (![![∑ i, t0 i, ∑ i, t1 i], ![∑ i, t2 i, ∑ i, t3 i]] k j))
    (R' := ![blockModSwitch basis 1 (∑ i, t4 i), c])
    (fun k j ↦ blockModSwitchRuns_fn hdown _)
    (fun j ↦ by fin_cases j; exacts [blockModSwitchRuns_fn hdown _, hdown])
  refine hfin.mono (le_of_eq ?_)
  simp [Finset.sum_const, Finset.card_univ]

end RefreshTopProof.Part_0_0
