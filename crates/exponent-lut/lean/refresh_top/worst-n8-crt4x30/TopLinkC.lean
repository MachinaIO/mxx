import TopLink

/-!
# The residues of a decoder's output, for every chain

Chain `c` decodes at the ring `R` (`Q₁` or `q₀ q₁ q₃`) and outputs residues modulo its prime
`q_c`. The online stage outputs, modulo `q_c`, `y = β A'₀ + A'₁ - T + e'` with
`T = k x W` and `W` the centered lift of `k₃ k₁ (G mod q_c)`, where `k k₃ k₁ ≡ q₃ (mod q_c)`.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh

/-- `T = k x W` at the ring `R`. -/
noncomputable def actionValueC {R qc : Nat} (k : Int) (x : ErrorPoly N) (w : ExactMatrix qc N 1 6) :
    ExactMatrix R N 1 6 :=
  reduceMatrix R N 1 6 (((k : ErrorPoly N) * x) • fun r c ↦ centeredIntLift (w r c))

theorem actionValue_eq (x : ErrorPoly N) (w : ExactMatrix q0 N 1 6) :
    actionValue x w = actionValueC (R := Q1) (actionConst / q1) x w := rfl

/-- The residue modulo `q` of a coefficient of a reduction modulo `Q`, `q ∣ Q`. -/
theorem val_reduce {Q qc : Nat} (hQ : 1 < Q) (hqc : 1 < qc) (hdvd : qc ∣ Q) (Y : ErrorPoly N)
    (i : Fin N) : ((((reducePoly Q N Y).coeff i).val : Nat) : ZMod qc) = (reducePoly qc N Y).coeff i := by
  haveI : NeZero Q := ⟨by omega⟩
  rw [reducePoly_coeff_eq hQ hN, reducePoly_coeff_eq hqc hN]
  have := val_cast_dvd hdvd (Y.coeff i)
  push_cast at this ⊢
  exact this

set_option maxHeartbeats 4000000 in
/-- The decode stage's residues of a chain, from the online stage's output modulo its prime. -/
theorem link_chain {R qc : Nat} (hR : 1 < R) (hqc : 1 < qc) (hqcR : qc ∣ R) (hqcQ0 : qc ∣ Q0)
    {A : ExactMatrix Q0 N (1 + 1) 6} {A' : ExactMatrix R N (1 + 1) 6} (hA' : modulusReduceRuns A A')
    {GR : ExactMatrix R N 1 6} {G : ExactMatrix Q1 N 1 6}
    (hLG : ∀ c, centeredIntLift (GR 0 c) = centeredIntLift (G 0 c))
    {w345 : ExactMatrix Q0 N 1 6} (h345 : centeredRebaseRuns GR w345) {w346 : ExactMatrix qc N 1 6}
    (h346 : modulusReduceRuns w345 w346) {k3 k1 k : Int} {w347 w349 : ExactMatrix qc N 1 1}
    (e347 : w347 = (matrixPolynomial [k3] : ExactMatrix qc N 1 1))
    (e349 : w349 = (matrixPolynomial [k1] : ExactMatrix qc N 1 1))
    {w348 w350 : ExactMatrix qc N 1 6} (e348 : w348 = matrixMulScalarRight w346 w347)
    (e350 : w350 = matrixMulScalarRight w348 w349)
    (hk : ((k * k3 * k1 : Int) : ZMod qc) = ((q3 : Int) : ZMod qc))
    {βI σI xI rI : ErrorPoly N} (hβ : reducePoly qc N βI = reducePoly qc N σI)
    {σ : ExactMatrix Q0 N 1 1} (hσ : σ = fun _ _ ↦ reducePoly Q0 N σI) {c e : Int}
    (hx : xI = message c e * rI) (E : ErrorMatrix N 1 6) {e' : ExactMatrix R N 1 6}
    (hE : e' = reduceMatrix R N 1 6 E) {res : ExactMatrix qc N 1 6}
    (hres : ∀ col i, (res 0 col).coeff i =
      ((((reducePoly R N βI • rowOf A' 0 + rowOf A' 1 - actionValueC k xI w350 + e' :
        ExactMatrix R N 1 6) 0 col).coeff i).val :
        ZMod qc)) :
    Residue res (expectedRow A σ (fun _ _ ↦ reducePoly Q0 N rI) c e G + reduceMatrix Q0 N 1 6 E) := by
  haveI : NeZero qc := ⟨by omega⟩
  subst hE
  set L : ErrorMatrix N (1 + 1) 6 := fun r k ↦ centeredIntLift (A r k) with hL
  have hA : A = reduceMatrix Q0 N (1 + 1) 6 L := (reduceMatrix_centeredIntLift (by norm_num) hN A).symm
  have hA'r := modulusReduceRuns_reduce (by norm_num) hR hN hA' L hA
  set Lg : ErrorMatrix N 1 6 := fun r k ↦ centeredIntLift (GR r k) with hLg
  have h345r := centeredRebaseRuns_reduce h345
  have h346r := modulusReduceRuns_reduce (by norm_num) hqc hN h346 Lg h345r
  intro j i
  rw [hres j i]
  set W : ErrorPoly N := centeredIntLift (w350 0 j) with hW
  have hWr : reducePoly qc N W = reducePoly qc N (Lg 0 j * (k3 : ErrorPoly N) * (k1 : ErrorPoly N)) := by
    rw [hW, reducePoly_centeredIntLift hqc hN, e350, e348, e347, e349, h346r]
    simp only [matrixMulScalarRight, matrixPolynomial_single, reduceMatrix, map_mul, map_intCast]
  have hy : (reducePoly R N βI • rowOf A' 0 + rowOf A' 1 - actionValueC k xI w350 +
      reduceMatrix R N 1 6 E : ExactMatrix R N 1 6) 0 j = reducePoly R N
        (βI * L 0 j + L 1 j - ((k : ErrorPoly N) * xI) * W + E 0 j) := by
    rw [hA'r]
    simp only [actionValueC, rowOf, reduceMatrix, Matrix.add_apply, Matrix.sub_apply,
      Matrix.smul_apply, Pi.smul_apply, smul_eq_mul, map_add, map_sub, map_mul, map_intCast, hW]
  have hv : (expectedRow A σ (fun _ _ ↦ reducePoly Q0 N rI) c e G + reduceMatrix Q0 N 1 6 E) 0 j =
      reducePoly Q0 N (σI * L 0 j + L 1 j - (((q3 : Int) : ErrorPoly N) * rI * message c e) *
        centeredIntLift (G 0 j) + E 0 j) := by
    rw [hσ, hA]
    simp only [expectedRow, matrixSub, matrixAdd, matrixMulScalarLeft, multiplyMonomial_liftInteger,
      matrixPolynomial_single, reduceMatrix, Matrix.add_apply, Matrix.sub_apply, map_add, map_sub,
      map_mul, map_intCast, centeredRebase_eq]
    rfl
  rw [hy, hv, val_reduce hR hqc hqcR, val_reduce (by norm_num) hqc hqcQ0]
  congr 1
  have hc : reducePoly qc N ((k : ErrorPoly N) * (k3 : ErrorPoly N) * (k1 : ErrorPoly N)) =
      reducePoly qc N ((q3 : Int) : ErrorPoly N) := by
    rw [← Int.cast_mul, ← Int.cast_mul, reducePoly_intCast, reducePoly_intCast, intCast_eq_algebraMap,
      intCast_eq_algebraMap, hk]
  have hl : Lg 0 j = centeredIntLift (G 0 j) := hLG j
  simp only [map_add, map_sub, map_mul, map_intCast, hWr, hx, ← hβ, hl] at hc ⊢
  linear_combination (-(reducePoly qc N (message c e) * reducePoly qc N rI *
    reducePoly qc N (centeredIntLift (G 0 j)))) * hc

set_option maxHeartbeats 4000000 in
/-- The decode stage's residues of a chain whose action vanishes modulo its prime (`q₃ ≡ 0`). -/
theorem link_chain0 {R qc : Nat} (hR : 1 < R) (hqc : 1 < qc) (hqcR : qc ∣ R) (hqcQ0 : qc ∣ Q0)
    {A : ExactMatrix Q0 N (1 + 1) 6} {A' : ExactMatrix R N (1 + 1) 6} (hA' : modulusReduceRuns A A')
    {GR : ExactMatrix R N 1 6} {G : ExactMatrix Q1 N 1 6}
    {w345 : ExactMatrix Q0 N 1 6} (h345 : centeredRebaseRuns GR w345) {w346 : ExactMatrix qc N 1 6}
    (h346 : modulusReduceRuns w345 w346) {k3 k1 k : Int} {w347 w349 : ExactMatrix qc N 1 1}
    (e347 : w347 = (matrixPolynomial [k3] : ExactMatrix qc N 1 1))
    (e349 : w349 = (matrixPolynomial [k1] : ExactMatrix qc N 1 1))
    {w348 w350 : ExactMatrix qc N 1 6} (e348 : w348 = matrixMulScalarRight w346 w347)
    (e350 : w350 = matrixMulScalarRight w348 w349)
    (hk : ((k * k3 * k1 : Int) : ZMod qc) = 0) (hq3 : ((q3 : Int) : ZMod qc) = 0)
    {βI σI xI rI : ErrorPoly N} (hβ : reducePoly qc N βI = reducePoly qc N σI)
    {σ : ExactMatrix Q0 N 1 1} (hσ : σ = fun _ _ ↦ reducePoly Q0 N σI) {c e : Int}
    (hx : xI = message c e * rI) (E : ErrorMatrix N 1 6) {e' : ExactMatrix R N 1 6}
    (hE : e' = reduceMatrix R N 1 6 E) {res : ExactMatrix qc N 1 6}
    (hres : ∀ col i, (res 0 col).coeff i =
      ((((reducePoly R N βI • rowOf A' 0 + rowOf A' 1 - actionValueC k xI w350 + e' :
        ExactMatrix R N 1 6) 0 col).coeff i).val :
        ZMod qc)) :
    Residue res (expectedRow A σ (fun _ _ ↦ reducePoly Q0 N rI) c e G + reduceMatrix Q0 N 1 6 E) := by
  haveI : NeZero qc := ⟨by omega⟩
  subst hE
  set L : ErrorMatrix N (1 + 1) 6 := fun r k ↦ centeredIntLift (A r k) with hL
  have hA : A = reduceMatrix Q0 N (1 + 1) 6 L := (reduceMatrix_centeredIntLift (by norm_num) hN A).symm
  have hA'r := modulusReduceRuns_reduce (by norm_num) hR hN hA' L hA
  set Lg : ErrorMatrix N 1 6 := fun r k ↦ centeredIntLift (GR r k) with hLg
  have h345r := centeredRebaseRuns_reduce h345
  have h346r := modulusReduceRuns_reduce (by norm_num) hqc hN h346 Lg h345r
  intro j i
  rw [hres j i]
  set W : ErrorPoly N := centeredIntLift (w350 0 j) with hW
  have hWr : reducePoly qc N W = reducePoly qc N (Lg 0 j * (k3 : ErrorPoly N) * (k1 : ErrorPoly N)) := by
    rw [hW, reducePoly_centeredIntLift hqc hN, e350, e348, e347, e349, h346r]
    simp only [matrixMulScalarRight, matrixPolynomial_single, reduceMatrix, map_mul, map_intCast]
  have hy : (reducePoly R N βI • rowOf A' 0 + rowOf A' 1 - actionValueC k xI w350 +
      reduceMatrix R N 1 6 E : ExactMatrix R N 1 6) 0 j = reducePoly R N
        (βI * L 0 j + L 1 j - ((k : ErrorPoly N) * xI) * W + E 0 j) := by
    rw [hA'r]
    simp only [actionValueC, rowOf, reduceMatrix, Matrix.add_apply, Matrix.sub_apply,
      Matrix.smul_apply, Pi.smul_apply, smul_eq_mul, map_add, map_sub, map_mul, map_intCast, hW]
  have hv : (expectedRow A σ (fun _ _ ↦ reducePoly Q0 N rI) c e G + reduceMatrix Q0 N 1 6 E) 0 j =
      reducePoly Q0 N (σI * L 0 j + L 1 j - (((q3 : Int) : ErrorPoly N) * rI * message c e) *
        centeredIntLift (G 0 j) + E 0 j) := by
    rw [hσ, hA]
    simp only [expectedRow, matrixSub, matrixAdd, matrixMulScalarLeft, multiplyMonomial_liftInteger,
      matrixPolynomial_single, reduceMatrix, Matrix.add_apply, Matrix.sub_apply, map_add, map_sub,
      map_mul, map_intCast, centeredRebase_eq]
    rfl
  rw [hy, hv, val_reduce hR hqc hqcR, val_reduce (by norm_num) hqc hqcQ0]
  congr 1
  have hc : reducePoly qc N ((k : ErrorPoly N) * (k3 : ErrorPoly N) * (k1 : ErrorPoly N)) = 0 := by
    rw [← Int.cast_mul, ← Int.cast_mul, reducePoly_intCast, intCast_eq_algebraMap, hk, map_zero]
  have hc3 : reducePoly qc N ((q3 : Int) : ErrorPoly N) = 0 := by
    rw [reducePoly_intCast, intCast_eq_algebraMap, hq3, map_zero]
  simp only [map_add, map_sub, map_mul, map_intCast, hWr, hx, ← hβ] at hc hc3 ⊢
  linear_combination (-(reducePoly qc N (message c e) * reducePoly qc N rI * reducePoly qc N (Lg 0 j))) * hc +
    (reducePoly qc N rI * reducePoly qc N (message c e) * reducePoly qc N (centeredIntLift (G 0 j))) * hc3

end RefreshTopProof
