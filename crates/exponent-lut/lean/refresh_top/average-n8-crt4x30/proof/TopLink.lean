import TopAction
import TopDecode
import TopResidue

/-!
# The residues of a decoder's output

The online stage of chain `0` outputs, residue by residue modulo `q₀`, the refresh's value
`y = β A'₀ + A'₁ - T + e'` at `Q₁`; the decode stage recombines the residues of
`σ A₀ + A₁ - q₃ r (c X^e) G + E` at `Q₀`. Both are reductions of integer polynomials, which agree
modulo `q₀` when `β ≡ σ`, `x = r (c X^e)` and `E = e'`: `A'` reduces `A`, and `T` reduces
`(c' / q₁) x W` with `W ≡ q₃ q₁ G` and `c' ≡ 1 (mod q₀)`.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh

/-- The residue modulo `q₀` of a coefficient of a reduction modulo `Q`, `q₀ ∣ Q`. -/
theorem val_reduce_q0 {Q : Nat} (hQ : 1 < Q) (hdvd : q0 ∣ Q) (Y : ErrorPoly N) (i : Fin N) :
    ((((reducePoly Q N Y).coeff i).val : Nat) : ZMod q0) = (reducePoly q0 N Y).coeff i := by
  haveI : NeZero Q := ⟨by omega⟩
  rw [reducePoly_coeff_eq hQ hN, reducePoly_coeff_eq (by norm_num) hN]
  have := val_cast_dvd hdvd (Y.coeff i)
  push_cast at this ⊢
  exact this

set_option maxHeartbeats 4000000 in
/-- The decode stage's residues of chain `0`, from the online stage's output modulo `q₀`. -/
theorem link_q0 {A : ExactMatrix Q0 N (1 + 1) 6} {A' : ExactMatrix Q1 N (1 + 1) 6}
    (hA' : modulusReduceRuns A A') {G : ExactMatrix Q1 N 1 6} {w345 : ExactMatrix Q0 N 1 6}
    (h345 : centeredRebaseRuns G w345) {w346 : ExactMatrix q0 N 1 6}
    (h346 : modulusReduceRuns w345 w346) {w347 w349 : ExactMatrix q0 N 1 1}
    (e347 : w347 = (matrixPolynomial [(q3 : Int)] : ExactMatrix q0 N 1 1))
    (e349 : w349 = (matrixPolynomial [(q1 : Int)] : ExactMatrix q0 N 1 1))
    {w348 w350 : ExactMatrix q0 N 1 6} (e348 : w348 = matrixMulScalarRight w346 w347)
    (e350 : w350 = matrixMulScalarRight w348 w349)
    {βI σI xI rI : ErrorPoly N} (hβ : reducePoly q0 N βI = reducePoly q0 N σI)
    {σ : ExactMatrix Q0 N 1 1} (hσ : σ = fun _ _ ↦ reducePoly Q0 N σI) {c e : Int}
    (hx : xI = message c e * rI) (E : ErrorMatrix N 1 6) {e' : ExactMatrix Q1 N 1 6}
    (hE : e' = reduceMatrix Q1 N 1 6 E) {res : ExactMatrix q0 N 1 6}
    (hres : ∀ k i, (res 0 k).coeff i =
      ((((reducePoly Q1 N βI • rowOf A' 0 + rowOf A' 1 - actionValue xI w350 + e') 0 k).coeff i).val :
        ZMod q0)) :
    Residue res (expectedRow A σ (fun _ _ ↦ reducePoly Q0 N rI) c e G + reduceMatrix Q0 N 1 6 E) := by
  set L : ErrorMatrix N (1 + 1) 6 := fun r k ↦ centeredIntLift (A r k) with hL
  have hA : A = reduceMatrix Q0 N (1 + 1) 6 L := (reduceMatrix_centeredIntLift (by norm_num) hN A).symm
  have hA'r := modulusReduceRuns_reduce (by norm_num) (by norm_num) hN hA' L hA
  set Lg : ErrorMatrix N 1 6 := fun r k ↦ centeredIntLift (G r k) with hLg
  have h345r := centeredRebaseRuns_reduce h345
  have h346r := modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h346 Lg h345r
  subst hE
  intro k i
  rw [hres k i]
  set W : ErrorPoly N := centeredIntLift (w350 0 k) with hW
  have hWr : reducePoly q0 N W = reducePoly q0 N (Lg 0 k * (q3 : Int) * (q1 : Int)) := by
    rw [hW, reducePoly_centeredIntLift (by norm_num) hN, e350, e348, e347, e349, h346r]
    simp only [matrixMulScalarRight, matrixPolynomial_single, reduceMatrix, map_mul, map_intCast]
  have hy : (reducePoly Q1 N βI • rowOf A' 0 + rowOf A' 1 - actionValue xI w350 +
      reduceMatrix Q1 N 1 6 E) 0 k = reducePoly Q1 N
        (βI * L 0 k + L 1 k - ((actionConst / q1 : Int) * xI) * W + E 0 k) := by
    rw [hA'r]
    simp only [actionValue, rowOf, reduceMatrix, Matrix.add_apply, Matrix.sub_apply,
      Matrix.smul_apply, Pi.smul_apply, smul_eq_mul, map_add, map_sub, map_mul, map_intCast, hW]
  have hv : (expectedRow A σ (fun _ _ ↦ reducePoly Q0 N rI) c e G + reduceMatrix Q0 N 1 6 E) 0 k =
      reducePoly Q0 N (σI * L 0 k + L 1 k - (((q3 : Int) : ErrorPoly N) * rI * message c e) * Lg 0 k +
        E 0 k) := by
    rw [hσ, hA]
    simp only [expectedRow, matrixSub, matrixAdd, matrixMulScalarLeft, multiplyMonomial_liftInteger,
      matrixPolynomial_single, reduceMatrix, Matrix.add_apply, Matrix.sub_apply, map_add, map_sub,
      map_mul, map_intCast, centeredRebase_eq, hLg]
    rfl
  rw [hy, hv, val_reduce_q0 (by norm_num) (by norm_num), val_reduce_q0 (by norm_num) (by norm_num)]
  congr 1
  have hc : reducePoly q0 N (((actionConst / q1 : Int) : ErrorPoly N) * ((q1 : Int) : ErrorPoly N)) =
      1 := by
    rw [← Int.cast_mul, reducePoly_intCast, intCast_eq_algebraMap]
    have h1 : (((actionConst / q1 : Int) * (q1 : Int) : Int) : ZMod q0) = 1 := by decide
    rw [h1, map_one]
  simp only [map_add, map_sub, map_mul, map_intCast, hWr, hx, ← hβ] at hc ⊢
  linear_combination (-(reducePoly q0 N (message c e) * reducePoly q0 N rI * reducePoly q0 N (Lg 0 k) *
    ((q3 : Int) : ExactPoly q0 N))) * hc

end RefreshTopProof
