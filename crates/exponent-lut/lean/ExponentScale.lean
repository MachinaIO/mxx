import PrimitivesBounds
import RuntimeMatrixOps
import RuntimeLemmas

/-!
# Modulus drops

A level of the modulus chain drops the factor `p = q / q'`. Multiplication by `p` is a well-defined
additive map `scaleUp : ℤ_{q'}[X]/(X^n + 1) → ℤ_q[X]/(X^n + 1)`, injective and compatible with
multiplication by reductions of integer polynomials. The runtime ModDown (`blockModSwitchRuns` with
plaintext modulus one) returns `y` with `p · y = x - r` for the centered remainder `r` of `x`
modulo `p`; together with injectivity this divides an encoding relation by `p`.

This module also identifies the runtime reductions between nested moduli (`modulusReduceRuns`,
`centeredRebaseRuns`) with reductions of integer polynomials.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime
open scoped BigOperators

variable {n : Nat}

/-- Multiplication by `p` from `ℤ_{q'}` to `ℤ_q`, through the canonical integer lift. -/
noncomputable def scaleUp (q p : Nat) {q' : Nat} (x : ExactPoly q' n) : ExactPoly q n :=
  reducePoly q n (((p : Int) : ErrorPoly n) * canonicalLift x)

/-- `scaleUp` applied entrywise. -/
noncomputable def scaleUpMatrix (q p : Nat) {q' rows columns : Nat}
    (x : ExactMatrix q' n rows columns) : ExactMatrix q n rows columns :=
  fun row column ↦ scaleUp q p (x row column)

section Scale

variable {q p q' : Nat}

/-- `scaleUp` of a reduction is the reduction of `p` times the integer polynomial. -/
theorem scaleUp_reducePoly (hq : q = p * q') (hq' : 1 < q') (hn : 0 < n) (z : ErrorPoly n) :
    scaleUp q p (reducePoly q' n z) = reducePoly q n (((p : Int) : ErrorPoly n) * z) := by
  obtain ⟨k, hk⟩ :=
    exists_of_reducePoly_eq hq' hn (reducePoly_canonicalLift hq' hn (reducePoly q' n z))
  unfold scaleUp
  have hpq : ((p : Int) : ErrorPoly n) * ((q' : Int) : ErrorPoly n) = ((q : Int) : ErrorPoly n) := by
    rw [hq]; push_cast; ring
  rw [hk, mul_add, map_add, ← mul_assoc, hpq, reducePoly_modulus_mul, add_zero]

theorem scaleUp_add (hq : q = p * q') (hq' : 1 < q') (hn : 0 < n) (x y : ExactPoly q' n) :
    scaleUp q p (x + y) = scaleUp q p x + scaleUp q p y := by
  rw [← reducePoly_canonicalLift hq' hn x, ← reducePoly_canonicalLift hq' hn y, ← map_add,
    scaleUp_reducePoly hq hq' hn, scaleUp_reducePoly hq hq' hn, scaleUp_reducePoly hq hq' hn,
    mul_add, map_add]

theorem scaleUp_neg (hq : q = p * q') (hq' : 1 < q') (hn : 0 < n) (x : ExactPoly q' n) :
    scaleUp q p (-x) = -scaleUp q p x := by
  rw [← reducePoly_canonicalLift hq' hn x, ← map_neg, scaleUp_reducePoly hq hq' hn,
    scaleUp_reducePoly hq hq' hn, mul_neg, map_neg]

theorem scaleUp_sub (hq : q = p * q') (hq' : 1 < q') (hn : 0 < n) (x y : ExactPoly q' n) :
    scaleUp q p (x - y) = scaleUp q p x - scaleUp q p y := by
  rw [sub_eq_add_neg, scaleUp_add hq hq' hn, scaleUp_neg hq hq' hn, ← sub_eq_add_neg]

theorem scaleUp_zero (hq : q = p * q') (hq' : 1 < q') (hn : 0 < n) :
    scaleUp q p (0 : ExactPoly q' n) = 0 := by
  rw [← map_zero (reducePoly q' n), scaleUp_reducePoly hq hq' hn, mul_zero, map_zero]

theorem scaleUp_sum {ι : Type} [DecidableEq ι] (hq : q = p * q') (hq' : 1 < q') (hn : 0 < n)
    (s : Finset ι) (f : ι → ExactPoly q' n) :
    scaleUp q p (∑ i ∈ s, f i) = ∑ i ∈ s, scaleUp q p (f i) := by
  induction s using Finset.induction_on with
  | empty => simp [scaleUp_zero hq hq' hn]
  | @insert i s hi ih => rw [Finset.sum_insert hi, Finset.sum_insert hi, scaleUp_add hq hq' hn, ih]

/-- Reductions of integer polynomials factor out of `scaleUp`. -/
theorem scaleUp_reducePoly_mul (hq : q = p * q') (hq' : 1 < q') (hn : 0 < n) (a : ErrorPoly n)
    (x : ExactPoly q' n) :
    scaleUp q p (reducePoly q' n a * x) = reducePoly q n a * scaleUp q p x := by
  rw [← reducePoly_canonicalLift hq' hn x, ← map_mul, scaleUp_reducePoly hq hq' hn,
    scaleUp_reducePoly hq hq' hn, ← map_mul]
  congr 1
  ring

/-- Division by `p`: an integer polynomial that `scaleUp x` reduces from is `p` times a lift of
`x`. -/
theorem scaleUp_eq_reducePoly (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q') (hn : 0 < n)
    {x : ExactPoly q' n} {w : ErrorPoly n} (h : scaleUp q p x = reducePoly q n w) :
    ∃ v : ErrorPoly n, w = ((p : Int) : ErrorPoly n) * v ∧ x = reducePoly q' n v := by
  obtain ⟨k, hk⟩ := exists_of_reducePoly_eq hq1 hn h.symm
  refine ⟨canonicalLift x + ((q' : Int) : ErrorPoly n) * k, ?_, ?_⟩
  · have hpq : ((q : Int) : ErrorPoly n) =
        ((p : Int) : ErrorPoly n) * ((q' : Int) : ErrorPoly n) := by
      rw [hq]; push_cast; ring
    rw [hk, hpq]
    ring
  · rw [map_add, reducePoly_canonicalLift hq' hn, reducePoly_modulus_mul, add_zero]

end Scale

/-! ## Runtime reductions between nested moduli -/

section Reductions

variable {q p : Nat}

/-- `modulusReduceRuns` reduces the integer polynomials a matrix is reduced from. -/
theorem modulusReduceRuns_reduce (hq : 1 < q) (hp : 1 < p) (hn : 0 < n) {rows columns : Nat}
    {input : ExactMatrix q n rows columns} {output : ExactMatrix p n rows columns}
    (h : modulusReduceRuns input output) (x : ErrorMatrix n rows columns)
    (hx : input = reduceMatrix q n rows columns x) :
    output = reduceMatrix p n rows columns x := by
  obtain ⟨_, hdvd, hout⟩ := h
  rw [hout, hx]
  funext row column
  exact modulusReduce_reducePoly hq hp hdvd hn (x row column)

/-- `centeredRebaseRuns` re-encodes the centered integer lift of every entry. -/
theorem centeredRebaseRuns_reduce {rows columns : Nat}
    {input : ExactMatrix q n rows columns} {output : ExactMatrix p n rows columns}
    (h : centeredRebaseRuns input output) :
    output = fun row column ↦ reducePoly p n (centeredIntLift (input row column)) := by
  obtain ⟨_, hout⟩ := h
  rw [hout]
  funext row column
  exact centeredRebase_eq input row column

end Reductions

/-! ## ModDown -/

section ModDown

variable {q p q' : Nat}

private theorem rnsInverse_one' {m : Nat} (hm : 1 < m) : rnsInverse m 1 = 1 := rnsInverse_one hm

/-- One coefficient of the runtime ModDown with plaintext modulus one: `p` times the output is
the input minus its centered remainder modulo `p`. -/
theorem blockModSwitch_coefficient (hq : q = p * q') (hp : 1 < p) (z : Int) :
    ((p : Int) * ((z - rnsCentered p (z * rnsInverse p 1)) / p % q') : Int) ≡
      z - rnsCentered p z [ZMOD q] := by
  rw [rnsInverse_one' hp, mul_one]
  have hdvd : (p : Int) ∣ z - rnsCentered p z := (rnsCentered_modEq p z).dvd
  obtain ⟨m, hm⟩ := hdvd
  rw [hm, Int.mul_ediv_cancel_left _ (by omega)]
  have hqi : (q : Int) = p * q' := by rw [hq]; push_cast; ring
  rw [hqi]
  exact (Int.mod_modEq m q').mul_left' (c := p)

/-- The runtime ModDown with plaintext modulus one: `scaleUp` of its output is its input minus a
centered remainder of norm at most `p / 2`, where `p` is the product of the dropped primes. -/
theorem blockModSwitchRuns_scaleUp (hn : 0 < n) {rows columns : Nat} {basis : List Nat}
    {input : ExactMatrix q n rows columns} {output : ExactMatrix q' n rows columns}
    (h : blockModSwitchRuns basis 1 input output) :
    ∃ remainder : ErrorMatrix n rows columns,
      CoeffBound remainder ((q / q') / 2) ∧
        scaleUpMatrix q (q / q') output = input - reduceMatrix q n rows columns remainder := by
  obtain ⟨hbasis, hne, _, _, hq'1, hlt, _, hdne, hprod, _, _, hout⟩ := h
  set dropped := basis.filter (fun prime ↦ q' % prime != 0)
  have hp : q / q' = dropped.prod := by
    rw [← hprod, Nat.mul_div_cancel_left _ (by omega)]
  set p := q / q' with hpdef
  have hq : q = p * q' := by rw [hp, Nat.mul_comm, hprod]
  have hpgt : 1 < p := by
    by_contra hle
    rcases (show p = 0 ∨ p = 1 by omega) with h0 | h1
    · rw [h0, zero_mul] at hq; omega
    · rw [h1, one_mul] at hq; omega
  have hq1 : 1 < q := by nlinarith
  refine ⟨fun row column ↦ intPoly fun i ↦
      rnsCentered p (Int.ofNat ((input row column).coeff i).val), ?_, ?_⟩
  · intro row column i
    rw [intPoly_coeff hn]
    exact rnsCentered_natAbs_le (by omega) _
  · funext row column
    have hentry := hout row column
    simp only at hentry
    rw [← hp] at hentry
    simp only [scaleUpMatrix, Matrix.sub_apply, reduceMatrix]
    rw [hentry, polynomialOfCoefficients_eq_reduce, scaleUp_reducePoly hq hq'1 hn]
    rw [show ∀ r : ErrorPoly n, input row column - reducePoly q n r =
        reducePoly q n (canonicalLift (input row column) - r) from fun r ↦ by
      rw [map_sub, reducePoly_canonicalLift hq1 hn]]
    rw [reducePoly_eq_iff hq1 hn]
    intro i
    rw [intCast_mul_coeff, intPoly_coeff hn, Negacyclic.coeff_sub, canonicalLift_coeff hn,
      intPoly_coeff hn]
    simpa only [one_mul, Int.ofNat_eq_natCast] using
      blockModSwitch_coefficient hq hpgt (((input row column).coeff i).val : Int)

end ModDown

end MxxExponentLut
