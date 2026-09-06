import PrimitivesBounds
import RuntimeMatrixOps
import Mathlib.Data.Nat.ModEq

namespace MxxExponentLut
open Mxx.Primitives
open scoped BigOperators

/-- Integer substitution with exactly the runtime's root-power convention. -/
noncomputable def errorAutomorphism {n : Nat} (index : Nat) (input : ErrorPoly n) :
    ErrorPoly n :=
  ∑ i : Fin n, algebraMap Int (ErrorPoly n) (input.coeff i) *
    AdjoinRoot.root (negacyclicModulus n Int) ^ (i.val * index)

private theorem root_power_remainder {n : Nat} (exponent : Nat) :
    AdjoinRoot.root (negacyclicModulus n Int) ^ exponent =
      algebraMap Int (ErrorPoly n) ((-1 : Int) ^ (exponent / n)) *
        AdjoinRoot.root (negacyclicModulus n Int) ^ (exponent % n) := by
  conv_lhs => rw [← Nat.div_add_mod exponent n]
  rw [pow_add, pow_mul, Negacyclic.root_pow_n]
  simp

private theorem substituted_term_coeff {n : Nat} (hn : 0 < n)
    (a : Int) (exponent : Nat) (k : Fin n) :
    (algebraMap Int (ErrorPoly n) a *
      AdjoinRoot.root (negacyclicModulus n Int) ^ exponent).coeff k =
      if exponent % n = k.val then a * (-1 : Int) ^ (exponent / n) else 0 := by
  rw [root_power_remainder]
  rw [← mul_assoc, ← map_mul, Negacyclic.coeff_smul]
  have hr := Negacyclic.coeff_root_pow (R := Int) hn
    (⟨exponent % n, Nat.mod_lt _ hn⟩ : Fin n) k
  rw [hr]
  simp [Fin.ext_iff]

private theorem multiply_index_injective {n index : Nat} (hcoprime : n.Coprime index) :
    Function.Injective (fun i : Fin n => (i.val * index) % n) := by
  intro i j hij
  have h := Nat.ModEq.cancel_right_of_coprime hcoprime hij
  apply Fin.ext
  simpa [Nat.ModEq, Nat.mod_eq_of_lt i.isLt, Nat.mod_eq_of_lt j.isLt] using h

/-- A coprime index permutes coefficients and changes only their signs. -/
theorem errorAutomorphism_norm_le {n index : Nat} (hn : 0 < n)
    (hcoprime : n.Coprime index) (input : ErrorPoly n) :
    polyNorm (errorAutomorphism index input) ≤ polyNorm input := by
  apply Finset.sup_le
  intro k hk
  rw [errorAutomorphism, Negacyclic.coeff_sum]
  simp_rw [substituted_term_coeff hn]
  by_cases hex : ∃ i : Fin n, (i.val * index) % n = k.val
  · obtain ⟨i, hi⟩ := hex
    rw [Finset.sum_eq_single i]
    · simp only [hi, if_true, Int.natAbs_mul, Int.natAbs_pow, Int.natAbs_neg,
        Int.natAbs_one, one_pow, mul_one]
      exact coeff_natAbs_le_polyNorm input i
    · intro j hj hji
      have hneq : (j.val * index) % n ≠ k.val := by
        intro heq
        exact hji (multiply_index_injective hcoprime (heq.trans hi.symm))
      simp [hneq]
    · simp
  · have hnone (i : Fin n) : (i.val * index) % n ≠ k.val := by
      intro h; exact hex ⟨i, h⟩
    simp [hnone]

/-- The runtime's odd indices are coprime to every power-of-two degree. -/
theorem errorAutomorphism_power_two_norm_le (log index : Nat) (hodd : Odd index)
    (input : ErrorPoly (2 ^ log)) :
    polyNorm (errorAutomorphism index input) ≤ polyNorm input := by
  apply errorAutomorphism_norm_le (by positivity)
  exact hodd.coprime_two_left.pow_left log

/-- Reduction of the integer substitution agrees with the actual runtime
operation on reduced inputs, including canonical residue representatives. -/
theorem errorAutomorphism_reduce {q n rows columns : Nat} (hq : 1 < q) (hn : 0 < n)
    (index : Nat) (input : ErrorMatrix n rows columns) :
    reduceMatrix q n rows columns (fun row column => errorAutomorphism index (input row column)) =
      MxxRuntime.ringAutomorphism index (reduceMatrix q n rows columns input) := by
  letI : NeZero q := ⟨by omega⟩
  ext row column
  simp only [reduceMatrix, errorAutomorphism, MxxRuntime.ringAutomorphism, map_sum, map_mul,
    map_pow, reducePoly_coeff hq hn]
  apply Finset.sum_congr rfl
  intro i hi
  have hcast (a : Int) :
      ((a : ZMod q).val : ExactPoly q n) = (a : ExactPoly q n) := by
    have h := congrArg (algebraMap (ZMod q) (ExactPoly q n)) (ZMod.natCast_zmod_val (a : ZMod q))
    simpa only [map_natCast, map_intCast] using h
  simpa [reducePoly] using
    congrArg (fun a : ExactPoly q n => a *
      AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (i.val * index))
      (hcast ((input row column).coeff i)).symm

/-- Canonical residue representatives disappear under the coefficient algebra map. -/
private theorem canonical_coefficient_cast {q n : Nat} (hq : 1 < q) (a : ZMod q) :
    (Int.ofNat a.val : ExactPoly q n) = algebraMap (ZMod q) (ExactPoly q n) a := by
  letI : NeZero q := ⟨by omega⟩
  have h := congrArg (algebraMap (ZMod q) (ExactPoly q n)) (ZMod.natCast_zmod_val a)
  simpa only [map_natCast, Int.ofNat_eq_natCast, Int.cast_natCast] using h

theorem ringAutomorphism_add {q n rows columns : Nat} (hq : 1 < q) (index : Nat)
    (left right : ExactMatrix q n rows columns) :
    MxxRuntime.ringAutomorphism index (left + right) =
      MxxRuntime.ringAutomorphism index left + MxxRuntime.ringAutomorphism index right := by
  ext row column
  simp only [MxxRuntime.ringAutomorphism, Matrix.add_apply, Negacyclic.coeff_add,
    canonical_coefficient_cast hq, map_add, add_mul, Finset.sum_add_distrib]

/-- Actual runtime automorphism preserves the coefficient error budget. -/
theorem approx_ringAutomorphism {q n rows columns index bound : Nat}
    (hq : 1 < q) (hn : 0 < n) (hcoprime : n.Coprime index)
    {actual ideal : ExactMatrix q n rows columns} (h : Approx actual ideal bound) :
    Approx (MxxRuntime.ringAutomorphism index actual)
      (MxxRuntime.ringAutomorphism index ideal) bound := by
  obtain ⟨error, heq, herror⟩ := h
  refine ⟨fun row column => errorAutomorphism index (error row column), ?_, ?_⟩
  · rw [heq, ringAutomorphism_add hq, errorAutomorphism_reduce hq hn]
  · intro row column coefficient
    apply (coeff_natAbs_le_polyNorm _ _).trans
    apply (errorAutomorphism_norm_le hn hcoprime _).trans
    exact Finset.sup_le (fun k _ => herror row column k)

/-- Exported odd-index execution on a power-of-two ring has unit noise gain. -/
theorem approx_ringAutomorphismRuns {q log rows columns bound : Nat}
    (hq : 1 < q) {index : Int} {actual ideal output : ExactMatrix q (2 ^ log) rows columns}
    (h : Approx actual ideal bound) (hrun : MxxRuntime.ringAutomorphismRuns index actual output) :
    Approx output (MxxRuntime.ringAutomorphism index.toNat ideal) bound := by
  obtain ⟨hpos, hlt, hodd, hout⟩ := hrun
  rw [hout]
  apply approx_ringAutomorphism hq (by positivity) _ h
  have ho : Odd index.toNat := by
    apply Nat.odd_iff.mpr
    omega
  exact ho.coprime_two_left.pow_left log

end MxxExponentLut
