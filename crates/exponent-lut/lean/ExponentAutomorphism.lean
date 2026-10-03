import PrimitivesBounds
import RuntimeMatrixOps
import RuntimeLemmas

/-!
# Ring automorphisms

The runtime substitutes `X ↦ X^index` coefficient by coefficient (`MxxRuntime.ringAutomorphism`).
For an odd index this is the ring automorphism of `ℤ_q[X]/(X^n + 1)` that sends the root to its
`index`-th power. This module identifies the runtime operation with that ring homomorphism, gives
the matching homomorphism on integer polynomials, and shows that it commutes with reduction and
preserves coefficient bounds.
-/

namespace MxxExponentLut

open Mxx.Primitives
open scoped BigOperators

variable {q n : Nat}

private theorem root_pow_odd_pow_n {R : Type} [CommRing R] {index : Nat} (hodd : Odd index) :
    (AdjoinRoot.root (negacyclicModulus n R) ^ index) ^ n = -1 := by
  rw [← pow_mul, Nat.mul_comm index n, pow_mul, Negacyclic.root_pow_n, hodd.neg_one_pow]

private theorem negacyclic_eval_root_pow {R : Type} [CommRing R] {index : Nat} (hodd : Odd index) :
    Polynomial.eval₂ (algebraMap R (Negacyclic n R))
      (AdjoinRoot.root (negacyclicModulus n R) ^ index) (negacyclicModulus n R) = 0 := by
  simpa [negacyclicModulus, Polynomial.eval₂_add, Polynomial.eval₂_pow] using
    congrArg (fun x : Negacyclic n R => x + 1) (root_pow_odd_pow_n (n := n) (R := R) hodd)

/-- The substitution `X ↦ X^index` of `R[X]/(X^n + 1)` for an odd index. -/
noncomputable def substitution {R : Type} [CommRing R] (index : Nat) (hodd : Odd index) :
    Negacyclic n R →+* Negacyclic n R :=
  AdjoinRoot.lift (algebraMap R (Negacyclic n R))
    (AdjoinRoot.root (negacyclicModulus n R) ^ index) (negacyclic_eval_root_pow hodd)

@[simp] theorem substitution_root {R : Type} [CommRing R] (index : Nat) (hodd : Odd index) :
    substitution (n := n) (R := R) index hodd (AdjoinRoot.root (negacyclicModulus n R)) =
      AdjoinRoot.root (negacyclicModulus n R) ^ index := by
  simp [substitution]

@[simp] theorem substitution_algebraMap {R : Type} [CommRing R] (index : Nat) (hodd : Odd index)
    (a : R) :
    substitution (n := n) (R := R) index hodd (algebraMap R (Negacyclic n R) a) =
      algebraMap R (Negacyclic n R) a := by
  simp [substitution]

@[simp] theorem substitution_intCast {R : Type} [CommRing R] (index : Nat) (hodd : Odd index)
    (a : Int) :
    substitution (n := n) (R := R) index hodd (a : Negacyclic n R) = (a : Negacyclic n R) :=
  map_intCast _ a

/-- The substitution fixes every constant polynomial. -/
theorem substitution_root_pow {R : Type} [CommRing R] (index : Nat) (hodd : Odd index)
    (k : Nat) :
    substitution (n := n) (R := R) index hodd (AdjoinRoot.root (negacyclicModulus n R) ^ k) =
      AdjoinRoot.root (negacyclicModulus n R) ^ (k * index) := by
  rw [map_pow, substitution_root, ← pow_mul, Nat.mul_comm]

/-- Substitution commutes with reduction modulo `q`. -/
theorem reducePoly_substitution (index : Nat) (hodd : Odd index) (x : ErrorPoly n) :
    reducePoly q n (substitution index hodd x) = substitution index hodd (reducePoly q n x) := by
  have h : (reducePoly q n).comp (substitution (n := n) (R := Int) index hodd) =
      (substitution (n := n) (R := ZMod q) index hodd).comp (reducePoly q n) := by
    apply AdjoinRoot.ringHom_ext
    · ext a
      simp [reducePoly]
    · simp [reducePoly, substitution]
  exact congrArg (fun f : ErrorPoly n →+* ExactPoly q n => f x) h

private theorem canonical_coefficient_cast (hq : 1 < q) (a : ZMod q) :
    (Int.ofNat a.val : ExactPoly q n) = algebraMap (ZMod q) (ExactPoly q n) a := by
  letI : NeZero q := ⟨by omega⟩
  have h := congrArg (algebraMap (ZMod q) (ExactPoly q n)) (ZMod.natCast_zmod_val a)
  simpa only [map_natCast, Int.ofNat_eq_natCast, Int.cast_natCast] using h

/-- The runtime operation is the substitution applied to every entry. -/
theorem ringAutomorphism_eq (hq : 1 < q) (hn : 0 < n) {rows columns : Nat} (index : Nat)
    (hodd : Odd index) (input : ExactMatrix q n rows columns) :
    MxxRuntime.ringAutomorphism index input =
      fun row column => substitution index hodd (input row column) := by
  letI : Fact (1 < q) := ⟨hq⟩
  ext row column
  conv_rhs => rw [Negacyclic.expansion hn (input row column)]
  simp only [MxxRuntime.ringAutomorphism, map_sum, map_mul, canonical_coefficient_cast hq,
    substitution_algebraMap, substitution_root_pow]

/-- The runtime automorphism relation at an odd in-range index. -/
theorem ringAutomorphismRuns_eq (hq : 1 < q) (hn : 0 < n) {rows columns : Nat} {index : Int}
    {input output : ExactMatrix q n rows columns}
    (h : MxxRuntime.ringAutomorphismRuns index input output) :
    ∃ hodd : Odd index.toNat,
      output = fun row column => substitution index.toNat hodd (input row column) := by
  obtain ⟨hpos, _, hparity, hout⟩ := h
  have hodd : Odd index.toNat := Nat.odd_iff.mpr (by omega)
  exact ⟨hodd, hout.trans (ringAutomorphism_eq hq hn _ hodd input)⟩

/-- Integer substitution permutes coefficients up to sign: the coefficient sum form. -/
private theorem substitution_coeff_sum (index : Nat) (hodd : Odd index) (hn : 0 < n)
    (x : ErrorPoly n) :
    substitution index hodd x = ∑ i : Fin n, algebraMap Int (ErrorPoly n) (x.coeff i) *
      AdjoinRoot.root (negacyclicModulus n Int) ^ (i.val * index) := by
  conv_lhs => rw [Negacyclic.expansion hn x]
  simp only [map_sum, map_mul, substitution_algebraMap, substitution_root_pow]

private theorem root_power_remainder {R : Type} [CommRing R] (exponent : Nat) :
    AdjoinRoot.root (negacyclicModulus n R) ^ exponent =
      algebraMap Int (Negacyclic n R) ((-1 : Int) ^ (exponent / n)) *
        AdjoinRoot.root (negacyclicModulus n R) ^ (exponent % n) := by
  conv_lhs => rw [← Nat.div_add_mod exponent n]
  rw [pow_add, pow_mul, Negacyclic.root_pow_n]
  simp

private theorem substituted_term_coeff (hn : 0 < n) (a : Int) (exponent : Nat) (k : Fin n) :
    (algebraMap Int (ErrorPoly n) a *
      AdjoinRoot.root (negacyclicModulus n Int) ^ exponent).coeff k =
      if exponent % n = k.val then a * (-1 : Int) ^ (exponent / n) else 0 := by
  rw [root_power_remainder]
  rw [← mul_assoc, ← map_mul, Negacyclic.coeff_smul]
  have hr := Negacyclic.coeff_root_pow (R := Int) hn
    (⟨exponent % n, Nat.mod_lt _ hn⟩ : Fin n) k
  rw [hr]
  simp [Fin.ext_iff]

private theorem multiply_index_injective {index : Nat} (hcoprime : n.Coprime index) :
    Function.Injective (fun i : Fin n => (i.val * index) % n) := by
  intro i j hij
  have h := Nat.ModEq.cancel_right_of_coprime hcoprime hij
  apply Fin.ext
  simpa [Nat.ModEq, Nat.mod_eq_of_lt i.isLt, Nat.mod_eq_of_lt j.isLt] using h

/-- A substitution with an index coprime to `n` does not increase the coefficient bound. -/
theorem polyNorm_substitution_le (hn : 0 < n) {index : Nat} (hodd : Odd index)
    (hcoprime : n.Coprime index) (x : ErrorPoly n) :
    polyNorm (substitution index hodd x) ≤ polyNorm x := by
  rw [substitution_coeff_sum index hodd hn]
  apply Finset.sup_le
  intro k hk
  rw [Negacyclic.coeff_sum]
  simp_rw [substituted_term_coeff hn]
  by_cases hex : ∃ i : Fin n, (i.val * index) % n = k.val
  · obtain ⟨i, hi⟩ := hex
    rw [Finset.sum_eq_single i]
    · simp only [hi, if_true, Int.natAbs_mul, Int.natAbs_pow, Int.natAbs_neg,
        Int.natAbs_one, one_pow, mul_one]
      exact coeff_natAbs_le_polyNorm x i
    · intro j hj hji
      have hneq : (j.val * index) % n ≠ k.val := by
        intro heq
        exact hji (multiply_index_injective hcoprime (heq.trans hi.symm))
      simp [hneq]
    · simp
  · have hnone (i : Fin n) : (i.val * index) % n ≠ k.val := by
      intro h; exact hex ⟨i, h⟩
    simp [hnone]

/-- Odd indices are coprime to power-of-two dimensions. -/
theorem coprime_of_odd_pow_two {log index : Nat} (hodd : Odd index) :
    (2 ^ log).Coprime index :=
  hodd.coprime_two_left.pow_left log

end MxxExponentLut
