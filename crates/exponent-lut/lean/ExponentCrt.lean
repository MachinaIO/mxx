import ExponentTransfers
import RuntimeMatrixOps

namespace MxxExponentLut

open Mxx.Primitives
open scoped BigOperators

/-- A CRT reconstruction weight annihilates multiples of its plaintext modulus.
Consequently taking the canonical plaintext residue does not change its weighted
contribution in the destination ring. -/
theorem crt_weighted_residue {destination : Nat} (coefficient plaintext value : Int)
    (annihilates : (coefficient : ZMod destination) * (plaintext : ZMod destination) = 0) :
    (coefficient : ZMod destination) * ((value % plaintext : Int) : ZMod destination) =
      (coefficient : ZMod destination) * (value : ZMod destination) := by
  have equation := congrArg (fun z : Int ↦ (coefficient : ZMod destination) *
    (z : ZMod destination)) (Int.mul_ediv_add_emod value plaintext)
  simpa only [Int.cast_add, Int.cast_mul, mul_add, ← mul_assoc, annihilates,
    zero_mul, zero_add] using equation

private theorem exact_poly_eq_of_coeff_eq {q n : Nat} (hq : 1 < q) (hn : 0 < n)
    {left right : ExactPoly q n}
    (equal : ∀ coefficient : Fin n, left.coeff coefficient = right.coeff coefficient) :
    left = right := by
  letI : Fact (1 < q) := ⟨hq⟩
  rw [Negacyclic.expansion hn left, Negacyclic.expansion hn right]
  apply Finset.sum_congr rfl
  intro i _
  rw [equal i]

/-- Exact reconstruction from the actual heterogeneous runtime operands. Each
slot rounds using its own source modulus, as in `crt_recompose_cpu`; only the
resulting plaintext residue is assumed correct. The CRT hypotheses are finite
identities of the concrete reconstruction table, not correctness of the output. -/
theorem crt_recompose_of_rounded_residues
    {destination n columns slots : Nat}
    (hq : 1 < destination) (hn : 0 < n)
    (source : Fin slots → Nat) (plaintext coefficient : Fin slots → Int)
    (input : (slot : Fin slots) → ExactMatrix (source slot) n 1 columns)
    (target : ErrorMatrix n 1 columns)
    (annihilates : ∀ slot, (coefficient slot : ZMod destination) *
      (plaintext slot : ZMod destination) = 0)
    (partition : (∑ slot : Fin slots, (coefficient slot : ZMod destination)) = 1)
    (rounded : ∀ slot row column (i : Fin n),
      ((plaintext slot * Int.ofNat ((input slot row column).coeff i).val +
        Int.ofNat (source slot / 2)) / Int.ofNat (source slot)) % plaintext slot =
      (target row column).coeff i % plaintext slot) :
    (∑ slot : Fin slots, MxxRuntime.crtRecomposeLevel
      (destination := destination) (plaintext slot) (coefficient slot) (input slot)) =
      reduceMatrix destination n 1 columns target := by
  apply exactMatrix_ext
  intro row column
  apply exact_poly_eq_of_coeff_eq hq hn
  intro i
  simp only [Matrix.sum_apply, MxxRuntime.crtRecomposeLevel]
  rw [Negacyclic.coeff_sum]
  have hcast (value : Int) : (value : ExactPoly destination n) =
      algebraMap (ZMod destination) (ExactPoly destination n) (value : ZMod destination) := by
    simp
  simp_rw [hcast, Negacyclic.coeff_smul,
    MxxRuntime.polynomialOfCoefficients_coeff hq hn, rounded]
  simp_rw [crt_weighted_residue _ _ _ (annihilates _)]
  rw [← Finset.sum_mul, partition, one_mul]
  exact (reducePoly_coeff hq hn (target row column) i).symm

/-- Once every source level's strict decoder recovers the coefficient of the
anchor plus fresh error, the final runtime CRT sum has precisely that fresh
error. This matches verification's `encoding.vector - anchor` endpoint. -/
theorem crt_recompose_fresh_error
    {destination n columns slots bound : Nat}
    (hq : 1 < destination) (hn : 0 < n)
    (source : Fin slots → Nat) (plaintext coefficient : Fin slots → Int)
    (input : (slot : Fin slots) → ExactMatrix (source slot) n 1 columns)
    (anchor fresh : ErrorMatrix n 1 columns)
    (actual : ExactMatrix destination n 1 columns)
    (runs : actual = ∑ slot : Fin slots, MxxRuntime.crtRecomposeLevel
      (plaintext slot) (coefficient slot) (input slot))
    (annihilates : ∀ slot, (coefficient slot : ZMod destination) *
      (plaintext slot : ZMod destination) = 0)
    (partition : (∑ slot : Fin slots, (coefficient slot : ZMod destination)) = 1)
    (rounded : ∀ slot row column (i : Fin n),
      ((plaintext slot * Int.ofNat ((input slot row column).coeff i).val +
        Int.ofNat (source slot / 2)) / Int.ofNat (source slot)) % plaintext slot =
      ((anchor + fresh) row column).coeff i % plaintext slot)
    (freshBounded : CoeffBound fresh bound) :
    Approx actual (reduceMatrix destination n 1 columns anchor) bound := by
  refine ⟨fresh, ?_, freshBounded⟩
  rw [runs, crt_recompose_of_rounded_residues hq hn source plaintext coefficient input
    (anchor + fresh) annihilates partition rounded, reduceMatrix_add]

end MxxExponentLut
