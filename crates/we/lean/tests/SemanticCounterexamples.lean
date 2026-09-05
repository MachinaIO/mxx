import MxxRuntime.Primitives
import MxxBgg.Multiplication

/- Small semantic regression checks, not parameter-security or runtime acceptance tests.
   All matrices inhabit the actual negacyclic quotient, with one row and one column.
   With the existing package dependencies built, run from crates/runtime/lean:
   lake env bash -c 'LEAN_PATH=../../bgg/lean/.lake/build/lib/lean:$LEAN_PATH \
     lean ../../we/lean/tests/SemanticCounterexamples.lean'
   No generated graph, GPU execution, or integration test is needed. -/
namespace SemanticCounterexamples

open Mxx.Primitives Mxx.Bgg MxxRuntime

private abbrev Scalar := ExactPoly 17 1
private abbrev Mat := ExactMatrix 17 1 1 1

private instance : Fact (1 < (17 : Nat)) := ⟨by decide⟩

private theorem matrix_one_ne_zero : (1 : Mat) ≠ 0 := by
  intro h
  have hc := congrArg (fun m : Mat ↦ Negacyclic.coeff (m 0 0) 0) h
  have hone : Negacyclic.coeff (1 : Scalar) 0 = 1 := by
    simpa using (Negacyclic.coeff_root_pow (R := ZMod 17) (by decide)
      (0 : Fin 1) (0 : Fin 1))
  simp [hone] at hc

/-- Here B = K = T = L = X = E = 1, while P = eX = 0.
The actual preimage equation holds, but omitting L * E changes the output. -/
theorem target_error_is_essential :
    RightPreimage (1 : Mat) (1 : Mat) 1 ∧
    (1 : Mat) * 1 = 1 * 0 + (1 * 1 + 0 * 1) ∧
    (1 : Mat) * 1 ≠ 1 * 0 + 0 * 1 := by
  refine ⟨⟨by simp⟩, ?_, ?_⟩
  · exact consume_rectangular (1 : Mat) 1 1 0 1 1 0 1
      (by simp) (by simp) (by simp)
  · simpa using matrix_one_ne_zero

/-- Both messages, secrets, public matrices, and the gadget/decomposition are one.
The left ciphertext/error are zero and the right ciphertext/error are one.
Thus the exact BGG multiplication output is entirely the RHS error term xL • eR. -/
theorem rhs_error_is_essential :
    (0 : Mat) * 1 + (1 : Scalar) • (1 : Mat) =
      1 * (1 * 1) - ((1 : Scalar) * 1) • ((1 : Mat) * 1) +
        ((0 : Mat) * 1 + (1 : Scalar) • (1 : Mat)) ∧
    (0 : Mat) * 1 + (1 : Scalar) • (1 : Mat) ≠
      1 * (1 * 1) - ((1 : Scalar) * 1) • ((1 : Mat) * 1) + (0 : Mat) * 1 := by
  constructor
  · exact multiplication_core
      (gadget := (1 : Mat)) (leftPublic := 1) (rightPublic := 1)
      (decomposition := 1) (leftCiphertext := 0) (rightCiphertext := 1)
      (leftSecret := 1) (rightSecret := 1) (rightPayload := 1)
      (leftMessage := (1 : Scalar)) (rightMessage := 1)
      (leftError := 0) (rightError := 1) (by simp) (by simp) (by simp)
  · simpa using matrix_one_ne_zero

/-- Two individually valid, identically shaped preimage/target pairs cannot be mixed.
The runtime relation rejects either crossed pair, regardless of token or cutoff. -/
theorem same_shape_does_not_authorize_substitution
    {Token : Type} (trapdoor : TrapdoorValue Mat Token) (cutoff : Nat) :
    RightPreimage (1 : Mat) (1 : Mat) 1 ∧ RightPreimage (1 : Mat) (0 : Mat) 0 ∧
    ¬preimageRuns (1 : Mat) trapdoor (1 : Mat) cutoff 0 ∧
    ¬preimageRuns (1 : Mat) trapdoor (0 : Mat) cutoff 1 := by
  refine ⟨⟨by simp⟩, ⟨by simp⟩, ?_, ?_⟩
  · intro h
    have heq := preimageRuns_equation h
    exact matrix_one_ne_zero (by simpa using heq.symm)
  · intro h
    have heq := preimageRuns_equation h
    exact matrix_one_ne_zero (by simpa using heq)

#print axioms target_error_is_essential
#print axioms rhs_error_is_essential
#print axioms same_shape_does_not_authorize_substitution

end SemanticCounterexamples
