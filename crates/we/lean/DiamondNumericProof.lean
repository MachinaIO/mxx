import Bounds
import DiamondProofParameters

namespace DiamondGeneratedProof

open DiamondProofParameters

open MxxWe
open scoped Matrix

set_option maxHeartbeats 600000

/-- Numeric transition for the actual bounded injector recurrence. `inner` is the
preimage inner dimension; it is not the witness digit base or batch width. -/
def injectorStepMatrix (n inner E K : Nat) : BoundMatrix2 :=
  ⟨n, 0, 2 * n * E, inner * n * K⟩

theorem injectorStepMatrix_eq_existing (n ell E K : Nat) :
    injectorStepMatrix n (2 * (ell + 2)) E K = injectorMatrix n ell E K := rfl

private def asMatrix (A : BoundMatrix2) : Matrix (Fin 2) (Fin 2) Nat :=
  !![A.a00, A.a01; A.a10, A.a11]

private theorem asMatrix_mul (A B : BoundMatrix2) :
    asMatrix (MxxWe.matrixMul A B) = asMatrix A * asMatrix B := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [asMatrix, MxxWe.matrixMul, Matrix.mul_apply, Fin.sum_univ_two]

private theorem asMatrix_one : asMatrix matrixOne = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [asMatrix, matrixOne]

/-- Proof-only interpretation of the existing binary matrix-power evaluator. -/
private theorem asMatrix_pow (A : BoundMatrix2) (L : Nat) :
    asMatrix (matrixPow A L) = asMatrix A ^ L := by
  induction L using Nat.strong_induction_on with
  | h L ih =>
    by_cases hz : L = 0
    · simp [matrixPow, hz, asMatrix_one]
    · have hh : L / 2 < L := Nat.div_lt_self (Nat.pos_of_ne_zero hz) (by decide)
      rw [matrixPow]
      simp only [hz, ↓reduceDIte]
      split <;> rename_i hparity
      · rw [asMatrix_mul, ih (L / 2) hh, ← pow_add]
        congr 1
        omega
      · rw [asMatrix_mul, asMatrix_mul, ih (L / 2) hh, ← pow_add, ← pow_succ]
        congr 1
        have hm := Nat.mod_two_eq_zero_or_one L
        omega

/-- A proof reference recurrence. Checker evaluation uses the binary definitions below. -/
def injectorSequence (n inner E K : Nat) : Nat → Nat × Nat
  | 0 => (1, E)
  | L + 1 =>
    let previous := injectorSequence n inner E K L
    (n * previous.1, 2 * n * E * previous.1 + inner * n * K * previous.2)

private theorem injectorSequence_matrix (n inner E K L : Nat) :
    asMatrix (injectorStepMatrix n inner E K) ^ L *ᵥ ![1, E] =
      ![(injectorSequence n inner E K L).1, (injectorSequence n inner E K L).2] := by
  induction L with
  | zero => simp [injectorSequence]
  | succ L ih =>
    rw [pow_succ', ← Matrix.mulVec_mulVec, ih]
    ext i
    fin_cases i <;>
      simp [asMatrix, injectorStepMatrix, injectorSequence, Matrix.mulVec,
        dotProduct, Fin.sum_univ_two]

def binaryInjectorP (n inner E K L : Nat) : Nat :=
  let A := matrixPow (injectorStepMatrix n inner E K) L
  A.a00 + A.a01 * E

def binaryInjectorN (n inner E K L : Nat) : Nat :=
  let A := matrixPow (injectorStepMatrix n inner E K) L
  A.a10 + A.a11 * E

theorem binaryInjector_eq_sequence (n inner E K L : Nat) :
    binaryInjectorP n inner E K L = (injectorSequence n inner E K L).1 ∧
    binaryInjectorN n inner E K L = (injectorSequence n inner E K L).2 := by
  have h := injectorSequence_matrix n inner E K L
  rw [← asMatrix_pow] at h
  constructor
  · simpa [binaryInjectorP, asMatrix, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
      using congrArg (fun v ↦ v 0) h
  · simpa [binaryInjectorN, asMatrix, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
      using congrArg (fun v ↦ v 1) h

/-- These are precisely the four numeric premises of the bounded injector loop,
with arbitrary ring dimension and preimage inner dimension. -/
theorem binaryInjector_recurrence (n inner E K : Nat) :
    binaryInjectorP n inner E K 0 = 1 ∧ binaryInjectorN n inner E K 0 = E ∧
    (∀ L, binaryInjectorP n inner E K (L + 1) = n * binaryInjectorP n inner E K L) ∧
    (∀ L, binaryInjectorN n inner E K (L + 1) =
      2 * n * E * binaryInjectorP n inner E K L +
        inner * n * K * binaryInjectorN n inner E K L) := by
  refine ⟨(binaryInjector_eq_sequence n inner E K 0).1,
    (binaryInjector_eq_sequence n inner E K 0).2, ?_, ?_⟩
  · intro L
    rw [(binaryInjector_eq_sequence n inner E K (L + 1)).1,
      (binaryInjector_eq_sequence n inner E K L).1]
    rfl
  · intro L
    rw [(binaryInjector_eq_sequence n inner E K (L + 1)).2,
      (binaryInjector_eq_sequence n inner E K L).1,
      (binaryInjector_eq_sequence n inner E K L).2]
    rfl

/-- Uniqueness connects any numeric sequences already supplied to the actual loop proof. -/
theorem injector_recurrence_unique (n inner E K : Nat) (P N : Nat → Nat)
    (hP0 : P 0 = 1) (hN0 : N 0 = E)
    (hP : ∀ L, P (L + 1) = n * P L)
    (hN : ∀ L, N (L + 1) = 2 * n * E * P L + inner * n * K * N L) (L : Nat) :
    P L = binaryInjectorP n inner E K L ∧ N L = binaryInjectorN n inner E K L := by
  obtain ⟨hp0, hn0, hp, hn⟩ := binaryInjector_recurrence n inner E K
  induction L with
  | zero => exact ⟨hP0.trans hp0.symm, hN0.trans hn0.symm⟩
  | succ L ih => simp only [hP, hN, hp, hn, ih.1, ih.2, and_self]

theorem binaryInjectorP_eq_binaryPow (n inner E K L : Nat) :
    binaryInjectorP n inner E K L = binaryPow n L := by
  rw [binaryPow_eq_pow]
  obtain ⟨hp0, _, hp, _⟩ := binaryInjector_recurrence n inner E K
  induction L with
  | zero => exact hp0
  | succ L ih => rw [hp, ih, pow_succ']

/-- The exact factor ordering expected by `generated_bounded_injector_loop`. -/
theorem fixture_binaryInjector_loop_premises (E K : Nat) :
    binaryInjectorP n inner E K 0 = 1 ∧ binaryInjectorN n inner E K 0 = E ∧
    (∀ L, binaryInjectorP n inner E K (L + 1) = n * binaryInjectorP n inner E K L) ∧
    (∀ L, binaryInjectorN n inner E K (L + 1) =
      2 * n * binaryInjectorP n inner E K L * E +
        projection * binaryInjectorN n inner E K L * K) := by
  simpa only [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using
    binaryInjector_recurrence n inner E K

theorem projected_factor_order (inner n K N : Nat) :
    inner * n * N * K = initialErrorBound inner n K N := by
  unfold initialErrorBound
  ring

/-- The circuit layer's scalar recurrence is the existing raw layer bound. -/
theorem circuit_recurrence_eq_raw (factor B0 : Nat) (B : Nat → Nat)
    (hzero : B 0 = B0) (hstep : ∀ layer, B (layer + 1) = factor * B layer) (H : Nat) :
    B H = rawLayerBound factor B0 H := by
  unfold rawLayerBound
  induction H with
  | zero => simpa using hzero
  | succ H ih => rw [hstep, ih, pow_succ']; ac_rfl

/-- Numeric composition of the two recurrences with the final algebraic error formula.
Every premise is a recurrence of natural numbers, not an execution/noise conclusion. -/
theorem final_bound_from_recurrences (n inner ell E K D L H : Nat)
    (P N B : Nat → Nat) (hP0 : P 0 = 1) (hN0 : N 0 = E)
    (hP : ∀ layer, P (layer + 1) = n * P layer)
    (hN : ∀ layer, N (layer + 1) =
      2 * n * E * P layer + inner * n * K * N layer)
    (hB0 : B 0 = initialErrorBound inner n K (N L))
    (hB : ∀ layer, B (layer + 1) = layerFactor ell n D * B layer) :
    2 * initialErrorBound inner n K (N L) +
        (ell * n * D) * (initialErrorBound inner n K (N L) + B H) =
      rawFinalBound (initialErrorBound inner n K (binaryInjectorN n inner E K L))
        (ell * n * D) (layerFactor ell n D) H := by
  rw [circuit_recurrence_eq_raw _ _ B hB0 hB H]
  rw [(injector_recurrence_unique n inner E K P N hP0 hN0 hP hN L).2]
  rfl

def cappedInjectorN (C n inner E K L : Nat) : Nat :=
  let A := matrixPowCap C (injectorStepMatrix n inner E K) L
  cadd C A.a10 (cmul C A.a11 (cap C E))

theorem cappedInjectorN_eq_cap (C n inner E K L : Nat) :
    cappedInjectorN C n inner E K L = cap C (binaryInjectorN n inner E K L) := by
  unfold cappedInjectorN binaryInjectorN
  rw [matrixPowCap_eq_cap_matrixPow]
  simp only [capMatrix, cadd, cmul]
  rw [← cap_mul_cap, ← cap_add_cap]

def projectedInjectorBound (n inner E K L : Nat) : Nat :=
  initialErrorBound inner n K (binaryInjectorN n inner E K L)

def cappedProjectedInjectorBound (C n inner E K L : Nat) : Nat :=
  cmul C (cap C (inner * n * K)) (cappedInjectorN C n inner E K L)

theorem cappedProjectedInjectorBound_eq_cap (C n inner E K L : Nat) :
    cappedProjectedInjectorBound C n inner E K L =
      cap C (projectedInjectorBound n inner E K L) := by
  unfold cappedProjectedInjectorBound projectedInjectorBound initialErrorBound cmul
  rw [cappedInjectorN_eq_cap, ← cap_mul_cap]

/-- The capped checker computes only binary matrix powers and binary circuit-layer powers. -/
def cappedDiamondBound (C n inner ell E K D L H : Nat) : Nat :=
  cappedFinalBound C (cappedProjectedInjectorBound C n inner E K L)
    (ell * n * D) (layerFactor ell n D) H

theorem cappedDiamondBound_eq_cap (C n inner ell E K D L H : Nat) :
    cappedDiamondBound C n inner ell E K D L H =
      cap C (rawFinalBound (projectedInjectorBound n inner E K L)
        (ell * n * D) (layerFactor ell n D) H) := by
  unfold cappedDiamondBound
  rw [cappedProjectedInjectorBound_eq_cap]
  have hid (x : Nat) : cap C (cap C x) = cap C x := by simp [cap]
  have hc (x a factor : Nat) : cappedFinalBound C (cap C x) a factor H =
      cappedFinalBound C x a factor H := by
    simp only [cappedFinalBound, cappedLayerBound, hid]
  rw [hc, cappedFinalBound_eq_cap_raw]

/-- Threshold equivalence only: no candidate is asserted to pass this gate. -/
theorem cappedDiamondBound_lt_iff (C n inner ell E K D L H : Nat) :
    cappedDiamondBound C n inner ell E K D L H < C ↔
      rawFinalBound (projectedInjectorBound n inner E K L)
        (ell * n * D) (layerFactor ell n D) H < C := by
  rw [cappedDiamondBound_eq_cap, cap_lt_iff]

#print axioms binaryInjector_recurrence
#print axioms injector_recurrence_unique
#print axioms binaryInjectorP_eq_binaryPow
#print axioms fixture_binaryInjector_loop_premises
#print axioms final_bound_from_recurrences
#print axioms cappedInjectorN_eq_cap
#print axioms cappedProjectedInjectorBound_eq_cap
#print axioms cappedDiamondBound_eq_cap
#print axioms cappedDiamondBound_lt_iff

end DiamondGeneratedProof
