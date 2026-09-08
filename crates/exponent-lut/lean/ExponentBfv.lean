import RuntimeMatrixOps
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Ring

namespace MxxExponentLut
open Mxx.Primitives
open scoped BigOperators

/-- Equal-level addition preserves the BFV relation with the sum of errors
and plaintext polynomials; no monomial premise is needed. -/
theorem bfv_add_relation {q n : Nat}
    (mask payload leftError rightError : ExactMatrix q n 1 2)
    (leftKey rightKey : ExactMatrix q n 2 2)
    (scale leftMessage rightMessage : ExactPoly q n) :
    (mask * leftKey - (scale * leftMessage) • payload + leftError) +
      (mask * rightKey - (scale * rightMessage) • payload + rightError) =
    mask * (leftKey + rightKey) - (scale * (leftMessage + rightMessage)) • payload +
      (leftError + rightError) := by
  simp only [Matrix.mul_add, mul_add, add_smul]
  abel

/-- Multiplication by a known public polynomial acts on the public matrix,
message, and error at the same level. No public ciphertext error enters. -/
theorem bfv_public_mul_relation {q n : Nat}
    (mask payload error : ExactMatrix q n 1 2) (key : ExactMatrix q n 2 2)
    (scale message multiplier : ExactPoly q n) :
    multiplier • (mask * key - (scale * message) • payload + error) =
    mask * (multiplier • key) - (scale * (multiplier * message)) • payload +
      (multiplier • error) := by
  simp only [smul_add, smul_sub, smul_smul, Matrix.mul_smul,
    mul_left_comm, mul_comm]

/-- BFV note (6.3)-(6.4), before the outer block switch. All quantities are in the
source ring. `rotatedMask`, `rotatedPayload`, and `rotatedError` are the
substituted input components; `residual` is the internal switch residual.
The premise is the exact block-switch equation, not an output invariant. -/
theorem bfv_branch_cancellation {q n : Nat}
    (mask rotatedMask rotatedPayload helperError rotatedError : ExactMatrix q n 1 2)
    (helperKey rotatedKey ciphertext intermediate residual : ExactMatrix q n 2 2)
    (scale message : ExactPoly q n)
    (rounding : rotatedKey * ciphertext - scale • intermediate =
      residual) :
    (mask * helperKey - scale • rotatedMask + helperError) * intermediate +
      (rotatedMask * rotatedKey - (scale * message) • rotatedPayload +
        rotatedError) * ciphertext =
    mask * (helperKey * intermediate) - (scale * message) • (rotatedPayload * ciphertext) +
      (helperError * intermediate + rotatedError * ciphertext +
        rotatedMask * residual) := by
  have hr : rotatedKey * ciphertext = residual + scale • intermediate :=
    sub_eq_iff_eq_add.mp rounding
  simp only [Matrix.add_mul, Matrix.sub_mul, Matrix.smul_mul, Matrix.mul_assoc,
    hr, Matrix.mul_add, Matrix.mul_smul]
  abel

/-- The fresh RHS equation initially holds in the next modulus. Multiplying
by the dropped scale kills its integer wrap in the source modulus, exactly as
in the BFV note, section 6. The fresh error retains the factor `scale * message`. -/
theorem bfv_fresh_rhs_substitution {q n : Nat}
    (rotatedPayloadProduct target freshError wrap : ExactMatrix q n 1 2)
    (scale nextModulus message : ExactPoly q n)
    (modulusProduct : scale * nextModulus = 0)
    (fresh : rotatedPayloadProduct = target + freshError +
      nextModulus • wrap) :
    (scale * message) • rotatedPayloadProduct =
      (scale * message) • target + ((scale * message) • freshError) := by
  have hwrap : (scale * message) * nextModulus = 0 := by
    calc
      _ = message * (scale * nextModulus) := by ring
      _ = 0 := by rw [modulusProduct, mul_zero]
  simp only [fresh, smul_add, smul_smul, hwrap, zero_smul, add_zero]

/-- Aggregation leaves one public accumulator and one state accumulator.
In particular, this statement applies before a single outer block switch;
it does not replace the aggregate by a sum of individually rounded outputs. -/
theorem bfv_aggregate_relation {q n count : Nat}
    (mask : ExactMatrix q n 1 2)
    (states targets errors : Fin count → ExactMatrix q n 1 2)
    (keys : Fin count → ExactMatrix q n 2 2)
    (branches : ∀ i, states i = mask * keys i - targets i + errors i) :
    (∑ i, states i) = mask * (∑ i, keys i) - (∑ i, targets i) +
      (∑ i, errors i) := by
  simp only [branches, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    Matrix.mul_sum]

/-- Outer public/state switching contributes exactly one residual for each
aggregate. This is the scaled algebraic relation; coefficient
integrality are separate obligations when cancelling the scale. -/
theorem bfv_outer_switch_relation {q n : Nat}
    (mask stateRaw stateOut target errorRaw stateResidual : ExactMatrix q n 1 2)
    (publicRaw publicOut publicResidual : ExactMatrix q n 2 2)
    (scale : ExactPoly q n)
    (relation : stateRaw = mask * publicRaw - scale • target + errorRaw)
    (stateSwitch : scale • stateOut = stateRaw - stateResidual)
    (publicSwitch : scale • publicOut = publicRaw - publicResidual) :
    scale • (stateOut - mask * publicOut + target) =
      (errorRaw + mask * publicResidual - stateResidual) := by
  rw [smul_add, smul_sub, ← Matrix.mul_smul, stateSwitch, publicSwitch, relation]
  simp only [Matrix.mul_sub]
  abel

/-- Coefficientwise integrality for ordinary ModDown. The source wrap is
an explicit multiple of scale * nextModulus; no plaintext modulus is needed. -/
theorem bfv_error_numerator_divisible
    (scale nextModulus outputDifference numerator wrap : Int)
    (equation : scale * outputDifference = numerator + scale * nextModulus * wrap) :
    scale ∣ numerator := by
  refine ⟨outputDifference - nextModulus * wrap, ?_⟩
  calc
    numerator = scale * outputDifference - scale * nextModulus * wrap :=
      eq_sub_of_add_eq equation.symm
    _ = scale * (outputDifference - nextModulus * wrap) := by ring

/-- The masked combined target has a direct two-column decoder. Its fresh
error is unscaled and remains part of the low-order rounding residual. -/
theorem bfv_direct_decoder_relation {q n : Nat}
    (mask payload fresh lowError decoderError : ExactMatrix q n 1 2)
    (combined output : ExactMatrix q n 2 2) (spacing : ExactPoly q n) :
    (mask * combined + spacing • payload + spacing • fresh + lowError) -
      (mask * (combined - spacing • output) + decoderError) =
    spacing • (mask * output + payload + fresh) + (lowError - decoderError) := by
  simp only [Matrix.mul_sub, Matrix.mul_smul, smul_add]
  abel

/-- Fresh target weighting leaves the encryption error unchanged. Adding the
public affine offset is distinct from scaling a completed noisy ciphertext. -/
theorem bfv_affine_target_readout {q n : Nat}
    (mask error digit offset : ExactMatrix q n 1 2)
    (key : ExactMatrix q n 2 2) (weight : ExactPoly q n) :
    (mask * key + weight • (digit - offset) + error) + weight • offset =
      mask * key + weight • digit + error := by
  rw [smul_sub]
  abel

/-- A strict low-order mask/error interval is the exact centered remainder.
This includes negative coefficients and does not assume a positive message. -/
theorem bfv_refresh_centered_residual (spacing : Nat) (message error : Int)
    (hs : 0 < spacing) (he : 2 * |error| < (spacing : Int)) :
    MxxRuntime.centeredResidue spacing ((spacing : Int) * message + error) = error := by
  have hs' : (0 : Int) < spacing := by exact_mod_cast hs
  have hlo := neg_abs_le error
  have hhi := le_abs_self error
  have hmod : ((spacing : Int) * message + error) % spacing = error % spacing := by simp
  unfold MxxRuntime.centeredResidue
  simp only [Int.ofNat_eq_natCast, Int.natCast_ediv, Nat.cast_ofNat, hmod]
  by_cases hn : error < 0
  · have hr : error % spacing = error + spacing := by
      have h := Int.emod_eq_of_lt (show 0 ≤ error + (spacing : Int) by omega)
        (show error + (spacing : Int) < spacing by omega)
      simpa using h
    rw [hr, if_pos (by omega)]
    omega
  · rw [Int.emod_eq_of_lt (by omega) (by omega), if_neg (by omega)]

/-- Ordinary ModDown removes the old error and mask exactly after direct
decoder subtraction. The remaining message includes the desired fresh error.
This is the integer coefficient formula used by the residue-only runtime. -/
theorem bfv_refresh_exact_moddown (spacing : Nat) (message error : Int)
    (hs : 0 < spacing) (he : 2 * |error| < (spacing : Int)) :
    (((spacing : Int) * message + error -
      MxxRuntime.centeredResidue spacing ((spacing : Int) * message + error)) /
      (spacing : Int)) = message := by
  rw [bfv_refresh_centered_residual spacing message error hs he, add_sub_cancel_right]
  exact Int.mul_ediv_cancel_left message (by exact_mod_cast Nat.ne_of_gt hs)

/-- Changing an integer lift by the full modulus H*pi does not change the
extracted residue. CRT assembly therefore recovers the same fresh state. -/
theorem bfv_refresh_moddown_wrap (spacing : Nat) (prime message error wrap : Int)
    (hs : 0 < spacing) (he : 2 * |error| < (spacing : Int)) :
    (((spacing : Int) * (message + prime * wrap) + error -
      MxxRuntime.centeredResidue spacing
        ((spacing : Int) * (message + prime * wrap) + error)) /
      (spacing : Int)) % prime = message % prime := by
  rw [bfv_refresh_exact_moddown spacing (message + prime * wrap) error hs he]
  simp

end MxxExponentLut
