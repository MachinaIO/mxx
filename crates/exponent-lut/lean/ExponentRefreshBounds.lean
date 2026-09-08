import PrimitivesBounds

namespace MxxExponentLut

/-- The exact natural-number ceiling used by `noise_refresh/simulation.rs`.
The zero branch also matches the implementation. -/
def refreshCeilDiv (numerator denominator : Nat) : Nat :=
  if numerator = 0 then 0 else (numerator + denominator - 1) / denominator

/-- Exact `switched_error_bound`: paired switch rounding followed by decoder error. -/
def switchedErrorBound (pre lift dimension decoder : Nat) : Nat :=
  refreshCeilDiv (2 * pre + (lift - 1) * (dimension + 2)) (2 * lift) + decoder

/-- Data exported from the candidate's actual noise report and route gains. -/
structure RefreshNumericSlot where
  stateGain : Nat
  initialBound : Nat
  maskGain : Nat
  freshGain : Nat
  readoutBound : Nat
  preBound : Nat
  decoderGain : Nat
  helperCutoff : Nat
  decoderBound : Nat
  lift : Nat
  dimension : Nat
  residualBound : Nat
  maskModulus : Nat
  maskBound : Nat
  spacing : Nat

/-- Rejected candidates retain arithmetic checks but do not claim strict decoding. -/
def RefreshNumericSlot.Valid (s : RefreshNumericSlot) (accepted : Bool) : Prop :=
  s.stateGain * s.initialBound + (s.maskGain + s.freshGain) * s.readoutBound = s.preBound ∧
  s.decoderGain * s.helperCutoff = s.decoderBound ∧
  switchedErrorBound s.preBound s.lift s.dimension s.decoderBound = s.residualBound ∧
  0 < s.lift ∧ s.maskModulus / 2 = s.maskBound ∧
  (accepted = true →
    2 * (s.maskBound + switchedErrorBound s.preBound s.lift s.dimension s.decoderBound) < s.spacing)

instance (s : RefreshNumericSlot) (accepted : Bool) : Decidable (s.Valid accepted) := by
  unfold RefreshNumericSlot.Valid
  infer_instance

theorem refreshCeilDiv_eq (numerator denominator : Nat) (hd : 0 < denominator) :
    refreshCeilDiv numerator denominator = (numerator + denominator - 1) / denominator := by
  unfold refreshCeilDiv
  split_ifs with h
  · subst numerator
    simp only [Nat.zero_add]
    symm
    exact Nat.div_eq_of_lt (by omega)
  · rfl

/-- Integer division introduces no unchecked rational-to-integer rounding step. -/
theorem scaled_le_refreshCeilDiv {error numerator denominator : Nat}
    (hd : 0 < denominator) (hscaled : denominator * error ≤ numerator) :
    error ≤ refreshCeilDiv numerator denominator := by
  rw [refreshCeilDiv_eq numerator denominator hd]
  apply (Nat.le_div_iff_mul_le hd).mpr
  have h : numerator ≤ numerator + denominator - 1 := by omega
  have he : error * denominator ≤ numerator := by simpa [Nat.mul_comm] using hscaled
  exact he.trans h

/-- The scaled paired-switch coefficient estimate implies the exact Rust route
budget. The premise is an arithmetic estimate of the actual signed coefficient,
not an assumption of simulator soundness. -/
theorem switched_error_bound {pre lift dimension decoder : Nat}
    {pairedError decoderError : Int} (hlift : 0 < lift)
    (hpaired : 2 * lift * pairedError.natAbs ≤
      2 * pre + (lift - 1) * (dimension + 2))
    (hdecoder : decoderError.natAbs ≤ decoder) :
    (pairedError + decoderError).natAbs ≤ switchedErrorBound pre lift dimension decoder := by
  exact (Int.natAbs_add_le _ _).trans (Nat.add_le_add
    (scaled_le_refreshCeilDiv (by omega) hpaired) hdecoder)

/-- Separate the pre-switch coefficient from the aggregate rounding residual.
The identity is the unscaled algebraic error equation for the actual switch. -/
theorem paired_scaled_error {pre lift dimension : Nat}
    {pairedError preError roundingError : Int}
    (hidentity : (lift : Int) * pairedError = preError + roundingError)
    (hpre : preError.natAbs ≤ pre)
    (hrounding : 2 * roundingError.natAbs ≤ (lift - 1) * (dimension + 2)) :
    2 * lift * pairedError.natAbs ≤ 2 * pre + (lift - 1) * (dimension + 2) := by
  have habs := Int.natAbs_add_le preError roundingError
  rw [← hidentity, Int.natAbs_mul] at habs
  simp only [Int.natAbs_natCast] at habs
  nlinarith

/-- Combine the actual switch error equation with its pre-switch and rounding
estimates, then include the decoder action. -/
theorem switched_error_of_decomposition {pre lift dimension decoder : Nat}
    {pairedError preError roundingError decoderError : Int} (hlift : 0 < lift)
    (hidentity : (lift : Int) * pairedError = preError + roundingError)
    (hpre : preError.natAbs ≤ pre)
    (hrounding : 2 * roundingError.natAbs ≤ (lift - 1) * (dimension + 2))
    (hdecoder : decoderError.natAbs ≤ decoder) :
    (pairedError + decoderError).natAbs ≤ switchedErrorBound pre lift dimension decoder := by
  exact switched_error_bound hlift (paired_scaled_error hidentity hpre hrounding) hdecoder

/-- The simulator checks twice the complete mask-plus-route budget strictly
below the integer plaintext spacing. This gives the coefficient condition
needed for nearest-plaintext rounding, including at odd spacings. -/
theorem strict_rounding_bound {mask route spacing : Nat} {maskError routeError : Int}
    (hmask : maskError.natAbs ≤ mask) (hroute : routeError.natAbs ≤ route)
    (haccepted : 2 * (mask + route) < spacing) :
    2 * (maskError + routeError).natAbs < spacing := by
  have hsum := (Int.natAbs_add_le maskError routeError).trans (Nat.add_le_add hmask hroute)
  exact (Nat.mul_le_mul_left 2 hsum).trans_lt haccepted

/-- Composition of the operational switch budget and the strict candidate
rounding check for one actual coefficient. -/
theorem switched_mask_strict_rounding {pre lift dimension decoder mask spacing : Nat}
    {pairedError decoderError maskError : Int} (hlift : 0 < lift)
    (hpaired : 2 * lift * pairedError.natAbs ≤
      2 * pre + (lift - 1) * (dimension + 2))
    (hdecoder : decoderError.natAbs ≤ decoder) (hmask : maskError.natAbs ≤ mask)
    (haccepted : 2 * (mask + switchedErrorBound pre lift dimension decoder) < spacing) :
    2 * (maskError + (pairedError + decoderError)).natAbs < spacing := by
  exact strict_rounding_bound hmask (switched_error_bound hlift hpaired hdecoder) haccepted

end MxxExponentLut
