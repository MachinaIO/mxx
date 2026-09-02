import MxxWe.DiamondWE.Parameters
import MxxPrimitives.Bounds

namespace Mxx.We.DiamondWE

def decoderNoiseThreshold (q : Nat) : Nat :=
  min (decoderQuarter q)
    (min (q - 3 * decoderQuarter q)
      (min (q / 2 - decoderQuarter q + 1)
        (3 * decoderQuarter q - q / 2 + 1)))

def DecoderSafe (q noise : Nat) : Prop :=
  decoderQuarter q > noise ∧
    q - (3 * decoderQuarter q + noise) > 0 ∧
    q / 2 ≥ decoderQuarter q + noise ∧
    3 * decoderQuarter q ≥ q / 2 + noise

def decodeInterval (q : Nat) (coefficient : Int) : Bool :=
  decide ((decoderQuarter q : Int) ≤ coefficient ∧
    coefficient ≤ (3 * decoderQuarter q : Nat))

theorem decoder_safe_iff {q noise : Nat} (geometry : DecoderGeometryValid q) :
    noise < decoderNoiseThreshold q ↔ DecoderSafe q noise := by
  unfold decoderNoiseThreshold DecoderSafe
  rcases geometry with ⟨quarter_le_half, half_le_three_quarters, three_quarters_le_q⟩
  omega

theorem decoder_threshold_le_half {q : Nat} (geometry : DecoderGeometryValid q) :
    decoderNoiseThreshold q ≤ q / 2 := by
  unfold decoderNoiseThreshold
  exact (min_le_left _ _).trans geometry.quarter_le_half

/- The threshold is strict.  This small bridge is used by the exact decoder proof, where the
   whole-matrix error bound is first projected to coefficient zero. -/
theorem decoder_safe_of_lt_threshold {q noise : Nat}
    (geometry : DecoderGeometryValid q) (hnoise : noise < decoderNoiseThreshold q) :
    DecoderSafe q noise :=
  decoder_safe_iff geometry |>.mp hnoise

/- The hypotheses below are the exact centered-lift cases produced by the coefficient-zero
   equation.  For a false bit a negative error is represented by its residue `q + error`; for a
   true bit the safe geometry keeps the centered value in the same residue interval. -/
theorem decode_interval_of_centered_error
    {q noise : Nat} (geometry : DecoderGeometryValid q)
    (hnoise : noise < decoderNoiseThreshold q) (message : Bool) (error : Int) (coefficient : Int)
    (error_lower : -(noise : Int) ≤ error) (error_upper : error ≤ noise)
    (coefficient_eq :
      if message then coefficient = q / 2 + error
      else (0 ≤ error ∧ coefficient = error) ∨ (error < 0 ∧ coefficient = q + error)) :
    decodeInterval q coefficient = message := by
  have safe := decoder_safe_of_lt_threshold geometry hnoise
  rcases safe with ⟨quarter_error, modulus_error, lower_error, upper_error⟩
  cases message with
  | false =>
      simp only [Bool.false_eq_true, ↓reduceIte] at coefficient_eq
      rcases coefficient_eq with hpositive | hnegative
      · simp [decodeInterval]
        omega
      · simp [decodeInterval]
        omega
  | true =>
      simp only [↓reduceIte] at coefficient_eq
      simp [decodeInterval]
      omega

theorem val_of_intCast_of_nonneg_lt {q : Nat} (hq : 0 < q) {z : Int}
    (hz_nonneg : 0 ≤ z) (hz_lt : z < q) :
    ((z : ZMod q).val : Int) = z := by
  letI : NeZero q := ⟨by omega⟩
  have hval := ZMod.val_intCast (n := q) z
  rw [Int.emod_eq_of_lt (by omega) hz_lt] at hval
  exact hval

theorem val_of_intCast_of_neg {q : Nat} (hq : 0 < q) {z : Int}
    (hz_lower : -(q : Int) < z) (hz_neg : z < 0) :
    ((z : ZMod q).val : Int) = q + z := by
  letI : NeZero q := ⟨by omega⟩
  have hval := ZMod.val_intCast (n := q) z
  rw [Int.emod_eq_add_self_emod, Int.emod_eq_of_lt (by omega) (by omega)] at hval
  simpa [add_comm] using hval

example : decoderQuarter 17 = 4 := by decide

example : DecoderGeometryValid 17 := by
  constructor <;> norm_num [decoderQuarter]

example : decoderNoiseThreshold 17 = 4 := by decide

example : DecoderSafe 17 3 := by
  norm_num [DecoderSafe, decoderQuarter]

example : decodeInterval 17 4 = true := by decide

example : decodeInterval 17 0 = false := by decide

end Mxx.We.DiamondWE
