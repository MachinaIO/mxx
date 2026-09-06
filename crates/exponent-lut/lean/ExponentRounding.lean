import ExponentCrt

namespace MxxExponentLut

private theorem strict_rounding_integer (p spacing anchor error value : Int)
    (hp : 0 < p) (hs : 0 < spacing)
    (he : 2 * |error| < spacing)
    (hv : value = (spacing * anchor + error) % (p * spacing)) :
    ((p * value + p * spacing / 2) / (p * spacing)) % p = anchor % p := by
  let k := (spacing * anchor + error) / (p * spacing)
  have hdivision := Int.mul_ediv_add_emod (spacing * anchor + error) (p * spacing)
  have hvalue : value = spacing * (anchor - p * k) + error := by
    rw [← hv] at hdivision
    dsimp [k]
    nlinarith
  have hq := mul_pos hp hs
  have habs1 := le_abs_self error
  have habs2 := neg_abs_le error
  have hlo : 0 ≤ p * error + p * spacing / 2 := by
    have h := mul_lt_mul_of_pos_left he hp
    have hhalf := Int.mul_ediv_add_emod (p * spacing) 2
    have hrem := Int.emod_lt_of_pos (p * spacing) (by decide : (0 : Int) < 2)
    nlinarith
  have hhi : p * error + p * spacing / 2 < p * spacing := by
    have h := mul_lt_mul_of_pos_left he hp
    have hhalf := Int.mul_ediv_add_emod (p * spacing) 2
    have hrem := Int.emod_nonneg (p * spacing) (by decide : (2 : Int) ≠ 0)
    nlinarith
  have hzero : (p * error + p * spacing / 2) / (p * spacing) = 0 :=
    Int.ediv_eq_zero_of_lt hlo hhi
  have hexpand : p * value + p * spacing / 2 =
      (p * error + p * spacing / 2) + (anchor - p * k) * (p * spacing) := by
    rw [hvalue]
    ring
  rw [hexpand, Int.add_mul_ediv_right _ _ (ne_of_gt hq), hzero, zero_add]
  simp

/-- Strict nearest-digit decoding for the runtime CRT formula, including
canonical residue wraparound and signed errors. No parity premise is needed. -/
theorem crt_rounded_digit_of_strict_error {q p spacing : Nat}
    (hp : 0 < p) (hs : 0 < spacing) (hq : q = p * spacing)
    (x : ZMod q) (anchor error : Int)
    (hx : x = ((spacing : Int) * anchor + error : Int))
    (he : 2 * error.natAbs < spacing) :
    (((p : Int) * Int.ofNat x.val + Int.ofNat (q / 2)) / Int.ofNat q) % (p : Int) =
      anchor % (p : Int) := by
  letI : NeZero q := ⟨by rw [hq]; positivity⟩
  have hv : (x.val : Int) = ((spacing : Int) * anchor + error) % ((p : Int) * spacing) := by
    rw [hx, ZMod.val_intCast, hq, Nat.cast_mul]
  have he' : 2 * |error| < (spacing : Int) := by
    rw [← Int.natCast_natAbs]
    exact_mod_cast he
  have result := strict_rounding_integer p spacing anchor error x.val
    (by exact_mod_cast hp) (by exact_mod_cast hs) he' hv
  simpa only [hq, Int.ofNat_eq_natCast, Nat.cast_mul, Int.natCast_ediv, Nat.cast_ofNat] using result

end MxxExponentLut
