import Mathlib

namespace MxxWe

/-!
Arithmetic support for the interval decoder used by the Diamond graph.

The Rust decoder evaluates `RoundDiv(q - 2, 4)`, which is `q / 4` for natural
moduli.  The definitions below keep the canonical representative explicit so
that the negative half-modulus convention is visible in the generated proof.
-/

def quarter (q : Nat) : Nat := q / 4

def half (q : Nat) : Nat := q / 2

def decoderRadius (q : Nat) : Nat :=
  min (quarter q) (min (q - 3 * quarter q)
    (min (half q - quarter q + 1) (3 * quarter q - half q + 1)))

def inDecoderInterval (q coefficient : Nat) : Prop :=
  quarter q ≤ coefficient ∧ coefficient ≤ 3 * quarter q

def canonicalCoeff (q : Nat) (value : Int) : Nat :=
  (value % (q : Int)).toNat

def messageCenter (q : Nat) (message : Bool) : Nat :=
  if message then half q else 0

def decoded (q : Nat) (coefficient : Nat) : Bool :=
  if quarter q ≤ coefficient ∧ coefficient ≤ 3 * quarter q then true else false

theorem decoderRadius_conditions {q B : Nat} (hB : B < decoderRadius q) :
    B < quarter q ∧
      B < q - 3 * quarter q ∧
      B ≤ half q - quarter q ∧
      B ≤ 3 * quarter q - half q := by
  simp only [decoderRadius, Nat.lt_min] at hB
  exact ⟨hB.1, hB.2.1, Nat.lt_succ_iff.mp hB.2.2.1,
    Nat.lt_succ_iff.mp hB.2.2.2⟩

theorem natAbs_bounds {e : Int} {B : Nat} (he : e.natAbs ≤ B) :
    -(B : Int) ≤ e ∧ e ≤ (B : Int) := by
  constructor
  · have h := Int.le_natAbs (a := -e)
    simp only [Int.natAbs_neg] at h
    omega
  · have h := Int.le_natAbs (a := e)
    omega

theorem canonicalCoeff_of_nonneg_lt {q : Nat} {value : Int}
    (hvalue : 0 ≤ value) (hbound : value < (q : Int)) :
    canonicalCoeff q value = value.toNat := by
  simp [canonicalCoeff, Int.emod_eq_of_lt hvalue hbound]

theorem canonicalCoeff_of_neg_gt {q : Nat} {value : Int}
    (hvalue : -(q : Int) < value) (hbound : value < 0) :
    canonicalCoeff q value = (value + (q : Int)).toNat := by
  have hnonneg : 0 ≤ value + (q : Int) := by omega
  have hlt : value + (q : Int) < (q : Int) := by omega
  have hmod : value % (q : Int) = value + (q : Int) := by
    calc
      value % (q : Int) = ((value + (q : Int)) - (q : Int)) % (q : Int) := by
        congr 1; omega
      _ = ((value + (q : Int)) % (q : Int) - (q : Int) % (q : Int)) % (q : Int) := by
        rw [Int.sub_emod]
      _ = value + (q : Int) := by
        simp [Int.emod_eq_of_lt hnonneg hlt]
  simp [canonicalCoeff, hmod]

theorem canonicalCoeff_zero_of_small_error {q B : Nat} (hq : 4 ≤ q)
    (hB : B < decoderRadius q) {e : Int} (he : e.natAbs ≤ B) :
    canonicalCoeff q e < quarter q ∨
      3 * quarter q < canonicalCoeff q e := by
  have hcond := decoderRadius_conditions hB
  have heb := natAbs_bounds he
  by_cases hnonneg : 0 ≤ e
  · left
    have heq : e < (q : Int) := by
      have hBq : B < q := by omega
      exact heb.2.trans_lt (by exact_mod_cast hBq)
    rw [canonicalCoeff_of_nonneg_lt hnonneg heq]
    have hBc : B < quarter q := hcond.1
    have hBcI : (B : Int) < (quarter q : Int) := by exact_mod_cast hBc
    have hnat : e.toNat < quarter q := by
      apply (Int.toNat_lt (n := quarter q) hnonneg).2
      exact heb.2.trans_lt hBcI
    exact hnat
  · right
    have heneg : e < 0 := by omega
    have hlower : -(q : Int) < e := by
      have hq3 : B < q - 3 * quarter q := hcond.2.1
      omega
    rw [canonicalCoeff_of_neg_gt hlower heneg]
    have hq3 : B < q - 3 * quarter q := hcond.2.1
    have hq3' : 3 * quarter q + B < q := by omega
    have hq3I : (3 * quarter q : Int) + (B : Int) < (q : Int) := by
      exact_mod_cast hq3'
    have hupper : (3 * quarter q : Int) < e + q := by omega
    exact by
      have hnonneg' : 0 ≤ e + (q : Int) := by omega
      have hto : (3 * quarter q : Int) <
          ((e + (q : Int)).toNat : Int) := by
        rw [Int.toNat_of_nonneg hnonneg']
        exact hupper
      exact_mod_cast hto

theorem canonicalCoeff_one_in_interval {q B : Nat} (hq : 4 ≤ q)
    (hB : B < decoderRadius q) {e : Int} (he : e.natAbs ≤ B) :
    inDecoderInterval q (canonicalCoeff q ((half q : Int) + e)) := by
  have hcond := decoderRadius_conditions hB
  have heb := natAbs_bounds he
  have hq4 := Nat.div_add_mod q 4
  have hq2 := Nat.div_add_mod q 2
  have hr4 : q % 4 < 4 := Nat.mod_lt q (by omega)
  have hr2 : q % 2 < 2 := Nat.mod_lt q (by omega)
  have hquarter_half : quarter q ≤ half q := by
    simp only [quarter, half]
    omega
  have hhalf_threequarter : half q ≤ 3 * quarter q := by
    simp only [quarter, half]
    omega
  have hlowNat : quarter q + B ≤ half q :=
    by simpa [Nat.add_comm] using
      (Nat.le_sub_iff_add_le hquarter_half).mp hcond.2.2.1
  have huppNat : half q + B ≤ 3 * quarter q :=
    by simpa [Nat.add_comm] using
      (Nat.le_sub_iff_add_le hhalf_threequarter).mp hcond.2.2.2
  have hlowInt : (quarter q : Int) + B ≤ (half q : Int) := by
    exact_mod_cast hlowNat
  have huppInt : (half q : Int) + B ≤ (3 * quarter q : Int) := by
    exact_mod_cast huppNat
  have hlow : (quarter q : Int) ≤ (half q : Int) + e := by
    omega
  have hupp : (half q : Int) + e ≤ (3 * quarter q : Int) := by
    omega
  have hnonneg : 0 ≤ (half q : Int) + e := by omega
  have hltq : (half q : Int) + e < (q : Int) := by
    omega
  rw [canonicalCoeff_of_nonneg_lt hnonneg hltq]
  exact ⟨by omega, by omega⟩

theorem decode_zero_of_small_error {q B : Nat} (hq : 4 ≤ q)
    (hB : B < decoderRadius q) {e : Int} (he : e.natAbs ≤ B) :
    decoded q (canonicalCoeff q e) = false := by
  simp only [decoded]
  split
  · rename_i hinterval
    have hcond := decoderRadius_conditions hB
    have hbad := canonicalCoeff_zero_of_small_error hq hB he
    omega
  · rfl

theorem decode_one_of_small_error {q B : Nat} (hq : 4 ≤ q)
    (hB : B < decoderRadius q) {e : Int} (he : e.natAbs ≤ B) :
    decoded q (canonicalCoeff q ((half q : Int) + e)) = true := by
  simp only [decoded]
  split
  · rfl
  · rename_i hnot
    have hinterval := canonicalCoeff_one_in_interval hq hB he
    exfalso
    exact hnot (by simpa [inDecoderInterval] using hinterval)

theorem ceilHalf_neg_mod (q : Nat) (hq : 0 < q) :
    canonicalCoeff q (-((q + 1) / 2 : Int)) = half q := by
  have hsum : (q + 1) / 2 + q / 2 = q := by omega
  have hqint : (q : Int) > 0 := by omega
  have hrep : -((q + 1) / 2 : Int) =
      ((q / 2 : Nat) : Int) - (q : Int) := by
    omega
  simp only [canonicalCoeff, hrep]
  rw [Int.sub_emod]
  have hhalf : 0 ≤ ((q / 2 : Nat) : Int) := by omega
  have hhalf_lt : ((q / 2 : Nat) : Int) < (q : Int) := by
    exact_mod_cast (Nat.div_lt_self hq (by omega : 1 < 2))
  simp only [Int.emod_self, sub_zero, Int.emod_eq_of_lt hhalf hhalf_lt]
  have hdiv : Int.toNat ((q : Int) / 2) = q / 2 := by
    have hcast : (↑q / 2 : Int) = (↑(q / 2) : Int) := by
      have h := Int.natCast_ediv q 2
      exact h.symm
    calc
      Int.toNat ((q : Int) / 2) = Int.toNat (↑(q / 2) : Int) :=
        congrArg Int.toNat hcast
      _ = q / 2 := by
        have hnonneg : (0 : Int) ≤ (q / 2 : Nat) := by omega
        have h := Int.toNat_of_nonneg hnonneg
        exact_mod_cast h
  change Int.toNat ((q : Int) / 2) = q / 2
  exact hdiv

end MxxWe
