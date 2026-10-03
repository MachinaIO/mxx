import ExponentEncoding

/-!
# Decryption of a terminal encoding

At the terminal level the gadget is one and the message scale is `⌊q / 2⌋`, so the second row
component minus the mask part is the phase `-⌊q / 2⌋ μ + e`. Threshold decoding modulo two returns
every binary coefficient of `μ` when `4 |e| + 4 < q`.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime
open scoped BigOperators

variable {n : Nat}

/-- Threshold decoding of one coefficient `-⌊q/2⌋ b + e` with `b ∈ {0, 1}` and `4 |e| + 4 < q`. -/
theorem threshold_decode_value {q : Nat} (hq : 1 < q) (hodd : q % 2 = 1) {b e : Int}
    (hb : b = 0 ∨ b = 1) (he : 4 * |e| + 4 < q) :
    ((2 * ((((-(((q / 2 : Nat) : Int) * b) + e : Int) : ZMod q).val : Nat) : Int) +
      (q : Int) / 2) / (q : Int)) % 2 = b := by
  rw [val_intCast_emod hq]
  have hq2 : ((q / 2 : Nat) : Int) = (q : Int) / 2 := by omega
  rw [hq2]
  have hqpos : (0 : Int) < q := by omega
  rcases hb with rfl | rfl
  · simp only [mul_zero, neg_zero, zero_add]
    by_cases hpos : 0 ≤ e
    · rw [abs_of_nonneg hpos] at he
      rw [Int.emod_eq_of_lt hpos (by omega)]
      rw [(Int.ediv_eq_iff_of_pos (y := 0) hqpos).mpr ⟨by omega, by omega⟩]
      rfl
    · have hneg : e < 0 := by omega
      rw [abs_of_neg hneg] at he
      have hmod : e % (q : Int) = e + q := by
        rw [← Int.add_emod_right, Int.emod_eq_of_lt (by omega) (by omega)]
      rw [hmod, (Int.ediv_eq_iff_of_pos (y := 2) hqpos).mpr ⟨by omega, by omega⟩]
      rfl
  · simp only [mul_one]
    have habs := abs_le.mp (le_refl |e|)
    have hmod : (-((q : Int) / 2) + e) % (q : Int) = -((q : Int) / 2) + e + q := by
      rw [← Int.add_emod_right, Int.emod_eq_of_lt (by omega) (by omega)]
    rw [hmod, (Int.ediv_eq_iff_of_pos (y := 1) hqpos).mpr ⟨by omega, by omega⟩]
    rfl

/-- Decryption of a terminal encoding: the threshold-decoded coefficient `k` of the phase
`c₁ - (s₀ A₀₁ + A₁₁)` is coefficient `k` of the message, when it is binary and the error is below
a quarter of the modulus. -/
theorem decode_terminal {q : Nat} (hq : 1 < q) (hodd : q % 2 = 1) (hn : 0 < n)
    {sm sp μ : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 1}
    {c : Fin 2 → ExactMatrix q n 1 1} {B : Nat}
    (hrel : Rel sm A (fun j ↦ reducePoly q n ((((q / 2 : Nat) : Int) : ErrorPoly n) * μ *
      unitPair sp j) • (1 : ExactMatrix q n 1 1)) c B)
    (hB : 4 * B + 4 < q) {k : Fin n} (hμ : μ.coeff k = 0 ∨ μ.coeff k = 1)
    {phase : ExactMatrix q n 1 1}
    (hphase : phase = c 1 - (reducePoly q n sm • A 0 1 + A 1 1)) {length : Int} {out : Int}
    (hdecode : thresholdDecode 2 length k.val phase out) : out = μ.coeff k := by
  obtain ⟨e, he, hc⟩ := hrel
  obtain ⟨_, _, _, _, index, hindex, _, hout⟩ := hdecode
  have hk : index = k := Fin.ext (by exact_mod_cast hindex)
  subst hk
  rw [hout]
  have hentry : (phase 0 0).coeff index =
      ((-(((q / 2 : Nat) : Int) * μ.coeff index) + (e 1 0 0).coeff index : Int) : ZMod q) := by
    rw [hphase, hc 1]
    simp only [unitPair, Matrix.cons_val_one, Matrix.sub_apply, Matrix.add_apply,
      Matrix.smul_apply, smul_eq_mul, reduceMatrix, mul_one, Matrix.one_apply_eq]
    rw [show ∀ x y z w : ExactPoly q n, x + y - z + w - (x + y) = -z + w from fun x y z w ↦ by ring,
      ← map_neg, ← map_add, reducePoly_coeff_eq hq hn]
    simp [intCast_mul_coeff]
  rw [hentry]
  have heabs : 4 * |(e 1 0 0).coeff index| + 4 < q := by
    have := he 1 0 0 index
    rw [Int.abs_eq_natAbs]
    omega
  exact threshold_decode_value hq hodd hμ heabs

end MxxExponentLut
