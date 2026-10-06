import ExponentEncoding

/-!
# Decryption of a terminal encoding

At the terminal level the gadget is one and the message scale is `⌊q / 2⌋`, so the second row
component minus the mask part is the phase `-⌊q / 2⌋ μ + e`. Threshold decoding modulo two returns
every binary coefficient of `μ` when `4 |e| + 4 < q`.

A coefficient output is finalized to the scale `k` with `t k = q - r` for its decryption modulus
`t`, so its phase is `-k μ + e`; threshold decoding modulo `t` of the negated phase `k μ - e`
returns `μ mod t` when `|μ| ≤ W` and `2 t |e| + 2 r W < q`.
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

/-- Threshold decoding of one coefficient `-⌊q/2⌋ b + e` with `|b| ≤ 2` and `4 |e| + 8 < q` returns
the parity of `b`. -/
theorem threshold_decode_parity {q : Nat} (hq : 1 < q) (hodd : q % 2 = 1) {b e : Int}
    (hb : -2 ≤ b ∧ b ≤ 2) (he : 4 * |e| + 8 < q) :
    ((2 * ((((-(((q / 2 : Nat) : Int) * b) + e : Int) : ZMod q).val : Nat) : Int) +
      (q : Int) / 2) / (q : Int)) % 2 = b % 2 := by
  rw [val_intCast_emod hq]
  have hq2 : ((q / 2 : Nat) : Int) = (q : Int) / 2 := by omega
  rw [hq2]
  have hqpos : (0 : Int) < q := by omega
  have hqodd : (q : Int) = 2 * ((q : Int) / 2) + 1 := by omega
  have habs := abs_le.mp (le_refl |e|)
  -- The centered residue `r` of the phase decodes to `0` near `0` and to `1` near `q/2`.
  have hdec : ∀ r : Int, 0 ≤ r → r < q →
      ((2 * r + (q : Int) / 2 < q ∨ 2 * (q : Int) ≤ 2 * r + (q : Int) / 2) →
        (2 * r + (q : Int) / 2) / q % 2 = 0) ∧
      ((q : Int) ≤ 2 * r + (q : Int) / 2 → 2 * r + (q : Int) / 2 < 2 * q →
        (2 * r + (q : Int) / 2) / q % 2 = 1) := by
    intro r h0 h1
    constructor
    · rintro (hr | hr)
      · rw [(Int.ediv_eq_iff_of_pos (y := 0) hqpos).mpr ⟨by omega, by omega⟩]; rfl
      · rw [(Int.ediv_eq_iff_of_pos (y := 2) hqpos).mpr ⟨by omega, by omega⟩]; rfl
    · intro hr1 hr2
      rw [(Int.ediv_eq_iff_of_pos (y := 1) hqpos).mpr ⟨by omega, by omega⟩]; rfl
  have hmod : ∀ x : Int, 0 ≤ x % (q : Int) ∧ x % (q : Int) < q := fun x ↦
    ⟨Int.emod_nonneg _ (by omega), Int.emod_lt_of_pos _ hqpos⟩
  rcases (show b = -2 ∨ b = -1 ∨ b = 0 ∨ b = 1 ∨ b = 2 by omega) with rfl | rfl | rfl | rfl | rfl
  · have hr : (-((q : Int) / 2 * -2) + e) % (q : Int) = (e - 1) % (q : Int) := by
      rw [show -((q : Int) / 2 * -2) + e = (e - 1) + q by omega, Int.add_emod_right]
    rw [hr]
    obtain ⟨h0, h1⟩ := hmod (e - 1)
    refine (hdec _ h0 h1).1 ?_
    by_cases he1 : 1 ≤ e
    · left; rw [Int.emod_eq_of_lt (by omega) (by omega)]; omega
    · right; rw [show e - 1 = (e - 1 + q) + (-1) * q by ring, Int.add_mul_emod_self_right,
        Int.emod_eq_of_lt (by omega) (by omega)]; omega
  · have hr : (-((q : Int) / 2 * -1) + e) % (q : Int) = (q : Int) / 2 + e := by
      rw [show -((q : Int) / 2 * -1) + e = (q : Int) / 2 + e by ring,
        Int.emod_eq_of_lt (by omega) (by omega)]
    rw [hr]
    exact (hdec _ (by omega) (by omega)).2 (by omega) (by omega)
  · have hr : (-((q : Int) / 2 * 0) + e) % (q : Int) = e % (q : Int) := by ring_nf
    rw [hr]
    obtain ⟨h0, h1⟩ := hmod e
    refine (hdec _ h0 h1).1 ?_
    by_cases hpos : 0 ≤ e
    · left; rw [Int.emod_eq_of_lt hpos (by omega)]; omega
    · right; rw [show e = (e + q) + (-1) * q by ring, Int.add_mul_emod_self_right,
        Int.emod_eq_of_lt (by omega) (by omega)]; omega
  · have hr : (-((q : Int) / 2 * 1) + e) % (q : Int) = -((q : Int) / 2) + e + q := by
      rw [← Int.add_emod_right, Int.emod_eq_of_lt (by omega) (by omega)]
      ring
    rw [hr]
    exact (hdec _ (by omega) (by omega)).2 (by omega) (by omega)
  · have hr : (-((q : Int) / 2 * 2) + e) % (q : Int) = (e + 1) % (q : Int) := by
      rw [show -((q : Int) / 2 * 2) + e = (e + 1) + (-1) * q by omega, Int.add_mul_emod_self_right]
    rw [hr]
    obtain ⟨h0, h1⟩ := hmod (e + 1)
    refine (hdec _ h0 h1).1 ?_
    by_cases he1 : -1 ≤ e
    · left; rw [Int.emod_eq_of_lt (by omega) (by omega)]; omega
    · right; rw [show e + 1 = (e + 1 + q) + (-1) * q by ring, Int.add_mul_emod_self_right,
        Int.emod_eq_of_lt (by omega) (by omega)]; omega

/-- Decryption of a terminal encoding whose coefficient `k` lies in `[-2, 2]`: threshold decoding
returns its parity when the error is below a quarter of the modulus, less two. -/
theorem decode_terminal_parity {q : Nat} (hq : 1 < q) (hodd : q % 2 = 1) (hn : 0 < n)
    {sm sp μ : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 1}
    {c : Fin 2 → ExactMatrix q n 1 1} {B : Nat}
    (hrel : Rel sm A (fun j ↦ reducePoly q n ((((q / 2 : Nat) : Int) : ErrorPoly n) * μ *
      unitPair sp j) • (1 : ExactMatrix q n 1 1)) c B)
    (hB : 4 * B + 8 < q) {k : Fin n} (hμ : -2 ≤ μ.coeff k ∧ μ.coeff k ≤ 2)
    {phase : ExactMatrix q n 1 1}
    (hphase : phase = c 1 - (reducePoly q n sm • A 0 1 + A 1 1)) {length : Int} {out : Int}
    (hdecode : thresholdDecode 2 length k.val phase out) : out = μ.coeff k % 2 := by
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
  have heabs : 4 * |(e 1 0 0).coeff index| + 8 < q := by
    have := he 1 0 0 index
    rw [Int.abs_eq_natAbs]
    omega
  exact threshold_decode_parity hq hodd hμ heabs


/-- Threshold decoding of one coefficient `-⌊q/2⌋ b + e` with `|b| ≤ T` and `4 |e| + 2 T + 12 < q`
returns the parity of `b`: with `b = 2 t + p`, the phase is `-⌊q/2⌋ p + (t + e)` modulo `q`. -/
theorem threshold_decode_parity_bounded {q : Nat} (hq : 1 < q) (hodd : q % 2 = 1) {b e : Int}
    {T : Nat} (hb : |b| ≤ T) (he : 4 * |e| + 2 * T + 12 < q) :
    ((2 * ((((-(((q / 2 : Nat) : Int) * b) + e : Int) : ZMod q).val : Nat) : Int) +
      (q : Int) / 2) / (q : Int)) % 2 = b % 2 := by
  obtain ⟨t, p, hbtp, hp0, hp1⟩ : ∃ t p : Int, b = 2 * t + p ∧ 0 ≤ p ∧ p < 2 :=
    ⟨b / 2, b % 2, (Int.mul_ediv_add_emod b 2).symm, Int.emod_nonneg _ (by norm_num),
      Int.emod_lt_of_pos _ (by norm_num)⟩
  have hq2 : ((q / 2 : Nat) : Int) = (q : Int) / 2 := by omega
  have hqodd : (q : Int) = 2 * ((q : Int) / 2) + 1 := by omega
  have hcast : (((-(((q / 2 : Nat) : Int) * b) + e : Int) : ZMod q)) =
      (((-(((q / 2 : Nat) : Int) * p) + (t + e) : Int) : ZMod q)) := by
    rw [ZMod.intCast_eq_intCast_iff_dvd_sub]
    refine ⟨t, ?_⟩
    rw [hq2, hbtp]
    generalize (q : Int) / 2 = h at hqodd ⊢
    rw [hqodd]
    ring
  have hbp : b % 2 = p % 2 := by omega
  rw [hcast, threshold_decode_parity hq hodd (by omega) ?_, hbp]
  rw [abs_le] at hb
  have habs : |t + e| ≤ |t| + |e| := abs_add_le _ _
  have ht : 2 * |t| ≤ (T : Int) + 1 := by
    rcases abs_cases t with ⟨h1, _⟩ | ⟨h1, _⟩ <;> omega
  omega

/-- Decryption of a terminal encoding whose coefficient `k` lies in `[-T, T]`: threshold decoding
returns its parity when `4 B + 2 T + 12 < q`. -/
theorem decode_terminal_parity_bounded {q : Nat} (hq : 1 < q) (hodd : q % 2 = 1) (hn : 0 < n)
    {sm sp μ : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 1}
    {c : Fin 2 → ExactMatrix q n 1 1} {B T : Nat}
    (hrel : Rel sm A (fun j ↦ reducePoly q n ((((q / 2 : Nat) : Int) : ErrorPoly n) * μ *
      unitPair sp j) • (1 : ExactMatrix q n 1 1)) c B)
    (hB : 4 * B + 2 * T + 12 < q) {k : Fin n} (hμ : |μ.coeff k| ≤ T)
    {phase : ExactMatrix q n 1 1}
    (hphase : phase = c 1 - (reducePoly q n sm • A 0 1 + A 1 1)) {length : Int} {out : Int}
    (hdecode : thresholdDecode 2 length k.val phase out) : out = μ.coeff k % 2 := by
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
  have heabs : 4 * |(e 1 0 0).coeff index| + 2 * T + 12 < q := by
    have := he 1 0 0 index
    rw [Int.abs_eq_natAbs]
    omega
  exact threshold_decode_parity_bounded hq hodd hμ heabs

/-- Threshold decoding modulo `t` of one coefficient `k v - e` of the negated phase, with
`t k = q - r`, `|v| ≤ W` and `2 t |e| + 2 r W < q`, returns `v mod t`: `t (k v - e)` is
`q v - d` with `2 |d| < q`. -/
theorem threshold_decode_scaled {q t k r : Nat} (hq : 1 < q) (ht : 0 < t) (hk : t * k + r = q)
    {v e : Int} {W : Nat} (hv : |v| ≤ W) (he : 2 * t * |e| + 2 * r * W < q) :
    (((t : Int) * ((((k * v - e : Int) : ZMod q).val : Nat) : Int) + (q : Int) / 2) / (q : Int)) %
      (t : Int) = v % t := by
  rw [val_intCast_emod hq]
  set x : Int := k * v - e with hx
  have hqpos : (0 : Int) < q := by omega
  have hsplit := Int.emod_add_mul_ediv x q
  set m := x / q
  set d : Int := (r : Int) * v + t * e
  have hkq : ((t : Int) * k + r) = q := by exact_mod_cast hk
  have key : (t : Int) * (x % q) + (q : Int) / 2 = q * (v - t * m) + ((q : Int) / 2 - d) := by
    have : (t : Int) * (x % q) = t * x - t * (q * m) := by linear_combination (t : Int) * hsplit
    rw [this, hx]
    simp only [d]
    rw [← hkq]
    ring
  have hrv : |(r : Int) * v| ≤ r * W := by
    rw [abs_mul, abs_of_nonneg (by positivity : (0 : Int) ≤ r)]
    exact mul_le_mul_of_nonneg_left hv (by positivity)
  have hte : |(t : Int) * e| = t * |e| := by
    rw [abs_mul, abs_of_nonneg (by positivity : (0 : Int) ≤ t)]
  have hd : 2 * |d| < q := by
    have := abs_add_le ((r : Int) * v) ((t : Int) * e)
    nlinarith
  have hdabs := abs_le.mp (le_refl |d|)
  have hquot : ((t : Int) * (x % q) + (q : Int) / 2) / q = v - t * m := by
    rw [key, Int.add_comm, Int.add_mul_ediv_left _ _ (by omega),
      Int.ediv_eq_zero_of_lt (by omega) (by omega)]
    ring
  rw [hquot]
  simp [Int.sub_emod, Int.mul_emod_right]

/-- Decryption of a finalized coefficient output: the encoding has scale `k` against the gadget
`1`, and threshold decoding modulo `t` of coefficient `i` of the negated phase
`-(c₁ - (s₀ A₀₁ + A₁₁))` returns `μ_i mod t` when `t k = q - r`, `|μ_i| ≤ W` and
`2 t B + 2 r W < q`. -/
theorem decode_terminal_scaled {q t k r W : Nat} (hq : 1 < q) (hn : 0 < n) (ht : 0 < t)
    (hk : t * k + r = q) {sm sp μ : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 1}
    {c : Fin 2 → ExactMatrix q n 1 1} {B : Nat}
    (hrel : Rel sm A (fun j ↦ reducePoly q n (((k : Int) : ErrorPoly n) * μ * unitPair sp j) •
      (1 : ExactMatrix q n 1 1)) c B)
    (hB : 2 * t * B + 2 * r * W < q) {i : Fin n} (hμ : |μ.coeff i| ≤ W)
    {phase : ExactMatrix q n 1 1}
    (hphase : phase = -(c 1 - (reducePoly q n sm • A 0 1 + A 1 1))) {length : Int} {out : Int}
    (hdecode : thresholdDecode t length i.val phase out) : out = μ.coeff i % t := by
  obtain ⟨e, he, hc⟩ := hrel
  obtain ⟨_, _, _, _, index, hindex, _, hout⟩ := hdecode
  have hi : index = i := Fin.ext (by exact_mod_cast hindex)
  subst hi
  rw [hout]
  have hentry : (phase 0 0).coeff index =
      ((k * μ.coeff index - (e 1 0 0).coeff index : Int) : ZMod q) := by
    rw [hphase, hc 1]
    simp only [unitPair, Matrix.cons_val_one, Matrix.neg_apply, Matrix.sub_apply,
      Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, reduceMatrix, mul_one,
      Matrix.one_apply_eq]
    rw [show ∀ x y z w : ExactPoly q n, -(x + y - z + w - (x + y)) = z - w from fun x y z w ↦ by
      ring, ← map_sub, reducePoly_coeff_eq hq hn]
    have hkμ := intCast_mul_coeff (k : Int) μ index
    push_cast at hkμ ⊢
    simp [hkμ]
  rw [hentry]
  have hnat : 2 * t * ((e 1 0 0).coeff index).natAbs + 2 * r * W < q := by
    have := Nat.mul_le_mul_left (2 * t) (he 1 0 0 index)
    omega
  have heabs : 2 * t * |(e 1 0 0).coeff index| + 2 * r * W < q := by
    rw [Int.abs_eq_natAbs]
    exact_mod_cast hnat
  exact threshold_decode_scaled hq ht hk hμ heabs

end MxxExponentLut
