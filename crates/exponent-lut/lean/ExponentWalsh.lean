import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Nat.ModEq
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity
import Mathlib.Algebra.Ring.Parity
import Mathlib.Logic.Equiv.Fin.Basic

/-!
# LUT interpolation by automorphism traces

A LUT of width `W = 2^log` is evaluated as `Σ_i σ_i(X^e) · C_{σ_i}` with
`C_σ = W⁻¹ Σ_k T(k) · σ(X^{-k})` over the `W` automorphisms `σ_i(X) = X^{1 + 2·quotient·i}` of the
native exponent group. In any commutative ring where the root `ζ` satisfies `ζ^{nativeWidth} = -1`,
the sum over the subgroup cancels every exponent outside `W ℤ`, which leaves `T(e)` on the lookup
domain (`lut_fourier_table_value`). No field or integral-domain hypothesis is needed.
-/

namespace MxxExponentLut

open scoped BigOperators

/-- The sum of the odd powers `ζ^{(2j + 1) e}`, `j < width`. -/
def oddTrace {R : Type} [CommRing R] (width : Nat) (root : R) (exponent : Nat) : R :=
  ∑ j ∈ Finset.range width, root ^ ((2 * j + 1) * exponent)

private theorem oddTrace_double {R : Type} [CommRing R] (width exponent : Nat) (root : R)
    (hroot : root ^ (2 * width) = -1) :
    oddTrace (2 * width) root exponent =
      oddTrace width root exponent + oddTrace width root exponent * (-1) ^ exponent := by
  unfold oddTrace
  rw [show 2 * width = width + width by omega, Finset.sum_range_add]
  congr 1
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro j hj
  rw [show (2 * (width + j) + 1) * exponent =
    (2 * j + 1) * exponent + (2 * width) * exponent by ring, pow_add, pow_mul root (2 * width) exponent, hroot]

private theorem oddTrace_even {R : Type} [CommRing R] (width exponent : Nat) (root : R) :
    oddTrace width root (2 * exponent) = oddTrace width (root ^ 2) exponent := by
  unfold oddTrace
  apply Finset.sum_congr rfl
  intro j hj
  rw [← pow_mul]
  congr 1
  ring

/-- Exact odd-residue trace for every power-of-two width. Cancellation is
proved inside the ring; no field or integral-domain hypothesis is required. -/
theorem oddTrace_power_two {R : Type} [CommRing R] (log exponent : Nat) (root : R)
    (hroot : root ^ (2 ^ log) = -1) (hexponent : exponent < 2 * 2 ^ log) :
    oddTrace (2 ^ log) root exponent =
      if exponent = 0 then ((2 ^ log : Nat) : R)
      else if exponent = 2 ^ log then -((2 ^ log : Nat) : R) else 0 := by
  induction log generalizing root exponent with
  | zero =>
    have he : exponent = 0 ∨ exponent = 1 := by omega
    rcases he with rfl | rfl
    · simp [oddTrace]
    · simpa [oddTrace] using hroot
  | succ log ih =>
    have hwidth : 2 ^ (log + 1) = 2 * 2 ^ log := by rw [pow_succ]; ring
    rw [hwidth] at hroot hexponent ⊢
    rw [oddTrace_double _ _ _ hroot]
    rcases Nat.even_or_odd exponent with heven | hodd
    · obtain ⟨half, rfl⟩ := heven
      have htwice : half + half = 2 * half := by omega
      rw [htwice, oddTrace_even]
      have hroot' : (root ^ 2) ^ (2 ^ log) = -1 := by rw [← pow_mul]; exact hroot
      rw [ih half (root ^ 2) hroot' (by omega)]
      have heven : Even (2 * half) := by exact even_two_mul half
      rw [heven.neg_one_pow]
      split_ifs <;> simp_all <;> ring
    · rw [hodd.neg_one_pow]
      have hparity := Nat.odd_iff.mp hodd
      have hnotzero : exponent ≠ 0 := by omega
      have hnotwidth : exponent ≠ 2 * 2 ^ log := by omega
      simp [hnotzero, hnotwidth]

private theorem power_mod {R : Type} [CommRing R] (root : R) (period exponent : Nat)
    (hperiod : root ^ period = 1) : root ^ exponent = root ^ (exponent % period) := by
  conv_lhs => rw [← Nat.div_add_mod exponent period]
  rw [pow_add, pow_mul, hperiod]
  simp

private def geometricTrace {R : Type} [CommRing R] (width : Nat) (root : R) (exponent : Nat) : R :=
  ∑ j ∈ Finset.range width, root ^ (j * exponent)

/-- Power-of-two Fourier orthogonality is proved by exact cancellation in
arbitrary commutative rings, including the composite CRT modulus used by Rust. -/
private theorem geometricTrace_power_two {R : Type} [CommRing R]
    (log exponent : Nat) (root : R) (hroot : root ^ (2 ^ log) = -1) :
    geometricTrace (2 ^ log) (root ^ 2) exponent =
      if exponent % (2 ^ log) = 0 then ((2 ^ log : Nat) : R) else 0 := by
  have hw : 0 < 2 ^ log := by positivity
  have hp : root ^ (2 * 2 ^ log) = 1 := by rw [Nat.mul_comm, pow_mul, hroot]; ring
  have hp2 : (root ^ 2) ^ (2 ^ log) = 1 := by rw [← pow_mul]; exact hp
  have hm : geometricTrace (2 ^ log) (root ^ 2) exponent =
      geometricTrace (2 ^ log) (root ^ 2) (exponent % (2 ^ log)) := by
    unfold geometricTrace
    apply Finset.sum_congr rfl
    intro j hj
    rw [Nat.mul_comm j exponent, pow_mul, power_mod (root ^ 2) (2 ^ log) exponent hp2,
      ← pow_mul, Nat.mul_comm]
  rw [hm]
  apply ((IsUnit.of_pow_eq_one hp (by omega)).pow (exponent % (2 ^ log))).mul_left_cancel
  have htrace : root ^ (exponent % (2 ^ log)) *
      geometricTrace (2 ^ log) (root ^ 2) (exponent % (2 ^ log)) =
      oddTrace (2 ^ log) root (exponent % (2 ^ log)) := by
    unfold geometricTrace oddTrace
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j hj
    rw [← pow_mul, ← pow_add]
    congr 1
    ring
  rw [htrace, oddTrace_power_two log _ root hroot (by have := Nat.mod_lt exponent hw; omega)]
  have hne : exponent % (2 ^ log) ≠ 2 ^ log := Nat.ne_of_lt (Nat.mod_lt _ hw)
  by_cases he : exponent % (2 ^ log) = 0 <;> simp [he, hne]

/-- On the strided root, the width-W subgroup permutes `1+(2M/W)j`, not
necessarily all odd native residues when the LUT width is smaller than M. -/
theorem lut_subgroup_trace {R : Type} [CommRing R]
    (log nativeWidth quotient exponent : Nat) (root : R)
    (hnative : nativeWidth = quotient * 2 ^ log) (hroot : root ^ nativeWidth = -1)
    (index : Fin (2 ^ log) → Nat) (permutation : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * nativeWidth) = 1 + 2 * quotient * (permutation i).val) :
    (∑ i, root ^ (index i * exponent)) =
      root ^ exponent * (if exponent % (2 ^ log) = 0 then ((2 ^ log : Nat) : R) else 0) := by
  have hp : root ^ (2 * nativeWidth) = 1 := by rw [Nat.mul_comm, pow_mul, hroot]; ring
  have hr : (root ^ quotient) ^ (2 ^ log) = -1 := by rw [← pow_mul, ← hnative]; exact hroot
  calc
    _ = ∑ i, root ^ ((1 + 2 * quotient * (permutation i).val) * exponent) := by
      apply Finset.sum_congr rfl
      intro i hi
      rw [pow_mul, power_mod root (2 * nativeWidth) (index i) hp, hindex i, ← pow_mul]
    _ = ∑ i : Fin (2 ^ log), root ^ ((1 + 2 * quotient * i.val) * exponent) :=
      permutation.sum_comp (fun i => root ^ ((1 + 2 * quotient * i.val) * exponent))
    _ = root ^ exponent * geometricTrace (2 ^ log) ((root ^ quotient) ^ 2) exponent := by
      unfold geometricTrace
      rw [← Fin.sum_univ_eq_sum_range, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i hi
      rw [← pow_mul, ← pow_mul, ← pow_add]
      congr 1
      ring
    _ = _ := by rw [geometricTrace_power_two log exponent (root ^ quotient) hr]

private theorem difference_divisible {width nativeWidth exponent index : Nat}
    (hnpos : 0 < nativeWidth) (hi : index < width) (hn : width ∣ nativeWidth) :
    (exponent + 2 * nativeWidth - index) % width = 0 ↔ exponent % width = index := by
  have hwn : width ≤ nativeWidth := Nat.le_of_dvd hnpos hn
  have hperiod : 2 * nativeWidth ≡ 0 [MOD width] :=
    Nat.modEq_zero_iff_dvd.mpr (dvd_mul_of_dvd_right hn 2)
  have htotal : (exponent + 2 * nativeWidth - index) + index ≡ exponent [MOD width] := by
    rw [Nat.sub_add_cancel (by omega)]
    simpa using (show exponent ≡ exponent [MOD width] from Nat.ModEq.rfl).add hperiod
  have hequiv : (exponent + 2 * nativeWidth - index) ≡ 0 [MOD width] ↔ exponent ≡ index [MOD width] := by
    constructor
    · intro h
      exact htotal.symm.trans (by simpa using h.add_right index)
    · intro h
      exact Nat.ModEq.add_right_cancel' index (by simpa using htotal.trans h)
  simpa only [Nat.ModEq, Nat.zero_mod, Nat.mod_eq_of_lt hi] using hequiv

/-- Exactly `encoding.rs::lut_coefficient`: W inverse times the sum of
`mono(L(k)) * sigma(mono(-k))`, retaining the declared table order. -/
def lutFourierCoefficient {R : Type} [CommRing R] {width : Nat}
    (nativeWidth : Nat) (root inverse : R) (table : Fin width → R) (index : Nat) : R :=
  inverse * ∑ k, table k * root ^ (index * (2 * nativeWidth - k.val))

private theorem lut_fourier_double_sum {R : Type} [CommRing R] {width : Nat}
    (nativeWidth exponent : Nat) (root inverse : R) (table : Fin width → R)
    (index : Fin width → Nat) (hwidth : width ≤ nativeWidth) :
    (∑ i, root ^ (index i * exponent) * lutFourierCoefficient nativeWidth root inverse table (index i)) =
      inverse * ∑ k, table k * (∑ i, root ^ (index i * (exponent + 2 * nativeWidth - k.val))) := by
  unfold lutFourierCoefficient
  simp_rw [← mul_assoc, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k hk
  apply Finset.sum_congr rfl
  intro i hi
  have hsum : index i * exponent + index i * (2 * nativeWidth - k.val) =
      index i * (exponent + 2 * nativeWidth - k.val) := by
    rw [← Nat.mul_add]
    have hkbound := k.isLt
    congr 1
    omega
  rw [← hsum, pow_add]
  ring

/-- Actual LUT evaluation on the full native exponent domain. Outside its
width-W lookup domain the implementation retains a monomial factor; this
faithful extension still preserves signed-monomial plaintexts. -/
theorem lut_fourier_interpolation {R : Type} [CommRing R]
    (log nativeWidth quotient exponent : Nat) (root inverse : R)
    (table : Fin (2 ^ log) → R) (hnative : nativeWidth = quotient * 2 ^ log)
    (hnpos : 0 < nativeWidth) (hroot : root ^ nativeWidth = -1)
    (hinverse : inverse * ((2 ^ log : Nat) : R) = 1)
    (index : Fin (2 ^ log) → Nat) (permutation : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * nativeWidth) = 1 + 2 * quotient * (permutation i).val) :
    (∑ i, root ^ (index i * exponent) * lutFourierCoefficient nativeWidth root inverse table (index i)) =
      root ^ (exponent - exponent % (2 ^ log)) * table ⟨exponent % (2 ^ log), Nat.mod_lt _ (by positivity)⟩ := by
  have hw : 0 < 2 ^ log := by positivity
  have hdvd : 2 ^ log ∣ nativeWidth := by rw [hnative]; exact dvd_mul_left _ _
  have hwidth : 2 ^ log ≤ nativeWidth := Nat.le_of_dvd hnpos hdvd
  rw [lut_fourier_double_sum nativeWidth exponent root inverse table index hwidth]
  simp_rw [lut_subgroup_trace log nativeWidth quotient _ root hnative hroot index permutation hindex]
  let chosen : Fin (2 ^ log) := ⟨exponent % (2 ^ log), Nat.mod_lt _ hw⟩
  rw [Finset.sum_eq_single chosen]
  · have hselected : (exponent + 2 * nativeWidth - chosen.val) % (2 ^ log) = 0 :=
      (difference_divisible hnpos chosen.isLt hdvd).mpr rfl
    rw [if_pos hselected]
    have hexp : exponent + 2 * nativeWidth - chosen.val =
        (exponent - chosen.val) + 2 * nativeWidth := by
      have := Nat.mod_le exponent (2 ^ log)
      dsimp [chosen]
      omega
    have hp : root ^ (2 * nativeWidth) = 1 := by rw [Nat.mul_comm, pow_mul, hroot]; ring
    rw [hexp, pow_add, hp, mul_one]
    calc
      _ = (inverse * ((2 ^ log : Nat) : R)) * (root ^ (exponent - chosen.val) * table chosen) := by ring
      _ = _ := by rw [hinverse, one_mul]
  · intro k hk hne
    have hnot : (exponent + 2 * nativeWidth - k.val) % (2 ^ log) ≠ 0 := by
      intro h
      apply hne
      apply Fin.ext
      exact ((difference_divisible hnpos k.isLt hdvd).mp h).symm
    rw [if_neg hnot, mul_zero, mul_zero]
  · simp

/-- In the declared lookup domain the monomial factor is one, giving the
actual table entry rather than an assumed interpolation axiom. -/
theorem lut_fourier_table_value {R : Type} [CommRing R]
    (log nativeWidth quotient : Nat) (root inverse : R)
    (table : Fin (2 ^ log) → R) (input : Fin (2 ^ log))
    (hnative : nativeWidth = quotient * 2 ^ log) (hnpos : 0 < nativeWidth)
    (hroot : root ^ nativeWidth = -1) (hinverse : inverse * ((2 ^ log : Nat) : R) = 1)
    (index : Fin (2 ^ log) → Nat) (permutation : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * nativeWidth) = 1 + 2 * quotient * (permutation i).val) :
    (∑ i, root ^ (index i * input.val) * lutFourierCoefficient nativeWidth root inverse table (index i)) =
      table input := by
  rw [lut_fourier_interpolation log nativeWidth quotient input.val root inverse table
    hnative hnpos hroot hinverse index permutation hindex]
  simp [Nat.mod_eq_of_lt input.isLt]

end MxxExponentLut
