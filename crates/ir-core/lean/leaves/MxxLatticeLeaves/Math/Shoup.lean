import Mathlib.Tactic

/-!
# The arithmetic of Shoup's modular multiplication

`lattice.scale_mod` computes `c' = floor(c 2^64 / q)` by binary long division and then, per
residue `x`, `c x - floor(x c' / 2^64) q`, which lies in `[0, 2 q)` and is congruent to `c x`.
-/

namespace MxxLatticeLeaves.Math.Shoup

/-- One step of the binary long division of `c 2^i` by `q`: the remainder doubles and the quotient
takes the bit that keeps the remainder below `q`. -/
theorem div_step {c q i qt r : Nat} (h : c * 2 ^ i = qt * q + r) (hr : r < q) :
    c * 2 ^ (i + 1) = (if 2 * r ≥ q then 2 * qt + 1 else 2 * qt) * q +
      (if 2 * r ≥ q then 2 * r - q else 2 * r) ∧
    (if 2 * r ≥ q then 2 * r - q else 2 * r) < q := by
  rw [pow_succ]
  split_ifs with hq
  · constructor
    · have : c * 2 ^ i * 2 = (2 * qt + 1) * q + (2 * r - q) := by rw [h]; zify [hq]; ring
      rw [← Nat.mul_assoc]; exact this
    · omega
  · constructor
    · rw [show c * (2 ^ i * 2) = 2 * (c * 2 ^ i) by ring, h]; ring
    · omega

/-- Shoup's bound: with `c 2^64 = c' q + ρ` (`ρ < q`) and `x < q < 2^64`, the product `c x` lies in
`[t q, t q + 2 q)` for `t = floor(x c' / 2^64)`. -/
theorem shoup {q c c' ρ x : Nat} (hq : q < 2 ^ 64) (hx : x < q) (hdiv : c * 2 ^ 64 = c' * q + ρ)
    (hρ : ρ < q) : x * c' / 2 ^ 64 * q ≤ c * x ∧ c * x < x * c' / 2 ^ 64 * q + 2 * q := by
  set t := x * c' / 2 ^ 64
  set e := x * c' % 2 ^ 64
  have hxc : x * c' = 2 ^ 64 * t + e := (Nat.div_add_mod _ _).symm
  have he : e < 2 ^ 64 := Nat.mod_lt _ (by positivity)
  -- (c x - t q) 2^64 = e q + x ρ, as integers.
  have h1 : (c : ℤ) * 2 ^ 64 = (c' : ℤ) * q + ρ := by exact_mod_cast hdiv
  have h2 : (x : ℤ) * c' = 2 ^ 64 * (t : ℤ) + e := by exact_mod_cast hxc
  have key : (c : ℤ) * x * 2 ^ 64 = (t : ℤ) * q * 2 ^ 64 + ((e : ℤ) * q + (x : ℤ) * ρ) := by
    linear_combination (x : ℤ) * h1 + (q : ℤ) * h2
  have hx' : (x : ℤ) < q := by exact_mod_cast hx
  have hρ' : (ρ : ℤ) < q := by exact_mod_cast hρ
  have hq' : (q : ℤ) < 2 ^ 64 := by exact_mod_cast hq
  have he' : (e : ℤ) < 2 ^ 64 := by exact_mod_cast he
  have hx0 : (0 : ℤ) ≤ x := by positivity
  have hρ0 : (0 : ℤ) ≤ ρ := by positivity
  have he0 : (0 : ℤ) ≤ e := by positivity
  have hq0 : (0 : ℤ) < q := by omega
  have hxρ : (x : ℤ) * ρ < q * 2 ^ 64 := by nlinarith
  have heq : (e : ℤ) * q < 2 ^ 64 * q := by nlinarith
  have hnn : (0 : ℤ) ≤ (e : ℤ) * q + (x : ℤ) * ρ := by positivity
  constructor
  · have : (t : ℤ) * q ≤ (c : ℤ) * x := by nlinarith
    exact_mod_cast this
  · have : (c : ℤ) * x < (t : ℤ) * q + 2 * q := by nlinarith
    exact_mod_cast this

/-- The reduced residue: `c x - t q`, less `q` when at least `q`, is `c x mod q`. -/
theorem shoup_mod {q c c' ρ x : Nat} (hq : q < 2 ^ 64) (hx : x < q) (hdiv : c * 2 ^ 64 = c' * q + ρ)
    (hρ : ρ < q) :
    (if c * x - x * c' / 2 ^ 64 * q ≥ q then c * x - x * c' / 2 ^ 64 * q - q
      else c * x - x * c' / 2 ^ 64 * q) = c * x % q := by
  obtain ⟨h1, h2⟩ := shoup hq hx hdiv hρ
  set t := x * c' / 2 ^ 64
  split_ifs with h
  · have e : c * x = (c * x - t * q - q) + q * (t + 1) := by
      have : q * (t + 1) = t * q + q := by ring
      omega
    conv_rhs => rw [e]
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt (by omega)]
  · have e : c * x = (c * x - t * q) + q * t := by
      have : q * t = t * q := by ring
      omega
    conv_rhs => rw [e]
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt (by omega)]

end MxxLatticeLeaves.Math.Shoup
