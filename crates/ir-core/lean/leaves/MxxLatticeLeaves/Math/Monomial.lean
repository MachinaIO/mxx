import Mathlib.Tactic

/-!
# The arithmetic of `lattice.multiply_monomial`

With `N = 2^m`, the leaf computes `-k mod 2 N` and `(p - k) mod 2 N` with a mask `2 N - 1` on
words, and walks the output as one flat array: word `j` is coefficient `j % N` of entry `j / N`.
-/

namespace MxxLatticeLeaves.Math.Monomial

/-- Masking with `2^m - 1` is the remainder modulo `2^m`, on words. -/
theorem and_mask {x m : Nat} (hm : m ≤ 64) :
    (BitVec.ofNat 64 x &&& BitVec.ofNat 64 (2 ^ m - 1)).toNat = x % 2 ^ 64 % 2 ^ m := by
  have hlt : 2 ^ m - 1 < 2 ^ 64 := by
    have := Nat.pow_le_pow_right (show 0 < 2 by norm_num) hm
    omega
  rw [BitVec.toNat_and, BitVec.toNat_ofNat, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hlt,
    Nat.and_two_pow_sub_one_eq_mod]

/-- `2^m` divides `2^64` for `m ≤ 64`, so reducing modulo `2^64` first does not change a remainder
modulo `2^m`. -/
theorem mod_mod_pow {x m : Nat} (hm : m ≤ 64) : x % 2 ^ 64 % 2 ^ m = x % 2 ^ m :=
  Nat.mod_mod_of_dvd x (Nat.pow_dvd_pow 2 hm)

/-- The negation of `k` modulo `2^m`, on words: `(2^m - k mod 2^m) mod 2^m`. -/
theorem neg_mask {k m : Nat} (hm : m ≤ 64) (hk : k < 2 ^ 64) :
    ((0#64 - BitVec.ofNat 64 k) &&& BitVec.ofNat 64 (2 ^ m - 1)).toNat =
      (2 ^ m - k % 2 ^ m) % 2 ^ m := by
  have hlt : 2 ^ m - 1 < 2 ^ 64 := by
    have := Nat.pow_le_pow_right (show 0 < 2 by norm_num) hm
    omega
  rw [BitVec.toNat_and, BitVec.toNat_ofNat (2 ^ m - 1), Nat.mod_eq_of_lt hlt,
    Nat.and_two_pow_sub_one_eq_mod, BitVec.toNat_sub, BitVec.toNat_ofNat, BitVec.toNat_ofNat,
    Nat.zero_mod, Nat.add_zero, Nat.mod_eq_of_lt hk, mod_mod_pow hm]
  -- (2^64 - k) mod 2^m = (2^m - k mod 2^m) mod 2^m, as 2^m divides 2^64.
  have hdvd : 2 ^ m ∣ 2 ^ 64 := Nat.pow_dvd_pow 2 hm
  obtain ⟨c, hc⟩ := hdvd
  have hpos : 0 < 2 ^ m := by positivity
  have hkm : k % 2 ^ m < 2 ^ m := Nat.mod_lt _ hpos
  have hsplit : k = 2 ^ m * (k / 2 ^ m) + k % 2 ^ m := (Nat.div_add_mod k (2 ^ m)).symm
  have hc1 : 1 ≤ c := by
    rcases Nat.eq_zero_or_pos c with h | h
    · rw [h, Nat.mul_zero] at hc; norm_num at hc
    · exact h
  have hkc : k / 2 ^ m < c := by
    rw [Nat.div_lt_iff_lt_mul hpos]; rw [hc] at hk; linarith [Nat.mul_comm c (2 ^ m)]
  have key : 2 ^ 64 - k = (2 ^ m - k % 2 ^ m) + 2 ^ m * (c - 1 - k / 2 ^ m) := by
    generalize hK1 : k / 2 ^ m = K1 at hsplit hkc
    generalize hK0 : k % 2 ^ m = K0 at hsplit hkm
    generalize hM : 2 ^ m = M at hsplit hkm hkc hc hpos
    subst hsplit
    rw [hc]
    have h1 : M * K1 + K0 ≤ M * c := by nlinarith
    have h2 : M * (c - 1 - K1) + M * K1 + M = M * c := by
      rw [← Nat.mul_add, ← Nat.mul_succ]; congr 1; omega
    omega
  rw [key, Nat.add_mul_mod_self_left]

/-- The flat index `e N + p` (`p < N`) is coefficient `p` of entry `e`. -/
theorem flat {e p N : Nat} (hp : p < N) : (e * N + p) / N = e ∧ (e * N + p) % N = p := by
  have hN : 0 < N := by omega
  refine ⟨?_, ?_⟩
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt hp, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hp]

end MxxLatticeLeaves.Math.Monomial
