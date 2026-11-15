import Mathlib.Tactic

/-!
# The arithmetic of the inline modular multiplication

`MxxLatticeLeaves.MulMod` reduces the 128-bit product `hi 2^64 + lo` modulo `q` by 64 steps of
binary long division: step `i` doubles the remainder and adds bit `63 - i` of `lo`, read as the top
bit of `lo 2^i` modulo `2^64`.
-/

namespace MxxLatticeLeaves.Math.MulMod

/-- The remainder after `i` steps. -/
def rem (q hi lo i : Nat) : Nat := (hi * 2 ^ i + lo / 2 ^ (64 - i)) % q

/-- Bit `63 - i` of `lo` is the top bit of `lo 2^i` modulo `2^64`. -/
theorem top_bit {lo i : Nat} (hi : i < 64) :
    lo * 2 ^ i % 2 ^ 64 / 2 ^ 63 = lo / 2 ^ (63 - i) % 2 := by
  have e64 : 2 ^ 64 = 2 ^ (64 - i) * 2 ^ i := by rw [← pow_add]; congr 1; omega
  have e63 : 2 ^ 63 = 2 ^ (63 - i) * 2 ^ i := by rw [← pow_add]; congr 1; omega
  have e2 : 2 ^ (64 - i) = 2 ^ (63 - i) * 2 := by rw [← pow_succ]; congr 1; omega
  rw [e64, Nat.mul_mod_mul_right, e63, Nat.mul_div_mul_right _ _ (by positivity), e2,
    Nat.mod_mul_right_div_self]

/-- The bits of `lo` from `63 - i` on are twice those from `64 - i` on, plus bit `63 - i`. -/
theorem shift_split {lo i : Nat} (hi : i < 64) :
    lo / 2 ^ (63 - i) = 2 * (lo / 2 ^ (64 - i)) + lo / 2 ^ (63 - i) % 2 := by
  have e2 : 2 ^ (64 - i) = 2 ^ (63 - i) * 2 := by rw [← pow_succ]; congr 1; omega
  rw [e2, ← Nat.div_div_eq_div_mul]
  exact (Nat.div_add_mod _ 2).symm

/-- One step: the doubled remainder plus the bit, reduced, is the next remainder. -/
theorem rem_step {q hi lo i : Nat} (hq : 0 < q) (hi64 : i < 64) :
    (2 * rem q hi lo i + lo * 2 ^ i % 2 ^ 64 / 2 ^ 63) % q = rem q hi lo (i + 1) := by
  unfold rem
  rw [top_bit hi64, show 64 - (i + 1) = 63 - i by omega, pow_succ]
  conv_rhs => rw [shift_split (lo := lo) hi64]
  have : hi * (2 ^ i * 2) + (2 * (lo / 2 ^ (64 - i)) + lo / 2 ^ (63 - i) % 2) =
      2 * (hi * 2 ^ i + lo / 2 ^ (64 - i)) + lo / 2 ^ (63 - i) % 2 := by ring
  rw [this]
  generalize hi * 2 ^ i + lo / 2 ^ (64 - i) = A
  generalize lo / 2 ^ (63 - i) % 2 = b
  have h := Nat.div_add_mod A q
  have e : 2 * A + b = (2 * (A % q) + b) + q * (2 * (A / q)) := by
    rw [show q * (2 * (A / q)) = 2 * (q * (A / q)) by ring]; omega
  rw [e, Nat.add_mul_mod_self_left]

theorem rem_zero {q hi lo : Nat} (hlo : lo < 2 ^ 64) (hhi : hi < q) : rem q hi lo 0 = hi := by
  unfold rem
  rw [Nat.sub_zero, pow_zero, Nat.mul_one, Nat.div_eq_of_lt hlo, Nat.add_zero,
    Nat.mod_eq_of_lt hhi]

theorem rem_lt {q hi lo i : Nat} (hq : 0 < q) : rem q hi lo i < q := Nat.mod_lt _ hq

/-- After 64 steps, the remainder of the whole product. -/
theorem rem_last {q x y : Nat} :
    rem q (x * y / 2 ^ 64) (x * y % 2 ^ 64) 64 = x * y % q := by
  unfold rem; simp only [Nat.sub_self, pow_zero, Nat.div_one]
  rw [Nat.mul_comm (x * y / 2 ^ 64), Nat.div_add_mod]

/-- The high word of a product of residues is below the modulus. -/
theorem hi_lt {q x y : Nat} (hx : x < q) (hy : y < q) (hq : q < 2 ^ 64) : x * y / 2 ^ 64 < q := by
  rw [Nat.div_lt_iff_lt_mul (by positivity)]
  calc x * y < q * q := Nat.mul_lt_mul'' hx hy
    _ ≤ q * 2 ^ 64 := Nat.mul_le_mul_left q hq.le

end MxxLatticeLeaves.Math.MulMod
