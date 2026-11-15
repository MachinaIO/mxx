import MathcalA.Lib.Prog
import MxxCore.Sym

/-!
# Words of naturals

Rewrites of machine operations on words `W x` of naturals, for the leaves' specifications.
-/

namespace MxxLatticeLeaves.Math.Words

open MathcalA RiscvZkvm.Rv64

/-- `MULHU` of two words: the high word of their product. -/
theorem mulhu_W {x y : Nat} (hx : x < 2 ^ 64) (hy : y < 2 ^ 64) :
    rv64_mulhu (W x) (W y) = W (x * y / 2 ^ 64) := by
  apply BitVec.eq_of_toNat_eq
  have hxy : x * y < 2 ^ 128 := by
    calc x * y < 2 ^ 64 * 2 ^ 64 := Nat.mul_lt_mul'' hx hy
      _ = 2 ^ 128 := by norm_num
  have hq : x * y / 2 ^ 64 < 2 ^ 64 := by omega
  have h1 : x % 18446744073709551616 = x := Nat.mod_eq_of_lt (by omega)
  have h2 : y % 18446744073709551616 = y := Nat.mod_eq_of_lt (by omega)
  have h3 : x * y % 340282366920938463463374607431768211456 = x * y := Nat.mod_eq_of_lt (by omega)
  have h4 : x * y / 18446744073709551616 % 18446744073709551616 = x * y / 18446744073709551616 :=
    Nat.mod_eq_of_lt (by omega)
  simp [rv64_mulhu, BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow, h1, h2, h3, h4]

/-- The difference of two words of naturals `b ≤ a`. -/
theorem sub_W {a b : Nat} (h : b ≤ a) : W a - W b = W (a - b) := by
  simp only [W]
  conv_lhs => rw [show a = (a - b) + b by omega]
  rw [← BitVec.ofNat_add_ofNat, BitVec.add_sub_cancel]

/-- `x << 3` of a word is `8 x`. -/
theorem shl3_W {x : Nat} (hx : 8 * x < 2 ^ 64) : BitVec.ofNat 64 x <<< 3 = W (8 * x) := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_shiftLeft, BitVec.toNat_ofNat, BitVec.toNat_ofNat, Nat.shiftLeft_eq,
    Nat.mod_eq_of_lt (show x < 2 ^ 64 by omega)]
  congr 1; ring

/-- `x << k` of a word is `x 2^k`. -/
theorem shl_W {x k : Nat} : BitVec.ofNat 64 x <<< k = W (x * 2 ^ k) := by
  apply BitVec.eq_of_toNat_eq
  simp only [W, BitVec.toNat_shiftLeft, BitVec.toNat_ofNat, Nat.shiftLeft_eq]
  rw [Nat.mod_mul_mod]

/-- `x >> k` of a word `x` is `x / 2^k`. -/
theorem srl_W {x k : Nat} (hx : x < 2 ^ 64) : BitVec.ofNat 64 x >>> k = W (x / 2 ^ k) := by
  apply BitVec.eq_of_toNat_eq
  have : x / 2 ^ k < 2 ^ 64 := lt_of_le_of_lt (Nat.div_le_self _ _) hx
  simp only [W, BitVec.toNat_ushiftRight, BitVec.toNat_ofNat, Nat.shiftRight_eq_div_pow,
    Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt this]

end MxxLatticeLeaves.Math.Words
