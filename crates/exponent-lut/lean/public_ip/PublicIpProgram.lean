import ExponentRuntime

/-!
# The public inner product

`PublicInnerProductProgram` multiplies the public exponent encodings of the coordinates of `a`
selected by its hash-sampled binary vector `t`. A plaintext `c X^e` with `c² = 1`, `0 ≤ e < n`, and
`2 e + n (1 - c) = 2 x` is the monomial `X^x`, so the product of the selected encodings has the
plaintext `X^(Σ x_k)`, and a claimed plaintext accepted for `x = stride (<a, t> mod Q)` is that
product.
-/

namespace MxxExponentLutPublicIp

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {n : Nat}

/-- An accepted signed monomial `c X^e` is `X^x`. -/
theorem message_eq_Xe {c e x : Int} (hc : c * c = 1) (hrel : e * 2 + (1 - c) * n = x * 2)
    (he : 0 ≤ e ∧ e < n) : message (n := n) c e = Xe x.toNat := by
  rcases Int.eq_one_or_neg_one_of_mul_eq_one hc with rfl | rfl
  · exact message_one rfl (by omega) (by omega)
  · have hmod : e % ((2 * n : Nat) : Int) = e := Int.emod_eq_of_lt he.1 (by push_cast; omega)
    rw [message, hmod]
    have hsplit : x.toNat = e.toNat + n := by omega
    rw [Xe, hsplit, pow_add, Negacyclic.root_pow_n]
    push_cast
    ring

theorem Xe_mul (a b : Nat) : Xe (n := n) a * Xe b = Xe (a + b) := by
  rw [Xe, Xe, Xe, pow_add]

/-- `X^(2n)` is one. -/
theorem Xe_period (a k : Nat) : Xe (n := n) (a + 2 * n * k) = Xe a := by
  rw [Xe, Xe, pow_add, show 2 * n * k = n * (2 * k) by ring, pow_mul, Negacyclic.root_pow_n,
    pow_mul, neg_one_sq, one_pow, mul_one]

end MxxExponentLutPublicIp
