import TopLemmas

/-!
# The label payloads

The online stage bakes the label's monomials `±X^e` in as constants; they are `X^(2 a_i)` with
`a = labelExp` (`X^8 = -1`).
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The label exponents: the payload of label member `i` is `X^(2 a_i)`. -/
def labelExp : Fin 5 → Nat := ![3, 3, 3, 6, 0]

theorem label_value {q : Nat} (i : Fin 5) :
    multiplyMonomial (liftInteger (MxxIR.selectInt (Int.ofNat i) [(1), (1), (1), (-1), (1)]) :
      ExactMatrix q N 1 1) (MxxIR.selectInt (Int.ofNat i) [(6), (6), (6), (4), (0)]) =
      fun _ _ ↦ reducePoly q N (Xe (n := N) (2 * labelExp i)) := by
  rw [multiplyMonomial_liftInteger]
  have h8 : (AdjoinRoot.root (negacyclicModulus N ℤ)) ^ 8 = -1 := Negacyclic.root_pow_n
  funext r c
  congr 1
  fin_cases i <;> simp [message, labelExp, MxxIR.selectInt, Xe, N]
  rw [show (12 : Nat) = 8 + 4 from rfl, pow_add, h8]; ring

end RefreshTopProof

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- A label encoding's plaintext `c X^e` with `c² = 1`, `2 e + 8 (1 - c) = 4 a`, `0 ≤ e ≤ 7` is
`X^(2 a)`. -/
theorem label_message {c e a : Int} (hc : c * c = 1) (hr : e * 2 + (1 - c) * 8 = a * 4)
    (he0 : 0 ≤ e) (he7 : e ≤ 7) :
    message (n := N) c e = Xe (n := N) (2 * a.toNat) := by
  have h8 : (AdjoinRoot.root (negacyclicModulus N ℤ)) ^ 8 = -1 := Negacyclic.root_pow_n
  have hc' : c = 1 ∨ c = -1 := by
    have : (c - 1) * (c + 1) = 0 := by ring_nf; linarith
    rcases mul_eq_zero.mp this with h | h
    · left; linarith
    · right; linarith
  unfold message Xe
  have hmod : (e % (2 * (N : Nat) : Nat)).toNat = e.toNat := by
    rw [Int.emod_eq_of_lt he0 (by simp only [N]; omega)]
  rw [hmod]
  rcases hc' with rfl | rfl
  · have : 2 * a.toNat = e.toNat := by omega
    rw [this]; simp
  · have : 2 * a.toNat = e.toNat + 8 := by omega
    rw [this, pow_add, h8]; simp

end RefreshTopProof

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem message_zero (c : Int) : message (n := N) c 0 = ((c : Int) : ErrorPoly N) := by
  simp [message]

theorem message_one_zero : message (n := N) 1 0 = 1 := by
  simp [message]

end RefreshTopProof

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem message_one_exp {e : Int} (h0 : 0 ≤ e) (h1 : e < 16) :
    message (n := N) 1 e = Xe (n := N) e.toNat := by
  unfold message Xe
  rw [Int.emod_eq_of_lt h0 (by simp only [N]; omega)]
  simp

end RefreshTopProof
