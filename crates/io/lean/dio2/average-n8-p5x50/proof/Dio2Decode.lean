import W_Stage_decode

/-!
# The decode stage

The decode stage reads the canonical coefficients `c_k` of the output decoder's plaintext
`mono(z) = X^(2 z)` modulo the first prime `q`, centers each to `v_k = c_k - q ⌊2 c_k / q⌋`, sums
`k v_k (v_k + 1) / 2 + (k + 8) v_k (v_k - 1) / 2` over the eight coefficients and halves the sum.
On the monomial `X^(2 z)` (`2 z < 8`) the sum is `2 z`, so the stage outputs `z`.
-/

namespace Dio2Proof

open W_Stage_decode

/-- On a coefficient `c ∈ {0, 1}` the loop adds `k c`. -/
theorem decode_step (params : Params) (k : Nat) (hk : k < 8) (current next : Int)
    (family : Fin 8 → Int) (hc : family ⟨k, hk⟩ = 0 ∨ family ⟨k, hk⟩ = 1)
    (h : sequential_generatedRoot_3 params k (current, (family, ())) next) :
    next = current + k * family ⟨k, hk⟩ := by
  obtain ⟨w3, hconstraints⟩ := h
  simp only [sequential_generatedRoot_3.constraints_0] at hconstraints
  obtain ⟨-, -, ⟨position, hposition, hw3⟩, -, -, -, hnext⟩ := hconstraints
  have hpos : position = ⟨k, hk⟩ := by
    apply Fin.ext
    have : ((position.val : Nat) : Int) = (k : Int) := by simpa using hposition
    exact_mod_cast this
  subst hpos
  rw [hnext, hw3]
  rcases hc with hzero | hone
  · rw [hzero]; norm_num
  · rw [hone]; norm_num

/-- The decode stage outputs `z` on the plaintext `X^(2 z)`. -/
theorem decode_monomial (params : Params) (plaintext : Mxx.Primitives.ExactMatrix modulus_0 8 1 1)
    (z : Nat) (hz : 2 * z < 8)
    (hcoeff : ∀ i : Fin 8, (((plaintext 0 0).coeff i).val : Int) = if i.val = 2 * z then 1 else 0)
    (output : Int) (h : generatedRoot params plaintext output) : output = z := by
  obtain ⟨witness, hbody⟩ := h
  simp only [generatedRoot.body, generatedRoot.constraints_0] at hbody
  obtain ⟨hvalues, -, hiter, -, hout⟩ := hbody
  simp only [MxxRuntime.polynomialValues, Bool.false_eq_true, ↓reduceIte] at hvalues
  have hfamily : ∀ i : Fin 8, witness.w_2_0 i = if i.val = 2 * z then 1 else 0 := by
    intro i; rw [hvalues i, hcoeff i]
  -- After `i` coefficients, the sum is `2 z` once `2 z < i`, else `0`.
  have hsum := MxxIR.IterRuns.invariant
    (Invariant := fun i (s : Int) => i ≤ 8 → s = if 2 * z < i then (2 * z : Int) else 0)
    (by intro _; simp)
    (by
      intro i current next hinv hstep hi
      have hi8 : i < 8 := by omega
      have hc := hfamily ⟨i, hi8⟩
      have hnext := decode_step params i hi8 current next witness.w_2_0
        (by rw [hc]; split_ifs <;> simp) hstep
      rw [hnext, hinv (by omega), hc]
      by_cases hiz : i = 2 * z
      · subst hiz; simp
      · simp only [hiz, if_false]; split_ifs <;> omega)
    hiter
  have h8 : Int.toNat 8 = 8 := rfl
  rw [h8] at hsum
  have hfinal := hsum le_rfl
  rw [hout, hfinal, if_pos hz]
  omega

end Dio2Proof
