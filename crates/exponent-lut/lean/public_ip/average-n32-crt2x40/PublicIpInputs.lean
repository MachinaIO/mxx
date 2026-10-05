import PublicIpSetup

/-!
# The requirement and the ideal output

The requirement checks every member of the input family in one loop: it accepts plaintext monomials
`c X^e` with `c X^e = X^(2 a)`. It also makes the last member zero and the claimed plaintext
`X^(2 ((a₀ + a₁ + a₃) mod 32))`, the inner product with `t = (1, 1, 0, 1)`. The ideal residual
count is zero.
-/

namespace PublicIpProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {tape : SampleTape} {path : List Nat}

theorem ite_add_ite_eq_two (p q : Prop) [Decidable p] [Decidable q] :
    ((if p then (1 : Int) else 0) + (if q then 1 else 0) = 2) ↔ p ∧ q := by
  by_cases hp : p <;> by_cases hq : q <;> simp [hp, hq]

/-- One iteration of the member loop keeps a true state only for an accepted member. -/
theorem member_step {ac ae a : Fin 5 → Int} {i : Nat} {current next : Bool}
    (h : Requirement_0.sequential_generatedRoot_6 tape path GeneratedClaim.requirement_0_params i
      (current, ac, ae, a, ()) next) (hnext : next = true) :
    current = true ∧ ∀ hi : i < 5, ac ⟨i, hi⟩ * ac ⟨i, hi⟩ = 1 ∧
      ae ⟨i, hi⟩ * 2 + (1 - ac ⟨i, hi⟩) * 32 = a ⟨i, hi⟩ * 4 := by
  unfold Requirement_0.sequential_generatedRoot_6 at h
  dsimp only at h
  obtain ⟨w4, w10, w19, -, -, h4, -, -, h10, -, -, h19, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h4
  obtain ⟨hi10, rfl⟩ := familyGet_eq h10
  obtain ⟨hi19, rfl⟩ := familyGet_eq h19
  subst hnext
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  obtain ⟨hc, hm, hr⟩ := hout
  exact ⟨by simpa using hc, fun _ ↦ ⟨hm, hr⟩⟩

set_option maxHeartbeats 4000000 in
/-- The requirement: every member is an accepted monomial, the last member is zero, and the claimed
plaintext is accepted for `(a₀ + a₁ + a₃) mod 32`. -/
theorem requirement_holds {ac ae a : Fin 5 → Int} {ec ee : Int} {out : Bool}
    (h : Requirement_0.generatedRoot tape path GeneratedClaim.requirement_0_params
      (ac, ae, a, ec, ee, ()) out) (ht : out = true) :
    (∀ k : Fin 5, ac k * ac k = 1 ∧ ae k * 2 + (1 - ac k) * 32 = a k * 4) ∧ a 4 = 0 ∧
      ec * ec = 1 ∧ ee * 2 + (1 - ec) * 32 = (a 0 + a 1 + a 3) % 32 * 4 := by
  obtain ⟨w, hb⟩ := h
  unfold Requirement_0.generatedRoot.body at hb
  dsimp only at hb
  obtain ⟨-, hm, h8, h30, h32, h34, -, hout⟩ := hb
  subst ht
  obtain ⟨_, e8⟩ := familyGetStatic_eq (k := 4) h8
  obtain ⟨_, e30⟩ := familyGetStatic_eq (k := 0) h30
  obtain ⟨_, e32⟩ := familyGetStatic_eq (k := 1) h32
  obtain ⟨_, e34⟩ := familyGetStatic_eq (k := 3) h34
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  obtain ⟨⟨hw6, ha4⟩, hc, hrel⟩ := hout
  have hmem := (IterRuns.all_of_step (fun i ↦ ∀ hi : i < 5, ac ⟨i, hi⟩ * ac ⟨i, hi⟩ = 1 ∧
      ae ⟨i, hi⟩ * 2 + (1 - ac ⟨i, hi⟩) * 32 = a ⟨i, hi⟩ * 4)
    (fun i current next _ hb hn ↦ member_step hb hn) hm (by simpa using hw6)).2
  refine ⟨fun k ↦ hmem k.val (lt_of_lt_of_eq k.isLt rfl) k.isLt, ?_, hc, ?_⟩
  · rw [show a 4 = a ⟨4, by decide⟩ from rfl, ← e8, ha4]
  · rw [hrel, e30, e32, e34, zero_add]
    rfl

/-- The ideal residual count is zero. -/
theorem ideal_value {out : Int}
    (h : Ideal.generatedRoot tape path GeneratedClaim.ideal_params () out) : out = 0 := by
  obtain ⟨_, h⟩ := h
  simpa [Ideal.generatedRoot.body, Ideal.generatedRoot.constraints_0] using h

end PublicIpProof
