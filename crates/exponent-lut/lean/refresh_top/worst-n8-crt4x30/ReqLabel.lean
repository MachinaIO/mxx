import RequirementFacts
import TopLabel

/-! The requirement's label check: the label's exponents are `labelExp`, and its encodings encode
`c X^e = X^(2 a)` (`c² = 1`, `2 e + 8 (1 - c) = 4 a`). -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem reqLabel {i : Nat} {cur next : Bool} {a c e : Fin 5 → Int}
    (h : Requirement_0.sequential_generatedRoot_6 GeneratedClaim.requirement_0_params i
      (cur, a, c, e, ()) next) (hn : next = true) :
    cur = true ∧ ∀ hi : i < 5, a ⟨i, hi⟩ = (labelExp ⟨i, hi⟩ : Int) ∧
      c ⟨i, hi⟩ * c ⟨i, hi⟩ = 1 ∧ e ⟨i, hi⟩ * 2 + (1 - c ⟨i, hi⟩) * 8 = a ⟨i, hi⟩ * 4 := by
  unfold Requirement_0.sequential_generatedRoot_6 at h
  dsimp only at h
  obtain ⟨w4, w20, w28, w38, w46, h⟩ := h
  simp only [Requirement_0.sequential_generatedRoot_6.constraints_0] at h
  obtain ⟨-, -, h4, -, -, -, h20, -, -, h28, -, -, h38, -, -, h46, hout⟩ := h
  obtain ⟨hi, e4⟩ := familyGet_eq h4
  obtain ⟨_, e28⟩ := familyGet_eq h28
  obtain ⟨_, e38⟩ := familyGet_eq h38
  obtain ⟨_, e46⟩ := familyGet_eq h46
  have e20 := select_eq_selectInt h20
  rw [e4, e28, e38, e46, e20] at hout
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  obtain ⟨⟨⟨hc, ha⟩, hm⟩, hr⟩ := hout
  refine ⟨by simpa using hc, fun hi' ↦ ⟨?_, hm, by linarith⟩⟩
  rw [ha]
  interval_cases i <;> rfl

end RefreshTopProof
