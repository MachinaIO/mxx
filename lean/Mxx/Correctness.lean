import Mxx.Assumptions
import Mathlib.Data.ENNReal.Basic

namespace Mxx

abbrev FailureProbability := ENNReal

def booleanFailureProbability {α : Type} (outcomes : List α) (bad : α → Bool) : ENNReal :=
  if outcomes.any bad then 1 else 0

theorem booleanFailureProbability_eq_zero {α : Type} (outcomes : List α) (bad : α → Bool)
    (safe : ∀ output ∈ outcomes, bad output = false) :
    booleanFailureProbability outcomes bad = 0 := by
  unfold booleanFailureProbability
  rw [if_neg]
  intro failure
  simp only [List.any_eq_true] at failure
  obtain ⟨output, member, failed⟩ := failure
  rw [safe output member] at failed
  contradiction

end Mxx
