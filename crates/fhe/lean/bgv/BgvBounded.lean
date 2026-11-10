import Simulators

/-! The cost of the simulators of `Simulators.lean`, by computation: each is at most the bound
`2^30` of `GeneratedScheme.BoundedSecurityClaim`. `BgvReductions` proves the rest of that claim. -/

set_option maxRecDepth 200000

namespace BgvBounded
open GeneratedSimulators

theorem simulator_0_cost : simulator_0.cost = some 605757440 := by decide +kernel
theorem simulator_1_cost : simulator_1.cost = some 605659136 := by decide +kernel
theorem simulator_2_cost : simulator_2.cost = some 202760192 := by decide +kernel

theorem simulators_within (i : Fin 3) :
    ∃ cost, ([simulator_0, simulator_1, simulator_2].get ⟨i, by simp⟩).cost = some cost ∧
      cost ≤ 1073741824 := by
  fin_cases i
  · exact ⟨_, simulator_0_cost, by decide⟩
  · exact ⟨_, simulator_1_cost, by decide⟩
  · exact ⟨_, simulator_2_cost, by decide⟩

end BgvBounded
