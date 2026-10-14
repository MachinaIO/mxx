import BridgeRuns
import Gen_Correct

/-!
# The average-case correctness claim of refresh_top

Every average-case run is a worst-case run (`GeneratedClaim.runs_bridge`), and every worst-case run
decodes the ideal output (`RefreshTopProof.GenCorrect.correct`, ported to the renamed worst-case
export by `generator/gen_port.py`): the set of tapes on which some run fails is empty, so its
measure is `0`.
-/

namespace RefreshTopProof

theorem certificate : GeneratedClaim.CorrectnessClaim := by
  intro model external
  have hempty : {tape | ∃ execution, GeneratedClaim.Runs model external tape execution ∧
      ¬ execution.«stage_9» = execution.«ideal»} = ∅ := by
    ext tape
    simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
    rintro ⟨execution, hruns, hne⟩
    exact hne (GenCorrect.correct (GeneratedClaim.runs_bridge hruns))
  rw [hempty, MeasureTheory.measure_empty]
  exact zero_le _

/-- info: 'RefreshTopProof.certificate' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms certificate

end RefreshTopProof
