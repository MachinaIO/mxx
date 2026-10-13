import Gen_Correct

/-!
# The worst-case correctness claim of refresh_top

Every worst-case run decodes the ideal output (`RefreshTopProof.GenCorrect.correct`, the part
theorem of the generated instance pipeline, see `generator/run.sh`).
-/

namespace RefreshTopProof

theorem certificate : GeneratedClaim.CorrectnessClaim :=
  fun _ _ _ h ↦ GenCorrect.correct h

/-- info: 'RefreshTopProof.certificate' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms certificate

end RefreshTopProof
