import DiamondClaimInjectorProof
import DiamondProofParameters

open Mxx.Primitives MxxRuntime GeneratedClaim
open DiamondProofParameters

namespace DiamondGeneratedProof

def claimInjectorNoise : Nat :=
  binaryInjectorN n inner stage_0_params.diamond_error_max_coefficient_bound.toNat
    stage_0_params.diamond_preimage_max_coefficient_bound.toNat
    stage_1_params.diamond_input_count.toNat

/-- The actual terminal zero state uses the same base as all final preimages. -/
theorem claim_zero_state_encoding {hashModel external execution}
    (w : ClaimInjectorWitness hashModel external execution) :
    ∃ selector : ExactMatrix q n 1 2,
      selector 0 0 = reducePoly q n w.commonSecret ∧
      selector 0 1 = (if external.input_7 then 1 else 0) ∧
      Approx (execution.stage_1.2.2.2.2.2.2.1 0)
        (selector * w.finalPublic.base) claimInjectorNoise := by
  obtain ⟨position, row, error, hposition, hsecret, hstate, _, herror, hmessage⟩ := w.states 0
  have hpos : position = w.terminal := by
    apply Fin.ext
    have ht := w.terminalAddress
    change (w.terminal.val : Int) = (inputCount * stateCount : Nat) at ht
    change position.val = inputCount * stateCount + 0 at hposition
    omega
  refine ⟨reduceMatrix q n 1 2 row, ?_, ?_, error, ?_, herror⟩
  · exact congrArg (reducePoly q n) hsecret
  · simpa only [Fin.val_zero, ite_true] using hmessage
  · simpa only [hpos, ← w.terminalBase] using hstate

#print axioms claim_zero_state_encoding

end DiamondGeneratedProof
