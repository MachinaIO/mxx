import DiamondProofParameters
import Stage_encrypt

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

/-- Distinct state lanes select the same sampled secret whenever their actual
    generated digit-secret indices coincide. No family is enumerated. -/
theorem generated_shared_digit_secret
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (firstSlot secondSlot : Nat) (firstIndex secondIndex : Int)
    (samples : Fin sampleCount → ExactMatrix q n 1 1)
    (firstSecret secondSecret : ExactMatrix q n 1 1)
    (hfirstIndex : Stage_encrypt.parallel_generatedRoot_67 backend hashModel params
      firstSlot () firstIndex)
    (hsecondIndex : Stage_encrypt.parallel_generatedRoot_67 backend hashModel params
      secondSlot () secondIndex)
    (hsameDigit : Int.ofNat firstSlot /
      (1 + params.diamond_batch_bits * params.diamond_input_count) =
      Int.ofNat secondSlot / (1 + params.diamond_batch_bits * params.diamond_input_count))
    (hfirst : Stage_encrypt.parallel_generatedRoot_69 backend hashModel params firstSlot
      (firstIndex, samples, ()) firstSecret)
    (hsecond : Stage_encrypt.parallel_generatedRoot_69 backend hashModel params secondSlot
      (secondIndex, samples, ()) secondSecret) : firstSecret = secondSecret := by
  have hindices : firstIndex = secondIndex :=
    hfirstIndex.2.trans (hsameDigit.trans hsecondIndex.2.symm)
  rcases hfirst with ⟨first, _, _, ⟨firstPosition, hfirstPosition, hfirstValue⟩, hfirstOut⟩
  rcases hsecond with ⟨second, _, _, ⟨secondPosition, hsecondPosition, hsecondValue⟩,
    hsecondOut⟩
  have hposition : firstPosition = secondPosition := by
    apply Fin.ext
    omega
  exact hfirstOut.trans (hfirstValue.trans
    ((congrArg samples hposition).trans (hsecondValue.symm.trans hsecondOut.symm)))

theorem generated_state_lanes_share_secret
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (stateCount digit : Nat) (hcount : 0 < stateCount)
    (hgeometry : (stateCount : Int) =
      1 + params.diamond_batch_bits * params.diamond_input_count)
    (firstState secondState : Fin stateCount) (firstIndex secondIndex : Int)
    (samples : Fin sampleCount → ExactMatrix q n 1 1)
    (firstSecret secondSecret : ExactMatrix q n 1 1)
    (hfirstIndex : Stage_encrypt.parallel_generatedRoot_67 backend hashModel params
      (digit * stateCount + firstState.val) () firstIndex)
    (hsecondIndex : Stage_encrypt.parallel_generatedRoot_67 backend hashModel params
      (digit * stateCount + secondState.val) () secondIndex)
    (hfirst : Stage_encrypt.parallel_generatedRoot_69 backend hashModel params
      (digit * stateCount + firstState.val) (firstIndex, samples, ()) firstSecret)
    (hsecond : Stage_encrypt.parallel_generatedRoot_69 backend hashModel params
      (digit * stateCount + secondState.val) (secondIndex, samples, ()) secondSecret) :
    firstSecret = secondSecret := by
  apply generated_shared_digit_secret backend hashModel params _ _ _ _ samples _ _
    hfirstIndex hsecondIndex _ hfirst hsecond
  rw [← hgeometry]
  change ((digit * stateCount + firstState.val : Nat) : Int) / (stateCount : Int) =
    ((digit * stateCount + secondState.val : Nat) : Int) / (stateCount : Int)
  rw [← Int.natCast_ediv, ← Int.natCast_ediv]
  congr 1
  simp [Nat.mul_comm, Nat.add_comm, Nat.add_mul_div_left, hcount, Nat.div_eq_of_lt firstState.isLt,
    Nat.div_eq_of_lt secondState.isLt]

#print axioms generated_state_lanes_share_secret

end DiamondGeneratedProof
