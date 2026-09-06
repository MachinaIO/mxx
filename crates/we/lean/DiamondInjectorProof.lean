import DiamondProofParameters
import Stage_encrypt

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

/-- Distinct state lanes select the same sampled secret whenever their actual
    generated digit-secret indices coincide. No family is enumerated. -/
theorem generated_shared_digit_secret
    (params : Stage_encrypt.Params)
    (firstSlot secondSlot : Nat) (firstIndex secondIndex : Int)
    (samples : Fin sampleCount → ExactMatrix q n 1 1)
    (firstSecret secondSecret : ExactMatrix q n 1 1)
    (hfirstIndex : firstIndex = Int.ofNat firstSlot /
      (1 + params.diamond_batch_bits * params.diamond_input_count))
    (hsecondIndex : secondIndex = Int.ofNat secondSlot /
      (1 + params.diamond_batch_bits * params.diamond_input_count))
    (hsameDigit : Int.ofNat firstSlot /
      (1 + params.diamond_batch_bits * params.diamond_input_count) =
      Int.ofNat secondSlot / (1 + params.diamond_batch_bits * params.diamond_input_count))
    (hfirst : familyGetDynamic samples firstIndex firstSecret)
    (hsecond : familyGetDynamic samples secondIndex secondSecret) : firstSecret = secondSecret := by
  have hindices : firstIndex = secondIndex :=
    hfirstIndex.trans (hsameDigit.trans hsecondIndex.symm)
  rcases hfirst with ⟨firstPosition, hfirstPosition, hfirstValue⟩
  rcases hsecond with ⟨secondPosition, hsecondPosition, hsecondValue⟩
  have hposition : firstPosition = secondPosition := by
    apply Fin.ext
    omega
  exact hfirstValue.trans ((congrArg samples hposition).trans hsecondValue.symm)

theorem generated_state_lanes_share_secret
    (params : Stage_encrypt.Params)
    (stateCount digit : Nat) (hcount : 0 < stateCount)
    (hgeometry : (stateCount : Int) =
      1 + params.diamond_batch_bits * params.diamond_input_count)
    (firstState secondState : Fin stateCount) (firstIndex secondIndex : Int)
    (samples : Fin sampleCount → ExactMatrix q n 1 1)
    (firstSecret secondSecret : ExactMatrix q n 1 1)
    (hfirstIndex : firstIndex = Int.ofNat (digit * stateCount + firstState.val) /
      (1 + params.diamond_batch_bits * params.diamond_input_count))
    (hsecondIndex : secondIndex = Int.ofNat (digit * stateCount + secondState.val) /
      (1 + params.diamond_batch_bits * params.diamond_input_count))
    (hfirst : familyGetDynamic samples firstIndex firstSecret)
    (hsecond : familyGetDynamic samples secondIndex secondSecret) :
    firstSecret = secondSecret := by
  apply generated_shared_digit_secret params _ _ _ _ samples _ _
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
