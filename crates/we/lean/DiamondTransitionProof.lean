import DiamondProofParameters
import DiamondAccumulatedSecretProof
import Stage_decrypt

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

/-- Local error transport through the actual sampled preimage and decryption multiply.
    The input equation is an induction premise; the target equation comes from
    the generated preprocessing scope, not an assumed transition identity. -/
theorem generated_injector_transition
    (backend : BackendContext) (hashModel : HashModel)
    (encryptParams : Stage_encrypt.Params) (decryptParams : Stage_decrypt.Params)
    (slot layer lane : Nat)
    (secret : ExactMatrix q n 1 1)
    (sourcePublic targetPublic target : ExactMatrix q n 2 inner)
    (trapdoor : TrapdoorValue (ExactMatrix q n 2 inner) Unit)
    (transition : ExactMatrix q n inner inner)
    (stateError current next : ExactMatrix q n 1 inner)
    (rowSecret : ExactMatrix q n 1 2)
    (hcurrent : current = rowSecret * sourcePublic + stateError)
    (htarget : Stage_encrypt.parallel_generatedRoot_72 backend hashModel encryptParams slot
      (secret, targetPublic, ()) target)
    (hpreimage : Stage_encrypt.parallel_generatedRoot_73 backend hashModel encryptParams slot
      (sourcePublic, trapdoor, target, ()) transition)
    (hstep : Stage_decrypt.parallel_sequential_generatedRoot_8_13 backend decryptParams
      layer lane (current, transition, ()) next) :
    ∃ (selector : ExactMatrix q n 2 2) (targetError : ExactMatrix q n 2 inner),
      gaussianSample encryptParams.diamond_error_sigma
        encryptParams.diamond_error_max_coefficient_bound targetError ∧
      target = selector * targetPublic + targetError ∧
      (rowSecret * selector) 0 0 = rowSecret 0 0 * secret 0 0 ∧
      next = (rowSecret * selector) * targetPublic +
        (rowSecret * targetError + stateError * transition) := by
  obtain ⟨selector, targetError, h00, h10, herror, htarget⟩ :=
    generated_target_shared_secret backend hashModel encryptParams slot secret
      targetPublic target htarget
  rcases hpreimage with ⟨value, _, hruns, hvalue⟩
  have heq : sourcePublic * transition = target := by
    rw [hvalue]
    exact preimageRunsDispatched_equation (by decide) (by decide) hruns
  have hnext : next = current * transition := hstep
  change target = selector * targetPublic + targetError at htarget
  refine ⟨selector, targetError, herror, htarget, ?_, ?_⟩
  · simp [Matrix.mul_apply, Fin.sum_univ_two, h00, h10]
  · rw [hnext, hcurrent, Matrix.add_mul, Matrix.mul_assoc, heq, htarget,
      Matrix.mul_add, ← Matrix.mul_assoc]
    exact add_assoc _ _ _

#print axioms generated_injector_transition

/-- A source gather retains the exact source index and the common secret premise;
    the public matrix is not replaced by an independently selected witness. -/
theorem generated_source_gather_common_secret
    (backend : BackendContext) (params : Stage_decrypt.Params) (layer lane : Nat)
    (index : Int) (states : Fin stateCount → ExactMatrix q n 1 inner)
    (publics : Fin stateCount → ExactMatrix q n 2 inner)
    (secret : ExactPoly q n) (selected : ExactMatrix q n 1 inner)
    (hindexBound : index ≤ Int.ofNat layer * params.diamond_batch_bits)
    (hinvariant : ∀ state : Fin stateCount,
      (state.val : Int) ≤ Int.ofNat layer * params.diamond_batch_bits →
      ∃ (row : ExactMatrix q n 1 2)
      (error : ExactMatrix q n 1 inner), row 0 0 = secret ∧
      states state = row * publics state + error)
    (hrun : Stage_decrypt.parallel_sequential_generatedRoot_8_7 backend params layer lane
      (index, states, ()) selected) :
    ∃ (state : Fin stateCount) (row : ExactMatrix q n 1 2)
      (error : ExactMatrix q n 1 inner), (state.val : Int) = index ∧
      row 0 0 = secret ∧ selected = row * publics state + error := by
  rcases hrun with ⟨value, _, _, ⟨state, hindex, hvalue⟩, hout⟩
  obtain ⟨row, error, hsecret, hequation⟩ := hinvariant state (by omega)
  exact ⟨state, row, error, hindex, hsecret, hout.trans (hvalue.trans hequation)⟩

#print axioms generated_source_gather_common_secret

theorem generated_source_index_bound
    (backend : BackendContext) (params : Stage_decrypt.Params) (layer lane : Nat)
    (index : Int) (hbatch : 0 ≤ params.diamond_batch_bits)
    (hrun : Stage_decrypt.parallel_sequential_generatedRoot_8_5 backend params layer lane
      (Int.ofNat layer * params.diamond_batch_bits + 1) index) :
    index ≤ Int.ofNat layer * params.diamond_batch_bits := by
  dsimp only [Stage_decrypt.parallel_sequential_generatedRoot_8_5] at hrun
  rcases hrun with ⟨value, _, _, _, hselect, hout⟩
  by_cases hnew : Int.ofNat layer * params.diamond_batch_bits + 1 ≤ Int.ofNat lane
  · have hflag : decide (Int.ofNat layer * params.diamond_batch_bits + 1 ≤
        Int.ofNat lane) = true := decide_eq_true hnew
    rw [hflag, if_pos rfl] at hselect
    rcases hselect with ⟨position, hposition, hvalue⟩
    have hp : position = (⟨1, by decide⟩ : Fin 2) := by
      apply Fin.ext
      dsimp at hposition ⊢
      omega
    subst position
    have hi : index = 0 := hout.trans hvalue
    rw [hi]
    exact mul_nonneg (Int.natCast_nonneg layer) hbatch
  · have hflag : decide (Int.ofNat layer * params.diamond_batch_bits + 1 ≤
        Int.ofNat lane) = false := decide_eq_false hnew
    rw [hflag, if_neg (by decide)] at hselect
    rcases hselect with ⟨position, hposition, hvalue⟩
    have hp : position = (⟨0, by decide⟩ : Fin 2) := by
      apply Fin.ext
      dsimp at hposition ⊢
      omega
    subst position
    have hi : index = Int.ofNat lane + 0 := hout.trans hvalue
    omega

#print axioms generated_source_index_bound

end DiamondGeneratedProof
