import DiamondProofParameters
import DiamondSelectorWitness
import Stage_decrypt

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

/-- Local error transport through the actual sampled preimage and decryption multiply.
    The input equation is an induction premise; the target equation comes from
    the generated preprocessing scope, not an assumed transition identity. -/
theorem generated_injector_transition
    (backend : BackendContext) (hashModel : HashModel)
    (encryptParams : Stage_encrypt.Params)
    (slot : Nat)
    (secret : ExactMatrix q n 1 1)
    (sourcePublic targetPublic target : ExactMatrix q n 2 inner)
    (trapdoor : TrapdoorValue (ExactMatrix q n 2 inner) Unit)
    (transition : ExactMatrix q n inner inner)
    (stateError current next : ExactMatrix q n 1 inner)
    (rowSecret : ExactMatrix q n 1 2)
    (hcurrent : current = rowSecret * sourcePublic + stateError)
    (htarget : Nonempty (InjectorSelectorWitness backend hashModel encryptParams slot secret targetPublic target))
    (hpreimage : preimageRunsDispatched backend sourcePublic trapdoor target
      encryptParams.diamond_preimage_max_coefficient_bound.toNat transition)
    (hstep : next = current * transition) :
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
  have heq : sourcePublic * transition = target :=
    preimageRunsDispatched_equation (by decide) (by decide) hpreimage
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
    (params : Stage_decrypt.Params) (layer : Nat)
    (index : Int) (states : Fin stateCount → ExactMatrix q n 1 inner)
    (publics : Fin stateCount → ExactMatrix q n 2 inner)
    (secret : ExactPoly q n) (selected : ExactMatrix q n 1 inner)
    (hindexBound : index ≤ Int.ofNat layer * params.diamond_batch_bits)
    (hinvariant : ∀ state : Fin stateCount,
      (state.val : Int) ≤ Int.ofNat layer * params.diamond_batch_bits →
      ∃ (row : ExactMatrix q n 1 2)
      (error : ExactMatrix q n 1 inner), row 0 0 = secret ∧
      states state = row * publics state + error)
    (hrun : familyGetDynamic states index selected) :
    ∃ (state : Fin stateCount) (row : ExactMatrix q n 1 2)
      (error : ExactMatrix q n 1 inner), (state.val : Int) = index ∧
      row 0 0 = secret ∧ selected = row * publics state + error := by
  rcases hrun with ⟨state, hindex, hvalue⟩
  obtain ⟨row, error, hsecret, hequation⟩ := hinvariant state (by omega)
  exact ⟨state, row, error, hindex, hsecret, hvalue.trans hequation⟩

#print axioms generated_source_gather_common_secret

theorem generated_source_index_bound
    (params : Stage_decrypt.Params) (layer lane : Nat)
    (index : Int) (hbatch : 0 ≤ params.diamond_batch_bits)
    (hrun : select (if decide (Int.ofNat layer * params.diamond_batch_bits + 1 ≤ Int.ofNat lane) then 1 else 0)
      [Int.ofNat lane, 0] index) :
    index ≤ Int.ofNat layer * params.diamond_batch_bits := by
  have hselect := hrun
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
    have hi : index = 0 := hvalue
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
    have hi : index = Int.ofNat lane := hvalue
    omega

#print axioms generated_source_index_bound

end DiamondGeneratedProof
