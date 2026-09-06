import DiamondProofParameters
import Claim
import DiamondInitialStateProof
import DiamondDecryptInjectorProof

open Mxx.Primitives MxxRuntime
open DiamondProofParameters
open GeneratedClaim

namespace DiamondGeneratedProof

/-- Facts derived jointly from the linked generated executions. No extra noise,
    packing, or common-public-matrix premise is supplied by the caller. -/
structure ClaimInjectorWitness (hashModel : HashModel) (external : ExternalInputs)
    (execution : Execution) where
  producer : InjectorRootWitness DiamondBackend.backend hashModel stage_0_params external.input_7
    execution.stage_0.2.1 execution.stage_0.2.2.2.2.2.2.1
  finalPublic : FinalPublicWitness DiamondBackend.backend stage_0_params execution.stage_0.1
    execution.stage_0.2.2.1 execution.stage_0.2.2.2.1 execution.stage_0.2.2.2.2.2.1
    execution.stage_0.2.2.2.2.2.2.2.2.1 execution.stage_0.2.2.2.2.1
    execution.stage_0.2.2.2.2.2.2.2.2.2.1
  terminal : Fin basePoolCount
  terminalAddress : (terminal.val : Int) = stage_0_params.diamond_input_count *
    (1 + stage_0_params.diamond_batch_bits * stage_0_params.diamond_input_count)
  terminalBase : finalPublic.base = producer.bases terminal
  witnessLinks : ∀ i : Fin witnessSlots,
    ∃ (state : Fin stateCount) (position : Fin basePoolCount) (target : ExactMatrix q n 2 ell),
      state.val = i.val + 1 ∧
      (position.val : Int) = stage_0_params.diamond_input_count *
        (1 + stage_0_params.diamond_batch_bits * stage_0_params.diamond_input_count) + state.val ∧
      concatRows (execution.stage_0.2.2.2.2.1 state) (-finalPublic.gadget) target ∧
      producer.bases position * execution.stage_0.2.2.2.2.2.2.2.1 i = target ∧
      PreimageWithin (execution.stage_0.2.2.2.2.2.2.2.1 i)
        stage_0_params.diamond_preimage_max_coefficient_bound.toNat
  commonSecret : ErrorPoly n
  states : ∀ state : Fin stateCount,
    ∃ (position : Fin basePoolCount) (row : ErrorMatrix n 1 2) (error : ErrorMatrix n 1 inner),
      position.val = stage_1_params.diamond_input_count.toNat * DiamondProofParameters.stateCount + state.val ∧
      row 0 0 = commonSecret ∧
      execution.stage_1.2.2.2.2.2.2.1 state =
        reduceMatrix q n 1 2 row * producer.bases position + reduceMatrix q n 1 inner error ∧
      CoeffBound row (binaryInjectorP n inner stage_0_params.diamond_error_max_coefficient_bound.toNat
        stage_0_params.diamond_preimage_max_coefficient_bound.toNat
        stage_1_params.diamond_input_count.toNat) ∧
      CoeffBound error (binaryInjectorN n inner stage_0_params.diamond_error_max_coefficient_bound.toNat
        stage_0_params.diamond_preimage_max_coefficient_bound.toNat
        stage_1_params.diamond_input_count.toNat) ∧
      reducePoly q n (row 0 1) =
        if state.val = 0 then (if external.input_7 then 1 else 0)
        else reducePoly q n commonSecret *
          (if rawWitnessBits external.input_6 (state.val - 1) then 1 else 0)

theorem generated_claim_injector
    (hashModel : HashModel) (external : ExternalInputs) (execution : Execution)
    (hrun : Runs hashModel external execution) :
    Nonempty (ClaimInjectorWitness hashModel external execution) := by
  obtain ⟨producer, finalPublic, terminal, hterminal, hbase, hwitnesses⟩ :=
    generated_injector_root DiamondBackend.backend hashModel stage_0_params _ _ hrun.2.1
  have hraw : ∀ position : Fin circuitWidth, 0 ≤ external.input_6 position ∧ external.input_6 position ≤ 1 :=
    hrun.1.2.2.2.2.2.2.1
  obtain ⟨secret, hstates⟩ := generated_decrypt_bounded_states DiamondBackend.backend hashModel
    stage_0_params stage_1_params _ _ hrun.2.2.1 external.input_7 producer digitBase (by decide)
    rfl rfl rfl (by decide) hraw
  refine ⟨{
    producer := producer
    finalPublic := finalPublic
    terminal := terminal
    terminalAddress := hterminal
    terminalBase := hbase
    witnessLinks := hwitnesses
    commonSecret := secret
    states := ?_ }⟩
  intro state
  apply hstates state
  change (state.val : Int) ≤ (witnessSlots : Int)
  have h : state.val < 1 + witnessSlots := state.isLt
  omega

#print axioms generated_claim_injector

end DiamondGeneratedProof
