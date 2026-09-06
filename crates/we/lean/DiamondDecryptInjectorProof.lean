import DiamondProofParameters
import DiamondBoundedLoopProof
import DiamondPackingProof
import DiamondNumericProof

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

set_option maxRecDepth 8192
set_option maxHeartbeats 1000000

theorem generated_decrypt_injector_context
    (backend : BackendContext) (params : Stage_decrypt.Params) {inputs outputs}
    (hraw : ∀ position : Fin circuitWidth, 0 ≤ inputs.2.2.1 position ∧ inputs.2.2.1 position ≤ 1)
    (rootWitness : Stage_decrypt.generatedRoot.Witness)
    (hbody : Stage_decrypt.generatedRoot.body backend params inputs outputs rootWitness) :
    ∃ (initialStates : Fin stateCount → ExactMatrix q n 1 inner) (packed : Fin inputCount → Int),
      0 ≤ params.diamond_batch_bits ∧
      (∀ state : Fin stateCount, Stage_decrypt.parallel_generatedRoot_2 backend params state.val
        inputs.1 (initialStates state)) ∧
      (∀ i, 0 ≤ packed i ∧ packed i < (2 : Int) ^ params.diamond_batch_bits.toNat) ∧
      (∀ i : Fin inputCount, ∀ bit, bit < params.diamond_batch_bits.toNat →
        rawWitnessBits inputs.2.2.1 (i.val * params.diamond_batch_bits.toNat + bit) =
          decide ((packed i / 2 ^ bit) % 2 = 1)) ∧
      MxxIR.IterRuns
        (fun layer current next ↦ Stage_decrypt.sequential_generatedRoot_6 backend params
          layer (current, inputs.2.1, packed, ()) next)
        params.diamond_input_count.toNat initialStates rootWitness.w_6_0 := by
  dsimp only [Stage_decrypt.generatedRoot.body, Stage_decrypt.generatedRoot.constraints_0] at hbody
  have hinit : ∀ state : Fin stateCount, Stage_decrypt.parallel_generatedRoot_2 backend params state.val
      inputs.1 (rootWitness.w_2_0 state) := by tauto
  have hpacking : ∀ i : Fin inputCount, Stage_decrypt.parallel_generatedRoot_5 backend params i.val
      inputs.2.2.1 (rootWitness.w_5_0 i) := by tauto
  have hbatch : 0 ≤ params.diamond_batch_bits := by
    rcases hpacking 0 with ⟨_, hbatch, _, _⟩
    exact hbatch
  have hpacked := generated_packed_raw_witness backend params inputs.2.2.1 rootWitness.w_5_0
    hraw hpacking
  have hloop : MxxIR.IterRuns
      (fun layer current next ↦ Stage_decrypt.sequential_generatedRoot_6 backend params
        layer (current, inputs.2.1, rootWitness.w_5_0, ()) next)
      params.diamond_input_count.toNat rootWitness.w_2_0 rootWitness.w_6_0 := by tauto
  exact ⟨rootWitness.w_2_0, rootWitness.w_5_0, hbatch, hinit, hpacked.1, hpacked.2, hloop⟩

theorem generated_decrypt_bounded_states
    (backend : BackendContext) (hashModel : HashModel)
    (encryptParams : Stage_encrypt.Params) (params : Stage_decrypt.Params)
    (inputs : _) (outputs : _)
    (rootWitness : Stage_decrypt.generatedRoot.Witness)
    (hbody : Stage_decrypt.generatedRoot.body backend params inputs outputs rootWitness)
    (message : Bool)
    (producer : InjectorRootWitness backend hashModel encryptParams message inputs.1 inputs.2.1)
    (digitBase : Nat) (hdigits : 0 < digitBase)
    (hbase : (digitBase : Int) = encryptParams.diamond_digit_base)
    (hbaseParams : encryptParams.diamond_digit_base = params.diamond_digit_base)
    (hbatch : encryptParams.diamond_batch_bits = params.diamond_batch_bits)
    (hradix : (2 : Int) ^ params.diamond_batch_bits.toNat ≤ params.diamond_digit_base)
    (hraw : ∀ position : Fin circuitWidth, 0 ≤ inputs.2.2.1 position ∧ inputs.2.2.1 position ≤ 1) :
    ∃ commonSecret : ErrorPoly n, ∀ state : Fin stateCount,
      (state.val : Int) ≤ (params.diamond_input_count.toNat : Int) * params.diamond_batch_bits →
      ∃ (position : Fin basePoolCount) (row : ErrorMatrix n 1 2) (error : ErrorMatrix n 1 inner),
        position.val = params.diamond_input_count.toNat * DiamondProofParameters.stateCount + state.val ∧
        row 0 0 = commonSecret ∧
        rootWitness.w_6_0 state = reduceMatrix q n 1 2 row * producer.bases position +
          reduceMatrix q n 1 inner error ∧
        CoeffBound row (binaryInjectorP n inner encryptParams.diamond_error_max_coefficient_bound.toNat
          encryptParams.diamond_preimage_max_coefficient_bound.toNat params.diamond_input_count.toNat) ∧
        CoeffBound error (binaryInjectorN n inner encryptParams.diamond_error_max_coefficient_bound.toNat
          encryptParams.diamond_preimage_max_coefficient_bound.toNat params.diamond_input_count.toNat) ∧
        reducePoly q n (row 0 1) =
          if state.val = 0 then (if message then 1 else 0)
          else reducePoly q n commonSecret *
            (if rawWitnessBits inputs.2.2.1 (state.val - 1) then 1 else 0) := by
  obtain ⟨initialStates, packed, hbatchNonneg, hinitial, hpacked, hbits, hloop⟩ :=
    generated_decrypt_injector_context backend params hraw rootWitness hbody
  obtain ⟨hPzero, hNzero, hPstep, hNstep⟩ := fixture_binaryInjector_loop_premises
    encryptParams.diamond_error_max_coefficient_bound.toNat
    encryptParams.diamond_preimage_max_coefficient_bound.toNat
  have hpackedBase : ∀ index, 0 ≤ packed index ∧ packed index < digitBase := by
    intro index
    refine ⟨(hpacked index).1, ?_⟩
    rw [hbase, hbaseParams]
    exact lt_of_lt_of_le (hpacked index).2 hradix
  have hbatchNat : (params.diamond_batch_bits.toNat : Int) =
      encryptParams.diamond_batch_bits := by
    rw [hbatch, Int.toNat_of_nonneg hbatchNonneg]
  exact generated_bounded_injector_loop backend hashModel encryptParams params message inputs.1
    inputs.2.1 producer digitBase params.diamond_input_count.toNat hdigits hbase hbaseParams
    hbatch hbatchNonneg packed hpackedBase params.diamond_batch_bits.toNat hbatchNat
    (rawWitnessBits inputs.2.2.1) hbits _ _ hPzero hNzero hPstep hNstep initialStates
    rootWitness.w_6_0 hinitial hloop

#print axioms generated_decrypt_bounded_states
#print axioms generated_decrypt_injector_context

end DiamondGeneratedProof
