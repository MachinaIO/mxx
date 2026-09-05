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
    (hraw : ∀ position : Fin circuitWidth, 0 ≤ inputs.2.1 position ∧ inputs.2.1 position ≤ 1)
    (hrun : Stage_decrypt.generatedRoot backend params inputs outputs) :
    ∃ (initialStates : Fin stateCount → ExactMatrix q n 1 inner) (packed : Fin inputCount → Int),
      0 ≤ params.diamond_batch_bits ∧
      (∀ state : Fin stateCount, Stage_decrypt.parallel_generatedRoot_2 backend params state.val
        inputs.1 (initialStates state)) ∧
      (∀ i, 0 ≤ packed i ∧ packed i < (2 : Int) ^ params.diamond_batch_bits.toNat) ∧
      (∀ i : Fin inputCount, ∀ bit, bit < params.diamond_batch_bits.toNat →
        rawWitnessBits inputs.2.1 (i.val * params.diamond_batch_bits.toNat + bit) =
          decide ((packed i / 2 ^ bit) % 2 = 1)) ∧
      MxxIR.IterRuns
        (fun layer current next ↦ Stage_decrypt.sequential_generatedRoot_8 backend params
          layer (current, packed, inputs.2.2.1, ()) next)
        params.diamond_input_count.toNat initialStates outputs.2.2.2.2.2.2.1 := by
  dsimp only [Stage_decrypt.generatedRoot] at hrun
  rcases hrun with ⟨w2, w3, w5, w6, w8, state, w19, w24, w26, w27, w28, w29,
    w31, w32, w33, w35, w36, w37, w38, w40, w42, w43, w44, w45, w46, w47,
    w48, w49, w50, w53, w54, w55, w56, w57, w58, w59, w60, w61, w62,
    w67a, w67b, w67c, w70, circuit, coefficient, h⟩
  have hinit : ∀ state : Fin stateCount, Stage_decrypt.parallel_generatedRoot_2 backend params state.val
      inputs.1 (w2 state) := by tauto
  have hindices : ∀ i : Fin witnessSlots, Stage_decrypt.parallel_generatedRoot_3 backend params i.val ()
      (w3 i) := by tauto
  have hprefix : ∀ i : Fin witnessSlots, Stage_decrypt.parallel_generatedRoot_5 backend params i.val
      (w3 i, inputs.2.1, ()) (w5 i) := by tauto
  have hpacking : ∀ i : Fin inputCount, Stage_decrypt.parallel_generatedRoot_6 backend params i.val
      w5 (w6 i) := by tauto
  have hbatch : 0 ≤ params.diamond_batch_bits := by
    rcases hpacking 0 with ⟨_, _, hbatch, _, _⟩
    exact hbatch
  have hpacked := generated_packed_raw_witness backend params inputs.2.1 w3 w5 w6
    hraw hindices hprefix hpacking
  have hloop : MxxIR.IterRuns
      (fun layer current next ↦ Stage_decrypt.sequential_generatedRoot_8 backend params
        layer (current, w6, inputs.2.2.1, ()) next)
      params.diamond_input_count.toNat w2 w8 := by tauto
  repeat' obtain ⟨_, h⟩ := h
  exact ⟨w2, w6, hbatch, hinit, hpacked.1, hpacked.2, hloop⟩

theorem generated_decrypt_bounded_states
    (backend : BackendContext) (hashModel : HashModel)
    (encryptParams : Stage_encrypt.Params) (params : Stage_decrypt.Params)
    (inputs : _) (outputs : _)
    (hrun : Stage_decrypt.generatedRoot backend params inputs outputs)
    (message : Bool)
    (producer : InjectorRootWitness backend hashModel encryptParams message inputs.1 inputs.2.2.1)
    (digitBase : Nat) (hdigits : 0 < digitBase)
    (hbase : (digitBase : Int) = encryptParams.diamond_digit_base)
    (hbaseParams : encryptParams.diamond_digit_base = params.diamond_digit_base)
    (hbatch : encryptParams.diamond_batch_bits = params.diamond_batch_bits)
    (hradix : (2 : Int) ^ params.diamond_batch_bits.toNat ≤ params.diamond_digit_base)
    (hraw : ∀ position : Fin circuitWidth, 0 ≤ inputs.2.1 position ∧ inputs.2.1 position ≤ 1) :
    ∃ commonSecret : ErrorPoly n, ∀ state : Fin stateCount,
      (state.val : Int) ≤ (params.diamond_input_count.toNat : Int) * params.diamond_batch_bits →
      ∃ (position : Fin basePoolCount) (row : ErrorMatrix n 1 2) (error : ErrorMatrix n 1 inner),
        position.val = params.diamond_input_count.toNat * DiamondProofParameters.stateCount + state.val ∧
        row 0 0 = commonSecret ∧
        outputs.2.2.2.2.2.2.1 state = reduceMatrix q n 1 2 row * producer.bases position +
          reduceMatrix q n 1 inner error ∧
        CoeffBound row (binaryInjectorP n inner encryptParams.diamond_error_max_coefficient_bound.toNat
          encryptParams.diamond_preimage_max_coefficient_bound.toNat params.diamond_input_count.toNat) ∧
        CoeffBound error (binaryInjectorN n inner encryptParams.diamond_error_max_coefficient_bound.toNat
          encryptParams.diamond_preimage_max_coefficient_bound.toNat params.diamond_input_count.toNat) ∧
        reducePoly q n (row 0 1) =
          if state.val = 0 then (if message then 1 else 0)
          else reducePoly q n commonSecret *
            (if rawWitnessBits inputs.2.1 (state.val - 1) then 1 else 0) := by
  obtain ⟨initialStates, packed, hbatchNonneg, hinitial, hpacked, hbits, hloop⟩ :=
    generated_decrypt_injector_context backend params hraw hrun
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
    inputs.2.2.1 producer digitBase params.diamond_input_count.toNat hdigits hbase hbaseParams
    hbatch hbatchNonneg packed hpackedBase params.diamond_batch_bits.toNat hbatchNat
    (rawWitnessBits inputs.2.1) hbits _ _ hPzero hNzero hPstep hNstep initialStates
    outputs.2.2.2.2.2.2.1 hinitial hloop

#print axioms generated_decrypt_bounded_states
#print axioms generated_decrypt_injector_context

end DiamondGeneratedProof
