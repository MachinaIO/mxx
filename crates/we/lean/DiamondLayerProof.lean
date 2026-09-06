import DiamondProofParameters
import DiamondInjectorWitness
import DiamondIndexProof
import DiamondTransitionProof

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

/-- One actual lane, using the producer's exact pool, sample family and preimage.
    Only the previously introduced states carry the induction premise. -/
theorem generated_injector_lane
    (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_encrypt.Params) (decryptParams : Stage_decrypt.Params)
    (message : Bool) (initial : ExactMatrix q n 1 inner)
    (transitions : Fin transitionCount → ExactMatrix q n inner inner)
    (producer : InjectorRootWitness backend hashModel params message initial transitions)
    (digitBase layer : Nat) (hdigits : 0 < digitBase)
    (hbase : (digitBase : Int) = params.diamond_digit_base)
    (hbatch : params.diamond_batch_bits = decryptParams.diamond_batch_bits)
    (hbatchNonneg : 0 ≤ decryptParams.diamond_batch_bits)
    (digit : Fin digitBase) (lane : Fin stateCount) (slot : Fin transitionCount)
    (hslot : slot.val = (layer * digitBase + digit.val) * DiamondProofParameters.stateCount + lane.val)
    (states : Fin stateCount → ExactMatrix q n 1 inner) (commonSecret : ExactPoly q n)
    (sourceIndex : Int) (current next : ExactMatrix q n 1 inner)
    (hinvariant : ∀ state : Fin stateCount,
      (state.val : Int) ≤ Int.ofNat layer * decryptParams.diamond_batch_bits →
      ∃ (position : Fin basePoolCount) (row : ExactMatrix q n 1 2)
        (error : ExactMatrix q n 1 inner), position.val = layer * DiamondProofParameters.stateCount + state.val ∧
        row 0 0 = commonSecret ∧ states state = row * producer.bases position + error)
    (hsource : Stage_decrypt.parallel_sequential_generatedRoot_8_5 backend decryptParams
      layer lane.val (Int.ofNat layer * decryptParams.diamond_batch_bits + 1) sourceIndex)
    (hgather : Stage_decrypt.parallel_sequential_generatedRoot_8_7 backend decryptParams
      layer lane.val (sourceIndex, states, ()) current)
    (hstep : Stage_decrypt.parallel_sequential_generatedRoot_8_13 backend decryptParams
      layer lane.val (current, transitions slot, ()) next) :
    ∃ (samplePosition : Fin sampleCount) (targetPosition : Fin basePoolCount)
      (row : ExactMatrix q n 1 2) (error : ExactMatrix q n 1 inner),
      samplePosition.val = layer * digitBase + digit.val ∧
      targetPosition.val = (layer + 1) * DiamondProofParameters.stateCount + lane.val ∧
      row 0 0 = commonSecret * producer.digitSamples samplePosition 0 0 ∧
      next = row * producer.bases targetPosition + error := by
  have hstateGeometry : (DiamondProofParameters.stateCount : Int) =
      1 + params.diamond_batch_bits * params.diamond_input_count := producer.stateCount.symm
  have hsetup := producer.sourceIndicesRun slot
  have htargetIndex := producer.targetIndicesRun slot
  change Stage_encrypt.parallel_generatedRoot_65 backend hashModel params slot.val ()
    (producer.sourceIndices slot) at hsetup
  change Stage_encrypt.parallel_generatedRoot_70 backend hashModel params slot.val ()
    (producer.targetIndices slot) at htargetIndex
  rw [hslot] at hsetup htargetIndex
  have hsource' : Stage_decrypt.parallel_sequential_generatedRoot_8_5 backend decryptParams
      layer lane.val (Int.ofNat layer * params.diamond_batch_bits + 1) sourceIndex := by
    simpa only [hbatch] using hsource
  have hsourceAddress := generated_source_index_agrees backend hashModel params decryptParams
    DiamondProofParameters.stateCount digitBase layer (by decide) hdigits hstateGeometry hbase digit lane _ _ hsetup hsource'
  have htargetAddress := generated_target_index backend hashModel params DiamondProofParameters.stateCount digitBase layer
    (by decide) hdigits hstateGeometry hbase digit lane _ htargetIndex
  obtain ⟨sourcePosition, hsourcePosition, hsourcePublic, _⟩ :=
    generated_source_pool_lookup backend hashModel params slot.val _ _ _ _ _
      (producer.sourcesRun slot)
  obtain ⟨targetPosition, htargetPosition, htargetPublic⟩ :=
    generated_target_pool_lookup backend hashModel params slot.val _ _ _
      (producer.targetPublicsRun slot)
  have hbound := generated_source_index_bound backend decryptParams layer lane.val sourceIndex
    hbatchNonneg hsource
  rcases hgather with ⟨value, _, _, ⟨state, hstate, hvalue⟩, hout⟩
  obtain ⟨basePosition, row, error, hbasePosition, hrow, hcurrent⟩ :=
    hinvariant state (by omega)
  have hpositions : sourcePosition = basePosition := by
    apply Fin.ext
    have haddress := hsourcePosition.trans hsourceAddress
    simp only [Nat.cast_mul] at haddress
    omega
  have hcurrent' : current = row * producer.sourcePublics slot + error := by
    rw [hsourcePublic, hpositions]
    exact hout.trans (hvalue.trans hcurrent)
  obtain ⟨selector, targetError, _, _, hfirst, hnext⟩ :=
    generated_injector_transition backend hashModel params decryptParams slot.val layer lane.val
      (producer.digitSecrets slot) (producer.sourcePublics slot) (producer.targetPublics slot)
      (producer.targets slot) (producer.sourceTrapdoors slot) (transitions slot) error current next
      row hcurrent' (producer.targetsRun slot) (producer.preimagesRun slot) hstep
  have hdigitIndex := (producer.digitIndicesRun slot).2
  change producer.digitIndices slot = Int.ofNat slot.val /
    (1 + params.diamond_batch_bits * params.diamond_input_count) at hdigitIndex
  rw [producer.stateCount, hslot] at hdigitIndex
  obtain ⟨_, _, hcoordinate⟩ := transition_coordinates DiamondProofParameters.stateCount digitBase layer (by decide) hdigits
    digit lane
  have hdigitAddress : producer.digitIndices slot = ((layer * digitBase + digit.val : Nat) : Int) := by
    have hc := congrArg (fun value : Nat ↦ (value : Int)) hcoordinate
    dsimp only at hc
    rw [Int.natCast_ediv] at hc
    exact hdigitIndex.trans hc
  rcases producer.secretsRun slot with ⟨secretValue, _, _,
    ⟨samplePosition, hsamplePosition, hsampleValue⟩, hsecretOut⟩
  have hsecret : producer.digitSecrets slot = producer.digitSamples samplePosition :=
    hsecretOut.trans hsampleValue
  refine ⟨samplePosition, targetPosition, row * selector,
    row * targetError + error * transitions slot, ?_, ?_, ?_, ?_⟩
  · exact_mod_cast hsamplePosition.trans hdigitAddress
  · exact_mod_cast htargetPosition.trans htargetAddress
  · simpa only [hrow, hsecret] using hfirst
  · simpa only [htargetPublic] using hnext

#print axioms generated_injector_lane

theorem generated_injector_layer
    (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_encrypt.Params) (decryptParams : Stage_decrypt.Params)
    (message : Bool) (initial : ExactMatrix q n 1 inner)
    (transitions : Fin transitionCount → ExactMatrix q n inner inner)
    (producer : InjectorRootWitness backend hashModel params message initial transitions)
    (digitBase layer : Nat) (hdigits : 0 < digitBase)
    (hbase : (digitBase : Int) = params.diamond_digit_base)
    (hbaseParams : params.diamond_digit_base = decryptParams.diamond_digit_base)
    (hbatch : params.diamond_batch_bits = decryptParams.diamond_batch_bits)
    (hbatchNonneg : 0 ≤ decryptParams.diamond_batch_bits)
    (packed : Fin inputCount → Int)
    (hpacked : ∀ index : Fin inputCount, 0 ≤ packed index ∧ packed index < digitBase)
    (states outputs : Fin stateCount → ExactMatrix q n 1 inner)
    (commonSecret : ExactPoly q n)
    (hinvariant : ∀ state : Fin stateCount,
      (state.val : Int) ≤ Int.ofNat layer * decryptParams.diamond_batch_bits →
      ∃ (position : Fin basePoolCount) (row : ExactMatrix q n 1 2)
        (error : ExactMatrix q n 1 inner), position.val = layer * DiamondProofParameters.stateCount + state.val ∧
        row 0 0 = commonSecret ∧ states state = row * producer.bases position + error)
    (hrun : Stage_decrypt.sequential_generatedRoot_8 backend decryptParams layer
      (states, packed, transitions, ()) outputs) :
    ∃ (digit : Fin digitBase) (samplePosition : Fin sampleCount),
      familyGetDynamic packed (Int.ofNat layer) (digit.val : Int) ∧
      samplePosition.val = layer * digitBase + digit.val ∧
      ∀ lane : Fin stateCount, ∃ (position : Fin basePoolCount) (row : ExactMatrix q n 1 2)
        (error : ExactMatrix q n 1 inner), position.val = (layer + 1) * DiamondProofParameters.stateCount + lane.val ∧
        row 0 0 = commonSecret * producer.digitSamples samplePosition 0 0 ∧
        outputs lane = row * producer.bases position + error := by
  dsimp only [Stage_decrypt.sequential_generatedRoot_8] at hrun
  rcases hrun with ⟨sourceIndices, sourceStates, digitValue, transitionIndices,
    selectedTransitions, nextStates, hstateGeometry, hsources, _, hgathers, _, _, hdigitGet,
    _, hindices, _, htransitions, _, hsteps, hout⟩
  obtain ⟨packedPosition, hpackedPosition, hpackedValue⟩ := hdigitGet
  have hdigitBounds : 0 ≤ digitValue ∧ digitValue < digitBase := by
    rw [hpackedValue]
    exact hpacked packedPosition
  let digit : Fin digitBase := ⟨digitValue.toNat, by omega⟩
  have hdigitValue : (digit.val : Int) = digitValue := by
    dsimp [digit]
    omega
  have hlanes : ∀ lane : Fin stateCount, ∃ (samplePosition : Fin sampleCount) (targetPosition : Fin basePoolCount)
      (row : ExactMatrix q n 1 2) (error : ExactMatrix q n 1 inner),
      samplePosition.val = layer * digitBase + digit.val ∧
      targetPosition.val = (layer + 1) * DiamondProofParameters.stateCount + lane.val ∧
      row 0 0 = commonSecret * producer.digitSamples samplePosition 0 0 ∧
      nextStates lane = row * producer.bases targetPosition + error := by
    intro lane
    have hindexRun := hindices lane
    change Stage_decrypt.parallel_sequential_generatedRoot_8_10 backend decryptParams layer
      lane.val (Int.ofNat layer, digitValue, ()) (transitionIndices lane) at hindexRun
    rw [← hdigitValue] at hindexRun
    have hindex := generated_runtime_transition_index backend decryptParams DiamondProofParameters.stateCount digitBase layer
      digit.val lane.val hstateGeometry.symm (hbase.trans hbaseParams) _ hindexRun
    obtain ⟨slot, hslotIndex, hslotValue⟩ := generated_selected_transition_lookup backend
      decryptParams layer lane.val _ _ _ (htransitions lane)
    have hslot : slot.val = (layer * digitBase + digit.val) * DiamondProofParameters.stateCount + lane.val := by
      exact_mod_cast hslotIndex.trans hindex
    have hstep : Stage_decrypt.parallel_sequential_generatedRoot_8_13 backend decryptParams
        layer lane.val (sourceStates lane, transitions slot, ()) (nextStates lane) := by
      simpa only [hslotValue] using hsteps lane
    exact generated_injector_lane backend hashModel params decryptParams message initial
      transitions producer digitBase layer hdigits hbase hbatch hbatchNonneg digit lane slot
      hslot states commonSecret (sourceIndices lane) (sourceStates lane) (nextStates lane)
      hinvariant (hsources lane) (hgathers lane) hstep
  obtain ⟨samplePosition, _, _, _, hsamplePosition, _⟩ := hlanes 0
  refine ⟨digit, samplePosition, ?_, hsamplePosition, ?_⟩
  · exact ⟨packedPosition, hpackedPosition, hdigitValue.trans hpackedValue⟩
  · intro lane
    obtain ⟨position, targetPosition, row, error, hposition, htarget, hrow, hequation⟩ :=
      hlanes lane
    have hp : position = samplePosition := by
      apply Fin.ext
      omega
    subst position
    exact ⟨targetPosition, row, error, htarget, hrow, by simpa only [hout] using hequation⟩

#print axioms generated_injector_layer

end DiamondGeneratedProof
