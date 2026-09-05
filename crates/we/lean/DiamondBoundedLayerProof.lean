import DiamondProofParameters
import DiamondIntegerInvariant
import DiamondLayerProof
import DiamondEncodingRowProof

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

theorem generated_bounded_injector_lane
    (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_encrypt.Params) (decryptParams : Stage_decrypt.Params)
    (message : Bool) (initial : ExactMatrix q n 1 inner)
    (transitions : Fin transitionCount → ExactMatrix q n inner inner)
    (producer : InjectorRootWitness backend hashModel params message initial transitions)
    (samples : Fin sampleCount → ErrorPoly n)
    (hsamples : ∀ position,
      producer.digitSamples position 0 0 = reducePoly q n (samples position) ∧
      polyNorm (samples position) ≤ 1)
    (digitBase layer : Nat) (hdigits : 0 < digitBase)
    (hbase : (digitBase : Int) = params.diamond_digit_base)
    (hbatch : params.diamond_batch_bits = decryptParams.diamond_batch_bits)
    (hbatchNonneg : 0 ≤ decryptParams.diamond_batch_bits)
    (digit : Fin digitBase) (lane : Fin stateCount) (slot : Fin transitionCount)
    (batch : Nat) (hbatchNat : (batch : Int) = params.diamond_batch_bits)
    (bits : Nat → Bool)
    (hbits : ∀ bit, bit < batch →
      bits (layer * batch + bit) = decide (((digit.val : Int) / (2 ^ bit)) % 2 = 1))
    (hslot : slot.val = (layer * digitBase + digit.val) * DiamondProofParameters.stateCount + lane.val)
    (states : Fin stateCount → ExactMatrix q n 1 inner) (commonSecret : ErrorPoly n)
    (rowBound errorBound : Nat)
    (sourceIndex : Int) (current next : ExactMatrix q n 1 inner)
    (hinvariant : ∀ state : Fin stateCount,
      (state.val : Int) ≤ Int.ofNat layer * decryptParams.diamond_batch_bits →
      ∃ (position : Fin basePoolCount) (row : ErrorMatrix n 1 2) (error : ErrorMatrix n 1 inner),
        position.val = layer * DiamondProofParameters.stateCount + state.val ∧ row 0 0 = commonSecret ∧
        states state = reduceMatrix q n 1 2 row * producer.bases position +
          reduceMatrix q n 1 inner error ∧
        CoeffBound row rowBound ∧ CoeffBound error errorBound ∧
        reducePoly q n (row 0 1) =
          if state.val = 0 then (if message then 1 else 0)
          else reducePoly q n commonSecret * (if bits (state.val - 1) then 1 else 0))
    (hsource : Stage_decrypt.parallel_sequential_generatedRoot_8_5 backend decryptParams
      layer lane.val (Int.ofNat layer * decryptParams.diamond_batch_bits + 1) sourceIndex)
    (hgather : Stage_decrypt.parallel_sequential_generatedRoot_8_7 backend decryptParams
      layer lane.val (sourceIndex, states, ()) current)
    (hstep : Stage_decrypt.parallel_sequential_generatedRoot_8_13 backend decryptParams
      layer lane.val (current, transitions slot, ()) next) :
    ∃ (samplePosition : Fin sampleCount) (targetPosition : Fin basePoolCount)
      (row : ErrorMatrix n 1 2) (error : ErrorMatrix n 1 inner),
      samplePosition.val = layer * digitBase + digit.val ∧
      targetPosition.val = (layer + 1) * DiamondProofParameters.stateCount + lane.val ∧
      row 0 0 = commonSecret * samples samplePosition ∧
      next = reduceMatrix q n 1 2 row * producer.bases targetPosition +
        reduceMatrix q n 1 inner error ∧
      CoeffBound row (n * rowBound) ∧
      CoeffBound error (2 * n * rowBound *
        params.diamond_error_max_coefficient_bound.toNat +
        inner * n * errorBound * params.diamond_preimage_max_coefficient_bound.toNat) ∧
      (lane.val ≤ (layer + 1) * batch → reducePoly q n (row 0 1) =
        if lane.val = 0 then (if message then 1 else 0)
        else reducePoly q n (row 0 0) * (if bits (lane.val - 1) then 1 else 0)) := by
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
  obtain ⟨basePosition, row, error, hbasePosition, hrow, hcurrent, hrowBound, herrorBound,
    hcoordinateMeaning⟩ :=
    hinvariant state (by omega)
  have hpositions : sourcePosition = basePosition := by
    apply Fin.ext
    have haddress := hsourcePosition.trans hsourceAddress
    simp only [Nat.cast_mul] at haddress
    omega
  have hcurrent' : current = reduceMatrix q n 1 2 row * producer.sourcePublics slot +
      reduceMatrix q n 1 inner error := by
    rw [hsourcePublic, hpositions]
    exact hout.trans (hvalue.trans hcurrent)
  obtain ⟨samplePosition, hsamplePosition, hsecret, hsecretBound⟩ :=
    producer_integer_selected_secret backend hashModel params message initial transitions
      producer samples hsamples slot
  obtain ⟨selectorWitness, nextRow, nextError, hrowReduce, hfirst, hnext,
    hnextRowBound, hnextErrorBound⟩ :=
    generated_integer_transition backend hashModel params decryptParams message initial
      transitions producer slot layer lane.val row error (samples samplePosition)
      rowBound errorBound current next hsecret hsecretBound hrowBound herrorBound hcurrent' hstep
  have hdigitIndex := (producer.digitIndicesRun slot).2
  change producer.digitIndices slot = Int.ofNat slot.val /
    (1 + params.diamond_batch_bits * params.diamond_input_count) at hdigitIndex
  rw [producer.stateCount, hslot] at hdigitIndex
  obtain ⟨_, _, hcoordinate⟩ := transition_coordinates DiamondProofParameters.stateCount digitBase layer (by decide) hdigits
    digit lane
  have hdigitAddress : producer.digitIndices slot =
      ((layer * digitBase + digit.val : Nat) : Int) := by
    have hc := congrArg (fun value : Nat ↦ (value : Int)) hcoordinate
    dsimp only at hc
    rw [Int.natCast_ediv] at hc
    exact hdigitIndex.trans hc
  refine ⟨samplePosition, targetPosition, nextRow, nextError, ?_, ?_, ?_, ?_,
    hnextRowBound, hnextErrorBound, ?_⟩
  · exact_mod_cast hsamplePosition.trans hdigitAddress
  · exact_mod_cast htargetPosition.trans htargetAddress
  · simpa only [hrow] using hfirst
  · simpa only [htargetPublic] using hnext
  · intro hactive
    obtain ⟨hselectorState, hselectorDigit, hselectorFirst⟩ := selector_witness_coordinates
      backend hashModel params slot.val _ _ _ selectorWitness DiamondProofParameters.stateCount digitBase layer
      (by decide) hdigits hstateGeometry hbase digit lane hslot
    apply selector_integer_row_encoding backend hashModel params slot.val _ _ _ selectorWitness
      row nextRow commonSecret (samples samplePosition) message bits batch layer lane.val
      digit.val hrow (by simpa only [hrow] using hfirst) hsecret hrowReduce
      (by rw [← hbatchNat]; simp) hselectorState
      (by simpa only [← hbatchNat, Nat.cast_add, Nat.cast_mul, Nat.cast_one]
        using hselectorFirst) hselectorDigit hactive _ hbits
    intro hexisting
    have hbefore : (lane.val : Int) < (layer : Int) * decryptParams.diamond_batch_bits + 1 := by
      rw [← hbatch, ← hbatchNat]
      exact_mod_cast Nat.lt_succ_of_le hexisting
    have hsourceEq := generated_existing_source_index backend decryptParams layer lane.val
      sourceIndex hbefore hsource
    have hs : state.val = lane.val := by exact_mod_cast hstate.trans hsourceEq
    simpa only [hs] using hcoordinateMeaning

#print axioms generated_bounded_injector_lane

theorem generated_bounded_injector_layer
    (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_encrypt.Params) (decryptParams : Stage_decrypt.Params)
    (message : Bool) (initial : ExactMatrix q n 1 inner)
    (transitions : Fin transitionCount → ExactMatrix q n inner inner)
    (producer : InjectorRootWitness backend hashModel params message initial transitions)
    (samples : Fin sampleCount → ErrorPoly n)
    (hsamples : ∀ position,
      producer.digitSamples position 0 0 = reducePoly q n (samples position) ∧
      polyNorm (samples position) ≤ 1)
    (digitBase layer : Nat) (hdigits : 0 < digitBase)
    (hbase : (digitBase : Int) = params.diamond_digit_base)
    (hbaseParams : params.diamond_digit_base = decryptParams.diamond_digit_base)
    (hbatch : params.diamond_batch_bits = decryptParams.diamond_batch_bits)
    (hbatchNonneg : 0 ≤ decryptParams.diamond_batch_bits)
    (packed : Fin inputCount → Int)
    (hpacked : ∀ index : Fin inputCount, 0 ≤ packed index ∧ packed index < digitBase)
    (batch : Nat) (hbatchNat : (batch : Int) = params.diamond_batch_bits)
    (bits : Nat → Bool)
    (hpackedBits : ∀ index : Fin inputCount, ∀ bit, bit < batch →
      bits (index.val * batch + bit) = decide ((packed index / (2 ^ bit)) % 2 = 1))
    (states outputs : Fin stateCount → ExactMatrix q n 1 inner)
    (commonSecret : ErrorPoly n) (rowBound errorBound : Nat)
    (hinvariant : ∀ state : Fin stateCount,
      (state.val : Int) ≤ Int.ofNat layer * decryptParams.diamond_batch_bits →
      ∃ (position : Fin basePoolCount) (row : ErrorMatrix n 1 2) (error : ErrorMatrix n 1 inner),
        position.val = layer * DiamondProofParameters.stateCount + state.val ∧ row 0 0 = commonSecret ∧
        states state = reduceMatrix q n 1 2 row * producer.bases position +
          reduceMatrix q n 1 inner error ∧
        CoeffBound row rowBound ∧ CoeffBound error errorBound ∧
        reducePoly q n (row 0 1) =
          if state.val = 0 then (if message then 1 else 0)
          else reducePoly q n commonSecret * (if bits (state.val - 1) then 1 else 0))
    (hrun : Stage_decrypt.sequential_generatedRoot_8 backend decryptParams layer
      (states, packed, transitions, ()) outputs) :
    ∃ (digit : Fin digitBase) (samplePosition : Fin sampleCount),
      familyGetDynamic packed (Int.ofNat layer) (digit.val : Int) ∧
      samplePosition.val = layer * digitBase + digit.val ∧
      ∀ lane : Fin stateCount, ∃ (position : Fin basePoolCount)
        (row : ErrorMatrix n 1 2) (error : ErrorMatrix n 1 inner),
        position.val = (layer + 1) * DiamondProofParameters.stateCount + lane.val ∧
        row 0 0 = commonSecret * samples samplePosition ∧
        outputs lane = reduceMatrix q n 1 2 row * producer.bases position +
          reduceMatrix q n 1 inner error ∧
        CoeffBound row (n * rowBound) ∧
        CoeffBound error (2 * n * rowBound *
          params.diamond_error_max_coefficient_bound.toNat +
          inner * n * errorBound * params.diamond_preimage_max_coefficient_bound.toNat) ∧
        (lane.val ≤ (layer + 1) * batch → reducePoly q n (row 0 1) =
          if lane.val = 0 then (if message then 1 else 0)
          else reducePoly q n (row 0 0) * (if bits (lane.val - 1) then 1 else 0)) := by
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
      (row : ErrorMatrix n 1 2) (error : ErrorMatrix n 1 inner),
      samplePosition.val = layer * digitBase + digit.val ∧
      targetPosition.val = (layer + 1) * DiamondProofParameters.stateCount + lane.val ∧
      row 0 0 = commonSecret * samples samplePosition ∧
      nextStates lane = reduceMatrix q n 1 2 row * producer.bases targetPosition +
        reduceMatrix q n 1 inner error ∧
      CoeffBound row (n * rowBound) ∧
      CoeffBound error (2 * n * rowBound *
        params.diamond_error_max_coefficient_bound.toNat +
        inner * n * errorBound * params.diamond_preimage_max_coefficient_bound.toNat) ∧
      (lane.val ≤ (layer + 1) * batch → reducePoly q n (row 0 1) =
        if lane.val = 0 then (if message then 1 else 0)
        else reducePoly q n (row 0 0) * (if bits (lane.val - 1) then 1 else 0)) := by
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
    exact generated_bounded_injector_lane backend hashModel params decryptParams message initial
      transitions producer samples hsamples digitBase layer hdigits hbase hbatch hbatchNonneg
      digit lane slot batch hbatchNat bits (by
        intro bit hbit
        have hposition : packedPosition.val = layer := by
          change (packedPosition.val : Int) = (layer : Int) at hpackedPosition
          exact_mod_cast hpackedPosition
        have hvalueEq : packed packedPosition = (digit.val : Int) :=
          hpackedValue.symm.trans hdigitValue.symm
        simpa only [hposition, hvalueEq] using hpackedBits packedPosition bit hbit)
      hslot states commonSecret rowBound errorBound (sourceIndices lane)
      (sourceStates lane) (nextStates lane) hinvariant (hsources lane) (hgathers lane) hstep
  obtain ⟨samplePosition, _, _, _, hsamplePosition, _⟩ := hlanes 0
  refine ⟨digit, samplePosition, ?_, hsamplePosition, ?_⟩
  · exact ⟨packedPosition, hpackedPosition, hdigitValue.trans hpackedValue⟩
  · intro lane
    obtain ⟨position, targetPosition, row, error, hposition, htarget, hrow, hequation,
      hrowBound, herrorBound, hmeaning⟩ := hlanes lane
    have hp : position = samplePosition := by
      apply Fin.ext
      omega
    subst position
    exact ⟨targetPosition, row, error, htarget, hrow,
      by simpa only [hout] using hequation, hrowBound, herrorBound, hmeaning⟩

#print axioms generated_bounded_injector_layer

end DiamondGeneratedProof
