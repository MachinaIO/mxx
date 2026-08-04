import MxxWe.Certificate.Verifier

namespace MxxWe.Certificate

/-- Kernel-checked structural consequences of an accepted Diamond certificate.  Each field is
the result of checking exact references against the emitted executable workflow. -/
structure VerifiedDiamondLayout (workflow : Mxx.Ir.Workflow)
    (certificate : DiamondCertificate) : Prop where
  workflowMatches : verifyWorkflow workflow certificate.workflow = true
  messageMatches :
    verifyMessageConstruction workflow certificate.workflow certificate.message = true
  inputPreprocessingMatches :
    verifyInputPreprocessing workflow certificate.workflow certificate.inputPreprocessing = true
  publicKeySamplingMatches :
    verifyPublicKeySampling workflow certificate.workflow certificate.publicKeySampling = true
  encryptionInitialPublicKeysMatch :
    verifyEncryptionInitialPublicKeys workflow certificate.workflow
      certificate.publicKeySampling certificate.encryptionInitialPublicKeys = true
  artifactPreprocessingMatches :
    verifyArtifactPreprocessing workflow certificate.workflow certificate.inputPreprocessing
      certificate.publicKeySampling certificate.booleanLayers certificate.artifactPreprocessing = true
  inputInjectionMatches : verifyInputInjection workflow certificate.inputInjection = true
  decryptionInitialEncodingsMatch :
    verifyDecryptionInitialEncodings workflow certificate.workflow certificate.inputInjection
      certificate.booleanLayers certificate.decryptionInitialEncodings = true
  booleanLayersMatch :
    verifyBooleanLayers workflow certificate.workflow certificate.booleanLayers = true
  decoderMatches :
    verifyDecoder workflow certificate.workflow certificate.booleanLayers certificate.decoder = true
  closureMatches : verifyOutputRootedClosure workflow certificate = true
  definitionNamesUnique : definitionsUnique workflow = true
  ssaOrderValid : verifyWorkflowSsaOrder workflow = true

theorem verifyDiamondCertificate_sound {workflow : Mxx.Ir.Workflow}
    {certificate : DiamondCertificate}
    (verified : verifyDiamondCertificate workflow certificate = true) :
    VerifiedDiamondLayout workflow certificate := by
  unfold verifyDiamondCertificate at verified
  simp only [Bool.and_eq_true] at verified
  refine {
    workflowMatches := ?_
    messageMatches := ?_
    inputPreprocessingMatches := ?_
    publicKeySamplingMatches := ?_
    encryptionInitialPublicKeysMatch := ?_
    artifactPreprocessingMatches := ?_
    inputInjectionMatches := ?_
    decryptionInitialEncodingsMatch := ?_
    booleanLayersMatch := ?_
    decoderMatches := ?_
    closureMatches := ?_
    definitionNamesUnique := ?_
    ssaOrderValid := ?_
  } <;> aesop

theorem VerifiedDiamondLayout.preprocessingSourceIndexFormula
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyPreprocessingSourceIndexFormula workflow
      certificate.inputPreprocessing.transitionSourceIndices = true := by
  have matched := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at matched
  simp only [Bool.and_eq_true] at matched
  aesop

theorem VerifiedDiamondLayout.preprocessingTargetIndexFormula
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyPreprocessingTargetIndexFormula workflow
      certificate.inputPreprocessing.transitionTargetIndices = true := by
  have matched := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at matched
  simp only [Bool.and_eq_true] at matched
  aesop

theorem VerifiedDiamondLayout.preprocessingDigitSecretIndexFormula
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyPreprocessingDigitSecretIndexFormula workflow
      certificate.inputPreprocessing.digitSecretIndices = true := by
  have matched := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at matched
  simp only [Bool.and_eq_true] at matched
  aesop

/-- A checked parallel index formula includes a checked enclosing parallel loop. -/
theorem verifyParallelIndexFormulaRef_loop
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyParallelIndexFormulaRef workflow reference = true) :
    verifyParallelLoop workflow reference.parallelLoop = true := by
  unfold verifyParallelIndexFormulaRef at verified
  simp only [Bool.and_eq_true] at verified
  aesop

/-- The accepted source-index formula includes its checked enclosing parallel loop. -/
theorem VerifiedDiamondLayout.preprocessingSourceIndexLoop
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyParallelLoop workflow
      certificate.inputPreprocessing.transitionSourceIndices.parallelLoop = true := by
  have formula := VerifiedDiamondLayout.preprocessingSourceIndexFormula
    (workflow := workflow) (certificate := certificate) verified
  unfold verifyPreprocessingSourceIndexFormula at formula
  simp only [Bool.and_eq_true] at formula
  have referenceChecked : verifyParallelIndexFormulaRef workflow
      certificate.inputPreprocessing.transitionSourceIndices = true := by
    aesop
  exact verifyParallelIndexFormulaRef_loop
    (workflow := workflow)
    (reference := certificate.inputPreprocessing.transitionSourceIndices)
    referenceChecked

/-- The accepted target-index formula includes its checked enclosing parallel loop. -/
theorem VerifiedDiamondLayout.preprocessingTargetIndexLoop
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyParallelLoop workflow
      certificate.inputPreprocessing.transitionTargetIndices.parallelLoop = true := by
  have formula := VerifiedDiamondLayout.preprocessingTargetIndexFormula
    (workflow := workflow) (certificate := certificate) verified
  unfold verifyPreprocessingTargetIndexFormula at formula
  simp only [Bool.and_eq_true] at formula
  have referenceChecked : verifyParallelIndexFormulaRef workflow
      certificate.inputPreprocessing.transitionTargetIndices = true := by
    aesop
  exact verifyParallelIndexFormulaRef_loop
    (workflow := workflow)
    (reference := certificate.inputPreprocessing.transitionTargetIndices)
    referenceChecked

/-- The accepted digit-secret-index formula includes its checked enclosing parallel loop. -/
theorem VerifiedDiamondLayout.preprocessingDigitSecretIndexLoop
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyParallelLoop workflow
      certificate.inputPreprocessing.digitSecretIndices.parallelLoop = true := by
  have formula := VerifiedDiamondLayout.preprocessingDigitSecretIndexFormula
    (workflow := workflow) (certificate := certificate) verified
  unfold verifyPreprocessingDigitSecretIndexFormula at formula
  simp only [Bool.and_eq_true] at formula
  have referenceChecked : verifyParallelIndexFormulaRef workflow
      certificate.inputPreprocessing.digitSecretIndices = true := by
    aesop
  exact verifyParallelIndexFormulaRef_loop
    (workflow := workflow)
    (reference := certificate.inputPreprocessing.digitSecretIndices)
    referenceChecked

/-- A checked parallel gather includes a checked enclosing parallel loop. -/
theorem verifyParallelGather_loop
    {workflow : Mxx.Ir.Workflow} {reference : ParallelGatherRef}
    (verified : verifyParallelGather workflow reference = true) :
    verifyParallelLoop workflow reference.parallelLoop = true := by
  unfold verifyParallelGather at verified
  simp only [Bool.and_eq_true] at verified
  aesop

/-- A checked transition-target builder includes a checked enclosing parallel loop. -/
theorem verifyParallelTransitionTarget_loop
    {workflow : Mxx.Ir.Workflow} {reference : ParallelTransitionTargetRef}
    (verified : verifyParallelTransitionTarget workflow reference = true) :
    verifyParallelLoop workflow reference.parallelLoop = true := by
  unfold verifyParallelTransitionTarget at verified
  simp only [Bool.and_eq_true] at verified
  aesop

/-- A checked parallel preimage sampler includes a checked enclosing parallel loop. -/
theorem verifyParallelPreimage_loop
    {workflow : Mxx.Ir.Workflow} {reference : ParallelPreimageRef}
    {columns : Mxx.Ir.IntExpr}
    (verified : verifyParallelPreimage workflow reference columns = true) :
    verifyParallelLoop workflow reference.parallelLoop = true := by
  unfold verifyParallelPreimage at verified
  simp only [Bool.and_eq_true] at verified
  aesop

/-- The accepted preprocessing layout checks the digit-secret gather. -/
theorem VerifiedDiamondLayout.preprocessingDigitSecretsGather
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyParallelGather workflow certificate.inputPreprocessing.digitSecrets = true := by
  have matched := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at matched
  simp only [Bool.and_eq_true] at matched
  aesop

/-- The accepted preprocessing layout checks the transition-source gather. -/
theorem VerifiedDiamondLayout.preprocessingTransitionSourcesGather
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyParallelGather workflow certificate.inputPreprocessing.transitionSources = true := by
  have matched := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at matched
  simp only [Bool.and_eq_true] at matched
  aesop

/-- The accepted preprocessing layout checks the target-public-matrix gather. -/
theorem VerifiedDiamondLayout.preprocessingTargetPublicMatricesGather
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyParallelGather workflow certificate.inputPreprocessing.targetPublicMatrices = true := by
  have matched := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at matched
  simp only [Bool.and_eq_true] at matched
  aesop

/-- The accepted preprocessing layout checks the transition-target builder. -/
theorem VerifiedDiamondLayout.preprocessingTransitionTargets
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyParallelTransitionTarget workflow certificate.inputPreprocessing.transitionTargets =
      true := by
  have matched := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at matched
  simp only [Bool.and_eq_true] at matched
  aesop

/-- The accepted preprocessing layout checks the transition-preimage sampler. -/
theorem VerifiedDiamondLayout.preprocessingTransitionPreimages
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyParallelPreimage workflow certificate.inputPreprocessing.transitionPreimages
      (.add (.constant 4) (.multiply (.constant 2)
        (.parameter "diamond_digit_count"))) = true := by
  have matched := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at matched
  simp only [Bool.and_eq_true] at matched
  aesop

/-- The accepted preprocessing layout checks the final-trapdoor gather. -/
theorem VerifiedDiamondLayout.preprocessingFinalTrapdoorsGather
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyParallelGather workflow certificate.inputPreprocessing.finalTrapdoors = true := by
  have matched := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at matched
  simp only [Bool.and_eq_true] at matched
  aesop

/-- The checked transition-selector scan binds its carried state and four invariant arguments to
the exact names read by the fixed bit-body input nodes. -/
theorem VerifiedDiamondLayout.transitionSelectorBitBodyInputNames
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    let scan :=
      certificate.inputPreprocessing.transitionTargets.body.selectorConstruction.bitScan
    ∃ body,
      resolveScope workflow { scan.operation with scope := scan.bodyScope } = some body ∧
      body.inputNames =
        ["arg-0-matrix", "arg-1-integer", "arg-2-integer", "arg-3-integer",
          "arg-4-matrix"] := by
  have matched := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at matched
  simp only [Bool.and_eq_true] at matched
  have targets : verifyParallelTransitionTarget workflow
      certificate.inputPreprocessing.transitionTargets = true := by aesop
  unfold verifyParallelTransitionTarget at targets
  simp only [Bool.and_eq_true] at targets
  have selector : verifyTransitionSelector workflow
      certificate.inputPreprocessing.transitionTargets.body.selectorConstruction = true := by
    aesop
  unfold verifyTransitionSelector at selector
  simp only [Bool.and_eq_true] at selector
  have names : verifyTransitionSelectorBitBodyInputNames workflow
      certificate.inputPreprocessing.transitionTargets.body.selectorConstruction = true := by
    aesop
  unfold verifyTransitionSelectorBitBodyInputNames at names
  cases resolved : resolveScope workflow {
    certificate.inputPreprocessing.transitionTargets.body.selectorConstruction.bitScan.operation
      with scope :=
        certificate.inputPreprocessing.transitionTargets.body.selectorConstruction.bitScan.bodyScope
  } with
  | none => simp [resolved] at names
  | some body =>
      refine ⟨body, resolved, ?_⟩
      simpa [resolved, decide_eq_true_eq] using names

theorem VerifiedDiamondLayout.onlineSourceIndexFormula
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyOnlineSourceIndexFormula workflow certificate.inputInjection.sourceIndices = true := by
  have matched := verified.inputInjectionMatches
  unfold verifyInputInjection at matched
  simp only [Bool.and_eq_true] at matched
  aesop

theorem VerifiedDiamondLayout.onlineTransitionIndexFormula
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyOnlineTransitionIndexFormula workflow
      certificate.inputInjection.transitionIndices = true := by
  have matched := verified.inputInjectionMatches
  unfold verifyInputInjection at matched
  simp only [Bool.and_eq_true] at matched
  aesop

theorem VerifiedDiamondLayout.initialStateExpansionFormula
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyInitialStateExpansionRef workflow
      certificate.inputInjection.initialStatesExpansion = true := by
  have matched := verified.inputInjectionMatches
  unfold verifyInputInjection at matched
  simp only [Bool.and_eq_true] at matched
  aesop

/-- The accepted online state scan is an executable decryption-root node. -/
theorem VerifiedDiamondLayout.inputInjectionStateScanLocation
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    certificate.inputInjection.stateScan.stage = certificate.workflow.decryption.stage ∧
      certificate.inputInjection.stateScan.scope = .root := by
  have matched := verified.inputInjectionMatches
  have workflowMatched := verified.workflowMatches
  unfold verifyInputInjection at matched
  unfold verifyWorkflow at workflowMatched
  simp only [Bool.and_eq_true, decide_eq_true_eq] at matched workflowMatched
  aesop

/-- The checked scan consumes the exact artifacts and the checked witness-digit output. -/
theorem VerifiedDiamondLayout.inputInjectionExternalSources
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    (match certificate.inputInjection.initialStatesExpansion.parallelLoop.arguments[0]? with
      | some initialState =>
          verifyInputWire workflow initialState.wire "decrypt" "diamond_initial_state"
      | none => false) = true ∧
    verifyInputWire workflow certificate.inputInjection.transitionFamily.wire "decrypt"
      "artifact:diamond_transitions" = true ∧
    certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.outputs[0]? =
      some certificate.inputInjection.packedDigits.wire := by
  have injectionMatched := verified.inputInjectionMatches
  have encodingsMatched := verified.decryptionInitialEncodingsMatch
  unfold verifyInputInjection at injectionMatched
  unfold verifyDecryptionInitialEncodings at encodingsMatched
  simp only [Bool.and_eq_true] at injectionMatched encodingsMatched
  have externalSources := injectionMatched.2
  have packedOutput := encodingsMatched.2
  unfold verifyInputInjectionExternalSources at externalSources
  unfold verifyWitnessDigitPackingSources at packedOutput
  simp only [Bool.and_eq_true, decide_eq_true_eq] at externalSources packedOutput
  exact ⟨externalSources.1, externalSources.2, packedOutput.2⟩

/-- The packed witness digits originate at the exact decryption-root witness input. -/
theorem VerifiedDiamondLayout.witnessDigitPackingSource
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    (match certificate.decryptionInitialEncodings.witnessBits.sourceFamilies[0]? with
    | some witness =>
        verifyInputWire workflow witness.wire certificate.workflow.decryption.stage
          "boolean-witness"
    | none => false) = true := by
  have matched := verified.decryptionInitialEncodingsMatch
  unfold verifyDecryptionInitialEncodings at matched
  simp only [Bool.and_eq_true] at matched
  have sources := matched.2
  unfold verifyWitnessDigitPackingSources at sources
  simp only [Bool.and_eq_true] at sources
  exact sources.1.2

/-- The witness gather index family is the exact loop index formula used by the DSL. -/
theorem VerifiedDiamondLayout.witnessDigitIndexFormula
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyDecryptionWitnessIndexFormula workflow certificate.decryptionInitialEncodings = true := by
  have matched := verified.decryptionInitialEncodingsMatch
  unfold verifyDecryptionInitialEncodings at matched
  simp only [Bool.and_eq_true] at matched
  have sources := matched.2
  unfold verifyWitnessDigitPackingSources at sources
  simp only [Bool.and_eq_true] at sources
  exact sources.1.1.2

/-- The witness-index producer is the exact one-output, argument-free loop used by the packing
pipeline.  This accessor avoids re-elaborating the complete decryption-layout conjunction in
execution bridges. -/
theorem VerifiedDiamondLayout.witnessIndexParentFacts
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyParallelLoop workflow certificate.decryptionInitialEncodings.witnessIndices = true ∧
      certificate.decryptionInitialEncodings.witnessIndices.indexSlot = 0 ∧
      certificate.decryptionInitialEncodings.witnessIndices.bindings = [] ∧
      certificate.decryptionInitialEncodings.witnessIndices.inputModes = [] ∧
      certificate.decryptionInitialEncodings.witnessIndices.bodyOutputs.length = 1 ∧
      certificate.decryptionInitialEncodings.witnessIndices.outputs.length = 1 := by
  have checked := verified.witnessDigitIndexFormula
  unfold verifyDecryptionWitnessIndexFormula at checked
  simp only [Bool.and_eq_true] at checked
  have role := checked.1.1.1
  unfold verifyExactParallelLoopRole at role
  simp only [Bool.and_eq_true, decide_eq_true_eq] at role
  aesop

/-- The witness-bit gather is the exact one-output zip/broadcast loop used by witness packing. -/
theorem VerifiedDiamondLayout.witnessGatherParentFacts
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyParallelGather workflow certificate.decryptionInitialEncodings.witnessBits = true ∧
      verifyParallelLoop workflow
        certificate.decryptionInitialEncodings.witnessBits.parallelLoop = true ∧
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.indexSlot = 0 ∧
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.bindings = [] ∧
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.inputModes =
        [.zip, .broadcast] ∧
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.bodyOutputs.length = 1 ∧
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.outputs.length = 1 := by
  have matched := verified.decryptionInitialEncodingsMatch
  unfold verifyDecryptionInitialEncodings at matched
  simp only [Bool.and_eq_true] at matched
  have sources := matched.2
  unfold verifyWitnessDigitPackingSources at sources
  simp only [Bool.and_eq_true] at sources
  have parent := sources.1.1.1.2
  unfold verifyWitnessGatherParent at parent
  simp only [Bool.and_eq_true] at parent
  have gather := parent.1
  have role := parent.2
  unfold verifyExactParallelLoopRole at role
  simp only [Bool.and_eq_true, decide_eq_true_eq] at role
  aesop

/-- Both witness-source loops are checked to occur in the decryption root scope. -/
theorem VerifiedDiamondLayout.witnessSourceLoopLocations
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    certificate.decryptionInitialEncodings.witnessIndices.operation.stage =
        certificate.workflow.decryption.stage ∧
      certificate.decryptionInitialEncodings.witnessIndices.operation.scope = .root ∧
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.operation.stage =
        certificate.workflow.decryption.stage ∧
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.operation.scope = .root := by
  have matched := verified.decryptionInitialEncodingsMatch
  unfold verifyDecryptionInitialEncodings at matched
  simp only [Bool.and_eq_true] at matched
  have sources := matched.2
  unfold verifyWitnessDigitPackingSources at sources
  simp only [Bool.and_eq_true] at sources
  have locations := sources.1.1.1.1
  unfold verifyWitnessSourceLoopLocations at locations
  simp only [Bool.and_eq_true, decide_eq_true_eq] at locations
  rcases locations with ⟨⟨⟨indicesStage, indicesScope⟩, bitsStage⟩, bitsScope⟩
  exact ⟨indicesStage, indicesScope, bitsStage, bitsScope⟩

/-- The checked witness-digit packing loop is an executable decryption-root node. -/
theorem VerifiedDiamondLayout.witnessDigitPackingLocation
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.operation.stage =
        certificate.workflow.decryption.stage ∧
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.operation.scope = .root ∧
      verifyWitnessDigitPackingRef workflow
        certificate.decryptionInitialEncodings.witnessDigits = true := by
  have matched := verified.decryptionInitialEncodingsMatch
  unfold verifyDecryptionInitialEncodings at matched
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_cons, List.all_nil, and_true] at matched
  aesop

theorem VerifiedDiamondLayout.witnessDigitPackingFormula
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyWitnessDigitPackingRef workflow
      certificate.decryptionInitialEncodings.witnessDigits = true := by
  have matched := verified.decryptionInitialEncodingsMatch
  unfold verifyDecryptionInitialEncodings at matched
  simp only [Bool.and_eq_true] at matched
  aesop

/-- The witness-digit packing formula includes a successfully checked parent parallel loop. -/
theorem VerifiedDiamondLayout.witnessDigitPackingParentVerified
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyParallelLoop workflow
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop = true := by
  have checked := verified.witnessDigitPackingFormula
  unfold verifyWitnessDigitPackingRef at checked
  simp only [Bool.and_eq_true] at checked
  aesop

/-- The witness-packing outer loop and inner scan bind the exact names read by their fixed bodies. -/
theorem VerifiedDiamondLayout.witnessDigitPackingInputNames
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyWitnessDigitPackingInputNames workflow
      certificate.decryptionInitialEncodings.witnessDigits = true := by
  have checked := verified.witnessDigitPackingFormula
  unfold verifyWitnessDigitPackingRef at checked
  simp only [Bool.and_eq_true] at checked
  exact checked.2

theorem verifyWorkflow_encryptOutputs {workflow : Mxx.Ir.Workflow}
    {layout : DiamondWorkflowLayout} (verified : verifyWorkflow workflow layout = true) :
    layout.encryption.outputs.map (·.name) = [
      "diamond_decoder_preimage", "diamond_initial_state", "diamond_k_preimage",
      "diamond_one_preimage", "diamond_public_keys", "diamond_r_decomposed",
      "diamond_transitions", "diamond_witness_preimages"] := by
  unfold verifyWorkflow at verified
  split at verified
  <;> simp_all [Bool.and_eq_true, decide_eq_true_eq]

/-- The message construction consumes exactly the protocol's `diamond-message` encryption input.
This excludes certificates that redirect `boolToInt` to another same-stage input. -/
theorem verifyMessageConstruction_messageInput
    {workflow : Mxx.Ir.Workflow} {workflowLayout : DiamondWorkflowLayout}
    {layout : MessageConstructionLayout}
    (verified : verifyMessageConstruction workflow workflowLayout layout = true) :
    ∃ message,
      layout.toInt.inputs = [message] ∧
      verifyInputWire workflow message.wire workflowLayout.encryption.stage
        "diamond-message" = true := by
  unfold verifyMessageConstruction at verified
  simp only [Bool.and_eq_true] at verified
  cases inputs : layout.toInt.inputs with
  | nil => simp [inputs] at verified
  | cons message tail =>
      cases tail with
      | nil =>
          simp only [inputs] at verified
          exact ⟨message, rfl, by aesop⟩
      | cons next rest => simp [inputs] at verified

theorem verifiedWorkflow_hasNoGateDecompositionArtifact {workflow : Mxx.Ir.Workflow}
    {layout : DiamondWorkflowLayout} (verified : verifyWorkflow workflow layout = true) :
    "diamond_gate_rhs_decomposition" ∉ layout.encryption.outputs.map (·.name) := by
  rw [verifyWorkflow_encryptOutputs verified]
  simp

theorem verifyOperation_resolves {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    (verified : verifyOperation workflow reference = true) :
    ∃ node,
      resolveNode workflow reference.operation = some node ∧
      node.arguments = reference.inputs.map (wireRef ∘ CoreOperandRef.wire) ∧
      node.outputCount = reference.outputs.length := by
  cases equation : resolveNode workflow reference.operation with
  | none => simp [verifyOperation, equation] at verified
  | some node =>
      rw [verifyOperation, equation] at verified
      simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
      exact ⟨node, rfl, verified.2.1, verified.2.2⟩

theorem verifyInputPreprocessing_artifacts
    {workflow : Mxx.Ir.Workflow} {workflowLayout : DiamondWorkflowLayout}
    {layout : DiamondInputPreprocessingLayout}
    (verified : verifyInputPreprocessing workflow workflowLayout layout = true) :
    layout.initialStateArtifact ∈ workflowLayout.artifacts ∧
      layout.transitionsArtifact ∈ workflowLayout.artifacts := by
  unfold verifyInputPreprocessing at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  aesop

/-- A checked parallel loop binds every runtime argument to exactly one distinct body input name.
This closes the gap between structural input-wire references and the `inputNames.zip inputs`
environment used by the executable child runner. -/
theorem verifyParallelLoop_bodyInputBindings
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef} {body : Mxx.Ir.Scope}
    (verified : verifyParallelLoop workflow reference = true)
    (bodyResolved :
      resolveScope workflow { reference.operation with scope := reference.bodyScope } = some body) :
    body.inputNames.Nodup ∧
      (scopeInputWires body).length = body.inputNames.length ∧
      reference.arguments.length = body.inputNames.length ∧
      reference.bodyInputs.map wireRef = scopeInputWires body := by
  unfold verifyParallelLoop at verified
  cases nodeResolved : resolveNode workflow reference.operation with
  | none => simp [nodeResolved, bodyResolved] at verified
  | some node =>
      rw [nodeResolved, bodyResolved] at verified
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> simp_all [Bool.and_eq_true, decide_eq_true_eq]

/-- A checked parallel loop's declared body outputs are exactly the executable scope outputs. -/
theorem verifyParallelLoop_bodyOutputs
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef} {body : Mxx.Ir.Scope}
    (verified : verifyParallelLoop workflow reference = true)
    (bodyResolved :
      resolveScope workflow { reference.operation with scope := reference.bodyScope } = some body) :
    reference.bodyOutputs.map wireRef = body.outputs.map Prod.snd := by
  unfold verifyParallelLoop at verified
  cases nodeResolved : resolveNode workflow reference.operation with
  | none => simp [nodeResolved, bodyResolved] at verified
  | some node =>
      rw [nodeResolved, bodyResolved] at verified
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> simp_all [Bool.and_eq_true, decide_eq_true_eq, scopeOutputWires]

end MxxWe.Certificate
