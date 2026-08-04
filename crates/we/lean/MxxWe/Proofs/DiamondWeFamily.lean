import MxxWe.DiamondGeneric
import MxxWe.DiamondFamilyChecker
import MxxWe.Generated.DiamondWeFamily.Certificate
import MxxWe.Generated.DiamondWeFamily.Statement
import MxxWe.GeneratedPreconditions
import MxxWe.Certificate.BooleanExecutionBridge
import MxxWe.Certificate.DecoderExecutionBridge
import MxxWe.Certificate.EncryptionExecutionBridge
import MxxWe.Certificate.InputInjectionWorkflowBridge
import MxxWe.Certificate.VerifierSound

open Mxx
open MxxWe.Generated.DiamondWeFamily

namespace MxxWe.Proofs.DiamondWeFamily

set_option maxRecDepth 100000 in
private theorem diamondWeFamily_workflow_verified :
    MxxWe.Certificate.verifyWorkflow DiamondWeFamily_workflow
      DiamondWeFamily_certificate.workflow = true := by
  rfl

set_option maxRecDepth 100000 in
private theorem diamondWeFamily_message_verified :
    MxxWe.Certificate.verifyMessageConstruction DiamondWeFamily_workflow
      DiamondWeFamily_certificate.workflow DiamondWeFamily_certificate.message = true := by
  rfl

set_option maxRecDepth 100000 in
private theorem diamondWeFamily_inputPreprocessing_verified :
    MxxWe.Certificate.verifyInputPreprocessing DiamondWeFamily_workflow
      DiamondWeFamily_certificate.workflow DiamondWeFamily_certificate.inputPreprocessing =
        true := by
  rfl

set_option maxRecDepth 100000 in
private theorem diamondWeFamily_publicKeySampling_verified :
    MxxWe.Certificate.verifyPublicKeySampling DiamondWeFamily_workflow
      DiamondWeFamily_certificate.workflow DiamondWeFamily_certificate.publicKeySampling =
        true := by
  rfl

set_option maxRecDepth 100000 in
private theorem diamondWeFamily_encryptionInitialPublicKeys_verified :
    MxxWe.Certificate.verifyEncryptionInitialPublicKeys DiamondWeFamily_workflow
      DiamondWeFamily_certificate.workflow DiamondWeFamily_certificate.publicKeySampling
      DiamondWeFamily_certificate.encryptionInitialPublicKeys = true := by
  rfl

set_option maxRecDepth 100000 in
private theorem diamondWeFamily_artifactPreprocessing_verified :
    MxxWe.Certificate.verifyArtifactPreprocessing DiamondWeFamily_workflow
      DiamondWeFamily_certificate.workflow DiamondWeFamily_certificate.inputPreprocessing
      DiamondWeFamily_certificate.publicKeySampling DiamondWeFamily_certificate.booleanLayers
      DiamondWeFamily_certificate.artifactPreprocessing = true := by
  rfl

set_option maxRecDepth 100000 in
private theorem diamondWeFamily_inputInjection_verified :
    MxxWe.Certificate.verifyInputInjection DiamondWeFamily_workflow
      DiamondWeFamily_certificate.inputInjection = true := by
  rfl

set_option maxRecDepth 100000 in
private theorem diamondWeFamily_decryptionInitialEncodings_verified :
    MxxWe.Certificate.verifyDecryptionInitialEncodings DiamondWeFamily_workflow
      DiamondWeFamily_certificate.workflow DiamondWeFamily_certificate.inputInjection
      DiamondWeFamily_certificate.booleanLayers
      DiamondWeFamily_certificate.decryptionInitialEncodings = true := by
  rfl

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 100000 in
private theorem diamondWeFamily_booleanLayers_verified :
    MxxWe.Certificate.verifyBooleanLayers DiamondWeFamily_workflow
      DiamondWeFamily_certificate.workflow DiamondWeFamily_certificate.booleanLayers = true := by
  rfl

set_option maxRecDepth 100000 in
private theorem diamondWeFamily_decoder_verified :
    MxxWe.Certificate.verifyDecoder DiamondWeFamily_workflow
      DiamondWeFamily_certificate.workflow DiamondWeFamily_certificate.booleanLayers
      DiamondWeFamily_certificate.decoder = true := by
  rfl

set_option maxHeartbeats 5000000 in
set_option maxRecDepth 100000 in
private theorem diamondWeFamily_closure_verified :
    MxxWe.Certificate.verifyOutputRootedClosure DiamondWeFamily_workflow
      DiamondWeFamily_certificate = true := by
  rfl

set_option maxRecDepth 100000 in
private theorem diamondWeFamily_definitionsUnique_verified :
    MxxWe.Certificate.definitionsUnique DiamondWeFamily_workflow = true := by
  rfl

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 100000 in
private theorem diamondWeFamily_ssaOrder_verified :
    MxxWe.Certificate.verifyWorkflowSsaOrder DiamondWeFamily_workflow = true := by
  rfl

/-- The frozen generated certificate is accepted by the executable structural verifier. -/
theorem diamondWeFamily_certificate_verified :
    MxxWe.Certificate.verifyDiamondCertificate DiamondWeFamily_workflow
      DiamondWeFamily_certificate = true := by
  unfold MxxWe.Certificate.verifyDiamondCertificate
  simp only [diamondWeFamily_workflow_verified, diamondWeFamily_message_verified,
    diamondWeFamily_inputPreprocessing_verified, diamondWeFamily_publicKeySampling_verified,
    diamondWeFamily_encryptionInitialPublicKeys_verified,
    diamondWeFamily_artifactPreprocessing_verified, diamondWeFamily_inputInjection_verified,
    diamondWeFamily_decryptionInitialEncodings_verified, diamondWeFamily_booleanLayers_verified,
    diamondWeFamily_decoder_verified, diamondWeFamily_closure_verified,
    diamondWeFamily_definitionsUnique_verified, diamondWeFamily_ssaOrder_verified,
    Bool.true_and]

/-- Kernel-checked structural facts extracted from the accepted generated certificate. -/
theorem diamondWeFamily_verifiedLayout :
    MxxWe.Certificate.VerifiedDiamondLayout DiamondWeFamily_workflow
      DiamondWeFamily_certificate :=
  MxxWe.Certificate.verifyDiamondCertificate_sound diamondWeFamily_certificate_verified

/-- Exact checked matrix nodes used by the concrete family decoder. -/
theorem diamondWeFamily_decoderMatrixOperations :
    MxxWe.Certificate.VerifiedDecoderMatrixOperations DiamondWeFamily_workflow
      DiamondWeFamily_certificate.decoder :=
  diamondWeFamily_verifiedLayout.decoderMatrixOperations

/-- Exact checked scalar nodes used by the concrete family decoder. -/
theorem diamondWeFamily_decoderScalarOperations :
    MxxWe.Certificate.VerifiedDecoderScalarOperations DiamondWeFamily_workflow
      DiamondWeFamily_certificate.decoder :=
  diamondWeFamily_verifiedLayout.decoderScalarOperations

/-- Exact metadata, candidate, masking, and decomposition wiring of both Boolean loops. -/
theorem diamondWeFamily_booleanExecutionWiring :
    MxxWe.Certificate.VerifiedBooleanExecutionWiring DiamondWeFamily_workflow
      DiamondWeFamily_certificate.booleanLayers :=
  diamondWeFamily_verifiedLayout.booleanExecutionWiring

/-- The public-key Boolean layer is the checked executable sequential loop. -/
theorem diamondWeFamily_publicKeyBooleanLoopResolution :
    Nonempty (MxxWe.Certificate.BooleanSequentialLoopResolution DiamondWeFamily_workflow
      DiamondWeFamily_certificate.booleanLayers.encryption.layerScan
      DiamondWeFamily_certificate.booleanLayers.encryption.bodyScope
      [DiamondWeFamily_certificate.booleanLayers.encryption.initialPublicKeys]
      [DiamondWeFamily_certificate.booleanLayers.encryption.activeGateCounts,
        DiamondWeFamily_certificate.booleanLayers.encryption.gateKinds,
        DiamondWeFamily_certificate.booleanLayers.encryption.leftSources,
        DiamondWeFamily_certificate.booleanLayers.encryption.rightSources,
        DiamondWeFamily_certificate.booleanLayers.encryption.onePublicKey]
      [DiamondWeFamily_certificate.booleanLayers.encryption.finalPublicKeys]) :=
  diamondWeFamily_verifiedLayout.publicKeyBooleanLoopResolution

/-- The encoding Boolean layer is the checked executable three-component sequential loop. -/
theorem diamondWeFamily_encodingBooleanLoopResolution :
    Nonempty (MxxWe.Certificate.BooleanSequentialLoopResolution DiamondWeFamily_workflow
      DiamondWeFamily_certificate.booleanLayers.decryption.layerScan
      DiamondWeFamily_certificate.booleanLayers.decryption.bodyScope
      [DiamondWeFamily_certificate.booleanLayers.decryption.initialVectors,
        DiamondWeFamily_certificate.booleanLayers.decryption.initialPublicKeys,
        DiamondWeFamily_certificate.booleanLayers.decryption.initialPlaintexts]
      [DiamondWeFamily_certificate.booleanLayers.decryption.activeGateCounts,
        DiamondWeFamily_certificate.booleanLayers.decryption.gateKinds,
        DiamondWeFamily_certificate.booleanLayers.decryption.leftSources,
        DiamondWeFamily_certificate.booleanLayers.decryption.rightSources,
        DiamondWeFamily_certificate.booleanLayers.decryption.oneVector,
        DiamondWeFamily_certificate.booleanLayers.decryption.onePublicKey,
        DiamondWeFamily_certificate.booleanLayers.decryption.onePlaintext]
      [DiamondWeFamily_certificate.booleanLayers.decryption.finalVectors,
        DiamondWeFamily_certificate.booleanLayers.decryption.finalPublicKeys,
        DiamondWeFamily_certificate.booleanLayers.decryption.finalPlaintexts]) :=
  diamondWeFamily_verifiedLayout.encodingBooleanLoopResolution

theorem diamondWeFamilyChecker_generatedValid (p : DiamondWeFamilyParams)
    (accepted : diamondWeFamilyChecker p = true) :
    DiamondWeFamilyParamsValid p = true := by
  simp only [diamondWeFamilyChecker, Bool.and_eq_true] at accepted
  exact accepted.1

theorem diamondWeFamilyChecker_genericAccepted (p : DiamondWeFamilyParams)
    (accepted : diamondWeFamilyChecker p = true) :
    diamondWeChecker (genericParameters p) = true := by
  simp only [diamondWeFamilyChecker, Bool.and_eq_true] at accepted
  exact accepted.2

theorem diamondWeFamily_accepts_valid_parameters :
    ∃ p : DiamondWeFamilyParams,
      diamondWeFamilyChecker p = true ∧ DiamondWeFamilyParamsValid p = true := by
  let p : DiamondWeFamilyParams :=
    { instanceWidth := 0
      witnessWidth := 1
      depth := 1
      maxLayerWidth := 1
      diamondRingDimension := 1
      diamondInputCount := 1
      diamondDigitBase := 2
      diamondBatchBits := 1
      diamondDigitCount := 1
      diamondModulus := 8
      diamondGadgetBase := 2
      diamondErrorMaxCoefficientBound := 0
      diamondPreimageMaxCoefficientBound := 0
      diamondTrapdoorSigma := 1
      diamondErrorSigma := 0 }
  refine ⟨p, ?_, ?_⟩
  · norm_num [diamondWeFamilyChecker, genericParameters, DiamondWeFamilyParamsValid,
      diamondWeChecker, diamondWeParametersValid, DiamondWeParameters.finalBound,
      DiamondWeParameters.oneEncodingBound, DiamondWeParameters.inputStateBound,
      DiamondWeParameters.circuitNoiseBound, DiamondWeParameters.digitBound,
      DiamondWeParameters.inputWidth, DiamondWeParameters.stateRows,
      DiamondWeParameters.stateColumns, DiamondWeParameters.publicColumns, injectionBound,
      injectionStep, circuitBound, gateStep, productBound]
  · norm_num [DiamondWeFamilyParamsValid]

theorem signedParameters_nonnegative (p : DiamondWeFamilyParams)
    (valid : DiamondWeFamilyParamsValid p = true) :
    0 ≤ p.diamondModulus ∧ 0 ≤ p.diamondGadgetBase ∧
      0 ≤ p.diamondErrorMaxCoefficientBound ∧
      0 ≤ p.diamondPreimageMaxCoefficientBound := by
  simp only [DiamondWeFamilyParamsValid, Bool.and_eq_true, decide_eq_true_eq] at valid
  omega

theorem genericParameters_modulus (p : DiamondWeFamilyParams)
    (valid : DiamondWeFamilyParamsValid p = true) :
    Int.ofNat (genericParameters p).modulus = p.diamondModulus := by
  simp only [genericParameters]
  exact Int.toNat_of_nonneg (signedParameters_nonnegative p valid).1

theorem genericParameters_gadgetBase (p : DiamondWeFamilyParams)
    (valid : DiamondWeFamilyParamsValid p = true) :
    Int.ofNat (genericParameters p).gadgetBase = p.diamondGadgetBase := by
  simp only [genericParameters]
  exact Int.toNat_of_nonneg (signedParameters_nonnegative p valid).2.1

theorem genericParameters_errorBound (p : DiamondWeFamilyParams)
    (valid : DiamondWeFamilyParamsValid p = true) :
    Int.ofNat (genericParameters p).errorMaxCoefficientBound =
      p.diamondErrorMaxCoefficientBound := by
  simp only [genericParameters]
  exact Int.toNat_of_nonneg (signedParameters_nonnegative p valid).2.2.1

theorem genericParameters_preimageBound (p : DiamondWeFamilyParams)
    (valid : DiamondWeFamilyParamsValid p = true) :
    Int.ofNat (genericParameters p).preimageMaxCoefficientBound =
      p.diamondPreimageMaxCoefficientBound := by
  simp only [genericParameters]
  exact Int.toNat_of_nonneg (signedParameters_nonnegative p valid).2.2.2

/-- The accepted family parameters expose exactly the two arithmetic facts used by executable
witness packing.  No bound or radix relation is supplied by a caller of the final theorem. -/
theorem genericParameters_witnessPackingRelations (p : DiamondWeFamilyParams)
    (accepted : diamondWeFamilyChecker p = true) :
    (genericParameters p).witnessWidth =
        (genericParameters p).inputCount * (genericParameters p).batchBits ∧
      2 ^ (genericParameters p).batchBits ≤ (genericParameters p).digitBase := by
  have checked := diamondWeChecker_parametersValid (genericParameters p)
    (diamondWeFamilyChecker_genericAccepted p accepted)
  simp only [diamondWeParametersValid, Bool.and_eq_true, decide_eq_true_eq] at checked
  aesop

/-- The generated input predicate and circuit predicate together supply the concrete source facts
needed by witness packing: the witness list has the fixed allocation length, and every active
witness entry is a canonical bit. -/
theorem diamondWeFamily_witnessSourceFacts
    (p : DiamondWeFamilyParams) (x : DiamondWeFamilyInputs p)
    (accepted : diamondWeFamilyChecker p = true)
    (inputsWellFormed : DiamondWeFamilyInputsWF p x)
    (preconditions : DiamondWeFamilyPreconditions p x) :
    x.booleanWitness.length = p.maxLayerWidth ∧
      (∀ slot, slot < p.witnessWidth →
        x.booleanWitness[slot]?.getD 0 = 0 ∨
          x.booleanWitness[slot]?.getD 0 = 1) := by
  have witnessLength : x.booleanWitness.length = p.maxLayerWidth := by
    unfold DiamondWeFamilyInputsWF at inputsWellFormed
    simpa using inputsWellFormed.2.2.2.2.2.2.1
  have checked := diamondWeChecker_parametersValid (genericParameters p)
    (diamondWeFamilyChecker_genericAccepted p accepted)
  simp only [diamondWeParametersValid, Bool.and_eq_true, decide_eq_true_eq] at checked
  have witnessLe : p.witnessWidth ≤ p.maxLayerWidth := by
    have inputWidthLe : p.instanceWidth + p.witnessWidth ≤ p.maxLayerWidth := by
      simpa [genericParameters, DiamondWeParameters.inputWidth] using checked.2.2.1
    omega
  have satisfied := DiamondWeFamilyPreconditions.circuitSatisfied preconditions
  refine ⟨witnessLength, ?_⟩
  intro slot slotLt
  have slotMax : slot < p.maxLayerWidth := lt_of_lt_of_le slotLt witnessLe
  simpa [DynamicBoolean.CanonicalValues, slotLt] using
    satisfied.witnessCanonical slot slotMax

/-- Every bit addressed by an accepted input level and bit offset is present in the generated
witness family. -/
theorem diamondWeFamily_witnessBitPresent
    (p : DiamondWeFamilyParams) (x : DiamondWeFamilyInputs p)
    (accepted : diamondWeFamilyChecker p = true)
    (inputsWellFormed : DiamondWeFamilyInputsWF p x)
    {level bit : Nat} (levelLt : level < p.diamondInputCount)
    (bitLt : bit < p.diamondBatchBits) :
    ∃ value, x.booleanWitness[level * p.diamondBatchBits + bit]? = some value := by
  have witnessLength : x.booleanWitness.length = p.maxLayerWidth := by
    unfold DiamondWeFamilyInputsWF at inputsWellFormed
    simpa using inputsWellFormed.2.2.2.2.2.2.1
  have checked := diamondWeChecker_parametersValid (genericParameters p)
    (diamondWeFamilyChecker_genericAccepted p accepted)
  simp only [diamondWeParametersValid, Bool.and_eq_true, decide_eq_true_eq] at checked
  have widthEq : p.witnessWidth = p.diamondInputCount * p.diamondBatchBits := by
    simpa [genericParameters] using checked.2.2.2.1
  have witnessLe : p.witnessWidth ≤ p.maxLayerWidth := by
    have inputWidthLe : p.instanceWidth + p.witnessWidth ≤ p.maxLayerWidth := by
      simpa [genericParameters, DiamondWeParameters.inputWidth] using checked.2.2.1
    omega
  have addressLtWitness : level * p.diamondBatchBits + bit < p.witnessWidth := by
    rw [widthEq]
    calc
      level * p.diamondBatchBits + bit <
          level * p.diamondBatchBits + p.diamondBatchBits :=
        Nat.add_lt_add_left bitLt _
      _ = (level + 1) * p.diamondBatchBits := by simp [Nat.add_mul]
      _ ≤ p.diamondInputCount * p.diamondBatchBits :=
        Nat.mul_le_mul_right p.diamondBatchBits (Nat.succ_le_iff.mpr levelLt)
  have addressLt : level * p.diamondBatchBits + bit < x.booleanWitness.length := by
    rw [witnessLength]
    exact lt_of_lt_of_le addressLtWitness witnessLe
  refine ⟨x.booleanWitness[level * p.diamondBatchBits + bit], ?_⟩
  rw [List.getElem?_eq_getElem addressLt]

/-- The generated witness-index and gather loops both range over the configured witness width. -/
private theorem diamondWeFamily_witnessSourceCountsEvaluate (p : DiamondWeFamilyParams) :
    DiamondWeFamily_certificate.decryptionInitialEncodings.witnessIndices.count.evaluate
        (DiamondWeFamilyParamEnvironment p) = some (Int.ofNat p.witnessWidth) ∧
      DiamondWeFamily_certificate.decryptionInitialEncodings.witnessBits.parallelLoop.count.evaluate
        (DiamondWeFamilyParamEnvironment p) = some (Int.ofNat p.witnessWidth) := by
  simp [DiamondWeFamily_certificate, DiamondWeFamilyParamEnvironment,
    Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupParam]

/-- The generated packing loops use the configured input-count and batch-bit parameters. -/
private theorem diamondWeFamily_witnessPackingCountsEvaluate (p : DiamondWeFamilyParams) :
    DiamondWeFamily_certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.count.evaluate
        (DiamondWeFamilyParamEnvironment p) = some (Int.ofNat p.diamondInputCount) ∧
      DiamondWeFamily_certificate.decryptionInitialEncodings.witnessDigits.bitScan.count.evaluate
        (DiamondWeFamilyParamEnvironment p) = some (Int.ofNat p.diamondBatchBits) := by
  simp [DiamondWeFamily_certificate, DiamondWeFamilyParamEnvironment,
    Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupParam]

def encryptStageInputs (p : DiamondWeFamilyParams) (x : DiamondWeFamilyInputs p) :
    Mxx.Ir.Environment :=
  [("diamond-hash-key", .bytes x.diamondHashKey),
    ("boolean-instance", .family (x.booleanInstance.map .integer)),
    ("circuit-active-gate-count", .family (x.circuitActiveGateCount.map .integer)),
    ("circuit-gate-kind", .family (x.circuitGateKind.map .integer)),
    ("circuit-left-source", .family (x.circuitLeftSource.map .integer)),
    ("circuit-right-source", .family (x.circuitRightSource.map .integer)),
    ("circuit-output-source", .family (x.circuitOutputSource.map .integer)),
    ("diamond-message", .boolean x.diamondMessage)]

def decryptStageInputs (p : DiamondWeFamilyParams) (x : DiamondWeFamilyInputs p)
    (encrypted : Mxx.Ir.Environment) : Mxx.Ir.Environment :=
  let artifact name := Mxx.Ir.resolveStageInput [] [("encrypt", encrypted)]
    (.artifact "encrypt" name)
  [("diamond_initial_state", artifact "diamond_initial_state"),
    ("boolean-witness", .family (x.booleanWitness.map .integer)),
    ("artifact:diamond_transitions", artifact "diamond_transitions"),
    ("diamond_decoder_preimage", artifact "diamond_decoder_preimage"),
    ("diamond_k_preimage", artifact "diamond_k_preimage"),
    ("diamond_one_preimage", artifact "diamond_one_preimage"),
    ("artifact:diamond_witness_preimages", artifact "diamond_witness_preimages"),
    ("boolean-instance", .family (x.booleanInstance.map .integer)),
    ("artifact:diamond_public_keys", artifact "diamond_public_keys"),
    ("circuit-active-gate-count", .family (x.circuitActiveGateCount.map .integer)),
    ("circuit-gate-kind", .family (x.circuitGateKind.map .integer)),
    ("circuit-left-source", .family (x.circuitLeftSource.map .integer)),
    ("circuit-right-source", .family (x.circuitRightSource.map .integer)),
    ("circuit-output-source", .family (x.circuitOutputSource.map .integer)),
    ("diamond_r_decomposed", artifact "diamond_r_decomposed")]

private theorem decryptStageInputs_witness (p : DiamondWeFamilyParams)
    (x : DiamondWeFamilyInputs p) (encrypted : Mxx.Ir.Environment) :
    Mxx.Ir.lookupEnvironment "boolean-witness" (decryptStageInputs p x encrypted) =
      some (.family (x.booleanWitness.map .integer)) := by
  simp [decryptStageInputs, Mxx.Ir.lookupEnvironment]

theorem workflowOutcomeMembers (samplers : MxxSamplerFamily) (p : DiamondWeFamilyParams)
    (x : DiamondWeFamilyInputs p) (output : Mxx.Ir.Environment)
    (member : output ∈ DiamondWeFamilyConcreteOutcomes samplers p x) :
    ∃ encrypted,
      encrypted ∈ Mxx.Ir.denote samplers DiamondWeFamily_stage_encrypt
        (DiamondWeFamilyParamEnvironment p) (encryptStageInputs p x) ∧
      output ∈ Mxx.Ir.denote samplers DiamondWeFamily_stage_decrypt
        (DiamondWeFamilyParamEnvironment p) (decryptStageInputs p x encrypted) := by
  simpa [DiamondWeFamilyConcreteOutcomes, DiamondWeFamily_workflow,
    DiamondWeFamilyInputEnvironment, Mxx.Ir.denoteWorkflow, Mxx.Ir.evaluateStages,
    Mxx.Ir.stageInputs, Mxx.Ir.resolveStageInput, Mxx.Ir.lookupStage, Mxx.Ir.lookupEnvironment,
    encryptStageInputs, decryptStageInputs] using member

def decodedOutputEnvironment (value : Bool) : Mxx.Ir.Environment :=
  [("diamond-decoded", .boolean value)]

theorem failureBoolSafe (p : DiamondWeFamilyParams) (x : DiamondWeFamilyInputs p)
    (decoded : Bool) (correct : decoded = x.diamondMessage) :
    DiamondWeFamilyFailureBool p x (decodedOutputEnvironment decoded) = false := by
  subst decoded
  cases message : x.diamondMessage <;>
    simp [DiamondWeFamilyFailureBool, DiamondWeFamilyIdealOutput,
      DiamondWeFamily_ideal, DiamondWeFamilyParamEnvironment,
      DiamondWeFamilyInputEnvironment, Mxx.Ir.denotePure, Mxx.Ir.denote,
      Mxx.Ir.denoteScopeWithFuel, Mxx.Ir.lookupDefinition, Mxx.Ir.evaluateNodes,
      Mxx.Ir.evaluateNode, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs,
      Mxx.Ir.collectOutputs, Mxx.Ir.lookupEnvironment,
      Mxx.Ir.projectOutputs, Mxx.Ir.environmentValues, Mxx.Ir.environmentValid,
      Mxx.Ir.Value.isValid, Mxx.Ir.valuesEqual, Mxx.Ir.Value.equal,
      decodedOutputEnvironment, message]

theorem failureBoolSafe_of_lookup (p : DiamondWeFamilyParams)
    (x : DiamondWeFamilyInputs p) (output : Mxx.Ir.Environment)
    (decoded : Mxx.Ir.lookupEnvironment "diamond-decoded" output =
      some (.boolean x.diamondMessage)) :
    DiamondWeFamilyFailureBool p x output = false := by
  simp [DiamondWeFamilyFailureBool, DiamondWeFamilyIdealOutput,
    DiamondWeFamily_ideal, DiamondWeFamilyParamEnvironment,
    DiamondWeFamilyInputEnvironment, Mxx.Ir.denotePure, Mxx.Ir.denote,
    Mxx.Ir.denoteScopeWithFuel, Mxx.Ir.lookupDefinition, Mxx.Ir.evaluateNodes,
    Mxx.Ir.evaluateNode, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs,
    Mxx.Ir.collectOutputs, Mxx.Ir.lookupEnvironment, Mxx.Ir.projectOutputs,
    Mxx.Ir.environmentValues, Mxx.Ir.environmentValid, Mxx.Ir.Value.isValid,
    Mxx.Ir.valuesEqual, Mxx.Ir.Value.equal, decoded]

/-- Typed output of the input-injection and Boolean execution bridges needed by the final
decoder bound.  The fields are algebraic trace results, not raw SSA lookups. -/
private structure ResidualRecurrenceResult (p : DiamondWeParameters) where
  initialSignal : Mxx.Matrix
  initialError : Mxx.Matrix
  finalSignal : Mxx.Matrix
  finalStateError : Mxx.Matrix
  onePreimage : Mxx.Matrix
  oneError : Mxx.Matrix
  states : Nat → List Mxx.Matrix
  circuitError : Mxx.Matrix
  circuitIndex : Fin (states p.depth).length
  decoderError : Mxx.Matrix
  kError : Mxx.Matrix
  rDecomposed : Mxx.Matrix
  residual : Mxx.Matrix
  inputTrace : InputInjectionTrace p.modulus p.ringDimension p.stateColumns
    p.errorMaxCoefficientBound p.preimageMaxCoefficientBound
    (1, p.errorMaxCoefficientBound) initialSignal initialError
    ((List.range p.inputCount).foldl
      (fun bounds _ ↦ injectionStep p.ringDimension 2 p.stateColumns
        p.errorMaxCoefficientBound p.preimageMaxCoefficientBound bounds)
      (1, p.errorMaxCoefficientBound))
    finalSignal finalStateError
  initialSignalNorm : Mxx.maxCenteredCoefficientNorm initialSignal ≤ 1
  initialErrorNorm :
    Mxx.maxCenteredCoefficientNorm initialError ≤ p.errorMaxCoefficientBound
  finalStateShape : Mxx.Toolkit.MatrixShape finalStateError p.modulus p.ringDimension
    1 p.stateColumns
  onePreimageShape : Mxx.Toolkit.MatrixShape onePreimage p.modulus p.ringDimension
    p.stateColumns p.publicColumns
  onePreimageNorm :
    Mxx.maxCenteredCoefficientNorm onePreimage ≤ p.preimageMaxCoefficientBound
  oneErrorEq : oneError = projectedStateNoise finalStateError onePreimage
  statesInitial : EveryNoiseBounded (states 0) (2 * p.oneEncodingBound)
  statesStep : ∀ layer : Nat, layer < p.depth →
    EveryNoiseBounded (states layer)
      ((List.range layer).foldl
        (fun bound _ ↦ gateStep p.ringDimension p.publicColumns p.digitBound
          p.oneEncodingBound bound)
        (2 * p.oneEncodingBound)) →
    EveryNoiseBounded (states (layer + 1))
      (gateStep p.ringDimension p.publicColumns p.digitBound p.oneEncodingBound
        ((List.range layer).foldl
          (fun bound _ ↦ gateStep p.ringDimension p.publicColumns p.digitBound
            p.oneEncodingBound bound)
          (2 * p.oneEncodingBound)))
  circuitErrorEq : circuitError = (states p.depth).get circuitIndex
  circuitShape : Mxx.Toolkit.MatrixShape circuitError p.modulus p.ringDimension
    1 p.publicColumns
  decoderShape : Mxx.Toolkit.MatrixShape decoderError p.modulus p.ringDimension 1 1
  kShape : Mxx.Toolkit.MatrixShape kError p.modulus p.ringDimension 1 1
  rShape : Mxx.Toolkit.MatrixShape rDecomposed p.modulus p.ringDimension
    p.publicColumns 1
  decoderNorm : Mxx.maxCenteredCoefficientNorm decoderError ≤ p.oneEncodingBound
  kNorm : Mxx.maxCenteredCoefficientNorm kError ≤ p.oneEncodingBound
  rNorm : Mxx.maxCenteredCoefficientNorm rDecomposed ≤ p.digitBound
  residualEq :
    residual = finalDecoderNoise decoderError kError oneError circuitError rDecomposed

/-- Compose the sampler-contract input trace, the arbitrary-depth Boolean recurrence, and the
final decoder recurrence.  Exact child executions only need to supply the already-refined trace,
the Boolean step facts, and the component matrices; no child-body semantics is repeated here. -/
theorem residual_bound_of_recurrences
    (p : DiamondWeParameters) [NeZero p.modulus]
    (initialSignal initialError finalSignal finalStateError onePreimage oneError : Mxx.Matrix)
    (states : Nat → List Mxx.Matrix) (circuitError : Mxx.Matrix)
    (circuitIndex : Fin (states p.depth).length)
    (decoderError kError rDecomposed residual : Mxx.Matrix)
    (inputTrace : InputInjectionTrace p.modulus p.ringDimension p.stateColumns
      p.errorMaxCoefficientBound p.preimageMaxCoefficientBound
      (1, p.errorMaxCoefficientBound) initialSignal initialError
      ((List.range p.inputCount).foldl
        (fun bounds _ ↦ injectionStep p.ringDimension 2 p.stateColumns
          p.errorMaxCoefficientBound p.preimageMaxCoefficientBound bounds)
        (1, p.errorMaxCoefficientBound))
      finalSignal finalStateError)
    (initialSignalNorm : Mxx.maxCenteredCoefficientNorm initialSignal ≤ 1)
    (initialErrorNorm :
      Mxx.maxCenteredCoefficientNorm initialError ≤ p.errorMaxCoefficientBound)
    (finalStateShape : Mxx.Toolkit.MatrixShape finalStateError p.modulus p.ringDimension
      1 p.stateColumns)
    (onePreimageShape : Mxx.Toolkit.MatrixShape onePreimage p.modulus p.ringDimension
      p.stateColumns p.publicColumns)
    (onePreimageNorm :
      Mxx.maxCenteredCoefficientNorm onePreimage ≤ p.preimageMaxCoefficientBound)
    (oneErrorEq : oneError = projectedStateNoise finalStateError onePreimage)
    (statesInitial : EveryNoiseBounded (states 0) (2 * p.oneEncodingBound))
    (statesStep : ∀ layer : Nat, layer < p.depth →
      EveryNoiseBounded (states layer)
        ((List.range layer).foldl
          (fun bound _ ↦ gateStep p.ringDimension p.publicColumns p.digitBound
            p.oneEncodingBound bound)
          (2 * p.oneEncodingBound)) →
      EveryNoiseBounded (states (layer + 1))
        (gateStep p.ringDimension p.publicColumns p.digitBound p.oneEncodingBound
          ((List.range layer).foldl
            (fun bound _ ↦ gateStep p.ringDimension p.publicColumns p.digitBound
              p.oneEncodingBound bound)
            (2 * p.oneEncodingBound))))
    (circuitErrorEq : circuitError = (states p.depth).get circuitIndex)
    (circuitShape : Mxx.Toolkit.MatrixShape circuitError p.modulus p.ringDimension
      1 p.publicColumns)
    (decoderShape : Mxx.Toolkit.MatrixShape decoderError p.modulus p.ringDimension 1 1)
    (kShape : Mxx.Toolkit.MatrixShape kError p.modulus p.ringDimension 1 1)
    (rShape : Mxx.Toolkit.MatrixShape rDecomposed p.modulus p.ringDimension
      p.publicColumns 1)
    (decoderNorm : Mxx.maxCenteredCoefficientNorm decoderError ≤ p.oneEncodingBound)
    (kNorm : Mxx.maxCenteredCoefficientNorm kError ≤ p.oneEncodingBound)
    (rNorm : Mxx.maxCenteredCoefficientNorm rDecomposed ≤ p.digitBound)
    (residualEq :
      residual = finalDecoderNoise decoderError kError oneError circuitError rDecomposed) :
    Mxx.Toolkit.MatrixShape residual p.modulus p.ringDimension 1 1 ∧
      Mxx.maxCenteredCoefficientNorm residual ≤ p.finalBound := by
  have inputNorms := inputTrace.norms initialSignalNorm initialErrorNorm
  have finalStateNorm :
      Mxx.maxCenteredCoefficientNorm finalStateError ≤ p.inputStateBound := by
    simpa [DiamondWeParameters.inputStateBound, injectionBound,
      DiamondWeParameters.stateRows] using inputNorms.2
  have projectedShape : Mxx.Toolkit.MatrixShape
      (projectedStateNoise finalStateError onePreimage) p.modulus p.ringDimension
      1 p.publicColumns :=
    Mxx.Toolkit.matrixMul_shape _ _ finalStateShape onePreimageShape
  have oneShape : Mxx.Toolkit.MatrixShape oneError p.modulus p.ringDimension
      1 p.publicColumns := by
    simpa [oneErrorEq] using projectedShape
  have oneNorm : Mxx.maxCenteredCoefficientNorm oneError ≤ p.oneEncodingBound := by
    rw [oneErrorEq]
    simpa [DiamondWeParameters.oneEncodingBound] using
      projectedStateNoise_norm_le p.modulus p.ringDimension p.stateColumns p.publicColumns
        p.inputStateBound p.preimageMaxCoefficientBound finalStateError onePreimage
        finalStateShape onePreimageShape finalStateNorm onePreimageNorm
  have circuitNorms := dynamicBooleanLayers_noise_le_circuitBound p.ringDimension
    p.publicColumns p.depth p.digitBound p.oneEncodingBound states statesInitial statesStep
  have circuitNorm :
      Mxx.maxCenteredCoefficientNorm circuitError ≤ p.circuitNoiseBound := by
    simpa [circuitErrorEq, DiamondWeParameters.circuitNoiseBound] using
      circuitNorms circuitIndex
  have differenceShape := Mxx.Toolkit.matrixSubtract_shape oneError circuitError
    oneShape circuitShape
  have projectedDifferenceShape := Mxx.Toolkit.matrixMul_shape
    (Mxx.matrixSubtract oneError circuitError) rDecomposed differenceShape rShape
  have sumShape := Mxx.Toolkit.matrixAdd_shape kError
    (Mxx.matrixMul (Mxx.matrixSubtract oneError circuitError) rDecomposed)
    kShape projectedDifferenceShape
  have finalShape := Mxx.Toolkit.matrixSubtract_shape decoderError
    (Mxx.matrixAdd kError
      (Mxx.matrixMul (Mxx.matrixSubtract oneError circuitError) rDecomposed))
    decoderShape sumShape
  constructor
  · simpa [residualEq, finalDecoderNoise] using finalShape
  · rw [residualEq]
    exact finalDecoderNoise_norm_le_finalBound p decoderError kError oneError circuitError
      rDecomposed decoderShape kShape oneShape circuitShape rShape decoderNorm kNorm oneNorm
      circuitNorm rNorm

private theorem ResidualRecurrenceResult.bound
    (p : DiamondWeParameters) [NeZero p.modulus]
    (result : ResidualRecurrenceResult p) :
    Mxx.Toolkit.MatrixShape result.residual p.modulus p.ringDimension 1 1 ∧
      Mxx.maxCenteredCoefficientNorm result.residual ≤ p.finalBound :=
  residual_bound_of_recurrences p result.initialSignal result.initialError result.finalSignal
    result.finalStateError result.onePreimage result.oneError result.states result.circuitError
    result.circuitIndex result.decoderError result.kError result.rDecomposed result.residual
    result.inputTrace result.initialSignalNorm result.initialErrorNorm result.finalStateShape
    result.onePreimageShape result.onePreimageNorm result.oneErrorEq result.statesInitial
    result.statesStep result.circuitErrorEq result.circuitShape result.decoderShape result.kShape
    result.rShape result.decoderNorm result.kNorm result.rNorm result.residualEq

/-- The executable entrywise addition has the expected first-coefficient equality modulo `q`.
This small bridge lets the algebraic signal-plus-error identity feed the concrete decoder without
assuming a second representation of the decoded coefficient. -/
theorem matrixAdd_headD_zmod_eq (q : Nat) [NeZero q] (left right : Mxx.Matrix)
    (leftModulus : left.modulus = q) :
    ((Mxx.matrixAdd left right).coefficients.headD 0 : ZMod q) =
      (left.coefficients.headD 0 : ZMod q) +
        (right.coefficients.headD 0 : ZMod q) := by
  rcases left with
    ⟨leftCoefficients, leftModulusValue, leftRing, leftRows, leftColumns⟩
  rcases right with
    ⟨rightCoefficients, rightModulus, rightRing, rightRows, rightColumns⟩
  simp only at leftModulus
  cases leftCoefficients <;> cases rightCoefficients <;>
    simp [Mxx.matrixAdd, Mxx.addCoefficients, Mxx.Toolkit.cast_reduce, leftModulus]

/-- Turn the exact negacyclic signal-plus-error matrix identity into the coefficient congruence
consumed by `decoded_eq_message_of_checker_bound`.  The centered representative denotes the same
class as the raw residual coefficient, so this step adds no magnitude assumption. -/
theorem message_congruence_of_matrixValue_eq
    (q n : Nat) [Fact (1 < q)] [NeZero q] [NeZero n]
    (message : Bool) (noisy signal residual : Mxx.Matrix)
    (signalShape : Mxx.Toolkit.MatrixShape signal q n 1 1)
    (valueEq : Mxx.Toolkit.matrixValue q n 1 1 noisy =
      Mxx.Toolkit.matrixValue q n 1 1 (Mxx.matrixAdd signal residual))
    (signalHead : (signal.coefficients.headD 0 : ZMod q) =
      ((if message then (q : Int) / 2 else 0 : Int) : ZMod q)) :
    (noisy.coefficients.headD 0 : ZMod q) =
      (((if message then (q : Int) / 2 else 0) +
        Mxx.centeredCoefficient q (residual.coefficients.headD 0) : Int) : ZMod q) := by
  have headEq := Mxx.Toolkit.matrixValue_headD_zmod_eq q n noisy
    (Mxx.matrixAdd signal residual) valueEq
  rw [matrixAdd_headD_zmod_eq q signal residual signalShape.modulus, signalHead] at headEq
  calc
    (noisy.coefficients.headD 0 : ZMod q) =
        ((if message then (q : Int) / 2 else 0 : Int) : ZMod q) +
          (residual.coefficients.headD 0 : ZMod q) := headEq
    _ = (((if message then (q : Int) / 2 else 0) +
        Mxx.centeredCoefficient q (residual.coefficients.headD 0) : Int) : ZMod q) := by
      rw [Mxx.Toolkit.centeredCoefficient_eq_valMinAbs]
      simp

private theorem roundDiv_two_eq_ceilHalf (value : Int) :
    Mxx.Ir.roundDiv value 2 = (value + 1) / 2 := by
  unfold Mxx.Ir.roundDiv
  norm_num
  have divided := Int.mul_ediv_mul_of_pos (a := 2) (value + 1) 2 (by norm_num)
  convert divided using 1
  all_goals ring_nf

private theorem ceilHalf_add_floorHalf (q : Nat) :
    ((q : Int) + 1) / 2 + (q : Int) / 2 = q := by
  omega

private theorem neg_roundDiv_two_eq_half_sub (q : Nat) :
    -Mxx.Ir.roundDiv (q : Int) 2 = (q : Int) / 2 - q := by
  rw [roundDiv_two_eq_ceilHalf]
  have halves := ceilHalf_add_floorHalf q
  omega

/-- The executable decoder subtracts the `ceil(q / 2)` K-message term.  Its negation is the same
residue as `floor(q / 2)` for every positive modulus, including odd moduli. -/
private theorem neg_roundDiv_two_zmod_eq_half (q : Nat) [NeZero q] :
    ((-Mxx.Ir.roundDiv (q : Int) 2 : Int) : ZMod q) =
      (((q : Int) / 2 : Int) : ZMod q) := by
  rw [neg_roundDiv_two_eq_half_sub]
  push_cast
  simp

/-- Algebraic output of the input-injection and Boolean bridges at the decoder boundary. -/
private structure SignalAlgebraResult (p : DiamondWeParameters) (message : Bool)
    (noisy residual : Mxx.Matrix) where
  signal : Mxx.Matrix
  noisyShape : Mxx.Toolkit.MatrixShape noisy p.modulus p.ringDimension 1 1
  signalShape : Mxx.Toolkit.MatrixShape signal p.modulus p.ringDimension 1 1
  valueEq : Mxx.Toolkit.matrixValue p.modulus p.ringDimension 1 1 noisy =
    Mxx.Toolkit.matrixValue p.modulus p.ringDimension 1 1
      (Mxx.matrixAdd signal residual)
  signalHead : (signal.coefficients.headD 0 : ZMod p.modulus) =
    ((if message then -Mxx.Ir.roundDiv (p.modulus : Int) 2 else 0 : Int) :
      ZMod p.modulus)

/-- Closed internal result expected from the execution bridges.  This packages the exact decoder
path, the input/Boolean residual recurrence, and its signal identity without turning any of those
implementation details into premises of the public correctness theorem. -/
private structure ClosedDiamondWeExecution
    (samplers : MxxSamplerFamily) (p : DiamondWeFamilyParams)
    (x : DiamondWeFamilyInputs p) (output : Mxx.Ir.Environment) where
  stage : Mxx.Ir.Stage
  stageInputs : Mxx.Ir.Environment
  bundle : MxxWe.Certificate.DecoderStageExecutionBundle DiamondWeFamily_workflow
    DiamondWeFamily_certificate samplers stage (DiamondWeFamilyParamEnvironment p)
    stageInputs output
  semantic : MxxWe.Certificate.DecoderSemanticOutcome bundle
  recurrence : ResidualRecurrenceResult (genericParameters p)
  algebra : SignalAlgebraResult (genericParameters p) x.diamondMessage semantic.noisy
    recurrence.residual

private theorem matrixSubtract_headD_canonical
    (q : Nat) [NeZero q] (left right : Mxx.Matrix) (leftModulus : left.modulus = q) :
    (Mxx.matrixSubtract left right).coefficients.headD 0 =
      Mxx.reduceCoefficient q ((Mxx.matrixSubtract left right).coefficients.headD 0) := by
  rcases left with ⟨leftCoefficients, leftModulusValue, leftRing, leftRows, leftColumns⟩
  rcases right with ⟨rightCoefficients, rightModulus, rightRing, rightRows, rightColumns⟩
  simp only at leftModulus
  cases leftCoefficients <;> cases rightCoefficients <;>
    simp [Mxx.matrixSubtract, Mxx.subtractCoefficients, Mxx.reduceCoefficient,
      leftModulus, NeZero.ne q]

/-- Final internal composition point.  Its two bridge results contain algebraic traces and
matrix identities, while all SSA lookups and scalar decoder semantics stay encapsulated in
`DecoderSemanticOutcome`. -/
private theorem failureBoolSafe_of_decoderSemanticOutcome
    {samplers : MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {inputs output : Mxx.Ir.Environment}
    (p : DiamondWeFamilyParams) (x : DiamondWeFamilyInputs p)
    (accepted : diamondWeFamilyChecker p = true)
    (valid : DiamondWeFamilyParamsValid p = true)
    (bundle : MxxWe.Certificate.DecoderStageExecutionBundle DiamondWeFamily_workflow
      DiamondWeFamily_certificate samplers stage (DiamondWeFamilyParamEnvironment p) inputs
      output)
    (semantic : MxxWe.Certificate.DecoderSemanticOutcome bundle)
    (recurrence : ResidualRecurrenceResult (genericParameters p))
    (algebra : SignalAlgebraResult (genericParameters p) x.diamondMessage semantic.noisy
      recurrence.residual) :
    DiamondWeFamilyFailureBool p x output = false := by
  let generic := genericParameters p
  have genericAccepted := diamondWeFamilyChecker_genericAccepted p accepted
  have modulusGe := diamondWeChecker_modulus_ge generic genericAccepted
  have modulusPositive : 0 < generic.modulus := lt_of_lt_of_le (by omega) modulusGe
  letI : NeZero generic.modulus := ⟨modulusPositive.ne'⟩
  letI : Fact (1 < generic.modulus) := ⟨lt_of_lt_of_le (by omega) modulusGe⟩
  have ringDimensionPositive : 0 < generic.ringDimension := by
    have sourcePositive : 0 < p.diamondRingDimension := by
      unfold DiamondWeFamilyParamsValid at valid
      simp only [Bool.and_eq_true, decide_eq_true_eq] at valid
      have sourcePositiveInt : (0 : Int) < Int.ofNat p.diamondRingDimension := by aesop
      simpa using sourcePositiveInt
    simpa [generic, genericParameters] using sourcePositive
  letI : NeZero generic.ringDimension := ⟨ringDimensionPositive.ne'⟩
  have residualFacts := recurrence.bound generic
  have familyModulus : p.diamondModulus = semantic.modulus := by
    simpa [DiamondWeFamilyParamEnvironment, Mxx.Ir.IntExpr.evaluate,
      Mxx.Ir.lookupParam] using semantic.modulusEvaluate
  have semanticModulus : semantic.modulus = Int.ofNat generic.modulus := by
    calc
      semantic.modulus = p.diamondModulus := familyModulus.symm
      _ = Int.ofNat (genericParameters p).modulus :=
        (genericParameters_modulus p valid).symm
      _ = Int.ofNat generic.modulus := rfl
  have noisyEqSubtract : semantic.noisy = Mxx.matrixSubtract
      semantic.matrixExecution.matrixOutcome.decoderVector
      semantic.matrixExecution.matrixOutcome.kPlusProjection := by
    rw [semantic.noisyEq, semantic.residualEq,
      semantic.matrixExecution.matrixOutcome.residualEq]
  have decoderModulus :
      semantic.matrixExecution.matrixOutcome.decoderVector.modulus = generic.modulus := by
    have noisyModulus := algebra.noisyShape.modulus
    rw [noisyEqSubtract] at noisyModulus
    simpa [Mxx.matrixSubtract] using noisyModulus
  have noisyCanonical : semantic.noisy.coefficients.headD 0 =
      Mxx.reduceCoefficient generic.modulus (semantic.noisy.coefficients.headD 0) := by
    rw [noisyEqSubtract]
    exact matrixSubtract_headD_canonical generic.modulus
      semantic.matrixExecution.matrixOutcome.decoderVector
      semantic.matrixExecution.matrixOutcome.kPlusProjection decoderModulus
  have normalizedSignalHead : (algebra.signal.coefficients.headD 0 : ZMod generic.modulus) =
      ((if x.diamondMessage then (generic.modulus : Int) / 2 else 0 : Int) :
        ZMod generic.modulus) := by
    by_cases messageEq : x.diamondMessage = true
    · calc
        (algebra.signal.coefficients.headD 0 : ZMod generic.modulus) =
            ((-Mxx.Ir.roundDiv (generic.modulus : Int) 2 : Int) :
              ZMod generic.modulus) := by simpa [messageEq] using algebra.signalHead
        _ = (((generic.modulus : Int) / 2 : Int) : ZMod generic.modulus) :=
          neg_roundDiv_two_zmod_eq_half generic.modulus
        _ = ((if x.diamondMessage then (generic.modulus : Int) / 2 else 0 : Int) :
            ZMod generic.modulus) := by simp [messageEq]
    · have messageFalse : x.diamondMessage = false := by
        cases message : x.diamondMessage <;> simp_all
      simpa [messageFalse] using algebra.signalHead
  have congruent := message_congruence_of_matrixValue_eq generic.modulus
    generic.ringDimension x.diamondMessage semantic.noisy algebra.signal recurrence.residual
    algebra.signalShape algebra.valueEq normalizedSignalHead
  have decodedValue : semantic.decoded = MxxWe.decodeBooleanInterval generic.modulus
      (semantic.noisy.coefficients.headD 0) := by
    simpa [semanticModulus, semantic.coefficientEq] using semantic.decodedEq
  have decodedEq := MxxWe.Certificate.decoded_eq_message_of_checker_bound generic
    x.diamondMessage genericAccepted generic.finalBound (le_refl generic.finalBound)
    recurrence.residual semantic.noisy residualFacts.1.modulus residualFacts.2 noisyCanonical
    congruent semantic.decoded decodedValue
  apply failureBoolSafe_of_lookup p x output
  simpa [decodedEq] using semantic.exportedDecoded

/-- Once the private execution bridges have produced their closed result, pointwise workflow
safety follows without exposing any execution-level premise at the public theorem boundary. -/
private theorem failureBoolSafe_of_closedExecution
    {samplers : MxxSamplerFamily} {p : DiamondWeFamilyParams}
    {x : DiamondWeFamilyInputs p} {output : Mxx.Ir.Environment}
    (accepted : diamondWeFamilyChecker p = true)
    (valid : DiamondWeFamilyParamsValid p = true)
    (closed : ClosedDiamondWeExecution samplers p x output) :
    DiamondWeFamilyFailureBool p x output = false :=
  failureBoolSafe_of_decoderSemanticOutcome p x accepted valid closed.bundle closed.semantic
    closed.recurrence closed.algebra

/-- Pointwise safety of exact concrete workflow outcomes implies zero finite-support failure
probability. -/
private theorem failureProbability_eq_zero_of_safe_outcomes
    (samplers : MxxSamplerFamily) (p : DiamondWeFamilyParams)
    (x : DiamondWeFamilyInputs p)
    (safe : ∀ output ∈ DiamondWeFamilyConcreteOutcomes samplers p x,
      DiamondWeFamilyFailureBool p x output = false) :
    DiamondWeFamilyFailureProbability samplers p x = 0 := by
  unfold DiamondWeFamilyFailureProbability
  apply Mxx.booleanFailureProbability_eq_zero
  intro output member
  exact safe output member

/-- Exact remaining closed-bridge obligation.  Its arguments are precisely the public statement's
sampler contract and generated predicates; all execution traces, wire identities, decompositions,
and semantic outcomes are existentially produced inside the result. -/
private def ClosedDiamondWeExecutionAvailable : Prop :=
  ∀ samplers : MxxSamplerFamily, MxxBoundedSamplerContract samplers →
  ∀ p : DiamondWeFamilyParams, diamondWeFamilyChecker p = true →
    DiamondWeFamilyParamsValid p = true →
  ∀ x : DiamondWeFamilyInputs p, DiamondWeFamilyInputsWF p x →
    DiamondWeFamilyPreconditions p x →
  ∀ output ∈ DiamondWeFamilyConcreteOutcomes samplers p x,
    Nonempty (ClosedDiamondWeExecution samplers p x output)

/-- Final theorem assembly after the one private closed-bridge obligation has been discharged.
The premise is deliberately private: the eventual public `diamondWeFamily_correct` theorem must
prove it from the executable bridges rather than accept it from a user. -/
private theorem diamondWeFamily_correct_of_closedExecutionAvailable
    (closedAvailable : ClosedDiamondWeExecutionAvailable) :
    DiamondWeFamilyCorrectStatement diamondWeFamilyChecker := by
  refine {
    accepts_valid_parameters := diamondWeFamily_accepts_valid_parameters
    probability_zero := ?_
  }
  intro samplers samplerContract p accepted valid x inputsWellFormed preconditions
  apply failureProbability_eq_zero_of_safe_outcomes
  intro output member
  obtain ⟨closed⟩ := closedAvailable samplers samplerContract p accepted valid x
    inputsWellFormed preconditions output member
  exact failureBoolSafe_of_closedExecution accepted valid closed

end MxxWe.Proofs.DiamondWeFamily
