import MxxWe.Certificate.DecoderExecutionBridge
import MxxWe.Certificate.InputInjectionExecutionBridge

namespace MxxWe.Certificate

/-! Exact execution lifting for the Diamond encryption stage.

This file retains one executable root path for the producer stage.  Consequently, the sampled
initial error, the symbolic signal, the public base matrix, the resulting initial state, and all
exported artifacts below are values from one and the same `denote` member.  No certificate field
is treated as an execution hypothesis.

`RootStageExecutionPath` currently lives in `DecoderExecutionBridge`; it is generic machinery and
will move to `ExecutionBridge` once the parallel proof work using that file has frozen.
-/

private def encryptionUnitMatrixType : Mxx.Ir.MatrixTypeExpr := {
  modulus := .parameter "diamond_modulus"
  ringDimension := .parameter "diamond_ring_dimension"
  rows := .constant 1
  columns := .constant 1
}

private def encryptionInitialErrorMatrixType : Mxx.Ir.MatrixTypeExpr := {
  modulus := .parameter "diamond_modulus"
  ringDimension := .parameter "diamond_ring_dimension"
  rows := .constant 1
  columns := .add (.constant 4)
    (.multiply (.constant 2) (.parameter "diamond_digit_count"))
}

/-- Concrete parameter values recovered while inverting the producer execution.  The public
bridge constructs this record from the checked sampler and parallel-loop nodes; it is not a
caller-provided collection of evaluation assumptions. -/
structure EncryptionParameterOutcome (params : Mxx.Ir.ParamEnvironment) where
  modulus : Nat
  ringDimension : Nat
  digitCount : Nat
  digitBase : Nat
  gadgetBase : Nat
  inputCount : Nat
  batchBits : Nat
  stateColumns : Nat
  stateWidth : Nat
  transitionStride : Nat
  errorBound : Nat
  preimageBound : Nat
  modulusEvaluate : (.parameter "diamond_modulus" : Mxx.Ir.IntExpr).evaluate params =
    some (Int.ofNat modulus)
  ringDimensionEvaluate :
    (.parameter "diamond_ring_dimension" : Mxx.Ir.IntExpr).evaluate params =
      some (Int.ofNat ringDimension)
  digitCountEvaluate : (.parameter "diamond_digit_count" : Mxx.Ir.IntExpr).evaluate params =
    some (Int.ofNat digitCount)
  digitBaseEvaluate : (.parameter "diamond_digit_base" : Mxx.Ir.IntExpr).evaluate params =
    some (Int.ofNat digitBase)
  gadgetBaseEvaluate : (.parameter "diamond_gadget_base" : Mxx.Ir.IntExpr).evaluate params =
    some (Int.ofNat gadgetBase)
  inputCountEvaluate : (.parameter "diamond_input_count" : Mxx.Ir.IntExpr).evaluate params =
    some (Int.ofNat inputCount)
  batchBitsEvaluate : (.parameter "diamond_batch_bits" : Mxx.Ir.IntExpr).evaluate params =
    some (Int.ofNat batchBits)
  errorBoundEvaluate :
    (.parameter "diamond_error_max_coefficient_bound" : Mxx.Ir.IntExpr).evaluate params =
      some (Int.ofNat errorBound)
  preimageBoundEvaluate :
    (.parameter "diamond_preimage_max_coefficient_bound" : Mxx.Ir.IntExpr).evaluate params =
      some (Int.ofNat preimageBound)
  stateColumnsEq : stateColumns = 4 + 2 * digitCount
  stateWidthEq : stateWidth = 1 + batchBits * inputCount
  transitionStrideEq : transitionStride = batchBits * digitBase * inputCount + digitBase

/-- Typed external values needed before the encryption Boolean circuit is evaluated.  A generated
family specialization constructs this record from its concrete parameter and input environments;
the generic execution theorem keeps it explicit because malformed IR environments evaluate to an
`invalid` value rather than having empty support. -/
structure EncryptionPreprocessingInputsWellFormed
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment) where
  parameters : EncryptionParameterOutcome params
  hashKey : ByteArray
  message : Bool
  hashKeyLookup : Mxx.Ir.lookupEnvironment "diamond-hash-key" inputs = some (.bytes hashKey)
  messageLookup : Mxx.Ir.lookupEnvironment "diamond-message" inputs = some (.boolean message)

private theorem verifyOperationKind_operation
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    {accept : Mxx.Ir.NodeKind → Bool}
    (verified : verifyOperationKind workflow reference accept = true) :
  verifyOperation workflow reference = true := by
  unfold verifyOperationKind at verified
  simp only [Bool.and_eq_true] at verified
  exact verified.1

private theorem verifySelectOperation_operation
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    (verified : verifySelectOperation workflow reference = true) :
    verifyOperation workflow reference = true := by
  unfold verifySelectOperation at verified
  simp only [Bool.and_eq_true] at verified
  exact verifyOperationKind_operation verified.1.1

private theorem checkedRootEncryptionOperation
    {workflowLayout : DiamondWorkflowLayout} {operations : List CoreNodeRef}
    (checked : operations.all (fun operation =>
      decide (operation.stage = workflowLayout.encryption.stage) &&
        decide (operation.scope = .root)) = true)
    (operation : CoreNodeRef) (member : operation ∈ operations) :
    operation.stage = workflowLayout.encryption.stage ∧ operation.scope = .root := by
  simp only [List.all_eq_true] at checked
  have operationChecked := checked operation member
  simpa [Bool.and_eq_true, decide_eq_true_eq] using operationChecked

/-- The concrete producer stage recovered from the checked workflow interface. -/
structure VerifiedEncryptionStage
    (workflow : Mxx.Ir.Workflow) (certificate : DiamondCertificate) where
  stage : Mxx.Ir.Stage
  resolved : resolveStage workflow certificate.workflow.encryption.stage = some stage
  outputNamesUnique : (stage.program.root.outputs.map Prod.fst).Nodup
  outputNamesEq : stage.program.root.outputs.map Prod.fst = [
    "diamond_decoder_preimage", "diamond_initial_state", "diamond_k_preimage",
    "diamond_one_preimage", "diamond_public_keys", "diamond_r_decomposed",
    "diamond_transitions", "diamond_witness_preimages"]
  outputsEq : certificate.workflow.encryption.outputs.map
    (fun output ↦ (output.name, wireRef output.wire)) = stage.program.root.outputs

/-- Workflow verification selects the encryption stage without indexing `workflow.stages`. -/
theorem VerifiedDiamondLayout.encryptionStage
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    Nonempty (VerifiedEncryptionStage workflow certificate) := by
  have workflowMatches := verified.workflowMatches
  have interfaceCheck :
      verifyStageInterface workflow certificate.workflow.encryption = true := by
    unfold verifyWorkflow at workflowMatches
    simp only [Bool.and_eq_true] at workflowMatches
    aesop
  have outputNames := verifyWorkflow_encryptOutputs verified.workflowMatches
  unfold verifyStageInterface at interfaceCheck
  cases resolved : resolveStage workflow certificate.workflow.encryption.stage with
  | none => simp [resolved] at interfaceCheck
  | some stage =>
      rw [resolved] at interfaceCheck
      simp only [Bool.and_eq_true, decide_eq_true_eq] at interfaceCheck
      have outputsEq : certificate.workflow.encryption.outputs.map
          (fun output ↦ (output.name, wireRef output.wire)) = stage.program.root.outputs := by
        aesop
      have stageNames : stage.program.root.outputs.map Prod.fst = [
          "diamond_decoder_preimage", "diamond_initial_state", "diamond_k_preimage",
          "diamond_one_preimage", "diamond_public_keys", "diamond_r_decomposed",
          "diamond_transitions", "diamond_witness_preimages"] := by
        rw [← outputsEq]
        simpa [Function.comp_def] using outputNames
      exact ⟨{
        stage
        resolved
        outputNamesUnique := by rw [stageNames]; simp
        outputNamesEq := stageNames
        outputsEq
      }⟩

private theorem encryptionRootReference_inBounds
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef} {stage : Mxx.Ir.Stage}
    (rootScope : reference.scope = .root)
    (stageResolved : resolveStage workflow reference.stage = some stage)
    (nodeResolved : ∃ node, resolveNode workflow reference = some node) :
    reference.node < stage.program.root.nodes.length := by
  obtain ⟨node, nodeResolved⟩ := nodeResolved
  rcases reference with ⟨stageName, scope, index⟩
  dsimp at rootScope
  subst scope
  have nodeAt : stage.program.root.nodes[index]? = some node := by
    simpa [resolveNode, resolveScope, scopeOwnerMatches, rawScope, stageResolved] using
      nodeResolved
  by_contra outOfBounds
  rw [List.getElem?_eq_none (Nat.le_of_not_gt outOfBounds)] at nodeAt
  contradiction

/-- Resolve one checked encryption-root operation on a retained producer path. -/
theorem RootStageExecutionPath.encryptionOperationExecution
    {workflow : Mxx.Ir.Workflow} {samplers : Mxx.MxxSamplerFamily}
    {stage : Mxx.Ir.Stage} {params : Mxx.Ir.ParamEnvironment}
    {inputs output : Mxx.Ir.Environment}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (reference : OperationRef) (rootScope : reference.operation.scope = .root)
    (stageResolved : resolveStage workflow reference.operation.stage = some stage)
    (verified : verifyOperation workflow reference = true) :
    ∃ execution : ReferencedNodeExecution workflow reference.operation
        (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
        samplers params inputs,
      RootedNodeExecution rootPath execution := by
  have resolved := verifyOperation_resolves verified
  apply rootPath.referencedRootNodeExecution reference.operation rootScope stageResolved
  exact encryptionRootReference_inBounds rootScope stageResolved
    ⟨resolved.choose, resolved.choose_spec.1⟩

private theorem verifyParallelLoop_nodeExists
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef}
    (verified : verifyParallelLoop workflow reference = true) :
    ∃ node, resolveNode workflow reference.operation = some node := by
  unfold verifyParallelLoop at verified
  simp only [Bool.and_eq_true] at verified
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [resolved] at verified
  | some node => exact ⟨node, rfl⟩

/-- Resolve any checked encryption-root node on a retained producer path. -/
private theorem RootStageExecutionPath.encryptionNodeExecution
    {workflow : Mxx.Ir.Workflow} {samplers : Mxx.MxxSamplerFamily}
    {stage : Mxx.Ir.Stage} {params : Mxx.Ir.ParamEnvironment}
    {inputs output : Mxx.Ir.Environment}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (reference : CoreNodeRef) (rootScope : reference.scope = .root)
    (stageResolved : resolveStage workflow reference.stage = some stage)
    (nodeResolved : ∃ node, resolveNode workflow reference = some node) :
    ∃ execution : ReferencedNodeExecution workflow reference
        (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
        samplers params inputs,
      RootedNodeExecution rootPath execution := by
  exact rootPath.referencedRootNodeExecution reference rootScope stageResolved
    (encryptionRootReference_inBounds rootScope stageResolved nodeResolved)

/-- Exact root input selected by the strengthened message-construction checker. -/
structure EncryptionMessageInputResolution
    (workflow : Mxx.Ir.Workflow) (workflowLayout : DiamondWorkflowLayout)
    (layout : MessageConstructionLayout) where
  input : CoreOperandRef
  inputsEq : layout.toInt.inputs = [input]
  stageEq : input.wire.node.stage = workflowLayout.encryption.stage
  scopeEq : input.wire.node.scope = .root
  portEq : input.wire.port = 0
  resolved : resolveNode workflow input.wire.node = some {
    kind := .input "diamond-message"
    arguments := []
    outputCount := 1
  }

theorem encryptionMessageInputResolution_of_verified
    {workflow : Mxx.Ir.Workflow} {workflowLayout : DiamondWorkflowLayout}
    {layout : MessageConstructionLayout}
    (verified : verifyMessageConstruction workflow workflowLayout layout = true) :
    Nonempty (EncryptionMessageInputResolution workflow workflowLayout layout) := by
  obtain ⟨input, inputsEq, inputVerified⟩ := verifyMessageConstruction_messageInput verified
  unfold verifyInputWire at inputVerified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at inputVerified
  cases resolved : resolveNode workflow input.wire.node with
  | none => simp [resolved] at inputVerified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> simp_all
      rename_i name
      exact ⟨{
        input
        inputsEq
        stageEq := by aesop
        scopeEq := by aesop
        portEq := by aesop
        resolved := by simp_all
      }⟩

/-- Exact uniform `{-1, 0, 1}` sampler node recovered from a checked operation. -/
structure UniformMinusOneOneResolution
    (workflow : Mxx.Ir.Workflow) (reference : OperationRef) where
  matrixType : Mxx.Ir.MatrixTypeExpr
  resolved : resolveNode workflow reference.operation = some {
    kind := .uniformSample matrixType (.constant (-1)) (.constant 1)
    arguments := reference.inputs.map (wireRef ∘ CoreOperandRef.wire)
    outputCount := reference.outputs.length
  }

theorem verifyUniformMinusOneOne_resolution
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    (verified : verifyOperationKind workflow reference (fun kind ↦ match kind with
      | .uniformSample _ (.constant (-1)) (.constant 1) => true
      | _ => false) = true) :
    Nonempty (UniformMinusOneOneResolution workflow reference) := by
  let accept : Mxx.Ir.NodeKind → Bool := fun kind ↦ match kind with
    | .uniformSample _ (.constant (-1)) (.constant 1) => true
    | _ => false
  have acceptExact : ∀ kind, accept kind = true →
      ∃ matrixType, kind =
        .uniformSample matrixType (.constant (-1)) (.constant 1) := by
    intro kind accepted
    grind
  change verifyOperationKind workflow reference accept = true at verified
  have operationVerified := verifyOperationKind_operation verified
  unfold verifyOperationKind at verified
  simp only [Bool.and_eq_true] at verified
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [verifyOperation, resolved] at verified
  | some node =>
      simp only [resolved] at verified
      have accepted : accept node.kind = true := verified.2
      obtain ⟨matrixType, kindEq⟩ := acceptExact node.kind accepted
      have operationFacts := verifyOperation_resolves operationVerified
      obtain ⟨actual, actualResolved, argumentsEq, outputsEq⟩ := operationFacts
      rw [resolved] at actualResolved
      have nodeEq : node = actual := Option.some.inj actualResolved
      subst actual
      exact ⟨{
        matrixType
        resolved := by
          rcases node with ⟨kind, arguments, outputCount⟩
          simp only at kindEq argumentsEq outputsEq
          subst kind
          subst arguments
          subst outputCount
          exact resolved
      }⟩

/-- Concrete uniform sample with the deterministic norm-one support fact. -/
structure UniformMinusOneOneOutcome
    (workflow : Mxx.Ir.Workflow) (reference : OperationRef)
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs) where
  matrixParams : Mxx.SamplerParams
  sample : Mxx.Matrix
  support : sample ∈ Mxx.Ir.uniformMatrixSupport matrixParams (-1) 1
  valuesEq : execution.values = [.matrix sample]
  shape : Mxx.Toolkit.MatrixShape sample matrixParams.modulus matrixParams.ringDimension
    matrixParams.rows matrixParams.columns
  norm : Mxx.maxCenteredCoefficientNorm sample ≤ 1

theorem uniformMinusOneOneOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : UniformMinusOneOneResolution workflow reference)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (inputsEmpty : reference.inputs = []) (outputsOne : reference.outputs.length = 1)
    (matrixParams : Mxx.SamplerParams)
    (matrixTypeEvaluate : resolution.matrixType.evaluate params = some matrixParams)
    (modulusGe : 2 ≤ matrixParams.modulus) :
    Nonempty (UniformMinusOneOneOutcome workflow reference runChild samplers params inputs
      execution) := by
  have executionResolved := execution.resolved
  have nodeResolved := resolution.resolved
  rw [executionResolved] at nodeResolved
  have nodeEq := Option.some.inj nodeResolved
  have member : execution.values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs
      execution.before {
        kind := .uniformSample resolution.matrixType (.constant (-1)) (.constant 1)
        arguments := []
        outputCount := 1
      } := by
    simpa [nodeEq, inputsEmpty, outputsOne] using execution.member
  obtain ⟨sample, sampleMember, valuesEq⟩ :=
    Mxx.Ir.mem_evaluateNode_uniformSample runChild samplers params inputs execution.before
      resolution.matrixType (.constant (-1)) (.constant 1) matrixParams (-1) 1 1
      matrixTypeEvaluate (by rfl) (by rfl) member
  exact ⟨{
    matrixParams
    sample
    support := sampleMember
    valuesEq
    shape :=
      (Mxx.Toolkit.uniformMatrixSupport_layout matrixParams (-1) 1 sample sampleMember).toMatrixShape
    norm := Mxx.Toolkit.uniformMatrixSupport_minusOneOne_norm_le matrixParams modulusGe sample
      sampleMember
  }⟩

/-- Values from the initial-state construction, all retained on one producer SSA path. -/
structure EncryptionInitialStateExecutions
    (workflow : Mxx.Ir.Workflow) (certificate : DiamondCertificate)
    (samplers : Mxx.MxxSamplerFamily) (stage : Mxx.Ir.Stage)
    (params : Mxx.Ir.ParamEnvironment) (inputs output : Mxx.Ir.Environment) where
  rootPath : RootStageExecutionPath samplers stage params inputs output
  secret : ReferencedNodeExecution workflow certificate.inputPreprocessing.secretSample.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  secretRooted : RootedNodeExecution rootPath secret
  messageInputRef : CoreOperandRef
  messageInputsEq : certificate.message.toInt.inputs = [messageInputRef]
  messageInput : ReferencedNodeExecution workflow messageInputRef.wire.node
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  messageInputRooted : RootedNodeExecution rootPath messageInput
  messageToInt : ReferencedNodeExecution workflow certificate.message.toInt.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  messageToIntRooted : RootedNodeExecution rootPath messageToInt
  messageZero : ReferencedNodeExecution workflow certificate.message.zero.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  messageZeroRooted : RootedNodeExecution rootPath messageZero
  messageOne : ReferencedNodeExecution workflow certificate.message.one.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  messageOneRooted : RootedNodeExecution rootPath messageOne
  messageSelect : ReferencedNodeExecution workflow certificate.message.select.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  messageSelectRooted : RootedNodeExecution rootPath messageSelect
  signal : ReferencedNodeExecution workflow certificate.inputPreprocessing.messageSelector.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  signalRooted : RootedNodeExecution rootPath signal
  initialError : ReferencedNodeExecution
    workflow certificate.inputPreprocessing.initialErrorSample.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  initialErrorRooted : RootedNodeExecution rootPath initialError
  initialProduct : ReferencedNodeExecution
    workflow certificate.inputPreprocessing.initialPublicProduct.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  initialProductRooted : RootedNodeExecution rootPath initialProduct
  initialState : ReferencedNodeExecution workflow certificate.inputPreprocessing.initialState.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  initialStateRooted : RootedNodeExecution rootPath initialState

/-- Recover the complete initial-state producer slice from one supplied encryption root path.
Every node is inverted from that exact path; the theorem has no per-node execution premise. -/
theorem encryptionInitialStateExecutions_of_rootPath
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved : resolveStage workflow certificate.workflow.encryption.stage = some stage)
    (rootPath : RootStageExecutionPath samplers stage params inputs output) :
    ∃ executions : EncryptionInitialStateExecutions workflow certificate samplers stage params
        inputs output,
      executions.rootPath = rootPath := by
  have messageChecked := verified.messageMatches
  have preprocessingChecked := verified.inputPreprocessingMatches
  obtain ⟨messageInput, messageInputsEq, messageInputChecked⟩ :=
    verifyMessageConstruction_messageInput messageChecked
  unfold verifyMessageConstruction at messageChecked
  simp only [Bool.and_eq_true] at messageChecked
  unfold verifyInputPreprocessing at preprocessingChecked
  simp only [Bool.and_eq_true] at preprocessingChecked
  have messageLocations :
      [certificate.message.toInt.operation, certificate.message.zero.operation,
        certificate.message.one.operation, certificate.message.select.operation].all
        (fun operation =>
          decide (operation.stage = certificate.workflow.encryption.stage) &&
            decide (operation.scope = .root)) = true := by tauto
  have messageToIntKind : verifyOperationKind workflow certificate.message.toInt
      (fun kind => match kind with | .boolToInt => true | _ => false) = true := by tauto
  have messageZeroKind : verifyOperationKind workflow certificate.message.zero
      (fun kind => match kind with
        | .zeroMatrix actualType => decide (actualType = encryptionUnitMatrixType)
        | _ => false) = true := by tauto
  have messageOneKind : verifyOperationKind workflow certificate.message.one
      (fun kind => match kind with
        | .identityMatrix actualType => decide (actualType = encryptionUnitMatrixType)
        | _ => false) = true := by tauto
  have messageSelectCheck : verifySelectOperation workflow certificate.message.select = true := by
    tauto
  have preprocessingLocations :
      [certificate.inputPreprocessing.trapdoorSamples.parallelLoop.operation,
        certificate.inputPreprocessing.secretSample.operation,
        certificate.inputPreprocessing.messageSelector.operation,
        certificate.inputPreprocessing.initialErrorSample.operation,
        certificate.inputPreprocessing.initialPublicProduct.operation,
        certificate.inputPreprocessing.initialState.operation,
        certificate.inputPreprocessing.transitionSourceIndices.parallelLoop.operation,
        certificate.inputPreprocessing.transitionTargetIndices.parallelLoop.operation,
        certificate.inputPreprocessing.digitSecretIndices.parallelLoop.operation,
        certificate.inputPreprocessing.digitSecretSamples.parallelLoop.operation,
        certificate.inputPreprocessing.digitSecrets.parallelLoop.operation,
        certificate.inputPreprocessing.transitionSources.parallelLoop.operation,
        certificate.inputPreprocessing.targetPublicMatrices.parallelLoop.operation,
        certificate.inputPreprocessing.transitionTargets.parallelLoop.operation,
        certificate.inputPreprocessing.transitionPreimages.parallelLoop.operation,
        certificate.inputPreprocessing.finalIndices.operation,
        certificate.inputPreprocessing.finalTrapdoors.parallelLoop.operation].all
        (fun operation =>
          decide (operation.stage = certificate.workflow.encryption.stage) &&
            decide (operation.scope = .root)) = true := by tauto
  have secretKind : verifyOperationKind workflow certificate.inputPreprocessing.secretSample
      (fun kind => match kind with
        | .uniformSample matrixType (.constant (-1)) (.constant 1) =>
            decide (matrixType = encryptionUnitMatrixType)
        | _ => false) = true := by tauto
  have signalKind : verifyOperationKind workflow certificate.inputPreprocessing.messageSelector
      (fun kind => match kind with | .concat .columns => true | _ => false) = true := by tauto
  have errorKind : verifyOperationKind workflow certificate.inputPreprocessing.initialErrorSample
      (fun kind => match kind with
        | .gaussianSample matrixType cutoff =>
            decide (matrixType = encryptionInitialErrorMatrixType) &&
              decide (cutoff = .parameter "diamond_error_max_coefficient_bound")
        | _ => false) = true := by tauto
  have productKind : verifyOperationKind workflow
      certificate.inputPreprocessing.initialPublicProduct
      (fun kind => match kind with | .matrixMultiply => true | _ => false) = true := by tauto
  have stateKind : verifyOperationKind workflow certificate.inputPreprocessing.initialState
      (fun kind => match kind with | .matrixAdd => true | _ => false) = true := by tauto
  have secretVerified : verifyOperation workflow
      certificate.inputPreprocessing.secretSample = true :=
    verifyOperationKind_operation secretKind
  have toIntVerified : verifyOperation workflow certificate.message.toInt = true :=
    verifyOperationKind_operation messageToIntKind
  have zeroVerified : verifyOperation workflow certificate.message.zero = true :=
    verifyOperationKind_operation messageZeroKind
  have oneVerified : verifyOperation workflow certificate.message.one = true :=
    verifyOperationKind_operation messageOneKind
  have selectVerified : verifyOperation workflow certificate.message.select = true :=
    verifySelectOperation_operation messageSelectCheck
  have signalVerified : verifyOperation workflow
      certificate.inputPreprocessing.messageSelector = true :=
    verifyOperationKind_operation signalKind
  have errorVerified : verifyOperation workflow
      certificate.inputPreprocessing.initialErrorSample = true :=
    verifyOperationKind_operation errorKind
  have productVerified : verifyOperation workflow
      certificate.inputPreprocessing.initialPublicProduct = true :=
    verifyOperationKind_operation productKind
  have stateVerified : verifyOperation workflow
      certificate.inputPreprocessing.initialState = true :=
    verifyOperationKind_operation stateKind
  have messageStage : messageInput.wire.node.stage =
      certificate.workflow.encryption.stage := by
    unfold verifyInputWire at messageInputChecked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at messageInputChecked
    aesop
  have messageScope : messageInput.wire.node.scope = .root := by
    unfold verifyInputWire at messageInputChecked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at messageInputChecked
    aesop
  have messageResolved : ∃ node, resolveNode workflow messageInput.wire.node = some node := by
    unfold verifyInputWire at messageInputChecked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at messageInputChecked
    cases resolved : resolveNode workflow messageInput.wire.node with
    | none => simp [resolved] at messageInputChecked
    | some node => exact ⟨node, rfl⟩
  have toIntLocation := checkedRootEncryptionOperation messageLocations
    certificate.message.toInt.operation (by simp)
  have zeroLocation := checkedRootEncryptionOperation messageLocations
    certificate.message.zero.operation (by simp)
  have oneLocation := checkedRootEncryptionOperation messageLocations
    certificate.message.one.operation (by simp)
  have selectLocation := checkedRootEncryptionOperation messageLocations
    certificate.message.select.operation (by simp)
  have secretLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.secretSample.operation (by simp)
  have signalLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.messageSelector.operation (by simp)
  have errorLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.initialErrorSample.operation (by simp)
  have productLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.initialPublicProduct.operation (by simp)
  have stateLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.initialState.operation (by simp)
  have messageInputStageResolved : resolveStage workflow messageInput.wire.node.stage =
      some stage := by simpa [messageStage] using stageResolved
  obtain ⟨messageInputExecution, messageInputRooted⟩ :=
    rootPath.referencedRootNodeExecution messageInput.wire.node messageScope
      messageInputStageResolved
      (encryptionRootReference_inBounds messageScope messageInputStageResolved messageResolved)
  obtain ⟨secret, secretRooted⟩ := rootPath.encryptionOperationExecution
    certificate.inputPreprocessing.secretSample secretLocation.2
    (by simpa [secretLocation.1] using stageResolved) secretVerified
  obtain ⟨messageToInt, messageToIntRooted⟩ := rootPath.encryptionOperationExecution
    certificate.message.toInt toIntLocation.2
    (by simpa [toIntLocation.1] using stageResolved) toIntVerified
  obtain ⟨messageZero, messageZeroRooted⟩ := rootPath.encryptionOperationExecution
    certificate.message.zero zeroLocation.2
    (by simpa [zeroLocation.1] using stageResolved) zeroVerified
  obtain ⟨messageOne, messageOneRooted⟩ := rootPath.encryptionOperationExecution
    certificate.message.one oneLocation.2
    (by simpa [oneLocation.1] using stageResolved) oneVerified
  obtain ⟨messageSelect, messageSelectRooted⟩ := rootPath.encryptionOperationExecution
    certificate.message.select selectLocation.2
    (by simpa [selectLocation.1] using stageResolved) selectVerified
  obtain ⟨signal, signalRooted⟩ := rootPath.encryptionOperationExecution
    certificate.inputPreprocessing.messageSelector signalLocation.2
    (by simpa [signalLocation.1] using stageResolved) signalVerified
  obtain ⟨initialError, initialErrorRooted⟩ := rootPath.encryptionOperationExecution
    certificate.inputPreprocessing.initialErrorSample errorLocation.2
    (by simpa [errorLocation.1] using stageResolved) errorVerified
  obtain ⟨initialProduct, initialProductRooted⟩ := rootPath.encryptionOperationExecution
    certificate.inputPreprocessing.initialPublicProduct productLocation.2
    (by simpa [productLocation.1] using stageResolved) productVerified
  obtain ⟨initialState, initialStateRooted⟩ := rootPath.encryptionOperationExecution
    certificate.inputPreprocessing.initialState stateLocation.2
    (by simpa [stateLocation.1] using stageResolved) stateVerified
  exact ⟨{
    rootPath
    secret
    secretRooted
    messageInputRef := messageInput
    messageInputsEq
    messageInput := messageInputExecution
    messageInputRooted
    messageToInt
    messageToIntRooted
    messageZero
    messageZeroRooted
    messageOne
    messageOneRooted
    messageSelect
    messageSelectRooted
    signal
    signalRooted
    initialError
    initialErrorRooted
    initialProduct
    initialProductRooted
    initialState
    initialStateRooted
  }, rfl⟩

/-- Membership wrapper for `encryptionInitialStateExecutions_of_rootPath`. -/
theorem encryptionInitialStateExecutions_of_member
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved : resolveStage workflow certificate.workflow.encryption.stage = some stage)
    (member : output ∈ Mxx.Ir.denote samplers stage.program params inputs) :
    Nonempty (EncryptionInitialStateExecutions workflow certificate samplers stage params inputs
      output) := by
  obtain ⟨rootPath⟩ := rootStageExecutionPath_of_member member
  obtain ⟨executions, _⟩ :=
    encryptionInitialStateExecutions_of_rootPath verified stageResolved rootPath
  exact ⟨executions⟩

/-- Exact message value selected by the producer graph.  This structure is deliberately tied to
the checked `diamond-message` input execution rather than accepting an unrelated Boolean. -/
structure EncryptionMessageOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (executions : EncryptionInitialStateExecutions workflow certificate samplers stage params
      inputs output) where
  message : Bool
  selected : Mxx.Matrix
  inputValues : executions.messageInput.values = [.boolean message]
  integerValues : executions.messageToInt.values = [.integer (if message then 1 else 0)]
  selectedValues : executions.messageSelect.values = [.matrix selected]

/-- Algebraic initial-state fact consumed by the input-injection bridge.  The equality is stated
through `matrixValue`, so it is independent of concrete coefficient storage and dimensions. -/
structure EncryptionInitialStateOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (executions : EncryptionInitialStateExecutions workflow certificate samplers stage params
      inputs output)
    (q ringDimension stateColumns errorBound : Nat) where
  signal : Mxx.Matrix
  base : Mxx.Matrix
  error : Mxx.Matrix
  state : Mxx.Matrix
  signalValues : executions.signal.values = [.matrix signal]
  errorValues : executions.initialError.values = [.matrix error]
  stateValues : executions.initialState.values = [.matrix state]
  signalShape : Mxx.Toolkit.MatrixShape signal q ringDimension 1 2
  baseShape : Mxx.Toolkit.MatrixShape base q ringDimension 2 stateColumns
  errorShape : Mxx.Toolkit.MatrixShape error q ringDimension 1 stateColumns
  stateShape : Mxx.Toolkit.MatrixShape state q ringDimension 1 stateColumns
  signalNorm : Mxx.maxCenteredCoefficientNorm signal ≤ 1
  errorNorm : Mxx.maxCenteredCoefficientNorm error ≤ errorBound
  stateEquation :
    Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns state =
      Mxx.Toolkit.matrixValue q ringDimension 1 2 signal *
          Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns base +
        Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns error

/-- Root executions that create every producer artifact needed by decryption.  Parallel sampler
families remain represented by their exact checked parent-loop execution; their child outcomes are
refined by `InputInjectionExecutionBridge` and the Boolean execution bridge. -/
structure EncryptionArtifactExecutions
    (workflow : Mxx.Ir.Workflow) (certificate : DiamondCertificate)
    (samplers : Mxx.MxxSamplerFamily) (stage : Mxx.Ir.Stage)
    (params : Mxx.Ir.ParamEnvironment) (inputs output : Mxx.Ir.Environment) where
  rootPath : RootStageExecutionPath samplers stage params inputs output
  trapdoorSamples : ReferencedNodeExecution
    workflow certificate.inputPreprocessing.trapdoorSamples.parallelLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  trapdoorSamplesRooted : RootedNodeExecution rootPath trapdoorSamples
  transitionSourceIndices : ReferencedNodeExecution
    workflow certificate.inputPreprocessing.transitionSourceIndices.parallelLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  transitionSourceIndicesRooted : RootedNodeExecution rootPath transitionSourceIndices
  transitionTargetIndices : ReferencedNodeExecution
    workflow certificate.inputPreprocessing.transitionTargetIndices.parallelLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  transitionTargetIndicesRooted : RootedNodeExecution rootPath transitionTargetIndices
  digitSecretIndices : ReferencedNodeExecution
    workflow certificate.inputPreprocessing.digitSecretIndices.parallelLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  digitSecretIndicesRooted : RootedNodeExecution rootPath digitSecretIndices
  digitSecretSamples : ReferencedNodeExecution
    workflow certificate.inputPreprocessing.digitSecretSamples.parallelLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  digitSecretSamplesRooted : RootedNodeExecution rootPath digitSecretSamples
  digitSecrets : ReferencedNodeExecution
    workflow certificate.inputPreprocessing.digitSecrets.parallelLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  digitSecretsRooted : RootedNodeExecution rootPath digitSecrets
  transitionSources : ReferencedNodeExecution
    workflow certificate.inputPreprocessing.transitionSources.parallelLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  transitionSourcesRooted : RootedNodeExecution rootPath transitionSources
  targetPublicMatrices : ReferencedNodeExecution
    workflow certificate.inputPreprocessing.targetPublicMatrices.parallelLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  targetPublicMatricesRooted : RootedNodeExecution rootPath targetPublicMatrices
  transitionTargets : ReferencedNodeExecution
    workflow certificate.inputPreprocessing.transitionTargets.parallelLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  transitionTargetsRooted : RootedNodeExecution rootPath transitionTargets
  packedPublicKeys : ReferencedNodeExecution
    workflow certificate.publicKeySampling.packedHash.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  packedPublicKeysRooted : RootedNodeExecution rootPath packedPublicKeys
  publicKeys : ReferencedNodeExecution
    workflow certificate.publicKeySampling.slices.parallelLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  publicKeysRooted : RootedNodeExecution rootPath publicKeys
  transitions : ReferencedNodeExecution
    workflow certificate.inputPreprocessing.transitionPreimages.parallelLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  transitionsRooted : RootedNodeExecution rootPath transitions
  finalTrapdoors : ReferencedNodeExecution
    workflow certificate.inputPreprocessing.finalTrapdoors.parallelLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  finalTrapdoorsRooted : RootedNodeExecution rootPath finalTrapdoors
  onePreimageSample : ReferencedNodeExecution
    workflow certificate.artifactPreprocessing.onePreimage.sample.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  onePreimageSampleRooted : RootedNodeExecution rootPath onePreimageSample
  onePreimage : ReferencedNodeExecution
    workflow certificate.artifactPreprocessing.onePreimage.materialize.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  onePreimageRooted : RootedNodeExecution rootPath onePreimage
  witnessPreimages : ReferencedNodeExecution
    workflow certificate.artifactPreprocessing.witnessPreimages.parallelLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  witnessPreimagesRooted : RootedNodeExecution rootPath witnessPreimages
  kPreimageSample : ReferencedNodeExecution
    workflow certificate.artifactPreprocessing.kPreimage.sample.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  kPreimageSampleRooted : RootedNodeExecution rootPath kPreimageSample
  kPreimage : ReferencedNodeExecution
    workflow certificate.artifactPreprocessing.kPreimage.materialize.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  kPreimageRooted : RootedNodeExecution rootPath kPreimage
  rDecomposition : ReferencedNodeExecution
    workflow certificate.artifactPreprocessing.rDecomposition.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  rDecompositionRooted : RootedNodeExecution rootPath rDecomposition
  rMaterialization : ReferencedNodeExecution
    workflow certificate.artifactPreprocessing.rMaterialization.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  rMaterializationRooted : RootedNodeExecution rootPath rMaterialization
  rDecomposed : ReferencedNodeExecution
    workflow certificate.artifactPreprocessing.rReshape.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  rDecomposedRooted : RootedNodeExecution rootPath rDecomposed
  decoderPreimageSample : ReferencedNodeExecution
    workflow certificate.artifactPreprocessing.decoderPreimage.sample.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  decoderPreimageSampleRooted : RootedNodeExecution rootPath decoderPreimageSample
  decoderPreimage : ReferencedNodeExecution
    workflow certificate.artifactPreprocessing.decoderPreimage.materialize.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  decoderPreimageRooted : RootedNodeExecution rootPath decoderPreimage

private structure EncryptionArtifactNodeChecks
    (workflow : Mxx.Ir.Workflow) (certificate : DiamondCertificate) : Prop where
  publicKeyLoop : verifyParallelLoop workflow
    certificate.publicKeySampling.slices.parallelLoop = true
  witnessLoop : verifyParallelLoop workflow
    certificate.artifactPreprocessing.witnessPreimages.parallelLoop = true
  packedHash : verifyOperation workflow certificate.publicKeySampling.packedHash = true
  oneSample : verifyOperation workflow certificate.artifactPreprocessing.onePreimage.sample = true
  oneMaterialize : verifyOperation workflow
    certificate.artifactPreprocessing.onePreimage.materialize = true
  kSample : verifyOperation workflow certificate.artifactPreprocessing.kPreimage.sample = true
  kMaterialize : verifyOperation workflow
    certificate.artifactPreprocessing.kPreimage.materialize = true
  rDecomposition : verifyOperation workflow certificate.artifactPreprocessing.rDecomposition = true
  rMaterialization : verifyOperation workflow
    certificate.artifactPreprocessing.rMaterialization = true
  rReshape : verifyOperation workflow certificate.artifactPreprocessing.rReshape = true
  decoderSample : verifyOperation workflow
    certificate.artifactPreprocessing.decoderPreimage.sample = true
  decoderMaterialize : verifyOperation workflow
    certificate.artifactPreprocessing.decoderPreimage.materialize = true

private theorem encryptionArtifactNodeChecks_of_verified
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    EncryptionArtifactNodeChecks workflow certificate := by
  have publicChecked := verified.publicKeySamplingMatches
  have artifactChecked := verified.artifactPreprocessingMatches
  unfold verifyPublicKeySampling at publicChecked
  unfold verifyArtifactPreprocessing at artifactChecked
  simp only [Bool.and_eq_true] at publicChecked artifactChecked
  have publicNodes := publicChecked.2
  have artifactNodes := artifactChecked.2
  clear publicChecked artifactChecked
  unfold verifyPublicKeySamplingProducerNodes at publicNodes
  unfold verifyArtifactPreprocessingProducerNodes at artifactNodes
  simp only [Bool.and_eq_true] at publicNodes artifactNodes
  refine {
    publicKeyLoop := by aesop
    witnessLoop := by aesop
    packedHash := by aesop
    oneSample := by aesop
    oneMaterialize := by aesop
    kSample := by aesop
    kMaterialize := by aesop
    rDecomposition := by aesop
    rMaterialization := by aesop
    rReshape := by aesop
    decoderSample := by aesop
    decoderMaterialize := by aesop
  }

/-- Recover every artifact-producing root execution from one supplied encryption-stage path. -/
theorem encryptionArtifactExecutions_of_rootPath
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved : resolveStage workflow certificate.workflow.encryption.stage = some stage)
    (rootPath : RootStageExecutionPath samplers stage params inputs output) :
    ∃ executions : EncryptionArtifactExecutions workflow certificate samplers stage params inputs
        output,
      executions.rootPath = rootPath := by
  have preprocessingChecked := verified.inputPreprocessingMatches
  have publicChecked := verified.publicKeySamplingMatches
  have artifactChecked := verified.artifactPreprocessingMatches
  unfold verifyInputPreprocessing at preprocessingChecked
  unfold verifyPublicKeySampling at publicChecked
  unfold verifyArtifactPreprocessing at artifactChecked
  simp only [Bool.and_eq_true] at preprocessingChecked publicChecked artifactChecked
  unfold verifyOperationKind at publicChecked
  unfold verifyPreimage verifyOperationKind at artifactChecked
  simp only [Bool.and_eq_true] at publicChecked artifactChecked
  have preprocessingLocations :
      [certificate.inputPreprocessing.trapdoorSamples.parallelLoop.operation,
        certificate.inputPreprocessing.secretSample.operation,
        certificate.inputPreprocessing.messageSelector.operation,
        certificate.inputPreprocessing.initialErrorSample.operation,
        certificate.inputPreprocessing.initialPublicProduct.operation,
        certificate.inputPreprocessing.initialState.operation,
        certificate.inputPreprocessing.transitionSourceIndices.parallelLoop.operation,
        certificate.inputPreprocessing.transitionTargetIndices.parallelLoop.operation,
        certificate.inputPreprocessing.digitSecretIndices.parallelLoop.operation,
        certificate.inputPreprocessing.digitSecretSamples.parallelLoop.operation,
        certificate.inputPreprocessing.digitSecrets.parallelLoop.operation,
        certificate.inputPreprocessing.transitionSources.parallelLoop.operation,
        certificate.inputPreprocessing.targetPublicMatrices.parallelLoop.operation,
        certificate.inputPreprocessing.transitionTargets.parallelLoop.operation,
        certificate.inputPreprocessing.transitionPreimages.parallelLoop.operation,
        certificate.inputPreprocessing.finalIndices.operation,
        certificate.inputPreprocessing.finalTrapdoors.parallelLoop.operation].all
        (fun operation =>
          decide (operation.stage = certificate.workflow.encryption.stage) &&
            decide (operation.scope = .root)) = true := by tauto
  have publicLocations :
      [certificate.publicKeySampling.packedHash.operation,
        certificate.publicKeySampling.slices.parallelLoop.operation].all
        (fun operation =>
          decide (operation.stage = certificate.workflow.encryption.stage) &&
            decide (operation.scope = .root)) = true := by tauto
  have artifactLocations :
      [certificate.artifactPreprocessing.onePreimage.sample.operation,
        certificate.artifactPreprocessing.onePreimage.materialize.operation,
        certificate.artifactPreprocessing.witnessPreimages.parallelLoop.operation,
        certificate.artifactPreprocessing.kPreimage.sample.operation,
        certificate.artifactPreprocessing.kPreimage.materialize.operation,
        certificate.artifactPreprocessing.rDecomposition.operation,
        certificate.artifactPreprocessing.rMaterialization.operation,
        certificate.artifactPreprocessing.rReshape.operation,
        certificate.artifactPreprocessing.decoderPreimage.sample.operation,
        certificate.artifactPreprocessing.decoderPreimage.materialize.operation].all
        (fun operation =>
          decide (operation.stage = certificate.workflow.encryption.stage) &&
            decide (operation.scope = .root)) = true := by tauto
  clear preprocessingChecked publicChecked artifactChecked
  have trapdoorLoop : verifyParallelLoop workflow
      certificate.inputPreprocessing.trapdoorSamples.parallelLoop = true := by
    exact verifyInputPreprocessing_trapdoorSamplesLoop verified.inputPreprocessingMatches
  have sourceIndexLoop : verifyParallelLoop workflow
      certificate.inputPreprocessing.transitionSourceIndices.parallelLoop = true := by
    exact verified.preprocessingSourceIndexLoop
  have targetIndexLoop : verifyParallelLoop workflow
      certificate.inputPreprocessing.transitionTargetIndices.parallelLoop = true := by
    exact verified.preprocessingTargetIndexLoop
  have digitIndexLoop : verifyParallelLoop workflow
      certificate.inputPreprocessing.digitSecretIndices.parallelLoop = true := by
    exact verified.preprocessingDigitSecretIndexLoop
  have digitSampleLoop : verifyParallelLoop workflow
      certificate.inputPreprocessing.digitSecretSamples.parallelLoop = true := by
    exact verifyInputPreprocessing_digitSecretSamplesLoop verified.inputPreprocessingMatches
  have digitGatherLoop : verifyParallelLoop workflow
      certificate.inputPreprocessing.digitSecrets.parallelLoop = true := by
    exact verifyParallelGather_loop verified.preprocessingDigitSecretsGather
  have sourceLoop : verifyParallelLoop workflow
      certificate.inputPreprocessing.transitionSources.parallelLoop = true := by
    exact verifyParallelGather_loop verified.preprocessingTransitionSourcesGather
  have targetPublicLoop : verifyParallelLoop workflow
      certificate.inputPreprocessing.targetPublicMatrices.parallelLoop = true := by
    exact verifyParallelGather_loop verified.preprocessingTargetPublicMatricesGather
  have targetLoop : verifyParallelLoop workflow
      certificate.inputPreprocessing.transitionTargets.parallelLoop = true := by
    exact verifyParallelTransitionTarget_loop verified.preprocessingTransitionTargets
  have transitionLoop : verifyParallelLoop workflow
      certificate.inputPreprocessing.transitionPreimages.parallelLoop = true := by
    exact verifyParallelPreimage_loop verified.preprocessingTransitionPreimages
  have finalLoop : verifyParallelLoop workflow
      certificate.inputPreprocessing.finalTrapdoors.parallelLoop = true := by
    exact verifyParallelGather_loop verified.preprocessingFinalTrapdoorsGather
  have artifactChecks := encryptionArtifactNodeChecks_of_verified verified
  have trapdoorLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.trapdoorSamples.parallelLoop.operation (by simp)
  have sourceIndexLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.transitionSourceIndices.parallelLoop.operation (by simp)
  have targetIndexLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.transitionTargetIndices.parallelLoop.operation (by simp)
  have digitIndexLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.digitSecretIndices.parallelLoop.operation (by simp)
  have digitSampleLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.digitSecretSamples.parallelLoop.operation (by simp)
  have digitGatherLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.digitSecrets.parallelLoop.operation (by simp)
  have sourceLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.transitionSources.parallelLoop.operation (by simp)
  have targetPublicLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.targetPublicMatrices.parallelLoop.operation (by simp)
  have targetLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.transitionTargets.parallelLoop.operation (by simp)
  have transitionLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.transitionPreimages.parallelLoop.operation (by simp)
  have finalLocation := checkedRootEncryptionOperation preprocessingLocations
    certificate.inputPreprocessing.finalTrapdoors.parallelLoop.operation (by simp)
  have packedLocation := checkedRootEncryptionOperation publicLocations
    certificate.publicKeySampling.packedHash.operation (by simp)
  have publicKeysLocation := checkedRootEncryptionOperation publicLocations
    certificate.publicKeySampling.slices.parallelLoop.operation (by simp)
  have witnessLocation := checkedRootEncryptionOperation artifactLocations
    certificate.artifactPreprocessing.witnessPreimages.parallelLoop.operation (by simp)
  have oneSampleLocation := checkedRootEncryptionOperation artifactLocations
    certificate.artifactPreprocessing.onePreimage.sample.operation (by simp)
  have oneMaterializeLocation := checkedRootEncryptionOperation artifactLocations
    certificate.artifactPreprocessing.onePreimage.materialize.operation (by simp)
  have kSampleLocation := checkedRootEncryptionOperation artifactLocations
    certificate.artifactPreprocessing.kPreimage.sample.operation (by simp)
  have kMaterializeLocation := checkedRootEncryptionOperation artifactLocations
    certificate.artifactPreprocessing.kPreimage.materialize.operation (by simp)
  have rDecompositionLocation := checkedRootEncryptionOperation artifactLocations
    certificate.artifactPreprocessing.rDecomposition.operation (by simp)
  have rMaterializationLocation := checkedRootEncryptionOperation artifactLocations
    certificate.artifactPreprocessing.rMaterialization.operation (by simp)
  have rReshapeLocation := checkedRootEncryptionOperation artifactLocations
    certificate.artifactPreprocessing.rReshape.operation (by simp)
  have decoderSampleLocation := checkedRootEncryptionOperation artifactLocations
    certificate.artifactPreprocessing.decoderPreimage.sample.operation (by simp)
  have decoderMaterializeLocation := checkedRootEncryptionOperation artifactLocations
    certificate.artifactPreprocessing.decoderPreimage.materialize.operation (by simp)
  obtain ⟨trapdoorSamples, trapdoorSamplesRooted⟩ := rootPath.encryptionNodeExecution
    _ trapdoorLocation.2 (by simpa [trapdoorLocation.1] using stageResolved)
    (verifyParallelLoop_nodeExists trapdoorLoop)
  obtain ⟨transitionSourceIndices, transitionSourceIndicesRooted⟩ :=
    rootPath.encryptionNodeExecution _ sourceIndexLocation.2
      (by simpa [sourceIndexLocation.1] using stageResolved)
      (verifyParallelLoop_nodeExists sourceIndexLoop)
  obtain ⟨transitionTargetIndices, transitionTargetIndicesRooted⟩ :=
    rootPath.encryptionNodeExecution _ targetIndexLocation.2
      (by simpa [targetIndexLocation.1] using stageResolved)
      (verifyParallelLoop_nodeExists targetIndexLoop)
  obtain ⟨digitSecretIndices, digitSecretIndicesRooted⟩ := rootPath.encryptionNodeExecution
    _ digitIndexLocation.2 (by simpa [digitIndexLocation.1] using stageResolved)
    (verifyParallelLoop_nodeExists digitIndexLoop)
  obtain ⟨digitSecretSamples, digitSecretSamplesRooted⟩ := rootPath.encryptionNodeExecution
    _ digitSampleLocation.2 (by simpa [digitSampleLocation.1] using stageResolved)
    (verifyParallelLoop_nodeExists digitSampleLoop)
  obtain ⟨digitSecrets, digitSecretsRooted⟩ := rootPath.encryptionNodeExecution
    _ digitGatherLocation.2 (by simpa [digitGatherLocation.1] using stageResolved)
    (verifyParallelLoop_nodeExists digitGatherLoop)
  obtain ⟨transitionSources, transitionSourcesRooted⟩ := rootPath.encryptionNodeExecution
    _ sourceLocation.2 (by simpa [sourceLocation.1] using stageResolved)
    (verifyParallelLoop_nodeExists sourceLoop)
  obtain ⟨targetPublicMatrices, targetPublicMatricesRooted⟩ :=
    rootPath.encryptionNodeExecution _ targetPublicLocation.2
      (by simpa [targetPublicLocation.1] using stageResolved)
      (verifyParallelLoop_nodeExists targetPublicLoop)
  obtain ⟨transitionTargets, transitionTargetsRooted⟩ := rootPath.encryptionNodeExecution
    _ targetLocation.2 (by simpa [targetLocation.1] using stageResolved)
    (verifyParallelLoop_nodeExists targetLoop)
  obtain ⟨packedPublicKeys, packedPublicKeysRooted⟩ := rootPath.encryptionOperationExecution
    _ packedLocation.2 (by simpa [packedLocation.1] using stageResolved) artifactChecks.packedHash
  obtain ⟨publicKeys, publicKeysRooted⟩ := rootPath.encryptionNodeExecution
    _ publicKeysLocation.2 (by simpa [publicKeysLocation.1] using stageResolved)
    (verifyParallelLoop_nodeExists artifactChecks.publicKeyLoop)
  obtain ⟨transitions, transitionsRooted⟩ := rootPath.encryptionNodeExecution
    _ transitionLocation.2 (by simpa [transitionLocation.1] using stageResolved)
    (verifyParallelLoop_nodeExists transitionLoop)
  obtain ⟨finalTrapdoors, finalTrapdoorsRooted⟩ := rootPath.encryptionNodeExecution
    _ finalLocation.2 (by simpa [finalLocation.1] using stageResolved)
    (verifyParallelLoop_nodeExists finalLoop)
  obtain ⟨onePreimageSample, onePreimageSampleRooted⟩ :=
    rootPath.encryptionOperationExecution _ oneSampleLocation.2
      (by simpa [oneSampleLocation.1] using stageResolved) artifactChecks.oneSample
  obtain ⟨onePreimage, onePreimageRooted⟩ := rootPath.encryptionOperationExecution _
    oneMaterializeLocation.2 (by simpa [oneMaterializeLocation.1] using stageResolved)
    artifactChecks.oneMaterialize
  obtain ⟨witnessPreimages, witnessPreimagesRooted⟩ := rootPath.encryptionNodeExecution
    _ witnessLocation.2 (by simpa [witnessLocation.1] using stageResolved)
    (verifyParallelLoop_nodeExists artifactChecks.witnessLoop)
  obtain ⟨kPreimageSample, kPreimageSampleRooted⟩ := rootPath.encryptionOperationExecution _
    kSampleLocation.2 (by simpa [kSampleLocation.1] using stageResolved) artifactChecks.kSample
  obtain ⟨kPreimage, kPreimageRooted⟩ := rootPath.encryptionOperationExecution _
    kMaterializeLocation.2 (by simpa [kMaterializeLocation.1] using stageResolved)
    artifactChecks.kMaterialize
  obtain ⟨rDecompositionExecution, rDecompositionRooted⟩ :=
    rootPath.encryptionOperationExecution _ rDecompositionLocation.2
      (by simpa [rDecompositionLocation.1] using stageResolved) artifactChecks.rDecomposition
  obtain ⟨rMaterializationExecution, rMaterializationRooted⟩ :=
    rootPath.encryptionOperationExecution _ rMaterializationLocation.2
      (by simpa [rMaterializationLocation.1] using stageResolved)
      artifactChecks.rMaterialization
  obtain ⟨rDecomposed, rDecomposedRooted⟩ := rootPath.encryptionOperationExecution _
    rReshapeLocation.2 (by simpa [rReshapeLocation.1] using stageResolved)
    artifactChecks.rReshape
  obtain ⟨decoderPreimageSample, decoderPreimageSampleRooted⟩ :=
    rootPath.encryptionOperationExecution _ decoderSampleLocation.2
      (by simpa [decoderSampleLocation.1] using stageResolved) artifactChecks.decoderSample
  obtain ⟨decoderPreimage, decoderPreimageRooted⟩ := rootPath.encryptionOperationExecution _
    decoderMaterializeLocation.2 (by simpa [decoderMaterializeLocation.1] using stageResolved)
    artifactChecks.decoderMaterialize
  exact ⟨{
    rootPath
    trapdoorSamples
    trapdoorSamplesRooted
    transitionSourceIndices
    transitionSourceIndicesRooted
    transitionTargetIndices
    transitionTargetIndicesRooted
    digitSecretIndices
    digitSecretIndicesRooted
    digitSecretSamples
    digitSecretSamplesRooted
    digitSecrets
    digitSecretsRooted
    transitionSources
    transitionSourcesRooted
    targetPublicMatrices
    targetPublicMatricesRooted
    transitionTargets
    transitionTargetsRooted
    packedPublicKeys
    packedPublicKeysRooted
    publicKeys
    publicKeysRooted
    transitions
    transitionsRooted
    finalTrapdoors
    finalTrapdoorsRooted
    onePreimageSample
    onePreimageSampleRooted
    onePreimage
    onePreimageRooted
    witnessPreimages
    witnessPreimagesRooted
    kPreimageSample
    kPreimageSampleRooted
    kPreimage
    kPreimageRooted
    rDecomposition := rDecompositionExecution
    rDecompositionRooted
    rMaterialization := rMaterializationExecution
    rMaterializationRooted
    rDecomposed
    rDecomposedRooted
    decoderPreimageSample
    decoderPreimageSampleRooted
    decoderPreimage
    decoderPreimageRooted
  }, rfl⟩

/-- Membership wrapper for `encryptionArtifactExecutions_of_rootPath`. -/
theorem encryptionArtifactExecutions_of_member
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved : resolveStage workflow certificate.workflow.encryption.stage = some stage)
    (member : output ∈ Mxx.Ir.denote samplers stage.program params inputs) :
    Nonempty (EncryptionArtifactExecutions workflow certificate samplers stage params inputs
      output) := by
  obtain ⟨rootPath⟩ := rootStageExecutionPath_of_member member
  obtain ⟨executions, _⟩ :=
    encryptionArtifactExecutions_of_rootPath verified stageResolved rootPath
  exact ⟨executions⟩

/-- Initial-state and artifact executions extracted from one retained root path.  This is the
same-path constructor consumed by the producer outcome; it cannot combine two independently
selected nondeterministic executions. -/
theorem encryptionSameRootExecutions_of_rootPath
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved : resolveStage workflow certificate.workflow.encryption.stage = some stage)
    (rootPath : RootStageExecutionPath samplers stage params inputs output) :
    ∃ initial : EncryptionInitialStateExecutions workflow certificate samplers stage params inputs
        output,
      ∃ artifacts : EncryptionArtifactExecutions workflow certificate samplers stage params inputs
        output,
        initial.rootPath = artifacts.rootPath := by
  obtain ⟨initial, initialRoot⟩ :=
    encryptionInitialStateExecutions_of_rootPath verified stageResolved rootPath
  obtain ⟨artifacts, artifactsRoot⟩ :=
    encryptionArtifactExecutions_of_rootPath verified stageResolved rootPath
  exact ⟨initial, artifacts, initialRoot.trans artifactsRoot.symm⟩

/-- Select one root path from stage membership once, then derive both producer slices from it. -/
theorem encryptionSameRootExecutions_of_member
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved : resolveStage workflow certificate.workflow.encryption.stage = some stage)
    (member : output ∈ Mxx.Ir.denote samplers stage.program params inputs) :
    ∃ initial : EncryptionInitialStateExecutions workflow certificate samplers stage params inputs
        output,
      ∃ artifacts : EncryptionArtifactExecutions workflow certificate samplers stage params inputs
        output,
        initial.rootPath = artifacts.rootPath := by
  obtain ⟨rootPath⟩ := rootStageExecutionPath_of_member member
  exact encryptionSameRootExecutions_of_rootPath verified stageResolved rootPath

/-- Exact public-key family selected by the checked packed-hash and slice loop.  The family is
kept as executable values because public keys are Large matrices and carry no noise bound. -/
structure EncryptionPublicKeyOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (executions : EncryptionArtifactExecutions workflow certificate samplers stage params inputs
      output) where
  packed : Mxx.Matrix
  keys : List Mxx.Ir.Value
  packedValues : executions.packedPublicKeys.values = [.matrix packed]
  keyValues : executions.publicKeys.values = [.family keys]

/-- Direct sampler facts available before evaluating the dynamic Boolean circuit.  The values are
outcomes of the exact producer nodes above, and every magnitude fact comes from the bounded
sampler contract (or the deterministic decomposition contract). -/
structure EncryptionPreBooleanSamplerOutcomes
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (executions : EncryptionArtifactExecutions workflow certificate samplers stage params inputs
      output) where
  onePreimage : PreimageSampleOutcome workflow certificate.artifactPreprocessing.onePreimage.sample
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs executions.onePreimageSample
  kPreimage : PreimageSampleOutcome workflow certificate.artifactPreprocessing.kPreimage.sample
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs executions.kPreimageSample
  rMatrixParams : Mxx.SamplerParams
  rBase : Int
  rDigitCount : Nat
  rDecomposed : Mxx.Matrix
  rValues : executions.rDecomposed.values = [.matrix rDecomposed]
  rShape : Mxx.Toolkit.MatrixShape rDecomposed
    rMatrixParams.modulus rMatrixParams.ringDimension rMatrixParams.rows rMatrixParams.columns
  rNorm : Mxx.maxCenteredCoefficientNorm rDecomposed ≤
    max (rBase.natAbs / 2) 1

/-- The decoder preimage is intentionally separate: its target contains the exact selected output
of the encryption Boolean circuit, so it can only be constructed after that execution has been
proved typed and exact. -/
structure EncryptionDecoderPreimageOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (executions : EncryptionArtifactExecutions workflow certificate samplers stage params inputs
      output) where
  decoderPreimage : PreimageSampleOutcome
    workflow certificate.artifactPreprocessing.decoderPreimage.sample
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs executions.decoderPreimageSample

/-- Exact values exported by encryption and consumed as artifacts by decryption.  Keeping the
producer lookup, rather than only the matrices, prevents accidentally combining values selected
from different nondeterministic producer paths. -/
structure EncryptionArtifactValues
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (rootPath : RootStageExecutionPath samplers stage params inputs output) where
  decoderPreimage : Mxx.Ir.Value
  initialState : Mxx.Ir.Value
  kPreimage : Mxx.Ir.Value
  onePreimage : Mxx.Ir.Value
  publicKeys : Mxx.Ir.Value
  rDecomposed : Mxx.Ir.Value
  transitions : Mxx.Ir.Value
  witnessPreimages : Mxx.Ir.Value
  decoderPreimageLookup : Mxx.Ir.lookupEnvironment "diamond_decoder_preimage" output =
    some decoderPreimage
  initialStateLookup : Mxx.Ir.lookupEnvironment "diamond_initial_state" output =
    some initialState
  kPreimageLookup : Mxx.Ir.lookupEnvironment "diamond_k_preimage" output = some kPreimage
  onePreimageLookup : Mxx.Ir.lookupEnvironment "diamond_one_preimage" output = some onePreimage
  publicKeysLookup : Mxx.Ir.lookupEnvironment "diamond_public_keys" output = some publicKeys
  rDecomposedLookup : Mxx.Ir.lookupEnvironment "diamond_r_decomposed" output = some rDecomposed
  transitionsLookup : Mxx.Ir.lookupEnvironment "diamond_transitions" output = some transitions
  witnessPreimagesLookup : Mxx.Ir.lookupEnvironment "diamond_witness_preimages" output =
    some witnessPreimages

private theorem lookupEnvironment_exists_of_name_mem
    {environment : Mxx.Ir.Environment} {name : String}
    (member : name ∈ environment.map Prod.fst) :
    ∃ value, Mxx.Ir.lookupEnvironment name environment = some value := by
  induction environment with
  | nil => simp at member
  | cons head tail induction =>
      rcases head with ⟨candidate, value⟩
      simp only [List.map_cons, List.mem_cons] at member
      rcases member with rfl | member
      · exact ⟨value, by simp [Mxx.Ir.lookupEnvironment]⟩
      · obtain ⟨found, foundLookup⟩ := induction member
        by_cases same : candidate = name
        · subst candidate
          exact ⟨value, by simp [Mxx.Ir.lookupEnvironment]⟩
        · exact ⟨found, by simp [Mxx.Ir.lookupEnvironment, same, foundLookup]⟩

private theorem RootStageExecutionPath.encryptionOutputExists
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (stageFacts : VerifiedEncryptionStage workflow certificate)
    (stageEq : stageFacts.stage = stage) (name : String)
    (named : name ∈ [
      "diamond_decoder_preimage", "diamond_initial_state", "diamond_k_preimage",
      "diamond_one_preimage", "diamond_public_keys", "diamond_r_decomposed",
      "diamond_transitions", "diamond_witness_preimages"]) :
    ∃ value, Mxx.Ir.lookupEnvironment name output = some value := by
  have stageNames : stage.program.root.outputs.map Prod.fst = [
      "diamond_decoder_preimage", "diamond_initial_state", "diamond_k_preimage",
      "diamond_one_preimage", "diamond_public_keys", "diamond_r_decomposed",
      "diamond_transitions", "diamond_witness_preimages"] := by
    simpa [stageEq] using stageFacts.outputNamesEq
  apply lookupEnvironment_exists_of_name_mem
  rw [rootPath.outputEq]
  have nameMember : name ∈ stage.program.root.outputs.map Prod.fst := by
    rw [stageNames]
    exact named
  simpa [Mxx.Ir.collectOutputs] using nameMember

/-- Recover the eight exact producer artifact values directly from the retained encryption-stage
output.  The values are chosen from actual environment lookups, not from certificate payloads. -/
theorem encryptionArtifactValues_of_rootPath
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (stageFacts : VerifiedEncryptionStage workflow certificate)
    (stageEq : stageFacts.stage = stage) :
    Nonempty (EncryptionArtifactValues (workflow := workflow) (certificate := certificate)
      rootPath) := by
  obtain ⟨decoderPreimage, decoderPreimageLookup⟩ := rootPath.encryptionOutputExists
    stageFacts stageEq "diamond_decoder_preimage" (by simp)
  obtain ⟨initialState, initialStateLookup⟩ := rootPath.encryptionOutputExists
    stageFacts stageEq "diamond_initial_state" (by simp)
  obtain ⟨kPreimage, kPreimageLookup⟩ := rootPath.encryptionOutputExists
    stageFacts stageEq "diamond_k_preimage" (by simp)
  obtain ⟨onePreimage, onePreimageLookup⟩ := rootPath.encryptionOutputExists
    stageFacts stageEq "diamond_one_preimage" (by simp)
  obtain ⟨publicKeys, publicKeysLookup⟩ := rootPath.encryptionOutputExists
    stageFacts stageEq "diamond_public_keys" (by simp)
  obtain ⟨rDecomposed, rDecomposedLookup⟩ := rootPath.encryptionOutputExists
    stageFacts stageEq "diamond_r_decomposed" (by simp)
  obtain ⟨transitions, transitionsLookup⟩ := rootPath.encryptionOutputExists
    stageFacts stageEq "diamond_transitions" (by simp)
  obtain ⟨witnessPreimages, witnessPreimagesLookup⟩ := rootPath.encryptionOutputExists
    stageFacts stageEq "diamond_witness_preimages" (by simp)
  exact ⟨{
    decoderPreimage
    initialState
    kPreimage
    onePreimage
    publicKeys
    rDecomposed
    transitions
    witnessPreimages
    decoderPreimageLookup
    initialStateLookup
    kPreimageLookup
    onePreimageLookup
    publicKeysLookup
    rDecomposedLookup
    transitionsLookup
    witnessPreimagesLookup
  }⟩

/-- Complete producer-side input-injection evidence recovered from one encryption execution.

The equality between the two retained root paths is intentional: it prevents a proof from
combining an initial state selected from one nondeterministic sampler path with transition or
artifact values selected from another.  The transition table is the exact family exported by the
producer, and its level-indexed bases are therefore the same matrices used by the online scan.
-/
structure EncryptionInputInjectionOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (q ringDimension stateColumns errorBound preimageBound inputCount : Nat)
    [NeZero q] [NeZero ringDimension] where
  parameterValues : EncryptionParameterOutcome params
  modulusEq : parameterValues.modulus = q
  ringDimensionEq : parameterValues.ringDimension = ringDimension
  stateColumnsEq : parameterValues.stateColumns = stateColumns
  errorBoundEq : parameterValues.errorBound = errorBound
  preimageBoundEq : parameterValues.preimageBound = preimageBound
  inputCountEq : parameterValues.inputCount = inputCount
  initialExecutions : EncryptionInitialStateExecutions workflow certificate samplers stage params
    inputs output
  artifactExecutions : EncryptionArtifactExecutions workflow certificate samplers stage params
    inputs output
  sameRootPath : initialExecutions.rootPath = artifactExecutions.rootPath
  message : EncryptionMessageOutcome (workflow := workflow) (certificate := certificate)
    initialExecutions
  table : InputInjectionTransitionTable q ringDimension stateColumns errorBound preimageBound
    inputCount
  initialValues : List Mxx.Ir.Value
  initialSignal : Mxx.Matrix
  initialError : Mxx.Matrix
  initialStateZero : InputInjectionStateZero q ringDimension stateColumns (table.baseAt 0)
    initialValues initialSignal initialError
  initialStateExecution :
    initialExecutions.initialState.values = [.matrix initialStateZero.state]
  initialSignalExecution : initialExecutions.signal.values = [.matrix initialSignal]
  initialErrorExecution : initialExecutions.initialError.values = [.matrix initialError]
  initialSignalNorm : Mxx.maxCenteredCoefficientNorm initialSignal ≤ 1
  initialErrorNorm : Mxx.maxCenteredCoefficientNorm initialError ≤ errorBound
  transitionExecution :
    artifactExecutions.transitions.values = [.family table.transitionFamily]
  artifacts : EncryptionArtifactValues (workflow := workflow) (certificate := certificate)
    artifactExecutions.rootPath
  initialArtifact : artifacts.initialState = .matrix initialStateZero.state
  transitionArtifact : artifacts.transitions = .family table.transitionFamily

/-- Producer facts independent of the dynamic Boolean evaluation. -/
structure EncryptionPreprocessingOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (q ringDimension stateColumns errorBound preimageBound inputCount : Nat)
    [NeZero q] [NeZero ringDimension] where
  inputInjection : EncryptionInputInjectionOutcome (workflow := workflow)
    (certificate := certificate) (samplers := samplers) (stage := stage) (params := params)
    (inputs := inputs) (output := output) q ringDimension stateColumns errorBound preimageBound
    inputCount
  publicKeys : EncryptionPublicKeyOutcome (workflow := workflow) (certificate := certificate)
    (samplers := samplers) (stage := stage) (params := params) (inputs := inputs)
    (output := output)
    inputInjection.artifactExecutions
  directSamplers : EncryptionPreBooleanSamplerOutcomes (workflow := workflow)
    (certificate := certificate) (samplers := samplers) (stage := stage) (params := params)
    (inputs := inputs) (output := output) inputInjection.artifactExecutions

/-- Complete producer result after the exact encryption Boolean outcome has also been used to
construct the decoder-preimage target. -/
structure EncryptionProducerOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (q ringDimension stateColumns errorBound preimageBound inputCount : Nat)
    [NeZero q] [NeZero ringDimension] where
  preprocessing : EncryptionPreprocessingOutcome (workflow := workflow)
    (certificate := certificate) (samplers := samplers) (stage := stage) (params := params)
    (inputs := inputs) (output := output) q ringDimension stateColumns errorBound preimageBound
    inputCount
  decoderPreimage :
    EncryptionDecoderPreimageOutcome (workflow := workflow) (certificate := certificate)
      (samplers := samplers) (stage := stage) (params := params) (inputs := inputs)
      (output := output) preprocessing.inputInjection.artifactExecutions

end MxxWe.Certificate
