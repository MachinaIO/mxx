import MxxWe.Certificate.DecoderExecutionBridge
import MxxWe.Certificate.InputInjectionExecutionBridge

namespace MxxWe.Certificate

/-! Same-path workflow lifting for Diamond input injection. -/

private theorem rootReference_inBounds_of_resolved
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
    simpa [resolveNode, resolveScope, scopeOwnerMatches, rawScope, stageResolved] using nodeResolved
  by_contra outOfBounds
  rw [List.getElem?_eq_none (Nat.le_of_not_gt outOfBounds)] at nodeAt
  contradiction

private theorem resolveNode_of_verifyParallelLoop
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef}
    (verified : verifyParallelLoop workflow reference = true) :
    ∃ node, resolveNode workflow reference.operation = some node := by
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [verifyParallelLoop, resolved] at verified
  | some node => exact ⟨node, rfl⟩

private theorem inputNode_of_verifyInputWire
    {workflow : Mxx.Ir.Workflow} {reference : CoreWireRef} {stage name : String}
    (verified : verifyInputWire workflow reference stage name = true) :
    reference.node.stage = stage ∧ reference.node.scope = .root ∧ reference.port = 0 ∧
      resolveNode workflow reference.node = some {
        kind := .input name
        arguments := []
        outputCount := 1
      } := by
  unfold verifyInputWire at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  cases resolved : resolveNode workflow reference.node with
  | none => simp [resolved] at verified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> simp_all

/-- The input scan and witness-digit packing loop selected from one decryption root path. -/
structure RetainedInputInjectionExecutions
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (rootPath : RootStageExecutionPath samplers stage params inputs output) where
  scanResolution : InputInjectionStateScanResolution workflow certificate.inputInjection
  scan : ReferencedNodeExecution workflow certificate.inputInjection.stateScan
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  scanRooted : RootedNodeExecution rootPath scan
  witnessIndices : ReferencedNodeExecution workflow
    certificate.decryptionInitialEncodings.witnessIndices.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  witnessIndicesRooted : RootedNodeExecution rootPath witnessIndices
  witnessBits : ReferencedNodeExecution workflow
    certificate.decryptionInitialEncodings.witnessBits.parallelLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  witnessBitsRooted : RootedNodeExecution rootPath witnessBits
  witnessSource : CoreOperandRef
  witnessSourceAt :
    certificate.decryptionInitialEncodings.witnessBits.sourceFamilies[0]? = some witnessSource
  witnessInput : ReferencedNodeExecution workflow witnessSource.wire.node
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  witnessInputRooted : RootedNodeExecution rootPath witnessInput
  witnessDigits : ReferencedNodeExecution workflow
    certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  witnessDigitsRooted : RootedNodeExecution rootPath witnessDigits

/-- Recover both online input-injection roots from the same concrete decryption execution. -/
theorem retainedInputInjectionExecutions_of_rootPath
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved : resolveStage workflow certificate.workflow.decryption.stage = some stage)
    (rootPath : RootStageExecutionPath samplers stage params inputs output) :
    Nonempty (RetainedInputInjectionExecutions (workflow := workflow)
      (certificate := certificate) rootPath) := by
  obtain ⟨scanResolution⟩ := verified.stateScanResolution
  have scanLocation := verified.inputInjectionStateScanLocation
  have scanStageResolved : resolveStage workflow certificate.inputInjection.stateScan.stage =
      some stage := by simpa [scanLocation.1] using stageResolved
  have scanInBounds := rootReference_inBounds_of_resolved scanLocation.2 scanStageResolved
    ⟨_, scanResolution.resolved⟩
  obtain ⟨scan, scanRooted⟩ := rootPath.referencedRootNodeExecution
    certificate.inputInjection.stateScan scanLocation.2 scanStageResolved scanInBounds
  obtain ⟨indicesVerified, _, _, _, _, _⟩ := verified.witnessIndexParentFacts
  obtain ⟨_, bitsVerified, _, _, _, _, _⟩ := verified.witnessGatherParentFacts
  obtain ⟨indicesStage, indicesScope, bitsStage, bitsScope⟩ :=
    verified.witnessSourceLoopLocations
  obtain ⟨indicesNode, indicesResolved⟩ := resolveNode_of_verifyParallelLoop indicesVerified
  have indicesStageResolved : resolveStage workflow
      certificate.decryptionInitialEncodings.witnessIndices.operation.stage = some stage := by
    simpa [indicesStage] using stageResolved
  have indicesInBounds := rootReference_inBounds_of_resolved indicesScope indicesStageResolved
    ⟨indicesNode, indicesResolved⟩
  obtain ⟨witnessIndices, witnessIndicesRooted⟩ := rootPath.referencedRootNodeExecution
    certificate.decryptionInitialEncodings.witnessIndices.operation indicesScope
    indicesStageResolved indicesInBounds
  obtain ⟨bitsNode, bitsResolved⟩ := resolveNode_of_verifyParallelLoop bitsVerified
  have bitsStageResolved : resolveStage workflow
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.operation.stage =
        some stage := by
    simpa [bitsStage] using stageResolved
  have bitsInBounds := rootReference_inBounds_of_resolved bitsScope bitsStageResolved
    ⟨bitsNode, bitsResolved⟩
  obtain ⟨witnessBits, witnessBitsRooted⟩ := rootPath.referencedRootNodeExecution
    certificate.decryptionInitialEncodings.witnessBits.parallelLoop.operation bitsScope
    bitsStageResolved bitsInBounds
  have witnessSourceChecked := verified.witnessDigitPackingSource
  cases sources : certificate.decryptionInitialEncodings.witnessBits.sourceFamilies with
  | nil => simp [sources] at witnessSourceChecked
  | cons witnessSource tail =>
    have witnessSourceAt :
        certificate.decryptionInitialEncodings.witnessBits.sourceFamilies[0]? =
          some witnessSource := by simp [sources]
    simp only [sources, List.getElem?_cons_zero] at witnessSourceChecked
    obtain ⟨sourceStage, sourceScope, sourcePort, sourceResolved⟩ :=
      inputNode_of_verifyInputWire witnessSourceChecked
    have sourceStageResolved :
        resolveStage workflow witnessSource.wire.node.stage = some stage := by
      simpa [sourceStage] using stageResolved
    have sourceInBounds := rootReference_inBounds_of_resolved sourceScope sourceStageResolved
      ⟨_, sourceResolved⟩
    obtain ⟨witnessInput, witnessInputRooted⟩ := rootPath.referencedRootNodeExecution
      witnessSource.wire.node sourceScope sourceStageResolved sourceInBounds
    have witnessLocation := verified.witnessDigitPackingLocation
    have witnessLoopVerified := verified.witnessDigitPackingParentVerified
    obtain ⟨witnessNode, witnessResolved⟩ :=
      resolveNode_of_verifyParallelLoop witnessLoopVerified
    have witnessStageResolved : resolveStage workflow
        certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.operation.stage =
          some stage := by simpa [witnessLocation.1] using stageResolved
    have witnessInBounds := rootReference_inBounds_of_resolved witnessLocation.2.1
      witnessStageResolved ⟨witnessNode, witnessResolved⟩
    obtain ⟨witnessDigits, witnessDigitsRooted⟩ := rootPath.referencedRootNodeExecution
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.operation
      witnessLocation.2.1 witnessStageResolved witnessInBounds
    exact ⟨{
      scanResolution
      scan
      scanRooted
      witnessIndices
      witnessIndicesRooted
      witnessBits
      witnessBitsRooted
      witnessSource
      witnessSourceAt
      witnessInput
      witnessInputRooted
      witnessDigits
      witnessDigitsRooted
    }⟩

private theorem list_singleton_of_length_one {α : Type} (values : List α)
    (lengthOne : values.length = 1) : ∃ value, values = [value] := by
  cases values with
  | nil => simp at lengthOne
  | cons value tail =>
    cases tail with
    | nil => exact ⟨value, rfl⟩
    | cons next rest => simp at lengthOne

private theorem parallelLoop_singleOutputWire
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef} {output : CoreWireRef}
    (verified : verifyParallelLoop workflow reference = true)
    (outputs : reference.outputs = [output]) :
    wireRef output = { node := reference.operation.node, port := 0 } := by
  unfold verifyParallelLoop at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have outputChecked := verified.1.1.1.2
  simp [outputs] at outputChecked
  unfold verifyOperationOutput at outputChecked
  simp only [Bool.and_eq_true, decide_eq_true_eq] at outputChecked
  have nodeEq := congrArg CoreNodeRef.node outputChecked.1.1
  have portEq := outputChecked.2
  simp [wireRef, nodeEq, portEq]

private theorem map_range_getElem_getD_eq_take {α : Type} (values : List α)
    (fallback : α) (count : Nat) (countLe : count ≤ values.length) :
    (List.range count).map (fun index ↦ values[index]?.getD fallback) = values.take count := by
  apply List.ext_getElem
  · simp [countLe]
  · intro index leftValid rightValid
    have indexLt : index < count := by simpa using leftValid
    have valueValid : index < values.length := lt_of_lt_of_le indexLt countLe
    simp [List.getElem_map, valueValid]

private theorem parallelLoopArgumentsEvaluate_of_twoLookups
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    {execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs}
    (left right : CoreOperandRef) (leftValue rightValue : Mxx.Ir.Value)
    (arguments : reference.arguments = [left, right])
    (leftLookup : Mxx.Ir.lookupWire (wireRef left.wire) execution.before = some leftValue)
    (rightLookup : Mxx.Ir.lookupWire (wireRef right.wire) execution.before = some rightValue) :
    (reference.arguments.map (wireRef ∘ CoreOperandRef.wire)).mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) = some [leftValue, rightValue] := by
  simp [arguments, leftLookup, rightLookup]

/-- Construct the witness-index trace directly from its retained execution and checked count. -/
theorem RetainedInputInjectionExecutions.witnessIndicesTrace
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {rootPath : RootStageExecutionPath samplers stage params inputs output}
    (retained : RetainedInputInjectionExecutions
      (workflow := workflow) (certificate := certificate) rootPath)
    (verified : VerifiedDiamondLayout workflow certificate)
    (evaluatedCount : Int)
    (countEvaluate :
      certificate.decryptionInitialEncodings.witnessIndices.count.evaluate params =
        some evaluatedCount) :
    Nonempty (CheckedParallelLoopTrace workflow
      certificate.decryptionInitialEncodings.witnessIndices
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs retained.witnessIndices) := by
  obtain ⟨loopVerified, _, _, modesEmpty, _, _⟩ := verified.witnessIndexParentFacts
  have argumentsEmpty :
      certificate.decryptionInitialEncodings.witnessIndices.arguments = [] :=
    verifyDecryptionWitnessIndexFormula_argumentsEmpty verified.witnessDigitIndexFormula
  obtain ⟨resolution⟩ := checkedParallelLoopResolution_of_verified loopVerified
  apply checkedParallelLoopTrace_of_execution resolution retained.witnessIndices []
    evaluatedCount
  · simp [argumentsEmpty]
  · exact countEvaluate

/-- Execute the witness-index producer without requiring a caller-provided trace. -/
theorem RetainedInputInjectionExecutions.witnessIndicesFamilyOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {rootPath : RootStageExecutionPath samplers stage params inputs output}
    (retained : RetainedInputInjectionExecutions
      (workflow := workflow) (certificate := certificate) rootPath)
    (verified : VerifiedDiamondLayout workflow certificate)
    (bodyResolved : resolveScope workflow {
      certificate.decryptionInitialEncodings.witnessIndices.operation with
      scope := certificate.decryptionInitialEncodings.witnessIndices.bodyScope } = some body)
    (definitionFound : Mxx.Ir.lookupDefinition
      certificate.decryptionInitialEncodings.witnessIndices.bodyScope.definitionName
        stage.program.definitions = some body)
    (witnessWidth : Nat)
    (countEvaluate :
      certificate.decryptionInitialEncodings.witnessIndices.count.evaluate params =
        some (Int.ofNat witnessWidth)) :
    retained.witnessIndices.values =
      [.family ((List.range witnessWidth).map fun index ↦ .integer (Int.ofNat index))] := by
  obtain ⟨trace⟩ := retained.witnessIndicesTrace (workflow := workflow)
    (certificate := certificate) verified (Int.ofNat witnessWidth) countEvaluate
  have definitionMember := lookupDefinition_mem definitionFound
  have definitionsNonempty : stage.program.definitions ≠ [] := by
    intro empty
    rw [empty] at definitionMember
    simp at definitionMember
  have runFuelPositive : 0 < stage.program.definitions.length := by
    cases definitionsEq : stage.program.definitions with
    | nil => exact (definitionsNonempty definitionsEq).elim
    | cons head tail => simp
  have evaluatedCountEq : trace.evaluatedCount = Int.ofNat witnessWidth :=
    Option.some.inj (trace.countEvaluate.symm.trans countEvaluate)
  have outcome := decryptionWitnessIndex_familyOutcome verified trace bodyResolved
    definitionFound runFuelPositive
  simpa [evaluatedCountEq] using outcome

private theorem VerifiedDiamondLayout.witnessGatherIndexWire
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    wireRef certificate.decryptionInitialEncodings.witnessBits.indexFamily.wire = {
      node := certificate.decryptionInitialEncodings.witnessIndices.operation.node
      port := 0
    } := by
  obtain ⟨loopVerified, _, _, _, _, outputsOne⟩ := verified.witnessIndexParentFacts
  have indexMember : certificate.decryptionInitialEncodings.witnessBits.indexFamily.wire ∈
      certificate.decryptionInitialEncodings.witnessIndices.outputs := by
    have matched := verified.decryptionInitialEncodingsMatch
    unfold verifyDecryptionInitialEncodings at matched
    simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_cons, List.all_nil,
      and_true] at matched
    aesop
  obtain ⟨output, outputs⟩ := list_singleton_of_length_one _ outputsOne
  have outputEq : certificate.decryptionInitialEncodings.witnessBits.indexFamily.wire =
      output := by
    simpa [outputs] using indexMember
  subst output
  exact parallelLoop_singleOutputWire loopVerified outputs

private theorem VerifiedDiamondLayout.witnessGatherSourcePort
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {source : CoreOperandRef}
    (verified : VerifiedDiamondLayout workflow certificate)
    (sourceAt : certificate.decryptionInitialEncodings.witnessBits.sourceFamilies[0]? =
      some source) :
    source.wire.port = 0 := by
  have checked := verified.witnessDigitPackingSource
  rw [sourceAt] at checked
  unfold verifyInputWire at checked
  simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
  aesop

/-- The retained witness source executes the checked `boolean-witness` input node exactly. -/
theorem RetainedInputInjectionExecutions.witnessInputOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {rootPath : RootStageExecutionPath samplers stage params inputs output}
    (retained : RetainedInputInjectionExecutions
      (workflow := workflow) (certificate := certificate) rootPath)
    (verified : VerifiedDiamondLayout workflow certificate)
    (values : List Mxx.Ir.Value)
    (witnessLookup : Mxx.Ir.lookupEnvironment "boolean-witness" inputs =
      some (.family values)) :
    retained.witnessInput.values = [.family values] := by
  have sourceChecked := verified.witnessDigitPackingSource
  rw [retained.witnessSourceAt] at sourceChecked
  obtain ⟨_, _, _, sourceResolved⟩ := inputNode_of_verifyInputWire sourceChecked
  have checkedResolved := sourceResolved
  rw [retained.witnessInput.resolved] at checkedResolved
  have nodeEq := Option.some.inj checkedResolved
  have member := retained.witnessInput.member
  simpa [nodeEq, Mxx.Ir.evaluateNode, witnessLookup] using member

/-- The checked witness gather reads the exact index and witness-input outputs retained on the
same decryption path.  No caller-supplied lookup or non-invalid premise is used. -/
private theorem RetainedInputInjectionExecutions.witnessGatherArgumentsEvaluate
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {rootPath : RootStageExecutionPath samplers stage params inputs output}
    (retained : RetainedInputInjectionExecutions
      (workflow := workflow) (certificate := certificate) rootPath)
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved : resolveStage workflow certificate.workflow.decryption.stage = some stage)
    (indexValues sourceValues : List Mxx.Ir.Value)
    (indicesOutcome : retained.witnessIndices.values = [.family indexValues])
    (sourceOutcome : retained.witnessInput.values = [.family sourceValues]) :
    (certificate.decryptionInitialEncodings.witnessBits.parallelLoop.arguments.map
      (wireRef ∘ CoreOperandRef.wire)).mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire retained.witnessBits.before) =
      some [.family indexValues, .family sourceValues] := by
  obtain ⟨gatherVerified, loopVerified, _, _, _, _, outputsOne⟩ :=
    verified.witnessGatherParentFacts
  have gatherFacts := gatherVerified
  unfold verifyParallelGather at gatherFacts
  simp only [Bool.and_eq_true, decide_eq_true_eq] at gatherFacts
  have sourceLength :
      certificate.decryptionInitialEncodings.witnessBits.sourceFamilies.length = 1 := by
    aesop
  obtain ⟨actualSource, sources⟩ := list_singleton_of_length_one _ sourceLength
  have actualSourceEq : actualSource = retained.witnessSource := by
    simpa [sources] using retained.witnessSourceAt
  subst actualSource
  have arguments : certificate.decryptionInitialEncodings.witnessBits.parallelLoop.arguments =
      [certificate.decryptionInitialEncodings.witnessBits.indexFamily,
        retained.witnessSource] := by
    simpa [sources] using gatherFacts.1.1.1.1.1.1.1.2
  obtain ⟨loopResolution⟩ := checkedParallelLoopResolution_of_verified loopVerified
  have consumerNodeEq : retained.witnessBits.node = {
      kind := .parallelLoop
        certificate.decryptionInitialEncodings.witnessBits.parallelLoop.bodyScope.definitionName
        certificate.decryptionInitialEncodings.witnessBits.parallelLoop.count
        certificate.decryptionInitialEncodings.witnessBits.parallelLoop.indexSlot
        certificate.decryptionInitialEncodings.witnessBits.parallelLoop.bindings
        (certificate.decryptionInitialEncodings.witnessBits.parallelLoop.inputModes.map
          CertifiedLoopInputMode.toIr)
      arguments := [wireRef
        certificate.decryptionInitialEncodings.witnessBits.indexFamily.wire,
        wireRef retained.witnessSource.wire]
      outputCount :=
        certificate.decryptionInitialEncodings.witnessBits.parallelLoop.outputs.length
    } := by
    have checkedResolved := loopResolution.resolved
    rw [retained.witnessBits.resolved] at checkedResolved
    simpa [arguments] using Option.some.inj checkedResolved
  obtain ⟨_, _, bitsStage, bitsScope⟩ := verified.witnessSourceLoopLocations
  have consumerScopeResolved : resolveScope workflow
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.operation =
        some stage.program.root := by
    simp [resolveScope, bitsStage, bitsScope, stageResolved, scopeOwnerMatches, rawScope]
  have consumerAt := resolveNode_scopeNode consumerScopeResolved retained.witnessBits.resolved
  have consumerLt := list_index_lt_of_getElem?_eq_some consumerAt
  have consumerRootNode : stage.program.root.nodes[
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.operation.node] =
        retained.witnessBits.node := by
    rw [List.getElem?_eq_getElem consumerLt] at consumerAt
    exact Option.some.inj consumerAt
  have stageMember := resolveStage_mem stageResolved
  have rootSsa : verifyScopeSsaOrder stage.program.root = true := by
    have checked := verified.ssaOrderValid
    unfold verifyWorkflowSsaOrder at checked
    simp only [List.all_eq_true, Bool.and_eq_true] at checked
    exact (checked stage stageMember).1
  have indexWire := verified.witnessGatherIndexWire
  have indexFinal : Mxx.Ir.lookupWire
      (wireRef certificate.decryptionInitialEncodings.witnessBits.indexFamily.wire)
      rootPath.finalWires = some (.family indexValues) := by
    have exactOutput := retained.witnessIndicesRooted.outputFinal 0 (by simp [indicesOutcome])
    simpa [indicesOutcome, indexWire] using exactOutput
  have indexMember : wireRef
      certificate.decryptionInitialEncodings.witnessBits.indexFamily.wire ∈
        retained.witnessBits.node.arguments := by
    simp [consumerNodeEq]
  have indexPast := verifyScopeSsaOrder_argument_lt rootSsa
    certificate.decryptionInitialEncodings.witnessBits.parallelLoop.operation.node consumerLt
    (wireRef certificate.decryptionInitialEncodings.witnessBits.indexFamily.wire)
    (by simpa [consumerRootNode] using indexMember)
  have indexLookup := retained.witnessBitsRooted.finalBefore _ _ indexPast indexFinal
  have sourcePort := verified.witnessGatherSourcePort retained.witnessSourceAt
  have sourceFinal : Mxx.Ir.lookupWire (wireRef retained.witnessSource.wire)
      rootPath.finalWires = some (.family sourceValues) := by
    have exactOutput := retained.witnessInputRooted.outputFinal 0 (by simp [sourceOutcome])
    simpa [sourceOutcome, wireRef, sourcePort] using exactOutput
  have sourceMember : wireRef retained.witnessSource.wire ∈
      retained.witnessBits.node.arguments := by
    simp [consumerNodeEq]
  have sourcePast := verifyScopeSsaOrder_argument_lt rootSsa
    certificate.decryptionInitialEncodings.witnessBits.parallelLoop.operation.node consumerLt
    (wireRef retained.witnessSource.wire) (by simpa [consumerRootNode] using sourceMember)
  have sourceLookup := retained.witnessBitsRooted.finalBefore _ _ sourcePast sourceFinal
  exact parallelLoopArgumentsEvaluate_of_twoLookups _ _ _ _ arguments indexLookup sourceLookup

/-- Construct the witness-gather trace from the exact same-path index and source outcomes. -/
theorem RetainedInputInjectionExecutions.witnessGatherTrace
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {rootPath : RootStageExecutionPath samplers stage params inputs output}
    (retained : RetainedInputInjectionExecutions
      (workflow := workflow) (certificate := certificate) rootPath)
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved : resolveStage workflow certificate.workflow.decryption.stage = some stage)
    (indexValues sourceValues : List Mxx.Ir.Value)
    (indicesOutcome : retained.witnessIndices.values = [.family indexValues])
    (sourceOutcome : retained.witnessInput.values = [.family sourceValues])
    (evaluatedCount : Int)
    (countEvaluate :
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.count.evaluate params =
        some evaluatedCount) :
    Nonempty (CheckedParallelLoopTrace workflow
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs retained.witnessBits) := by
  obtain ⟨_, loopVerified, _, _, _, _, _⟩ := verified.witnessGatherParentFacts
  obtain ⟨resolution⟩ := checkedParallelLoopResolution_of_verified loopVerified
  have argumentsEvaluate := retained.witnessGatherArgumentsEvaluate verified stageResolved
    indexValues sourceValues indicesOutcome sourceOutcome
  exact checkedParallelLoopTrace_of_execution resolution retained.witnessBits
    [.family indexValues, .family sourceValues] evaluatedCount argumentsEvaluate countEvaluate

/-- Execute the complete checked witness gather from the two exact producers retained on the
same decryption path. -/
theorem RetainedInputInjectionExecutions.witnessGatherFamilyOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {rootPath : RootStageExecutionPath samplers stage params inputs output}
    (retained : RetainedInputInjectionExecutions
      (workflow := workflow) (certificate := certificate) rootPath)
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved : resolveStage workflow certificate.workflow.decryption.stage = some stage)
    (body : Mxx.Ir.Scope)
    (bodyResolved : resolveScope workflow {
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.operation with
      scope := certificate.decryptionInitialEncodings.witnessBits.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.bodyScope.definitionName
        stage.program.definitions = some body)
    (evaluatedCount : Int)
    (countEvaluate :
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.count.evaluate params =
        some evaluatedCount)
    (sourceValues : List Mxx.Ir.Value)
    (indicesOutcome : retained.witnessIndices.values =
      [.family ((List.range evaluatedCount.toNat).map fun index =>
        .integer (Int.ofNat index))])
    (sourceOutcome : retained.witnessInput.values = [.family sourceValues]) :
    retained.witnessBits.values =
      [.family ((List.range evaluatedCount.toNat).map fun index =>
        sourceValues[index]?.getD (.invalid "FamilyGetDynamic index out of range"))] := by
  obtain ⟨trace⟩ := retained.witnessGatherTrace (workflow := workflow)
    (certificate := certificate) verified stageResolved
    ((List.range evaluatedCount.toNat).map fun index => .integer (Int.ofNat index)) sourceValues
    indicesOutcome sourceOutcome evaluatedCount countEvaluate
  obtain ⟨gatherVerified, _, _, bindings, modes, _, outputsOne⟩ :=
    verified.witnessGatherParentFacts
  have gatherFacts := gatherVerified
  unfold verifyParallelGather at gatherFacts
  simp only [Bool.and_eq_true, decide_eq_true_eq] at gatherFacts
  have sourceLength :
      certificate.decryptionInitialEncodings.witnessBits.sourceFamilies.length = 1 := by
    aesop
  have bodySourceLength :
      certificate.decryptionInitialEncodings.witnessBits.bodySources.length = 1 := by
    aesop
  have getLength : certificate.decryptionInitialEncodings.witnessBits.gets.length = 1 := by
    aesop
  have outputLength :
      certificate.decryptionInitialEncodings.witnessBits.outputFamilies.length = 1 := by
    aesop
  obtain ⟨actualSource, sources⟩ := list_singleton_of_length_one _ sourceLength
  have actualSourceEq : actualSource = retained.witnessSource := by
    simpa [sources] using retained.witnessSourceAt
  subst actualSource
  obtain ⟨bodySource, bodySources⟩ :=
    list_singleton_of_length_one _ bodySourceLength
  obtain ⟨get, gets⟩ := list_singleton_of_length_one _ getLength
  obtain ⟨familyOutput, familyOutputs⟩ :=
    list_singleton_of_length_one _ outputLength
  have stageMember := resolveStage_mem stageResolved
  have definitionMember := lookupDefinition_mem definitionFound
  have ssaOrder := verifyWorkflowSsaOrder_definition verified.ssaOrderValid stageMember
    definitionMember
  have definitionsNonempty : stage.program.definitions ≠ [] := by
    intro empty
    rw [empty] at definitionMember
    simp at definitionMember
  have runFuelPositive : 0 < stage.program.definitions.length := by
    cases definitionsEq : stage.program.definitions with
    | nil => exact (definitionsNonempty definitionsEq).elim
    | cons head tail => simp
  have argumentsEvaluate := retained.witnessGatherArgumentsEvaluate (workflow := workflow)
    (certificate := certificate) verified stageResolved
    ((List.range evaluatedCount.toNat).map fun index => .integer (Int.ofNat index))
    sourceValues indicesOutcome sourceOutcome
  have evaluatedCountEq : trace.evaluatedCount = evaluatedCount :=
    Option.some.inj (trace.countEvaluate.symm.trans countEvaluate)
  have traceArguments : trace.argumentValues =
      [.family ((List.range trace.evaluatedCount.toNat).map fun index =>
        .integer (Int.ofNat index)), .family sourceValues] := by
    have exactArguments := Option.some.inj
      (trace.argumentsEvaluate.symm.trans argumentsEvaluate)
    simpa [evaluatedCountEq] using exactArguments
  have outcome := oneSourceParallelGather_familyOutcome gatherVerified sources bodySources gets
    familyOutputs
    modes bindings trace bodyResolved definitionFound runFuelPositive ssaOrder traceArguments
  simpa [evaluatedCountEq] using outcome

private theorem VerifiedDiamondLayout.witnessDigitPackingArgument
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    ∃ argument,
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.arguments = [argument] ∧
      wireRef argument.wire = {
        node := certificate.decryptionInitialEncodings.witnessBits.parallelLoop.operation.node
        port := 0
      } := by
  have loopVerified := verified.witnessDigitPackingParentVerified
  have namesVerified := verified.witnessDigitPackingInputNames
  unfold verifyWitnessDigitPackingInputNames at namesVerified
  simp only [Bool.and_eq_true] at namesVerified
  have argumentLength :
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.arguments.length = 1 := by
    cases bodyResolved : resolveScope workflow {
        certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.operation with
        scope := certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.bodyScope } with
    | none => simp [bodyResolved] at namesVerified
    | some body =>
        have outerNames : body.inputNames = ["pack-bit-source"] := by
          simpa [bodyResolved] using namesVerified.1
        have inputBindings := verifyParallelLoop_bodyInputBindings loopVerified bodyResolved
        calc
          certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.arguments.length =
              body.inputNames.length := inputBindings.2.2.1
          _ = 1 := by simp [outerNames]
  obtain ⟨argument, arguments⟩ := list_singleton_of_length_one _ argumentLength
  refine ⟨argument, arguments, ?_⟩
  have gatherParent := verified.witnessGatherParentFacts
  have gatherVerified := gatherParent.1
  unfold verifyParallelGather at gatherVerified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at gatherVerified
  have outputLength :
      certificate.decryptionInitialEncodings.witnessBits.outputFamilies.length = 1 := by
    aesop
  obtain ⟨familyOutput, familyOutputs⟩ := list_singleton_of_length_one _ outputLength
  have sourceWiring := verified.decryptionInitialEncodingsMatch
  unfold verifyDecryptionInitialEncodings at sourceWiring
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_cons, List.all_nil,
    and_true] at sourceWiring
  have argumentWire : argument.wire = familyOutput := by
    simp [arguments, familyOutputs] at sourceWiring
    aesop
  have gatherLoopVerified := gatherParent.2.1
  have familyOutputWire : wireRef familyOutput = {
      node := certificate.decryptionInitialEncodings.witnessBits.parallelLoop.operation.node
      port := 0
    } := by
    have loopOutputs :
        certificate.decryptionInitialEncodings.witnessBits.parallelLoop.outputs =
          [familyOutput] := by
      simpa [familyOutputs] using gatherVerified.1.2
    exact parallelLoop_singleOutputWire gatherLoopVerified loopOutputs
  simpa [argumentWire] using familyOutputWire

private theorem RetainedInputInjectionExecutions.witnessDigitPackingArgumentsEvaluate
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {rootPath : RootStageExecutionPath samplers stage params inputs output}
    (retained : RetainedInputInjectionExecutions
      (workflow := workflow) (certificate := certificate) rootPath)
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved : resolveStage workflow certificate.workflow.decryption.stage = some stage)
    (bits : List Int)
    (witnessBitsOutcome : retained.witnessBits.values =
      [.family (bits.map Mxx.Ir.Value.integer)]) :
    (certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.arguments.map
      (wireRef ∘ CoreOperandRef.wire)).mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire retained.witnessDigits.before) =
      some [.family (bits.map Mxx.Ir.Value.integer)] := by
  obtain ⟨argument, arguments, argumentWire⟩ := verified.witnessDigitPackingArgument
  have packingLocation := verified.witnessDigitPackingLocation
  have digitsStage := packingLocation.1
  have digitsScope := packingLocation.2.1
  have consumerScopeResolved : resolveScope workflow
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.operation =
        some stage.program.root := by
    simp [resolveScope, digitsStage, digitsScope, stageResolved, scopeOwnerMatches, rawScope]
  obtain ⟨loopResolution⟩ := checkedParallelLoopResolution_of_verified
    verified.witnessDigitPackingParentVerified
  have consumerNodeEq : retained.witnessDigits.node = {
      kind := .parallelLoop
        certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.bodyScope.definitionName
        certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.count
        certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.indexSlot
        certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.bindings
        (certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.inputModes.map
          CertifiedLoopInputMode.toIr)
      arguments := [wireRef argument.wire]
      outputCount :=
        certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.outputs.length
    } := by
    have checkedResolved := loopResolution.resolved
    rw [retained.witnessDigits.resolved] at checkedResolved
    simpa [arguments] using Option.some.inj checkedResolved
  have consumerAt := resolveNode_scopeNode consumerScopeResolved retained.witnessDigits.resolved
  have consumerLt := list_index_lt_of_getElem?_eq_some consumerAt
  have consumerRootNode : stage.program.root.nodes[
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.operation.node] =
        retained.witnessDigits.node := by
    rw [List.getElem?_eq_getElem consumerLt] at consumerAt
    exact Option.some.inj consumerAt
  have stageMember := resolveStage_mem stageResolved
  have rootSsa : verifyScopeSsaOrder stage.program.root = true := by
    have checked := verified.ssaOrderValid
    unfold verifyWorkflowSsaOrder at checked
    simp only [List.all_eq_true, Bool.and_eq_true] at checked
    exact (checked stage stageMember).1
  have bitsFinal : Mxx.Ir.lookupWire (wireRef argument.wire) rootPath.finalWires =
      some (.family (bits.map Mxx.Ir.Value.integer)) := by
    have exactOutput := retained.witnessBitsRooted.outputFinal 0 (by simp [witnessBitsOutcome])
    simpa [witnessBitsOutcome, argumentWire] using exactOutput
  have argumentMember : wireRef argument.wire ∈ retained.witnessDigits.node.arguments := by
    simp [consumerNodeEq]
  have argumentPast := verifyScopeSsaOrder_argument_lt rootSsa
    certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.operation.node consumerLt
    (wireRef argument.wire) (by simpa [consumerRootNode] using argumentMember)
  have argumentLookup := retained.witnessDigitsRooted.finalBefore _ _ argumentPast bitsFinal
  simp [arguments, argumentLookup]

/-- Construct the witness-digit packing trace from the exact same-path gather outcome. -/
theorem RetainedInputInjectionExecutions.witnessDigitPackingTrace
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {rootPath : RootStageExecutionPath samplers stage params inputs output}
    (retained : RetainedInputInjectionExecutions
      (workflow := workflow) (certificate := certificate) rootPath)
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved : resolveStage workflow certificate.workflow.decryption.stage = some stage)
    (bits : List Int)
    (witnessBitsOutcome : retained.witnessBits.values =
      [.family (bits.map Mxx.Ir.Value.integer)])
    (evaluatedCount : Int)
    (countEvaluate :
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.count.evaluate params =
        some evaluatedCount) :
    Nonempty (CheckedParallelLoopTrace workflow
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs retained.witnessDigits) := by
  obtain ⟨resolution⟩ := checkedParallelLoopResolution_of_verified
    verified.witnessDigitPackingParentVerified
  have argumentsEvaluate := retained.witnessDigitPackingArgumentsEvaluate (workflow := workflow)
    (certificate := certificate) verified stageResolved bits witnessBitsOutcome
  exact checkedParallelLoopTrace_of_execution resolution retained.witnessDigits
    [.family (bits.map Mxx.Ir.Value.integer)] evaluatedCount argumentsEvaluate countEvaluate

/-- Execute witness-digit packing from the exact gathered witness family retained on the same
decryption path. -/
private theorem RetainedInputInjectionExecutions.witnessDigitPackingFamilyOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {rootPath : RootStageExecutionPath samplers stage params inputs output}
    (retained : RetainedInputInjectionExecutions
      (workflow := workflow) (certificate := certificate) rootPath)
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved : resolveStage workflow certificate.workflow.decryption.stage = some stage)
    (outerBody scanBody : Mxx.Ir.Scope)
    (outerBodyResolved : resolveScope workflow {
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.operation with
      scope := certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.bodyScope } =
        some outerBody)
    (outerDefinitionFound : Mxx.Ir.lookupDefinition
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.bodyScope.definitionName
        stage.program.definitions = some outerBody)
    (scanBodyResolved : resolveScope workflow {
      certificate.decryptionInitialEncodings.witnessDigits.bitScan.operation with
      scope := certificate.decryptionInitialEncodings.witnessDigits.bitScan.bodyScope } =
        some scanBody)
    (scanDefinitionFound : Mxx.Ir.lookupDefinition
      certificate.decryptionInitialEncodings.witnessDigits.bitScan.bodyScope.definitionName
        stage.program.definitions = some scanBody)
    (runFuelAtLeastTwo : 2 ≤ stage.program.definitions.length)
    (bits : List Int) (batch : Nat)
    (witnessBitsOutcome : retained.witnessBits.values =
      [.family (bits.map Mxx.Ir.Value.integer)])
    (evaluatedCount : Int)
    (countEvaluate :
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.count.evaluate params =
        some evaluatedCount)
    (batchEvaluate :
      certificate.decryptionInitialEncodings.witnessDigits.bitScan.count.evaluate params =
        some (Int.ofNat batch))
    (bitsPresent : ∀ level ∈ List.range evaluatedCount.toNat,
      ∀ index, index < batch → ∃ bit, bits[level * batch + index]? = some bit) :
    retained.witnessDigits.values =
      [.family ((List.range evaluatedCount.toNat).map fun level =>
        .integer (packedWitnessPrefix bits (level * batch) batch))] := by
  obtain ⟨trace⟩ := retained.witnessDigitPackingTrace (workflow := workflow)
    (certificate := certificate) verified stageResolved bits witnessBitsOutcome evaluatedCount
    countEvaluate
  have argumentsEvaluate := retained.witnessDigitPackingArgumentsEvaluate
    (workflow := workflow) (certificate := certificate) verified stageResolved bits
    witnessBitsOutcome
  have traceArguments : trace.argumentValues =
      [.family (bits.map Mxx.Ir.Value.integer)] :=
    Option.some.inj (trace.argumentsEvaluate.symm.trans argumentsEvaluate)
  have evaluatedCountEq : trace.evaluatedCount = evaluatedCount :=
    Option.some.inj (trace.countEvaluate.symm.trans countEvaluate)
  have traceBitsPresent : ∀ level ∈ List.range trace.evaluatedCount.toNat,
      ∀ index, index < batch → ∃ bit, bits[level * batch + index]? = some bit := by
    simpa [evaluatedCountEq] using bitsPresent
  have outcome := witnessDigitPacking_familyOutcome verified.witnessDigitPackingFormula trace
    outerBodyResolved outerDefinitionFound
    scanBodyResolved scanDefinitionFound runFuelAtLeastTwo traceArguments batchEvaluate
    traceBitsPresent
  simpa [evaluatedCountEq] using outcome

/-- Exact retained outputs of the complete witness-index, gather, and digit-packing pipeline. -/
structure WitnessPackingExecutionOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {rootPath : RootStageExecutionPath samplers stage params inputs output}
    (retained : RetainedInputInjectionExecutions
      (workflow := workflow) (certificate := certificate) rootPath)
    (bits : List Int) (witnessWidth inputCount batch : Nat) where
  indices : retained.witnessIndices.values =
    [.family ((List.range witnessWidth).map fun index ↦ .integer (Int.ofNat index))]
  gatheredBits : retained.witnessBits.values =
    [.family ((bits.take witnessWidth).map Mxx.Ir.Value.integer)]
  packedDigits : retained.witnessDigits.values =
    [.family ((List.range inputCount).map fun level ↦
      .integer (packedWitnessPrefix (bits.take witnessWidth) (level * batch) batch))]

/-- Execute the complete retained witness-packing pipeline from the checked input environment.
No raw node outcome, lookup-at-wire, non-invalid result, or loop trace is supplied by the caller. -/
theorem RetainedInputInjectionExecutions.witnessPackingExecutionOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {rootPath : RootStageExecutionPath samplers stage params inputs output}
    (retained : RetainedInputInjectionExecutions
      (workflow := workflow) (certificate := certificate) rootPath)
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved : resolveStage workflow certificate.workflow.decryption.stage = some stage)
    (indexBody gatherBody outerBody scanBody : Mxx.Ir.Scope)
    (indexBodyResolved : resolveScope workflow {
      certificate.decryptionInitialEncodings.witnessIndices.operation with
      scope := certificate.decryptionInitialEncodings.witnessIndices.bodyScope } = some indexBody)
    (indexDefinitionFound : Mxx.Ir.lookupDefinition
      certificate.decryptionInitialEncodings.witnessIndices.bodyScope.definitionName
        stage.program.definitions = some indexBody)
    (gatherBodyResolved : resolveScope workflow {
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.operation with
      scope := certificate.decryptionInitialEncodings.witnessBits.parallelLoop.bodyScope } =
        some gatherBody)
    (gatherDefinitionFound : Mxx.Ir.lookupDefinition
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.bodyScope.definitionName
        stage.program.definitions = some gatherBody)
    (outerBodyResolved : resolveScope workflow {
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.operation with
      scope := certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.bodyScope } =
        some outerBody)
    (outerDefinitionFound : Mxx.Ir.lookupDefinition
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.bodyScope.definitionName
        stage.program.definitions = some outerBody)
    (scanBodyResolved : resolveScope workflow {
      certificate.decryptionInitialEncodings.witnessDigits.bitScan.operation with
      scope := certificate.decryptionInitialEncodings.witnessDigits.bitScan.bodyScope } =
        some scanBody)
    (scanDefinitionFound : Mxx.Ir.lookupDefinition
      certificate.decryptionInitialEncodings.witnessDigits.bitScan.bodyScope.definitionName
        stage.program.definitions = some scanBody)
    (runFuelAtLeastTwo : 2 ≤ stage.program.definitions.length)
    (bits : List Int) (witnessWidth inputCount batch : Nat)
    (indexCountEvaluate :
      certificate.decryptionInitialEncodings.witnessIndices.count.evaluate params =
        some (Int.ofNat witnessWidth))
    (gatherCountEvaluate :
      certificate.decryptionInitialEncodings.witnessBits.parallelLoop.count.evaluate params =
        some (Int.ofNat witnessWidth))
    (packingCountEvaluate :
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.count.evaluate params =
        some (Int.ofNat inputCount))
    (batchEvaluate :
      certificate.decryptionInitialEncodings.witnessDigits.bitScan.count.evaluate params =
        some (Int.ofNat batch))
    (witnessLookup : Mxx.Ir.lookupEnvironment "boolean-witness" inputs =
      some (.family (bits.map Mxx.Ir.Value.integer)))
    (bitsPresent : ∀ index, index < witnessWidth → ∃ bit, bits[index]? = some bit) :
    WitnessPackingExecutionOutcome (workflow := workflow) (certificate := certificate)
      retained bits witnessWidth inputCount batch := by
  have sourceOutcome := retained.witnessInputOutcome (workflow := workflow)
    (certificate := certificate) verified
    (bits.map Mxx.Ir.Value.integer) witnessLookup
  have indicesOutcome := retained.witnessIndicesFamilyOutcome (workflow := workflow)
    (certificate := certificate) verified indexBodyResolved indexDefinitionFound witnessWidth
    indexCountEvaluate
  have widthLe : witnessWidth ≤ bits.length := by
    by_cases widthZero : witnessWidth = 0
    · omega
    · obtain ⟨bit, bitAt⟩ := bitsPresent (witnessWidth - 1) (by omega)
      have bitLt := list_index_lt_of_getElem?_eq_some bitAt
      omega
  have rawGatherOutcome := retained.witnessGatherFamilyOutcome (workflow := workflow)
    (certificate := certificate) verified stageResolved gatherBody gatherBodyResolved
    gatherDefinitionFound (Int.ofNat witnessWidth) gatherCountEvaluate
    (bits.map Mxx.Ir.Value.integer) indicesOutcome sourceOutcome
  have selectedValues :
      (List.range witnessWidth).map (fun index ↦
        (bits.map Mxx.Ir.Value.integer)[index]?.getD
          (.invalid "FamilyGetDynamic index out of range")) =
        (bits.take witnessWidth).map Mxx.Ir.Value.integer := by
    rw [map_range_getElem_getD_eq_take]
    · simp
    · simpa using widthLe
  have gatheredBits : retained.witnessBits.values =
      [.family ((bits.take witnessWidth).map Mxx.Ir.Value.integer)] := by
    calc
      retained.witnessBits.values = [.family ((List.range witnessWidth).map (fun index ↦
          (bits.map Mxx.Ir.Value.integer)[index]?.getD
            (.invalid "FamilyGetDynamic index out of range")))] := rawGatherOutcome
      _ = [.family ((bits.take witnessWidth).map Mxx.Ir.Value.integer)] := by
        rw [selectedValues]
  have widthFormula := verified.witnessDigitPackingFormula
  unfold verifyWitnessDigitPackingRef at widthFormula
  simp only [Bool.and_eq_true, decide_eq_true_eq] at widthFormula
  have outerCount : certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.count =
      Mxx.Ir.IntExpr.parameter "diamond_input_count" := by aesop
  have scanCount : certificate.decryptionInitialEncodings.witnessDigits.bitScan.count =
      Mxx.Ir.IntExpr.parameter "diamond_batch_bits" := by aesop
  have indexFormula := verified.witnessDigitIndexFormula
  have indexCount : certificate.decryptionInitialEncodings.witnessIndices.count =
      Mxx.Ir.IntExpr.multiply (.parameter "diamond_batch_bits")
        (.parameter "diamond_input_count") :=
    verifyDecryptionWitnessIndexFormula_count indexFormula
  have widthEq : witnessWidth = inputCount * batch := by
    have inputParamEvaluate :
        (Mxx.Ir.IntExpr.parameter "diamond_input_count").evaluate params =
          some (Int.ofNat inputCount) := by
      simpa only [outerCount] using packingCountEvaluate
    have batchParamEvaluate :
        (Mxx.Ir.IntExpr.parameter "diamond_batch_bits").evaluate params =
          some (Int.ofNat batch) := by
      simpa only [scanCount] using batchEvaluate
    have productEvaluate :
        (Mxx.Ir.IntExpr.multiply (.parameter "diamond_batch_bits")
          (.parameter "diamond_input_count")).evaluate params =
            some (Int.ofNat witnessWidth) := by
      simpa only [indexCount] using indexCountEvaluate
    change (do
      let left ← (Mxx.Ir.IntExpr.parameter "diamond_batch_bits").evaluate params
      let right ← (Mxx.Ir.IntExpr.parameter "diamond_input_count").evaluate params
      pure (left * right)) = some (Int.ofNat witnessWidth) at productEvaluate
    rw [batchParamEvaluate, inputParamEvaluate] at productEvaluate
    have productEqInt : Int.ofNat (batch * inputCount) = Int.ofNat witnessWidth := by
      simpa using productEvaluate
    have productEq : batch * inputCount = witnessWidth :=
      Int.ofNat_inj.mp productEqInt
    simpa [Nat.mul_comm] using productEq.symm
  have selectedBitsPresent : ∀ level ∈ List.range inputCount,
      ∀ index, index < batch →
        ∃ bit, (bits.take witnessWidth)[level * batch + index]? = some bit := by
    intro level levelMember index indexLt
    have levelLt : level < inputCount := by simpa using levelMember
    have positionLt : level * batch + index < witnessWidth := by
      calc
        level * batch + index < level * batch + batch := Nat.add_lt_add_left indexLt _
        _ = (level + 1) * batch := by simp [Nat.add_mul]
        _ ≤ inputCount * batch := Nat.mul_le_mul_right batch (Nat.succ_le_iff.mpr levelLt)
        _ = witnessWidth := widthEq.symm
    obtain ⟨bit, bitAt⟩ := bitsPresent _ positionLt
    exact ⟨bit, by simpa [List.getElem?_take, positionLt] using bitAt⟩
  have packedDigits := retained.witnessDigitPackingFamilyOutcome (workflow := workflow)
    (certificate := certificate) verified stageResolved outerBody scanBody outerBodyResolved
    outerDefinitionFound scanBodyResolved scanDefinitionFound runFuelAtLeastTwo
    (bits.take witnessWidth) batch gatheredBits (Int.ofNat inputCount) packingCountEvaluate
    batchEvaluate selectedBitsPresent
  exact { indices := indicesOutcome, gatheredBits, packedDigits }

/-- The packed witness family consumed by the online scan is the exact output of the retained
packing execution on the same decryption root path. -/
theorem RetainedInputInjectionExecutions.packedDigitsFinal
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {rootPath : RootStageExecutionPath samplers stage params inputs output}
    (retained : RetainedInputInjectionExecutions
      (workflow := workflow) (certificate := certificate) rootPath)
    (verified : VerifiedDiamondLayout workflow certificate)
    (packedValues : List Mxx.Ir.Value)
    (packedOutcome : retained.witnessDigits.values = [.family packedValues]) :
    Mxx.Ir.lookupWire (wireRef certificate.inputInjection.packedDigits.wire)
      rootPath.finalWires = some (.family packedValues) := by
  have packingFormula := verified.witnessDigitPackingFormula
  unfold verifyWitnessDigitPackingRef at packingFormula
  simp only [Bool.and_eq_true, decide_eq_true_eq] at packingFormula
  have outputsOne :
      certificate.decryptionInitialEncodings.witnessDigits.parallelLoop.outputs.length = 1 := by
    aesop
  obtain ⟨packingOutput, outputs⟩ := list_singleton_of_length_one _ outputsOne
  have linked := verified.inputInjectionExternalSources.2.2
  rw [outputs] at linked
  have outputEq : packingOutput = certificate.inputInjection.packedDigits.wire := by
    simpa using linked
  subst packingOutput
  have outputWire := parallelLoop_singleOutputWire
    verified.witnessDigitPackingParentVerified outputs
  have exactOutput := retained.witnessDigitsRooted.outputFinal 0 (by simp [packedOutcome])
  simpa [packedOutcome, outputWire] using exactOutput

/-- Invert the retained online state scan from the three exact values produced on the same root
path.  The caller identifies values already present in that root environment; this theorem moves
them to the scan's `before` environment using checked SSA order and therefore introduces no
detached semantic premise. -/
theorem RetainedInputInjectionExecutions.stateScanTrace_of_finalArguments
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {rootPath : RootStageExecutionPath samplers stage params inputs output}
    (retained : RetainedInputInjectionExecutions
      (workflow := workflow) (certificate := certificate) rootPath)
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved : resolveStage workflow certificate.workflow.decryption.stage = some stage)
    (initialStates packedDigits transitionFamily : Mxx.Ir.Value)
    (initialFinal : Mxx.Ir.lookupWire
      (wireRef certificate.inputInjection.initialStates.wire) rootPath.finalWires =
        some initialStates)
    (packedFinal : Mxx.Ir.lookupWire
      (wireRef certificate.inputInjection.packedDigits.wire) rootPath.finalWires =
        some packedDigits)
    (transitionFinal : Mxx.Ir.lookupWire
      (wireRef certificate.inputInjection.transitionFamily.wire) rootPath.finalWires =
        some transitionFamily)
    (evaluatedCount : Int)
    (countEvaluate :
      (Mxx.Ir.IntExpr.parameter "diamond_input_count").evaluate params =
        some evaluatedCount) :
    Nonempty (InputInjectionStateScanTrace workflow certificate.inputInjection
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs retained.scan) := by
  obtain ⟨resolution⟩ := verified.stateScanResolution
  have location := verified.inputInjectionStateScanLocation
  have rootSsa : verifyScopeSsaOrder stage.program.root = true := by
    have checked := verified.ssaOrderValid
    unfold verifyWorkflowSsaOrder at checked
    simp only [List.all_eq_true, Bool.and_eq_true] at checked
    exact (checked stage (resolveStage_mem stageResolved)).1
  have nodeEq : retained.scan.node = {
      kind := .sequentialLoop certificate.inputInjection.bodyScope.definitionName
        resolution.count resolution.indexSlot resolution.bindings 1
      arguments := [wireRef certificate.inputInjection.initialStates.wire,
        wireRef certificate.inputInjection.packedDigits.wire,
        wireRef certificate.inputInjection.transitionFamily.wire]
      outputCount := 1
    } := by
    have checkedResolved := resolution.resolved
    rw [retained.scan.resolved] at checkedResolved
    exact Option.some.inj checkedResolved
  have scanScopeResolved : resolveScope workflow certificate.inputInjection.stateScan =
      some stage.program.root := by
    simp [resolveScope, location.1, location.2, stageResolved, scopeOwnerMatches, rawScope]
  have scanAt := resolveNode_scopeNode scanScopeResolved retained.scan.resolved
  have scanLt := list_index_lt_of_getElem?_eq_some scanAt
  have scanRootNode : stage.program.root.nodes[certificate.inputInjection.stateScan.node] =
      retained.scan.node := by
    rw [List.getElem?_eq_getElem scanLt] at scanAt
    exact Option.some.inj scanAt
  have argumentBefore (argument : Mxx.Ir.WireRef) (value : Mxx.Ir.Value)
      (member : argument ∈ retained.scan.node.arguments)
      (finalLookup : Mxx.Ir.lookupWire argument rootPath.finalWires = some value) :
      Mxx.Ir.lookupWire argument retained.scan.before = some value := by
    apply retained.scanRooted.finalBefore argument value
    · apply verifyScopeSsaOrder_argument_lt rootSsa
        certificate.inputInjection.stateScan.node scanLt argument
      simpa [scanRootNode] using member
    · exact finalLookup
  have initialBefore := argumentBefore
    (wireRef certificate.inputInjection.initialStates.wire) initialStates (by simp [nodeEq])
    initialFinal
  have packedBefore := argumentBefore
    (wireRef certificate.inputInjection.packedDigits.wire) packedDigits (by simp [nodeEq])
    packedFinal
  have transitionBefore := argumentBefore
    (wireRef certificate.inputInjection.transitionFamily.wire) transitionFamily
    (by simp [nodeEq]) transitionFinal
  have argumentsEvaluate :
      [wireRef certificate.inputInjection.initialStates.wire,
        wireRef certificate.inputInjection.packedDigits.wire,
        wireRef certificate.inputInjection.transitionFamily.wire].mapM
          (fun wire ↦ Mxx.Ir.lookupWire wire retained.scan.before) =
        some [initialStates, packedDigits, transitionFamily] := by
    simp [initialBefore, packedBefore, transitionBefore]
  have resolutionCount : resolution.count =
      Mxx.Ir.IntExpr.parameter "diamond_input_count" := by
    have checked := verified.inputInjectionMatches
    unfold verifyInputInjection at checked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
    have exactRole : verifyExactSequentialNodeRole workflow
        certificate.inputInjection.stateScan (.parameter "diamond_input_count") 0 = true := by
      aesop
    unfold verifyExactSequentialNodeRole at exactRole
    rw [resolution.resolved] at exactRole
    simp only [decide_eq_true_eq, Bool.and_eq_true] at exactRole
    exact exactRole.1.1
  apply inputInjectionStateScanTrace_of_resolution resolution retained.scan
    [initialStates, packedDigits, transitionFamily] evaluatedCount argumentsEvaluate
  simpa [resolutionCount] using countEvaluate

end MxxWe.Certificate
