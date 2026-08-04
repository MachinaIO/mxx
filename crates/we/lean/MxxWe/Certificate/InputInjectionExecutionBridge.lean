import MxxWe.Certificate.ExecutionBridge

namespace MxxWe.Certificate

/-! Execution lifting dedicated to Diamond input injection.

This module intentionally sits after `ExecutionBridge`: the shared Boolean and decoder bridges
depend only on the frozen foundational API, while the protocol-specific family and recurrence
proofs can evolve here without invalidating those targets.
-/

private def injectionInputCount : Mxx.Ir.IntExpr := .parameter "diamond_input_count"
private def injectionBatchBits : Mxx.Ir.IntExpr := .parameter "diamond_batch_bits"
private def injectionDigitBase : Mxx.Ir.IntExpr := .parameter "diamond_digit_base"
private def injectionStateWidth : Mxx.Ir.IntExpr :=
  .add (.constant 1) (.multiply injectionBatchBits injectionInputCount)
private def injectionTransitionStride : Mxx.Ir.IntExpr :=
  .add (.multiply (.multiply injectionBatchBits injectionDigitBase) injectionInputCount)
    injectionDigitBase

private theorem indexFormula_loopAndOutput
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyParallelIndexFormulaRef workflow reference = true) :
    verifyParallelLoop workflow reference.parallelLoop = true ∧
      reference.parallelLoop.bodyOutputs = [reference.bodyOutput] := by
  unfold verifyParallelIndexFormulaRef at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  aesop

private theorem initialExpansion_loopAndOutput
    {workflow : Mxx.Ir.Workflow} {reference : InitialStateExpansionRef}
    (verified : verifyInitialStateExpansionRef workflow reference = true) :
    verifyParallelLoop workflow reference.parallelLoop = true ∧
      reference.parallelLoop.bodyOutputs = [reference.bodyOutput] := by
  unfold verifyInitialStateExpansionRef at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  aesop

private theorem preprocessingSource_baseCheck
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyPreprocessingSourceIndexFormula workflow reference = true) :
    verifyParallelIndexFormulaRef workflow reference = true := by
  unfold verifyPreprocessingSourceIndexFormula at verified
  simp only [Bool.and_eq_true] at verified
  aesop

private theorem preprocessingTarget_baseCheck
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyPreprocessingTargetIndexFormula workflow reference = true) :
    verifyParallelIndexFormulaRef workflow reference = true := by
  unfold verifyPreprocessingTargetIndexFormula at verified
  simp only [Bool.and_eq_true] at verified
  aesop

private theorem preprocessingDigit_baseCheck
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyPreprocessingDigitSecretIndexFormula workflow reference = true) :
    verifyParallelIndexFormulaRef workflow reference = true := by
  unfold verifyPreprocessingDigitSecretIndexFormula at verified
  simp only [Bool.and_eq_true] at verified
  aesop

private theorem onlineSource_baseCheck
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyOnlineSourceIndexFormula workflow reference = true) :
    verifyParallelIndexFormulaRef workflow reference = true := by
  unfold verifyOnlineSourceIndexFormula at verified
  simp only [Bool.and_eq_true] at verified
  aesop

private theorem onlineTransition_baseCheck
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyOnlineTransitionIndexFormula workflow reference = true) :
    verifyParallelIndexFormulaRef workflow reference = true := by
  unfold verifyOnlineTransitionIndexFormula at verified
  simp only [Bool.and_eq_true] at verified
  aesop

private def inputInjectionSameCoreScopeWire (context : CoreNodeRef)
    (wire : Mxx.Ir.WireRef) : CoreWireRef := {
  node := { context with node := wire.node }
  port := wire.port
}

private theorem inputInjectionConstantIntWire_resolution
    {workflow : Mxx.Ir.Workflow} {wire : CoreWireRef} {value : Int}
    (verified : verifyConstantIntWire workflow wire value = true) :
    wire.port = 0 ∧ resolveNode workflow wire.node = some {
      kind := .constantInt value
      arguments := []
      outputCount := 1
    } := by
  unfold verifyConstantIntWire at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  obtain ⟨port, verified⟩ := verified
  cases nodeResolved : resolveNode workflow wire.node with
  | none => simp [nodeResolved] at verified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> simp_all

/-
Structural resolution of the checked lower-bound arithmetic is intentionally proved next to
the same-child constructor.
private theorem inputInjectionOnlineSourceLowerBound_resolution
    {workflow : Mxx.Ir.Workflow} {wire : CoreWireRef}
    (verified : verifyOnlineSourceLowerBound workflow wire = true) :
    ∃ product one level width : CoreWireRef,
      wire.port = 0 ∧ product.port = 0 ∧ one.port = 0 ∧ level.port = 0 ∧
      width.port = 0 ∧
      product.node.stage = wire.node.stage ∧ product.node.scope = wire.node.scope ∧
      one.node.stage = wire.node.stage ∧ one.node.scope = wire.node.scope ∧
      level.node.stage = wire.node.stage ∧ level.node.scope = wire.node.scope ∧
      width.node.stage = wire.node.stage ∧ width.node.scope = wire.node.scope ∧
      resolveNode workflow wire.node = some {
        kind := .intBinary .add
        arguments := [wireRef product, wireRef one]
        outputCount := 1
      } ∧
      resolveNode workflow product.node = some {
        kind := .intBinary .multiply
        arguments := [wireRef level, wireRef width]
        outputCount := 1
      } ∧
      resolveNode workflow one.node = some {
        kind := .constantInt 1
        arguments := []
        outputCount := 1
      } ∧
      resolveNode workflow level.node = some {
        kind := .evaluateInt (.loopIndex 0)
        arguments := []
        outputCount := 1
      } ∧
      resolveNode workflow width.node = some {
        kind := .evaluateInt injectionBatchBits
        arguments := []
        outputCount := 1
      } := by
  unfold verifyOnlineSourceLowerBound at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  obtain ⟨wirePort, verified⟩ := verified
  have addShape : ∃ productWire oneWire,
      resolveNode workflow wire.node = some {
        kind := .intBinary .add, arguments := [productWire, oneWire], outputCount := 1
      } := by grind
  obtain ⟨productWire, oneWire, addResolved⟩ := addShape
  let product := inputInjectionSameCoreScopeWire wire.node productWire
  let one := inputInjectionSameCoreScopeWire wire.node oneWire
  have outerFacts : (product.port = 0 ∧ verifyConstantIntWire workflow one 1 = true) ∧
      (match resolveNode workflow product.node with
      | some productNode => match productNode.kind, productNode.arguments,
          productNode.outputCount with
        | .intBinary .multiply, [level, width], 1 =>
            let level := inputInjectionSameCoreScopeWire wire.node level
            let width := inputInjectionSameCoreScopeWire wire.node width
            decide (level.port = 0) && decide (width.port = 0) &&
              match resolveNode workflow level.node, resolveNode workflow width.node with
              | some levelNode, some widthNode =>
                  decide (levelNode.kind = .evaluateInt (.loopIndex 0)) &&
                  decide levelNode.arguments.isEmpty && decide (levelNode.outputCount = 1) &&
                  decide (widthNode.kind = .evaluateInt injectionBatchBits) &&
                  decide widthNode.arguments.isEmpty && decide (widthNode.outputCount = 1)
              | _, _ => false
        | _, _, _ => false
      | none => false) = true := by
    simpa [addResolved, product, one, inputInjectionSameCoreScopeWire, injectionBatchBits] using
      verified
  obtain ⟨⟨productPort, oneChecked⟩, productChecked⟩ := outerFacts
  obtain ⟨onePort, oneResolved⟩ := inputInjectionConstantIntWire_resolution oneChecked
  have productShape : ∃ levelWire widthWire,
      resolveNode workflow product.node = some {
        kind := .intBinary .multiply, arguments := [levelWire, widthWire], outputCount := 1
      } := by grind
  obtain ⟨levelWire, widthWire, productResolved⟩ := productShape
  let level := inputInjectionSameCoreScopeWire wire.node levelWire
  let width := inputInjectionSameCoreScopeWire wire.node widthWire
  have innerFacts : (level.port = 0 ∧ width.port = 0) ∧
      resolveNode workflow level.node = some {
        kind := .evaluateInt (.loopIndex 0), arguments := [], outputCount := 1
      } ∧
      resolveNode workflow width.node = some {
        kind := .evaluateInt injectionBatchBits, arguments := [], outputCount := 1
      } := by
    simp [productResolved, level, width, inputInjectionSameCoreScopeWire] at productChecked
    grind
  obtain ⟨⟨levelPort, widthPort⟩, levelResolved, widthResolved⟩ := innerFacts
  refine ⟨product, one, level, width, ?_⟩
  simp [product, one, level, width, inputInjectionSameCoreScopeWire, wireRef, wirePort,
    productPort, onePort, levelPort, widthPort, addResolved, productResolved, oneResolved,
    levelResolved, widthResolved]
-/

private theorem inputInjectionOnlineSourceIndexFormula_lowerBound
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyOnlineSourceIndexFormula workflow reference = true) :
    ∃ lowerBound : CoreOperandRef,
      reference.parallelLoop.arguments = [lowerBound] ∧
        verifyOnlineSourceLowerBound workflow lowerBound.wire = true := by
  unfold verifyOnlineSourceIndexFormula at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have lowerBoundChecked :
      (match reference.parallelLoop.arguments with
      | [lowerBound] => verifyOnlineSourceLowerBound workflow lowerBound.wire
      | _ => false) = true := by aesop
  cases argumentsEq : reference.parallelLoop.arguments with
  | nil => simp [argumentsEq] at lowerBoundChecked
  | cons lowerBound tail =>
      cases tail with
      | nil =>
          refine ⟨lowerBound, rfl, ?_⟩
          simpa [argumentsEq] using lowerBoundChecked
      | cons next rest => simp [argumentsEq] at lowerBoundChecked

/-- Evaluate one exact integer-expression node on a retained child path and keep the resulting
wire identity in the shared final environment. -/
private theorem ChildExecutionPath.evaluateIntOutputFinal
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value} {expression : Mxx.Ir.IntExpr} {value : Int}
    {arguments : List Mxx.Ir.WireRef} {outputCount : Nat}
    (path : ChildExecutionPath stage scope fuel samplers params inputs outputs)
    (scopeResolved : resolveScope workflow reference = some scope)
    (nodeResolved : resolveNode workflow reference = some {
      kind := .evaluateInt expression
      arguments
      outputCount
    })
    (evaluated : expression.evaluate params = some value) :
    Mxx.Ir.lookupWire ⟨reference.node, 0⟩ path.finalWires = some (.integer value) := by
  have nodeAt := resolveNode_scopeNode scopeResolved nodeResolved
  obtain ⟨execution, rooted⟩ := path.rootedReferencedNodeExecution nodeAt nodeResolved
  have valuesEq : execution.values = [.integer value] := by
    have member := execution.member
    rw [show execution.node = {
        kind := .evaluateInt expression
        arguments
        outputCount
      } by
        rw [execution.resolved] at nodeResolved
        exact Option.some.inj nodeResolved] at member
    simpa [Mxx.Ir.evaluateNode, evaluated] using member
  have final := rooted.outputFinal 0 (by simp [valuesEq])
  simpa [valuesEq] using final

/-- Evaluate one exact integer constant on a retained child path. -/
private theorem ChildExecutionPath.constantIntOutputFinal
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value} {value : Int}
    (path : ChildExecutionPath stage scope fuel samplers params inputs outputs)
    (scopeResolved : resolveScope workflow reference = some scope)
    (nodeResolved : resolveNode workflow reference = some {
      kind := .constantInt value
      arguments := []
      outputCount := 1
    }) :
    Mxx.Ir.lookupWire ⟨reference.node, 0⟩ path.finalWires = some (.integer value) := by
  have nodeAt := resolveNode_scopeNode scopeResolved nodeResolved
  obtain ⟨execution, rooted⟩ := path.rootedReferencedNodeExecution nodeAt nodeResolved
  have nodeEq : execution.node = {
      kind := .constantInt value
      arguments := []
      outputCount := 1
    } := by
    rw [execution.resolved] at nodeResolved
    exact Option.some.inj nodeResolved
  have valuesEq : execution.values = [.integer value] := by
    have member := execution.member
    rw [nodeEq] at member
    simpa [Mxx.Ir.evaluateNode] using member
  simpa [valuesEq] using rooted.outputFinal 0 (by simp [valuesEq])

/-- Evaluate an exact integer binary node from operand values already established on the same
retained SSA path. -/
private theorem ChildExecutionPath.intBinaryOutputFinal
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value} {operation : Mxx.Ir.IntBinaryOp}
    {left right : Mxx.Ir.WireRef} {leftValue rightValue result : Int}
    (path : ChildExecutionPath stage scope fuel samplers params inputs outputs)
    (scopeResolved : resolveScope workflow reference = some scope)
    (ssaOrder : verifyScopeSsaOrder scope = true)
    (nodeResolved : resolveNode workflow reference = some {
      kind := .intBinary operation
      arguments := [left, right]
      outputCount := 1
    })
    (leftFinal : Mxx.Ir.lookupWire left path.finalWires = some (.integer leftValue))
    (rightFinal : Mxx.Ir.lookupWire right path.finalWires = some (.integer rightValue))
    (evaluated : Mxx.Ir.evaluateIntBinary operation leftValue rightValue = some result) :
    Mxx.Ir.lookupWire ⟨reference.node, 0⟩ path.finalWires = some (.integer result) := by
  have nodeAt := resolveNode_scopeNode scopeResolved nodeResolved
  obtain ⟨execution, rooted⟩ := path.rootedReferencedNodeExecution nodeAt nodeResolved
  have nodeEq : execution.node = {
      kind := .intBinary operation
      arguments := [left, right]
      outputCount := 1
    } := by
    rw [execution.resolved] at nodeResolved
    exact Option.some.inj nodeResolved
  have leftBefore := rooted.argumentFromFinal ssaOrder scopeResolved left (by simp [nodeEq])
    leftFinal
  have rightBefore := rooted.argumentFromFinal ssaOrder scopeResolved right (by simp [nodeEq])
    rightFinal
  have valuesEq : execution.values = [.integer result] := by
    have member := execution.member
    rw [nodeEq] at member
    simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, leftBefore, rightBefore, evaluated] using member
  simpa [valuesEq] using rooted.outputFinal 0 (by simp [valuesEq])

/-- Recover a structural body input by position, without exposing its generated local name. -/
private theorem ChildExecutionPath.inputWireValueAt
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value}
    (path : ChildExecutionPath stage scope fuel samplers params inputs outputs)
    (namesNodup : scope.inputNames.Nodup)
    (wireLength : (scopeInputWires scope).length = scope.inputNames.length)
    (inputLength : scope.inputNames.length = inputs.length)
    (index : Nat) (indexLt : index < scope.inputNames.length)
    (wire : Mxx.Ir.WireRef) (value : Mxx.Ir.Value)
    (wireAt : (scopeInputWires scope)[index]? = some wire)
    (valueAt : inputs[index]? = some value)
    (wireValid : ∃ node, scope.nodes[wire.node]? = some node ∧ wire.port < node.outputCount) :
    Mxx.Ir.lookupWire wire path.finalWires = some value := by
  let name := scope.inputNames[index]
  apply path.inputWireValue namesNodup wireLength inputLength index name wire value
  · simp [name, List.getElem?_eq_getElem indexLt]
  · exact wireAt
  · exact valueAt
  · exact wireValid

/-- The input-injection verifier ties the dynamic lookup index to the exact loop-index expression
of the surrounding state scan. -/
private theorem inputInjectionSelectedIndexNode
    {workflow : Mxx.Ir.Workflow} {layout : InputInjectionLayout}
    (verified : verifyInputInjection workflow layout = true)
    (resolution : InputInjectionStateScanResolution workflow layout) :
    resolveNode workflow layout.selectedDigit.index.wire.node = some {
      kind := .evaluateInt (.loopIndex resolution.indexSlot)
      arguments := []
      outputCount := 1
    } ∧ layout.selectedDigit.index.wire.port = 0 := by
  have selectedMatch :
      (match resolveNode workflow layout.stateScan,
          resolveNode workflow layout.selectedDigit.index.wire.node with
      | some { kind := .sequentialLoop _ _ indexSlot _ _, .. },
          some { kind := .evaluateInt (.loopIndex selectedSlot), arguments, outputCount } =>
          decide (selectedSlot = indexSlot) && decide arguments.isEmpty &&
            decide (outputCount = 1) && decide (layout.selectedDigit.index.wire.port = 0)
      | _, _ => false) = true := by
    unfold verifyInputInjection at verified
    simp only [Bool.and_eq_true] at verified
    exact verified.1.2
  clear verified
  rw [resolution.resolved] at selectedMatch
  cases indexResolved : resolveNode workflow layout.selectedDigit.index.wire.node with
  | none => simp [indexResolved] at selectedMatch
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> simp_all
      rename_i value
      cases value <;> simp_all

/-- Structural facts for the three state-scan body inputs, recovered from the accepted loop-body
mapping rather than from generated local names. -/
private theorem inputInjectionBodyInputFacts
    {workflow : Mxx.Ir.Workflow} {layout : InputInjectionLayout} {body : Mxx.Ir.Scope}
    (verified : verifyInputInjection workflow layout = true)
    (bodyResolved : resolveScope workflow { layout.stateScan with scope := layout.bodyScope } =
      some body) :
    body.inputNames.Nodup ∧
      (scopeInputWires body).length = body.inputNames.length ∧
      body.inputNames.length = 3 ∧
      [wireRef layout.bodyInitialStates, wireRef layout.bodyPackedDigits,
        wireRef layout.bodyTransitionFamily] = scopeInputWires body := by
  have loopBody : verifyLoopBody workflow layout.stateScan layout.bodyScope
      [layout.initialStates, layout.packedDigits, layout.transitionFamily]
      [layout.bodyInitialStates, layout.bodyPackedDigits, layout.bodyTransitionFamily]
      [layout.bodyFinalStates] [layout.finalStates] = true := by
    unfold verifyInputInjection at verified
    simp only [Bool.and_eq_true] at verified
    aesop
  unfold verifyLoopBody at loopBody
  rw [bodyResolved] at loopBody
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_cons, List.all_nil, and_true] at loopBody
  have wires :
      [wireRef layout.bodyInitialStates, wireRef layout.bodyPackedDigits,
        wireRef layout.bodyTransitionFamily] = scopeInputWires body := by aesop
  refine ⟨by aesop, by aesop, ?_, wires⟩
  have lengths := congrArg List.length wires
  symm
  simpa using (lengths.trans (by aesop : (scopeInputWires body).length = body.inputNames.length))

/-- The three semantic scan arguments are the exact three structural input wires on the retained
child path. -/
private theorem InputInjectionIterationExecutions.bodyInputFinalValues
    {workflow : Mxx.Ir.Workflow} {layout : InputInjectionLayout}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {childInputs : List Mxx.Ir.Value}
    (executions : InputInjectionIterationExecutions workflow layout stage scope fuel samplers
      params childInputs)
    (verified : verifyInputInjection workflow layout = true)
    (bodyResolved : resolveScope workflow { layout.stateScan with scope := layout.bodyScope } =
      some scope)
    (initialStates packedDigits transitionFamily : List Mxx.Ir.Value)
    (childInputsEq : childInputs =
      [.family initialStates, .family packedDigits, .family transitionFamily]) :
    Mxx.Ir.lookupWire (wireRef layout.bodyInitialStates) executions.final =
        some (.family initialStates) ∧
      Mxx.Ir.lookupWire (wireRef layout.bodyPackedDigits) executions.final =
        some (.family packedDigits) ∧
      Mxx.Ir.lookupWire (wireRef layout.bodyTransitionFamily) executions.final =
        some (.family transitionFamily) := by
  let path : ChildExecutionPath stage scope fuel samplers params childInputs executions.outputs := {
    finalWires := executions.final
    path := executions.path
    outputs := executions.outputsEq
  }
  obtain ⟨namesNodup, wireLength, inputLength, inputWires⟩ :=
    inputInjectionBodyInputFacts verified bodyResolved
  have checked := verified
  unfold verifyInputInjection at checked
  simp only [Bool.and_eq_true] at checked
  have loopBody : verifyLoopBody workflow layout.stateScan layout.bodyScope
      [layout.initialStates, layout.packedDigits, layout.transitionFamily]
      [layout.bodyInitialStates, layout.bodyPackedDigits, layout.bodyTransitionFamily]
      [layout.bodyFinalStates] [layout.finalStates] = true := by aesop
  unfold verifyLoopBody at loopBody
  rw [bodyResolved] at loopBody
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_cons, List.all_nil, and_true] at loopBody
  clear checked verified
  have inputPairs := loopBody.1.1.1.2
  clear loopBody
  simp at inputPairs
  have initialChecked : verifyWire workflow layout.bodyInitialStates = true := by aesop
  have packedChecked : verifyWire workflow layout.bodyPackedDigits = true := by aesop
  have transitionChecked : verifyWire workflow layout.bodyTransitionFamily = true := by aesop
  have initialStage : layout.bodyInitialStates.node.stage = layout.stateScan.stage := by aesop
  have initialScopeRef : layout.bodyInitialStates.node.scope = layout.bodyScope := by aesop
  have packedStage : layout.bodyPackedDigits.node.stage = layout.stateScan.stage := by aesop
  have packedScopeRef : layout.bodyPackedDigits.node.scope = layout.bodyScope := by aesop
  have transitionStage : layout.bodyTransitionFamily.node.stage = layout.stateScan.stage := by aesop
  have transitionScopeRef : layout.bodyTransitionFamily.node.scope = layout.bodyScope := by aesop
  have initialScope : resolveScope workflow layout.bodyInitialStates.node = some scope := by
    simpa [resolveScope, initialStage, initialScopeRef] using bodyResolved
  have packedScope : resolveScope workflow layout.bodyPackedDigits.node = some scope := by
    simpa [resolveScope, packedStage, packedScopeRef] using bodyResolved
  have transitionScope : resolveScope workflow layout.bodyTransitionFamily.node = some scope := by
    simpa [resolveScope, transitionStage, transitionScopeRef] using bodyResolved
  refine ⟨?_, ?_, ?_⟩
  · apply path.inputWireValueAt namesNodup wireLength (by simpa [childInputsEq] using inputLength)
      0 (by omega) (wireRef layout.bodyInitialStates) (.family initialStates)
    · rw [← inputWires]
      simp
    · simp [childInputsEq]
    · exact verifyWire_scopeValid initialChecked initialScope
  · apply path.inputWireValueAt namesNodup wireLength (by simpa [childInputsEq] using inputLength)
      1 (by omega) (wireRef layout.bodyPackedDigits) (.family packedDigits)
    · rw [← inputWires]
      simp
    · simp [childInputsEq]
    · exact verifyWire_scopeValid packedChecked packedScope
  · apply path.inputWireValueAt namesNodup wireLength (by simpa [childInputsEq] using inputLength)
      2 (by omega) (wireRef layout.bodyTransitionFamily) (.family transitionFamily)
    · rw [← inputWires]
      simp
    · simp [childInputsEq]
    · exact verifyWire_scopeValid transitionChecked transitionScope

/-- The selected digit in one retained scan child is the exact packed digit at the surrounding
sequential-loop level. -/
theorem InputInjectionIterationExecutions.selectedDigitOutcome
    {workflow : Mxx.Ir.Workflow} {layout : InputInjectionLayout}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {childInputs : List Mxx.Ir.Value}
    (executions : InputInjectionIterationExecutions workflow layout stage scope fuel samplers
      params childInputs)
    (verified : verifyInputInjection workflow layout = true)
    (bodyResolved : resolveScope workflow { layout.stateScan with scope := layout.bodyScope } =
      some scope)
    (ssaOrder : verifyScopeSsaOrder scope = true)
    (resolution : InputInjectionStateScanResolution workflow layout)
    (initialStates packedDigits transitionFamily : List Mxx.Ir.Value)
    (childInputsEq : childInputs =
      [.family initialStates, .family packedDigits, .family transitionFamily])
    (level : Nat) (selectedDigit : Int)
    (levelEvaluate : (Mxx.Ir.IntExpr.loopIndex resolution.indexSlot).evaluate params =
      some (Int.ofNat level))
    (packedAt : packedDigits[level]? = some (.integer selectedDigit)) :
    executions.selectedDigit.execution.values = [.integer selectedDigit] := by
  let path : ChildExecutionPath stage scope fuel samplers params childInputs executions.outputs := {
    finalWires := executions.final
    path := executions.path
    outputs := executions.outputsEq
  }
  obtain ⟨_, packedFinal, _⟩ := executions.bodyInputFinalValues verified bodyResolved
    initialStates packedDigits transitionFamily childInputsEq
  have selectedVerified : verifyDynamicGet workflow layout.selectedDigit
      layout.bodyPackedDigits = true := by
    unfold verifyInputInjection at verified
    simp only [Bool.and_eq_true] at verified
    aesop
  obtain ⟨selectedResolution⟩ :=
    checkedDynamicFamilyGetResolution_of_verified selectedVerified
  obtain ⟨indexNode, indexPort⟩ := inputInjectionSelectedIndexNode verified resolution
  have selectedScope : resolveScope workflow layout.selectedDigit.operation = some scope := by
    have checked := verified
    unfold verifyInputInjection at checked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
    unfold resolveScope at bodyResolved ⊢
    have selectedStage : layout.selectedDigit.operation.stage = layout.stateScan.stage := by aesop
    have selectedBody : layout.selectedDigit.operation.scope = layout.bodyScope := by aesop
    simpa [selectedStage, selectedBody] using bodyResolved
  have indexOwner :
      layout.selectedDigit.index.wire.node.stage = layout.selectedDigit.operation.stage ∧
        layout.selectedDigit.index.wire.node.scope = layout.selectedDigit.operation.scope := by
    have checked := selectedVerified
    unfold verifyDynamicGet verifyOperand at checked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
    aesop
  have indexScope : resolveScope workflow layout.selectedDigit.index.wire.node = some scope := by
    unfold resolveScope at selectedScope ⊢
    simpa [indexOwner.1, indexOwner.2] using selectedScope
  have indexFinalRaw := path.evaluateIntOutputFinal indexScope indexNode levelEvaluate
  have indexFinal : Mxx.Ir.lookupWire (wireRef layout.selectedDigit.index.wire)
      executions.final = some (.integer (Int.ofNat level)) := by
    simpa [wireRef, indexPort] using indexFinalRaw
  have selectedNodeAt := resolveNode_scopeNode selectedScope selectedResolution.resolved
  have selectedLt := list_index_lt_of_getElem?_eq_some selectedNodeAt
  have selectedNode : scope.nodes[layout.selectedDigit.operation.node] =
      executions.selectedDigit.execution.node :=
    list_getElem_eq_of_getElem?_eq_some
      (resolveNode_scopeNode selectedScope executions.selectedDigit.execution.resolved) selectedLt
  have familyPast : (wireRef layout.selectedDigit.family.wire).node <
      layout.selectedDigit.operation.node := by
    apply verifyScopeSsaOrder_argument_lt ssaOrder layout.selectedDigit.operation.node selectedLt
    simpa [selectedNode] using (show wireRef layout.selectedDigit.family.wire ∈
      executions.selectedDigit.execution.node.arguments by
        have exactNode := selectedResolution.resolved
        rw [executions.selectedDigit.execution.resolved] at exactNode
        have nodeEq := Option.some.inj exactNode
        simp [nodeEq])
  have indexPast : (wireRef layout.selectedDigit.index.wire).node <
      layout.selectedDigit.operation.node := by
    apply verifyScopeSsaOrder_argument_lt ssaOrder layout.selectedDigit.operation.node selectedLt
    simpa [selectedNode] using (show wireRef layout.selectedDigit.index.wire ∈
      executions.selectedDigit.execution.node.arguments by
        have exactNode := selectedResolution.resolved
        rw [executions.selectedDigit.execution.resolved] at exactNode
        have nodeEq := Option.some.inj exactNode
        simp [nodeEq])
  have familyBefore := executions.selectedDigit.finalBefore familyPast (by
    simpa [show layout.selectedDigit.family.wire = layout.bodyPackedDigits by
      unfold verifyDynamicGet at selectedVerified
      simp only [Bool.and_eq_true, decide_eq_true_eq] at selectedVerified
      aesop] using packedFinal)
  have indexBefore := executions.selectedDigit.finalBefore indexPast indexFinal
  have argumentsEvaluate :
      [wireRef layout.selectedDigit.family.wire,
        wireRef layout.selectedDigit.index.wire].mapM
          (fun wire => Mxx.Ir.lookupWire wire executions.selectedDigit.execution.before) =
        some [.family packedDigits, .integer (Int.ofNat level)] := by
    simp [familyBefore, indexBefore]
  have outcome := checkedDynamicFamilyGetOutcome selectedResolution
    executions.selectedDigit.execution packedDigits (Int.ofNat level) argumentsEvaluate
  simpa [packedAt] using outcome

/-! ## Witness packing semantics -/

/-- Little-endian integer represented by `count` consecutive witness bits.  This is the pure
counterpart of the fixed inner sequential loop used by witness-digit packing. -/
def packedWitnessPrefix (bits : List Int) (offset : Nat) : Nat → Int
  | 0 => 0
  | count + 1 =>
      packedWitnessPrefix bits offset count +
        bits[offset + count]?.getD 0 * Int.ofNat (2 ^ count)

/-- Canonical source bits make every packed prefix a nonnegative integer strictly below its
power-of-two radix. -/
theorem packedWitnessPrefix_bounds
    {bits : List Int} {offset count : Nat}
    (canonical : ∀ bit, bit < count →
      bits[offset + bit]?.getD 0 = 0 ∨ bits[offset + bit]?.getD 0 = 1) :
    0 ≤ packedWitnessPrefix bits offset count ∧
      packedWitnessPrefix bits offset count < Int.ofNat (2 ^ count) := by
  induction count with
  | zero => simp [packedWitnessPrefix]
  | succ count induction =>
      have prefixCanonical : ∀ bit, bit < count →
          bits[offset + bit]?.getD 0 = 0 ∨ bits[offset + bit]?.getD 0 = 1 := by
        intro bit bitLt
        exact canonical bit (by omega)
      have prefixBounds := induction prefixCanonical
      have lastCanonical := canonical count (by omega)
      have castPower : Int.ofNat (2 ^ count) = (2 : Int) ^ count := by simp
      have castPowerSucc : Int.ofNat (2 ^ (count + 1)) = (2 : Int) ^ (count + 1) := by simp
      rw [castPower] at prefixBounds
      rw [packedWitnessPrefix]
      rw [castPowerSucc, pow_succ]
      rcases lastCanonical with lastZero | lastOne
      · rw [lastZero]
        have powerPositive : 0 < (2 : Int) ^ count := pow_pos (by norm_num) count
        constructor <;> norm_num <;> nlinarith
      · rw [lastOne]
        have powerPositive : 0 < (2 : Int) ^ count := pow_pos (by norm_num) count
        constructor <;> norm_num <;> nlinarith

/-- A canonical witness split into `inputCount` little-endian blocks yields a legal digit at
every input level.  The upper bound is strong enough for any configured digit base satisfying
`2 ^ batchBits ≤ digitBase`. -/
theorem packedWitnessDigit_bounds_of_canonical
    {bits : List Int} {witnessWidth inputCount batchBits digitBase level : Nat}
    (canonical : ∀ slot, slot < witnessWidth →
      bits[slot]?.getD 0 = 0 ∨ bits[slot]?.getD 0 = 1)
    (widthEq : witnessWidth = inputCount * batchBits)
    (levelLt : level < inputCount)
    (radixLe : 2 ^ batchBits ≤ digitBase) :
    0 ≤ packedWitnessPrefix bits (level * batchBits) batchBits ∧
      packedWitnessPrefix bits (level * batchBits) batchBits < Int.ofNat digitBase := by
  have blockCanonical : ∀ bit, bit < batchBits →
      bits[level * batchBits + bit]?.getD 0 = 0 ∨
        bits[level * batchBits + bit]?.getD 0 = 1 := by
    intro bit bitLt
    apply canonical
    rw [widthEq]
    calc
      level * batchBits + bit < level * batchBits + batchBits :=
        Nat.add_lt_add_left bitLt _
      _ = (level + 1) * batchBits := by simp [Nat.add_mul]
      _ ≤ inputCount * batchBits :=
        Nat.mul_le_mul_right batchBits (Nat.succ_le_iff.mpr levelLt)
  have bounds := packedWitnessPrefix_bounds blockCanonical
  exact ⟨bounds.1, lt_of_lt_of_le bounds.2 (Int.ofNat_le.mpr radixLe)⟩

/-- One child of the checked witness-index loop returns its loop index unchanged.  The proof
executes the verifier's fixed three-node body; callers do not supply an index equation. -/
theorem decryptionWitnessIndex_childOutcome
    {workflow : Mxx.Ir.Workflow} {layout : DecryptionInitialEncodingsLayout}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value} {index : Nat}
    (verified : verifyDecryptionWitnessIndexFormula workflow layout = true)
    (loopVerified : verifyParallelLoop workflow layout.witnessIndices = true)
    (bodyResolved : resolveScope workflow
      { layout.witnessIndices.operation with scope := layout.witnessIndices.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition
      layout.witnessIndices.bodyScope.definitionName stage.program.definitions = some body)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      layout.witnessIndices.bodyScope.definitionName
      ((.loopIndex layout.witnessIndices.indexSlot, .integer index) :: params) inputs)
    (slotZero : layout.witnessIndices.indexSlot = 0) :
    values = [.integer (Int.ofNat index)] := by
  have checked := verified
  unfold verifyDecryptionWitnessIndexFormula at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have outputDescription := verified.1.2
  cases outputs : layout.witnessIndices.bodyOutputs with
  | nil => simp [outputs] at outputDescription
  | cons bodyOutput tail =>
      cases tail with
      | cons next rest => simp [outputs] at outputDescription
      | nil =>
          apply parallelLoopSingleBodyOutput_of_childOutcome loopVerified outputs bodyResolved
            definitionFound childMember
          intro final path
          apply verifyDecryptionWitnessIndexFormula_pathOutput checked outputs bodyResolved path
          simp [slotZero, Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupLoopIndex]

/-- The complete checked witness-index loop produces the ordered family `0, ..., count - 1`.
This family is the exact index input subsequently consumed by the witness gather. -/
theorem decryptionWitnessIndex_familyOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {runFuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {execution : ReferencedNodeExecution workflow
      certificate.decryptionInitialEncodings.witnessIndices.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program runFuel) samplers params inputs}
    (verified : VerifiedDiamondLayout workflow certificate)
    (trace : CheckedParallelLoopTrace workflow
      certificate.decryptionInitialEncodings.witnessIndices
      (Mxx.Ir.childRunnerWithFuel samplers stage.program runFuel) samplers params inputs
      execution)
    (bodyResolved : resolveScope workflow
      { certificate.decryptionInitialEncodings.witnessIndices.operation with
        scope := certificate.decryptionInitialEncodings.witnessIndices.bodyScope } = some body)
    (definitionFound : Mxx.Ir.lookupDefinition
      certificate.decryptionInitialEncodings.witnessIndices.bodyScope.definitionName
      stage.program.definitions = some body)
    (runFuelPositive : 0 < runFuel) :
    execution.values = [.family ((List.range trace.evaluatedCount.toNat).map fun index =>
      .integer (Int.ofNat index))] := by
  obtain ⟨fuel, rfl⟩ :=
    Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt runFuelPositive)
  obtain ⟨loopVerified, slotZero, bindingsEmpty, modesEmpty, _bodyOutputsOne, outputsOne⟩ :=
    verified.witnessIndexParentFacts
  apply checkedParallelLoop_onePortFamily trace outputsOne
    (fun index => .integer (Int.ofNat index))
  intro index evaluatedBindings childValues bindingsEvaluate childMember
  rw [bindingsEmpty] at bindingsEvaluate
  simp only [Mxx.Ir.evaluateBindings] at bindingsEvaluate
  have evaluatedBindingsEmpty : evaluatedBindings = [] :=
    Option.some.inj bindingsEvaluate.symm
  subst evaluatedBindings
  apply decryptionWitnessIndex_childOutcome verified.witnessDigitIndexFormula loopVerified
    bodyResolved definitionFound childMember slotZero

/-- One preprocessing source-index child execution returns the checked source formula. -/
theorem preprocessingSourceIndex_childOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value} {index stride width batch : Int}
    (verified : verifyPreprocessingSourceIndexFormula workflow reference = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.parallelLoop.bodyScope.definitionName
      stage.program.definitions = some body)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.parallelLoop.bodyScope.definitionName params inputs)
    (indexEvaluate : (Mxx.Ir.IntExpr.loopIndex 0).evaluate params = some index)
    (strideEvaluate : injectionTransitionStride.evaluate params = some stride)
    (widthEvaluate : injectionStateWidth.evaluate params = some width)
    (batchEvaluate : injectionBatchBits.evaluate params = some batch)
    (strideNonzero : stride ≠ 0) (widthNonzero : width ≠ 0) :
    values = [.integer ((index / stride) * width +
      if (index / stride) * batch + 1 ≤ index % width then 0 else index % width)] := by
  have base := preprocessingSource_baseCheck verified
  have checked := indexFormula_loopAndOutput base
  apply parallelLoopSingleBodyOutput_of_childOutcome checked.1 checked.2 bodyResolved
    definitionFound childMember
  intro final path
  exact verifyPreprocessingSourceIndexFormula_pathOutput verified bodyResolved path indexEvaluate
    strideEvaluate widthEvaluate batchEvaluate strideNonzero widthNonzero

/-- One preprocessing target-index child execution returns the checked target formula. -/
theorem preprocessingTargetIndex_childOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value} {index stride width : Int}
    (verified : verifyPreprocessingTargetIndexFormula workflow reference = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.parallelLoop.bodyScope.definitionName
      stage.program.definitions = some body)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.parallelLoop.bodyScope.definitionName params inputs)
    (indexEvaluate : (Mxx.Ir.IntExpr.loopIndex 0).evaluate params = some index)
    (strideEvaluate : injectionTransitionStride.evaluate params = some stride)
    (widthEvaluate : injectionStateWidth.evaluate params = some width)
    (strideNonzero : stride ≠ 0) (widthNonzero : width ≠ 0) :
    values = [.integer ((index / stride + 1) * width + index % width)] := by
  have checked := indexFormula_loopAndOutput (preprocessingTarget_baseCheck verified)
  apply parallelLoopSingleBodyOutput_of_childOutcome checked.1 checked.2 bodyResolved
    definitionFound childMember
  intro final path
  exact verifyPreprocessingTargetIndexFormula_pathOutput verified bodyResolved path indexEvaluate
    strideEvaluate widthEvaluate strideNonzero widthNonzero

/-- One preprocessing digit-secret child execution returns the checked quotient. -/
theorem preprocessingDigitSecretIndex_childOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value} {index width : Int}
    (verified : verifyPreprocessingDigitSecretIndexFormula workflow reference = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.parallelLoop.bodyScope.definitionName
      stage.program.definitions = some body)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.parallelLoop.bodyScope.definitionName params inputs)
    (indexEvaluate : (Mxx.Ir.IntExpr.loopIndex 0).evaluate params = some index)
    (widthEvaluate : injectionStateWidth.evaluate params = some width)
    (widthNonzero : width ≠ 0) : values = [.integer (index / width)] := by
  have checked := indexFormula_loopAndOutput (preprocessingDigit_baseCheck verified)
  apply parallelLoopSingleBodyOutput_of_childOutcome checked.1 checked.2 bodyResolved
    definitionFound childMember
  intro final path
  exact verifyPreprocessingDigitSecretIndexFormula_pathOutput verified bodyResolved path
    indexEvaluate widthEvaluate widthNonzero

/-- One online source-index child execution returns the checked masked local index. -/
theorem onlineSourceIndex_childOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value} {lowerBound index : Int}
    (verified : verifyOnlineSourceIndexFormula workflow reference = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.parallelLoop.bodyScope.definitionName
      stage.program.definitions = some body)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.parallelLoop.bodyScope.definitionName params inputs)
    (lowerBoundInput : Mxx.Ir.lookupEnvironment "__capture_0"
      (body.inputNames.zip inputs) = some (.integer lowerBound))
    (indexEvaluate : (Mxx.Ir.IntExpr.loopIndex 1).evaluate params = some index) :
    values = [.integer (if lowerBound ≤ index then 0 else index)] := by
  have checked := indexFormula_loopAndOutput (onlineSource_baseCheck verified)
  apply parallelLoopSingleBodyOutput_of_childOutcome checked.1 checked.2 bodyResolved
    definitionFound childMember
  intro final path
  exact verifyOnlineSourceIndexFormula_pathOutput verified bodyResolved path lowerBoundInput
    indexEvaluate

/-- One online transition-index child execution returns the checked flattened index. -/
theorem onlineTransitionIndex_childOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value} {level digit stride width index : Int}
    (verified : verifyOnlineTransitionIndexFormula workflow reference = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.parallelLoop.bodyScope.definitionName
      stage.program.definitions = some body)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.parallelLoop.bodyScope.definitionName params inputs)
    (levelInput : Mxx.Ir.lookupEnvironment "__capture_0" (body.inputNames.zip inputs) =
      some (.integer level))
    (digitInput : Mxx.Ir.lookupEnvironment "__capture_1" (body.inputNames.zip inputs) =
      some (.integer digit))
    (strideEvaluate : injectionTransitionStride.evaluate params = some stride)
    (widthEvaluate : injectionStateWidth.evaluate params = some width)
    (indexEvaluate : (Mxx.Ir.IntExpr.loopIndex 1).evaluate params = some index) :
    values = [.integer (level * stride + digit * width + index)] := by
  have checked := indexFormula_loopAndOutput (onlineTransition_baseCheck verified)
  apply parallelLoopSingleBodyOutput_of_childOutcome checked.1 checked.2 bodyResolved
    definitionFound childMember
  intro final path
  exact verifyOnlineTransitionIndexFormula_pathOutput verified bodyResolved path levelInput
    digitInput strideEvaluate widthEvaluate indexEvaluate

/-- One initial-state expansion child execution returns the initial matrix at index zero and the
exact zero matrix elsewhere. -/
theorem initialStateExpansion_childOutcome
    {workflow : Mxx.Ir.Workflow} {reference : InitialStateExpansionRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value} {index : Int} {initial : Mxx.Matrix}
    {matrixParams : Mxx.SamplerParams}
    (verified : verifyInitialStateExpansionRef workflow reference = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.parallelLoop.bodyScope.definitionName
      stage.program.definitions = some body)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.parallelLoop.bodyScope.definitionName params inputs)
    (indexEvaluate : (Mxx.Ir.IntExpr.loopIndex 0).evaluate params = some index)
    (initialInput : Mxx.Ir.lookupEnvironment "__capture_0" (body.inputNames.zip inputs) =
      some (.matrix initial))
    (matrixTypeEvaluate :
      ({ modulus := .parameter "diamond_modulus"
         ringDimension := .parameter "diamond_ring_dimension"
         rows := .constant 1
         columns := .add (.constant 4) (.multiply (.constant 2)
           (.parameter "diamond_digit_count")) } : Mxx.Ir.MatrixTypeExpr).evaluate params =
        some matrixParams) :
    values = [.matrix (if index = 0 then initial else
      Mxx.Matrix.withSamplerParams {
        coefficients := List.replicate
          (matrixParams.rows * matrixParams.columns * matrixParams.ringDimension) 0
      } matrixParams)] := by
  have checked := initialExpansion_loopAndOutput verified
  apply parallelLoopSingleBodyOutput_of_childOutcome checked.1 checked.2 bodyResolved
    definitionFound childMember
  intro final path
  exact verifyInitialStateExpansionRef_pathOutput verified bodyResolved path indexEvaluate
    initialInput matrixTypeEvaluate

private theorem preprocessingDigit_parentFacts
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyPreprocessingDigitSecretIndexFormula workflow reference = true) :
    reference.parallelLoop.indexSlot = 0 ∧ reference.parallelLoop.bindings = [] ∧
      reference.parallelLoop.outputs.length = 1 := by
  unfold verifyPreprocessingDigitSecretIndexFormula verifyParallelIndexFormulaRef at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  aesop

/-- The full preprocessing digit-index parent loop returns the ordered checked quotient family. -/
theorem preprocessingDigitSecretIndex_familyOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {execution : ReferencedNodeExecution workflow reference.parallelLoop.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs}
    {width : Int}
    (verified : verifyPreprocessingDigitSecretIndexFormula workflow reference = true)
    (trace : CheckedParallelLoopTrace workflow reference.parallelLoop
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs
      execution)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.parallelLoop.bodyScope.definitionName
      stage.program.definitions = some body)
    (widthEvaluate : injectionStateWidth.evaluate params = some width)
    (widthNonzero : width ≠ 0) :
    execution.values = [.family ((List.range trace.evaluatedCount.toNat).map fun index =>
      .integer (Int.ofNat index / width))] := by
  obtain ⟨slotZero, bindingsEmpty, outputsOne⟩ := preprocessingDigit_parentFacts verified
  apply checkedParallelLoop_onePortFamily trace outputsOne
    (fun index => .integer (Int.ofNat index / width))
  intro index evaluatedBindings childValues bindingsEvaluate childMember
  rw [bindingsEmpty] at bindingsEvaluate
  simp only [Mxx.Ir.evaluateBindings] at bindingsEvaluate
  have evaluatedBindingsEmpty : evaluatedBindings = [] :=
    Option.some.inj bindingsEvaluate.symm
  subst evaluatedBindings
  apply preprocessingDigitSecretIndex_childOutcome verified bodyResolved definitionFound childMember
  · simp [slotZero, Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupLoopIndex]
  · simpa [injectionStateWidth, injectionBatchBits, injectionInputCount,
      Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupParam] using widthEvaluate
  · exact widthNonzero

private theorem preprocessingSource_parentFacts
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyPreprocessingSourceIndexFormula workflow reference = true) :
    reference.parallelLoop.indexSlot = 0 ∧ reference.parallelLoop.bindings = [] ∧
      reference.parallelLoop.outputs.length = 1 := by
  unfold verifyPreprocessingSourceIndexFormula verifyParallelIndexFormulaRef at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  aesop

private theorem preprocessingTarget_parentFacts
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyPreprocessingTargetIndexFormula workflow reference = true) :
    reference.parallelLoop.indexSlot = 0 ∧ reference.parallelLoop.bindings = [] ∧
      reference.parallelLoop.outputs.length = 1 := by
  unfold verifyPreprocessingTargetIndexFormula verifyParallelIndexFormulaRef at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  aesop

/-- The full preprocessing source-index loop returns its ordered checked family. -/
theorem preprocessingSourceIndex_familyOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {execution : ReferencedNodeExecution workflow reference.parallelLoop.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs}
    {stride width batch : Int}
    (verified : verifyPreprocessingSourceIndexFormula workflow reference = true)
    (trace : CheckedParallelLoopTrace workflow reference.parallelLoop
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs
      execution)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.parallelLoop.bodyScope.definitionName
      stage.program.definitions = some body)
    (strideEvaluate : injectionTransitionStride.evaluate params = some stride)
    (widthEvaluate : injectionStateWidth.evaluate params = some width)
    (batchEvaluate : injectionBatchBits.evaluate params = some batch)
    (strideNonzero : stride ≠ 0) (widthNonzero : width ≠ 0) :
    execution.values = [.family ((List.range trace.evaluatedCount.toNat).map fun index =>
      .integer ((Int.ofNat index / stride) * width +
        if (Int.ofNat index / stride) * batch + 1 ≤ Int.ofNat index % width then 0
        else Int.ofNat index % width))] := by
  obtain ⟨slotZero, bindingsEmpty, outputsOne⟩ := preprocessingSource_parentFacts verified
  apply checkedParallelLoop_onePortFamily trace outputsOne (fun index =>
    .integer ((Int.ofNat index / stride) * width +
      if (Int.ofNat index / stride) * batch + 1 ≤ Int.ofNat index % width then 0
      else Int.ofNat index % width))
  intro index evaluatedBindings childValues bindingsEvaluate childMember
  rw [bindingsEmpty] at bindingsEvaluate
  simp only [Mxx.Ir.evaluateBindings] at bindingsEvaluate
  have evaluatedBindingsEmpty : evaluatedBindings = [] := Option.some.inj bindingsEvaluate.symm
  subst evaluatedBindings
  apply preprocessingSourceIndex_childOutcome verified bodyResolved definitionFound childMember
  · simp [slotZero, Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupLoopIndex]
  · simpa [injectionTransitionStride, injectionBatchBits, injectionDigitBase,
      injectionInputCount, Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupParam] using strideEvaluate
  · simpa [injectionStateWidth, injectionBatchBits, injectionInputCount,
      Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupParam] using widthEvaluate
  · simpa [injectionBatchBits, Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupParam] using batchEvaluate
  · exact strideNonzero
  · exact widthNonzero

/-- The full preprocessing target-index loop returns its ordered checked family. -/
theorem preprocessingTargetIndex_familyOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {execution : ReferencedNodeExecution workflow reference.parallelLoop.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs}
    {stride width : Int}
    (verified : verifyPreprocessingTargetIndexFormula workflow reference = true)
    (trace : CheckedParallelLoopTrace workflow reference.parallelLoop
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs
      execution)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.parallelLoop.bodyScope.definitionName
      stage.program.definitions = some body)
    (strideEvaluate : injectionTransitionStride.evaluate params = some stride)
    (widthEvaluate : injectionStateWidth.evaluate params = some width)
    (strideNonzero : stride ≠ 0) (widthNonzero : width ≠ 0) :
    execution.values = [.family ((List.range trace.evaluatedCount.toNat).map fun index =>
      .integer ((Int.ofNat index / stride + 1) * width + Int.ofNat index % width))] := by
  obtain ⟨slotZero, bindingsEmpty, outputsOne⟩ := preprocessingTarget_parentFacts verified
  apply checkedParallelLoop_onePortFamily trace outputsOne (fun index =>
    .integer ((Int.ofNat index / stride + 1) * width + Int.ofNat index % width))
  intro index evaluatedBindings childValues bindingsEvaluate childMember
  rw [bindingsEmpty] at bindingsEvaluate
  simp only [Mxx.Ir.evaluateBindings] at bindingsEvaluate
  have evaluatedBindingsEmpty : evaluatedBindings = [] := Option.some.inj bindingsEvaluate.symm
  subst evaluatedBindings
  apply preprocessingTargetIndex_childOutcome verified bodyResolved definitionFound childMember
  · simp [slotZero, Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupLoopIndex]
  · simpa [injectionTransitionStride, injectionBatchBits, injectionDigitBase,
      injectionInputCount, Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupParam] using strideEvaluate
  · simpa [injectionStateWidth, injectionBatchBits, injectionInputCount,
      Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupParam] using widthEvaluate
  · exact strideNonzero
  · exact widthNonzero

private theorem onlineSource_parentFacts
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyOnlineSourceIndexFormula workflow reference = true) :
    reference.parallelLoop.indexSlot = 1 ∧ reference.parallelLoop.bindings = [] ∧
      reference.parallelLoop.inputModes = [.broadcast] ∧
      reference.parallelLoop.outputs.length = 1 := by
  unfold verifyOnlineSourceIndexFormula verifyParallelIndexFormulaRef at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  aesop

private theorem onlineTransition_parentFacts
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyOnlineTransitionIndexFormula workflow reference = true) :
    reference.parallelLoop.indexSlot = 1 ∧ reference.parallelLoop.bindings = [] ∧
      reference.parallelLoop.inputModes = [.broadcast, .broadcast] ∧
      reference.parallelLoop.outputs.length = 1 := by
  unfold verifyOnlineTransitionIndexFormula verifyParallelIndexFormulaRef at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  aesop

private theorem initialExpansion_parentFacts
    {workflow : Mxx.Ir.Workflow} {reference : InitialStateExpansionRef}
    (verified : verifyInitialStateExpansionRef workflow reference = true) :
    reference.parallelLoop.indexSlot = 0 ∧ reference.parallelLoop.bindings = [] ∧
      reference.parallelLoop.inputModes = [.broadcast] ∧
      reference.parallelLoop.outputs.length = 1 := by
  unfold verifyInitialStateExpansionRef at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  aesop

/-- The full online source-index loop masks every position using the checked lower bound. -/
theorem onlineSourceIndex_familyOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {execution : ReferencedNodeExecution workflow reference.parallelLoop.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs}
    {lowerBound : Int}
    (verified : verifyOnlineSourceIndexFormula workflow reference = true)
    (trace : CheckedParallelLoopTrace workflow reference.parallelLoop
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs
      execution)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.parallelLoop.bodyScope.definitionName
      stage.program.definitions = some body)
    (argumentValues : trace.argumentValues = [.integer lowerBound])
    (bodyInputNames : body.inputNames = ["__capture_0"]) :
    execution.values = [.family ((List.range trace.evaluatedCount.toNat).map fun index =>
      .integer (if lowerBound ≤ Int.ofNat index then 0 else Int.ofNat index))] := by
  obtain ⟨slotOne, bindingsEmpty, modes, outputsOne⟩ := onlineSource_parentFacts verified
  apply checkedParallelLoop_onePortFamily trace outputsOne (fun index =>
    .integer (if lowerBound ≤ Int.ofNat index then 0 else Int.ofNat index))
  intro index evaluatedBindings childValues bindingsEvaluate childMember
  rw [bindingsEmpty] at bindingsEvaluate
  simp only [Mxx.Ir.evaluateBindings] at bindingsEvaluate
  have evaluatedBindingsEmpty : evaluatedBindings = [] := Option.some.inj bindingsEvaluate.symm
  subst evaluatedBindings
  apply onlineSourceIndex_childOutcome verified bodyResolved definitionFound childMember
  · simp [bodyInputNames, modes, argumentValues, CertifiedLoopInputMode.toIr,
      Mxx.Ir.loopArgument, Mxx.Ir.lookupEnvironment]
  · simp [slotOne, Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupLoopIndex]

/-- The full online transition-index loop computes one flattened transition index per state. -/
theorem onlineTransitionIndex_familyOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {execution : ReferencedNodeExecution workflow reference.parallelLoop.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs}
    {level digit stride width : Int}
    (verified : verifyOnlineTransitionIndexFormula workflow reference = true)
    (trace : CheckedParallelLoopTrace workflow reference.parallelLoop
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs
      execution)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.parallelLoop.bodyScope.definitionName
      stage.program.definitions = some body)
    (argumentValues : trace.argumentValues = [.integer level, .integer digit])
    (bodyInputNames : body.inputNames = ["__capture_0", "__capture_1"])
    (strideEvaluate : injectionTransitionStride.evaluate params = some stride)
    (widthEvaluate : injectionStateWidth.evaluate params = some width) :
    execution.values = [.family ((List.range trace.evaluatedCount.toNat).map fun index =>
      .integer (level * stride + digit * width + Int.ofNat index))] := by
  obtain ⟨slotOne, bindingsEmpty, modes, outputsOne⟩ := onlineTransition_parentFacts verified
  apply checkedParallelLoop_onePortFamily trace outputsOne (fun index =>
    .integer (level * stride + digit * width + Int.ofNat index))
  intro index evaluatedBindings childValues bindingsEvaluate childMember
  rw [bindingsEmpty] at bindingsEvaluate
  simp only [Mxx.Ir.evaluateBindings] at bindingsEvaluate
  have evaluatedBindingsEmpty : evaluatedBindings = [] := Option.some.inj bindingsEvaluate.symm
  subst evaluatedBindings
  apply onlineTransitionIndex_childOutcome verified bodyResolved definitionFound childMember
  · simp [bodyInputNames, modes, argumentValues, CertifiedLoopInputMode.toIr,
      Mxx.Ir.loopArgument, Mxx.Ir.lookupEnvironment]
  · simp [bodyInputNames, modes, argumentValues, CertifiedLoopInputMode.toIr,
      Mxx.Ir.loopArgument, Mxx.Ir.lookupEnvironment]
  · simpa [injectionTransitionStride, injectionBatchBits, injectionDigitBase,
      injectionInputCount, Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupParam] using strideEvaluate
  · simpa [injectionStateWidth, injectionBatchBits, injectionInputCount,
      Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupParam] using widthEvaluate
  · simp [slotOne, Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupLoopIndex]

/-- The full initial-state expansion loop places the input state at zero and exact zeros in all
remaining family positions. -/
theorem initialStateExpansion_familyOutcome
    {workflow : Mxx.Ir.Workflow} {reference : InitialStateExpansionRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {execution : ReferencedNodeExecution workflow reference.parallelLoop.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs}
    {initial : Mxx.Matrix} {matrixParams : Mxx.SamplerParams}
    (verified : verifyInitialStateExpansionRef workflow reference = true)
    (trace : CheckedParallelLoopTrace workflow reference.parallelLoop
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs
      execution)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.parallelLoop.bodyScope.definitionName
      stage.program.definitions = some body)
    (argumentValues : trace.argumentValues = [.matrix initial])
    (bodyInputNames : body.inputNames = ["__capture_0"])
    (matrixTypeEvaluate :
      ({ modulus := .parameter "diamond_modulus"
         ringDimension := .parameter "diamond_ring_dimension"
         rows := .constant 1
         columns := .add (.constant 4) (.multiply (.constant 2)
           (.parameter "diamond_digit_count")) } : Mxx.Ir.MatrixTypeExpr).evaluate params =
        some matrixParams) :
    execution.values = [.family ((List.range trace.evaluatedCount.toNat).map fun index =>
      .matrix (if index = 0 then initial else Mxx.Matrix.withSamplerParams {
        coefficients := List.replicate
          (matrixParams.rows * matrixParams.columns * matrixParams.ringDimension) 0
      } matrixParams))] := by
  obtain ⟨slotZero, bindingsEmpty, modes, outputsOne⟩ := initialExpansion_parentFacts verified
  apply checkedParallelLoop_onePortFamily trace outputsOne (fun index =>
    .matrix (if index = 0 then initial else Mxx.Matrix.withSamplerParams {
      coefficients := List.replicate
        (matrixParams.rows * matrixParams.columns * matrixParams.ringDimension) 0
    } matrixParams))
  intro index evaluatedBindings childValues bindingsEvaluate childMember
  rw [bindingsEmpty] at bindingsEvaluate
  simp only [Mxx.Ir.evaluateBindings] at bindingsEvaluate
  have evaluatedBindingsEmpty : evaluatedBindings = [] := Option.some.inj bindingsEvaluate.symm
  subst evaluatedBindings
  have outcome := initialStateExpansion_childOutcome
    (index := Int.ofNat index) (initial := initial) (matrixParams := matrixParams)
    verified bodyResolved definitionFound childMember
    (by simp [slotZero, Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupLoopIndex])
    (by simp [bodyInputNames, modes, argumentValues, CertifiedLoopInputMode.toIr,
      Mxx.Ir.loopArgument, Mxx.Ir.lookupEnvironment])
    (by simpa [Mxx.Ir.MatrixTypeExpr.evaluate, Mxx.Ir.IntExpr.evaluate,
      Mxx.Ir.lookupParam] using matrixTypeEvaluate)
  simpa using outcome

/-- A checked one-source gather returns the pointwise family selected by its exact index family. -/
theorem oneSourceParallelGather_familyOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelGatherRef}
    {source : CoreOperandRef} {bodySource output : CoreWireRef}
    {get : DynamicFamilyGetRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {runFuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {execution : ReferencedNodeExecution workflow reference.parallelLoop.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program runFuel) samplers params inputs}
    {sourceValues : List Mxx.Ir.Value}
    (verified : verifyParallelGather workflow reference = true)
    (sources : reference.sourceFamilies = [source])
    (bodySources : reference.bodySources = [bodySource])
    (gets : reference.gets = [get])
    (outputs : reference.outputFamilies = [output])
    (modes : reference.parallelLoop.inputModes = [.zip, .broadcast])
    (bindings : reference.parallelLoop.bindings = [])
    (trace : CheckedParallelLoopTrace workflow reference.parallelLoop
      (Mxx.Ir.childRunnerWithFuel samplers stage.program runFuel) samplers params inputs
      execution)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition
      reference.parallelLoop.bodyScope.definitionName stage.program.definitions = some body)
    (runFuelPositive : 0 < runFuel)
    (ssaOrder : verifyScopeSsaOrder body = true)
    (argumentValues : trace.argumentValues =
      [.family ((List.range trace.evaluatedCount.toNat).map fun index =>
        .integer (Int.ofNat index)), .family sourceValues]) :
    execution.values = [.family ((List.range trace.evaluatedCount.toNat).map fun index =>
      sourceValues[index]?.getD (.invalid "FamilyGetDynamic index out of range"))] := by
  obtain ⟨fuel, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt runFuelPositive)
  obtain ⟨_, _, _, _, _, outputFamiliesEq, _, _⟩ :=
    oneSourceParallelGatherFacts verified sources bodySources gets outputs
  have outputsOne : reference.parallelLoop.outputs.length = 1 := by
    simp [outputFamiliesEq]
  have initialEq : List.replicate reference.parallelLoop.outputs.length [] =
      ([[]] : List (List Mxx.Ir.Value)) := by simp [outputsOne]
  have finalEq := parallelIterationsTrace_singlePortValues_mem
    (fun index => sourceValues[index]?.getD (.invalid "FamilyGetDynamic index out of range"))
    trace.iterations initialEq
  rw [trace.valuesEq, finalEq]
  · simp
  · intro index indexMember evaluatedBindings childValues bindingsEvaluate childMember
    rw [bindings] at bindingsEvaluate
    simp only [Mxx.Ir.evaluateBindings] at bindingsEvaluate
    have evaluatedBindingsEmpty : evaluatedBindings = [] :=
      Option.some.inj bindingsEvaluate.symm
    subst evaluatedBindings
    have childInputsEq :
        (((reference.parallelLoop.inputModes.map CertifiedLoopInputMode.toIr).zip
          trace.argumentValues).map
          fun (mode, value) => Mxx.Ir.loopArgument mode index value) =
          [.integer (Int.ofNat index), .family sourceValues] := by
      have indexLt : index < trace.evaluatedCount.toNat := List.mem_range.mp indexMember
      simp [modes, argumentValues, CertifiedLoopInputMode.toIr, Mxx.Ir.loopArgument, indexLt]
    exact oneSourceParallelGather_childOutcome verified sources bodySources gets outputs
      bodyResolved definitionFound ssaOrder childInputsEq
      (by simpa [modes, argumentValues, CertifiedLoopInputMode.toIr] using childMember)

/-! ## Exact witness-digit packing execution -/

private theorem witnessDigitScan_childOutcome
    {workflow : Mxx.Ir.Workflow} {reference : WitnessDigitPackingRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {childInputs childValues : List Mxx.Ir.Value}
    {bits : List Int} {level batch index : Nat} {accumulator weight bit : Int}
    (verified : verifyWitnessDigitPackingRef workflow reference = true)
    (bodyResolved : resolveScope workflow
      { reference.bitScan.operation with scope := reference.bitScan.bodyScope } = some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.bitScan.bodyScope.definitionName
      stage.program.definitions = some body)
    (childInputsEq : childInputs = [
      .integer accumulator, .integer weight, .family (bits.map Mxx.Ir.Value.integer),
      .integer (Int.ofNat level), .integer (Int.ofNat batch)])
    (bitAt : bits[level * batch + index]? = some bit)
    (childMember : childValues ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.bitScan.bodyScope.definitionName params childInputs)
    (indexEvaluate : (Mxx.Ir.IntExpr.loopIndex 1).evaluate params =
      some (Int.ofNat index)) :
    childValues = [.integer (accumulator + bit * weight), .integer (weight * 2)] := by
  obtain ⟨path⟩ := childExecutionPath_of_outcome definitionFound childMember
  have namesChecked := verified
  unfold verifyWitnessDigitPackingRef at namesChecked
  simp only [Bool.and_eq_true, decide_eq_true_eq] at namesChecked
  have inputNamesChecked := namesChecked.2
  unfold verifyWitnessDigitPackingInputNames at inputNamesChecked
  rw [bodyResolved] at inputNamesChecked
  simp only [Bool.and_eq_true, decide_eq_true_eq] at inputNamesChecked
  have scanNames : body.inputNames = [
      "arg-0-integer", "arg-1-integer", "arg-2-family", "__capture_0", "__capture_1"] :=
    inputNamesChecked.2
  have bitValueAt : (bits.map Mxx.Ir.Value.integer)[
      (Int.ofNat level * Int.ofNat batch + Int.ofNat index).toNat]? =
        some (.integer bit) := by
    have indexCast : Int.ofNat level * Int.ofNat batch + Int.ofNat index =
        Int.ofNat (level * batch + index) := by norm_num
    rw [indexCast]
    norm_num
    simpa only [List.getElem?_map, bitAt, Option.map_some]
  have scanOutputs := verifyWitnessDigitPackingRef_scanPathOutputs verified bodyResolved path.path
    (accumulator := accumulator) (outerIndex := Int.ofNat level)
    (batchSize := Int.ofNat batch) (bitIndex := Int.ofNat index) (weight := weight)
    (bitValue := bit) (bits := bits.map Mxx.Ir.Value.integer)
    (by simp [scanNames, childInputsEq, Mxx.Ir.lookupEnvironment])
    (by simp [scanNames, childInputsEq, Mxx.Ir.lookupEnvironment])
    (by simp [scanNames, childInputsEq, Mxx.Ir.lookupEnvironment])
    (by simp [scanNames, childInputsEq, Mxx.Ir.lookupEnvironment])
    (by simp [scanNames, childInputsEq, Mxx.Ir.lookupEnvironment]) indexEvaluate bitValueAt
  have scanLoopVerified : verifySequentialLoopRef workflow reference.bitScan = true := by
    aesop
  have scopeOutputs : body.outputs.map Prod.snd =
      [({ node := 10, port := 0 } : Mxx.Ir.WireRef),
        ({ node := 12, port := 0 } : Mxx.Ir.WireRef)] := by
    unfold verifySequentialLoopRef at scanLoopVerified
    simp only [Bool.and_eq_true, decide_eq_true_eq] at scanLoopVerified
    rw [bodyResolved] at scanLoopVerified
    have bodyOutputWires : reference.bitScan.bodyOutputs.map wireRef = scopeOutputWires body := by
      aesop
    have checkedOutputs : reference.bitScan.bodyOutputs.map wireRef =
        [({ node := 10, port := 0 } : Mxx.Ir.WireRef),
          ({ node := 12, port := 0 } : Mxx.Ir.WireRef)] := by
      aesop
    change reference.bitScan.bodyOutputs.map wireRef = body.outputs.map Prod.snd at bodyOutputWires
    exact bodyOutputWires.symm.trans checkedOutputs
  change Mxx.Ir.lookupWire ({ node := 10, port := 0 } : Mxx.Ir.WireRef) path.finalWires =
      some (.integer (accumulator + bit * weight)) ∧
    Mxx.Ir.lookupWire ({ node := 12, port := 0 } : Mxx.Ir.WireRef) path.finalWires =
      some (.integer (weight * 2)) at scanOutputs
  rw [path.outputs]
  cases outputsEq : body.outputs with
  | nil => simp [outputsEq] at scopeOutputs
  | cons first tail =>
    cases tail with
    | nil => simp [outputsEq] at scopeOutputs
    | cons second rest =>
      cases rest with
      | cons third more => simp [outputsEq] at scopeOutputs
      | nil =>
        rcases first with ⟨firstName, firstWire⟩
        rcases second with ⟨secondName, secondWire⟩
        simp [outputsEq] at scopeOutputs
        rcases scopeOutputs with ⟨rfl, rfl⟩
        simp only [Mxx.Ir.collectOutputs, List.map_cons, List.map_nil]
        rw [scanOutputs.1, scanOutputs.2]
        rfl

private theorem witnessDigitScan_traceOutcomeFrom
    {workflow : Mxx.Ir.Workflow} {reference : WitnessDigitPackingRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {bits : List Int} {level batch start count : Nat} {final : List Mxx.Ir.Value}
    (verified : verifyWitnessDigitPackingRef workflow reference = true)
    (bodyResolved : resolveScope workflow
      { reference.bitScan.operation with scope := reference.bitScan.bodyScope } = some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.bitScan.bodyScope.definitionName
      stage.program.definitions = some body)
    (bitsPresent : ∀ index, index < batch →
      ∃ bit, bits[level * batch + index]? = some bit)
    (trace : Mxx.Ir.SequentialIterationsTrace
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1))
      reference.bitScan.bodyScope.definitionName params 1 []
      [.family (bits.map Mxx.Ir.Value.integer), .integer (Int.ofNat level),
        .integer (Int.ofNat batch)]
      (List.range' start count)
      [.integer (packedWitnessPrefix bits (level * batch) start),
        .integer (Int.ofNat (2 ^ start))] final)
    (endLe : start + count ≤ batch) :
    final = [.integer (packedWitnessPrefix bits (level * batch) (start + count)),
      .integer (Int.ofNat (2 ^ (start + count)))] := by
  induction count generalizing start final with
  | zero =>
      cases trace
      simp
  | succ count induction =>
      rw [List.range'_succ] at trace
      cases trace with
      | cons index tail state evaluatedBindings next final bindingsEvaluate childMember rest =>
        have indexLt : start < batch := by omega
        obtain ⟨bit, bitAt⟩ := bitsPresent start indexLt
        simp only [Mxx.Ir.evaluateBindings] at bindingsEvaluate
        have evaluatedBindingsEmpty : evaluatedBindings = [] :=
          Option.some.inj bindingsEvaluate.symm
        subst evaluatedBindings
        have nextEq := witnessDigitScan_childOutcome verified bodyResolved definitionFound rfl bitAt
          childMember (by simp [Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupLoopIndex])
        have prefixStep :
            packedWitnessPrefix bits (level * batch) start + bit * Int.ofNat (2 ^ start) =
              packedWitnessPrefix bits (level * batch) (start + 1) := by
          rw [packedWitnessPrefix]
          simp [bitAt]
        have weightStep : Int.ofNat (2 ^ start) * 2 = Int.ofNat (2 ^ (start + 1)) := by
          simp [pow_succ]
        rw [nextEq, prefixStep, weightStep] at rest
        have tailOutcome := induction (start := start + 1) rest (by omega)
        have endEq : (start + 1) + count = start + (count + 1) := by omega
        simpa only [endEq] using tailOutcome

/-- The inner scan starting at zero computes exactly one little-endian packed witness digit. -/
theorem witnessDigitScan_traceOutcome
    {workflow : Mxx.Ir.Workflow} {reference : WitnessDigitPackingRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {bits : List Int} {level batch : Nat} {final : List Mxx.Ir.Value}
    (verified : verifyWitnessDigitPackingRef workflow reference = true)
    (bodyResolved : resolveScope workflow
      { reference.bitScan.operation with scope := reference.bitScan.bodyScope } = some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.bitScan.bodyScope.definitionName
      stage.program.definitions = some body)
    (bitsPresent : ∀ index, index < batch →
      ∃ bit, bits[level * batch + index]? = some bit)
    (trace : Mxx.Ir.SequentialIterationsTrace
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1))
      reference.bitScan.bodyScope.definitionName params 1 []
      [.family (bits.map Mxx.Ir.Value.integer), .integer (Int.ofNat level),
        .integer (Int.ofNat batch)]
      (List.range batch) [.integer 0, .integer 1] final) :
    final = [.integer (packedWitnessPrefix bits (level * batch) batch),
      .integer (Int.ofNat (2 ^ batch))] := by
  have rangeTrace : Mxx.Ir.SequentialIterationsTrace
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1))
      reference.bitScan.bodyScope.definitionName params 1 []
      [.family (bits.map Mxx.Ir.Value.integer), .integer (Int.ofNat level),
        .integer (Int.ofNat batch)]
      (List.range' 0 batch) [.integer 0, .integer 1] final := by
    rw [← List.range_eq_range']
    exact trace
  simpa only [Nat.zero_add] using
    witnessDigitScan_traceOutcomeFrom (start := 0) (count := batch) verified bodyResolved
      definitionFound bitsPresent rangeTrace (show 0 + batch ≤ batch by omega)

private theorem witnessDigitPacking_childOutcome
    {workflow : Mxx.Ir.Workflow} {reference : WitnessDigitPackingRef}
    {stage : Mxx.Ir.Stage} {outerBody scanBody : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {childInputs childValues : List Mxx.Ir.Value}
    {bits : List Int} {level batch : Nat}
    (verified : verifyWitnessDigitPackingRef workflow reference = true)
    (outerBodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some outerBody)
    (outerDefinitionFound : Mxx.Ir.lookupDefinition
      reference.parallelLoop.bodyScope.definitionName stage.program.definitions = some outerBody)
    (scanBodyResolved : resolveScope workflow
      { reference.bitScan.operation with scope := reference.bitScan.bodyScope } = some scanBody)
    (scanDefinitionFound : Mxx.Ir.lookupDefinition reference.bitScan.bodyScope.definitionName
      stage.program.definitions = some scanBody)
    (bitsPresent : ∀ index, index < batch →
      ∃ bit, bits[level * batch + index]? = some bit)
    (childInputsEq : childInputs = [.family (bits.map Mxx.Ir.Value.integer)])
    (childMember : childValues ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.parallelLoop.bodyScope.definitionName params childInputs)
    (fuelPositive : 0 < fuel)
    (outerIndexEvaluate :
      (Mxx.Ir.IntExpr.loopIndex reference.parallelLoop.indexSlot).evaluate params =
        some (Int.ofNat level))
    (batchEvaluate : reference.bitScan.count.evaluate params = some (Int.ofNat batch)) :
    childValues = [.integer (packedWitnessPrefix bits (level * batch) batch)] := by
  have loopVerified : verifyParallelLoop workflow reference.parallelLoop = true := by
    have checked := verified
    unfold verifyWitnessDigitPackingRef at checked
    simp only [Bool.and_eq_true] at checked
    aesop
  have bodyOutputs : reference.parallelLoop.bodyOutputs = [reference.bodyOutput] := by
    have checked := verified
    unfold verifyWitnessDigitPackingRef at checked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
    aesop
  have outerIndexSlot : reference.parallelLoop.indexSlot = 0 := by
    have checked := verified
    unfold verifyWitnessDigitPackingRef at checked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
    aesop
  have scanCount : reference.bitScan.count =
      Mxx.Ir.IntExpr.parameter "diamond_batch_bits" := by
    have checked := verified
    unfold verifyWitnessDigitPackingRef at checked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
    aesop
  have exactOuterIndexEvaluate : (Mxx.Ir.IntExpr.loopIndex 0).evaluate params =
      some (Int.ofNat level) := by
    simpa only [outerIndexSlot] using outerIndexEvaluate
  have exactBatchEvaluate :
      (Mxx.Ir.IntExpr.parameter "diamond_batch_bits").evaluate params =
        some (Int.ofNat batch) := by
    simpa only [scanCount] using batchEvaluate
  apply parallelLoopSingleBodyOutput_of_childOutcome loopVerified bodyOutputs outerBodyResolved
    outerDefinitionFound childMember
  intro final path
  have outerNamesChecked := verified
  unfold verifyWitnessDigitPackingRef at outerNamesChecked
  simp only [Bool.and_eq_true] at outerNamesChecked
  have inputNamesChecked := outerNamesChecked.2
  unfold verifyWitnessDigitPackingInputNames at inputNamesChecked
  rw [outerBodyResolved] at inputNamesChecked
  simp only [Bool.and_eq_true, decide_eq_true_eq] at inputNamesChecked
  have outerNames : outerBody.inputNames = ["pack-bit-source"] := inputNamesChecked.1
  obtain ⟨finalValues, scanTrace, outputAt⟩ :=
    verifyWitnessDigitPackingRef_outerPathTrace verified outerBodyResolved path
      (bits := bits.map Mxx.Ir.Value.integer) (outerIndex := Int.ofNat level)
      (batchSize := Int.ofNat batch)
      (by simp [outerNames, childInputsEq, Mxx.Ir.lookupEnvironment])
      exactOuterIndexEvaluate exactBatchEvaluate
  obtain ⟨scanFuel, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt fuelPositive)
  have finalValuesEq := witnessDigitScan_traceOutcome verified scanBodyResolved scanDefinitionFound
    bitsPresent scanTrace
  simpa [finalValuesEq] using outputAt

/-- The checked outer packing loop returns one exact little-endian digit for every level. -/
theorem witnessDigitPacking_familyOutcome
    {workflow : Mxx.Ir.Workflow} {reference : WitnessDigitPackingRef}
    {stage : Mxx.Ir.Stage} {outerBody scanBody : Mxx.Ir.Scope} {runFuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {execution : ReferencedNodeExecution workflow reference.parallelLoop.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program runFuel) samplers params inputs}
    {bits : List Int} {batch : Nat}
    (verified : verifyWitnessDigitPackingRef workflow reference = true)
    (trace : CheckedParallelLoopTrace workflow reference.parallelLoop
      (Mxx.Ir.childRunnerWithFuel samplers stage.program runFuel) samplers params inputs execution)
    (outerBodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some outerBody)
    (outerDefinitionFound : Mxx.Ir.lookupDefinition
      reference.parallelLoop.bodyScope.definitionName stage.program.definitions = some outerBody)
    (scanBodyResolved : resolveScope workflow
      { reference.bitScan.operation with scope := reference.bitScan.bodyScope } = some scanBody)
    (scanDefinitionFound : Mxx.Ir.lookupDefinition reference.bitScan.bodyScope.definitionName
      stage.program.definitions = some scanBody)
    (runFuelAtLeastTwo : 2 ≤ runFuel)
    (argumentValues : trace.argumentValues = [.family (bits.map Mxx.Ir.Value.integer)])
    (batchEvaluate : reference.bitScan.count.evaluate params = some (Int.ofNat batch))
    (bitsPresent : ∀ level ∈ List.range trace.evaluatedCount.toNat,
      ∀ index, index < batch → ∃ bit, bits[level * batch + index]? = some bit) :
    execution.values = [.family ((List.range trace.evaluatedCount.toNat).map fun level =>
      .integer (packedWitnessPrefix bits (level * batch) batch))] := by
  obtain ⟨fuel, rfl⟩ := Nat.exists_eq_succ_of_ne_zero
    (Nat.ne_of_gt (show 0 < runFuel by omega))
  have checked := verified
  unfold verifyWitnessDigitPackingRef at checked
  simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
  have modes : reference.parallelLoop.inputModes = [.broadcast] := by aesop
  have bindings : reference.parallelLoop.bindings = [] := by aesop
  have outputsOne : reference.parallelLoop.outputs.length = 1 := by aesop
  have initialEq : List.replicate reference.parallelLoop.outputs.length [] =
      ([[]] : List (List Mxx.Ir.Value)) := by simp [outputsOne]
  have finalEq := parallelIterationsTrace_singlePortValues_mem
    (fun level => .integer (packedWitnessPrefix bits (level * batch) batch))
    trace.iterations initialEq
  rw [trace.valuesEq, finalEq]
  · simp
  · intro level levelMember evaluatedBindings childValues bindingsEvaluate childMember
    rw [bindings] at bindingsEvaluate
    simp only [Mxx.Ir.evaluateBindings] at bindingsEvaluate
    have evaluatedBindingsEmpty : evaluatedBindings = [] :=
      Option.some.inj bindingsEvaluate.symm
    subst evaluatedBindings
    have scanCount : reference.bitScan.count =
        Mxx.Ir.IntExpr.parameter "diamond_batch_bits" := by aesop
    have batchEvaluateAtLoop : reference.bitScan.count.evaluate
        ((.loopIndex reference.parallelLoop.indexSlot, .integer level) :: params) =
          some (Int.ofNat batch) := by
      rw [scanCount] at batchEvaluate ⊢
      simpa [Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupParam] using batchEvaluate
    apply witnessDigitPacking_childOutcome verified outerBodyResolved outerDefinitionFound
      scanBodyResolved scanDefinitionFound (bitsPresent level levelMember) rfl
    · simpa [modes, argumentValues, CertifiedLoopInputMode.toIr, Mxx.Ir.loopArgument] using
        childMember
    · omega
    · simp [Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupLoopIndex]
    · exact batchEvaluateAtLoop

/-! ## State-zero input-injection semantics

The executable input injector carries a family of states.  The Diamond decryptor consumes the
state at index zero, so the correctness argument follows that entry while retaining the exact
family value produced by every loop iteration.  In particular, the public base matrix is indexed
by the loop level: it is never hidden behind an existential representation predicate.
-/

/-- The state-zero entry of one executable state family represents `signal * base + error`.
The equation is stated in the negacyclic matrix algebra, which is the representation preserved by
matrix multiplication even when concrete coefficient lists use different representatives. -/
structure InputInjectionStateZero
    (q ringDimension stateColumns : Nat) [NeZero q] [NeZero ringDimension]
    (base : Mxx.Matrix) (values : List Mxx.Ir.Value)
    (signal error : Mxx.Matrix) where
  family : List Mxx.Ir.Value
  state : Mxx.Matrix
  valuesEq : values = [.family family]
  stateZero : family[0]? = some (.matrix state)
  stateShape : Mxx.Toolkit.MatrixShape state q ringDimension 1 stateColumns
  signalShape : Mxx.Toolkit.MatrixShape signal q ringDimension 1 2
  baseShape : Mxx.Toolkit.MatrixShape base q ringDimension 2 stateColumns
  errorShape : Mxx.Toolkit.MatrixShape error q ringDimension 1 stateColumns
  equation :
    Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns state =
      Mxx.Toolkit.matrixValue q ringDimension 1 2 signal *
          Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns base +
        Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns error

/-- One exact preprocessing-table entry.  `base` and `nextBase` are explicit fields so the
preimage equation remains tied to the same public matrices used by adjacent scan levels. -/
structure InputInjectionTransitionEntry
    (q ringDimension stateColumns errorBound preimageBound : Nat)
    [NeZero q] [NeZero ringDimension]
    (base nextBase : Mxx.Matrix) where
  transitionFamily : List Mxx.Ir.Value
  flatIndex : Nat
  transition : Mxx.Matrix
  selector : Mxx.Matrix
  transitionError : Mxx.Matrix
  transitionAt : transitionFamily[flatIndex]? = some (.matrix transition)
  baseShape : Mxx.Toolkit.MatrixShape base q ringDimension 2 stateColumns
  nextBaseShape : Mxx.Toolkit.MatrixShape nextBase q ringDimension 2 stateColumns
  transitionShape :
    Mxx.Toolkit.MatrixShape transition q ringDimension stateColumns stateColumns
  selectorShape : Mxx.Toolkit.MatrixShape selector q ringDimension 2 2
  transitionErrorShape :
    Mxx.Toolkit.MatrixShape transitionError q ringDimension 2 stateColumns
  equation :
    Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns base *
        Mxx.Toolkit.matrixValue q ringDimension stateColumns stateColumns transition =
      Mxx.Toolkit.matrixValue q ringDimension 2 2 selector *
          Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns nextBase +
        Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns transitionError
  selectorNorm : Mxx.maxCenteredCoefficientNorm selector ≤ 1
  transitionErrorNorm :
    Mxx.maxCenteredCoefficientNorm transitionError ≤ errorBound
  transitionNorm : Mxx.maxCenteredCoefficientNorm transition ≤ preimageBound

/-- Concrete preprocessing result consumed by the online scan proof.  The producer-side
encryption bridge must construct this object from the checked parallel target and preimage paths;
the online proof never accepts detached transition equations or a caller-chosen base sequence. -/
structure InputInjectionTransitionTable
    (q ringDimension stateColumns errorBound preimageBound inputCount : Nat)
    [NeZero q] [NeZero ringDimension] where
  stateWidth : Nat
  transitionStride : Nat
  digitBase : Nat
  baseAt : Nat → Mxx.Matrix
  transitionFamily : List Mxx.Ir.Value
  entry : ∀ (level : Nat), level < inputCount → ∀ (digit : Nat), digit < digitBase →
    ∃ exactEntry : InputInjectionTransitionEntry q ringDimension stateColumns errorBound
        preimageBound (baseAt level) (baseAt (level + 1)),
      exactEntry.transitionFamily = transitionFamily ∧
      exactEntry.flatIndex = level * transitionStride + digit * stateWidth

/-- Construct the initial state-zero representation from the checked initial Gaussian outcome and
the exact initial-state expansion family.  The initial error bound is therefore the sampler
contract's hard-support bound, not a separately asserted estimate. -/
theorem inputInjectionInitialStateZero_of_samplerOutcome
    {workflow : Mxx.Ir.Workflow} {errorReference : OperationRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    {execution : ReferencedNodeExecution workflow errorReference.operation runChild samplers
      params inputs}
    {q ringDimension stateColumns errorBound : Nat} [NeZero q] [NeZero ringDimension]
    [Fact (1 < q)]
    {base signal initialState : Mxx.Matrix} (initialFamily : List Mxx.Ir.Value)
    (errorOutcome : GaussianSampleOutcome workflow errorReference runChild samplers params inputs
      execution)
    (errorParams : errorOutcome.matrixParams = {
      modulus := q, ringDimension := ringDimension, rows := 1, columns := stateColumns,
      maxCoefficientBound := errorBound })
    (baseShape : Mxx.Toolkit.MatrixShape base q ringDimension 2 stateColumns)
    (signalShape : Mxx.Toolkit.MatrixShape signal q ringDimension 1 2)
    (initialStateEq :
      initialState = Mxx.matrixAdd (Mxx.matrixMul signal base) errorOutcome.sample)
    (initialAt : initialFamily[0]? = some (.matrix initialState)) :
    Nonempty (InputInjectionStateZero q ringDimension stateColumns base
        [.family initialFamily] signal errorOutcome.sample) ∧
      Mxx.maxCenteredCoefficientNorm errorOutcome.sample ≤ errorBound := by
  have errorShape : Mxx.Toolkit.MatrixShape errorOutcome.sample q ringDimension 1 stateColumns := by
    simpa [errorParams] using errorOutcome.shape
  have productShape := Mxx.Toolkit.matrixMul_shape signal base signalShape baseShape
  have stateShape := Mxx.Toolkit.matrixAdd_shape (Mxx.matrixMul signal base)
    errorOutcome.sample productShape errorShape
  constructor
  · refine ⟨{
      family := initialFamily
      state := initialState
      valuesEq := rfl
      stateZero := initialAt
      stateShape := by simpa [initialStateEq] using stateShape
      signalShape
      baseShape
      errorShape
      equation := ?_
    }⟩
    calc
      Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns initialState =
          Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns
            (Mxx.matrixAdd (Mxx.matrixMul signal base) errorOutcome.sample) := by
              rw [initialStateEq]
      _ = Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns
            (Mxx.matrixMul signal base) +
          Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns errorOutcome.sample :=
        Mxx.Toolkit.matrixValue_add q ringDimension 1 stateColumns _ _
          ⟨productShape.modulus, productShape.ringDimension, productShape.rows,
            productShape.columns⟩
          ⟨errorShape.modulus, errorShape.ringDimension, errorShape.rows, errorShape.columns⟩
      _ = Mxx.Toolkit.matrixValue q ringDimension 1 2 signal *
            Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns base +
          Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns errorOutcome.sample := by
        rw [Mxx.Toolkit.matrixValue_mul q ringDimension 1 2 stateColumns signal base
          ⟨signalShape.modulus, signalShape.ringDimension, signalShape.rows,
            signalShape.columns⟩
          ⟨baseShape.modulus, baseShape.ringDimension, baseShape.rows, baseShape.columns⟩]
  · simpa [errorParams] using errorOutcome.norm

/-- Build one preprocessing-table entry from the exact Gaussian and preimage outcomes selected by
the typed certificate.  Thus the two hard bounds come from `MxxBoundedSamplerContract`; this
constructor does not accept independent numerical norm assumptions for either sampled matrix. -/
theorem inputInjectionTransitionEntry_of_samplerOutcomes
    {workflow : Mxx.Ir.Workflow} {errorReference preimageReference : OperationRef}
    {errorRunChild preimageRunChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {errorInputs preimageInputs : Mxx.Ir.Environment}
    {errorExecution : ReferencedNodeExecution workflow errorReference.operation errorRunChild
      samplers params errorInputs}
    {preimageExecution : ReferencedNodeExecution workflow preimageReference.operation
      preimageRunChild samplers params preimageInputs}
    {q ringDimension stateColumns errorBound preimageBound : Nat}
    [NeZero q] [NeZero ringDimension] [Fact (1 < q)]
    {base nextBase selector : Mxx.Matrix}
    (transitionFamily : List Mxx.Ir.Value) (flatIndex : Nat)
    (errorOutcome : GaussianSampleOutcome workflow errorReference errorRunChild samplers params
      errorInputs errorExecution)
    (preimageOutcome : PreimageSampleOutcome workflow preimageReference preimageRunChild samplers
      params preimageInputs preimageExecution)
    (errorParams : errorOutcome.matrixParams = {
      modulus := q, ringDimension := ringDimension, rows := 2, columns := stateColumns,
      maxCoefficientBound := errorBound })
    (preimageParams : preimageOutcome.matrixParams = {
      modulus := q, ringDimension := ringDimension, rows := stateColumns,
      columns := stateColumns, maxCoefficientBound := preimageBound })
    (publicMatrix : preimageOutcome.publicMatrix = base)
    (target : preimageOutcome.target =
      Mxx.matrixAdd (Mxx.matrixMul selector nextBase) errorOutcome.sample)
    (transitionAt : transitionFamily[flatIndex]? = some (.matrix preimageOutcome.preimage))
    (baseShape : Mxx.Toolkit.MatrixShape base q ringDimension 2 stateColumns)
    (nextBaseShape : Mxx.Toolkit.MatrixShape nextBase q ringDimension 2 stateColumns)
    (selectorShape : Mxx.Toolkit.MatrixShape selector q ringDimension 2 2)
    (selectorNorm : Mxx.maxCenteredCoefficientNorm selector ≤ 1) :
    Nonempty (InputInjectionTransitionEntry q ringDimension stateColumns errorBound
      preimageBound base nextBase) := by
  have errorShape : Mxx.Toolkit.MatrixShape errorOutcome.sample q ringDimension 2 stateColumns := by
    simpa [errorParams] using errorOutcome.shape
  have transitionShape : Mxx.Toolkit.MatrixShape preimageOutcome.preimage q ringDimension
      stateColumns stateColumns := by
    simpa [preimageParams] using preimageOutcome.shape
  have selectorProductShape :=
    Mxx.Toolkit.matrixMul_shape selector nextBase selectorShape nextBaseShape
  refine ⟨{
    transitionFamily
    flatIndex
    transition := preimageOutcome.preimage
    selector
    transitionError := errorOutcome.sample
    transitionAt
    baseShape
    nextBaseShape
    transitionShape
    selectorShape
    transitionErrorShape := errorShape
    equation := ?_
    selectorNorm
    transitionErrorNorm := by simpa [errorParams] using errorOutcome.norm
    transitionNorm := by simpa [preimageParams] using preimageOutcome.norm
  }⟩
  have rawEquation : Mxx.matrixMul base preimageOutcome.preimage =
      Mxx.matrixAdd (Mxx.matrixMul selector nextBase) errorOutcome.sample := by
    simpa [publicMatrix, target] using preimageOutcome.equation
  calc
    Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns base *
        Mxx.Toolkit.matrixValue q ringDimension stateColumns stateColumns
          preimageOutcome.preimage =
      Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns
        (Mxx.matrixMul base preimageOutcome.preimage) := by
          rw [Mxx.Toolkit.matrixValue_mul q ringDimension 2 stateColumns stateColumns base
            preimageOutcome.preimage
            ⟨baseShape.modulus, baseShape.ringDimension, baseShape.rows, baseShape.columns⟩
            ⟨transitionShape.modulus, transitionShape.ringDimension, transitionShape.rows,
              transitionShape.columns⟩]
    _ = Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns
        (Mxx.matrixAdd (Mxx.matrixMul selector nextBase) errorOutcome.sample) := by
          rw [rawEquation]
    _ = Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns
          (Mxx.matrixMul selector nextBase) +
        Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns errorOutcome.sample :=
      Mxx.Toolkit.matrixValue_add q ringDimension 2 stateColumns _ _
        ⟨selectorProductShape.modulus, selectorProductShape.ringDimension,
          selectorProductShape.rows, selectorProductShape.columns⟩
        ⟨errorShape.modulus, errorShape.ringDimension, errorShape.rows, errorShape.columns⟩
    _ = Mxx.Toolkit.matrixValue q ringDimension 2 2 selector *
          Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns nextBase +
        Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns errorOutcome.sample := by
      rw [Mxx.Toolkit.matrixValue_mul q ringDimension 2 2 stateColumns selector nextBase
        ⟨selectorShape.modulus, selectorShape.ringDimension, selectorShape.rows,
          selectorShape.columns⟩
        ⟨nextBaseShape.modulus, nextBaseShape.ringDimension, nextBaseShape.rows,
          nextBaseShape.columns⟩]

/-- Exact online evidence for state zero in one checked scan-body execution.  The family and
transition indices are retained, and the flattened transition index is tied to the preprocessing
entry rather than merely to a numerically equal matrix. -/
structure InputInjectionStateZeroIteration
    (workflow : Mxx.Ir.Workflow) (layout : InputInjectionLayout)
    (stage : Mxx.Ir.Stage) (scope : Mxx.Ir.Scope) (fuel : Nat)
    (samplers : Mxx.MxxSamplerFamily) (params : Mxx.Ir.ParamEnvironment)
    (childInputs state invariantArguments next : List Mxx.Ir.Value)
    (q ringDimension stateColumns errorBound preimageBound : Nat)
    [NeZero q] [NeZero ringDimension]
    (level : Nat) (base nextBase signal stateError : Mxx.Matrix) where
  executions : InputInjectionIterationExecutions workflow layout stage scope fuel samplers params
    childInputs
  childInputsEq : childInputs = state ++ invariantArguments
  current : InputInjectionStateZero q ringDimension stateColumns base state signal stateError
  transition : InputInjectionTransitionEntry q ringDimension stateColumns errorBound
    preimageBound base nextBase
  packedDigits : List Mxx.Ir.Value
  invariantArgumentsEq :
    invariantArguments = [.family packedDigits, .family transition.transitionFamily]
  selectedDigit : Int
  packedDigitAt : packedDigits[level]? = some (.integer selectedDigit)
  selectedDigitOutcome : executions.selectedDigit.execution.values = [.integer selectedDigit]
  sourceIndexFamily : List Mxx.Ir.Value
  sourceIndicesOutcome : executions.sourceIndices.execution.values = [.family sourceIndexFamily]
  sourceIndexZero : sourceIndexFamily[0]? = some (.integer 0)
  sourceStateFamily : List Mxx.Ir.Value
  sourceStatesOutcome : executions.sourceStates.execution.values = [.family sourceStateFamily]
  sourceStateZero : sourceStateFamily[0]? = some (.matrix current.state)
  transitionStride : Int
  stateWidth : Int
  transitionIndex : Int
  transitionIndexNonnegative : 0 ≤ transitionIndex
  transitionIndexFormula : transitionIndex =
    Int.ofNat level * transitionStride + selectedDigit * stateWidth
  transitionIndexFamily : List Mxx.Ir.Value
  transitionIndicesOutcome :
    executions.transitionIndices.execution.values = [.family transitionIndexFamily]
  transitionIndexZero : transitionIndexFamily[0]? = some (.integer transitionIndex)
  transitionIndexEq : transitionIndex.toNat = transition.flatIndex
  selectedTransitionFamily : List Mxx.Ir.Value
  selectedTransitionsOutcome :
    executions.selectedTransitions.execution.values = [.family selectedTransitionFamily]
  selectedTransitionAt :
    selectedTransitionFamily[0]? = some (.matrix transition.transition)
  preprocessingFamilyEq : selectedTransitionFamily[0]? =
    transition.transitionFamily[transitionIndex.toNat]?
  nextFamily : List Mxx.Ir.Value
  nextState : Mxx.Matrix
  nextEq : next = [.family nextFamily]
  stateProductOutcome : executions.stateProduct.execution.values = [.family nextFamily]
  nextZero : nextFamily[0]? = some (.matrix nextState)
  nextStateEq : nextState = Mxx.matrixMul current.state transition.transition

/-- One exact state-zero iteration yields the algebraic step and the next level's representation.
All sampled bounds are inherited from `InputInjectionTransitionEntry`; no callback can inject a
different semantic transition. -/
theorem InputInjectionStateZeroIteration.stepAndNext
    {workflow : Mxx.Ir.Workflow} {layout : InputInjectionLayout}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {childInputs state invariantArguments next : List Mxx.Ir.Value}
    {q ringDimension stateColumns errorBound preimageBound : Nat}
    [NeZero q] [NeZero ringDimension] [Fact (1 < q)]
    {level : Nat} {base nextBase signal stateError : Mxx.Matrix}
    (iteration : InputInjectionStateZeroIteration workflow layout stage scope fuel samplers params
      childInputs state invariantArguments next q ringDimension stateColumns errorBound
      preimageBound level base nextBase signal stateError) :
    ∃ step : MxxWe.InputInjectionStep q ringDimension stateColumns,
      step.signal = signal ∧ step.stateError = stateError ∧
      step.selector = iteration.transition.selector ∧
      step.transitionError = iteration.transition.transitionError ∧
      step.transition = iteration.transition.transition ∧
      Mxx.maxCenteredCoefficientNorm step.selector ≤ 1 ∧
      Mxx.maxCenteredCoefficientNorm step.transitionError ≤ errorBound ∧
      Mxx.maxCenteredCoefficientNorm step.transition ≤ preimageBound ∧
      Nonempty (InputInjectionStateZero q ringDimension stateColumns nextBase next
        (Mxx.matrixMul signal step.selector)
        (MxxWe.propagatedStateNoise signal step.transitionError stateError step.transition)) := by
  let step : MxxWe.InputInjectionStep q ringDimension stateColumns := {
    state := iteration.current.state
    signal
    base
    stateError
    transition := iteration.transition.transition
    selector := iteration.transition.selector
    nextBase
    transitionError := iteration.transition.transitionError
    stateShape := iteration.current.stateShape
    signalShape := iteration.current.signalShape
    baseShape := iteration.current.baseShape
    stateErrorShape := iteration.current.errorShape
    transitionShape := iteration.transition.transitionShape
    selectorShape := iteration.transition.selectorShape
    nextBaseShape := iteration.transition.nextBaseShape
    transitionErrorShape := iteration.transition.transitionErrorShape
    stateEquation := iteration.current.equation
    transitionEquation := iteration.transition.equation
  }
  have nextStateShape := Mxx.Toolkit.matrixMul_shape iteration.current.state
    iteration.transition.transition iteration.current.stateShape
    iteration.transition.transitionShape
  have nextSignalShape := Mxx.Toolkit.matrixMul_shape signal iteration.transition.selector
    iteration.current.signalShape iteration.transition.selectorShape
  have signalErrorShape := Mxx.Toolkit.matrixMul_shape signal iteration.transition.transitionError
    iteration.current.signalShape iteration.transition.transitionErrorShape
  have oldErrorShape := Mxx.Toolkit.matrixMul_shape stateError iteration.transition.transition
    iteration.current.errorShape iteration.transition.transitionShape
  have nextErrorShape := Mxx.Toolkit.matrixAdd_shape
    (Mxx.matrixMul signal iteration.transition.transitionError)
    (Mxx.matrixMul stateError iteration.transition.transition) signalErrorShape oldErrorShape
  refine ⟨step, rfl, rfl, rfl, rfl, rfl, iteration.transition.selectorNorm,
    iteration.transition.transitionErrorNorm, iteration.transition.transitionNorm, ⟨?_⟩⟩
  exact {
    family := iteration.nextFamily
    state := iteration.nextState
    valuesEq := iteration.nextEq
    stateZero := iteration.nextZero
    stateShape := by simpa [iteration.nextStateEq] using nextStateShape
    signalShape := nextSignalShape
    baseShape := iteration.transition.nextBaseShape
    errorShape := nextErrorShape
    equation := by
      rw [iteration.nextStateEq,
        Mxx.Toolkit.matrixValue_mul q ringDimension 1 2 2 signal
          iteration.transition.selector
          ⟨iteration.current.signalShape.modulus,
            iteration.current.signalShape.ringDimension, iteration.current.signalShape.rows,
            iteration.current.signalShape.columns⟩
          ⟨iteration.transition.selectorShape.modulus,
            iteration.transition.selectorShape.ringDimension,
            iteration.transition.selectorShape.rows,
            iteration.transition.selectorShape.columns⟩]
      simpa [step] using step.value_equation
  }

/-- A callback-free, level-indexed refinement of the executable sequential-loop trace.  Each
constructor stores the exact checked child execution, the exact preprocessing-table entry, and the
state-zero family equation for that iteration. -/
inductive CheckedInputInjectionIterations
    (workflow : Mxx.Ir.Workflow) (layout : InputInjectionLayout)
    (stage : Mxx.Ir.Stage) (scope : Mxx.Ir.Scope) (fuel : Nat)
    (samplers : Mxx.MxxSamplerFamily) (params : Mxx.Ir.ParamEnvironment)
    (q ringDimension stateColumns errorBound preimageBound : Nat)
    [NeZero q] [NeZero ringDimension] [Fact (1 < q)]
    (baseAt : Nat → Mxx.Matrix) (indexSlot : Nat)
    (bindings : List (String × Mxx.Ir.IntExpr))
    (invariantArguments : List Mxx.Ir.Value) :
    Nat → List Nat → List Mxx.Ir.Value → List Mxx.Ir.Value →
      Nat × Nat → Mxx.Matrix → Mxx.Matrix → Prop
  | nil (level bounds signal stateError values)
      (represents : InputInjectionStateZero q ringDimension stateColumns (baseAt level) values
        signal stateError) :
      CheckedInputInjectionIterations workflow layout stage scope fuel samplers params q
        ringDimension stateColumns errorBound preimageBound baseAt indexSlot bindings
        invariantArguments level [] values values bounds signal stateError
  | cons (level index : Nat) (indices : List Nat) (state next final : List Mxx.Ir.Value)
      (bounds : Nat × Nat) (signal stateError : Mxx.Matrix)
      (indexEq : index = level)
      (evaluatedBindings : Mxx.Ir.ParamEnvironment)
      (bindingsEvaluate :
        Mxx.Ir.evaluateBindings ((.loopIndex indexSlot, .integer index) :: params) bindings =
          some evaluatedBindings)
      (childMember : next ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
        layout.bodyScope.definitionName
        (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
        (state ++ invariantArguments))
      (iteration : InputInjectionStateZeroIteration workflow layout stage scope fuel samplers
        (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
        (state ++ invariantArguments) state invariantArguments next q ringDimension stateColumns
        errorBound preimageBound level (baseAt level) (baseAt (level + 1)) signal stateError)
      (signalNorm : Mxx.maxCenteredCoefficientNorm signal ≤ bounds.1)
      (stateErrorNorm : Mxx.maxCenteredCoefficientNorm stateError ≤ bounds.2)
      (tail : CheckedInputInjectionIterations workflow layout stage scope fuel samplers params q
        ringDimension stateColumns errorBound preimageBound baseAt indexSlot bindings
        invariantArguments (level + 1) indices next final (MxxWe.injectionStep ringDimension 2
          stateColumns errorBound preimageBound bounds)
        (Mxx.matrixMul signal iteration.transition.selector)
        (MxxWe.propagatedStateNoise signal iteration.transition.transitionError stateError
          iteration.transition.transition)) :
      CheckedInputInjectionIterations workflow layout stage scope fuel samplers params q
        ringDimension stateColumns errorBound preimageBound baseAt indexSlot bindings
        invariantArguments level (index :: indices) state final bounds signal stateError

/-- The loop indices expected by the input injector, stated independently of the concrete list
constructor used by the IR interpreter. -/
inductive ConsecutiveInputLevels : Nat → List Nat → Prop
  | nil (level) : ConsecutiveInputLevels level []
  | cons (level : Nat) (indices : List Nat)
      (tail : ConsecutiveInputLevels (level + 1) indices) :
      ConsecutiveInputLevels level (level :: indices)

/-- `List.range'` is exactly the consecutive level sequence expected by the input injector. -/
theorem consecutiveInputLevels_range' (level count : Nat) :
    ConsecutiveInputLevels level (List.range' level count) := by
  induction count generalizing level with
  | zero => exact .nil level
  | succ count induction =>
      rw [List.range'_succ]
      exact .cons level (List.range' (level + 1) count) (induction (level := level + 1))

/-- `List.range` is the zero-based instance executed by the IR sequential loop. -/
theorem consecutiveInputLevels_range (count : Nat) :
    ConsecutiveInputLevels 0 (List.range count) := by
  simpa [List.range_eq_range'] using consecutiveInputLevels_range' 0 count

/-- Refine an exact sequential-loop trace into the checked input-injection recurrence.

The only protocol-specific premise constructs the semantic evidence for one already-retained
child execution.  The recursion itself derives both next-state bounds from that evidence and the
generic matrix hard-bound lemmas; it does not accept caller-supplied bounds for later levels. -/
theorem checkedInputInjectionIterations_of_sequentialTrace
    {workflow : Mxx.Ir.Workflow} {layout : InputInjectionLayout}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {q ringDimension stateColumns errorBound preimageBound : Nat}
    [NeZero q] [NeZero ringDimension] [Fact (1 < q)]
    {baseAt : Nat → Mxx.Matrix} {indexSlot : Nat}
    {bindings : List (String × Mxx.Ir.IntExpr)}
    {invariantArguments : List Mxx.Ir.Value}
    {level : Nat} {indices : List Nat} {initial final : List Mxx.Ir.Value}
    {initialBounds : Nat × Nat} {initialSignal initialError : Mxx.Matrix}
    (trace : Mxx.Ir.SequentialIterationsTrace
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1))
      layout.bodyScope.definitionName params indexSlot bindings invariantArguments
      indices initial final)
    (levels : ConsecutiveInputLevels level indices)
    (initialState : InputInjectionStateZero q ringDimension stateColumns (baseAt level)
      initial initialSignal initialError)
    (initialSignalNorm : Mxx.maxCenteredCoefficientNorm initialSignal ≤ initialBounds.1)
    (initialErrorNorm : Mxx.maxCenteredCoefficientNorm initialError ≤ initialBounds.2)
    (iterationOfChild :
      ∀ (currentLevel : Nat) (state next : List Mxx.Ir.Value)
        (signal stateError : Mxx.Matrix)
        (evaluatedBindings : Mxx.Ir.ParamEnvironment),
        Mxx.Ir.evaluateBindings
            ((.loopIndex indexSlot, .integer currentLevel) :: params) bindings =
          some evaluatedBindings →
        next ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
          layout.bodyScope.definitionName
          (evaluatedBindings ++
            ((.loopIndex indexSlot, .integer currentLevel) :: params))
          (state ++ invariantArguments) →
        InputInjectionStateZero q ringDimension stateColumns (baseAt currentLevel)
          state signal stateError →
        Nonempty (InputInjectionStateZeroIteration workflow layout stage scope fuel samplers
          (evaluatedBindings ++
            ((.loopIndex indexSlot, .integer currentLevel) :: params))
          (state ++ invariantArguments) state invariantArguments next q ringDimension
          stateColumns errorBound preimageBound currentLevel (baseAt currentLevel)
          (baseAt (currentLevel + 1)) signal stateError)) :
    Nonempty (CheckedInputInjectionIterations workflow layout stage scope fuel samplers params q
      ringDimension stateColumns errorBound preimageBound baseAt indexSlot bindings
      invariantArguments level indices initial final initialBounds initialSignal initialError) := by
  induction trace generalizing level initialBounds initialSignal initialError with
  | nil state =>
      exact ⟨.nil level initialBounds initialSignal initialError state initialState⟩
  | cons index indices state evaluatedBindings next final bindingsEvaluate childMember rest
      induction =>
      cases levels with
      | cons _ _ tailLevels =>
          obtain ⟨iteration⟩ := iterationOfChild index state next initialSignal initialError
            evaluatedBindings bindingsEvaluate childMember initialState
          obtain ⟨step, signalEq, stateErrorEq, selectorEq, transitionErrorEq, transitionEq,
            _selectorNorm, _transitionErrorNorm, _transitionNorm, nextStateFromStep⟩ :=
            iteration.stepAndNext
          have nextStateExact : Nonempty (InputInjectionStateZero q ringDimension stateColumns
              (baseAt (index + 1)) next
              (Mxx.matrixMul initialSignal iteration.transition.selector)
              (MxxWe.propagatedStateNoise initialSignal
                iteration.transition.transitionError initialError
                iteration.transition.transition)) := by
            simpa [signalEq, stateErrorEq, selectorEq, transitionErrorEq, transitionEq] using
              nextStateFromStep
          obtain ⟨nextState⟩ := nextStateExact
          have nextSignalNorm :
              Mxx.maxCenteredCoefficientNorm
                  (Mxx.matrixMul initialSignal iteration.transition.selector) ≤
                (MxxWe.injectionStep ringDimension 2 stateColumns errorBound preimageBound
                  initialBounds).1 := by
            simpa [MxxWe.injectionStep, MxxWe.productBound] using
              Mxx.Toolkit.matrixMul_norm_le q ringDimension 2 initialBounds.1 1 initialSignal
                iteration.transition.selector iteration.current.signalShape.modulus
                iteration.transition.selectorShape.modulus
                iteration.current.signalShape.ringDimension
                iteration.transition.selectorShape.ringDimension
                iteration.current.signalShape.columns iteration.transition.selectorShape.rows
                initialSignalNorm iteration.transition.selectorNorm
          have nextErrorNorm :
              Mxx.maxCenteredCoefficientNorm
                  (MxxWe.propagatedStateNoise initialSignal iteration.transition.transitionError
                    initialError iteration.transition.transition) ≤
                (MxxWe.injectionStep ringDimension 2 stateColumns errorBound preimageBound
                  initialBounds).2 := by
            simpa [MxxWe.injectionStep, MxxWe.productBound, Nat.add_comm] using
              MxxWe.propagatedStateNoise_norm_le q ringDimension 2 stateColumns initialBounds.1
                errorBound initialBounds.2 preimageBound initialSignal
                iteration.transition.transitionError initialError iteration.transition.transition
                iteration.current.signalShape iteration.transition.transitionErrorShape
                iteration.current.errorShape iteration.transition.transitionShape initialSignalNorm
                iteration.transition.transitionErrorNorm initialErrorNorm
                iteration.transition.transitionNorm
          obtain ⟨tail⟩ := induction tailLevels nextState nextSignalNorm nextErrorNorm
          exact ⟨.cons index index indices state next final initialBounds initialSignal initialError
            rfl evaluatedBindings bindingsEvaluate childMember iteration initialSignalNorm
            initialErrorNorm tail⟩

/-- Completed online input-injection result after all execution witnesses have been retained. -/
structure InputInjectionTraceOutcome
    {q ringDimension stateColumns errorBound preimageBound : Nat}
    [NeZero q] [NeZero ringDimension]
    (base : Mxx.Matrix) (finalValues : List Mxx.Ir.Value)
    (initialBounds finalBounds : Nat × Nat)
    (initialSignal initialError : Mxx.Matrix) where
  finalSignal : Mxx.Matrix
  finalError : Mxx.Matrix
  finalStateZero : InputInjectionStateZero q ringDimension stateColumns base finalValues
    finalSignal finalError
  trace : MxxWe.InputInjectionTrace q ringDimension stateColumns errorBound preimageBound
    initialBounds initialSignal initialError finalBounds finalSignal finalError

/-- Forget the executable witnesses only after every iteration has been tied to its typed child
execution and exact preprocessing-table entry. -/
theorem CheckedInputInjectionIterations.toInputInjectionTrace
    {workflow : Mxx.Ir.Workflow} {layout : InputInjectionLayout}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {q ringDimension stateColumns errorBound preimageBound : Nat}
    [NeZero q] [NeZero ringDimension] [Fact (1 < q)]
    {baseAt : Nat → Mxx.Matrix} {indexSlot : Nat}
    {bindings : List (String × Mxx.Ir.IntExpr)}
    {invariantArguments : List Mxx.Ir.Value} {level : Nat} {indices : List Nat}
    {initial final : List Mxx.Ir.Value} {initialBounds : Nat × Nat}
    {initialSignal initialError : Mxx.Matrix}
    (checked : CheckedInputInjectionIterations workflow layout stage scope fuel samplers params q
      ringDimension stateColumns errorBound preimageBound baseAt indexSlot bindings
      invariantArguments level indices initial final initialBounds initialSignal initialError) :
    ∃ finalSignal finalError,
      Nonempty (InputInjectionStateZero q ringDimension stateColumns
        (baseAt (level + indices.length)) final finalSignal finalError) ∧
      MxxWe.InputInjectionTrace q ringDimension stateColumns errorBound preimageBound
        initialBounds initialSignal initialError
        (indices.foldl (fun bounds _ ↦
          MxxWe.injectionStep ringDimension 2 stateColumns errorBound preimageBound bounds)
          initialBounds)
        finalSignal finalError := by
  induction checked with
  | nil level bounds signal stateError values represents =>
      exact ⟨signal, stateError, ⟨by simpa using represents⟩,
        .nil bounds signal stateError⟩
  | cons level index indices state next final bounds signal stateError indexEq evaluatedBindings
      bindingsEvaluate childMember iteration signalNorm stateErrorNorm tail induction =>
      obtain ⟨step, signalEq, stateErrorEq, selectorEq, transitionErrorEq, transitionEq,
        selectorNorm, transitionErrorNorm, transitionNorm, nextRepresents⟩ :=
        iteration.stepAndNext
      obtain ⟨finalSignal, finalError, finalRepresents, tailTrace⟩ := induction
      rw [← selectorEq, ← transitionErrorEq, ← transitionEq] at tailTrace
      refine ⟨finalSignal, finalError, ?_, ?_⟩
      · simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using finalRepresents
      · simpa [List.foldl] using MxxWe.InputInjectionTrace.cons step signalEq stateErrorEq
          signalNorm stateErrorNorm selectorNorm transitionErrorNorm transitionNorm tailTrace

/-- Package the completed abstract trace together with the exact final executable state-zero
identity. -/
theorem CheckedInputInjectionIterations.toOutcome
    {workflow : Mxx.Ir.Workflow} {layout : InputInjectionLayout}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {q ringDimension stateColumns errorBound preimageBound : Nat}
    [NeZero q] [NeZero ringDimension] [Fact (1 < q)]
    {baseAt : Nat → Mxx.Matrix} {indexSlot : Nat}
    {bindings : List (String × Mxx.Ir.IntExpr)}
    {invariantArguments : List Mxx.Ir.Value} {level : Nat} {indices : List Nat}
    {initial final : List Mxx.Ir.Value} {initialBounds : Nat × Nat}
    {initialSignal initialError : Mxx.Matrix}
    (checked : CheckedInputInjectionIterations workflow layout stage scope fuel samplers params q
      ringDimension stateColumns errorBound preimageBound baseAt indexSlot bindings
      invariantArguments level indices initial final initialBounds initialSignal initialError) :
    Nonempty (InputInjectionTraceOutcome (q := q) (ringDimension := ringDimension)
      (stateColumns := stateColumns) (errorBound := errorBound) (preimageBound := preimageBound)
      (baseAt (level + indices.length)) final initialBounds
      (indices.foldl (fun bounds _ ↦ MxxWe.injectionStep ringDimension 2 stateColumns errorBound
        preimageBound bounds) initialBounds) initialSignal initialError) := by
  obtain ⟨finalSignal, finalError, ⟨finalStateZero⟩, trace⟩ :=
    checked.toInputInjectionTrace
  exact ⟨{ finalSignal, finalError, finalStateZero, trace }⟩

end MxxWe.Certificate
