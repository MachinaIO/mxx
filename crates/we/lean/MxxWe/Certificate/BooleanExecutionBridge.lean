import MxxWe.Certificate.ExecutionBridge
import MxxWe.Certificate.VerifierSound
import MxxWe.GenericBooleanLayers

namespace MxxWe.Certificate

/-- Read the sole child result from a retained path once the checked scope output wire and its
value are known. -/
theorem ChildExecutionPath.singleOutput
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value}
    (path : ChildExecutionPath stage scope fuel samplers params inputs values)
    (wire : Mxx.Ir.WireRef) (value : Mxx.Ir.Value)
    (outputWire : scope.outputs.map Prod.snd = [wire])
    (lookup : Mxx.Ir.lookupWire wire path.finalWires = some value) :
    values = [value] := by
  rw [path.outputs]
  cases outputsEq : scope.outputs with
  | nil => simp [outputsEq] at outputWire
  | cons output tail =>
      cases tail with
      | nil =>
          rcases output with ⟨name, actualWire⟩
          rw [outputsEq] at outputWire
          have actualWireEq : actualWire = wire := by simpa using outputWire
          subst actualWire
          simp [Mxx.Ir.collectOutputs, lookup]
      | cons next rest =>
          have lengthEq := congrArg List.length outputWire
          simp [outputsEq] at lengthEq
/-- Exact executable node recovered from any checked local matrix-binary reference. -/
structure LocalMatrixBinaryResolution
    (workflow : Mxx.Ir.Workflow) (reference : MatrixBinaryRef)
    (expected : Mxx.Ir.NodeKind) : Prop where
  resolved : resolveNode workflow reference.operation = some {
    kind := expected
    arguments := [wireRef reference.left.wire, wireRef reference.right.wire]
    outputCount := 1
  }

theorem localMatrixBinaryResolution_of_verified
    {workflow : Mxx.Ir.Workflow} {reference : MatrixBinaryRef}
    {expected : Mxx.Ir.NodeKind}
    (verified : verifyMatrixBinary workflow reference expected = true) :
    LocalMatrixBinaryResolution workflow reference expected := by
  unfold verifyMatrixBinary at verified
  simp only [Bool.and_eq_true] at verified
  rcases verified with ⟨binary, kindAndCount⟩
  unfold verifyBinaryNode at binary
  simp only [Bool.and_eq_true, decide_eq_true_eq] at binary
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [resolved] at kindAndCount
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      simp only [resolved] at kindAndCount
      simp only [resolved] at binary
      exact ⟨by simp_all⟩

/-- Concrete subtraction selected at an exact local Boolean operation. -/
theorem LocalMatrixBinaryResolution.subtractOutcome
    {workflow : Mxx.Ir.Workflow} {reference : MatrixBinaryRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : LocalMatrixBinaryResolution workflow reference .matrixSubtract)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (left right : Mxx.Matrix)
    (argumentsEvaluate :
      [wireRef reference.left.wire, wireRef reference.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
        some [.matrix left, .matrix right]) :
    execution.values = [.matrix (Mxx.matrixSubtract left right)] := by
  have executionResolved := execution.resolved
  have operationResolved := resolution.resolved
  rw [executionResolved] at operationResolved
  have nodeEq := Option.some.inj operationResolved
  apply Mxx.Ir.mem_evaluateNode_matrixSubtract_of_arguments runChild samplers params inputs
    execution.before (wireRef reference.left.wire) (wireRef reference.right.wire) left right 1
    argumentsEvaluate
  simpa [nodeEq] using execution.member

/-- Concrete addition selected at an exact local Boolean operation. -/
theorem LocalMatrixBinaryResolution.addOutcome
    {workflow : Mxx.Ir.Workflow} {reference : MatrixBinaryRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : LocalMatrixBinaryResolution workflow reference .matrixAdd)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (left right : Mxx.Matrix)
    (argumentsEvaluate :
      [wireRef reference.left.wire, wireRef reference.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
        some [.matrix left, .matrix right]) :
    execution.values = [.matrix (Mxx.matrixAdd left right)] := by
  have executionResolved := execution.resolved
  have operationResolved := resolution.resolved
  rw [executionResolved] at operationResolved
  have nodeEq := Option.some.inj operationResolved
  apply Mxx.Ir.mem_evaluateNode_matrixAdd_of_arguments runChild samplers params inputs
    execution.before (wireRef reference.left.wire) (wireRef reference.right.wire) left right 1
    argumentsEvaluate
  simpa [nodeEq] using execution.member

/-- Concrete multiplication selected at an exact local Boolean operation. -/
theorem LocalMatrixBinaryResolution.multiplyOutcome
    {workflow : Mxx.Ir.Workflow} {reference : MatrixBinaryRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : LocalMatrixBinaryResolution workflow reference .matrixMultiply)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (left right : Mxx.Matrix)
    (argumentsEvaluate :
      [wireRef reference.left.wire, wireRef reference.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
        some [.matrix left, .matrix right]) :
    execution.values = [.matrix (Mxx.matrixMultiply left right)] := by
  have executionResolved := execution.resolved
  have operationResolved := resolution.resolved
  rw [executionResolved] at operationResolved
  have nodeEq := Option.some.inj operationResolved
  apply Mxx.Ir.mem_evaluateNode_matrixMultiply_of_arguments runChild samplers params inputs
    execution.before (wireRef reference.left.wire) (wireRef reference.right.wire) left right 1
    argumentsEvaluate
  simpa [nodeEq] using execution.member

/-- Read the exact argument equality carried by one checked operand. -/
theorem verifyOperand_argument_of_resolved
    {workflow : Mxx.Ir.Workflow} {reference : CoreOperandRef} {node : Mxx.Ir.Node}
    (verified : verifyOperand workflow reference = true)
    (resolved : resolveNode workflow reference.node = some node) :
    node.arguments[reference.operand]? = some (wireRef reference.wire) := by
  unfold verifyOperand at verified
  simp only [Bool.and_eq_true] at verified
  simp [resolved] at verified
  exact verified.2

/-- Exact executable node recovered from a checked dynamic family lookup. -/
structure DynamicFamilyGetResolution
    (workflow : Mxx.Ir.Workflow) (reference : DynamicFamilyGetRef)
    (family : CoreWireRef) : Prop where
  resolved : resolveNode workflow reference.operation = some {
    kind := .familyGetDynamic
    arguments := [wireRef family, wireRef reference.index.wire]
    outputCount := 1
  }

theorem dynamicFamilyGetResolution_of_verified
    {workflow : Mxx.Ir.Workflow} {reference : DynamicFamilyGetRef}
    {family : CoreWireRef} (verified : verifyDynamicGet workflow reference family = true) :
    DynamicFamilyGetResolution workflow reference family := by
  unfold verifyDynamicGet at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [resolved] at verified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> try simp_all
      have familyWire : reference.family.wire = family := by aesop
      have argumentsEq :
          arguments = [wireRef reference.family.wire, wireRef reference.index.wire] := by
        aesop
      have outputCountEq : outputCount = 1 := by aesop
      subst arguments
      subst outputCount
      exact ⟨by simpa [familyWire] using resolved⟩

/-- Runtime semantics of a checked dynamic family lookup, including the interpreter's explicit
out-of-range value. -/
theorem DynamicFamilyGetResolution.outcome
    {workflow : Mxx.Ir.Workflow} {reference : DynamicFamilyGetRef}
    {family : CoreWireRef} {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (resolution : DynamicFamilyGetResolution workflow reference family)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (members : List Mxx.Ir.Value) (index : Int)
    (argumentsEvaluate :
      [wireRef family, wireRef reference.index.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
        some [.family members, .integer index]) :
    execution.values = [members[index.toNat]?.getD
      (.invalid "FamilyGetDynamic index out of range")] := by
  have executionResolved := execution.resolved
  have lookupResolved := resolution.resolved
  rw [executionResolved] at lookupResolved
  have nodeEq := Option.some.inj lookupResolved
  simpa [nodeEq, Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate] using execution.member

/-- A checked parallel family-get child returns the interpreter's exact dynamic lookup result.
The proof reuses the one-source gather bridge because `ParallelFamilyGetRef` is precisely that
single-source shape. -/
theorem parallelFamilyGetChildOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelFamilyGetRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel index : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {source : List Mxx.Ir.Value} {values : List Mxx.Ir.Value}
    (verified : verifyParallelFamilyGet workflow reference = true)
    (ssaOrder : verifyScopeSsaOrder body = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition
      reference.parallelLoop.bodyScope.definitionName stage.program.definitions = some body)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.parallelLoop.bodyScope.definitionName params
      [.integer (Int.ofNat index), .family source]) :
    values = [source[index]?.getD (.invalid "FamilyGetDynamic index out of range")] := by
  let gather : ParallelGatherRef := {
    parallelLoop := reference.parallelLoop
    indexFamily := reference.indexFamily
    sourceFamilies := [reference.sourceFamily]
    bodyIndex := reference.bodyIndex
    bodySources := [reference.bodySource]
    gets := [reference.get]
    outputFamilies := [reference.outputFamily]
  }
  have gatherVerified : verifyParallelGather workflow gather = true := by
    have checked := verified
    unfold verifyParallelFamilyGet at checked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
    unfold verifyParallelGather
    simp only [gather, List.isEmpty_cons, Bool.not_false, List.length_cons, List.length_nil,
      Nat.reduceAdd, decide_true, List.map_cons, List.map_nil, List.zip_cons_cons,
      List.zip_nil_left, List.all_cons, List.all_nil, Bool.and_true]
    aesop
  exact oneSourceParallelGather_childOutcome gatherVerified rfl rfl rfl rfl bodyResolved
    definitionFound ssaOrder rfl childMember

/-- One dynamic family lookup selected from an actual child execution path. -/
structure DynamicFamilyGetExecution
    (workflow : Mxx.Ir.Workflow) (reference : DynamicFamilyGetRef)
    (family : CoreWireRef) (stage : Mxx.Ir.Stage) (scope : Mxx.Ir.Scope)
    (fuel : Nat) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : List Mxx.Ir.Value) where
  resolution : DynamicFamilyGetResolution workflow reference family
  execution : ReferencedNodeExecution workflow reference.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
    (scope.inputNames.zip inputs)

/-- Extract a checked lookup without asking the caller to identify an interpreter node or
provide its outcome. -/
theorem ChildExecutionPath.dynamicFamilyGetExecution
    {workflow : Mxx.Ir.Workflow} {reference : DynamicFamilyGetRef}
    {family : CoreWireRef} {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope}
    {fuel : Nat} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs outputs : List Mxx.Ir.Value}
    (path : ChildExecutionPath stage scope fuel samplers params inputs outputs)
    (verified : verifyDynamicGet workflow reference family = true)
    (scopeResolved : resolveScope workflow reference.operation = some scope) :
    Nonempty (DynamicFamilyGetExecution workflow reference family stage scope fuel samplers
      params inputs) := by
  let resolution := dynamicFamilyGetResolution_of_verified verified
  have nodeInScope := resolveNode_scopeNode scopeResolved resolution.resolved
  obtain ⟨execution⟩ := path.referencedNodeExecution nodeInScope resolution.resolved
  exact ⟨{ resolution, execution }⟩

/-- Exact executable parent node recovered from a role-checked parallel operation. -/
structure ExactParallelNodeResolution
    (workflow : Mxx.Ir.Workflow) (operation : CoreNodeRef)
    (count : Mxx.Ir.IntExpr) (indexSlot : Nat)
    (inputModes : List Mxx.Ir.LoopInputMode) where
  arguments : List Mxx.Ir.WireRef
  resolved : resolveNode workflow operation = some {
    kind := .parallelLoop
      (ScopeRef.parallelBody operation.scope operation.node).definitionName
      count indexSlot [] inputModes
    arguments
    outputCount := 1
  }

theorem exactParallelNodeResolution_of_verified
    {workflow : Mxx.Ir.Workflow} {operation : CoreNodeRef}
    {count : Mxx.Ir.IntExpr} {indexSlot : Nat}
    {inputModes : List Mxx.Ir.LoopInputMode}
    (verified : verifyExactParallelNodeRole workflow operation count indexSlot inputModes = true) :
    Nonempty (ExactParallelNodeResolution workflow operation count indexSlot inputModes) := by
  unfold verifyExactParallelNodeRole at verified
  cases resolved : resolveNode workflow operation with
  | none => simp [resolved] at verified
  | some node =>
      cases bodyResolved : resolveScope workflow {
          operation with scope := ScopeRef.parallelBody operation.scope operation.node } with
      | none =>
          simp [resolved, bodyResolved] at verified
      | some body =>
          rcases node with ⟨kind, arguments, outputCount⟩
          simp only [resolved, bodyResolved] at verified
          cases kind <;> try simp_all [Bool.and_eq_true, decide_eq_true_eq]
          exact ⟨{
            arguments
            resolved := by simp_all
          }⟩

/-- A checked `ParallelLoopRef` role induces the equivalent raw-node role used by retained-path
execution witnesses. -/
theorem exactParallelNodeRole_of_loopRole
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef}
    {count : Mxx.Ir.IntExpr} {indexSlot : Nat}
    {inputModes : List CertifiedLoopInputMode}
    (verified : verifyExactParallelLoopRole workflow reference count indexSlot inputModes = true) :
    verifyExactParallelNodeRole workflow reference.operation count indexSlot
      (inputModes.map CertifiedLoopInputMode.toIr) = true := by
  unfold verifyExactParallelLoopRole at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have loopVerified : verifyParallelLoop workflow reference = true := by aesop
  unfold verifyParallelLoop at loopVerified
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [resolved] at loopVerified
  | some node =>
      cases bodyResolved : resolveScope workflow {
          reference.operation with scope := reference.bodyScope } with
      | none => simp [resolved, bodyResolved] at loopVerified
      | some body =>
          rcases node with ⟨kind, arguments, outputCount⟩
          cases kind <;> try simp_all [verifyExactParallelNodeRole]
          have outputWires : List.map wireRef reference.bodyOutputs = scopeOutputWires body := by
            aesop
          have outputLength : reference.bodyOutputs.length = 1 := by aesop
          have mappedLength := congrArg List.length outputWires
          simpa [outputLength] using mappedLength.symm

/-- One role-checked parallel node selected from the retained child path. -/
structure ExactParallelNodeExecution
    (workflow : Mxx.Ir.Workflow) (operation : CoreNodeRef)
    (count : Mxx.Ir.IntExpr) (indexSlot : Nat)
    (inputModes : List Mxx.Ir.LoopInputMode)
    (stage : Mxx.Ir.Stage) (scope : Mxx.Ir.Scope) (fuel : Nat)
    (samplers : Mxx.MxxSamplerFamily) (params : Mxx.Ir.ParamEnvironment)
    (inputs : List Mxx.Ir.Value) where
  resolution : ExactParallelNodeResolution workflow operation count indexSlot inputModes
  execution : ReferencedNodeExecution workflow operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
    (scope.inputNames.zip inputs)

/-- Exact iteration trace of a role-checked one-output parallel node. -/
structure ExactParallelNodeTrace
    (workflow : Mxx.Ir.Workflow) (operation : CoreNodeRef)
    (count : Mxx.Ir.IntExpr) (indexSlot : Nat)
    (inputModes : List Mxx.Ir.LoopInputMode)
    (stage : Mxx.Ir.Stage) (scope : Mxx.Ir.Scope) (fuel : Nat)
    (samplers : Mxx.MxxSamplerFamily) (params : Mxx.Ir.ParamEnvironment)
    (inputs : List Mxx.Ir.Value)
    (execution : ExactParallelNodeExecution workflow operation count indexSlot inputModes
      stage scope fuel samplers params inputs) where
  argumentValues : List Mxx.Ir.Value
  evaluatedCount : Int
  argumentsEvaluate : execution.resolution.arguments.mapM
    (fun wire ↦ Mxx.Ir.lookupWire wire execution.execution.before) = some argumentValues
  countEvaluate : count.evaluate params = some evaluatedCount
  final : List (List Mxx.Ir.Value)
  iterations : Mxx.Ir.ParallelIterationsTrace
    (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel)
    (ScopeRef.parallelBody operation.scope operation.node).definitionName params indexSlot []
    inputModes argumentValues (List.range evaluatedCount.toNat) [[]] final
  valuesEq : execution.execution.values = final.map Mxx.Ir.Value.family

/-- Invert a successful exact parallel node once its concrete arguments and count are known. -/
theorem ExactParallelNodeExecution.trace
    {workflow : Mxx.Ir.Workflow} {operation : CoreNodeRef}
    {count : Mxx.Ir.IntExpr} {indexSlot : Nat}
    {inputModes : List Mxx.Ir.LoopInputMode}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : List Mxx.Ir.Value}
    (execution : ExactParallelNodeExecution workflow operation count indexSlot inputModes
      stage scope fuel samplers params inputs)
    (argumentValues : List Mxx.Ir.Value) (evaluatedCount : Int)
    (argumentsEvaluate : execution.resolution.arguments.mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire execution.execution.before) = some argumentValues)
    (countEvaluate : count.evaluate params = some evaluatedCount) :
    Nonempty (ExactParallelNodeTrace workflow operation count indexSlot inputModes stage scope
      fuel samplers params inputs execution) := by
  have executionResolved := execution.execution.resolved
  have loopResolved := execution.resolution.resolved
  rw [executionResolved] at loopResolved
  have nodeEq := Option.some.inj loopResolved
  have member : execution.execution.values ∈ Mxx.Ir.evaluateNode
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs) execution.execution.before {
        kind := .parallelLoop
          (ScopeRef.parallelBody operation.scope operation.node).definitionName
          count indexSlot [] inputModes
        arguments := execution.resolution.arguments
        outputCount := 1
      } := by
    simpa [nodeEq] using execution.execution.member
  obtain ⟨final, iterations, valuesEq⟩ :=
    (Mxx.Ir.mem_evaluateNode_parallelLoop_iff_trace
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs) execution.execution.before
      (ScopeRef.parallelBody operation.scope operation.node).definitionName count indexSlot []
      inputModes execution.resolution.arguments 1 argumentValues evaluatedCount argumentsEvaluate
      countEvaluate execution.execution.values).mp member
  exact ⟨{
    argumentValues := argumentValues
    evaluatedCount := evaluatedCount
    argumentsEvaluate := argumentsEvaluate
    countEvaluate := countEvaluate
    final := final
    iterations := iterations
    valuesEq := valuesEq
  }⟩

/-- A role-checked one-output parallel trace with exact child semantics returns the ordered
family of those child values. -/
theorem ExactParallelNodeTrace.onePortFamily
    {workflow : Mxx.Ir.Workflow} {operation : CoreNodeRef}
    {count : Mxx.Ir.IntExpr} {indexSlot : Nat}
    {inputModes : List Mxx.Ir.LoopInputMode}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : List Mxx.Ir.Value}
    {execution : ExactParallelNodeExecution workflow operation count indexSlot inputModes
      stage scope fuel samplers params inputs}
    (trace : ExactParallelNodeTrace workflow operation count indexSlot inputModes stage scope
      fuel samplers params inputs execution)
    (valueAt : Nat → Mxx.Ir.Value)
    (childExact : ∀ (index : Nat) evaluatedBindings childValues,
      Mxx.Ir.evaluateBindings
          ((.loopIndex indexSlot, .integer index) :: params) [] =
          some evaluatedBindings →
      childValues ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program fuel
        (ScopeRef.parallelBody operation.scope operation.node).definitionName
        (evaluatedBindings ++
          ((.loopIndex indexSlot, .integer index) :: params))
        ((inputModes.zip trace.argumentValues).map fun (mode, value) ↦
          Mxx.Ir.loopArgument mode index value) →
      childValues = [valueAt index]) :
    execution.execution.values =
      [.family ((List.range trace.evaluatedCount.toNat).map valueAt)] := by
  have initialEq : ([[]] : List (List Mxx.Ir.Value)) = [[]] := rfl
  have finalEq := parallelIterationsTrace_singlePortValues valueAt trace.iterations initialEq
    childExact
  rw [trace.valuesEq, finalEq]
  simp

/-- A concrete family output derives the successful argument/count evaluations of an exact
parallel node and excludes its invalid fallback. -/
theorem ExactParallelNodeExecution.trace_of_familyOutput
    {workflow : Mxx.Ir.Workflow} {operation : CoreNodeRef}
    {count : Mxx.Ir.IntExpr} {indexSlot : Nat}
    {inputModes : List Mxx.Ir.LoopInputMode}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : List Mxx.Ir.Value}
    (execution : ExactParallelNodeExecution workflow operation count indexSlot inputModes
      stage scope fuel samplers params inputs)
    (outputFamily : List Mxx.Ir.Value)
    (outputEq : execution.execution.values = [.family outputFamily]) :
    Nonempty (ExactParallelNodeTrace workflow operation count indexSlot inputModes stage scope
      fuel samplers params inputs execution) := by
  have executionResolved := execution.execution.resolved
  have loopResolved := execution.resolution.resolved
  rw [executionResolved] at loopResolved
  have nodeEq := Option.some.inj loopResolved
  have member : execution.execution.values ∈ Mxx.Ir.evaluateNode
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs) execution.execution.before {
        kind := .parallelLoop
          (ScopeRef.parallelBody operation.scope operation.node).definitionName
          count indexSlot [] inputModes
        arguments := execution.resolution.arguments
        outputCount := 1
      } := by
    simpa [nodeEq] using execution.execution.member
  cases argumentsEvaluate : execution.resolution.arguments.mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire execution.execution.before) with
  | none =>
      simp [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate] at member
      rw [member] at outputEq
      simp at outputEq
  | some argumentValues =>
      cases countEvaluate : count.evaluate params with
      | none =>
          simp [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate, countEvaluate] at member
          rw [member] at outputEq
          simp at outputEq
      | some evaluatedCount =>
          obtain ⟨final, iterations, valuesEq⟩ :=
            (Mxx.Ir.mem_evaluateNode_parallelLoop_iff_trace
              (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
              (scope.inputNames.zip inputs) execution.execution.before
              (ScopeRef.parallelBody operation.scope operation.node).definitionName count
              indexSlot [] inputModes execution.resolution.arguments 1 argumentValues
              evaluatedCount argumentsEvaluate countEvaluate execution.execution.values).mp
                member
          exact ⟨{
            argumentValues
            evaluatedCount
            argumentsEvaluate
            countEvaluate
            final
            iterations
            valuesEq
          }⟩

theorem ChildExecutionPath.exactParallelNodeExecution
    {workflow : Mxx.Ir.Workflow} {operation : CoreNodeRef}
    {count : Mxx.Ir.IntExpr} {indexSlot : Nat}
    {inputModes : List Mxx.Ir.LoopInputMode}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value}
    (path : ChildExecutionPath stage scope fuel samplers params inputs outputs)
    (verified : verifyExactParallelNodeRole workflow operation count indexSlot inputModes = true)
    (scopeResolved : resolveScope workflow operation = some scope) :
    Nonempty (ExactParallelNodeExecution workflow operation count indexSlot inputModes
      stage scope fuel samplers params inputs) := by
  obtain ⟨resolution⟩ := exactParallelNodeResolution_of_verified verified
  have nodeInScope := resolveNode_scopeNode scopeResolved resolution.resolved
  obtain ⟨execution⟩ := path.referencedNodeExecution nodeInScope resolution.resolved
  exact ⟨{ resolution, execution }⟩

/-- Role-checked parallel execution retained on the same selected child path. -/
theorem ChildExecutionPath.rootedExactParallelNodeExecution
    {workflow : Mxx.Ir.Workflow} {operation : CoreNodeRef}
    {count : Mxx.Ir.IntExpr} {indexSlot : Nat}
    {inputModes : List Mxx.Ir.LoopInputMode}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value}
    (path : ChildExecutionPath stage scope fuel samplers params inputs outputs)
    (verified : verifyExactParallelNodeRole workflow operation count indexSlot inputModes = true)
    (scopeResolved : resolveScope workflow operation = some scope) :
    ∃ execution : ExactParallelNodeExecution workflow operation count indexSlot inputModes
        stage scope fuel samplers params inputs,
      ChildPathRootedNodeExecution path execution.execution := by
  obtain ⟨resolution⟩ := exactParallelNodeResolution_of_verified verified
  have nodeInScope := resolveNode_scopeNode scopeResolved resolution.resolved
  obtain ⟨execution, rooted⟩ :=
    path.rootedReferencedNodeExecution nodeInScope resolution.resolved
  exact ⟨{ resolution, execution }, rooted⟩

/-- Exact executable node recovered from a checked six-way selection. -/
structure SixWaySelectResolution
    (workflow : Mxx.Ir.Workflow) (reference : SixWaySelectRef) : Prop where
  resolved : resolveNode workflow reference.operation = some {
    kind := .select
    arguments := wireRef reference.selector.wire ::
      (List.ofFn reference.branches).map (wireRef ∘ CoreOperandRef.wire)
    outputCount := 1
  }

theorem sixWaySelectResolution_of_verified
    {workflow : Mxx.Ir.Workflow} {reference : SixWaySelectRef}
    (verified : verifySixWaySelect workflow reference = true) :
    SixWaySelectResolution workflow reference := by
  unfold verifySixWaySelect verifySelect at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at verified
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [resolved] at verified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> try simp_all
      have selectorNode : reference.selector.node = reference.operation := by aesop
      have selectorOperand : reference.selector.operand = 0 := by aesop
      have selectorVerified : verifyOperand workflow reference.selector = true := by aesop
      have selectorArgument := verifyOperand_argument_of_resolved selectorVerified
        (by simpa [selectorNode] using resolved)
      have branchNode (i : Fin 6) : (reference.branches i).node = reference.operation := by
        fin_cases i <;> aesop
      have branchOperand (i : Fin 6) : (reference.branches i).operand = i.val + 1 := by
        fin_cases i <;> aesop
      have branchVerified (i : Fin 6) : verifyOperand workflow (reference.branches i) = true := by
        fin_cases i <;> aesop
      have branchArgument (i : Fin 6) :
          arguments[(reference.branches i).operand]? =
            some (wireRef (reference.branches i).wire) :=
        verifyOperand_argument_of_resolved (branchVerified i)
          (by simpa [branchNode i] using resolved)
      refine ⟨?_⟩
      rw [resolved]
      congr 2
      apply List.ext_getElem?'
      intro index indexLt
      simp only [List.length_cons, List.length_map, List.length_ofFn] at indexLt
      have indexBound : index < 7 := by omega
      interval_cases index
      all_goals simp only [List.getElem?_cons_zero, List.getElem?_cons_succ,
        List.getElem?_map, List.getElem?_ofFn]
      · rw [selectorOperand] at selectorArgument
        exact selectorArgument
      · have argument := branchArgument (0 : Fin 6)
        rw [branchOperand 0] at argument
        simpa using argument
      · have argument := branchArgument (1 : Fin 6)
        rw [branchOperand 1] at argument
        simpa using argument
      · have argument := branchArgument (2 : Fin 6)
        rw [branchOperand 2] at argument
        simpa using argument
      · have argument := branchArgument (3 : Fin 6)
        rw [branchOperand 3] at argument
        simpa using argument
      · have argument := branchArgument (4 : Fin 6)
        rw [branchOperand 4] at argument
        simpa using argument
      · have argument := branchArgument (5 : Fin 6)
        rw [branchOperand 5] at argument
        simpa using argument

/-- Exact executable node recovered from a checked two-way selection. -/
structure TwoWaySelectResolution
    (workflow : Mxx.Ir.Workflow) (reference : TwoWaySelectRef) : Prop where
  resolved : resolveNode workflow reference.operation = some {
    kind := .select
    arguments := wireRef reference.selector.wire ::
      (List.ofFn reference.branches).map (wireRef ∘ CoreOperandRef.wire)
    outputCount := 1
  }

theorem twoWaySelectResolution_of_verified
    {workflow : Mxx.Ir.Workflow} {reference : TwoWaySelectRef}
    (verified : verifyTwoWaySelect workflow reference = true) :
    TwoWaySelectResolution workflow reference := by
  unfold verifyTwoWaySelect verifySelect at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at verified
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [resolved] at verified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> try simp_all
      have selectorNode : reference.selector.node = reference.operation := by aesop
      have selectorOperand : reference.selector.operand = 0 := by aesop
      have selectorVerified : verifyOperand workflow reference.selector = true := by aesop
      have selectorArgument := verifyOperand_argument_of_resolved selectorVerified
        (by simpa [selectorNode] using resolved)
      have branchNode (i : Fin 2) : (reference.branches i).node = reference.operation := by
        fin_cases i <;> aesop
      have branchOperand (i : Fin 2) : (reference.branches i).operand = i.val + 1 := by
        fin_cases i <;> aesop
      have branchVerified (i : Fin 2) : verifyOperand workflow (reference.branches i) = true := by
        fin_cases i <;> aesop
      have branchArgument (i : Fin 2) :
          arguments[(reference.branches i).operand]? =
            some (wireRef (reference.branches i).wire) :=
        verifyOperand_argument_of_resolved (branchVerified i)
          (by simpa [branchNode i] using resolved)
      refine ⟨?_⟩
      rw [resolved]
      congr 2
      apply List.ext_getElem?'
      intro index indexLt
      simp only [List.length_cons, List.length_map, List.length_ofFn] at indexLt
      have indexBound : index < 3 := by omega
      interval_cases index
      all_goals simp only [List.getElem?_cons_zero, List.getElem?_cons_succ,
        List.getElem?_map, List.getElem?_ofFn]
      · rw [selectorOperand] at selectorArgument
        exact selectorArgument
      · have argument := branchArgument (0 : Fin 2)
        rw [branchOperand 0] at argument
        simpa using argument
      · have argument := branchArgument (1 : Fin 2)
        rw [branchOperand 1] at argument
        simpa using argument

/-- Concrete six-way selector semantics for a checked Boolean opcode.  Bounds are obtained from
the dynamic-circuit precondition, so an active gate never takes the interpreter's fallback. -/
theorem sixWaySelectOutcome_of_arguments
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    {wires : Mxx.Ir.WireEnvironment} {argumentRefs : List Mxx.Ir.WireRef}
    {index : Int} {branches values : List Mxx.Ir.Value}
    (branchCount : branches.length = 6)
    (indexLower : 0 ≤ index) (indexUpper : index ≤ 5)
    (argumentsEvaluate : argumentRefs.mapM (fun wire ↦ Mxx.Ir.lookupWire wire wires) =
      some (.integer index :: branches))
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .select
      arguments := argumentRefs
      outputCount := 1
    }) :
    ∃ indexLt : index.toNat < branches.length, values = [branches[index.toNat]] := by
  have indexLt : index.toNat < branches.length := by
    rw [branchCount]
    omega
  refine ⟨indexLt, ?_⟩
  have outcome := Mxx.Ir.mem_evaluateNode_select_of_arguments runChild samplers params inputs
    wires argumentRefs index branches 1 argumentsEvaluate member
  rw [List.getElem?_eq_getElem indexLt] at outcome
  simpa using outcome

/-- Concrete two-way selector semantics for an active-mask bit. -/
theorem twoWaySelectOutcome_of_arguments
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    {wires : Mxx.Ir.WireEnvironment} {argumentRefs : List Mxx.Ir.WireRef}
    {index : Int} {branches values : List Mxx.Ir.Value}
    (branchCount : branches.length = 2)
    (indexLower : 0 ≤ index) (indexUpper : index ≤ 1)
    (argumentsEvaluate : argumentRefs.mapM (fun wire ↦ Mxx.Ir.lookupWire wire wires) =
      some (.integer index :: branches))
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .select
      arguments := argumentRefs
      outputCount := 1
    }) :
    ∃ indexLt : index.toNat < branches.length, values = [branches[index.toNat]] := by
  have indexLt : index.toNat < branches.length := by
    rw [branchCount]
    omega
  refine ⟨indexLt, ?_⟩
  have outcome := Mxx.Ir.mem_evaluateNode_select_of_arguments runChild samplers params inputs
    wires argumentRefs index branches 1 argumentsEvaluate member
  rw [List.getElem?_eq_getElem indexLt] at outcome
  simpa using outcome

/-- The executable six-branch order is exactly the dynamic Boolean opcode order. -/
theorem booleanCandidate_at_opcode
    {α : Type} (zero one copy not product xor : α) (opcode : Int)
    (opcodeLower : 0 ≤ opcode) (opcodeUpper : opcode ≤ 5) :
    [zero, one, copy, not, product, xor][opcode.toNat]?.getD zero =
      match MxxWe.DynamicBoolean.gateKind opcode with
      | .constantFalse => zero
      | .constantTrue => one
      | .copy => copy
      | .not => not
      | .and => product
      | .xor => xor := by
  interval_cases opcode <;> rfl

/-- The executable active-mask branch order is inactive-zero followed by the opcode candidate. -/
theorem booleanActiveCandidate
    {α : Type} (zero candidate : α) (active : Bool) :
    [zero, candidate][Bool.toNat active]?.getD zero = if active then candidate else zero := by
  cases active <;> rfl

/-- The concrete checked six-way selector returns exactly the candidate named by the dynamic
Boolean opcode. -/
theorem SixWaySelectResolution.booleanGateOutcome
    {workflow : Mxx.Ir.Workflow} {reference : SixWaySelectRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : SixWaySelectResolution workflow reference)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (opcode : Int) (opcodeLower : 0 ≤ opcode) (opcodeUpper : opcode ≤ 5)
    (zero one copy not product xor : Mxx.Matrix)
    (argumentsEvaluate :
      (wireRef reference.selector.wire ::
        (List.ofFn reference.branches).map (wireRef ∘ CoreOperandRef.wire)).mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
        some [.integer opcode, .matrix zero, .matrix one, .matrix copy, .matrix not,
          .matrix product, .matrix xor]) :
    execution.values = [match MxxWe.DynamicBoolean.gateKind opcode with
      | .constantFalse => .matrix zero
      | .constantTrue => .matrix one
      | .copy => .matrix copy
      | .not => .matrix not
      | .and => .matrix product
      | .xor => .matrix xor] := by
  have executionResolved := execution.resolved
  have selectResolved := resolution.resolved
  rw [executionResolved] at selectResolved
  have nodeEq := Option.some.inj selectResolved
  have member : execution.values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs
      execution.before {
        kind := .select
        arguments := wireRef reference.selector.wire ::
          (List.ofFn reference.branches).map (wireRef ∘ CoreOperandRef.wire)
        outputCount := 1
      } := by simpa [nodeEq] using execution.member
  obtain ⟨_, outcome⟩ := sixWaySelectOutcome_of_arguments (by simp) opcodeLower
    opcodeUpper argumentsEvaluate member
  interval_cases opcode <;> simpa [MxxWe.DynamicBoolean.gateKind] using outcome

/-- The concrete checked active-mask selector returns zero for inactive slots and the opcode
candidate for active slots. -/
theorem TwoWaySelectResolution.activeGateOutcome
    {workflow : Mxx.Ir.Workflow} {reference : TwoWaySelectRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : TwoWaySelectResolution workflow reference)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (active : Bool) (zero candidate : Mxx.Matrix)
    (argumentsEvaluate :
      (wireRef reference.selector.wire ::
        (List.ofFn reference.branches).map (wireRef ∘ CoreOperandRef.wire)).mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
        some [.integer (Bool.toNat active), .matrix zero, .matrix candidate]) :
    execution.values = [.matrix (if active then candidate else zero)] := by
  have executionResolved := execution.resolved
  have selectResolved := resolution.resolved
  rw [executionResolved] at selectResolved
  have nodeEq := Option.some.inj selectResolved
  have member : execution.values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs
      execution.before {
        kind := .select
        arguments := wireRef reference.selector.wire ::
          (List.ofFn reference.branches).map (wireRef ∘ CoreOperandRef.wire)
        outputCount := 1
      } := by simpa [nodeEq] using execution.member
  have activeLower : 0 ≤ Int.ofNat (Bool.toNat active) := Int.natCast_nonneg _
  have activeUpper : Int.ofNat (Bool.toNat active) ≤ 1 := by cases active <;> simp
  obtain ⟨_, outcome⟩ := twoWaySelectOutcome_of_arguments (by simp) activeLower
    activeUpper argumentsEvaluate member
  cases active <;> simpa using outcome

/-- All executable local operations that implement one certified public-key Boolean gate. -/
structure LocalBooleanGateResolutions
    (workflow : Mxx.Ir.Workflow) (layout : LocalBooleanGateLayout) : Prop where
  zero : LocalMatrixBinaryResolution workflow layout.zero .matrixSubtract
  not : LocalMatrixBinaryResolution workflow layout.not .matrixSubtract
  product : LocalMatrixBinaryResolution workflow layout.product .matrixMultiply
  sum : LocalMatrixBinaryResolution workflow layout.sum .matrixAdd
  twoProduct : LocalMatrixBinaryResolution workflow layout.twoProduct .matrixMultiply
  xor : LocalMatrixBinaryResolution workflow layout.xor .matrixSubtract
  candidates : SixWaySelectResolution workflow layout.candidateSelect
  active : TwoWaySelectResolution workflow layout.activeSelect

/-- Runtime loop contract required for pointwise public-key gate semantics. -/
structure LocalBooleanParentLoopContract (layout : LocalBooleanGateLayout) : Prop where
  count : layout.parentLoop.count = .parameter "max_layer_width"
  bindings : layout.parentLoop.bindings = []
  inputModes : layout.parentLoop.inputModes =
    [.zip, .zip, .zip, .broadcast, .broadcast]

theorem localBooleanParentLoopContract_of_verified
    {workflow : Mxx.Ir.Workflow} {layout : LocalBooleanGateLayout}
    (verified : verifyLocalBooleanGate workflow layout = true) :
    LocalBooleanParentLoopContract layout := by
  unfold verifyLocalBooleanGate at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have role : verifyExactParallelLoopRole workflow layout.parentLoop
      (.parameter "max_layer_width") 1
      [.zip, .zip, .zip, .broadcast, .broadcast] = true := by
    aesop
  unfold verifyExactParallelLoopRole at role
  simp only [Bool.and_eq_true, decide_eq_true_eq] at role
  exact ⟨by aesop, by aesop, by aesop⟩

/-- Pointwise child arguments selected by the certified local Boolean loop modes. -/
theorem localBooleanLoopArguments
    (opcode left right : List Mxx.Ir.Value) (one active : Mxx.Ir.Value) (index : Nat) :
    (([.zip, .zip, .zip, .broadcast, .broadcast].zip
      [.family opcode, .family left, .family right, one, active]).map fun (mode, value) ↦
        Mxx.Ir.loopArgument mode index value) =
      [opcode[index]?.getD (.invalid "parallel zip index out of range"),
        left[index]?.getD (.invalid "parallel zip index out of range"),
        right[index]?.getD (.invalid "parallel zip index out of range"), one, active] := by
  rfl

theorem localBooleanGateResolutions_of_verified
    {workflow : Mxx.Ir.Workflow} {layout : LocalBooleanGateLayout}
    (verified : verifyLocalBooleanGate workflow layout = true) :
    LocalBooleanGateResolutions workflow layout := by
  unfold verifyLocalBooleanGate at verified
  simp only [Bool.and_eq_true] at verified
  have zero : verifyMatrixBinary workflow layout.zero .matrixSubtract = true := by aesop
  have not : verifyMatrixBinary workflow layout.not .matrixSubtract = true := by aesop
  have product : verifyMatrixBinary workflow layout.product .matrixMultiply = true := by aesop
  have sum : verifyMatrixBinary workflow layout.sum .matrixAdd = true := by aesop
  have twoProduct : verifyMatrixBinary workflow layout.twoProduct .matrixMultiply = true := by aesop
  have xor : verifyMatrixBinary workflow layout.xor .matrixSubtract = true := by aesop
  have candidates : verifySixWaySelect workflow layout.candidateSelect = true := by aesop
  have active : verifyTwoWaySelect workflow layout.activeSelect = true := by aesop
  exact {
    zero := localMatrixBinaryResolution_of_verified zero
    not := localMatrixBinaryResolution_of_verified not
    product := localMatrixBinaryResolution_of_verified product
    sum := localMatrixBinaryResolution_of_verified sum
    twoProduct := localMatrixBinaryResolution_of_verified twoProduct
    xor := localMatrixBinaryResolution_of_verified xor
    candidates := sixWaySelectResolution_of_verified candidates
    active := twoWaySelectResolution_of_verified active
  }

/-- The six public-key candidates are the exact results of the executable local matrix nodes.
Every equality is derived from the selected support member of that node. -/
theorem LocalBooleanGateResolutions.candidateOutcomes
    {workflow : Mxx.Ir.Workflow} {layout : LocalBooleanGateLayout}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolutions : LocalBooleanGateResolutions workflow layout)
    (zeroExecution : ReferencedNodeExecution workflow layout.zero.operation runChild samplers
      params inputs)
    (notExecution : ReferencedNodeExecution workflow layout.not.operation runChild samplers
      params inputs)
    (productExecution : ReferencedNodeExecution workflow layout.product.operation runChild samplers
      params inputs)
    (sumExecution : ReferencedNodeExecution workflow layout.sum.operation runChild samplers
      params inputs)
    (twoProductExecution : ReferencedNodeExecution workflow layout.twoProduct.operation runChild
      samplers params inputs)
    (xorExecution : ReferencedNodeExecution workflow layout.xor.operation runChild samplers
      params inputs)
    (one copy right decomposition two : Mxx.Matrix)
    (zeroArguments :
      [wireRef layout.zero.left.wire, wireRef layout.zero.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire zeroExecution.before) =
        some [.matrix one, .matrix one])
    (notArguments :
      [wireRef layout.not.left.wire, wireRef layout.not.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire notExecution.before) =
        some [.matrix one, .matrix copy])
    (productArguments :
      [wireRef layout.product.left.wire, wireRef layout.product.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire productExecution.before) =
        some [.matrix copy, .matrix decomposition])
    (sumArguments :
      [wireRef layout.sum.left.wire, wireRef layout.sum.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire sumExecution.before) =
        some [.matrix copy, .matrix right])
    (twoProductArguments :
      [wireRef layout.twoProduct.left.wire, wireRef layout.twoProduct.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire twoProductExecution.before) =
        some [.matrix (Mxx.matrixMultiply copy decomposition), .matrix two])
    (xorArguments :
      [wireRef layout.xor.left.wire, wireRef layout.xor.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire xorExecution.before) =
        some [.matrix (Mxx.matrixAdd copy right),
          .matrix (Mxx.matrixMultiply (Mxx.matrixMultiply copy decomposition) two)]) :
    zeroExecution.values = [.matrix (Mxx.matrixSubtract one one)] ∧
      notExecution.values = [.matrix (Mxx.matrixSubtract one copy)] ∧
      productExecution.values = [.matrix (Mxx.matrixMultiply copy decomposition)] ∧
      sumExecution.values = [.matrix (Mxx.matrixAdd copy right)] ∧
      twoProductExecution.values =
        [.matrix (Mxx.matrixMultiply (Mxx.matrixMultiply copy decomposition) two)] ∧
      xorExecution.values = [.matrix (Mxx.matrixSubtract (Mxx.matrixAdd copy right)
        (Mxx.matrixMultiply (Mxx.matrixMultiply copy decomposition) two))] := by
  exact ⟨resolutions.zero.subtractOutcome zeroExecution one one zeroArguments,
    resolutions.not.subtractOutcome notExecution one copy notArguments,
    resolutions.product.multiplyOutcome productExecution copy decomposition productArguments,
    resolutions.sum.addOutcome sumExecution copy right sumArguments,
    resolutions.twoProduct.multiplyOutcome twoProductExecution
      (Mxx.matrixMultiply copy decomposition) two twoProductArguments,
    resolutions.xor.subtractOutcome xorExecution (Mxx.matrixAdd copy right)
      (Mxx.matrixMultiply (Mxx.matrixMultiply copy decomposition) two) xorArguments⟩

/-- The checked local gate returns the exact dynamic Boolean public-key equation, including the
inactive-slot zero branch. -/
theorem LocalBooleanGateResolutions.gateOutcome
    {workflow : Mxx.Ir.Workflow} {layout : LocalBooleanGateLayout}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolutions : LocalBooleanGateResolutions workflow layout)
    (zeroExecution : ReferencedNodeExecution workflow layout.zero.operation runChild samplers
      params inputs)
    (notExecution : ReferencedNodeExecution workflow layout.not.operation runChild samplers
      params inputs)
    (productExecution : ReferencedNodeExecution workflow layout.product.operation runChild samplers
      params inputs)
    (sumExecution : ReferencedNodeExecution workflow layout.sum.operation runChild samplers
      params inputs)
    (twoProductExecution : ReferencedNodeExecution workflow layout.twoProduct.operation runChild
      samplers params inputs)
    (xorExecution : ReferencedNodeExecution workflow layout.xor.operation runChild samplers
      params inputs)
    (candidateExecution : ReferencedNodeExecution workflow layout.candidateSelect.operation
      runChild samplers params inputs)
    (activeExecution : ReferencedNodeExecution workflow layout.activeSelect.operation runChild
      samplers params inputs)
    (opcode : Int) (opcodeLower : 0 ≤ opcode) (opcodeUpper : opcode ≤ 5) (active : Bool)
    (one copy right decomposition two : Mxx.Matrix)
    (zeroArguments :
      [wireRef layout.zero.left.wire, wireRef layout.zero.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire zeroExecution.before) =
        some [.matrix one, .matrix one])
    (notArguments :
      [wireRef layout.not.left.wire, wireRef layout.not.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire notExecution.before) =
        some [.matrix one, .matrix copy])
    (productArguments :
      [wireRef layout.product.left.wire, wireRef layout.product.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire productExecution.before) =
        some [.matrix copy, .matrix decomposition])
    (sumArguments :
      [wireRef layout.sum.left.wire, wireRef layout.sum.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire sumExecution.before) =
        some [.matrix copy, .matrix right])
    (twoProductArguments :
      [wireRef layout.twoProduct.left.wire, wireRef layout.twoProduct.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire twoProductExecution.before) =
        some [.matrix (Mxx.matrixMultiply copy decomposition), .matrix two])
    (xorArguments :
      [wireRef layout.xor.left.wire, wireRef layout.xor.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire xorExecution.before) =
        some [.matrix (Mxx.matrixAdd copy right),
          .matrix (Mxx.matrixMultiply (Mxx.matrixMultiply copy decomposition) two)])
    (candidateArguments :
      (wireRef layout.candidateSelect.selector.wire ::
        (List.ofFn layout.candidateSelect.branches).map
          (wireRef ∘ CoreOperandRef.wire)).mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire candidateExecution.before) =
        some [.integer opcode, .matrix (Mxx.matrixSubtract one one), .matrix one, .matrix copy,
          .matrix (Mxx.matrixSubtract one copy),
          .matrix (Mxx.matrixMultiply copy decomposition),
          .matrix (Mxx.matrixSubtract (Mxx.matrixAdd copy right)
            (Mxx.matrixMultiply (Mxx.matrixMultiply copy decomposition) two))])
    (activeArguments :
      (wireRef layout.activeSelect.selector.wire ::
        (List.ofFn layout.activeSelect.branches).map
          (wireRef ∘ CoreOperandRef.wire)).mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire activeExecution.before) =
        some [.integer (Bool.toNat active), .matrix (Mxx.matrixSubtract one one),
          .matrix (match MxxWe.DynamicBoolean.gateKind opcode with
          | .constantFalse => Mxx.matrixSubtract one one
          | .constantTrue => one
          | .copy => copy
          | .not => Mxx.matrixSubtract one copy
          | .and => Mxx.matrixMultiply copy decomposition
          | .xor => Mxx.matrixSubtract (Mxx.matrixAdd copy right)
              (Mxx.matrixMultiply (Mxx.matrixMultiply copy decomposition) two))]) :
    activeExecution.values = [.matrix (if active then
      match MxxWe.DynamicBoolean.gateKind opcode with
      | .constantFalse => Mxx.matrixSubtract one one
      | .constantTrue => one
      | .copy => copy
      | .not => Mxx.matrixSubtract one copy
      | .and => Mxx.matrixMultiply copy decomposition
      | .xor => Mxx.matrixSubtract (Mxx.matrixAdd copy right)
          (Mxx.matrixMultiply (Mxx.matrixMultiply copy decomposition) two)
      else Mxx.matrixSubtract one one)] := by
  obtain ⟨zeroOutcome, notOutcome, productOutcome, sumOutcome, twoProductOutcome, xorOutcome⟩ :=
    resolutions.candidateOutcomes zeroExecution notExecution productExecution sumExecution
      twoProductExecution xorExecution one copy right decomposition two zeroArguments notArguments
      productArguments sumArguments twoProductArguments xorArguments
  have candidateOutcome := resolutions.candidates.booleanGateOutcome candidateExecution opcode
    opcodeLower opcodeUpper (Mxx.matrixSubtract one one) one copy
    (Mxx.matrixSubtract one copy) (Mxx.matrixMultiply copy decomposition)
    (Mxx.matrixSubtract (Mxx.matrixAdd copy right)
      (Mxx.matrixMultiply (Mxx.matrixMultiply copy decomposition) two)) candidateArguments
  have activeOutcome := resolutions.active.activeGateOutcome activeExecution active
    (Mxx.matrixSubtract one one)
    (match MxxWe.DynamicBoolean.gateKind opcode with
    | .constantFalse => Mxx.matrixSubtract one one
    | .constantTrue => one
    | .copy => copy
    | .not => Mxx.matrixSubtract one copy
    | .and => Mxx.matrixMultiply copy decomposition
    | .xor => Mxx.matrixSubtract (Mxx.matrixAdd copy right)
        (Mxx.matrixMultiply (Mxx.matrixMultiply copy decomposition) two)) activeArguments
  simpa using activeOutcome

/-! Concrete Boolean-encoding ghost invariant.

Runtime values carry only the three executable components.  The error matrix below is an
existential proof witness, retained across layers by the interpreted BGG equation. -/

/-- One runtime `(vector, publicKey, plaintext)` triple together with its concrete ghost error. -/
structure RuntimeBooleanEncoding (q ringDimension publicColumns : Nat) where
  vector : Mxx.Matrix
  publicKey : Mxx.Matrix
  plaintext : Mxx.Matrix
  error : Mxx.Matrix
  vectorLayout : Mxx.Toolkit.MatrixLayout vector q ringDimension 1 publicColumns
  publicKeyLayout : Mxx.Toolkit.MatrixLayout publicKey q ringDimension 1 publicColumns
  plaintextLayout : Mxx.Toolkit.MatrixLayout plaintext q ringDimension 1 1
  errorLayout : Mxx.Toolkit.MatrixLayout error q ringDimension 1 publicColumns

/-- Interpret the four concrete matrices as the reusable algebraic Boolean encoding. -/
noncomputable def RuntimeBooleanEncoding.toAlgebra
    {q ringDimension publicColumns : Nat}
    (encoding : RuntimeBooleanEncoding q ringDimension publicColumns) :
    MxxWe.BooleanEncoding (Mxx.Toolkit.Negacyclic q ringDimension) publicColumns where
  vector := Mxx.Toolkit.matrixValue q ringDimension 1 publicColumns encoding.vector
  publicKey := Mxx.Toolkit.matrixValue q ringDimension 1 publicColumns encoding.publicKey
  plaintext := Mxx.Toolkit.matrixValue q ringDimension 1 1 encoding.plaintext
  error := Mxx.Toolkit.matrixValue q ringDimension 1 publicColumns encoding.error

private theorem booleanEncoding_eq_of_fields
    {R : Type} {columns : Nat} {left right : MxxWe.BooleanEncoding R columns}
    (vector : left.vector = right.vector)
    (publicKey : left.publicKey = right.publicKey)
    (plaintext : left.plaintext = right.plaintext)
    (error : left.error = right.error) : left = right := by
  cases left
  cases right
  simp_all

/-- The inductive ghost invariant for one exact executable family triple. -/
structure RuntimeEncodingState
    (q ringDimension publicColumns : Nat)
    (secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1)
    (gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns) where
  encodings : List (RuntimeBooleanEncoding q ringDimension publicColumns)
  holds : ∀ i : Fin encodings.length,
    (encodings.get i).toAlgebra.Holds secret gadget

/-- The exact three family values carried by the executable encoding Boolean loop.  This is a
value-level relation, not a second semantic graph: every entry is one concrete runtime matrix. -/
def runtimeEncodingFamilyValues
    {q ringDimension publicColumns : Nat}
    (encodings : List (RuntimeBooleanEncoding q ringDimension publicColumns)) :
    List Mxx.Ir.Value := [
  .family (encodings.map fun encoding ↦ .matrix encoding.vector),
  .family (encodings.map fun encoding ↦ .matrix encoding.publicKey),
  .family (encodings.map fun encoding ↦ .matrix encoding.plaintext)]

/-- A sequential-loop state is represented only when its concrete vector, public-key, and
plaintext families are exactly the families of one runtime encoding state. -/
def RuntimeEncodingState.Represents
    {q ringDimension publicColumns : Nat}
    {secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1}
    {gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns}
    (state : RuntimeEncodingState q ringDimension publicColumns secret gadget)
    (values : List Mxx.Ir.Value) : Prop :=
  values = runtimeEncodingFamilyValues state.encodings

theorem RuntimeEncodingState.represents_length
    {q ringDimension publicColumns : Nat}
    {secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1}
    {gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns}
    (state : RuntimeEncodingState q ringDimension publicColumns secret gadget)
    {values : List Mxx.Ir.Value} (represents : state.Represents values) :
    values.length = 3 := by
  unfold RuntimeEncodingState.Represents at represents
  rw [represents]
  rfl

theorem RuntimeEncodingState.represents_families
    {q ringDimension publicColumns : Nat}
    {secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1}
    {gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns}
    (state : RuntimeEncodingState q ringDimension publicColumns secret gadget)
    {values : List Mxx.Ir.Value} (represents : state.Represents values) :
    values = [
      .family (state.encodings.map fun encoding ↦ .matrix encoding.vector),
      .family (state.encodings.map fun encoding ↦ .matrix encoding.publicKey),
      .family (state.encodings.map fun encoding ↦ .matrix encoding.plaintext)] := by
  exact represents

/-- Raw product triple used by the executable encoding-vector, public-key, and plaintext loops. -/
noncomputable def RuntimeBooleanEncoding.multiply
    {q ringDimension publicColumns : Nat}
    (left right : RuntimeBooleanEncoding q ringDimension publicColumns)
    (rightDecomposition : Mxx.Matrix)
    (decompositionLayout : Mxx.Toolkit.MatrixLayout rightDecomposition q ringDimension
      publicColumns publicColumns) :
    RuntimeBooleanEncoding q ringDimension publicColumns where
  vector := Mxx.matrixAdd (Mxx.matrixMultiply left.vector rightDecomposition)
    (Mxx.matrixMultiply left.plaintext right.vector)
  publicKey := Mxx.matrixMultiply left.publicKey rightDecomposition
  plaintext := Mxx.matrixMultiply left.plaintext right.plaintext
  error := MxxWe.productGateNoise left.error right.error left.plaintext rightDecomposition
  vectorLayout := Mxx.Toolkit.matrixAdd_layout _ _
    (Mxx.Toolkit.matrixMultiply_layout left.vector rightDecomposition left.vectorLayout
      decompositionLayout)
    (Mxx.Toolkit.matrixMultiply_layout left.plaintext right.vector left.plaintextLayout
      right.vectorLayout)
  publicKeyLayout := Mxx.Toolkit.matrixMultiply_layout left.publicKey rightDecomposition
    left.publicKeyLayout decompositionLayout
  plaintextLayout := Mxx.Toolkit.matrixMultiply_layout left.plaintext right.plaintext
    left.plaintextLayout right.plaintextLayout
  errorLayout := Mxx.Toolkit.matrixAdd_layout _ _
    (Mxx.Toolkit.matrixMul_layout left.error rightDecomposition left.errorLayout
      decompositionLayout)
    (Mxx.Toolkit.matrixMul_layout left.plaintext right.error left.plaintextLayout
      right.errorLayout)

private theorem matrixMul_one_by_one_as_algebraScale
    (q ringDimension columns : Nat) [NeZero q] [NeZero ringDimension]
    (scalar matrix : Mxx.Matrix)
    (scalarLayout : Mxx.Toolkit.MatrixLayout scalar q ringDimension 1 1)
    (matrixLayout : Mxx.Toolkit.MatrixLayout matrix q ringDimension 1 columns) :
    Mxx.Toolkit.matrixValue q ringDimension 1 columns (Mxx.matrixMul scalar matrix) =
      MxxWe.algebraScale (Mxx.Toolkit.matrixValue q ringDimension 1 1 scalar)
        (Mxx.Toolkit.matrixValue q ringDimension 1 columns matrix) := by
  rw [Mxx.Toolkit.matrixValue_mul q ringDimension 1 1 columns scalar matrix
    ⟨scalarLayout.modulus, scalarLayout.ringDimension, scalarLayout.rows,
      scalarLayout.columns⟩
    ⟨matrixLayout.modulus, matrixLayout.ringDimension, matrixLayout.rows,
      matrixLayout.columns⟩]
  ext row column
  fin_cases row
  simp [MxxWe.algebraScale, _root_.Matrix.mul_apply]

private theorem matrixMultiply_one_by_one_as_algebraScale
    (q ringDimension columns : Nat) [NeZero q] [NeZero ringDimension]
    (scalar matrix : Mxx.Matrix)
    (scalarLayout : Mxx.Toolkit.MatrixLayout scalar q ringDimension 1 1)
    (matrixLayout : Mxx.Toolkit.MatrixLayout matrix q ringDimension 1 columns) :
    Mxx.Toolkit.matrixValue q ringDimension 1 columns
        (Mxx.matrixMultiply scalar matrix) =
      MxxWe.algebraScale (Mxx.Toolkit.matrixValue q ringDimension 1 1 scalar)
        (Mxx.Toolkit.matrixValue q ringDimension 1 columns matrix) := by
  rw [Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension 1 1 columns scalar matrix
    scalarLayout matrixLayout]
  ext row column
  fin_cases row
  simp [MxxWe.algebraScale, _root_.Matrix.mul_apply]

/-- The raw runtime product has exactly the reusable algebraic product components. -/
theorem RuntimeBooleanEncoding.multiply_toAlgebra
    {q ringDimension publicColumns : Nat} [NeZero q] [NeZero ringDimension]
    [Fact (1 < q)]
    (left right : RuntimeBooleanEncoding q ringDimension publicColumns)
    (rightDecomposition : Mxx.Matrix)
    (decompositionLayout : Mxx.Toolkit.MatrixLayout rightDecomposition q ringDimension
      publicColumns publicColumns) :
    (left.multiply right rightDecomposition decompositionLayout).toAlgebra =
      left.toAlgebra.multiply right.toAlgebra
        (Mxx.Toolkit.matrixValue q ringDimension publicColumns publicColumns
          rightDecomposition) := by
  apply booleanEncoding_eq_of_fields
  · simp only [RuntimeBooleanEncoding.toAlgebra, RuntimeBooleanEncoding.multiply,
      MxxWe.BooleanEncoding.multiply]
    rw [Mxx.Toolkit.matrixValue_add q ringDimension 1 publicColumns]
    · rw [Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension 1 publicColumns
        publicColumns left.vector rightDecomposition left.vectorLayout decompositionLayout]
      rw [matrixMultiply_one_by_one_as_algebraScale q ringDimension publicColumns
        left.plaintext right.vector left.plaintextLayout right.vectorLayout]
    · exact ⟨(Mxx.Toolkit.matrixMultiply_layout left.vector rightDecomposition
        left.vectorLayout decompositionLayout).modulus,
        (Mxx.Toolkit.matrixMultiply_layout left.vector rightDecomposition
          left.vectorLayout decompositionLayout).ringDimension,
        (Mxx.Toolkit.matrixMultiply_layout left.vector rightDecomposition
          left.vectorLayout decompositionLayout).rows,
        (Mxx.Toolkit.matrixMultiply_layout left.vector rightDecomposition
          left.vectorLayout decompositionLayout).columns⟩
    · exact ⟨(Mxx.Toolkit.matrixMultiply_layout left.plaintext right.vector
        left.plaintextLayout right.vectorLayout).modulus,
        (Mxx.Toolkit.matrixMultiply_layout left.plaintext right.vector
          left.plaintextLayout right.vectorLayout).ringDimension,
        (Mxx.Toolkit.matrixMultiply_layout left.plaintext right.vector
          left.plaintextLayout right.vectorLayout).rows,
        (Mxx.Toolkit.matrixMultiply_layout left.plaintext right.vector
          left.plaintextLayout right.vectorLayout).columns⟩
  · simp only [RuntimeBooleanEncoding.toAlgebra, RuntimeBooleanEncoding.multiply,
      MxxWe.BooleanEncoding.multiply]
    exact Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension 1 publicColumns publicColumns
      left.publicKey rightDecomposition left.publicKeyLayout decompositionLayout
  · simp only [RuntimeBooleanEncoding.toAlgebra, RuntimeBooleanEncoding.multiply,
      MxxWe.BooleanEncoding.multiply]
    exact Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension 1 1 1 left.plaintext
      right.plaintext left.plaintextLayout right.plaintextLayout
  · simp only [RuntimeBooleanEncoding.toAlgebra, RuntimeBooleanEncoding.multiply,
      MxxWe.BooleanEncoding.multiply, MxxWe.productGateNoise]
    rw [Mxx.Toolkit.matrixValue_add q ringDimension 1 publicColumns]
    · rw [Mxx.Toolkit.matrixValue_mul q ringDimension 1 publicColumns publicColumns]
      · rw [matrixMul_one_by_one_as_algebraScale q ringDimension publicColumns
          left.plaintext right.error left.plaintextLayout right.errorLayout]
      · exact ⟨left.errorLayout.modulus, left.errorLayout.ringDimension,
          left.errorLayout.rows, left.errorLayout.columns⟩
      · exact ⟨decompositionLayout.modulus, decompositionLayout.ringDimension,
          decompositionLayout.rows, decompositionLayout.columns⟩
    · exact ⟨(Mxx.Toolkit.matrixMul_layout left.error rightDecomposition
        left.errorLayout decompositionLayout).modulus,
        (Mxx.Toolkit.matrixMul_layout left.error rightDecomposition
          left.errorLayout decompositionLayout).ringDimension,
        (Mxx.Toolkit.matrixMul_layout left.error rightDecomposition
          left.errorLayout decompositionLayout).rows,
        (Mxx.Toolkit.matrixMul_layout left.error rightDecomposition
          left.errorLayout decompositionLayout).columns⟩
    · exact ⟨(Mxx.Toolkit.matrixMul_layout left.plaintext right.error
        left.plaintextLayout right.errorLayout).modulus,
        (Mxx.Toolkit.matrixMul_layout left.plaintext right.error
          left.plaintextLayout right.errorLayout).ringDimension,
        (Mxx.Toolkit.matrixMul_layout left.plaintext right.error
          left.plaintextLayout right.errorLayout).rows,
        (Mxx.Toolkit.matrixMul_layout left.plaintext right.error
          left.plaintextLayout right.errorLayout).columns⟩

/-- The contract-backed decomposition equation preserves the BGG encoding invariant through the
runtime product operations. -/
theorem RuntimeBooleanEncoding.multiply_holds
    {q ringDimension publicColumns : Nat} [NeZero q] [NeZero ringDimension]
    [Fact (1 < q)]
    (secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1)
    (gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns)
    (left right : RuntimeBooleanEncoding q ringDimension publicColumns)
    (rightDecomposition : Mxx.Matrix)
    (decompositionLayout : Mxx.Toolkit.MatrixLayout rightDecomposition q ringDimension
      publicColumns publicColumns)
    (leftHolds : left.toAlgebra.Holds secret gadget)
    (rightHolds : right.toAlgebra.Holds secret gadget)
    (decomposes : gadget *
      Mxx.Toolkit.matrixValue q ringDimension publicColumns publicColumns rightDecomposition =
        right.toAlgebra.publicKey) :
    (left.multiply right rightDecomposition decompositionLayout).toAlgebra.Holds
      secret gadget := by
  rw [left.multiply_toAlgebra right rightDecomposition decompositionLayout]
  exact MxxWe.BooleanEncoding.multiply_holds secret gadget left.toAlgebra right.toAlgebra
    _ leftHolds rightHolds decomposes

/-- Componentwise runtime addition, including the corresponding ghost error. -/
noncomputable def RuntimeBooleanEncoding.add
    {q ringDimension publicColumns : Nat}
    (left right : RuntimeBooleanEncoding q ringDimension publicColumns) :
    RuntimeBooleanEncoding q ringDimension publicColumns where
  vector := Mxx.matrixAdd left.vector right.vector
  publicKey := Mxx.matrixAdd left.publicKey right.publicKey
  plaintext := Mxx.matrixAdd left.plaintext right.plaintext
  error := Mxx.matrixAdd left.error right.error
  vectorLayout := Mxx.Toolkit.matrixAdd_layout _ _ left.vectorLayout right.vectorLayout
  publicKeyLayout := Mxx.Toolkit.matrixAdd_layout _ _ left.publicKeyLayout right.publicKeyLayout
  plaintextLayout := Mxx.Toolkit.matrixAdd_layout _ _ left.plaintextLayout right.plaintextLayout
  errorLayout := Mxx.Toolkit.matrixAdd_layout _ _ left.errorLayout right.errorLayout

/-- Componentwise runtime subtraction, including the corresponding ghost error. -/
noncomputable def RuntimeBooleanEncoding.sub
    {q ringDimension publicColumns : Nat}
    (left right : RuntimeBooleanEncoding q ringDimension publicColumns) :
    RuntimeBooleanEncoding q ringDimension publicColumns where
  vector := Mxx.matrixSubtract left.vector right.vector
  publicKey := Mxx.matrixSubtract left.publicKey right.publicKey
  plaintext := Mxx.matrixSubtract left.plaintext right.plaintext
  error := Mxx.matrixSubtract left.error right.error
  vectorLayout := Mxx.Toolkit.matrixSubtract_layout _ _ left.vectorLayout right.vectorLayout
  publicKeyLayout := Mxx.Toolkit.matrixSubtract_layout _ _ left.publicKeyLayout
    right.publicKeyLayout
  plaintextLayout := Mxx.Toolkit.matrixSubtract_layout _ _ left.plaintextLayout
    right.plaintextLayout
  errorLayout := Mxx.Toolkit.matrixSubtract_layout _ _ left.errorLayout right.errorLayout

private theorem RuntimeBooleanEncoding.add_toAlgebra
    {q ringDimension publicColumns : Nat} [Fact (1 < q)] [NeZero ringDimension]
    (left right : RuntimeBooleanEncoding q ringDimension publicColumns) :
    (left.add right).toAlgebra = left.toAlgebra.add right.toAlgebra := by
  apply booleanEncoding_eq_of_fields
  all_goals simp only [RuntimeBooleanEncoding.toAlgebra, RuntimeBooleanEncoding.add,
    MxxWe.BooleanEncoding.add]
  · exact Mxx.Toolkit.matrixValue_add q ringDimension 1 publicColumns left.vector right.vector
      ⟨left.vectorLayout.modulus, left.vectorLayout.ringDimension, left.vectorLayout.rows,
        left.vectorLayout.columns⟩
      ⟨right.vectorLayout.modulus, right.vectorLayout.ringDimension, right.vectorLayout.rows,
        right.vectorLayout.columns⟩
  · exact Mxx.Toolkit.matrixValue_add q ringDimension 1 publicColumns left.publicKey
      right.publicKey
      ⟨left.publicKeyLayout.modulus, left.publicKeyLayout.ringDimension,
        left.publicKeyLayout.rows, left.publicKeyLayout.columns⟩
      ⟨right.publicKeyLayout.modulus, right.publicKeyLayout.ringDimension,
        right.publicKeyLayout.rows, right.publicKeyLayout.columns⟩
  · exact Mxx.Toolkit.matrixValue_add q ringDimension 1 1 left.plaintext right.plaintext
      ⟨left.plaintextLayout.modulus, left.plaintextLayout.ringDimension,
        left.plaintextLayout.rows, left.plaintextLayout.columns⟩
      ⟨right.plaintextLayout.modulus, right.plaintextLayout.ringDimension,
        right.plaintextLayout.rows, right.plaintextLayout.columns⟩
  · exact Mxx.Toolkit.matrixValue_add q ringDimension 1 publicColumns left.error right.error
      ⟨left.errorLayout.modulus, left.errorLayout.ringDimension, left.errorLayout.rows,
        left.errorLayout.columns⟩
      ⟨right.errorLayout.modulus, right.errorLayout.ringDimension, right.errorLayout.rows,
        right.errorLayout.columns⟩

private theorem RuntimeBooleanEncoding.sub_toAlgebra
    {q ringDimension publicColumns : Nat} [Fact (1 < q)] [NeZero ringDimension]
    (left right : RuntimeBooleanEncoding q ringDimension publicColumns) :
    (left.sub right).toAlgebra = left.toAlgebra.sub right.toAlgebra := by
  apply booleanEncoding_eq_of_fields
  all_goals simp only [RuntimeBooleanEncoding.toAlgebra, RuntimeBooleanEncoding.sub,
    MxxWe.BooleanEncoding.sub]
  · exact Mxx.Toolkit.matrixValue_subtract q ringDimension 1 publicColumns left.vector
      right.vector
      ⟨left.vectorLayout.modulus, left.vectorLayout.ringDimension, left.vectorLayout.rows,
        left.vectorLayout.columns⟩
      ⟨right.vectorLayout.modulus, right.vectorLayout.ringDimension, right.vectorLayout.rows,
        right.vectorLayout.columns⟩
  · exact Mxx.Toolkit.matrixValue_subtract q ringDimension 1 publicColumns left.publicKey
      right.publicKey
      ⟨left.publicKeyLayout.modulus, left.publicKeyLayout.ringDimension,
        left.publicKeyLayout.rows, left.publicKeyLayout.columns⟩
      ⟨right.publicKeyLayout.modulus, right.publicKeyLayout.ringDimension,
        right.publicKeyLayout.rows, right.publicKeyLayout.columns⟩
  · exact Mxx.Toolkit.matrixValue_subtract q ringDimension 1 1 left.plaintext right.plaintext
      ⟨left.plaintextLayout.modulus, left.plaintextLayout.ringDimension,
        left.plaintextLayout.rows, left.plaintextLayout.columns⟩
      ⟨right.plaintextLayout.modulus, right.plaintextLayout.ringDimension,
        right.plaintextLayout.rows, right.plaintextLayout.columns⟩
  · exact Mxx.Toolkit.matrixValue_subtract q ringDimension 1 publicColumns left.error
      right.error
      ⟨left.errorLayout.modulus, left.errorLayout.ringDimension, left.errorLayout.rows,
        left.errorLayout.columns⟩
      ⟨right.errorLayout.modulus, right.errorLayout.ringDimension, right.errorLayout.rows,
        right.errorLayout.columns⟩

private theorem RuntimeBooleanEncoding.add_holds
    {q ringDimension publicColumns : Nat} [Fact (1 < q)] [NeZero ringDimension]
    (secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1)
    (gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns)
    (left right : RuntimeBooleanEncoding q ringDimension publicColumns)
    (leftHolds : left.toAlgebra.Holds secret gadget)
    (rightHolds : right.toAlgebra.Holds secret gadget) :
    (left.add right).toAlgebra.Holds secret gadget := by
  rw [left.add_toAlgebra right]
  exact MxxWe.BooleanEncoding.add_holds secret gadget left.toAlgebra right.toAlgebra
    leftHolds rightHolds

private theorem RuntimeBooleanEncoding.sub_holds
    {q ringDimension publicColumns : Nat} [Fact (1 < q)] [NeZero ringDimension]
    (secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1)
    (gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns)
    (left right : RuntimeBooleanEncoding q ringDimension publicColumns)
    (leftHolds : left.toAlgebra.Holds secret gadget)
    (rightHolds : right.toAlgebra.Holds secret gadget) :
    (left.sub right).toAlgebra.Holds secret gadget := by
  rw [left.sub_toAlgebra right]
  exact MxxWe.BooleanEncoding.sub_holds secret gadget left.toAlgebra right.toAlgebra
    leftHolds rightHolds

/-- Runtime scaling by the checked constant-two matrix.  Executable components use the actual
matrix multiplication nodes; the ghost error uses the equivalent exact integer scale. -/
noncomputable def RuntimeBooleanEncoding.scaleTwo
    {q ringDimension publicColumns : Nat}
    (encoding : RuntimeBooleanEncoding q ringDimension publicColumns)
    (two : Mxx.Matrix) (twoLayout : Mxx.Toolkit.MatrixLayout two q ringDimension 1 1) :
    RuntimeBooleanEncoding q ringDimension publicColumns where
  vector := Mxx.matrixMultiply encoding.vector two
  publicKey := Mxx.matrixMultiply encoding.publicKey two
  plaintext := Mxx.matrixMultiply encoding.plaintext two
  error := Mxx.matrixScale 2 encoding.error
  vectorLayout := by
    by_cases columnsOne : publicColumns = 1
    · subst publicColumns
      exact Mxx.Toolkit.matrixMultiply_layout encoding.vector two encoding.vectorLayout twoLayout
    · rw [Mxx.Toolkit.matrixMultiply_rightScalar encoding.vector two
          encoding.vectorLayout.toMatrixShape twoLayout.toMatrixShape columnsOne]
      exact Mxx.Toolkit.matrixMul_layout two encoding.vector twoLayout encoding.vectorLayout
  publicKeyLayout := by
    by_cases columnsOne : publicColumns = 1
    · subst publicColumns
      exact Mxx.Toolkit.matrixMultiply_layout encoding.publicKey two encoding.publicKeyLayout
        twoLayout
    · rw [Mxx.Toolkit.matrixMultiply_rightScalar encoding.publicKey two
          encoding.publicKeyLayout.toMatrixShape twoLayout.toMatrixShape columnsOne]
      exact Mxx.Toolkit.matrixMul_layout two encoding.publicKey twoLayout
        encoding.publicKeyLayout
  plaintextLayout := Mxx.Toolkit.matrixMultiply_layout encoding.plaintext two
    encoding.plaintextLayout twoLayout
  errorLayout := Mxx.Toolkit.matrixScale_layout 2 encoding.error encoding.errorLayout

private theorem matrixTwo_entry {R : Type} [CommRing R] :
    (2 : MxxWe.AlgebraMatrix R 1 1) 0 0 = (2 : R) := by
  rfl

private theorem matrixTwo_mul {R : Type} [CommRing R] {columns : Nat}
    (matrix : MxxWe.AlgebraMatrix R 1 columns) :
    (2 : MxxWe.AlgebraMatrix R 1 1) * matrix =
      MxxWe.algebraScale (2 : MxxWe.AlgebraMatrix R 1 1) matrix := by
  ext row column
  fin_cases row
  simp [MxxWe.algebraScale, _root_.Matrix.mul_apply, matrixTwo_entry]

private theorem intSmulTwo_error {R : Type} [CommRing R] {columns : Nat}
    (matrix : MxxWe.AlgebraMatrix R 1 columns) :
    ((2 : Int) : R) • matrix =
      MxxWe.algebraScale (2 : MxxWe.AlgebraMatrix R 1 1) matrix := by
  ext row column
  fin_cases row
  simp [MxxWe.algebraScale, matrixTwo_entry]

private theorem oneByOne_mul_two_comm {R : Type} [CommRing R]
    (matrix : MxxWe.AlgebraMatrix R 1 1) :
    matrix * (2 : MxxWe.AlgebraMatrix R 1 1) =
      (2 : MxxWe.AlgebraMatrix R 1 1) * matrix := by
  ext row column
  fin_cases row
  fin_cases column
  simp [_root_.Matrix.mul_apply, matrixTwo_entry, mul_comm]

private theorem matrixMultiply_by_two_as_algebraScale
    (q ringDimension columns : Nat) [NeZero q] [NeZero ringDimension]
    (matrix two : Mxx.Matrix)
    (matrixLayout : Mxx.Toolkit.MatrixLayout matrix q ringDimension 1 columns)
    (twoLayout : Mxx.Toolkit.MatrixLayout two q ringDimension 1 1)
    (twoValue : Mxx.Toolkit.matrixValue q ringDimension 1 1 two =
      (2 : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1)) :
    Mxx.Toolkit.matrixValue q ringDimension 1 columns (Mxx.matrixMultiply matrix two) =
      MxxWe.algebraScale
        (2 : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1)
        (Mxx.Toolkit.matrixValue q ringDimension 1 columns matrix) := by
  by_cases columnsOne : columns = 1
  · subst columns
    rw [Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension 1 1 1 matrix two
      matrixLayout twoLayout, twoValue]
    rw [oneByOne_mul_two_comm, matrixTwo_mul]
  · rw [Mxx.Toolkit.matrixMultiply_rightScalar matrix two matrixLayout.toMatrixShape
      twoLayout.toMatrixShape columnsOne]
    rw [Mxx.Toolkit.matrixValue_mul q ringDimension 1 1 columns two matrix
      ⟨twoLayout.modulus, twoLayout.ringDimension, twoLayout.rows, twoLayout.columns⟩
      ⟨matrixLayout.modulus, matrixLayout.ringDimension, matrixLayout.rows,
        matrixLayout.columns⟩, twoValue]
    exact matrixTwo_mul _

private theorem RuntimeBooleanEncoding.scaleTwo_toAlgebra
    {q ringDimension publicColumns : Nat} [NeZero q] [NeZero ringDimension]
    [Fact (1 < q)]
    (encoding : RuntimeBooleanEncoding q ringDimension publicColumns)
    (two : Mxx.Matrix) (twoLayout : Mxx.Toolkit.MatrixLayout two q ringDimension 1 1)
    (twoValue : Mxx.Toolkit.matrixValue q ringDimension 1 1 two =
      (2 : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1)) :
    (encoding.scaleTwo two twoLayout).toAlgebra = encoding.toAlgebra.scale 2 := by
  apply booleanEncoding_eq_of_fields
  · exact matrixMultiply_by_two_as_algebraScale q ringDimension publicColumns
      encoding.vector two encoding.vectorLayout twoLayout twoValue
  · exact matrixMultiply_by_two_as_algebraScale q ringDimension publicColumns
      encoding.publicKey two encoding.publicKeyLayout twoLayout twoValue
  · calc
      Mxx.Toolkit.matrixValue q ringDimension 1 1
          (Mxx.matrixMultiply encoding.plaintext two) =
        MxxWe.algebraScale
          (2 : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1)
          (Mxx.Toolkit.matrixValue q ringDimension 1 1 encoding.plaintext) :=
            matrixMultiply_by_two_as_algebraScale q ringDimension 1 encoding.plaintext two
              encoding.plaintextLayout twoLayout twoValue
      _ = (2 : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1) *
          Mxx.Toolkit.matrixValue q ringDimension 1 1 encoding.plaintext :=
            (matrixTwo_mul _).symm
  · simp only [RuntimeBooleanEncoding.toAlgebra, RuntimeBooleanEncoding.scaleTwo,
      MxxWe.BooleanEncoding.scale]
    rw [Mxx.Toolkit.matrixValue_scale q ringDimension 1 publicColumns 2 encoding.error
      ⟨encoding.errorLayout.modulus, encoding.errorLayout.ringDimension,
        encoding.errorLayout.rows, encoding.errorLayout.columns⟩]
    exact intSmulTwo_error (R := Mxx.Toolkit.Negacyclic q ringDimension)
      (columns := publicColumns)
      (Mxx.Toolkit.matrixValue q ringDimension 1 publicColumns encoding.error)

/-- Concrete runtime Boolean candidate selected by the opcode. -/
noncomputable def RuntimeBooleanEncoding.applyGate
    {q ringDimension publicColumns : Nat}
    (gate : MxxWe.BooleanGate)
    (one left right product : RuntimeBooleanEncoding q ringDimension publicColumns)
    (two : Mxx.Matrix) (twoLayout : Mxx.Toolkit.MatrixLayout two q ringDimension 1 1) :
    RuntimeBooleanEncoding q ringDimension publicColumns :=
  match gate with
  | .constantFalse => one.sub one
  | .constantTrue => one
  | .copy => left
  | .not => one.sub left
  | .and => product
  | .xor => (left.add right).sub (product.scaleTwo two twoLayout)

private theorem RuntimeBooleanEncoding.applyGate_toAlgebra
    {q ringDimension publicColumns : Nat} [NeZero q] [NeZero ringDimension]
    [Fact (1 < q)]
    (gate : MxxWe.BooleanGate)
    (one left right product : RuntimeBooleanEncoding q ringDimension publicColumns)
    (two : Mxx.Matrix) (twoLayout : Mxx.Toolkit.MatrixLayout two q ringDimension 1 1)
    (twoValue : Mxx.Toolkit.matrixValue q ringDimension 1 1 two =
      (2 : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1)) :
    (applyGate gate one left right product two twoLayout).toAlgebra =
      MxxWe.BooleanEncoding.applyGate gate one.toAlgebra left.toAlgebra right.toAlgebra
        product.toAlgebra := by
  cases gate
  · exact one.sub_toAlgebra one
  · rfl
  · rfl
  · exact one.sub_toAlgebra left
  · rfl
  · rw [RuntimeBooleanEncoding.applyGate, RuntimeBooleanEncoding.sub_toAlgebra,
      RuntimeBooleanEncoding.add_toAlgebra,
      RuntimeBooleanEncoding.scaleTwo_toAlgebra product two twoLayout twoValue,
      MxxWe.BooleanEncoding.applyGate]

theorem RuntimeBooleanEncoding.applyGate_holds
    {q ringDimension publicColumns : Nat} [NeZero q] [NeZero ringDimension]
    [Fact (1 < q)]
    (secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1)
    (gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns)
    (gate : MxxWe.BooleanGate)
    (one left right product : RuntimeBooleanEncoding q ringDimension publicColumns)
    (two : Mxx.Matrix) (twoLayout : Mxx.Toolkit.MatrixLayout two q ringDimension 1 1)
    (twoValue : Mxx.Toolkit.matrixValue q ringDimension 1 1 two =
      (2 : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1))
    (oneHolds : one.toAlgebra.Holds secret gadget)
    (leftHolds : left.toAlgebra.Holds secret gadget)
    (rightHolds : right.toAlgebra.Holds secret gadget)
    (productHolds : product.toAlgebra.Holds secret gadget) :
    (applyGate gate one left right product two twoLayout).toAlgebra.Holds secret gadget := by
  rw [applyGate_toAlgebra gate one left right product two twoLayout twoValue]
  exact MxxWe.BooleanEncoding.applyGate_holds secret gadget gate one.toAlgebra left.toAlgebra
    right.toAlgebra product.toAlgebra oneHolds leftHolds rightHolds productHolds

/-- The ghost error selected by the concrete runtime candidate is exactly the simulator's dynamic
Boolean gate error, not merely bounded by it. -/
theorem RuntimeBooleanEncoding.applyGate_error
    {q ringDimension publicColumns : Nat} (gate : MxxWe.BooleanGate)
    (one left right product : RuntimeBooleanEncoding q ringDimension publicColumns)
    (rightDecomposition two : Mxx.Matrix)
    (decompositionLayout : Mxx.Toolkit.MatrixLayout rightDecomposition q ringDimension
      publicColumns publicColumns)
    (twoLayout : Mxx.Toolkit.MatrixLayout two q ringDimension 1 1)
    (productEq : product = left.multiply right rightDecomposition decompositionLayout) :
    (applyGate gate one left right product two twoLayout).error =
      MxxWe.booleanGateNoiseMatrix gate one.error left.error right.error left.plaintext
        rightDecomposition := by
  subst product
  cases gate <;> rfl

/-- Concrete encoding state produced by one full dynamic Boolean layer. -/
noncomputable def runtimeBooleanEncodingLayer
    {q ringDimension publicColumns maxWidth : Nat}
    (layer : MxxWe.BooleanLayerProgram)
    (previous : List (RuntimeBooleanEncoding q ringDimension publicColumns))
    (one : RuntimeBooleanEncoding q ringDimension publicColumns)
    (valid : layer.Valid previous.length maxWidth)
    (rightDecompositions : Fin layer.activeWidth → Mxx.Matrix)
    (decompositionLayouts : ∀ i : Fin layer.activeWidth,
      Mxx.Toolkit.MatrixLayout (rightDecompositions i) q ringDimension
        publicColumns publicColumns)
    (two : Mxx.Matrix) (twoLayout : Mxx.Toolkit.MatrixLayout two q ringDimension 1 1) :
    List (RuntimeBooleanEncoding q ringDimension publicColumns) :=
  List.ofFn fun i : Fin maxWidth ↦
    if active : i < layer.activeWidth then
      let gate : Fin layer.activeWidth := ⟨i, active⟩
      let left := previous.get (layer.leftIndex valid gate)
      let right := previous.get (layer.rightIndex valid gate)
      let product := left.multiply right (rightDecompositions gate)
        (decompositionLayouts gate)
      RuntimeBooleanEncoding.applyGate (layer.kinds.get gate) one left right product two
        twoLayout
    else one.sub one

/-- One checked dynamic layer preserves the interpreted BGG equation for every active and padded
slot. -/
theorem runtimeBooleanEncodingLayer_holds
    {q ringDimension publicColumns maxWidth : Nat} [NeZero q] [NeZero ringDimension]
    [Fact (1 < q)]
    (secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1)
    (gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns)
    (layer : MxxWe.BooleanLayerProgram)
    (previous : List (RuntimeBooleanEncoding q ringDimension publicColumns))
    (one : RuntimeBooleanEncoding q ringDimension publicColumns)
    (valid : layer.Valid previous.length maxWidth)
    (rightDecompositions : Fin layer.activeWidth → Mxx.Matrix)
    (decompositionLayouts : ∀ i : Fin layer.activeWidth,
      Mxx.Toolkit.MatrixLayout (rightDecompositions i) q ringDimension
        publicColumns publicColumns)
    (two : Mxx.Matrix) (twoLayout : Mxx.Toolkit.MatrixLayout two q ringDimension 1 1)
    (twoValue : Mxx.Toolkit.matrixValue q ringDimension 1 1 two =
      (2 : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1))
    (oneHolds : one.toAlgebra.Holds secret gadget)
    (previousHolds : ∀ i : Fin previous.length,
      (previous.get i).toAlgebra.Holds secret gadget)
    (decomposes : ∀ i : Fin layer.activeWidth,
      gadget * Mxx.Toolkit.matrixValue q ringDimension publicColumns publicColumns
        (rightDecompositions i) =
      (previous.get (layer.rightIndex valid i)).toAlgebra.publicKey) :
    ∀ i : Fin (runtimeBooleanEncodingLayer layer previous one valid rightDecompositions
      decompositionLayouts two twoLayout).length,
      ((runtimeBooleanEncodingLayer layer previous one valid rightDecompositions
        decompositionLayouts two twoLayout).get i).toAlgebra.Holds secret gadget := by
  intro i
  have indexLt : i.val < maxWidth := by
    simpa [runtimeBooleanEncodingLayer] using i.isLt
  by_cases active : i.val < layer.activeWidth
  · let gate : Fin layer.activeWidth := ⟨i, active⟩
    let left := previous.get (layer.leftIndex valid gate)
    let right := previous.get (layer.rightIndex valid gate)
    let product := left.multiply right (rightDecompositions gate)
      (decompositionLayouts gate)
    have productHolds : product.toAlgebra.Holds secret gadget :=
      left.multiply_holds secret gadget right (rightDecompositions gate)
        (decompositionLayouts gate) (previousHolds (layer.leftIndex valid gate))
        (previousHolds (layer.rightIndex valid gate)) (decomposes gate)
    simpa [runtimeBooleanEncodingLayer, active, gate, left, right, product] using
      RuntimeBooleanEncoding.applyGate_holds secret gadget (layer.kinds.get gate) one left right
        product two twoLayout twoValue oneHolds
        (previousHolds (layer.leftIndex valid gate))
        (previousHolds (layer.rightIndex valid gate)) productHolds
  · simpa [runtimeBooleanEncodingLayer, active] using
      one.sub_holds secret gadget one oneHolds oneHolds

/-- Concrete error family produced by one dynamic Boolean layer, including inactive slots. -/
noncomputable def runtimeBooleanNoiseLayer
    (maxWidth : Nat) (layer : MxxWe.BooleanLayerProgram)
    (previousErrors : List Mxx.Matrix) (oneError : Mxx.Matrix)
    (valid : layer.Valid previousErrors.length maxWidth)
    (leftPlaintexts rightDecompositions : Fin layer.activeWidth → Mxx.Matrix) :
    List Mxx.Matrix :=
  List.ofFn fun i : Fin maxWidth ↦
    if active : i < layer.activeWidth then
      let gate : Fin layer.activeWidth := ⟨i, active⟩
      MxxWe.booleanGateNoiseMatrix (layer.kinds.get gate) oneError
        (previousErrors.get (layer.leftIndex valid gate))
        (previousErrors.get (layer.rightIndex valid gate))
        (leftPlaintexts gate) (rightDecompositions gate)
    else Mxx.matrixSubtract oneError oneError

theorem runtimeBooleanEncodingLayer_errors
    {q ringDimension publicColumns maxWidth : Nat}
    (layer : MxxWe.BooleanLayerProgram)
    (previous : List (RuntimeBooleanEncoding q ringDimension publicColumns))
    (one : RuntimeBooleanEncoding q ringDimension publicColumns)
    (valid : layer.Valid previous.length maxWidth)
    (rightDecompositions : Fin layer.activeWidth → Mxx.Matrix)
    (decompositionLayouts : ∀ i : Fin layer.activeWidth,
      Mxx.Toolkit.MatrixLayout (rightDecompositions i) q ringDimension
        publicColumns publicColumns)
    (two : Mxx.Matrix) (twoLayout : Mxx.Toolkit.MatrixLayout two q ringDimension 1 1) :
    (runtimeBooleanEncodingLayer layer previous one valid rightDecompositions
      decompositionLayouts two twoLayout).map RuntimeBooleanEncoding.error =
      runtimeBooleanNoiseLayer maxWidth layer (previous.map RuntimeBooleanEncoding.error)
        one.error
        (by simpa using valid)
        (fun i ↦ (previous.get (layer.leftIndex valid i)).plaintext)
        rightDecompositions := by
  apply List.ext_get
  · simp [runtimeBooleanEncodingLayer, runtimeBooleanNoiseLayer]
  · intro i leftBound rightBound
    have indexLt : i < maxWidth := by
      simpa [runtimeBooleanEncodingLayer] using leftBound
    by_cases active : i < layer.activeWidth
    · let gate : Fin layer.activeWidth := ⟨i, active⟩
      simp only [List.get_eq_getElem, List.getElem_map, runtimeBooleanEncodingLayer,
        List.getElem_ofFn, active, dite_true, runtimeBooleanNoiseLayer]
      exact RuntimeBooleanEncoding.applyGate_error (layer.kinds.get gate) one
        (previous.get (layer.leftIndex valid gate))
        (previous.get (layer.rightIndex valid gate))
        ((previous.get (layer.leftIndex valid gate)).multiply
          (previous.get (layer.rightIndex valid gate)) (rightDecompositions gate)
          (decompositionLayouts gate))
        (rightDecompositions gate) two (decompositionLayouts gate) twoLayout rfl
    · simp [runtimeBooleanEncodingLayer, runtimeBooleanNoiseLayer, active,
        RuntimeBooleanEncoding.sub]

theorem runtimeBooleanNoiseLayer_pointwise_bound
    (q ringDimension publicColumns digitBound oneBound inputBound maxWidth : Nat)
    [NeZero q] (layer : MxxWe.BooleanLayerProgram)
    (previousErrors : List Mxx.Matrix) (oneError : Mxx.Matrix)
    (valid : layer.Valid previousErrors.length maxWidth)
    (leftPlaintexts rightDecompositions : Fin layer.activeWidth → Mxx.Matrix)
    (oneShape : Mxx.Toolkit.MatrixShape oneError q ringDimension 1 publicColumns)
    (previousShape : ∀ i : Fin previousErrors.length,
      Mxx.Toolkit.MatrixShape (previousErrors.get i) q ringDimension 1 publicColumns)
    (plaintextShape : ∀ i : Fin layer.activeWidth,
      Mxx.Toolkit.MatrixShape (leftPlaintexts i) q ringDimension 1 1)
    (decompositionShape : ∀ i : Fin layer.activeWidth,
      Mxx.Toolkit.MatrixShape (rightDecompositions i) q ringDimension
        publicColumns publicColumns)
    (oneNorm : Mxx.maxCenteredCoefficientNorm oneError ≤ oneBound)
    (previousNorm : MxxWe.EveryNoiseBounded previousErrors inputBound)
    (plaintextNorm : ∀ i : Fin layer.activeWidth,
      Mxx.maxCenteredCoefficientNorm (leftPlaintexts i) ≤ 1)
    (decompositionNorm : ∀ i : Fin layer.activeWidth,
      Mxx.maxCenteredCoefficientNorm (rightDecompositions i) ≤ digitBound)
    (i : Fin (runtimeBooleanNoiseLayer maxWidth layer previousErrors oneError valid
      leftPlaintexts rightDecompositions).length) :
    Mxx.maxCenteredCoefficientNorm
        ((runtimeBooleanNoiseLayer maxWidth layer previousErrors oneError valid
          leftPlaintexts rightDecompositions).get i) ≤
      MxxWe.gateStep ringDimension publicColumns digitBound oneBound inputBound := by
  have outputLength :
      (runtimeBooleanNoiseLayer maxWidth layer previousErrors oneError valid
        leftPlaintexts rightDecompositions).length = maxWidth := by
    simp [runtimeBooleanNoiseLayer]
  let slot : Fin maxWidth := ⟨i, by simpa [outputLength] using i.isLt⟩
  change Mxx.maxCenteredCoefficientNorm
      ((List.ofFn fun j : Fin maxWidth ↦
        if active : j < layer.activeWidth then
          let gate : Fin layer.activeWidth := ⟨j, active⟩
          MxxWe.booleanGateNoiseMatrix (layer.kinds.get gate) oneError
            (previousErrors.get (layer.leftIndex valid gate))
            (previousErrors.get (layer.rightIndex valid gate))
            (leftPlaintexts gate) (rightDecompositions gate)
        else Mxx.matrixSubtract oneError oneError).get
          ⟨slot, by simp⟩) ≤ _
  simp only [List.get_ofFn]
  split_ifs with active
  · let gate : Fin layer.activeWidth := ⟨slot, active⟩
    exact MxxWe.dynamicBooleanLayer_noise_norm_le q ringDimension publicColumns digitBound
      oneBound inputBound maxWidth layer previousErrors valid oneError leftPlaintexts
      rightDecompositions oneShape previousShape plaintextShape decompositionShape oneNorm
      previousNorm plaintextNorm decompositionNorm gate
  · have differenceNorm : Mxx.maxCenteredCoefficientNorm
        (Mxx.matrixSubtract oneError oneError) ≤ oneBound + oneBound :=
      le_trans
        (Mxx.Toolkit.matrixSubtract_norm_le q oneError oneError oneShape.modulus
          oneShape.modulus)
        (Nat.add_le_add oneNorm oneNorm)
    have zeroGateBound := MxxWe.gateNoise_le_gateStep MxxWe.BooleanGate.constantFalse
      ringDimension publicColumns digitBound oneBound 0 0 inputBound
      (Nat.zero_le _) (Nat.zero_le _)
    exact le_trans differenceNorm (by simpa [MxxWe.gateNoise, two_mul] using zeroGateBound)

/-- The concrete layer above discharges exactly one `statesStep` obligation. -/
theorem runtimeBooleanNoiseLayer_bounded
    (q ringDimension publicColumns digitBound oneBound inputBound maxWidth : Nat)
    [NeZero q] (layer : MxxWe.BooleanLayerProgram)
    (previousErrors : List Mxx.Matrix) (oneError : Mxx.Matrix)
    (valid : layer.Valid previousErrors.length maxWidth)
    (leftPlaintexts rightDecompositions : Fin layer.activeWidth → Mxx.Matrix)
    (oneShape : Mxx.Toolkit.MatrixShape oneError q ringDimension 1 publicColumns)
    (previousShape : ∀ i : Fin previousErrors.length,
      Mxx.Toolkit.MatrixShape (previousErrors.get i) q ringDimension 1 publicColumns)
    (plaintextShape : ∀ i : Fin layer.activeWidth,
      Mxx.Toolkit.MatrixShape (leftPlaintexts i) q ringDimension 1 1)
    (decompositionShape : ∀ i : Fin layer.activeWidth,
      Mxx.Toolkit.MatrixShape (rightDecompositions i) q ringDimension
        publicColumns publicColumns)
    (oneNorm : Mxx.maxCenteredCoefficientNorm oneError ≤ oneBound)
    (previousNorm : MxxWe.EveryNoiseBounded previousErrors inputBound)
    (plaintextNorm : ∀ i : Fin layer.activeWidth,
      Mxx.maxCenteredCoefficientNorm (leftPlaintexts i) ≤ 1)
    (decompositionNorm : ∀ i : Fin layer.activeWidth,
      Mxx.maxCenteredCoefficientNorm (rightDecompositions i) ≤ digitBound) :
    MxxWe.EveryNoiseBounded
      (runtimeBooleanNoiseLayer maxWidth layer previousErrors oneError valid
        leftPlaintexts rightDecompositions)
      (MxxWe.gateStep ringDimension publicColumns digitBound oneBound inputBound) := by
  intro i
  simpa [runtimeBooleanNoiseLayer] using
    runtimeBooleanNoiseLayer_pointwise_bound q ringDimension publicColumns digitBound
      oneBound inputBound maxWidth layer previousErrors oneError valid leftPlaintexts
      rightDecompositions oneShape previousShape plaintextShape decompositionShape oneNorm
      previousNorm plaintextNorm decompositionNorm i

/-- All concrete data needed to identify one runtime error-state transition with the exact
dynamic Boolean noise equation. -/
structure RuntimeBooleanNoiseStep
    (q ringDimension publicColumns digitBound oneBound maxWidth : Nat)
    (layer : MxxWe.BooleanLayerProgram) (previous next : List Mxx.Matrix)
    (oneError : Mxx.Matrix) where
  valid : layer.Valid previous.length maxWidth
  leftPlaintexts : Fin layer.activeWidth → Mxx.Matrix
  rightDecompositions : Fin layer.activeWidth → Mxx.Matrix
  nextEq : next = runtimeBooleanNoiseLayer maxWidth layer previous oneError valid
    leftPlaintexts rightDecompositions
  oneShape : Mxx.Toolkit.MatrixShape oneError q ringDimension 1 publicColumns
  previousShape : ∀ i : Fin previous.length,
    Mxx.Toolkit.MatrixShape (previous.get i) q ringDimension 1 publicColumns
  plaintextShape : ∀ i : Fin layer.activeWidth,
    Mxx.Toolkit.MatrixShape (leftPlaintexts i) q ringDimension 1 1
  decompositionShape : ∀ i : Fin layer.activeWidth,
    Mxx.Toolkit.MatrixShape (rightDecompositions i) q ringDimension
      publicColumns publicColumns
  oneNorm : Mxx.maxCenteredCoefficientNorm oneError ≤ oneBound
  plaintextNorm : ∀ i : Fin layer.activeWidth,
    Mxx.maxCenteredCoefficientNorm (leftPlaintexts i) ≤ 1
  decompositionNorm : ∀ i : Fin layer.activeWidth,
    Mxx.maxCenteredCoefficientNorm (rightDecompositions i) ≤ digitBound

/-- Build the next invariant state from the exact runtime gate algebra. -/
noncomputable def RuntimeEncodingState.next
    {q ringDimension publicColumns maxWidth : Nat} [NeZero q] [NeZero ringDimension]
    [Fact (1 < q)]
    {secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1}
    {gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns}
    (state : RuntimeEncodingState q ringDimension publicColumns secret gadget)
    (layer : MxxWe.BooleanLayerProgram)
    (one : RuntimeBooleanEncoding q ringDimension publicColumns)
    (valid : layer.Valid state.encodings.length maxWidth)
    (rightDecompositions : Fin layer.activeWidth → Mxx.Matrix)
    (decompositionLayouts : ∀ i : Fin layer.activeWidth,
      Mxx.Toolkit.MatrixLayout (rightDecompositions i) q ringDimension
        publicColumns publicColumns)
    (two : Mxx.Matrix) (twoLayout : Mxx.Toolkit.MatrixLayout two q ringDimension 1 1)
    (twoValue : Mxx.Toolkit.matrixValue q ringDimension 1 1 two =
      (2 : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1))
    (oneHolds : one.toAlgebra.Holds secret gadget)
    (decomposes : ∀ i : Fin layer.activeWidth,
      gadget * Mxx.Toolkit.matrixValue q ringDimension publicColumns publicColumns
        (rightDecompositions i) =
      (state.encodings.get (layer.rightIndex valid i)).toAlgebra.publicKey) :
    RuntimeEncodingState q ringDimension publicColumns secret gadget where
  encodings := runtimeBooleanEncodingLayer layer state.encodings one valid rightDecompositions
    decompositionLayouts two twoLayout
  holds := runtimeBooleanEncodingLayer_holds secret gadget layer state.encodings one valid
    rightDecompositions decompositionLayouts two twoLayout twoValue oneHolds state.holds decomposes

/-- The invariant-preserving concrete layer supplies `RuntimeBooleanNoiseStep` without a caller
transition equation. -/
noncomputable def RuntimeEncodingState.noiseStep
    {q ringDimension publicColumns digitBound oneBound maxWidth : Nat}
    [NeZero q] [NeZero ringDimension] [Fact (1 < q)]
    {secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1}
    {gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns}
    (state : RuntimeEncodingState q ringDimension publicColumns secret gadget)
    (layer : MxxWe.BooleanLayerProgram)
    (one : RuntimeBooleanEncoding q ringDimension publicColumns)
    (valid : layer.Valid state.encodings.length maxWidth)
    (rightDecompositions : Fin layer.activeWidth → Mxx.Matrix)
    (decompositionLayouts : ∀ i : Fin layer.activeWidth,
      Mxx.Toolkit.MatrixLayout (rightDecompositions i) q ringDimension
        publicColumns publicColumns)
    (two : Mxx.Matrix) (twoLayout : Mxx.Toolkit.MatrixLayout two q ringDimension 1 1)
    (twoValue : Mxx.Toolkit.matrixValue q ringDimension 1 1 two =
      (2 : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1))
    (oneHolds : one.toAlgebra.Holds secret gadget)
    (decomposes : ∀ i : Fin layer.activeWidth,
      gadget * Mxx.Toolkit.matrixValue q ringDimension publicColumns publicColumns
        (rightDecompositions i) =
      (state.encodings.get (layer.rightIndex valid i)).toAlgebra.publicKey)
    (oneNorm : Mxx.maxCenteredCoefficientNorm one.error ≤ oneBound)
    (plaintextNorm : ∀ i : Fin layer.activeWidth,
      Mxx.maxCenteredCoefficientNorm
        (state.encodings.get (layer.leftIndex valid i)).plaintext ≤ 1)
    (decompositionNorm : ∀ i : Fin layer.activeWidth,
      Mxx.maxCenteredCoefficientNorm (rightDecompositions i) ≤ digitBound) :
    RuntimeBooleanNoiseStep q ringDimension publicColumns digitBound oneBound maxWidth layer
      (state.encodings.map RuntimeBooleanEncoding.error)
      ((state.next layer one valid rightDecompositions decompositionLayouts two twoLayout
        twoValue oneHolds decomposes).encodings.map RuntimeBooleanEncoding.error) one.error := by
  let nextState := state.next layer one valid rightDecompositions decompositionLayouts two
    twoLayout twoValue oneHolds decomposes
  refine {
    valid := by simpa using valid
    leftPlaintexts := fun i ↦ (state.encodings.get (layer.leftIndex valid i)).plaintext
    rightDecompositions
    nextEq := ?_
    oneShape := one.errorLayout.toMatrixShape
    previousShape := ?_
    plaintextShape := fun i ↦
      (state.encodings.get (layer.leftIndex valid i)).plaintextLayout.toMatrixShape
    decompositionShape := fun i ↦ (decompositionLayouts i).toMatrixShape
    oneNorm
    plaintextNorm
    decompositionNorm
  }
  · exact runtimeBooleanEncodingLayer_errors layer state.encodings one valid
      rightDecompositions decompositionLayouts two twoLayout
  · intro i
    let source : Fin state.encodings.length := ⟨i, by simpa using i.isLt⟩
    simpa [source] using (state.encodings.get source).errorLayout.toMatrixShape

/-- One exact invariant-preserving runtime transition.  This is the target produced by the
execution bridge for a certified Boolean child; the next state is definitionally the existing
runtime evaluator rather than an independently asserted semantic relation. -/
structure RuntimeEncodingTransition
    (q ringDimension publicColumns digitBound oneBound maxWidth : Nat)
    [NeZero q] [NeZero ringDimension] [Fact (1 < q)]
    {secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1}
    {gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns}
    (layer : MxxWe.BooleanLayerProgram)
    (previous next : RuntimeEncodingState q ringDimension publicColumns secret gadget)
    (one : RuntimeBooleanEncoding q ringDimension publicColumns) where
  valid : layer.Valid previous.encodings.length maxWidth
  rightDecompositions : Fin layer.activeWidth → Mxx.Matrix
  decompositionLayouts : ∀ i : Fin layer.activeWidth,
    Mxx.Toolkit.MatrixLayout (rightDecompositions i) q ringDimension
      publicColumns publicColumns
  two : Mxx.Matrix
  twoLayout : Mxx.Toolkit.MatrixLayout two q ringDimension 1 1
  twoValue : Mxx.Toolkit.matrixValue q ringDimension 1 1 two =
    (2 : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1)
  oneHolds : one.toAlgebra.Holds secret gadget
  decomposes : ∀ i : Fin layer.activeWidth,
    gadget * Mxx.Toolkit.matrixValue q ringDimension publicColumns publicColumns
      (rightDecompositions i) =
    (previous.encodings.get (layer.rightIndex valid i)).toAlgebra.publicKey
  nextEq : next = previous.next layer one valid rightDecompositions decompositionLayouts two
    twoLayout twoValue oneHolds decomposes
  oneNorm : Mxx.maxCenteredCoefficientNorm one.error ≤ oneBound
  plaintextNorm : ∀ i : Fin layer.activeWidth,
    Mxx.maxCenteredCoefficientNorm
      (previous.encodings.get (layer.leftIndex valid i)).plaintext ≤ 1
  decompositionNorm : ∀ i : Fin layer.activeWidth,
    Mxx.maxCenteredCoefficientNorm (rightDecompositions i) ≤ digitBound

/-- The exact encoding transition carries the matching concrete simulator step automatically. -/
noncomputable def RuntimeEncodingTransition.noiseStep
    {q ringDimension publicColumns digitBound oneBound maxWidth : Nat}
    [NeZero q] [NeZero ringDimension] [Fact (1 < q)]
    {secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1}
    {gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns}
    {layer : MxxWe.BooleanLayerProgram}
    {previous next : RuntimeEncodingState q ringDimension publicColumns secret gadget}
    {one : RuntimeBooleanEncoding q ringDimension publicColumns}
    (transition : RuntimeEncodingTransition q ringDimension publicColumns digitBound oneBound
      maxWidth layer previous next one) :
    RuntimeBooleanNoiseStep q ringDimension publicColumns digitBound oneBound maxWidth layer
      (previous.encodings.map RuntimeBooleanEncoding.error)
      (next.encodings.map RuntimeBooleanEncoding.error) one.error := by
  rw [transition.nextEq]
  exact previous.noiseStep layer one transition.valid transition.rightDecompositions
    transition.decompositionLayouts transition.two transition.twoLayout transition.twoValue
    transition.oneHolds transition.decomposes transition.oneNorm transition.plaintextNorm
    transition.decompositionNorm

theorem RuntimeBooleanNoiseStep.bounded
    {q ringDimension publicColumns digitBound oneBound maxWidth inputBound : Nat}
    [NeZero q] {layer : MxxWe.BooleanLayerProgram} {previous next : List Mxx.Matrix}
    {oneError : Mxx.Matrix}
    (step : RuntimeBooleanNoiseStep q ringDimension publicColumns digitBound oneBound
      maxWidth layer previous next oneError)
    (previousNorm : MxxWe.EveryNoiseBounded previous inputBound) :
    MxxWe.EveryNoiseBounded next
      (MxxWe.gateStep ringDimension publicColumns digitBound oneBound inputBound) := by
  rw [step.nextEq]
  exact runtimeBooleanNoiseLayer_bounded q ringDimension publicColumns digitBound oneBound
    inputBound maxWidth layer previous oneError step.valid step.leftPlaintexts
    step.rightDecompositions step.oneShape step.previousShape step.plaintextShape
    step.decompositionShape step.oneNorm previousNorm step.plaintextNorm
    step.decompositionNorm

/-- A runtime step object for every depth discharges the `statesStep` premise consumed by the
closed-form circuit-noise recurrence. -/
theorem statesStep_of_runtimeBooleanNoiseSteps
    (q ringDimension publicColumns depth digitBound oneBound maxWidth : Nat) [NeZero q]
    (layers : Fin depth → MxxWe.BooleanLayerProgram)
    (states : Nat → List Mxx.Matrix) (oneError : Mxx.Matrix)
    (steps : ∀ layer : Nat, (layerLt : layer < depth) →
      RuntimeBooleanNoiseStep q ringDimension publicColumns digitBound oneBound maxWidth
        (layers ⟨layer, layerLt⟩) (states layer) (states (layer + 1)) oneError) :
    ∀ layer : Nat, layer < depth →
      MxxWe.EveryNoiseBounded (states layer)
        ((List.range layer).foldl
          (fun bound _ ↦ MxxWe.gateStep ringDimension publicColumns digitBound oneBound bound)
          (2 * oneBound)) →
      MxxWe.EveryNoiseBounded (states (layer + 1))
        (MxxWe.gateStep ringDimension publicColumns digitBound oneBound
          ((List.range layer).foldl
            (fun bound _ ↦ MxxWe.gateStep ringDimension publicColumns digitBound oneBound bound)
            (2 * oneBound))) := by
  intro layer layerLt previousNorm
  exact (steps layer layerLt).bounded previousNorm

/-- Exact runtime layer transitions and the initial state bound imply the selected circuit-output
noise bound used by the Diamond decoder. -/
theorem selectedCircuitNoiseBound_of_runtimeSteps
    (q ringDimension publicColumns depth digitBound oneBound maxWidth : Nat) [NeZero q]
    (layers : Fin depth → MxxWe.BooleanLayerProgram)
    (states : Nat → List Mxx.Matrix) (oneError : Mxx.Matrix)
    (initial : MxxWe.EveryNoiseBounded (states 0) (2 * oneBound))
    (steps : ∀ layer : Nat, (layerLt : layer < depth) →
      RuntimeBooleanNoiseStep q ringDimension publicColumns digitBound oneBound maxWidth
        (layers ⟨layer, layerLt⟩) (states layer) (states (layer + 1)) oneError)
    (circuitIndex : Fin (states depth).length) :
    Mxx.maxCenteredCoefficientNorm ((states depth).get circuitIndex) ≤
      MxxWe.circuitBound ringDimension publicColumns depth digitBound oneBound := by
  have statesStep := statesStep_of_runtimeBooleanNoiseSteps q ringDimension publicColumns depth
    digitBound oneBound maxWidth layers states oneError steps
  have finalBound := MxxWe.dynamicBooleanLayers_noise_le_circuitBound ringDimension
    publicColumns depth digitBound oneBound states initial statesStep
  exact finalBound circuitIndex

/-- Every initial Boolean input is either an independently bounded witness/instance encoding or
the exact checked zero encoding `one - one`.  Distinct preimages may produce distinct bounded
errors, so this deliberately does not identify every nonzero input error with `oneError`. -/
structure RuntimeBooleanInitialState
    (q ringDimension publicColumns oneBound : Nat) (errors : List Mxx.Matrix)
    (oneError : Mxx.Matrix) : Prop where
  oneShape : Mxx.Toolkit.MatrixShape oneError q ringDimension 1 publicColumns
  oneNorm : Mxx.maxCenteredCoefficientNorm oneError ≤ oneBound
  source : ∀ i : Fin errors.length,
    Mxx.maxCenteredCoefficientNorm (errors.get i) ≤ oneBound ∨
      errors.get i = Mxx.matrixSubtract oneError oneError

/-- Exact arbitrary-depth runtime encoding execution.  Each adjacent state is the concrete
`RuntimeEncodingState.next` result stored by `RuntimeEncodingTransition`; the selected output is
therefore an actual member of the final certified state, not a separately postulated encoding. -/
structure RuntimeEncodingExecution
    (q ringDimension publicColumns depth digitBound oneBound maxWidth : Nat)
    [NeZero q] [NeZero ringDimension] [Fact (1 < q)]
    {secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1}
    {gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns}
    (layers : Fin depth → MxxWe.BooleanLayerProgram)
    (one : RuntimeBooleanEncoding q ringDimension publicColumns) where
  states : Nat → RuntimeEncodingState q ringDimension publicColumns secret gadget
  transitions : ∀ layer : Nat, (layerLt : layer < depth) →
    RuntimeEncodingTransition q ringDimension publicColumns digitBound oneBound maxWidth
      (layers ⟨layer, layerLt⟩) (states layer) (states (layer + 1)) one
  initial : RuntimeBooleanInitialState q ringDimension publicColumns oneBound
    ((states 0).encodings.map RuntimeBooleanEncoding.error) one.error
  circuitIndex : Fin (states depth).encodings.length
  circuitEncoding : RuntimeBooleanEncoding q ringDimension publicColumns
  circuitEncodingEq : circuitEncoding = (states depth).encodings.get circuitIndex

/-- The state-indexed ghost errors of an exact runtime encoding execution. -/
def RuntimeEncodingExecution.errorStates
    {q ringDimension publicColumns depth digitBound oneBound maxWidth : Nat}
    [NeZero q] [NeZero ringDimension] [Fact (1 < q)]
    {secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1}
    {gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns}
    {layers : Fin depth → MxxWe.BooleanLayerProgram}
    {one : RuntimeBooleanEncoding q ringDimension publicColumns}
    (execution : RuntimeEncodingExecution q ringDimension publicColumns depth digitBound
      oneBound maxWidth (secret := secret) (gadget := gadget) layers one)
    (layer : Nat) : List Mxx.Matrix :=
  (execution.states layer).encodings.map RuntimeBooleanEncoding.error

/-- The exact initial-selector alternatives imply the `2 * oneBound` premise of the Boolean
recurrence. -/
theorem RuntimeBooleanInitialState.bounded
    {q ringDimension publicColumns oneBound : Nat} [NeZero q]
    {errors : List Mxx.Matrix} {oneError : Mxx.Matrix}
    (initial : RuntimeBooleanInitialState q ringDimension publicColumns oneBound errors oneError) :
    MxxWe.EveryNoiseBounded errors (2 * oneBound) := by
  intro i
  rcases initial.source i with bounded | errorEq
  · exact le_trans bounded (by omega)
  · rw [errorEq]
    have differenceNorm := Mxx.Toolkit.matrixSubtract_norm_le q oneError oneError
      initial.oneShape.modulus initial.oneShape.modulus
    have sumNorm := Nat.add_le_add initial.oneNorm initial.oneNorm
    exact le_trans differenceNorm (by simpa [two_mul] using sumNorm)

/-- Complete ghost-error witness for an executed arbitrary-depth Boolean circuit.  The selected
error is definitionally tied to the final runtime state; all transitions use the exact dynamic
gate equation above. -/
structure RuntimeBooleanNoiseExecution
    (q ringDimension publicColumns depth digitBound oneBound maxWidth : Nat)
    (layers : Fin depth → MxxWe.BooleanLayerProgram) (oneError : Mxx.Matrix) where
  states : Nat → List Mxx.Matrix
  initial : RuntimeBooleanInitialState q ringDimension publicColumns oneBound (states 0) oneError
  steps : ∀ layer : Nat, (layerLt : layer < depth) →
    RuntimeBooleanNoiseStep q ringDimension publicColumns digitBound oneBound maxWidth
      (layers ⟨layer, layerLt⟩) (states layer) (states (layer + 1)) oneError
  circuitIndex : Fin (states depth).length
  circuitError : Mxx.Matrix
  circuitErrorEq : circuitError = (states depth).get circuitIndex
  circuitShape :
    Mxx.Toolkit.MatrixShape circuitError q ringDimension 1 publicColumns

/-- Exact encoding transitions induce the existing arbitrary-depth noise execution without any
additional per-layer semantic premise. -/
noncomputable def RuntimeEncodingExecution.noiseExecution
    {q ringDimension publicColumns depth digitBound oneBound maxWidth : Nat}
    [NeZero q] [NeZero ringDimension] [Fact (1 < q)]
    {secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1}
    {gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns}
    {layers : Fin depth → MxxWe.BooleanLayerProgram}
    {one : RuntimeBooleanEncoding q ringDimension publicColumns}
    (execution : RuntimeEncodingExecution q ringDimension publicColumns depth digitBound
      oneBound maxWidth (secret := secret) (gadget := gadget) layers one) :
    RuntimeBooleanNoiseExecution q ringDimension publicColumns depth digitBound oneBound
      maxWidth layers one.error := by
  let finalIndex : Fin (execution.errorStates depth).length :=
    ⟨execution.circuitIndex, by
      simp [RuntimeEncodingExecution.errorStates]⟩
  exact {
    states := execution.errorStates
    initial := execution.initial
    steps := fun layer layerLt ↦ (execution.transitions layer layerLt).noiseStep
    circuitIndex := finalIndex
    circuitError := execution.circuitEncoding.error
    circuitErrorEq := by
      rw [execution.circuitEncodingEq]
      simp [RuntimeEncodingExecution.errorStates, finalIndex]
    circuitShape := execution.circuitEncoding.errorLayout.toMatrixShape
  }

/-- The theorem-level bundle consumed by `residual_bound_of_recurrences`: initial bound, exact
step recurrence, selected-output identity, shape, and final norm. -/
structure RuntimeBooleanNoiseFacts
    (q ringDimension publicColumns depth digitBound oneBound : Nat)
    (states : Nat → List Mxx.Matrix) (circuitError : Mxx.Matrix) where
  statesInitial : MxxWe.EveryNoiseBounded (states 0) (2 * oneBound)
  statesStep : ∀ layer : Nat, layer < depth →
    MxxWe.EveryNoiseBounded (states layer)
      ((List.range layer).foldl
        (fun bound _ ↦ MxxWe.gateStep ringDimension publicColumns digitBound oneBound bound)
        (2 * oneBound)) →
    MxxWe.EveryNoiseBounded (states (layer + 1))
      (MxxWe.gateStep ringDimension publicColumns digitBound oneBound
        ((List.range layer).foldl
          (fun bound _ ↦ MxxWe.gateStep ringDimension publicColumns digitBound oneBound bound)
          (2 * oneBound)))
  circuitIndex : Fin (states depth).length
  circuitErrorEq : circuitError = (states depth).get circuitIndex
  circuitShape : Mxx.Toolkit.MatrixShape circuitError q ringDimension 1 publicColumns
  circuitNorm : Mxx.maxCenteredCoefficientNorm circuitError ≤
    MxxWe.circuitBound ringDimension publicColumns depth digitBound oneBound

/-- Package an exact runtime Boolean noise execution into the premises expected by the final
Diamond residual theorem. -/
def RuntimeBooleanNoiseExecution.facts
    {q ringDimension publicColumns depth digitBound oneBound maxWidth : Nat} [NeZero q]
    {layers : Fin depth → MxxWe.BooleanLayerProgram} {oneError : Mxx.Matrix}
    (execution : RuntimeBooleanNoiseExecution q ringDimension publicColumns depth digitBound
      oneBound maxWidth layers oneError) :
    RuntimeBooleanNoiseFacts q ringDimension publicColumns depth digitBound oneBound
      execution.states execution.circuitError := by
  have statesInitial := execution.initial.bounded
  have statesStep := statesStep_of_runtimeBooleanNoiseSteps q ringDimension publicColumns depth
    digitBound oneBound maxWidth layers execution.states oneError execution.steps
  have selectedNorm := selectedCircuitNoiseBound_of_runtimeSteps q ringDimension publicColumns
    depth digitBound oneBound maxWidth layers execution.states oneError statesInitial
    execution.steps execution.circuitIndex
  exact {
    statesInitial
    statesStep
    circuitIndex := execution.circuitIndex
    circuitErrorEq := execution.circuitErrorEq
    circuitShape := execution.circuitShape
    circuitNorm := by simpa [execution.circuitErrorEq] using selectedNorm
  }
/-- Exact executable gadget-decomposition node selected by a checked local decomposition
reference.  The verifier fixes its dimensions and parameters; this resolution retains the exact
checked payload without asking the proof caller to reconstruct private verifier abbreviations. -/
structure LocalGadgetDecompositionResolution
    (workflow : Mxx.Ir.Workflow) (reference : LocalGadgetDecompositionRef) where
  matrixType : Mxx.Ir.MatrixTypeExpr
  base : Mxx.Ir.IntExpr
  digitCount : Mxx.Ir.IntExpr
  resolved : resolveNode workflow reference.decompositionNode = some {
    kind := .gadgetDecompose matrixType base digitCount
    arguments := [wireRef reference.rightPublicKey.wire]
    outputCount := 1
  }

theorem localGadgetDecompositionResolution_of_verified
    {workflow : Mxx.Ir.Workflow} {expectedScope : ScopeRef}
    {reference : LocalGadgetDecompositionRef}
    (verified : verifyLocalDecomposition workflow expectedScope reference = true) :
    Nonempty (LocalGadgetDecompositionResolution workflow reference) := by
  unfold verifyLocalDecomposition at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  cases resolved : resolveNode workflow reference.decompositionNode with
  | none => simp [resolved] at verified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> try simp_all
      rename_i actualType base count
      exact ⟨⟨actualType, base, count, by simp_all⟩⟩

/-- Contract-backed outcome of the exact runtime RHS decomposition. -/
structure LocalGadgetDecompositionOutcome
    (workflow : Mxx.Ir.Workflow) (reference : LocalGadgetDecompositionRef)
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (execution : ReferencedNodeExecution workflow reference.decompositionNode runChild samplers
      params inputs) where
  matrixParams : Mxx.SamplerParams
  evaluatedBase : Int
  evaluatedDigitCount : Int
  rightPublicKey : Mxx.Matrix
  decomposition : Mxx.Matrix
  valuesEq : execution.values = [.matrix decomposition]
  equation : Mxx.matrixMul
    (Mxx.gadgetMatrix {
      matrixParams with
      rows := rightPublicKey.rows
      columns := rightPublicKey.rows * evaluatedDigitCount.toNat
    } evaluatedBase evaluatedDigitCount.toNat) decomposition = rightPublicKey
  shape : Mxx.Toolkit.MatrixShape decomposition matrixParams.modulus
    matrixParams.ringDimension matrixParams.rows matrixParams.columns
  norm : Mxx.maxCenteredCoefficientNorm decomposition ≤ max (evaluatedBase.natAbs / 2) 1

theorem localGadgetDecompositionOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : LocalGadgetDecompositionRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (resolution : LocalGadgetDecompositionResolution workflow reference)
    (execution : ReferencedNodeExecution workflow reference.decompositionNode runChild samplers
      params inputs)
    (rightPublicKey : Mxx.Matrix) (matrixParams : Mxx.SamplerParams)
    (evaluatedBase evaluatedDigitCount : Int)
    (argumentsEvaluate :
      [wireRef reference.rightPublicKey.wire].mapM
          (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
        some [.matrix rightPublicKey])
    (matrixTypeEvaluate : resolution.matrixType.evaluate params (.constant 0) =
      some matrixParams)
    (baseEvaluate : resolution.base.evaluate params = some evaluatedBase)
    (digitCountEvaluate : resolution.digitCount.evaluate params = some evaluatedDigitCount) :
    Nonempty (LocalGadgetDecompositionOutcome workflow reference runChild samplers params inputs
      execution) := by
  have executionResolved := execution.resolved
  have decompositionResolved := resolution.resolved
  rw [executionResolved] at decompositionResolved
  have nodeEq := Option.some.inj decompositionResolved
  have member := execution.member
  rw [nodeEq] at member
  obtain ⟨raw, rawMember, valuesEq⟩ :=
    Mxx.Ir.mem_evaluateNode_gadgetDecompose_of_arguments runChild samplers params inputs
      execution.before (wireRef reference.rightPublicKey.wire) rightPublicKey
      resolution.matrixType resolution.base resolution.digitCount matrixParams evaluatedBase
      evaluatedDigitCount 1 argumentsEvaluate matrixTypeEvaluate baseEvaluate digitCountEvaluate
      member
  let decomposition := raw.withSamplerParams matrixParams
  have sampled := contract.gadgetDecomposeContract matrixParams evaluatedBase
    evaluatedDigitCount.toNat rightPublicKey raw rawMember
  exact ⟨{
    matrixParams
    evaluatedBase
    evaluatedDigitCount
    rightPublicKey
    decomposition
    valuesEq := by simpa [decomposition] using valuesEq
    equation := by simpa [decomposition] using sampled.1
    shape := Mxx.Toolkit.withSamplerParams_shape raw matrixParams
    norm := by simpa [decomposition] using sampled.2
  }⟩

/-- Exact executable fields of an arbitrary checked parallel-loop reference. -/
structure CertifiedParallelLoopResolution
    (workflow : Mxx.Ir.Workflow) (reference : ParallelLoopRef) : Prop where
  resolved : resolveNode workflow reference.operation = some {
    kind := .parallelLoop reference.bodyScope.definitionName reference.count
      reference.indexSlot reference.bindings
      (reference.inputModes.map CertifiedLoopInputMode.toIr)
    arguments := reference.arguments.map (wireRef ∘ CoreOperandRef.wire)
    outputCount := reference.outputs.length
  }

set_option maxHeartbeats 800000 in
theorem certifiedParallelLoopResolution_of_verified
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef}
    (verified : verifyParallelLoop workflow reference = true) :
    CertifiedParallelLoopResolution workflow reference := by
  unfold verifyParallelLoop at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at verified
  have bodyScopeEq :
      reference.bodyScope = reference.operation.scope.parallelBody reference.operation.node := by
    aesop
  have nodeChecked := verified.2
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [resolved] at nodeChecked
  | some node =>
      cases bodyResolved :
          resolveScope workflow { reference.operation with scope := reference.bodyScope } with
      | none => simp [resolved, bodyResolved] at nodeChecked
      | some body =>
          rcases node with ⟨kind, arguments, outputCount⟩
          simp only [resolved, bodyResolved] at nodeChecked
          cases kind <;> try contradiction
          rename_i definition count indexSlot bindings inputModes
          simp only [Bool.and_eq_true, decide_eq_true_eq] at nodeChecked
          have definitionEq : definition = reference.bodyScope.definitionName := by aesop
          have countEq : count = reference.count := by aesop
          have indexSlotEq : indexSlot = reference.indexSlot := by aesop
          have bindingsEq : bindings = reference.bindings := by aesop
          have inputModesEq :
              inputModes = reference.inputModes.map CertifiedLoopInputMode.toIr := by aesop
          have argumentsEq :
              arguments = reference.arguments.map (wireRef ∘ CoreOperandRef.wire) := by aesop
          have outputCountEq : outputCount = reference.outputs.length := by aesop
          subst definition
          subst count
          subst indexSlot
          subst bindings
          subst inputModes
          subst arguments
          subst outputCount
          exact ⟨by simpa [bodyScopeEq] using resolved⟩

/-- Exact selected parallel trace for any certified loop, including Boolean candidate loops. -/
structure CertifiedParallelLoopTrace
    (workflow : Mxx.Ir.Workflow) (reference : ParallelLoopRef)
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs) where
  argumentValues : List Mxx.Ir.Value
  evaluatedCount : Int
  argumentsEvaluate :
    (reference.arguments.map (wireRef ∘ CoreOperandRef.wire)).mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) = some argumentValues
  countEvaluate : reference.count.evaluate params = some evaluatedCount
  final : List (List Mxx.Ir.Value)
  iterations : Mxx.Ir.ParallelIterationsTrace runChild reference.bodyScope.definitionName params
    reference.indexSlot reference.bindings
    (reference.inputModes.map CertifiedLoopInputMode.toIr) argumentValues
    (List.range evaluatedCount.toNat) (List.replicate reference.outputs.length []) final
  valuesEq : execution.values = final.map Mxx.Ir.Value.family

theorem certifiedParallelLoopTrace_of_resolution
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : CertifiedParallelLoopResolution workflow reference)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (argumentValues : List Mxx.Ir.Value) (evaluatedCount : Int)
    (argumentsEvaluate :
      (reference.arguments.map (wireRef ∘ CoreOperandRef.wire)).mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) = some argumentValues)
    (countEvaluate : reference.count.evaluate params = some evaluatedCount) :
    Nonempty (CertifiedParallelLoopTrace workflow reference runChild samplers params inputs
      execution) := by
  have executionResolved := execution.resolved
  have loopResolved := resolution.resolved
  rw [executionResolved] at loopResolved
  have nodeEq := Option.some.inj loopResolved
  have member : execution.values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs
      execution.before {
        kind := .parallelLoop reference.bodyScope.definitionName reference.count
          reference.indexSlot reference.bindings
          (reference.inputModes.map CertifiedLoopInputMode.toIr)
        arguments := reference.arguments.map (wireRef ∘ CoreOperandRef.wire)
        outputCount := reference.outputs.length
      } := by simpa [nodeEq] using execution.member
  obtain ⟨final, iterations, valuesEq⟩ :=
    (Mxx.Ir.mem_evaluateNode_parallelLoop_iff_trace runChild samplers params inputs
      execution.before reference.bodyScope.definitionName reference.count reference.indexSlot
      reference.bindings (reference.inputModes.map CertifiedLoopInputMode.toIr)
      (reference.arguments.map (wireRef ∘ CoreOperandRef.wire)) reference.outputs.length
      argumentValues evaluatedCount argumentsEvaluate countEvaluate execution.values).mp member
  exact ⟨{
    argumentValues
    evaluatedCount
    argumentsEvaluate
    countEvaluate
    final
    iterations
    valuesEq
  }⟩

/-- A concrete one-family output rules out the parallel interpreter's invalid fallback and
recovers its exact iteration trace without exposing argument/count evaluation as premises. -/
theorem certifiedParallelLoopTrace_of_familyOutput
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : CertifiedParallelLoopResolution workflow reference)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (outputFamily : List Mxx.Ir.Value)
    (valuesEq : execution.values = [.family outputFamily]) :
    Nonempty (CertifiedParallelLoopTrace workflow reference runChild samplers params inputs
      execution) := by
  have executionResolved := execution.resolved
  have loopResolved := resolution.resolved
  rw [executionResolved] at loopResolved
  have nodeEq := Option.some.inj loopResolved
  have member : execution.values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs
      execution.before {
        kind := .parallelLoop reference.bodyScope.definitionName reference.count
          reference.indexSlot reference.bindings
          (reference.inputModes.map CertifiedLoopInputMode.toIr)
        arguments := reference.arguments.map (wireRef ∘ CoreOperandRef.wire)
        outputCount := reference.outputs.length
      } := by
    simpa [nodeEq] using execution.member
  cases argumentsEvaluate :
      (reference.arguments.map (wireRef ∘ CoreOperandRef.wire)).mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) with
  | none =>
      simp [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate] at member
      rw [member] at valuesEq
      simp at valuesEq
  | some argumentValues =>
      cases countEvaluate : reference.count.evaluate params with
      | none =>
          simp [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate, countEvaluate] at member
          rw [member] at valuesEq
          simp at valuesEq
      | some evaluatedCount =>
          exact certifiedParallelLoopTrace_of_resolution resolution execution argumentValues
            evaluatedCount argumentsEvaluate countEvaluate

/-- The exact executable sequential loop recovered from a checked reference.  This structure is
parameter-generic: neither the loop count nor any generated node number is fixed by the theorem. -/
structure BooleanSequentialLoopResolution
    (workflow : Mxx.Ir.Workflow) (operation : CoreNodeRef) (bodyScope : ScopeRef)
    (carried invariants : List CoreOperandRef) (outputs : List CoreWireRef) where
  count : Mxx.Ir.IntExpr
  indexSlot : Nat
  bindings : List (String × Mxx.Ir.IntExpr)
  resolved : resolveNode workflow operation = some {
    kind := .sequentialLoop bodyScope.definitionName count indexSlot bindings carried.length
    arguments := (carried ++ invariants).map (wireRef ∘ CoreOperandRef.wire)
    outputCount := outputs.length
  }

/-- A successful generic sequential-loop check resolves the referenced executable node exactly. -/
theorem booleanSequentialLoopResolution_of_verified
    {workflow : Mxx.Ir.Workflow} {operation : CoreNodeRef} {bodyScope : ScopeRef}
    {carried invariants : List CoreOperandRef} {outputs : List CoreWireRef}
    (verified :
      verifySequentialLoop workflow operation bodyScope carried invariants outputs = true) :
    Nonempty (BooleanSequentialLoopResolution workflow operation bodyScope carried invariants
      outputs) := by
  unfold verifySequentialLoop at verified
  cases nodeResolved : resolveNode workflow operation with
  | none => simp [nodeResolved] at verified
  | some node =>
      cases scopeResolved : resolveScope workflow { operation with scope := bodyScope } with
      | none => simp [nodeResolved, scopeResolved] at verified
      | some scope =>
          rcases node with ⟨kind, arguments, outputCount⟩
          cases kind <;> simp_all [Bool.and_eq_true, decide_eq_true_eq]
          rename_i definition count indexSlot bindings carriedCount
          exact ⟨{
            count
            indexSlot
            bindings
            resolved := by simp_all
          }⟩

/-- Invert one selected execution of the exact checked Boolean loop into the interpreter's
arbitrary-length sequential trace.  This is the execution-path bridge consumed by the two generic
recurrence lifting theorems below. -/
theorem booleanSequentialTrace_of_resolution
    {workflow : Mxx.Ir.Workflow} {operation : CoreNodeRef} {bodyScope : ScopeRef}
    {carried invariants : List CoreOperandRef} {outputs : List CoreWireRef}
    (resolution : BooleanSequentialLoopResolution workflow operation bodyScope carried invariants
      outputs)
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment) (argumentValues values : List Mxx.Ir.Value)
    (evaluatedCount : Int)
    (argumentsEvaluate :
      ((carried ++ invariants).map (wireRef ∘ CoreOperandRef.wire)).mapM
          (fun wire ↦ Mxx.Ir.lookupWire wire wires) = some argumentValues)
    (countEvaluate : resolution.count.evaluate params = some evaluatedCount)
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .sequentialLoop bodyScope.definitionName resolution.count resolution.indexSlot
        resolution.bindings carried.length
      arguments := (carried ++ invariants).map (wireRef ∘ CoreOperandRef.wire)
      outputCount := outputs.length
    }) :
    Mxx.Ir.SequentialIterationsTrace runChild bodyScope.definitionName params
      resolution.indexSlot resolution.bindings (argumentValues.drop carried.length)
      (List.range evaluatedCount.toNat) (argumentValues.take carried.length) values :=
  (Mxx.Ir.mem_evaluateNode_sequentialLoop_iff_trace runChild samplers params inputs wires
    bodyScope.definitionName resolution.count resolution.indexSlot resolution.bindings
    carried.length ((carried ++ invariants).map (wireRef ∘ CoreOperandRef.wire)) outputs.length
    argumentValues evaluatedCount argumentsEvaluate countEvaluate values).mp member

/-- A concrete three-family encoding output rules out the interpreter's invalid fallback and
therefore derives both argument and count evaluation from the selected sequential-loop
execution.  No raw lookup or expression-evaluation premise is exposed to protocol proofs. -/
theorem booleanSequentialTrace_of_runtimeEncodingOutput
    {workflow : Mxx.Ir.Workflow} {operation : CoreNodeRef} {bodyScope : ScopeRef}
    {carried invariants : List CoreOperandRef} {outputs : List CoreWireRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    {q ringDimension publicColumns : Nat}
    {secret : MxxWe.AlgebraMatrix (Mxx.Toolkit.Negacyclic q ringDimension) 1 1}
    {gadget : MxxWe.AlgebraMatrix
      (Mxx.Toolkit.Negacyclic q ringDimension) 1 publicColumns}
    (resolution : BooleanSequentialLoopResolution workflow operation bodyScope carried invariants
      outputs)
    (execution : ReferencedNodeExecution workflow operation runChild samplers params inputs)
    (outputState : RuntimeEncodingState q ringDimension publicColumns secret gadget)
    (represented : outputState.Represents execution.values) :
    ∃ argumentValues evaluatedCount,
      ((carried ++ invariants).map (wireRef ∘ CoreOperandRef.wire)).mapM
          (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) = some argumentValues ∧
      resolution.count.evaluate params = some evaluatedCount ∧
      Mxx.Ir.SequentialIterationsTrace runChild bodyScope.definitionName params
        resolution.indexSlot resolution.bindings (argumentValues.drop carried.length)
        (List.range evaluatedCount.toNat) (argumentValues.take carried.length)
        execution.values := by
  have executionResolved := execution.resolved
  have loopResolved := resolution.resolved
  rw [executionResolved] at loopResolved
  have nodeEq := Option.some.inj loopResolved
  have member : execution.values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs
      execution.before {
        kind := .sequentialLoop bodyScope.definitionName resolution.count resolution.indexSlot
          resolution.bindings carried.length
        arguments := (carried ++ invariants).map (wireRef ∘ CoreOperandRef.wire)
        outputCount := outputs.length
      } := by
    simpa [nodeEq] using execution.member
  cases argumentsEvaluate :
      ((carried ++ invariants).map (wireRef ∘ CoreOperandRef.wire)).mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) with
  | none =>
      simp only [List.map_append, List.mapM_append] at argumentsEvaluate
      have invalid : execution.values = [.invalid "SequentialLoop argument mismatch"] := by
        cases leftEvaluate :
            (carried.map (wireRef ∘ CoreOperandRef.wire)).mapM
              (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) with
        | none =>
            simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, List.map_append,
              List.mapM_append, leftEvaluate] using member
        | some leftValues =>
            cases invariantEvaluate :
                (invariants.map (wireRef ∘ CoreOperandRef.wire)).mapM
                  (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) with
            | none =>
                simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, List.map_append,
                  List.mapM_append, leftEvaluate, invariantEvaluate] using member
            | some invariantValues =>
                simp [leftEvaluate, invariantEvaluate] at argumentsEvaluate
      rw [invalid] at represented
      simp [RuntimeEncodingState.Represents, runtimeEncodingFamilyValues] at represented
  | some argumentValues =>
      cases countEvaluate : resolution.count.evaluate params with
      | none =>
          have invalid : execution.values = [.invalid "SequentialLoop argument mismatch"] := by
            simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate, countEvaluate] using
              member
          rw [invalid] at represented
          simp [RuntimeEncodingState.Represents, runtimeEncodingFamilyValues] at represented
      | some evaluatedCount =>
          refine ⟨argumentValues, evaluatedCount, ?_, ?_, ?_⟩
          · rfl
          · rfl
          exact booleanSequentialTrace_of_resolution resolution runChild samplers params inputs
            execution.before argumentValues execution.values evaluatedCount argumentsEvaluate
            countEvaluate member

/-- The accepted public-key Boolean layer scan is the exact existing sequential loop. -/
theorem VerifiedDiamondLayout.publicKeyBooleanLoopResolution
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    Nonempty (BooleanSequentialLoopResolution workflow
      certificate.booleanLayers.encryption.layerScan
      certificate.booleanLayers.encryption.bodyScope
      [certificate.booleanLayers.encryption.initialPublicKeys]
      [certificate.booleanLayers.encryption.activeGateCounts,
        certificate.booleanLayers.encryption.gateKinds,
        certificate.booleanLayers.encryption.leftSources,
        certificate.booleanLayers.encryption.rightSources,
        certificate.booleanLayers.encryption.onePublicKey]
      [certificate.booleanLayers.encryption.finalPublicKeys]) := by
  apply booleanSequentialLoopResolution_of_verified
  have layers := verified.booleanLayersMatch
  unfold verifyBooleanLayers at layers
  simp only [Bool.and_eq_true] at layers
  have loop : verifyPublicKeyBooleanLoop workflow certificate.booleanLayers.encryption = true := by
    aesop
  unfold verifyPublicKeyBooleanLoop at loop
  simp only [Bool.and_eq_true] at loop
  aesop

/-- The accepted encoding Boolean layer scan is the exact existing three-component sequential
loop. -/
theorem VerifiedDiamondLayout.encodingBooleanLoopResolution
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    Nonempty (BooleanSequentialLoopResolution workflow
      certificate.booleanLayers.decryption.layerScan
      certificate.booleanLayers.decryption.bodyScope
      [certificate.booleanLayers.decryption.initialVectors,
        certificate.booleanLayers.decryption.initialPublicKeys,
        certificate.booleanLayers.decryption.initialPlaintexts]
      [certificate.booleanLayers.decryption.activeGateCounts,
        certificate.booleanLayers.decryption.gateKinds,
        certificate.booleanLayers.decryption.leftSources,
        certificate.booleanLayers.decryption.rightSources,
        certificate.booleanLayers.decryption.oneVector,
        certificate.booleanLayers.decryption.onePublicKey,
        certificate.booleanLayers.decryption.onePlaintext]
      [certificate.booleanLayers.decryption.finalVectors,
        certificate.booleanLayers.decryption.finalPublicKeys,
        certificate.booleanLayers.decryption.finalPlaintexts]) := by
  apply booleanSequentialLoopResolution_of_verified
  have layers := verified.booleanLayersMatch
  unfold verifyBooleanLayers at layers
  simp only [Bool.and_eq_true] at layers
  have loop : verifyEncodingBooleanLoop workflow certificate.booleanLayers.decryption = true := by
    aesop
  unfold verifyEncodingBooleanLoop at loop
  simp only [Bool.and_eq_true] at loop
  aesop

/-- Exact metadata paths used by one certified Boolean loop.  These equalities state that active
width, opcode, and predecessor selections come from the corresponding loop operands; they do not
reconstruct metadata by searching node contents. -/
structure ExactBooleanMetadataSelections (workflow : Mxx.Ir.Workflow)
    (sequential : CoreNodeRef) (layout : BooleanLayerMetadataLayout)
    (activeOuter opcodeOuter leftOuter rightOuter : CoreOperandRef)
    (activeInner opcodeInner leftInner rightInner : CoreWireRef) : Prop where
  active : verifyScalarMetadata workflow sequential activeOuter activeInner
    layout.activeGateCount = true
  opcode : verifyFamilyMetadata workflow sequential opcodeOuter opcodeInner layout.opcode = true
  left : verifyFamilyMetadata workflow sequential leftOuter leftInner layout.leftSource = true
  right : verifyFamilyMetadata workflow sequential rightOuter rightInner layout.rightSource = true

theorem exactBooleanMetadataSelections_of_verified
    {workflow : Mxx.Ir.Workflow} {sequential : CoreNodeRef}
    {layout : BooleanLayerMetadataLayout}
    {activeOuter opcodeOuter leftOuter rightOuter : CoreOperandRef}
    {activeInner opcodeInner leftInner rightInner : CoreWireRef}
    (verified : verifyBooleanMetadata workflow sequential layout
      [(activeOuter, activeInner), (opcodeOuter, opcodeInner),
        (leftOuter, leftInner), (rightOuter, rightInner)] = true) :
    ExactBooleanMetadataSelections workflow sequential layout
      activeOuter opcodeOuter leftOuter rightOuter
      activeInner opcodeInner leftInner rightInner := by
  unfold verifyBooleanMetadata at verified
  simp only [Bool.and_eq_true] at verified
  exact ⟨verified.1.1.1, verified.1.1.2, verified.1.2, verified.2⟩

/-- Exact six-candidate and inactive-mask wiring for one local public-key gate. -/
structure ExactLocalBooleanGateWiring (workflow : Mxx.Ir.Workflow)
    (layout : LocalBooleanGateLayout) : Prop where
  candidates : (List.ofFn layout.candidateSelect.branches).map (·.wire) = [
    layout.zero.output, layout.one, layout.copy, layout.not.output,
    layout.product.output, layout.xor.output]
  activeBranches : (List.ofFn layout.activeSelect.branches).map (·.wire) = [
    layout.zero.output, layout.candidateSelect.output]
  opcode : layout.bodyOpcode = layout.candidateSelect.selector.wire
  activeMask : verifyActiveMaskFormula workflow layout.parentLoop layout.bodyActiveGateCount
    layout.activeSelect.selector.wire = true

theorem exactLocalBooleanGateWiring_of_verified {workflow : Mxx.Ir.Workflow}
    {layout : LocalBooleanGateLayout} (verified : verifyLocalBooleanGate workflow layout = true) :
    ExactLocalBooleanGateWiring workflow layout := by
  unfold verifyLocalBooleanGate at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at verified
  refine { candidates := ?_, activeBranches := ?_, opcode := ?_, activeMask := ?_ }
  all_goals aesop

/-- Exact six-candidate and inactive-mask wiring for one family-valued encoding component. -/
structure ExactFamilyBooleanGateWiring (workflow : Mxx.Ir.Workflow)
    (layout : FamilyBooleanGateLayout) : Prop where
  candidates : (List.ofFn layout.candidateSelect.branchFamilies).map (·.wire) = [
    layout.zero.outputFamily, layout.oneFamily, layout.copyFamily,
    layout.not.outputFamily, layout.product.outputFamily, layout.xor.outputFamily]
  activeBranches : (List.ofFn layout.activeSelect.branchFamilies).map (·.wire) = [
    layout.zero.outputFamily, layout.candidateSelect.outputFamily]
  opcode : layout.candidateSelect.selectorFamily.wire = layout.opcodeFamily
  activeMaskSelector : layout.activeSelect.selectorFamily.wire = layout.activeFamily
  activeMaskFormula :
    (match layout.activeMask.bodyInputs, layout.activeMask.bodyOutputs with
    | [activeCount], [selector] =>
        verifyActiveMaskFormula workflow layout.activeMask activeCount selector
    | _, _ => false) = true
  output : layout.activeSelect.outputFamily = layout.stateOutput

theorem exactFamilyBooleanGateWiring_of_verified {workflow : Mxx.Ir.Workflow}
    {layout : FamilyBooleanGateLayout} (verified : verifyFamilyBooleanGate workflow layout = true) :
    ExactFamilyBooleanGateWiring workflow layout := by
  unfold verifyFamilyBooleanGate at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  refine {
    candidates := ?_
    activeBranches := ?_
    opcode := ?_
    activeMaskSelector := ?_
    activeMaskFormula := ?_
    output := ?_
  }
  all_goals aesop

theorem activeSelectRole_of_verifiedFamilyGate
    {workflow : Mxx.Ir.Workflow} {layout : FamilyBooleanGateLayout}
    (verified : verifyFamilyBooleanGate workflow layout = true) :
    verifyExactParallelNodeRole workflow layout.activeSelect.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip, .zip] = true := by
  unfold verifyFamilyBooleanGate at verified
  simp only [Bool.and_eq_true] at verified
  aesop

/-- The checked Boolean layout exposes all execution-local choices needed by the generic Boolean
recurrences: exact metadata selection, six candidates, inactive masking, and RHS decomposition
consumers. -/
structure VerifiedBooleanExecutionWiring (workflow : Mxx.Ir.Workflow)
    (layout : BooleanLayersLayout) : Prop where
  encryptionMetadata : ExactBooleanMetadataSelections workflow layout.encryption.layerScan
    layout.encryption.metadata
    layout.encryption.activeGateCounts layout.encryption.gateKinds
    layout.encryption.leftSources layout.encryption.rightSources
    layout.encryption.bodyActiveGateCounts layout.encryption.bodyGateKinds
    layout.encryption.bodyLeftSources layout.encryption.bodyRightSources
  decryptionMetadata : ExactBooleanMetadataSelections workflow layout.decryption.layerScan
    layout.decryption.metadata
    layout.decryption.activeGateCounts layout.decryption.gateKinds
    layout.decryption.leftSources layout.decryption.rightSources
    layout.decryption.bodyActiveGateCounts layout.decryption.bodyGateKinds
    layout.decryption.bodyLeftSources layout.decryption.bodyRightSources
  encryptionGate : ExactLocalBooleanGateWiring workflow layout.encryptionGate
  vectorGate : ExactFamilyBooleanGateWiring workflow layout.decryptionVectors
  publicKeyGate : ExactFamilyBooleanGateWiring workflow layout.decryptionPublicKeys
  plaintextGate : ExactFamilyBooleanGateWiring workflow layout.decryptionPlaintexts
  encryptDecomposition : verifyEncryptDecomposition workflow layout.encryption.bodyScope
    layout.encryptPublicKeyRhsDecomposition = true
  decryptDecomposition : verifyDecryptDecomposition workflow layout.decryption.bodyScope
    layout.decryptEncodingRhsDecomposition = true
  encryptProductUsesLocalRhs : layout.encryptionGate.product.right.wire =
    layout.encryptPublicKeyRhsDecomposition.localDecomposition.materialized

theorem VerifiedDiamondLayout.booleanExecutionWiring
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    VerifiedBooleanExecutionWiring workflow certificate.booleanLayers := by
  have layers := verified.booleanLayersMatch
  unfold verifyBooleanLayers at layers
  simp only [Bool.and_eq_true, decide_eq_true_eq] at layers
  have encryptionLoop :
      verifyPublicKeyBooleanLoop workflow certificate.booleanLayers.encryption = true := by
    aesop
  have decryptionLoop :
      verifyEncodingBooleanLoop workflow certificate.booleanLayers.decryption = true := by
    aesop
  unfold verifyPublicKeyBooleanLoop at encryptionLoop
  unfold verifyEncodingBooleanLoop at decryptionLoop
  simp only [Bool.and_eq_true] at encryptionLoop decryptionLoop
  have encryptionMetadataCheck : verifyBooleanMetadata workflow
      certificate.booleanLayers.encryption.layerScan
      certificate.booleanLayers.encryption.metadata [
        (certificate.booleanLayers.encryption.activeGateCounts,
          certificate.booleanLayers.encryption.bodyActiveGateCounts),
        (certificate.booleanLayers.encryption.gateKinds,
          certificate.booleanLayers.encryption.bodyGateKinds),
        (certificate.booleanLayers.encryption.leftSources,
          certificate.booleanLayers.encryption.bodyLeftSources),
        (certificate.booleanLayers.encryption.rightSources,
          certificate.booleanLayers.encryption.bodyRightSources)] = true := by
    aesop
  have decryptionMetadataCheck : verifyBooleanMetadata workflow
      certificate.booleanLayers.decryption.layerScan
      certificate.booleanLayers.decryption.metadata [
        (certificate.booleanLayers.decryption.activeGateCounts,
          certificate.booleanLayers.decryption.bodyActiveGateCounts),
        (certificate.booleanLayers.decryption.gateKinds,
          certificate.booleanLayers.decryption.bodyGateKinds),
        (certificate.booleanLayers.decryption.leftSources,
          certificate.booleanLayers.decryption.bodyLeftSources),
        (certificate.booleanLayers.decryption.rightSources,
          certificate.booleanLayers.decryption.bodyRightSources)] = true := by
    aesop
  have encryptionGateCheck :
      verifyLocalBooleanGate workflow certificate.booleanLayers.encryptionGate = true := by
    aesop
  have vectorGateCheck :
      verifyFamilyBooleanGate workflow certificate.booleanLayers.decryptionVectors = true := by
    aesop
  have publicKeyGateCheck :
      verifyFamilyBooleanGate workflow certificate.booleanLayers.decryptionPublicKeys = true := by
    aesop
  have plaintextGateCheck :
      verifyFamilyBooleanGate workflow certificate.booleanLayers.decryptionPlaintexts = true := by
    aesop
  have encryptDecompositionCheck : verifyEncryptDecomposition workflow
      certificate.booleanLayers.encryption.bodyScope
      certificate.booleanLayers.encryptPublicKeyRhsDecomposition = true := by
    aesop
  have decryptDecompositionCheck : verifyDecryptDecomposition workflow
      certificate.booleanLayers.decryption.bodyScope
      certificate.booleanLayers.decryptEncodingRhsDecomposition = true := by
    aesop
  have encryptProductUsesLocalRhs :
      certificate.booleanLayers.encryptionGate.product.right.wire =
      certificate.booleanLayers.encryptPublicKeyRhsDecomposition.localDecomposition.materialized :=
    by
    aesop
  exact {
    encryptionMetadata := exactBooleanMetadataSelections_of_verified encryptionMetadataCheck
    decryptionMetadata := exactBooleanMetadataSelections_of_verified decryptionMetadataCheck
    encryptionGate := exactLocalBooleanGateWiring_of_verified encryptionGateCheck
    vectorGate := exactFamilyBooleanGateWiring_of_verified vectorGateCheck
    publicKeyGate := exactFamilyBooleanGateWiring_of_verified publicKeyGateCheck
    plaintextGate := exactFamilyBooleanGateWiring_of_verified plaintextGateCheck
    encryptDecomposition := encryptDecompositionCheck
    decryptDecomposition := decryptDecompositionCheck
    encryptProductUsesLocalRhs
  }

/-- One concrete execution of the certified public-key gate parent loop, recovered from the exact
selected sequential-body path.  No semantic step relation is supplied by the caller: both the
loop node and its selected support member come from the executable IR. -/
structure LocalBooleanParentLoopExecution
    (workflow : Mxx.Ir.Workflow) (layout : LocalBooleanGateLayout)
    (stage : Mxx.Ir.Stage) (scope : Mxx.Ir.Scope) (fuel : Nat)
    (samplers : Mxx.MxxSamplerFamily) (params : Mxx.Ir.ParamEnvironment)
    (inputs outputs : List Mxx.Ir.Value) where
  body : ChildExecutionPath stage scope fuel samplers params inputs outputs
  resolution : CertifiedParallelLoopResolution workflow layout.parentLoop
  loop : ReferencedNodeExecution workflow layout.parentLoop.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
    (scope.inputNames.zip inputs)

/-- Invert an executed Boolean layer body at its certified local public-key parent loop. -/
theorem localBooleanParentLoopExecution_of_childOutcome
    {workflow : Mxx.Ir.Workflow} {layout : LocalBooleanGateLayout}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value} {definition : String}
    (verified : verifyLocalBooleanGate workflow layout = true)
    (definitionFound :
      Mxx.Ir.lookupDefinition definition stage.program.definitions = some scope)
    (scopeResolved : resolveScope workflow layout.parentLoop.operation = some scope)
    (childMember : outputs ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      definition params inputs) :
    Nonempty (LocalBooleanParentLoopExecution workflow layout stage scope fuel samplers params
      inputs outputs) := by
  obtain ⟨body⟩ := childExecutionPath_of_outcome definitionFound childMember
  have parentVerified : verifyParallelLoop workflow layout.parentLoop = true := by
    unfold verifyLocalBooleanGate at verified
    simp only [Bool.and_eq_true] at verified
    have role : verifyExactParallelLoopRole workflow layout.parentLoop
        (.parameter "max_layer_width") 1
        [.zip, .zip, .zip, .broadcast, .broadcast] = true := by
      aesop
    unfold verifyExactParallelLoopRole at role
    simp only [Bool.and_eq_true] at role
    exact role.1.1.1.1.1.1
  let resolution := certifiedParallelLoopResolution_of_verified parentVerified
  have nodeInScope := resolveNode_scopeNode scopeResolved resolution.resolved
  obtain ⟨loop⟩ := body.referencedNodeExecution nodeInScope resolution.resolved
  exact ⟨{ body, resolution, loop }⟩

/-- Recover the public-key gate execution directly from an accepted Diamond certificate and one
actual encryption-layer child outcome. -/
theorem VerifiedDiamondLayout.publicKeyChildExecution
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value}
    (verified : VerifiedDiamondLayout workflow certificate)
    (definitionFound : Mxx.Ir.lookupDefinition
      certificate.booleanLayers.encryption.bodyScope.definitionName
      stage.program.definitions = some scope)
    (scopeResolved : resolveScope workflow
      certificate.booleanLayers.encryptionGate.parentLoop.operation = some scope)
    (childMember : outputs ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      certificate.booleanLayers.encryption.bodyScope.definitionName params inputs) :
    Nonempty (LocalBooleanParentLoopExecution workflow
      certificate.booleanLayers.encryptionGate stage scope fuel samplers params inputs outputs) :=
  by
  have layers := verified.booleanLayersMatch
  unfold verifyBooleanLayers at layers
  simp only [Bool.and_eq_true] at layers
  have gateVerified : verifyLocalBooleanGate workflow
      certificate.booleanLayers.encryptionGate = true := by
    aesop
  exact localBooleanParentLoopExecution_of_childOutcome gateVerified definitionFound
    scopeResolved childMember

/-- The three component-producing active selectors executed by one accepted encoding-layer child.
Each execution is extracted from the retained interpreter path; no semantic transition is supplied
by the caller. -/
structure EncodingBooleanChildExecutions
    (workflow : Mxx.Ir.Workflow) (layout : BooleanLayersLayout)
    (stage : Mxx.Ir.Stage) (scope : Mxx.Ir.Scope) (fuel : Nat)
    (samplers : Mxx.MxxSamplerFamily) (params : Mxx.Ir.ParamEnvironment)
    (inputs outputs : List Mxx.Ir.Value) where
  path : ChildExecutionPath stage scope fuel samplers params inputs outputs
  vectors : ExactParallelNodeExecution workflow
    layout.decryptionVectors.activeSelect.parallelLoop (.parameter "max_layer_width") 1
    [.zip, .zip, .zip] stage scope fuel samplers params inputs
  vectorsRooted : ChildPathRootedNodeExecution path vectors.execution
  publicKeys : ExactParallelNodeExecution workflow
    layout.decryptionPublicKeys.activeSelect.parallelLoop (.parameter "max_layer_width") 1
    [.zip, .zip, .zip] stage scope fuel samplers params inputs
  publicKeysRooted : ChildPathRootedNodeExecution path publicKeys.execution
  plaintexts : ExactParallelNodeExecution workflow
    layout.decryptionPlaintexts.activeSelect.parallelLoop (.parameter "max_layer_width") 1
    [.zip, .zip, .zip] stage scope fuel samplers params inputs
  plaintextsRooted : ChildPathRootedNodeExecution path plaintexts.execution
  decompositions : ExactParallelNodeExecution workflow
    layout.decryptEncodingRhsDecomposition.decompositionLoop
    (.parameter "max_layer_width") 1 [.zip] stage scope fuel samplers params inputs
  decompositionsRooted : ChildPathRootedNodeExecution path decompositions.execution

/-- One exact parallel parent execution, retained on the selected encoding-child path. -/
structure RootedExactParallelExecution
    {workflow : Mxx.Ir.Workflow} {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value}
    (path : ChildExecutionPath stage scope fuel samplers params inputs outputs)
    (operation : CoreNodeRef) (count : Mxx.Ir.IntExpr) (indexSlot : Nat)
    (inputModes : List Mxx.Ir.LoopInputMode) where
  execution : ExactParallelNodeExecution workflow operation count indexSlot inputModes
    stage scope fuel samplers params inputs
  rooted : ChildPathRootedNodeExecution path execution.execution

/-- Product-family parent loops, including both terms of the encoding-vector product. -/
inductive RootedFamilyProductExecutions
    {workflow : Mxx.Ir.Workflow} {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value}
    (path : ChildExecutionPath stage scope fuel samplers params inputs outputs) :
    FamilyProductRef → Type
  | direct (operation : ParallelMatrixBinaryRef)
      (parent : RootedExactParallelExecution (workflow := workflow) path operation.parallelLoop
        (.parameter "max_layer_width") 1 [.zip, .zip]) :
      RootedFamilyProductExecutions (workflow := workflow) path (.direct operation)
  | encodingVector
      (leftTimesRightDecomposition rightTimesLeftPlaintext sum : ParallelMatrixBinaryRef)
      (leftProduct : RootedExactParallelExecution (workflow := workflow) path
        leftTimesRightDecomposition.parallelLoop (.parameter "max_layer_width") 1
        [.zip, .zip])
      (rightProduct : RootedExactParallelExecution (workflow := workflow) path
        rightTimesLeftPlaintext.parallelLoop (.parameter "max_layer_width") 1
        [.zip, .zip])
      (sumProduct : RootedExactParallelExecution (workflow := workflow) path sum.parallelLoop
        (.parameter "max_layer_width") 1 [.zip, .zip]) :
      RootedFamilyProductExecutions (workflow := workflow) path
        (.encodingVector leftTimesRightDecomposition rightTimesLeftPlaintext sum)

/-- Every family-producing parent loop needed for one component's six Boolean candidates.
All executions share the same selected child path. -/
structure RootedFamilyBooleanGateExecutions
    {workflow : Mxx.Ir.Workflow} {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value}
    (path : ChildExecutionPath stage scope fuel samplers params inputs outputs)
    (layout : FamilyBooleanGateLayout) where
  leftSelection : RootedExactParallelExecution (workflow := workflow) path
    layout.leftSelection.parallelLoop.operation
    (.parameter "max_layer_width") 1 [.zip, .broadcast]
  rightSelection : RootedExactParallelExecution (workflow := workflow) path
    layout.rightSelection.parallelLoop.operation
    (.parameter "max_layer_width") 1 [.zip, .broadcast]
  oneRepetition : RootedExactParallelExecution (workflow := workflow) path
    layout.oneRepetition.operation
    (.parameter "max_layer_width") 1 [.broadcast]
  activeMask : RootedExactParallelExecution (workflow := workflow) path layout.activeMask.operation
    (.parameter "max_layer_width") 1 [.broadcast]
  zero : RootedExactParallelExecution (workflow := workflow) path layout.zero.parallelLoop
    (.parameter "max_layer_width") 1 [.zip, .zip]
  not : RootedExactParallelExecution (workflow := workflow) path layout.not.parallelLoop
    (.parameter "max_layer_width") 1 [.zip, .zip]
  product : RootedFamilyProductExecutions (workflow := workflow) path layout.product
  sum : RootedExactParallelExecution (workflow := workflow) path layout.sum.parallelLoop
    (.parameter "max_layer_width") 1 [.zip, .zip]
  twoProduct : RootedExactParallelExecution (workflow := workflow) path
    layout.twoProduct.parallelLoop
    (.parameter "max_layer_width") 1 [.zip, .broadcast]
  xor : RootedExactParallelExecution (workflow := workflow) path layout.xor.parallelLoop
    (.parameter "max_layer_width") 1 [.zip, .zip]
  candidateSelect : RootedExactParallelExecution (workflow := workflow) path
    layout.candidateSelect.parallelLoop
    (.parameter "max_layer_width") 1 (List.replicate 7 .zip)
  activeSelect : RootedExactParallelExecution (workflow := workflow) path
    layout.activeSelect.parallelLoop
    (.parameter "max_layer_width") 1 [.zip, .zip, .zip]

theorem ChildExecutionPath.rootedExactParallelExecution
    {workflow : Mxx.Ir.Workflow} {operation : CoreNodeRef}
    {count : Mxx.Ir.IntExpr} {indexSlot : Nat}
    {inputModes : List Mxx.Ir.LoopInputMode}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value}
    (path : ChildExecutionPath stage scope fuel samplers params inputs outputs)
    (verified : verifyExactParallelNodeRole workflow operation count indexSlot inputModes = true)
    (scopeResolved : resolveScope workflow operation = some scope) :
    Nonempty (RootedExactParallelExecution (workflow := workflow) path operation count indexSlot
      inputModes) := by
  obtain ⟨execution, rooted⟩ :=
    path.rootedExactParallelNodeExecution verified scopeResolved
  exact ⟨{ execution, rooted }⟩

def familyProductParentNodes : FamilyProductRef → List CoreNodeRef
  | .direct operation => [operation.parallelLoop]
  | .encodingVector left right sum =>
      [left.parallelLoop, right.parallelLoop, sum.parallelLoop]

def familyGateParentNodes (layout : FamilyBooleanGateLayout) : List CoreNodeRef :=
  [layout.leftSelection.parallelLoop.operation, layout.rightSelection.parallelLoop.operation,
    layout.oneRepetition.operation, layout.activeMask.operation, layout.zero.parallelLoop,
    layout.not.parallelLoop] ++ familyProductParentNodes layout.product ++
    [layout.sum.parallelLoop, layout.twoProduct.parallelLoop, layout.xor.parallelLoop,
      layout.candidateSelect.parallelLoop, layout.activeSelect.parallelLoop]

theorem rootedFamilyProductExecutions_of_verified
    {workflow : Mxx.Ir.Workflow} {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value}
    (path : ChildExecutionPath stage scope fuel samplers params inputs outputs)
    (product : FamilyProductRef) (verified : verifyFamilyProduct workflow product = true)
    (scopes : ∀ operation ∈ familyProductParentNodes product,
      resolveScope workflow operation = some scope) :
    Nonempty (RootedFamilyProductExecutions (workflow := workflow) path product) := by
  cases product with
  | direct operation =>
      have role : verifyExactParallelNodeRole workflow operation.parallelLoop
          (.parameter "max_layer_width") 1 [.zip, .zip] = true := by
        unfold verifyFamilyProduct at verified
        simp only [Bool.and_eq_true] at verified
        aesop
      obtain ⟨parent⟩ := path.rootedExactParallelExecution role
        (scopes operation.parallelLoop (by simp [familyProductParentNodes]))
      exact ⟨.direct operation parent⟩
  | encodingVector left right sum =>
      have leftRole : verifyExactParallelNodeRole workflow left.parallelLoop
          (.parameter "max_layer_width") 1 [.zip, .zip] = true := by
        unfold verifyFamilyProduct at verified
        simp only [Bool.and_eq_true] at verified
        aesop
      have rightRole : verifyExactParallelNodeRole workflow right.parallelLoop
          (.parameter "max_layer_width") 1 [.zip, .zip] = true := by
        unfold verifyFamilyProduct at verified
        simp only [Bool.and_eq_true] at verified
        aesop
      have sumRole : verifyExactParallelNodeRole workflow sum.parallelLoop
          (.parameter "max_layer_width") 1 [.zip, .zip] = true := by
        unfold verifyFamilyProduct at verified
        simp only [Bool.and_eq_true] at verified
        aesop
      obtain ⟨leftProduct⟩ := path.rootedExactParallelExecution leftRole
        (scopes left.parallelLoop (by simp [familyProductParentNodes]))
      obtain ⟨rightProduct⟩ := path.rootedExactParallelExecution rightRole
        (scopes right.parallelLoop (by simp [familyProductParentNodes]))
      obtain ⟨sumProduct⟩ := path.rootedExactParallelExecution sumRole
        (scopes sum.parallelLoop (by simp [familyProductParentNodes]))
      exact ⟨.encodingVector left right sum leftProduct rightProduct sumProduct⟩

set_option maxHeartbeats 800000 in
theorem rootedFamilyBooleanGateExecutions_of_verified
    {workflow : Mxx.Ir.Workflow} {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value}
    (path : ChildExecutionPath stage scope fuel samplers params inputs outputs)
    (layout : FamilyBooleanGateLayout) (verified : verifyFamilyBooleanGate workflow layout = true)
    (scopes : ∀ operation ∈ familyGateParentNodes layout,
      resolveScope workflow operation = some scope) :
    Nonempty (RootedFamilyBooleanGateExecutions (workflow := workflow) path layout) := by
  have role (operation : CoreNodeRef) (modes : List Mxx.Ir.LoopInputMode)
      (member : operation ∈ familyGateParentNodes layout)
      (checked : verifyExactParallelNodeRole workflow operation
        (.parameter "max_layer_width") 1 modes = true) :
      Nonempty (RootedExactParallelExecution (workflow := workflow) path operation
        (.parameter "max_layer_width") 1 modes) :=
    path.rootedExactParallelExecution checked (scopes operation member)
  have leftRole : verifyExactParallelNodeRole workflow
      layout.leftSelection.parallelLoop.operation (.parameter "max_layer_width") 1
      [.zip, .broadcast] = true := by
    have checked : verifyExactParallelLoopRole workflow layout.leftSelection.parallelLoop
        (.parameter "max_layer_width") 1 [.zip, .broadcast] = true := by
      unfold verifyFamilyBooleanGate verifyExactFamilyGetRole at verified
      simp only [Bool.and_eq_true] at verified
      aesop
    simpa [CertifiedLoopInputMode.toIr] using exactParallelNodeRole_of_loopRole checked
  have rightRole : verifyExactParallelNodeRole workflow
      layout.rightSelection.parallelLoop.operation (.parameter "max_layer_width") 1
      [.zip, .broadcast] = true := by
    have checked : verifyExactParallelLoopRole workflow layout.rightSelection.parallelLoop
        (.parameter "max_layer_width") 1 [.zip, .broadcast] = true := by
      unfold verifyFamilyBooleanGate verifyExactFamilyGetRole at verified
      simp only [Bool.and_eq_true] at verified
      aesop
    simpa [CertifiedLoopInputMode.toIr] using exactParallelNodeRole_of_loopRole checked
  have oneRole : verifyExactParallelNodeRole workflow layout.oneRepetition.operation
      (.parameter "max_layer_width") 1 [.broadcast] = true := by
    have checked : verifyExactParallelLoopRole workflow layout.oneRepetition
        (.parameter "max_layer_width") 1 [.broadcast] = true := by
      unfold verifyFamilyBooleanGate at verified
      simp only [Bool.and_eq_true] at verified
      aesop
    simpa [CertifiedLoopInputMode.toIr] using exactParallelNodeRole_of_loopRole checked
  have activeMaskRole : verifyExactParallelNodeRole workflow layout.activeMask.operation
      (.parameter "max_layer_width") 1 [.broadcast] = true := by
    have checked : verifyExactParallelLoopRole workflow layout.activeMask
        (.parameter "max_layer_width") 1 [.broadcast] = true := by
      unfold verifyFamilyBooleanGate at verified
      simp only [Bool.and_eq_true] at verified
      aesop
    simpa [CertifiedLoopInputMode.toIr] using exactParallelNodeRole_of_loopRole checked
  have zeroRole : verifyExactParallelNodeRole workflow layout.zero.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip] = true := by
    unfold verifyFamilyBooleanGate at verified
    simp only [Bool.and_eq_true] at verified
    aesop
  have notRole : verifyExactParallelNodeRole workflow layout.not.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip] = true := by
    unfold verifyFamilyBooleanGate at verified
    simp only [Bool.and_eq_true] at verified
    aesop
  have productVerified : verifyFamilyProduct workflow layout.product = true := by
    unfold verifyFamilyBooleanGate at verified
    simp only [Bool.and_eq_true] at verified
    aesop
  have sumRole : verifyExactParallelNodeRole workflow layout.sum.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip] = true := by
    unfold verifyFamilyBooleanGate at verified
    simp only [Bool.and_eq_true] at verified
    aesop
  have twoRole : verifyExactParallelNodeRole workflow layout.twoProduct.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .broadcast] = true := by
    unfold verifyFamilyBooleanGate at verified
    simp only [Bool.and_eq_true] at verified
    aesop
  have xorRole : verifyExactParallelNodeRole workflow layout.xor.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip] = true := by
    unfold verifyFamilyBooleanGate at verified
    simp only [Bool.and_eq_true] at verified
    aesop
  have candidateRole : verifyExactParallelNodeRole workflow
      layout.candidateSelect.parallelLoop (.parameter "max_layer_width") 1
      (List.replicate 7 .zip) = true := by
    unfold verifyFamilyBooleanGate at verified
    simp only [Bool.and_eq_true] at verified
    aesop
  have activeRole : verifyExactParallelNodeRole workflow layout.activeSelect.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip, .zip] = true := by
    unfold verifyFamilyBooleanGate at verified
    simp only [Bool.and_eq_true] at verified
    aesop
  obtain ⟨leftSelection⟩ := role _ _ (by simp [familyGateParentNodes]) leftRole
  obtain ⟨rightSelection⟩ := role _ _ (by simp [familyGateParentNodes]) rightRole
  obtain ⟨oneRepetition⟩ := role _ _ (by simp [familyGateParentNodes]) oneRole
  obtain ⟨activeMask⟩ := role _ _ (by simp [familyGateParentNodes]) activeMaskRole
  obtain ⟨zero⟩ := role _ _ (by simp [familyGateParentNodes]) zeroRole
  obtain ⟨not⟩ := role _ _ (by simp [familyGateParentNodes]) notRole
  obtain ⟨product⟩ := rootedFamilyProductExecutions_of_verified path layout.product
    productVerified (fun operation member ↦ scopes operation (by
      simp only [familyGateParentNodes, List.mem_append]
      exact Or.inl (Or.inr member)))
  obtain ⟨sum⟩ := role _ _ (by simp [familyGateParentNodes]) sumRole
  obtain ⟨twoProduct⟩ := role _ _ (by simp [familyGateParentNodes]) twoRole
  obtain ⟨xor⟩ := role _ _ (by simp [familyGateParentNodes]) xorRole
  obtain ⟨candidateSelect⟩ := role _ _ (by simp [familyGateParentNodes]) candidateRole
  obtain ⟨activeSelect⟩ := role _ _ (by simp [familyGateParentNodes]) activeRole
  exact ⟨{
    leftSelection
    rightSelection
    oneRepetition
    activeMask
    zero
    not
    product
    sum
    twoProduct
    xor
    candidateSelect
    activeSelect
  }⟩

/-- Complete same-path executable witness for one encoding Boolean child. -/
structure CompleteEncodingBooleanChildExecutions
    (workflow : Mxx.Ir.Workflow) (layout : BooleanLayersLayout)
    (stage : Mxx.Ir.Stage) (scope : Mxx.Ir.Scope) (fuel : Nat)
    (samplers : Mxx.MxxSamplerFamily) (params : Mxx.Ir.ParamEnvironment)
    (inputs outputs : List Mxx.Ir.Value) where
  base : EncodingBooleanChildExecutions workflow layout stage scope fuel samplers params inputs
    outputs
  vectorCandidates : RootedFamilyBooleanGateExecutions (workflow := workflow) base.path
    layout.decryptionVectors
  publicKeyCandidates : RootedFamilyBooleanGateExecutions (workflow := workflow) base.path
    layout.decryptionPublicKeys
  plaintextCandidates : RootedFamilyBooleanGateExecutions (workflow := workflow) base.path
    layout.decryptionPlaintexts

/-- A concrete matrix-family output recovers the exact pointwise binary-operation trace.  The
family shape excludes the interpreter's invalid fallback, so lookup and count success are
derived rather than required from the caller. -/
theorem parallelMatrixBinaryTrace_of_familyOutput
    {workflow : Mxx.Ir.Workflow} {reference : ParallelMatrixBinaryRef}
    {expected : Mxx.Ir.NodeKind}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : ParallelMatrixBinaryResolution workflow reference expected)
    (execution : ReferencedNodeExecution workflow reference.parallelLoop runChild samplers params
      inputs)
    (outputFamily : List Mxx.Ir.Value)
    (valuesEq : execution.values = [.family outputFamily]) :
    Nonempty (ParallelMatrixBinaryTrace workflow reference expected runChild samplers params
      inputs execution) := by
  have executionResolved := execution.resolved
  have loopResolved := resolution.loopResolved
  rw [executionResolved] at loopResolved
  have nodeEq := Option.some.inj loopResolved
  have member : execution.values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs
      execution.before {
        kind := .parallelLoop reference.bodyScope.definitionName resolution.count
          resolution.indexSlot resolution.bindings resolution.modes
        arguments := [wireRef reference.leftFamily.wire, wireRef reference.rightFamily.wire]
        outputCount := 1
      } := by
    simpa [nodeEq] using execution.member
  cases argumentsEvaluate :
      [wireRef reference.leftFamily.wire, wireRef reference.rightFamily.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) with
  | none =>
      simp [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate] at member
      rw [member] at valuesEq
      simp at valuesEq
  | some argumentValues =>
      cases countEvaluate : resolution.count.evaluate params with
      | none =>
          simp [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate, countEvaluate] at member
          rw [member] at valuesEq
          simp at valuesEq
      | some evaluatedCount =>
          exact parallelMatrixBinaryTrace_of_resolution resolution execution argumentValues
            evaluatedCount argumentsEvaluate countEvaluate

/-- A checked parallel matrix-multiplication body is deterministic on two concrete matrix
arguments.  The proof follows the retained child path: the boundary identifies the two body
inputs, the checked local node supplies multiplication, and the checked scope output identifies
that node's result. -/
theorem parallelMatrixMultiplyChildOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelMatrixBinaryRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {left right : Mxx.Matrix} {values : List Mxx.Ir.Value}
    (verified : verifyParallelMatrixBinary workflow reference .matrixMultiply = true)
    (ssaOrder : verifyScopeSsaOrder body = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop with scope := reference.bodyScope } = some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.bodyScope.definitionName
      stage.program.definitions = some body)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.bodyScope.definitionName params [.matrix left, .matrix right]) :
    values = [.matrix (Mxx.matrixMultiply left right)] := by
  obtain ⟨path⟩ := childExecutionPath_of_outcome definitionFound childMember
  obtain ⟨resolution⟩ := verifyParallelMatrixBinary_resolution verified
  have operationLeftEq : reference.operation.left.wire = reference.bodyLeft := by
    have wiring := verified
    unfold verifyParallelMatrixBinary at wiring
    simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
    aesop
  have operationRightEq : reference.operation.right.wire = reference.bodyRight := by
    have wiring := verified
    unfold verifyParallelMatrixBinary at wiring
    simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
    aesop
  let localResolution : LocalMatrixBinaryResolution workflow reference.operation
      .matrixMultiply := {
    resolved := by
      simpa [operationLeftEq, operationRightEq] using resolution.operationResolved
  }
  have boundaryVerified : verifyParallelBoundary workflow reference.parallelLoop
      reference.bodyScope [reference.leftFamily, reference.rightFamily]
      [reference.bodyLeft, reference.bodyRight] reference.bodyOutput reference.outputFamily =
        true := by
    have wiring := verified
    unfold verifyParallelMatrixBinary at wiring
    simp only [Bool.and_eq_true] at wiring
    aesop
  have checked := boundaryVerified
  unfold verifyParallelBoundary at checked
  rw [resolution.loopResolved, bodyResolved] at checked
  simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
  ·
      have namesNodup : body.inputNames.Nodup := by aesop
      have namesLength : body.inputNames.length = 2 := by aesop
      have inputWires : scopeInputWires body =
          [wireRef reference.bodyLeft, wireRef reference.bodyRight] := by
        aesop
      have bodyOutputs : body.outputs.map Prod.snd = [wireRef reference.bodyOutput] := by
        change scopeOutputWires body = [wireRef reference.bodyOutput]
        aesop
      have outputWire : reference.operation.output = reference.bodyOutput := by
        have wiring := verified
        unfold verifyParallelMatrixBinary at wiring
        simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
        aesop
      have outputOwner : reference.operation.output.node =
          reference.operation.operation := by
        have wiring := verified
        unfold verifyParallelMatrixBinary verifyMatrixBinary verifyBinaryNode
          verifyOperationOutput at wiring
        simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
        aesop
      have outputPort : reference.operation.output.port = 0 := by
        have outputVerified : verifyWire workflow reference.operation.output = true := by
          have wiring := verified
          unfold verifyParallelMatrixBinary verifyMatrixBinary verifyBinaryNode
            verifyOperationOutput at wiring
          simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
          aesop
        have portLt : reference.operation.output.port < 1 := by
          simpa [verifyWire, outputOwner, resolution.operationResolved] using outputVerified
        omega
      have bodyOutputOwner : reference.bodyOutput.node = reference.operation.operation := by
        rw [← outputWire]
        exact outputOwner
      have bodyOutputPort : reference.bodyOutput.port = 0 := by
        rw [← outputWire]
        exact outputPort
      have operationScope : reference.operation.operation.scope = reference.bodyScope := by
        calc
          reference.operation.operation.scope = reference.operation.output.node.scope :=
            congrArg CoreNodeRef.scope outputOwner.symm
          _ = reference.bodyOutput.node.scope := by rw [outputWire]
          _ = reference.bodyScope := by aesop
      have operationStage : reference.operation.operation.stage = reference.parallelLoop.stage := by
        calc
          reference.operation.operation.stage = reference.operation.output.node.stage :=
            congrArg CoreNodeRef.stage outputOwner.symm
          _ = reference.bodyOutput.node.stage := by rw [outputWire]
          _ = reference.parallelLoop.stage := by aesop
      have operationScopeResolved : resolveScope workflow reference.operation.operation =
          some body := by
        have resolved := bodyResolved
        unfold resolveScope at resolved ⊢
        rw [operationScope, operationStage]
        exact resolved
      have operationInScope := resolveNode_scopeNode operationScopeResolved
        resolution.operationResolved
      obtain ⟨operationExecution, operationRooted⟩ :=
        path.rootedReferencedNodeExecution operationInScope resolution.operationResolved
      have operationNodeEq : operationExecution.node = {
          kind := .matrixMultiply
          arguments := [wireRef reference.bodyLeft, wireRef reference.bodyRight]
          outputCount := 1
        } := Option.some.inj
          (operationExecution.resolved.symm.trans resolution.operationResolved)
      let leftName := body.inputNames.get ⟨0, by omega⟩
      let rightName := body.inputNames.get ⟨1, by omega⟩
      have leftNameAt : body.inputNames[0]? = some leftName := by
        simp [leftName, namesLength]
      have rightNameAt : body.inputNames[1]? = some rightName := by
        simp [rightName, namesLength]
      have leftWireAt : (scopeInputWires body)[0]? = some (wireRef reference.bodyLeft) := by
        simp [inputWires]
      have rightWireAt : (scopeInputWires body)[1]? = some (wireRef reference.bodyRight) := by
        simp [inputWires]
      have leftVerified : verifyWire workflow reference.bodyLeft = true := by aesop
      have rightVerified : verifyWire workflow reference.bodyRight = true := by aesop
      have leftScope : resolveScope workflow reference.bodyLeft.node = some body := by
        have leftStage : reference.bodyLeft.node.stage = reference.parallelLoop.stage := by aesop
        have leftBodyScope : reference.bodyLeft.node.scope = reference.bodyScope := by aesop
        simpa [resolveScope, leftStage, leftBodyScope] using bodyResolved
      have rightScope : resolveScope workflow reference.bodyRight.node = some body := by
        have rightStage : reference.bodyRight.node.stage = reference.parallelLoop.stage := by aesop
        have rightBodyScope : reference.bodyRight.node.scope = reference.bodyScope := by aesop
        simpa [resolveScope, rightStage, rightBodyScope] using bodyResolved
      have leftValid : ∃ node, body.nodes[(wireRef reference.bodyLeft).node]? = some node ∧
          (wireRef reference.bodyLeft).port < node.outputCount := by
        exact verifyWire_scopeValid leftVerified leftScope
      have rightValid : ∃ node, body.nodes[(wireRef reference.bodyRight).node]? = some node ∧
          (wireRef reference.bodyRight).port < node.outputCount := by
        exact verifyWire_scopeValid rightVerified rightScope
      have leftFinal := path.inputWireValue namesNodup (by simp [namesLength, inputWires])
        namesLength 0 leftName (wireRef reference.bodyLeft) (.matrix left) leftNameAt leftWireAt
        (by simp) leftValid
      have rightFinal := path.inputWireValue namesNodup (by simp [namesLength, inputWires])
        namesLength 1 rightName (wireRef reference.bodyRight) (.matrix right) rightNameAt
        rightWireAt (by simp) rightValid
      have leftBefore := operationRooted.argumentFromFinal ssaOrder operationScopeResolved
        (wireRef reference.bodyLeft) (by simp [operationNodeEq]) leftFinal
      have rightBefore := operationRooted.argumentFromFinal ssaOrder operationScopeResolved
        (wireRef reference.bodyRight) (by simp [operationNodeEq]) rightFinal
      have argumentsEvaluate :
          [wireRef reference.operation.left.wire,
            wireRef reference.operation.right.wire].mapM
              (fun wire ↦ Mxx.Ir.lookupWire wire operationExecution.before) =
            some [.matrix left, .matrix right] := by
        have leftWire : reference.operation.left.wire = reference.bodyLeft := by aesop
        have rightWire : reference.operation.right.wire = reference.bodyRight := by aesop
        simp [leftWire, rightWire, leftBefore, rightBefore]
      have operationOutcome := localResolution.multiplyOutcome operationExecution left right
        argumentsEvaluate
      have outputLookup : Mxx.Ir.lookupWire (wireRef reference.bodyOutput) path.finalWires =
          some (.matrix (Mxx.matrixMultiply left right)) := by
        have portValid : 0 < operationExecution.values.length := by
          rw [operationOutcome]
          simp
        have operationValue : operationExecution.values.get ⟨0, portValid⟩ =
            .matrix (Mxx.matrixMultiply left right) := by
          simp [operationOutcome]
        have observed := operationRooted.outputFinal 0 portValid
        rw [operationValue] at observed
        have outputRef : wireRef reference.bodyOutput =
            ({ node := reference.operation.operation.node, port := 0 } : Mxx.Ir.WireRef) := by
          have bodyNodeId : reference.bodyOutput.node.node =
              reference.operation.operation.node :=
            congrArg (fun node : CoreNodeRef ↦ node.node) bodyOutputOwner
          show ({ node := reference.bodyOutput.node.node, port := reference.bodyOutput.port } :
            Mxx.Ir.WireRef) = _
          rw [bodyNodeId, bodyOutputPort]
        rw [outputRef]
        exact observed
      exact path.singleOutput (wireRef reference.bodyOutput)
        (.matrix (Mxx.matrixMultiply left right)) bodyOutputs outputLookup

/-- Exact child semantics of a checked parallel matrix addition. -/
theorem parallelMatrixAddChildOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelMatrixBinaryRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {left right : Mxx.Matrix} {values : List Mxx.Ir.Value}
    (verified : verifyParallelMatrixBinary workflow reference .matrixAdd = true)
    (ssaOrder : verifyScopeSsaOrder body = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop with scope := reference.bodyScope } = some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.bodyScope.definitionName
      stage.program.definitions = some body)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.bodyScope.definitionName params [.matrix left, .matrix right]) :
    values = [.matrix (Mxx.matrixAdd left right)] := by
  obtain ⟨path⟩ := childExecutionPath_of_outcome definitionFound childMember
  obtain ⟨resolution⟩ := verifyParallelMatrixBinary_resolution verified
  have operationLeftEq : reference.operation.left.wire = reference.bodyLeft := by
    have wiring := verified
    unfold verifyParallelMatrixBinary at wiring
    simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
    aesop
  have operationRightEq : reference.operation.right.wire = reference.bodyRight := by
    have wiring := verified
    unfold verifyParallelMatrixBinary at wiring
    simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
    aesop
  let localResolution : LocalMatrixBinaryResolution workflow reference.operation
      .matrixAdd := {
    resolved := by
      simpa [operationLeftEq, operationRightEq] using resolution.operationResolved
  }
  have boundaryVerified : verifyParallelBoundary workflow reference.parallelLoop
      reference.bodyScope [reference.leftFamily, reference.rightFamily]
      [reference.bodyLeft, reference.bodyRight] reference.bodyOutput reference.outputFamily =
        true := by
    have wiring := verified
    unfold verifyParallelMatrixBinary at wiring
    simp only [Bool.and_eq_true] at wiring
    aesop
  have checked := boundaryVerified
  unfold verifyParallelBoundary at checked
  rw [resolution.loopResolved, bodyResolved] at checked
  simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
  ·
      have namesNodup : body.inputNames.Nodup := by aesop
      have namesLength : body.inputNames.length = 2 := by aesop
      have inputWires : scopeInputWires body =
          [wireRef reference.bodyLeft, wireRef reference.bodyRight] := by
        aesop
      have bodyOutputs : body.outputs.map Prod.snd = [wireRef reference.bodyOutput] := by
        change scopeOutputWires body = [wireRef reference.bodyOutput]
        aesop
      have outputWire : reference.operation.output = reference.bodyOutput := by
        have wiring := verified
        unfold verifyParallelMatrixBinary at wiring
        simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
        aesop
      have outputOwner : reference.operation.output.node =
          reference.operation.operation := by
        have wiring := verified
        unfold verifyParallelMatrixBinary verifyMatrixBinary verifyBinaryNode
          verifyOperationOutput at wiring
        simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
        aesop
      have outputPort : reference.operation.output.port = 0 := by
        have outputVerified : verifyWire workflow reference.operation.output = true := by
          have wiring := verified
          unfold verifyParallelMatrixBinary verifyMatrixBinary verifyBinaryNode
            verifyOperationOutput at wiring
          simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
          aesop
        have portLt : reference.operation.output.port < 1 := by
          simpa [verifyWire, outputOwner, resolution.operationResolved] using outputVerified
        omega
      have bodyOutputOwner : reference.bodyOutput.node = reference.operation.operation := by
        rw [← outputWire]
        exact outputOwner
      have bodyOutputPort : reference.bodyOutput.port = 0 := by
        rw [← outputWire]
        exact outputPort
      have operationScope : reference.operation.operation.scope = reference.bodyScope := by
        calc
          reference.operation.operation.scope = reference.operation.output.node.scope :=
            congrArg CoreNodeRef.scope outputOwner.symm
          _ = reference.bodyOutput.node.scope := by rw [outputWire]
          _ = reference.bodyScope := by aesop
      have operationStage : reference.operation.operation.stage = reference.parallelLoop.stage := by
        calc
          reference.operation.operation.stage = reference.operation.output.node.stage :=
            congrArg CoreNodeRef.stage outputOwner.symm
          _ = reference.bodyOutput.node.stage := by rw [outputWire]
          _ = reference.parallelLoop.stage := by aesop
      have operationScopeResolved : resolveScope workflow reference.operation.operation =
          some body := by
        have resolved := bodyResolved
        unfold resolveScope at resolved ⊢
        rw [operationScope, operationStage]
        exact resolved
      have operationInScope := resolveNode_scopeNode operationScopeResolved
        resolution.operationResolved
      obtain ⟨operationExecution, operationRooted⟩ :=
        path.rootedReferencedNodeExecution operationInScope resolution.operationResolved
      have operationNodeEq : operationExecution.node = {
          kind := .matrixAdd
          arguments := [wireRef reference.bodyLeft, wireRef reference.bodyRight]
          outputCount := 1
        } := Option.some.inj
          (operationExecution.resolved.symm.trans resolution.operationResolved)
      let leftName := body.inputNames.get ⟨0, by omega⟩
      let rightName := body.inputNames.get ⟨1, by omega⟩
      have leftNameAt : body.inputNames[0]? = some leftName := by
        simp [leftName, namesLength]
      have rightNameAt : body.inputNames[1]? = some rightName := by
        simp [rightName, namesLength]
      have leftWireAt : (scopeInputWires body)[0]? = some (wireRef reference.bodyLeft) := by
        simp [inputWires]
      have rightWireAt : (scopeInputWires body)[1]? = some (wireRef reference.bodyRight) := by
        simp [inputWires]
      have leftVerified : verifyWire workflow reference.bodyLeft = true := by aesop
      have rightVerified : verifyWire workflow reference.bodyRight = true := by aesop
      have leftScope : resolveScope workflow reference.bodyLeft.node = some body := by
        have leftStage : reference.bodyLeft.node.stage = reference.parallelLoop.stage := by aesop
        have leftBodyScope : reference.bodyLeft.node.scope = reference.bodyScope := by aesop
        simpa [resolveScope, leftStage, leftBodyScope] using bodyResolved
      have rightScope : resolveScope workflow reference.bodyRight.node = some body := by
        have rightStage : reference.bodyRight.node.stage = reference.parallelLoop.stage := by aesop
        have rightBodyScope : reference.bodyRight.node.scope = reference.bodyScope := by aesop
        simpa [resolveScope, rightStage, rightBodyScope] using bodyResolved
      have leftValid : ∃ node, body.nodes[(wireRef reference.bodyLeft).node]? = some node ∧
          (wireRef reference.bodyLeft).port < node.outputCount := by
        exact verifyWire_scopeValid leftVerified leftScope
      have rightValid : ∃ node, body.nodes[(wireRef reference.bodyRight).node]? = some node ∧
          (wireRef reference.bodyRight).port < node.outputCount := by
        exact verifyWire_scopeValid rightVerified rightScope
      have leftFinal := path.inputWireValue namesNodup (by simp [namesLength, inputWires])
        namesLength 0 leftName (wireRef reference.bodyLeft) (.matrix left) leftNameAt leftWireAt
        (by simp) leftValid
      have rightFinal := path.inputWireValue namesNodup (by simp [namesLength, inputWires])
        namesLength 1 rightName (wireRef reference.bodyRight) (.matrix right) rightNameAt
        rightWireAt (by simp) rightValid
      have leftBefore := operationRooted.argumentFromFinal ssaOrder operationScopeResolved
        (wireRef reference.bodyLeft) (by simp [operationNodeEq]) leftFinal
      have rightBefore := operationRooted.argumentFromFinal ssaOrder operationScopeResolved
        (wireRef reference.bodyRight) (by simp [operationNodeEq]) rightFinal
      have argumentsEvaluate :
          [wireRef reference.operation.left.wire,
            wireRef reference.operation.right.wire].mapM
              (fun wire ↦ Mxx.Ir.lookupWire wire operationExecution.before) =
            some [.matrix left, .matrix right] := by
        have leftWire : reference.operation.left.wire = reference.bodyLeft := by aesop
        have rightWire : reference.operation.right.wire = reference.bodyRight := by aesop
        simp [leftWire, rightWire, leftBefore, rightBefore]
      have operationOutcome := localResolution.addOutcome operationExecution left right
        argumentsEvaluate
      have outputLookup : Mxx.Ir.lookupWire (wireRef reference.bodyOutput) path.finalWires =
          some (.matrix (Mxx.matrixAdd left right)) := by
        have portValid : 0 < operationExecution.values.length := by
          rw [operationOutcome]
          simp
        have operationValue : operationExecution.values.get ⟨0, portValid⟩ =
            .matrix (Mxx.matrixAdd left right) := by
          simp [operationOutcome]
        have observed := operationRooted.outputFinal 0 portValid
        rw [operationValue] at observed
        have outputRef : wireRef reference.bodyOutput =
            ({ node := reference.operation.operation.node, port := 0 } : Mxx.Ir.WireRef) := by
          have bodyNodeId : reference.bodyOutput.node.node =
              reference.operation.operation.node :=
            congrArg (fun node : CoreNodeRef ↦ node.node) bodyOutputOwner
          show ({ node := reference.bodyOutput.node.node, port := reference.bodyOutput.port } :
            Mxx.Ir.WireRef) = _
          rw [bodyNodeId, bodyOutputPort]
        rw [outputRef]
        exact observed
      exact path.singleOutput (wireRef reference.bodyOutput)
        (.matrix (Mxx.matrixAdd left right)) bodyOutputs outputLookup

/-- Exact child semantics of a checked parallel matrix subtraction. -/
theorem parallelMatrixSubtractChildOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelMatrixBinaryRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {left right : Mxx.Matrix} {values : List Mxx.Ir.Value}
    (verified : verifyParallelMatrixBinary workflow reference .matrixSubtract = true)
    (ssaOrder : verifyScopeSsaOrder body = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop with scope := reference.bodyScope } = some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.bodyScope.definitionName
      stage.program.definitions = some body)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.bodyScope.definitionName params [.matrix left, .matrix right]) :
    values = [.matrix (Mxx.matrixSubtract left right)] := by
  obtain ⟨path⟩ := childExecutionPath_of_outcome definitionFound childMember
  obtain ⟨resolution⟩ := verifyParallelMatrixBinary_resolution verified
  have operationLeftEq : reference.operation.left.wire = reference.bodyLeft := by
    have wiring := verified
    unfold verifyParallelMatrixBinary at wiring
    simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
    aesop
  have operationRightEq : reference.operation.right.wire = reference.bodyRight := by
    have wiring := verified
    unfold verifyParallelMatrixBinary at wiring
    simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
    aesop
  let localResolution : LocalMatrixBinaryResolution workflow reference.operation
      .matrixSubtract := {
    resolved := by
      simpa [operationLeftEq, operationRightEq] using resolution.operationResolved
  }
  have boundaryVerified : verifyParallelBoundary workflow reference.parallelLoop
      reference.bodyScope [reference.leftFamily, reference.rightFamily]
      [reference.bodyLeft, reference.bodyRight] reference.bodyOutput reference.outputFamily =
        true := by
    have wiring := verified
    unfold verifyParallelMatrixBinary at wiring
    simp only [Bool.and_eq_true] at wiring
    aesop
  have checked := boundaryVerified
  unfold verifyParallelBoundary at checked
  rw [resolution.loopResolved, bodyResolved] at checked
  simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
  ·
      have namesNodup : body.inputNames.Nodup := by aesop
      have namesLength : body.inputNames.length = 2 := by aesop
      have inputWires : scopeInputWires body =
          [wireRef reference.bodyLeft, wireRef reference.bodyRight] := by
        aesop
      have bodyOutputs : body.outputs.map Prod.snd = [wireRef reference.bodyOutput] := by
        change scopeOutputWires body = [wireRef reference.bodyOutput]
        aesop
      have outputWire : reference.operation.output = reference.bodyOutput := by
        have wiring := verified
        unfold verifyParallelMatrixBinary at wiring
        simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
        aesop
      have outputOwner : reference.operation.output.node =
          reference.operation.operation := by
        have wiring := verified
        unfold verifyParallelMatrixBinary verifyMatrixBinary verifyBinaryNode
          verifyOperationOutput at wiring
        simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
        aesop
      have outputPort : reference.operation.output.port = 0 := by
        have outputVerified : verifyWire workflow reference.operation.output = true := by
          have wiring := verified
          unfold verifyParallelMatrixBinary verifyMatrixBinary verifyBinaryNode
            verifyOperationOutput at wiring
          simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
          aesop
        have portLt : reference.operation.output.port < 1 := by
          simpa [verifyWire, outputOwner, resolution.operationResolved] using outputVerified
        omega
      have bodyOutputOwner : reference.bodyOutput.node = reference.operation.operation := by
        rw [← outputWire]
        exact outputOwner
      have bodyOutputPort : reference.bodyOutput.port = 0 := by
        rw [← outputWire]
        exact outputPort
      have operationScope : reference.operation.operation.scope = reference.bodyScope := by
        calc
          reference.operation.operation.scope = reference.operation.output.node.scope :=
            congrArg CoreNodeRef.scope outputOwner.symm
          _ = reference.bodyOutput.node.scope := by rw [outputWire]
          _ = reference.bodyScope := by aesop
      have operationStage : reference.operation.operation.stage = reference.parallelLoop.stage := by
        calc
          reference.operation.operation.stage = reference.operation.output.node.stage :=
            congrArg CoreNodeRef.stage outputOwner.symm
          _ = reference.bodyOutput.node.stage := by rw [outputWire]
          _ = reference.parallelLoop.stage := by aesop
      have operationScopeResolved : resolveScope workflow reference.operation.operation =
          some body := by
        have resolved := bodyResolved
        unfold resolveScope at resolved ⊢
        rw [operationScope, operationStage]
        exact resolved
      have operationInScope := resolveNode_scopeNode operationScopeResolved
        resolution.operationResolved
      obtain ⟨operationExecution, operationRooted⟩ :=
        path.rootedReferencedNodeExecution operationInScope resolution.operationResolved
      have operationNodeEq : operationExecution.node = {
          kind := .matrixSubtract
          arguments := [wireRef reference.bodyLeft, wireRef reference.bodyRight]
          outputCount := 1
        } := Option.some.inj
          (operationExecution.resolved.symm.trans resolution.operationResolved)
      let leftName := body.inputNames.get ⟨0, by omega⟩
      let rightName := body.inputNames.get ⟨1, by omega⟩
      have leftNameAt : body.inputNames[0]? = some leftName := by
        simp [leftName, namesLength]
      have rightNameAt : body.inputNames[1]? = some rightName := by
        simp [rightName, namesLength]
      have leftWireAt : (scopeInputWires body)[0]? = some (wireRef reference.bodyLeft) := by
        simp [inputWires]
      have rightWireAt : (scopeInputWires body)[1]? = some (wireRef reference.bodyRight) := by
        simp [inputWires]
      have leftVerified : verifyWire workflow reference.bodyLeft = true := by aesop
      have rightVerified : verifyWire workflow reference.bodyRight = true := by aesop
      have leftScope : resolveScope workflow reference.bodyLeft.node = some body := by
        have leftStage : reference.bodyLeft.node.stage = reference.parallelLoop.stage := by aesop
        have leftBodyScope : reference.bodyLeft.node.scope = reference.bodyScope := by aesop
        simpa [resolveScope, leftStage, leftBodyScope] using bodyResolved
      have rightScope : resolveScope workflow reference.bodyRight.node = some body := by
        have rightStage : reference.bodyRight.node.stage = reference.parallelLoop.stage := by aesop
        have rightBodyScope : reference.bodyRight.node.scope = reference.bodyScope := by aesop
        simpa [resolveScope, rightStage, rightBodyScope] using bodyResolved
      have leftValid : ∃ node, body.nodes[(wireRef reference.bodyLeft).node]? = some node ∧
          (wireRef reference.bodyLeft).port < node.outputCount := by
        exact verifyWire_scopeValid leftVerified leftScope
      have rightValid : ∃ node, body.nodes[(wireRef reference.bodyRight).node]? = some node ∧
          (wireRef reference.bodyRight).port < node.outputCount := by
        exact verifyWire_scopeValid rightVerified rightScope
      have leftFinal := path.inputWireValue namesNodup (by simp [namesLength, inputWires])
        namesLength 0 leftName (wireRef reference.bodyLeft) (.matrix left) leftNameAt leftWireAt
        (by simp) leftValid
      have rightFinal := path.inputWireValue namesNodup (by simp [namesLength, inputWires])
        namesLength 1 rightName (wireRef reference.bodyRight) (.matrix right) rightNameAt
        rightWireAt (by simp) rightValid
      have leftBefore := operationRooted.argumentFromFinal ssaOrder operationScopeResolved
        (wireRef reference.bodyLeft) (by simp [operationNodeEq]) leftFinal
      have rightBefore := operationRooted.argumentFromFinal ssaOrder operationScopeResolved
        (wireRef reference.bodyRight) (by simp [operationNodeEq]) rightFinal
      have argumentsEvaluate :
          [wireRef reference.operation.left.wire,
            wireRef reference.operation.right.wire].mapM
              (fun wire ↦ Mxx.Ir.lookupWire wire operationExecution.before) =
            some [.matrix left, .matrix right] := by
        have leftWire : reference.operation.left.wire = reference.bodyLeft := by aesop
        have rightWire : reference.operation.right.wire = reference.bodyRight := by aesop
        simp [leftWire, rightWire, leftBefore, rightBefore]
      have operationOutcome := localResolution.subtractOutcome operationExecution left right
        argumentsEvaluate
      have outputLookup : Mxx.Ir.lookupWire (wireRef reference.bodyOutput) path.finalWires =
          some (.matrix (Mxx.matrixSubtract left right)) := by
        have portValid : 0 < operationExecution.values.length := by
          rw [operationOutcome]
          simp
        have operationValue : operationExecution.values.get ⟨0, portValid⟩ =
            .matrix (Mxx.matrixSubtract left right) := by
          simp [operationOutcome]
        have observed := operationRooted.outputFinal 0 portValid
        rw [operationValue] at observed
        have outputRef : wireRef reference.bodyOutput =
            ({ node := reference.operation.operation.node, port := 0 } : Mxx.Ir.WireRef) := by
          have bodyNodeId : reference.bodyOutput.node.node =
              reference.operation.operation.node :=
            congrArg (fun node : CoreNodeRef ↦ node.node) bodyOutputOwner
          show ({ node := reference.bodyOutput.node.node, port := reference.bodyOutput.port } :
            Mxx.Ir.WireRef) = _
          rw [bodyNodeId, bodyOutputPort]
        rw [outputRef]
        exact observed
      exact path.singleOutput (wireRef reference.bodyOutput)
        (.matrix (Mxx.matrixSubtract left right)) bodyOutputs outputLookup

private theorem verifySelect_outputFacts
    {workflow : Mxx.Ir.Workflow} {operation : CoreNodeRef}
    {selector : CoreOperandRef} {branches : List CoreOperandRef} {output : CoreWireRef}
    (verified : verifySelect workflow operation selector branches output = true) :
    output.node = operation ∧ verifyWire workflow output = true := by
  have outputChecked : verifyOperationOutput workflow operation output = true := by
    unfold verifySelect at verified
    simp only [Bool.and_eq_true] at verified
    exact verified.1.2
  unfold verifyOperationOutput at outputChecked
  simpa only [Bool.and_eq_true, decide_eq_true_eq] using outputChecked

private theorem verifyParallelBoundary_innerFact
    {workflow : Mxx.Ir.Workflow} {operation : CoreNodeRef} {bodyScope : ScopeRef}
    {outer : List CoreOperandRef} {inner : List CoreWireRef}
    {bodyOutput output : CoreWireRef}
    (verified : verifyParallelBoundary workflow operation bodyScope outer inner bodyOutput output =
      true) (wire : CoreWireRef) (member : wire ∈ inner) :
    verifyWire workflow wire = true ∧ wire.node.stage = operation.stage ∧
      wire.node.scope = bodyScope := by
  have checked := verified
  unfold verifyParallelBoundary at checked
  simp only [Bool.and_eq_true] at checked
  have innerChecked : inner.all (fun input ↦
      verifyWire workflow input && decide (input.node.stage = operation.stage) &&
        decide (input.node.scope = bodyScope)) = true := by
    aesop
  have wireChecked := (List.all_eq_true.mp innerChecked) wire member
  simp only [Bool.and_eq_true, decide_eq_true_eq] at wireChecked
  exact ⟨wireChecked.1.1, wireChecked.1.2, wireChecked.2⟩

private theorem verifyParallelBoundary_bodyOutputLocation
    {workflow : Mxx.Ir.Workflow} {operation : CoreNodeRef} {bodyScope : ScopeRef}
    {outer : List CoreOperandRef} {inner : List CoreWireRef}
    {bodyOutput output : CoreWireRef}
    (verified : verifyParallelBoundary workflow operation bodyScope outer inner bodyOutput output =
      true) :
    bodyOutput.node.stage = operation.stage ∧ bodyOutput.node.scope = bodyScope := by
  unfold verifyParallelBoundary at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  aesop

/-- Exact child semantics of a checked parallel six-way selector. -/
theorem parallelSixWaySelectChildOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelSixWaySelectRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {opcode : Int} {branches : Fin 6 → Mxx.Matrix} {values : List Mxx.Ir.Value}
    (verified : verifyParallelSixWaySelect workflow reference = true)
    (ssaOrder : verifyScopeSsaOrder body = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop with scope := reference.bodyScope } = some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.bodyScope.definitionName
      stage.program.definitions = some body)
    (opcodeLower : 0 ≤ opcode) (opcodeUpper : opcode ≤ 5)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.bodyScope.definitionName params
        (.integer opcode :: (List.ofFn branches).map Mxx.Ir.Value.matrix)) :
    values = [match MxxWe.DynamicBoolean.gateKind opcode with
      | .constantFalse => .matrix (branches 0)
      | .constantTrue => .matrix (branches 1)
      | .copy => .matrix (branches 2)
      | .not => .matrix (branches 3)
      | .and => .matrix (branches 4)
      | .xor => .matrix (branches 5)] := by
  obtain ⟨path⟩ := childExecutionPath_of_outcome definitionFound childMember
  have selectVerified : verifySixWaySelect workflow reference.select = true := by
    have checked := verified
    unfold verifyParallelSixWaySelect at checked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
    aesop
  let resolution := sixWaySelectResolution_of_verified selectVerified
  have boundaryVerified : verifyParallelBoundary workflow reference.parallelLoop
      reference.bodyScope
      (reference.selectorFamily :: List.ofFn reference.branchFamilies)
      (reference.bodySelector :: List.ofFn reference.bodyBranches)
      reference.bodyOutput reference.outputFamily = true := by
    have checked := verified
    unfold verifyParallelSixWaySelect at checked
    simp only [Bool.and_eq_true] at checked
    aesop
  have checked := boundaryVerified
  unfold verifyParallelBoundary at checked
  rw [bodyResolved] at checked
  cases loopResolved : resolveNode workflow reference.parallelLoop with
  | none => simp [loopResolved] at checked
  | some loopNode =>
    rw [loopResolved] at checked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
    ·
      have namesNodup : body.inputNames.Nodup := by aesop
      have namesLength : body.inputNames.length = 7 := by aesop
      have inputWires : scopeInputWires body =
          (reference.bodySelector :: List.ofFn reference.bodyBranches).map wireRef := by
        aesop
      have bodyOutputs : body.outputs.map Prod.snd = [wireRef reference.bodyOutput] := by
        change scopeOutputWires body = [wireRef reference.bodyOutput]
        aesop
      have selectorWire : reference.select.selector.wire = reference.bodySelector := by
        have wiring := verified
        unfold verifyParallelSixWaySelect at wiring
        simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
        aesop
      have branchWires : (List.ofFn reference.select.branches).map (fun branch ↦ branch.wire) =
          List.ofFn reference.bodyBranches := by
        have wiring := verified
        unfold verifyParallelSixWaySelect at wiring
        simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
        aesop
      have branchWireRefs :
          (List.ofFn reference.select.branches).map
              (wireRef ∘ CoreOperandRef.wire) =
            (List.ofFn reference.bodyBranches).map wireRef := by
        have mapped := congrArg (List.map wireRef) branchWires
        simpa only [List.map_map, Function.comp_apply] using mapped
      have outputWire : reference.select.output = reference.bodyOutput := by
        have wiring := verified
        unfold verifyParallelSixWaySelect at wiring
        simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
        aesop
      have outputOwner : reference.select.output.node = reference.select.operation := by
        exact (verifySelect_outputFacts selectVerified).1
      have outputPort : reference.select.output.port = 0 := by
        have outputVerified : verifyWire workflow reference.select.output = true := by
          exact (verifySelect_outputFacts selectVerified).2
        have portLt : reference.select.output.port < 1 := by
          simpa [verifyWire, outputOwner, resolution.resolved] using outputVerified
        omega
      have bodyOutputOwner : reference.bodyOutput.node = reference.select.operation := by
        rw [← outputWire]
        exact outputOwner
      have bodyOutputPort : reference.bodyOutput.port = 0 := by
        rw [← outputWire]
        exact outputPort
      have operationScope : reference.select.operation.scope = reference.bodyScope := by
        calc
          reference.select.operation.scope = reference.select.output.node.scope :=
            congrArg CoreNodeRef.scope outputOwner.symm
          _ = reference.bodyOutput.node.scope := by rw [outputWire]
          _ = reference.bodyScope :=
            (verifyParallelBoundary_bodyOutputLocation boundaryVerified).2
      have operationStage : reference.select.operation.stage = reference.parallelLoop.stage := by
        calc
          reference.select.operation.stage = reference.select.output.node.stage :=
            congrArg CoreNodeRef.stage outputOwner.symm
          _ = reference.bodyOutput.node.stage := by rw [outputWire]
          _ = reference.parallelLoop.stage :=
            (verifyParallelBoundary_bodyOutputLocation boundaryVerified).1
      have operationScopeResolved : resolveScope workflow reference.select.operation =
          some body := by
        have resolved := bodyResolved
        unfold resolveScope at resolved ⊢
        rw [operationScope, operationStage]
        exact resolved
      have operationInScope := resolveNode_scopeNode operationScopeResolved resolution.resolved
      obtain ⟨operationExecution, operationRooted⟩ :=
        path.rootedReferencedNodeExecution operationInScope resolution.resolved
      have operationNodeEq : operationExecution.node = {
          kind := .select
          arguments := wireRef reference.select.selector.wire ::
            (List.ofFn reference.select.branches).map
              (wireRef ∘ CoreOperandRef.wire)
          outputCount := 1
        } := Option.some.inj (operationExecution.resolved.symm.trans resolution.resolved)
      let selectorName := body.inputNames.get ⟨0, by omega⟩
      have selectorFinal : Mxx.Ir.lookupWire (wireRef reference.bodySelector)
          path.finalWires = some (.integer opcode) := by
        apply path.inputWireValue namesNodup (by simp [namesLength, inputWires])
          namesLength 0 selectorName (wireRef reference.bodySelector) (.integer opcode)
        · simp [selectorName, namesLength]
        · simp [inputWires]
        · simp
        · have selectorFacts := verifyParallelBoundary_innerFact boundaryVerified
            reference.bodySelector (by simp)
          have selectorVerified : verifyWire workflow reference.bodySelector = true :=
            selectorFacts.1
          have selectorScope : resolveScope workflow reference.bodySelector.node = some body := by
            simpa [resolveScope, selectorFacts.2.1, selectorFacts.2.2] using bodyResolved
          exact verifyWire_scopeValid selectorVerified selectorScope
      have branchFinal (i : Fin 6) :
          Mxx.Ir.lookupWire (wireRef (reference.bodyBranches i)) path.finalWires =
            some (.matrix (branches i)) := by
        let branchName := body.inputNames.get ⟨i.val + 1, by omega⟩
        apply path.inputWireValue namesNodup (by simp [namesLength, inputWires])
          namesLength (i.val + 1) branchName (wireRef (reference.bodyBranches i))
          (.matrix (branches i))
        · simp [branchName]
        · simp only [inputWires, List.map_cons, List.getElem?_cons_succ,
            List.getElem?_map, List.getElem?_ofFn]
          rw [dif_pos i.isLt]
          rfl
        · simp only [List.getElem?_cons_succ, List.getElem?_map, List.getElem?_ofFn]
          rw [dif_pos i.isLt]
          rfl
        · have branchFacts := verifyParallelBoundary_innerFact boundaryVerified
            (reference.bodyBranches i) (by
              exact List.mem_cons.mpr (Or.inr (List.mem_ofFn.mpr ⟨i, rfl⟩)))
          have branchVerified : verifyWire workflow (reference.bodyBranches i) = true :=
            branchFacts.1
          have branchScope : resolveScope workflow (reference.bodyBranches i).node =
              some body := by
            simpa [resolveScope, branchFacts.2.1, branchFacts.2.2] using bodyResolved
          exact verifyWire_scopeValid branchVerified branchScope
      have selectorBefore := operationRooted.argumentFromFinal ssaOrder
        operationScopeResolved (wireRef reference.bodySelector) (by
          simp [operationNodeEq, selectorWire]) selectorFinal
      have branchBefore (i : Fin 6) := operationRooted.argumentFromFinal ssaOrder
        operationScopeResolved (wireRef (reference.bodyBranches i)) (by
          simp only [operationNodeEq, List.mem_cons]
          right
          rw [branchWireRefs]
          exact List.mem_map_of_mem (List.mem_ofFn.mpr ⟨i, rfl⟩)) (branchFinal i)
      have argumentsEvaluate :
          (wireRef reference.select.selector.wire ::
            (List.ofFn reference.select.branches).map
              (wireRef ∘ CoreOperandRef.wire)).mapM
              (fun wire ↦ Mxx.Ir.lookupWire wire operationExecution.before) =
            some (.integer opcode :: (List.ofFn branches).map Mxx.Ir.Value.matrix) := by
        rw [selectorWire]
        rw [branchWireRefs]
        have branch0Before := branchBefore (0 : Fin 6)
        have branch1Before := branchBefore (1 : Fin 6)
        have branch2Before := branchBefore (2 : Fin 6)
        have branch3Before := branchBefore (3 : Fin 6)
        have branch4Before := branchBefore (4 : Fin 6)
        have branch5Before := branchBefore (5 : Fin 6)
        simp [selectorBefore, branch0Before, branch1Before, branch2Before, branch3Before,
          branch4Before, branch5Before]
      have operationOutcome := resolution.booleanGateOutcome operationExecution opcode opcodeLower
        opcodeUpper (branches 0) (branches 1) (branches 2) (branches 3) (branches 4)
        (branches 5) argumentsEvaluate
      let result := match MxxWe.DynamicBoolean.gateKind opcode with
        | .constantFalse => branches 0
        | .constantTrue => branches 1
        | .copy => branches 2
        | .not => branches 3
        | .and => branches 4
        | .xor => branches 5
      have outputLookup : Mxx.Ir.lookupWire (wireRef reference.bodyOutput) path.finalWires =
          some (.matrix result) := by
        have portValid : 0 < operationExecution.values.length := by
          rw [operationOutcome]
          simp
        have operationValue : operationExecution.values.get ⟨0, portValid⟩ =
            .matrix result := by simp [operationOutcome, result]
        have observed := operationRooted.outputFinal 0 portValid
        rw [operationValue] at observed
        have outputRef : wireRef reference.bodyOutput =
            ({ node := reference.select.operation.node, port := 0 } : Mxx.Ir.WireRef) := by
          have bodyNodeId : reference.bodyOutput.node.node = reference.select.operation.node :=
            congrArg (fun node : CoreNodeRef ↦ node.node) bodyOutputOwner
          show ({ node := reference.bodyOutput.node.node, port := reference.bodyOutput.port } :
            Mxx.Ir.WireRef) = _
          rw [bodyNodeId, bodyOutputPort]
        rw [outputRef]
        exact observed
      exact path.singleOutput (wireRef reference.bodyOutput) (.matrix result) bodyOutputs
        outputLookup

/-- Exact child semantics of a checked parallel two-way selector. -/
theorem parallelTwoWaySelectChildOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelTwoWaySelectRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {active : Bool} {zero candidate : Mxx.Matrix} {values : List Mxx.Ir.Value}
    (verified : verifyParallelTwoWaySelect workflow reference = true)
    (ssaOrder : verifyScopeSsaOrder body = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop with scope := reference.bodyScope } = some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.bodyScope.definitionName
      stage.program.definitions = some body)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.bodyScope.definitionName params
        [.integer (Bool.toNat active), .matrix zero, .matrix candidate]) :
    values = [.matrix (if active then candidate else zero)] := by
  obtain ⟨path⟩ := childExecutionPath_of_outcome definitionFound childMember
  have selectVerified : verifyTwoWaySelect workflow reference.select = true := by
    have checked := verified
    unfold verifyParallelTwoWaySelect at checked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
    aesop
  let resolution := twoWaySelectResolution_of_verified selectVerified
  have boundaryVerified : verifyParallelBoundary workflow reference.parallelLoop
      reference.bodyScope
      (reference.selectorFamily :: List.ofFn reference.branchFamilies)
      (reference.bodySelector :: List.ofFn reference.bodyBranches)
      reference.bodyOutput reference.outputFamily = true := by
    have checked := verified
    unfold verifyParallelTwoWaySelect at checked
    simp only [Bool.and_eq_true] at checked
    aesop
  have checked := boundaryVerified
  unfold verifyParallelBoundary at checked
  rw [bodyResolved] at checked
  cases loopResolved : resolveNode workflow reference.parallelLoop with
  | none => simp [loopResolved] at checked
  | some loopNode =>
    rw [loopResolved] at checked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
    ·
      have namesNodup : body.inputNames.Nodup := by aesop
      have namesLength : body.inputNames.length = 3 := by aesop
      have inputWires : scopeInputWires body =
          (reference.bodySelector :: List.ofFn reference.bodyBranches).map wireRef := by
        aesop
      have bodyOutputs : body.outputs.map Prod.snd = [wireRef reference.bodyOutput] := by
        change scopeOutputWires body = [wireRef reference.bodyOutput]
        aesop
      have selectorWire : reference.select.selector.wire = reference.bodySelector := by
        have wiring := verified
        unfold verifyParallelTwoWaySelect at wiring
        simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
        aesop
      have branchWires : (List.ofFn reference.select.branches).map (fun branch ↦ branch.wire) =
          List.ofFn reference.bodyBranches := by
        have wiring := verified
        unfold verifyParallelTwoWaySelect at wiring
        simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
        aesop
      have branchWireRefs :
          (List.ofFn reference.select.branches).map
              (wireRef ∘ CoreOperandRef.wire) =
            (List.ofFn reference.bodyBranches).map wireRef := by
        have mapped := congrArg (List.map wireRef) branchWires
        simpa only [List.map_map, Function.comp_apply] using mapped
      have outputWire : reference.select.output = reference.bodyOutput := by
        have wiring := verified
        unfold verifyParallelTwoWaySelect at wiring
        simp only [Bool.and_eq_true, decide_eq_true_eq] at wiring
        aesop
      have outputOwner : reference.select.output.node = reference.select.operation := by
        exact (verifySelect_outputFacts selectVerified).1
      have outputPort : reference.select.output.port = 0 := by
        have outputVerified : verifyWire workflow reference.select.output = true := by
          exact (verifySelect_outputFacts selectVerified).2
        have portLt : reference.select.output.port < 1 := by
          simpa [verifyWire, outputOwner, resolution.resolved] using outputVerified
        omega
      have bodyOutputOwner : reference.bodyOutput.node = reference.select.operation := by
        rw [← outputWire]
        exact outputOwner
      have bodyOutputPort : reference.bodyOutput.port = 0 := by
        rw [← outputWire]
        exact outputPort
      have operationScope : reference.select.operation.scope = reference.bodyScope := by
        calc
          reference.select.operation.scope = reference.select.output.node.scope :=
            congrArg CoreNodeRef.scope outputOwner.symm
          _ = reference.bodyOutput.node.scope := by rw [outputWire]
          _ = reference.bodyScope :=
            (verifyParallelBoundary_bodyOutputLocation boundaryVerified).2
      have operationStage : reference.select.operation.stage = reference.parallelLoop.stage := by
        calc
          reference.select.operation.stage = reference.select.output.node.stage :=
            congrArg CoreNodeRef.stage outputOwner.symm
          _ = reference.bodyOutput.node.stage := by rw [outputWire]
          _ = reference.parallelLoop.stage :=
            (verifyParallelBoundary_bodyOutputLocation boundaryVerified).1
      have operationScopeResolved : resolveScope workflow reference.select.operation =
          some body := by
        have resolved := bodyResolved
        unfold resolveScope at resolved ⊢
        rw [operationScope, operationStage]
        exact resolved
      have operationInScope := resolveNode_scopeNode operationScopeResolved resolution.resolved
      obtain ⟨operationExecution, operationRooted⟩ :=
        path.rootedReferencedNodeExecution operationInScope resolution.resolved
      have operationNodeEq : operationExecution.node = {
          kind := .select
          arguments := wireRef reference.select.selector.wire ::
            (List.ofFn reference.select.branches).map
              (wireRef ∘ CoreOperandRef.wire)
          outputCount := 1
        } := Option.some.inj (operationExecution.resolved.symm.trans resolution.resolved)
      have selectorFacts := verifyParallelBoundary_innerFact boundaryVerified
        reference.bodySelector (by simp)
      have selectorVerified : verifyWire workflow reference.bodySelector = true := selectorFacts.1
      have selectorScope : resolveScope workflow reference.bodySelector.node = some body := by
        simpa [resolveScope, selectorFacts.2.1, selectorFacts.2.2] using bodyResolved
      let selectorName := body.inputNames.get ⟨0, by omega⟩
      have wireLength : (scopeInputWires body).length = body.inputNames.length := by
        rw [inputWires, namesLength]
        simp
      have selectorNameAt : body.inputNames[0]? = some selectorName := by
        simp [selectorName, namesLength]
      have selectorWireAt : (scopeInputWires body)[0]? =
          some (wireRef reference.bodySelector) := by
        simp [inputWires]
      have selectorFinal := path.inputWireValue namesNodup wireLength namesLength 0 selectorName
        (wireRef reference.bodySelector) (.integer (Bool.toNat active)) selectorNameAt
        selectorWireAt (by simp) (verifyWire_scopeValid selectorVerified selectorScope)
      have branchFinal (i : Fin 2) :
          Mxx.Ir.lookupWire (wireRef (reference.bodyBranches i)) path.finalWires =
            some (.matrix ([zero, candidate][i.val]'i.isLt)) := by
        have branchFacts := verifyParallelBoundary_innerFact boundaryVerified
          (reference.bodyBranches i) (by
            fin_cases i <;> simp)
        have branchVerified : verifyWire workflow (reference.bodyBranches i) = true :=
          branchFacts.1
        have branchScope : resolveScope workflow (reference.bodyBranches i).node = some body := by
          simpa [resolveScope, branchFacts.2.1, branchFacts.2.2] using bodyResolved
        apply path.inputWireValue namesNodup wireLength
          namesLength (i.val + 1) (body.inputNames.get ⟨i.val + 1, by omega⟩)
          (wireRef (reference.bodyBranches i)) (.matrix ([zero, candidate][i.val]'i.isLt))
        · simp
        · fin_cases i <;> simp [inputWires]
        · fin_cases i <;> simp
        · exact verifyWire_scopeValid branchVerified branchScope
      have selectorBefore := operationRooted.argumentFromFinal ssaOrder
        operationScopeResolved (wireRef reference.bodySelector) (by
          simp [operationNodeEq, selectorWire]) selectorFinal
      have branchBefore (i : Fin 2) := operationRooted.argumentFromFinal ssaOrder
        operationScopeResolved (wireRef (reference.bodyBranches i)) (by
          simp only [operationNodeEq, List.mem_cons]
          right
          rw [branchWireRefs]
          exact List.mem_map_of_mem (List.mem_ofFn.mpr ⟨i, rfl⟩)) (branchFinal i)
      have argumentsEvaluate :
          (wireRef reference.select.selector.wire ::
            (List.ofFn reference.select.branches).map
              (wireRef ∘ CoreOperandRef.wire)).mapM
              (fun wire ↦ Mxx.Ir.lookupWire wire operationExecution.before) =
            some [.integer (Bool.toNat active), .matrix zero, .matrix candidate] := by
        rw [selectorWire]
        rw [branchWireRefs]
        have zeroBefore := branchBefore (0 : Fin 2)
        have candidateBefore := branchBefore (1 : Fin 2)
        simp [selectorBefore, zeroBefore, candidateBefore]
      have operationOutcome := resolution.activeGateOutcome operationExecution active zero
        candidate argumentsEvaluate
      have outputLookup : Mxx.Ir.lookupWire (wireRef reference.bodyOutput) path.finalWires =
          some (.matrix (if active then candidate else zero)) := by
        have portValid : 0 < operationExecution.values.length := by
          rw [operationOutcome]
          simp
        have operationValue : operationExecution.values.get ⟨0, portValid⟩ =
            .matrix (if active then candidate else zero) := by simp [operationOutcome]
        have observed := operationRooted.outputFinal 0 portValid
        rw [operationValue] at observed
        have outputRef : wireRef reference.bodyOutput =
            ({ node := reference.select.operation.node, port := 0 } : Mxx.Ir.WireRef) := by
          have bodyNodeId : reference.bodyOutput.node.node = reference.select.operation.node :=
            congrArg (fun node : CoreNodeRef ↦ node.node) bodyOutputOwner
          show ({ node := reference.bodyOutput.node.node, port := reference.bodyOutput.port } :
            Mxx.Ir.WireRef) = _
          rw [bodyNodeId, bodyOutputPort]
        rw [outputRef]
        exact observed
      exact path.singleOutput (wireRef reference.bodyOutput)
        (.matrix (if active then candidate else zero)) bodyOutputs outputLookup

/-- A checked zip/zip parallel multiplication returns the pointwise product family. -/
theorem ExactParallelNodeTrace.matrixMultiplyFamilyOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelMatrixBinaryRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : List Mxx.Ir.Value}
    {execution : ExactParallelNodeExecution workflow reference.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip] stage body (fuel + 1) samplers params inputs}
    (verified : verifyParallelMatrixBinary workflow reference .matrixMultiply = true)
    (ssaOrder : verifyScopeSsaOrder body = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop with scope := reference.bodyScope } = some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.bodyScope.definitionName
      stage.program.definitions = some body)
    (trace : ExactParallelNodeTrace workflow reference.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip] stage body (fuel + 1) samplers params inputs
      execution)
    (left right : Nat → Mxx.Matrix)
    (argumentsEq : trace.argumentValues =
      [.family ((List.range trace.evaluatedCount.toNat).map fun index ↦ .matrix (left index)),
        .family ((List.range trace.evaluatedCount.toNat).map fun index ↦ .matrix (right index))]) :
    execution.execution.values =
      [.family ((List.range trace.evaluatedCount.toNat).map fun index ↦
        .matrix (Mxx.matrixMultiply (left index) (right index)))] := by
  have finalEq := parallelIterationsTrace_singlePortValues_mem
    (fun index ↦ .matrix (Mxx.matrixMultiply (left index) (right index))) trace.iterations
    (initialValues := []) rfl (by
      intro index indexMember evaluatedBindings childValues bindingsEvaluate childMember
      have indexLt := List.mem_range.mp indexMember
      have bodyScopeEq : reference.bodyScope =
          reference.parallelLoop.scope.parallelBody reference.parallelLoop.node := by
        have checked := verified
        unfold verifyParallelMatrixBinary verifyParallelBoundary at checked
        simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
        aesop
      exact parallelMatrixMultiplyChildOutcome (samplers := samplers)
        (params := evaluatedBindings ++ ((.loopIndex 1, .integer index) :: params)) verified
        ssaOrder bodyResolved definitionFound
        (by simpa [argumentsEq, Mxx.Ir.loopArgument, indexLt, bodyScopeEq] using childMember))
  rw [trace.valuesEq, finalEq]
  simp

/-- A checked zip/zip parallel addition returns the pointwise sum family. -/
theorem ExactParallelNodeTrace.matrixAddFamilyOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelMatrixBinaryRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : List Mxx.Ir.Value}
    {execution : ExactParallelNodeExecution workflow reference.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip] stage body (fuel + 1) samplers params inputs}
    (verified : verifyParallelMatrixBinary workflow reference .matrixAdd = true)
    (ssaOrder : verifyScopeSsaOrder body = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop with scope := reference.bodyScope } = some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.bodyScope.definitionName
      stage.program.definitions = some body)
    (trace : ExactParallelNodeTrace workflow reference.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip] stage body (fuel + 1) samplers params inputs
      execution)
    (left right : Nat → Mxx.Matrix)
    (argumentsEq : trace.argumentValues =
      [.family ((List.range trace.evaluatedCount.toNat).map fun index ↦ .matrix (left index)),
        .family ((List.range trace.evaluatedCount.toNat).map fun index ↦ .matrix (right index))]) :
    execution.execution.values =
      [.family ((List.range trace.evaluatedCount.toNat).map fun index ↦
        .matrix (Mxx.matrixAdd (left index) (right index)))] := by
  have finalEq := parallelIterationsTrace_singlePortValues_mem
    (fun index ↦ .matrix (Mxx.matrixAdd (left index) (right index))) trace.iterations
    (initialValues := []) rfl (by
      intro index indexMember evaluatedBindings childValues bindingsEvaluate childMember
      have indexLt := List.mem_range.mp indexMember
      have bodyScopeEq : reference.bodyScope =
          reference.parallelLoop.scope.parallelBody reference.parallelLoop.node := by
        have checked := verified
        unfold verifyParallelMatrixBinary verifyParallelBoundary at checked
        simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
        aesop
      exact parallelMatrixAddChildOutcome (samplers := samplers)
        (params := evaluatedBindings ++ ((.loopIndex 1, .integer index) :: params)) verified
        ssaOrder bodyResolved definitionFound
        (by simpa [argumentsEq, Mxx.Ir.loopArgument, indexLt, bodyScopeEq] using childMember))
  rw [trace.valuesEq, finalEq]
  simp

/-- A checked zip/zip parallel subtraction returns the pointwise difference family. -/
theorem ExactParallelNodeTrace.matrixSubtractFamilyOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelMatrixBinaryRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : List Mxx.Ir.Value}
    {execution : ExactParallelNodeExecution workflow reference.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip] stage body (fuel + 1) samplers params inputs}
    (verified : verifyParallelMatrixBinary workflow reference .matrixSubtract = true)
    (ssaOrder : verifyScopeSsaOrder body = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop with scope := reference.bodyScope } = some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.bodyScope.definitionName
      stage.program.definitions = some body)
    (trace : ExactParallelNodeTrace workflow reference.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip] stage body (fuel + 1) samplers params inputs
      execution)
    (left right : Nat → Mxx.Matrix)
    (argumentsEq : trace.argumentValues =
      [.family ((List.range trace.evaluatedCount.toNat).map fun index ↦ .matrix (left index)),
        .family ((List.range trace.evaluatedCount.toNat).map fun index ↦ .matrix (right index))]) :
    execution.execution.values =
      [.family ((List.range trace.evaluatedCount.toNat).map fun index ↦
        .matrix (Mxx.matrixSubtract (left index) (right index)))] := by
  have finalEq := parallelIterationsTrace_singlePortValues_mem
    (fun index ↦ .matrix (Mxx.matrixSubtract (left index) (right index))) trace.iterations
    (initialValues := []) rfl (by
      intro index indexMember evaluatedBindings childValues bindingsEvaluate childMember
      have indexLt := List.mem_range.mp indexMember
      have bodyScopeEq : reference.bodyScope =
          reference.parallelLoop.scope.parallelBody reference.parallelLoop.node := by
        have checked := verified
        unfold verifyParallelMatrixBinary verifyParallelBoundary at checked
        simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
        aesop
      exact parallelMatrixSubtractChildOutcome (samplers := samplers)
        (params := evaluatedBindings ++ ((.loopIndex 1, .integer index) :: params)) verified
        ssaOrder bodyResolved definitionFound
        (by simpa [argumentsEq, Mxx.Ir.loopArgument, indexLt, bodyScopeEq] using childMember))
  rw [trace.valuesEq, finalEq]
  simp

/-- A checked zip/broadcast parallel family lookup returns the pointwise selected family. -/
theorem ExactParallelNodeTrace.dynamicFamilyGetOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelFamilyGetRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : List Mxx.Ir.Value}
    {execution : ExactParallelNodeExecution workflow reference.parallelLoop.operation
      (.parameter "max_layer_width") 1 [.zip, .broadcast] stage body (fuel + 1) samplers params
      inputs}
    (verified : verifyParallelFamilyGet workflow reference = true)
    (ssaOrder : verifyScopeSsaOrder body = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition
      reference.parallelLoop.bodyScope.definitionName stage.program.definitions = some body)
    (trace : ExactParallelNodeTrace workflow reference.parallelLoop.operation
      (.parameter "max_layer_width") 1 [.zip, .broadcast] stage body (fuel + 1) samplers params
      inputs execution)
    (source : List Mxx.Ir.Value) (indexAt : Nat → Nat)
    (argumentsEq : trace.argumentValues =
      [.family ((List.range trace.evaluatedCount.toNat).map fun index ↦
        .integer (Int.ofNat (indexAt index))), .family source]) :
    execution.execution.values =
      [.family ((List.range trace.evaluatedCount.toNat).map fun index ↦
        source[indexAt index]?.getD (.invalid "FamilyGetDynamic index out of range"))] := by
  have finalEq := parallelIterationsTrace_singlePortValues_mem
    (fun index ↦ source[indexAt index]?.getD
      (.invalid "FamilyGetDynamic index out of range")) trace.iterations
    (initialValues := []) rfl (by
      intro index indexMember evaluatedBindings childValues bindingsEvaluate childMember
      have indexLt := List.mem_range.mp indexMember
      have bodyScopeEq : reference.parallelLoop.bodyScope =
          reference.parallelLoop.operation.scope.parallelBody
            reference.parallelLoop.operation.node := by
        have checked := verified
        unfold verifyParallelFamilyGet verifyParallelLoop at checked
        simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
        aesop
      exact parallelFamilyGetChildOutcome (samplers := samplers)
        (params := evaluatedBindings ++ ((.loopIndex 1, .integer index) :: params)) verified
        ssaOrder bodyResolved definitionFound
        (by simpa [argumentsEq, Mxx.Ir.loopArgument, indexLt, bodyScopeEq] using childMember))
  rw [trace.valuesEq, finalEq]
  simp

/-- Recover all three encoding-component child executions directly from the certificate and an
actual decryption-layer child outcome. -/
theorem VerifiedDiamondLayout.encodingChildExecutions
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value}
    (verified : VerifiedDiamondLayout workflow certificate)
    (definitionFound : Mxx.Ir.lookupDefinition
      certificate.booleanLayers.decryption.bodyScope.definitionName
      stage.program.definitions = some scope)
    (vectorsScope : resolveScope workflow
      certificate.booleanLayers.decryptionVectors.activeSelect.parallelLoop = some scope)
    (publicKeysScope : resolveScope workflow
      certificate.booleanLayers.decryptionPublicKeys.activeSelect.parallelLoop = some scope)
    (plaintextsScope : resolveScope workflow
      certificate.booleanLayers.decryptionPlaintexts.activeSelect.parallelLoop = some scope)
    (decompositionsScope : resolveScope workflow
      certificate.booleanLayers.decryptEncodingRhsDecomposition.decompositionLoop = some scope)
    (childMember : outputs ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      certificate.booleanLayers.decryption.bodyScope.definitionName params inputs) :
    Nonempty (EncodingBooleanChildExecutions workflow certificate.booleanLayers
      stage scope fuel samplers params inputs outputs) := by
  obtain ⟨path⟩ := childExecutionPath_of_outcome definitionFound childMember
  have layers := verified.booleanLayersMatch
  unfold verifyBooleanLayers at layers
  simp only [Bool.and_eq_true] at layers
  have vectorsVerified : verifyFamilyBooleanGate workflow
      certificate.booleanLayers.decryptionVectors = true := by aesop
  have publicKeysVerified : verifyFamilyBooleanGate workflow
      certificate.booleanLayers.decryptionPublicKeys = true := by aesop
  have plaintextsVerified : verifyFamilyBooleanGate workflow
      certificate.booleanLayers.decryptionPlaintexts = true := by aesop
  have decompositionsVerified : verifyExactParallelNodeRole workflow
      certificate.booleanLayers.decryptEncodingRhsDecomposition.decompositionLoop
      (.parameter "max_layer_width") 1 [.zip] = true := by
    have decomposition := verified.booleanExecutionWiring.decryptDecomposition
    unfold verifyDecryptDecomposition at decomposition
    simp only [Bool.and_eq_true, decide_eq_true_eq] at decomposition
    aesop
  obtain ⟨vectors, vectorsRooted⟩ := path.rootedExactParallelNodeExecution
    (activeSelectRole_of_verifiedFamilyGate vectorsVerified) vectorsScope
  obtain ⟨publicKeys, publicKeysRooted⟩ := path.rootedExactParallelNodeExecution
    (activeSelectRole_of_verifiedFamilyGate publicKeysVerified) publicKeysScope
  obtain ⟨plaintexts, plaintextsRooted⟩ := path.rootedExactParallelNodeExecution
    (activeSelectRole_of_verifiedFamilyGate plaintextsVerified) plaintextsScope
  obtain ⟨decompositions, decompositionsRooted⟩ := path.rootedExactParallelNodeExecution
    decompositionsVerified decompositionsScope
  exact ⟨{
    path
    vectors
    vectorsRooted
    publicKeys
    publicKeysRooted
    plaintexts
    plaintextsRooted
    decompositions
    decompositionsRooted
  }⟩

/-- Recover every candidate-family parent loop from the same accepted encoding child. -/
theorem VerifiedDiamondLayout.completeEncodingChildExecutions
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value}
    (verified : VerifiedDiamondLayout workflow certificate)
    (definitionFound : Mxx.Ir.lookupDefinition
      certificate.booleanLayers.decryption.bodyScope.definitionName
      stage.program.definitions = some scope)
    (vectorsScope : resolveScope workflow
      certificate.booleanLayers.decryptionVectors.activeSelect.parallelLoop = some scope)
    (publicKeysScope : resolveScope workflow
      certificate.booleanLayers.decryptionPublicKeys.activeSelect.parallelLoop = some scope)
    (plaintextsScope : resolveScope workflow
      certificate.booleanLayers.decryptionPlaintexts.activeSelect.parallelLoop = some scope)
    (decompositionsScope : resolveScope workflow
      certificate.booleanLayers.decryptEncodingRhsDecomposition.decompositionLoop = some scope)
    (vectorCandidateScopes : ∀ operation ∈
      familyGateParentNodes certificate.booleanLayers.decryptionVectors,
      resolveScope workflow operation = some scope)
    (publicKeyCandidateScopes : ∀ operation ∈
      familyGateParentNodes certificate.booleanLayers.decryptionPublicKeys,
      resolveScope workflow operation = some scope)
    (plaintextCandidateScopes : ∀ operation ∈
      familyGateParentNodes certificate.booleanLayers.decryptionPlaintexts,
      resolveScope workflow operation = some scope)
    (childMember : outputs ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      certificate.booleanLayers.decryption.bodyScope.definitionName params inputs) :
    Nonempty (CompleteEncodingBooleanChildExecutions workflow certificate.booleanLayers
      stage scope fuel samplers params inputs outputs) := by
  obtain ⟨base⟩ := verified.encodingChildExecutions definitionFound vectorsScope publicKeysScope
    plaintextsScope decompositionsScope childMember
  have layers := verified.booleanLayersMatch
  unfold verifyBooleanLayers at layers
  simp only [Bool.and_eq_true] at layers
  have vectorsVerified : verifyFamilyBooleanGate workflow
      certificate.booleanLayers.decryptionVectors = true := by aesop
  have publicKeysVerified : verifyFamilyBooleanGate workflow
      certificate.booleanLayers.decryptionPublicKeys = true := by aesop
  have plaintextsVerified : verifyFamilyBooleanGate workflow
      certificate.booleanLayers.decryptionPlaintexts = true := by aesop
  obtain ⟨vectorCandidates⟩ := rootedFamilyBooleanGateExecutions_of_verified base.path
    certificate.booleanLayers.decryptionVectors vectorsVerified vectorCandidateScopes
  obtain ⟨publicKeyCandidates⟩ := rootedFamilyBooleanGateExecutions_of_verified base.path
    certificate.booleanLayers.decryptionPublicKeys publicKeysVerified publicKeyCandidateScopes
  obtain ⟨plaintextCandidates⟩ := rootedFamilyBooleanGateExecutions_of_verified base.path
    certificate.booleanLayers.decryptionPlaintexts plaintextsVerified plaintextCandidateScopes
  exact ⟨{ base, vectorCandidates, publicKeyCandidates, plaintextCandidates }⟩

/-- The recovered parent-loop execution selects the interpreter's exact parallel trace whenever
its executable argument and count expressions evaluate. -/
theorem LocalBooleanParentLoopExecution.parallelTrace
    {workflow : Mxx.Ir.Workflow} {layout : LocalBooleanGateLayout}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value}
    (execution : LocalBooleanParentLoopExecution workflow layout stage scope fuel samplers params
      inputs outputs)
    (argumentValues : List Mxx.Ir.Value) (evaluatedCount : Int)
    (argumentsEvaluate :
      (layout.parentLoop.arguments.map (wireRef ∘ CoreOperandRef.wire)).mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.loop.before) = some argumentValues)
    (countEvaluate : layout.parentLoop.count.evaluate params = some evaluatedCount) :
    Nonempty (CertifiedParallelLoopTrace workflow layout.parentLoop
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs) execution.loop) :=
  certifiedParallelLoopTrace_of_resolution execution.resolution execution.loop argumentValues
    evaluatedCount argumentsEvaluate countEvaluate

/-- Every selected iteration of an arbitrary-length Boolean layer scan contains the exact
certificate-checked public-key parent loop.  This quantifies over all runtime layer indices and
retains each iteration's dynamic parameter and input environments. -/
theorem Mxx.Ir.SequentialIterationsTrace.everyLocalBooleanParentLoop
    {workflow : Mxx.Ir.Workflow} {layout : LocalBooleanGateLayout}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {indexSlot : Nat} {bindings : List (String × Mxx.Ir.IntExpr)}
    {invariants : List Mxx.Ir.Value} {indices : List Nat}
    {initial final : List Mxx.Ir.Value}
    (verified : verifyLocalBooleanGate workflow layout = true)
    (definitionFound : Mxx.Ir.lookupDefinition layout.bodyScope.definitionName
      stage.program.definitions = some scope)
    (scopeResolved : resolveScope workflow layout.parentLoop.operation = some scope)
    (trace : Mxx.Ir.SequentialIterationsTrace
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1))
      layout.bodyScope.definitionName params indexSlot bindings invariants indices initial final) :
    ∀ index ∈ indices, ∃ childParams childInputs childOutputs,
      childOutputs ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
        layout.bodyScope.definitionName childParams childInputs ∧
      Nonempty (LocalBooleanParentLoopExecution workflow layout stage scope fuel samplers
        childParams childInputs childOutputs) := by
  apply trace.everyChild (fun _ childParams childInputs childOutputs ↦
    Nonempty (LocalBooleanParentLoopExecution workflow layout stage scope fuel samplers
      childParams childInputs childOutputs))
  intro index evaluatedBindings state next bindingsEvaluate childMember
  exact localBooleanParentLoopExecution_of_childOutcome verified definitionFound scopeResolved
    childMember

/-- Every runtime iteration of the accepted public-key layer scan contains the certificate's
exact local Boolean gate execution. -/
theorem Mxx.Ir.SequentialIterationsTrace.everyCertifiedPublicKeyChild
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {indexSlot : Nat} {bindings : List (String × Mxx.Ir.IntExpr)}
    {invariants : List Mxx.Ir.Value} {indices : List Nat}
    {initial final : List Mxx.Ir.Value}
    (verified : VerifiedDiamondLayout workflow certificate)
    (definitionFound : Mxx.Ir.lookupDefinition
      certificate.booleanLayers.encryption.bodyScope.definitionName
      stage.program.definitions = some scope)
    (scopeResolved : resolveScope workflow
      certificate.booleanLayers.encryptionGate.parentLoop.operation = some scope)
    (trace : Mxx.Ir.SequentialIterationsTrace
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1))
      certificate.booleanLayers.encryption.bodyScope.definitionName params indexSlot bindings
      invariants indices initial final) :
    ∀ index ∈ indices, ∃ childParams childInputs childOutputs,
      childOutputs ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
        certificate.booleanLayers.encryption.bodyScope.definitionName childParams childInputs ∧
      Nonempty (LocalBooleanParentLoopExecution workflow
        certificate.booleanLayers.encryptionGate stage scope fuel samplers
        childParams childInputs childOutputs) := by
  apply trace.everyChild (fun _ childParams childInputs childOutputs ↦
    Nonempty (LocalBooleanParentLoopExecution workflow
      certificate.booleanLayers.encryptionGate stage scope fuel samplers
      childParams childInputs childOutputs))
  intro index evaluatedBindings state next bindingsEvaluate childMember
  exact verified.publicKeyChildExecution definitionFound scopeResolved childMember

/-- Every runtime iteration of the accepted encoding scan executes all three component selectors
from the same retained child path. -/
theorem Mxx.Ir.SequentialIterationsTrace.everyCertifiedEncodingChild
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {indexSlot : Nat} {bindings : List (String × Mxx.Ir.IntExpr)}
    {invariants : List Mxx.Ir.Value} {indices : List Nat}
    {initial final : List Mxx.Ir.Value}
    (verified : VerifiedDiamondLayout workflow certificate)
    (definitionFound : Mxx.Ir.lookupDefinition
      certificate.booleanLayers.decryption.bodyScope.definitionName
      stage.program.definitions = some scope)
    (vectorsScope : resolveScope workflow
      certificate.booleanLayers.decryptionVectors.activeSelect.parallelLoop = some scope)
    (publicKeysScope : resolveScope workflow
      certificate.booleanLayers.decryptionPublicKeys.activeSelect.parallelLoop = some scope)
    (plaintextsScope : resolveScope workflow
      certificate.booleanLayers.decryptionPlaintexts.activeSelect.parallelLoop = some scope)
    (decompositionsScope : resolveScope workflow
      certificate.booleanLayers.decryptEncodingRhsDecomposition.decompositionLoop = some scope)
    (trace : Mxx.Ir.SequentialIterationsTrace
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1))
      certificate.booleanLayers.decryption.bodyScope.definitionName params indexSlot bindings
      invariants indices initial final) :
    ∀ index ∈ indices, ∃ childParams childInputs childOutputs,
      childOutputs ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
        certificate.booleanLayers.decryption.bodyScope.definitionName childParams childInputs ∧
      Nonempty (EncodingBooleanChildExecutions workflow certificate.booleanLayers
        stage scope fuel samplers childParams childInputs childOutputs) := by
  apply trace.everyChild (fun _ childParams childInputs childOutputs ↦
    Nonempty (EncodingBooleanChildExecutions workflow certificate.booleanLayers
      stage scope fuel samplers childParams childInputs childOutputs))
  intro index evaluatedBindings state next bindingsEvaluate childMember
  exact verified.encodingChildExecutions definitionFound vectorsScope publicKeysScope
    plaintextsScope decompositionsScope childMember

/-- Every runtime iteration of the accepted encoding scan retains all parent-loop executions
needed to reconstruct the three concrete Boolean-encoding components.  Unlike
`everyCertifiedEncodingChild`, this includes the complete six-candidate data for each component,
so a later semantic bridge does not need to ask its caller for any operation equation. -/
theorem Mxx.Ir.SequentialIterationsTrace.everyCompleteCertifiedEncodingChild
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {indexSlot : Nat} {bindings : List (String × Mxx.Ir.IntExpr)}
    {invariants : List Mxx.Ir.Value} {indices : List Nat}
    {initial final : List Mxx.Ir.Value}
    (verified : VerifiedDiamondLayout workflow certificate)
    (definitionFound : Mxx.Ir.lookupDefinition
      certificate.booleanLayers.decryption.bodyScope.definitionName
      stage.program.definitions = some scope)
    (vectorsScope : resolveScope workflow
      certificate.booleanLayers.decryptionVectors.activeSelect.parallelLoop = some scope)
    (publicKeysScope : resolveScope workflow
      certificate.booleanLayers.decryptionPublicKeys.activeSelect.parallelLoop = some scope)
    (plaintextsScope : resolveScope workflow
      certificate.booleanLayers.decryptionPlaintexts.activeSelect.parallelLoop = some scope)
    (decompositionsScope : resolveScope workflow
      certificate.booleanLayers.decryptEncodingRhsDecomposition.decompositionLoop = some scope)
    (vectorCandidateScopes : ∀ operation ∈
      familyGateParentNodes certificate.booleanLayers.decryptionVectors,
      resolveScope workflow operation = some scope)
    (publicKeyCandidateScopes : ∀ operation ∈
      familyGateParentNodes certificate.booleanLayers.decryptionPublicKeys,
      resolveScope workflow operation = some scope)
    (plaintextCandidateScopes : ∀ operation ∈
      familyGateParentNodes certificate.booleanLayers.decryptionPlaintexts,
      resolveScope workflow operation = some scope)
    (trace : Mxx.Ir.SequentialIterationsTrace
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1))
      certificate.booleanLayers.decryption.bodyScope.definitionName params indexSlot bindings
      invariants indices initial final) :
    ∀ index ∈ indices, ∃ childParams childInputs childOutputs,
      childOutputs ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
        certificate.booleanLayers.decryption.bodyScope.definitionName childParams childInputs ∧
      Nonempty (CompleteEncodingBooleanChildExecutions workflow certificate.booleanLayers
        stage scope fuel samplers childParams childInputs childOutputs) := by
  apply trace.everyChild (fun _ childParams childInputs childOutputs ↦
    Nonempty (CompleteEncodingBooleanChildExecutions workflow certificate.booleanLayers
      stage scope fuel samplers childParams childInputs childOutputs))
  intro index evaluatedBindings state next bindingsEvaluate childMember
  exact verified.completeEncodingChildExecutions definitionFound vectorsScope publicKeysScope
    plaintextsScope decompositionsScope vectorCandidateScopes publicKeyCandidateScopes
    plaintextCandidateScopes childMember

/-- Lift an exact arbitrary-length public-key loop trace to the reusable algebraic recurrence.
The only protocol-specific premise is the step refinement discharged from the checked body path. -/
private theorem publicKeyLayersEvaluation_of_sequentialTrace
    {R : Type} [CommRing R] {columns maxWidth : Nat}
    {runChild : Mxx.Ir.ChildRunner} {definition : String}
    {params : Mxx.Ir.ParamEnvironment} {indexSlot : Nat}
    {bindings : List (String × Mxx.Ir.IntExpr)} {invariants : List Mxx.Ir.Value}
    {indices : List Nat} {initialValues finalValues : List Mxx.Ir.Value}
    (one : MxxWe.AlgebraMatrix R 1 columns)
    (layerAt : Nat → MxxWe.BooleanLayerProgram)
    (represents : List Mxx.Ir.Value → List (MxxWe.AlgebraMatrix R 1 columns) → Prop)
    (trace : Mxx.Ir.SequentialIterationsTrace runChild definition params indexSlot bindings
      invariants indices initialValues finalValues)
    (stepSemantics : ∀ (index : Nat) evaluatedBindings state next,
      Mxx.Ir.evaluateBindings
          ((.loopIndex indexSlot, .integer (Int.ofNat index)) :: params) bindings =
          some evaluatedBindings →
      next ∈ runChild definition
        (evaluatedBindings ++
          ((.loopIndex indexSlot, .integer (Int.ofNat index)) :: params))
        (state ++ invariants) →
      ∀ previous, represents state previous →
        ∃ valid : (layerAt index).Valid previous.length maxWidth,
          ∃ rightDecompositions : Fin (layerAt index).activeWidth →
              MxxWe.AlgebraMatrix R columns columns,
            represents next
              (MxxWe.evaluateBooleanPublicKeyLayer one previous (layerAt index) valid
                rightDecompositions))
    (initial : List (MxxWe.AlgebraMatrix R 1 columns))
    (initialRepresents : represents initialValues initial) :
    ∃ final,
      represents finalValues final ∧
      MxxWe.BooleanPublicKeyLayersEvaluation columns maxWidth one (indices.map layerAt)
        initial final := by
  induction trace generalizing initial with
  | nil => exact ⟨initial, initialRepresents, .nil initial⟩
  | cons index tail state evaluatedBindings next final bindingsEvaluate childMember rest ih =>
      obtain ⟨valid, rightDecompositions, nextRepresents⟩ :=
        stepSemantics index evaluatedBindings state next bindingsEvaluate childMember initial
          initialRepresents
      let middle := MxxWe.evaluateBooleanPublicKeyLayer one initial (layerAt index) valid
        rightDecompositions
      obtain ⟨last, lastRepresents, restEvaluation⟩ := ih middle nextRepresents
      exact ⟨last, lastRepresents,
        .cons (layerAt index) (tail.map layerAt) initial middle last valid rightDecompositions
          rfl restEvaluation⟩

/-- Lift an exact arbitrary-length encoding loop trace to the reusable three-component Boolean
recurrence. -/
private theorem booleanLayersEvaluation_of_sequentialTrace
    {R : Type} [CommRing R] {columns maxWidth : Nat}
    {runChild : Mxx.Ir.ChildRunner} {definition : String}
    {params : Mxx.Ir.ParamEnvironment} {indexSlot : Nat}
    {bindings : List (String × Mxx.Ir.IntExpr)} {invariants : List Mxx.Ir.Value}
    {indices : List Nat} {initialValues finalValues : List Mxx.Ir.Value}
    (one : MxxWe.BooleanEncoding R columns)
    (layerAt : Nat → MxxWe.BooleanLayerProgram)
    (represents : List Mxx.Ir.Value → List (MxxWe.BooleanEncoding R columns) → Prop)
    (trace : Mxx.Ir.SequentialIterationsTrace runChild definition params indexSlot bindings
      invariants indices initialValues finalValues)
    (stepSemantics : ∀ (index : Nat) evaluatedBindings state next,
      Mxx.Ir.evaluateBindings
          ((.loopIndex indexSlot, .integer (Int.ofNat index)) :: params) bindings =
          some evaluatedBindings →
      next ∈ runChild definition
        (evaluatedBindings ++
          ((.loopIndex indexSlot, .integer (Int.ofNat index)) :: params))
        (state ++ invariants) →
      ∀ previous, represents state previous →
        ∃ valid : (layerAt index).Valid previous.length maxWidth,
          ∃ rightDecompositions : Fin (layerAt index).activeWidth →
              MxxWe.AlgebraMatrix R columns columns,
            represents next
              (MxxWe.evaluateBooleanLayer one previous (layerAt index) valid rightDecompositions))
    (initial : List (MxxWe.BooleanEncoding R columns))
    (initialRepresents : represents initialValues initial) :
    ∃ final,
      represents finalValues final ∧
      MxxWe.BooleanLayersEvaluation columns maxWidth one (indices.map layerAt) initial final := by
  induction trace generalizing initial with
  | nil => exact ⟨initial, initialRepresents, .nil initial⟩
  | cons index tail state evaluatedBindings next final bindingsEvaluate childMember rest ih =>
      obtain ⟨valid, rightDecompositions, nextRepresents⟩ :=
        stepSemantics index evaluatedBindings state next bindingsEvaluate childMember initial
          initialRepresents
      let middle := MxxWe.evaluateBooleanLayer one initial (layerAt index) valid rightDecompositions
      obtain ⟨last, lastRepresents, restEvaluation⟩ := ih middle nextRepresents
      exact ⟨last, lastRepresents,
        .cons (layerAt index) (tail.map layerAt) initial middle last valid rightDecompositions
          rfl restEvaluation⟩

end MxxWe.Certificate
