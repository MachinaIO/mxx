import Mxx.Certificate.FrozenDependencySlice
import Mxx.Certificate.Rules.NestedSequentialTrace

namespace Mxx.Certificate

/-!
# Trace-indexed pointwise formula execution

These types retain the runtime instance in which a provenance-preserving pointwise formula is
interpreted.  In particular, a nested scope carries its actual parameter environment and a
parallel child carries its actual coordinate.  No callback can replace a runner, wire value, or
child execution.
-/

/-- One concrete execution of a named scope, including its static identity. -/
structure ExecutedScope
    (samplers : Mxx.MxxSamplerFamily)
    (program : Mxx.Ir.Prog) where
  scopeId : StaticScopeId
  fuel : Nat
  definition : String
  params : Mxx.Ir.ParamEnvironment
  arguments : List Mxx.Ir.Value
  outputs : List Mxx.Ir.Value
  execution : ChildScopeExecutionPath samplers program fuel definition params arguments outputs

/-- An exact subgraph-call edge from one executed scope to the child selected by that call.
The child support member is the selected SSA node's output list, not an independently supplied
execution. -/
structure ExactSubgraphExecutionEdge
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (parent : ExecutedScope samplers program) where
  child : ExecutedScope samplers program
  nodeIndex : Nat
  nodeInBounds : nodeIndex < parent.execution.scope.nodes.length
  definition : String
  bindings : List (String × Mxx.Ir.IntExpr)
  argumentRefs : List Mxx.Ir.WireRef
  argumentValues : List Mxx.Ir.Value
  evaluatedBindings : Mxx.Ir.ParamEnvironment
  nodeEq : parent.execution.scope.nodes[nodeIndex] = {
    kind := .subgraphCall definition bindings
    arguments := argumentRefs
    outputCount := parent.execution.scope.nodes[nodeIndex].outputCount
    outputTypes := parent.execution.scope.nodes[nodeIndex].outputTypes
  }
  argumentsEvaluate : argumentRefs.mapM (fun wire ↦ Mxx.Ir.lookupWire wire
    (parent.execution.nodeExecutionAt nodeIndex nodeInBounds).before) = some argumentValues
  bindingsEvaluate : Mxx.Ir.evaluateBindings parent.params bindings = some evaluatedBindings
  fuelEq : parent.fuel = child.fuel + 1
  definitionEq : child.definition = definition
  paramsEq : child.params = evaluatedBindings ++ parent.params
  argumentsEq : child.arguments = argumentValues
  outputsEq : child.outputs =
    (parent.execution.nodeExecutionAt nodeIndex nodeInBounds).nodeValues

/-- An exact coordinate of one parallel-loop execution and the child support member selected at
that coordinate. -/
structure ExactParallelLaneExecutionEdge
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (parent : ExecutedScope samplers program) where
  nodeIndex : Nat
  nodeInBounds : nodeIndex < parent.execution.scope.nodes.length
  view : ParallelLoopNodeView parent.execution.scope.nodes[nodeIndex]
  trace : NestedParallelTrace parent.execution nodeIndex nodeInBounds view
  position : Nat
  positionInBounds : position < trace.evaluatedCount.toNat
  evaluatedBindings : Mxx.Ir.ParamEnvironment
  childValues : List Mxx.Ir.Value
  bindingsEvaluate : Mxx.Ir.evaluateBindings
      ((.loopIndex view.indexSlot, .integer position) :: parent.params) view.bindings =
    some evaluatedBindings
  childMember : childValues ∈ Mxx.Ir.childRunnerWithFuel samplers program parent.fuel
    view.definition
    (evaluatedBindings ++ ((.loopIndex view.indexSlot, .integer position) :: parent.params))
    ((view.modes.zip trace.argumentValues).map fun (mode, value) ↦
      Mxx.Ir.loopArgument mode position value)
  child : ExecutedScope samplers program
  fuelEq : parent.fuel = child.fuel + 1
  definitionEq : child.definition = view.definition
  paramsEq : child.params =
    evaluatedBindings ++ ((.loopIndex view.indexSlot, .integer position) :: parent.params)
  argumentsEq : child.arguments =
    ((view.modes.zip trace.argumentValues).map fun (mode, value) ↦
      Mxx.Ir.loopArgument mode position value)
  outputsEq : child.outputs = childValues

/-- Build a subgraph edge only by inverting the selected parent SSA node. -/
theorem ExactSubgraphExecutionEdge.nonempty
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (parent : ExecutedScope samplers program)
    {fuel : Nat}
    (fuelEq : parent.fuel = fuel + 1)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < parent.execution.scope.nodes.length)
    (definition : String)
    (bindings : List (String × Mxx.Ir.IntExpr))
    (argumentRefs : List Mxx.Ir.WireRef)
    (argumentValues : List Mxx.Ir.Value)
    (evaluatedBindings : Mxx.Ir.ParamEnvironment)
    (body : Mxx.Ir.Scope)
    (nodeEq : parent.execution.scope.nodes[nodeIndex] = {
      kind := .subgraphCall definition bindings
      arguments := argumentRefs
      outputCount := parent.execution.scope.nodes[nodeIndex].outputCount
      outputTypes := parent.execution.scope.nodes[nodeIndex].outputTypes
    })
    (argumentsEvaluate : argumentRefs.mapM (fun wire ↦ Mxx.Ir.lookupWire wire
      (parent.execution.nodeExecutionAt nodeIndex nodeInBounds).before) = some argumentValues)
    (bindingsEvaluate : Mxx.Ir.evaluateBindings parent.params bindings = some evaluatedBindings)
    (bodyFound : Mxx.Ir.lookupDefinition definition program.definitions = some body) :
    Nonempty (ExactSubgraphExecutionEdge parent) := by
  obtain ⟨scopeId, parentFuel, parentDefinition, parentParams, parentArguments, parentOutputs,
      parentExecution⟩ := parent
  dsimp at fuelEq
  subst parentFuel
  obtain ⟨childExecution⟩ := parentExecution.subgraphChildExecutionExists nodeIndex nodeInBounds
    definition bindings argumentRefs argumentValues evaluatedBindings body nodeEq argumentsEvaluate
    bindingsEvaluate bodyFound
  let child : ExecutedScope samplers program := {
    scopeId := ⟨scopeId.path ++ [definition]⟩
    fuel
    definition
    params := evaluatedBindings ++ parentParams
    arguments := argumentValues
    outputs := (parentExecution.nodeExecutionAt nodeIndex nodeInBounds).nodeValues
    execution := childExecution
  }
  exact ⟨{
    child
    nodeIndex
    nodeInBounds
    definition
    bindings
    argumentRefs
    argumentValues
    evaluatedBindings
    nodeEq
    argumentsEvaluate
    bindingsEvaluate
    fuelEq := rfl
    definitionEq := rfl
    paramsEq := rfl
    argumentsEq := rfl
    outputsEq := rfl
  }⟩

/-- Build one parallel-lane edge from the exact coordinate stored by an actual parallel trace. -/
theorem ExactParallelLaneExecutionEdge.nonempty
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (parent : ExecutedScope samplers program)
    {fuel : Nat}
    (fuelEq : parent.fuel = fuel + 1)
    {nodeIndex : Nat}
    {nodeInBounds : nodeIndex < parent.execution.scope.nodes.length}
    {view : ParallelLoopNodeView parent.execution.scope.nodes[nodeIndex]}
    (trace : NestedParallelTrace parent.execution nodeIndex nodeInBounds view)
    (position : Nat)
    (positionInBounds : position < trace.evaluatedCount.toNat)
    (body : Mxx.Ir.Scope)
    (bodyFound : Mxx.Ir.lookupDefinition view.definition program.definitions = some body) :
    Nonempty (ExactParallelLaneExecutionEdge parent) := by
  have rangeMember : position ∈ List.range trace.evaluatedCount.toNat := by
    simpa using positionInBounds
  obtain ⟨childParams, childInputs, childOutputs, childMember,
      ⟨evaluatedBindings, bindingsEvaluate, paramsEq⟩, inputsEq⟩ :=
    trace.executionTrace.everyChild
    (fun index params inputs _ ↦
      (∃ evaluatedBindings,
        Mxx.Ir.evaluateBindings
            ((.loopIndex view.indexSlot, .integer index) :: parent.params) view.bindings =
          some evaluatedBindings ∧
        params = evaluatedBindings ++
          ((.loopIndex view.indexSlot, .integer index) :: parent.params)) ∧
      inputs = ((view.modes.zip trace.argumentValues).map fun (mode, value) ↦
        Mxx.Ir.loopArgument mode index value))
    (by
      intro index evaluatedBindings childValues evaluated bindingsMember
      constructor
      · exact ⟨evaluatedBindings, evaluated, rfl⟩
      · rfl)
    position rangeMember
  subst childParams
  subst childInputs
  obtain ⟨scopeId, parentFuel, parentDefinition, parentParams, parentArguments, parentOutputs,
      parentExecution⟩ := parent
  dsimp at fuelEq
  subst parentFuel
  obtain ⟨childExecution⟩ := ChildScopeExecutionPath.nonempty_of_childMember samplers program
    fuel view.definition body
      (evaluatedBindings ++ ((.loopIndex view.indexSlot, .integer position) :: parentParams))
      ((view.modes.zip trace.argumentValues).map fun (mode, value) ↦
        Mxx.Ir.loopArgument mode position value)
      childOutputs bodyFound childMember
  let child : ExecutedScope samplers program := {
    scopeId := ⟨scopeId.path ++ [view.definition]⟩
    fuel
    definition := view.definition
    params := evaluatedBindings ++
      ((.loopIndex view.indexSlot, .integer position) :: parentParams)
    arguments := (view.modes.zip trace.argumentValues).map fun (mode, value) ↦
      Mxx.Ir.loopArgument mode position value
    outputs := childOutputs
    execution := childExecution
  }
  exact ⟨{
    nodeIndex
    nodeInBounds
    view
    trace
    position
    positionInBounds
    evaluatedBindings
    childValues := childOutputs
    bindingsEvaluate
    childMember
    child
    fuelEq := rfl
    definitionEq := rfl
    paramsEq := rfl
    argumentsEq := rfl
    outputsEq := rfl
  }⟩

/-- A path of actual runtime scope instances.  Moving into a child retains the exact parent edge,
so formal-input substitution can move back to the unique calling instance. -/
inductive FormulaExecutionFrame
    (samplers : Mxx.MxxSamplerFamily)
    (program : Mxx.Ir.Prog) : ExecutedScope samplers program → Type where
  | root (current : ExecutedScope samplers program) : FormulaExecutionFrame samplers program current
  | subgraph
      {parentScope : ExecutedScope samplers program}
      (parent : FormulaExecutionFrame samplers program parentScope)
      (edge : ExactSubgraphExecutionEdge parentScope) :
      FormulaExecutionFrame samplers program edge.child
  | parallelLane
      {parentScope : ExecutedScope samplers program}
      (parent : FormulaExecutionFrame samplers program parentScope)
      (edge : ExactParallelLaneExecutionEdge parentScope) :
      FormulaExecutionFrame samplers program edge.child

/-- A matrix value selected solely from actual trace evidence.  Arithmetic constructors are
added by the semantic elaborator; this foundation handles leaves and execution boundaries. -/
inductive FrozenPointwiseMatrixProgramFormula.PointwiseRuntimeResult
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog} :
    {current : ExecutedScope samplers program} → FormulaExecutionFrame samplers program current →
      FrozenPointwiseMatrixProgramFormula → Mxx.Matrix → Prop where
  | atom
      {current : ExecutedScope samplers program}
      (frame : FormulaExecutionFrame samplers program current)
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (matrix : Mxx.Matrix)
      (scopeEq : scope = current.scopeId)
      (found : Mxx.Ir.lookupWire wire current.execution.wires = some (.matrix matrix)) :
      PointwiseRuntimeResult frame (.atom scope wire) matrix
  | gatheredAtom
      {current : ExecutedScope samplers program}
      (frame : FormulaExecutionFrame samplers program current)
      {nodeIndex : Nat}
      {nodeInBounds : nodeIndex < current.execution.scope.nodes.length}
      {view : ParallelLoopNodeView current.execution.scope.nodes[nodeIndex]}
      (trace : NestedParallelTrace current.execution nodeIndex nodeInBounds view)
      (scope : StaticScopeId)
      (port position : Nat)
      (matrix : Mxx.Matrix)
      (scopeEq : scope = current.scopeId)
      (portInBounds : port < view.outputCount)
      (gathered : List Mxx.Ir.Value)
      (finalPort : trace.final[port]? = some gathered)
      (elementFound : gathered[position]? = some (.matrix matrix)) :
      PointwiseRuntimeResult frame (.atom scope ⟨nodeIndex, port⟩) matrix
  | inputSubstitutionSubgraph
      {parentScope : ExecutedScope samplers program}
      (parent : FormulaExecutionFrame samplers program parentScope)
      (edge : ExactSubgraphExecutionEdge parentScope)
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (slot : Nat)
      (value : FrozenPointwiseMatrixProgramFormula)
      (matrix : Mxx.Matrix)
      (parentResult : PointwiseRuntimeResult parent value matrix)
      (scopeEq : scope = edge.child.scopeId)
      (name : String)
      (inputNameFound : edge.child.execution.scope.inputNames[slot]? = some name)
      (argumentFound : edge.child.arguments[slot]? = some (.matrix matrix))
      (formalInputFound : Mxx.Ir.lookupEnvironment name
        (edge.child.execution.scope.inputNames.zip edge.child.arguments) =
          some (.matrix matrix))
      (nodeInBounds : wire.node < edge.child.execution.scope.nodes.length)
      (outputCount : Nat)
      (nodeEq : edge.child.execution.scope.nodes[wire.node] = {
        kind := .input name
        arguments := []
        outputCount
        outputTypes := edge.child.execution.scope.nodes[wire.node].outputTypes
      })
      (portInBounds : wire.port < outputCount) :
      PointwiseRuntimeResult (.subgraph parent edge)
        (.inputSubstitution scope wire slot value) matrix
  | inputSubstitutionParallel
      {parentScope : ExecutedScope samplers program}
      (parent : FormulaExecutionFrame samplers program parentScope)
      (edge : ExactParallelLaneExecutionEdge parentScope)
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (slot : Nat)
      (value : FrozenPointwiseMatrixProgramFormula)
      (matrix : Mxx.Matrix)
      (parentResult : PointwiseRuntimeResult parent value matrix)
      (scopeEq : scope = edge.child.scopeId)
      (name : String)
      (inputNameFound : edge.child.execution.scope.inputNames[slot]? = some name)
      (argumentFound : edge.child.arguments[slot]? = some (.matrix matrix))
      (formalInputFound : Mxx.Ir.lookupEnvironment name
        (edge.child.execution.scope.inputNames.zip edge.child.arguments) =
          some (.matrix matrix))
      (nodeInBounds : wire.node < edge.child.execution.scope.nodes.length)
      (outputCount : Nat)
      (nodeEq : edge.child.execution.scope.nodes[wire.node] = {
        kind := .input name
        arguments := []
        outputCount
        outputTypes := edge.child.execution.scope.nodes[wire.node].outputTypes
      })
      (portInBounds : wire.port < outputCount) :
      PointwiseRuntimeResult (.parallelLane parent edge)
        (.inputSubstitution scope wire slot value) matrix
  | subgraphCall
      {parentScope : ExecutedScope samplers program}
      (parent : FormulaExecutionFrame samplers program parentScope)
      (edge : ExactSubgraphExecutionEdge parentScope)
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (definition : String)
      (outputPort : Nat)
      (arguments : List FrozenPointwiseMatrixProgramFormula)
      (output : FrozenPointwiseMatrixProgramFormula)
      (matrix : Mxx.Matrix)
      (sourceEq : (scope, wire) = (parentScope.scopeId, ⟨edge.nodeIndex, outputPort⟩))
      (definitionEq : definition = edge.definition)
      (childResult : PointwiseRuntimeResult (.subgraph parent edge) output matrix)
      (outputName : String)
      (outputWire : Mxx.Ir.WireRef)
      (outputEntry : edge.child.execution.scope.outputs[outputPort]? =
        some (outputName, outputWire))
      (childOutputFound : Mxx.Ir.lookupWire outputWire edge.child.execution.wires =
        some (.matrix matrix)) :
      PointwiseRuntimeResult parent
        (.subgraphCall scope wire definition outputPort arguments output) matrix
  | parallelLoopElement
      {parentScope : ExecutedScope samplers program}
      (parent : FormulaExecutionFrame samplers program parentScope)
      (edge : ExactParallelLaneExecutionEdge parentScope)
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (definition : String)
      (outputPort : Nat)
      (arguments : List FrozenPointwiseMatrixProgramFormula)
      (output : FrozenPointwiseMatrixProgramFormula)
      (matrix : Mxx.Matrix)
      (sourceEq : (scope, wire) =
        (parentScope.scopeId, ⟨edge.nodeIndex, outputPort⟩))
      (definitionEq : definition = edge.view.definition)
      (childResult : PointwiseRuntimeResult (.parallelLane parent edge) output matrix)
      (portInBounds : outputPort < edge.view.outputCount)
      (childArity : ∀ (index : Nat) evaluatedBindings childValues,
        Mxx.Ir.evaluateBindings
            ((.loopIndex edge.view.indexSlot, .integer index) :: parentScope.params)
            edge.view.bindings = some evaluatedBindings →
        childValues ∈ Mxx.Ir.childRunnerWithFuel samplers program parentScope.fuel
          edge.view.definition
          (evaluatedBindings ++
            ((.loopIndex edge.view.indexSlot, .integer index) :: parentScope.params))
          ((edge.view.modes.zip edge.trace.argumentValues).map fun (mode, value) ↦
            Mxx.Ir.loopArgument mode index value) →
        childValues.length = edge.view.outputCount)
      (gathered : List Mxx.Ir.Value)
      (finalPort : edge.trace.final[outputPort]? = some gathered)
      (outputName : String)
      (outputWire : Mxx.Ir.WireRef)
      (outputEntry : edge.child.execution.scope.outputs[outputPort]? =
        some (outputName, outputWire))
      (childOutputFound : Mxx.Ir.lookupWire outputWire edge.child.execution.wires =
        some (.matrix matrix))
      (coordinateOutput : edge.child.outputs[outputPort]? = gathered[edge.position]?) :
      PointwiseRuntimeResult parent
        (.parallelLoop scope wire definition outputPort arguments output) matrix

end Mxx.Certificate
