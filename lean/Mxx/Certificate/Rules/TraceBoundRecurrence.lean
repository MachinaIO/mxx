import Mxx.Certificate.Execution
import Mxx.Certificate.Analyzer
import Mxx.Certificate.Rules.Family
import Mxx.Certificate.Rules.CarriedFactSubstitution
import Mxx.Certificate.Rules.LoopRecurrence
import Mxx.Certificate.SymbolicRecurrence

namespace Mxx.Certificate

/-!
# Trace-bound sequential recurrence evidence

This module connects a sequential recurrence to one node in an actual stage execution.  It does
not accept a child runner, a child definition, invariant values, or a preservation callback.  The
runner is definitionally the runner used by the root-stage evaluator, while the definition and
invariant values are projections of the selected executable node and its evaluated arguments.
-/

/-- The sequential-loop fields obtained from one exact executable node.  `nodeEq` prevents the
fields from naming a different loop than `node`; in particular, `definition` is not a recurrence
or certificate identity. -/
structure RootSequentialLoopNodeView (node : Mxx.Ir.Node) where
  definition : String
  count : Mxx.Ir.IntExpr
  indexSlot : Nat
  bindings : List (String × Mxx.Ir.IntExpr)
  carriedCount : Nat
  argumentRefs : List Mxx.Ir.WireRef
  outputCount : Nat
  outputTypes : List Mxx.Ir.WireTypeExpr
  nodeEq : node = {
    kind := .sequentialLoop definition count indexSlot bindings carriedCount
    arguments := argumentRefs
    outputCount
    outputTypes
  }

/-- The actual sequential trace selected from a successful pure requirement program. This is
execution-only; analyzer roles are attached only after the frozen site has been matched. -/
structure PureRootSequentialTrace
    (execution : PureProgramExecution)
    (root : PureProgramRootExecutionPath execution)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < execution.program.root.nodes.length)
    (view : RootSequentialLoopNodeView execution.program.root.nodes[nodeIndex]) where
  before : Mxx.Ir.WireEnvironment
  prefixPath : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel Mxx.Ir.emptySamplerFamily execution.program
      execution.program.definitions.length)
    Mxx.Ir.emptySamplerFamily execution.params execution.inputs 0
    (execution.program.root.nodes.take nodeIndex) [] before
  argumentValues : List Mxx.Ir.Value
  argumentsEvaluate :
    view.argumentRefs.mapM (fun wire => Mxx.Ir.lookupWire wire before) = some argumentValues
  evaluatedCount : Int
  countEvaluate : view.count.evaluate execution.params = some evaluatedCount
  nodeValues : List Mxx.Ir.Value
  nodeMember : nodeValues ∈ Mxx.Ir.evaluateNode
    (Mxx.Ir.childRunnerWithFuel Mxx.Ir.emptySamplerFamily execution.program
      execution.program.definitions.length)
    Mxx.Ir.emptySamplerFamily execution.params execution.inputs before
    execution.program.root.nodes[nodeIndex]
  suffixPath : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel Mxx.Ir.emptySamplerFamily execution.program
      execution.program.definitions.length)
    Mxx.Ir.emptySamplerFamily execution.params execution.inputs (nodeIndex + 1)
    (execution.program.root.nodes.drop (nodeIndex + 1))
    (before ++ Mxx.Ir.bindOutputs nodeIndex nodeValues) root.wires
  executionTrace : Mxx.Ir.SequentialIterationsTrace
    (Mxx.Ir.childRunnerWithFuel Mxx.Ir.emptySamplerFamily execution.program
      execution.program.definitions.length)
    view.definition execution.params view.indexSlot view.bindings
    (argumentValues.drop view.carriedCount) (List.range evaluatedCount.toNat)
    (argumentValues.take view.carriedCount) nodeValues

/-- Invert the selected pure-program node member into its exact sequential trace. -/
theorem PureRootSequentialTrace.nonempty_atNode
    (execution : PureProgramExecution)
    (root : PureProgramRootExecutionPath execution)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < execution.program.root.nodes.length)
    (view : RootSequentialLoopNodeView execution.program.root.nodes[nodeIndex])
    (argumentValues : List Mxx.Ir.Value)
    (argumentsEvaluate : view.argumentRefs.mapM
      (fun wire => Mxx.Ir.lookupWire wire
        (root.nodeExecutionAt nodeIndex nodeInBounds).before) = some argumentValues)
    (evaluatedCount : Int)
    (countEvaluate : view.count.evaluate execution.params = some evaluatedCount) :
    Nonempty (PureRootSequentialTrace execution root nodeIndex nodeInBounds view) := by
  classical
  let selected := root.nodeExecutionAt nodeIndex nodeInBounds
  have baseMember : selected.nodeValues ∈ Mxx.Ir.evaluateNode
      (Mxx.Ir.childRunnerWithFuel Mxx.Ir.emptySamplerFamily execution.program
        execution.program.definitions.length)
      Mxx.Ir.emptySamplerFamily execution.params execution.inputs selected.before {
        kind := .sequentialLoop view.definition view.count view.indexSlot view.bindings
          view.carriedCount
        arguments := view.argumentRefs
        outputCount := view.outputCount
      } := by
    have typedMember : selected.nodeValues ∈ Mxx.Ir.evaluateNode
        (Mxx.Ir.childRunnerWithFuel Mxx.Ir.emptySamplerFamily execution.program
          execution.program.definitions.length)
        Mxx.Ir.emptySamplerFamily execution.params execution.inputs selected.before {
          kind := .sequentialLoop view.definition view.count view.indexSlot view.bindings
            view.carriedCount
          arguments := view.argumentRefs
          outputCount := view.outputCount
          outputTypes := view.outputTypes
        } := by
      simpa [view.nodeEq] using selected.nodeMember
    rw [← evaluateNode_outputTypes_irrelevant
      (Mxx.Ir.childRunnerWithFuel Mxx.Ir.emptySamplerFamily execution.program
        execution.program.definitions.length)
      Mxx.Ir.emptySamplerFamily execution.params execution.inputs selected.before {
        kind := .sequentialLoop view.definition view.count view.indexSlot view.bindings
          view.carriedCount
        arguments := view.argumentRefs
        outputCount := view.outputCount
      } view.outputTypes]
    exact typedMember
  have loopTrace :=
    (Mxx.Ir.mem_evaluateNode_sequentialLoop_iff_trace
      (Mxx.Ir.childRunnerWithFuel Mxx.Ir.emptySamplerFamily execution.program
        execution.program.definitions.length)
      Mxx.Ir.emptySamplerFamily execution.params execution.inputs selected.before
      view.definition view.count view.indexSlot view.bindings view.carriedCount
      view.argumentRefs view.outputCount argumentValues evaluatedCount argumentsEvaluate
      countEvaluate selected.nodeValues).mp baseMember
  exact ⟨{
    before := selected.before
    prefixPath := selected.prefixPath
    argumentValues
    argumentsEvaluate
    evaluatedCount
    countEvaluate
    nodeValues := selected.nodeValues
    nodeMember := selected.nodeMember
    suffixPath := selected.suffixPath
    executionTrace := loopTrace
  }⟩

/-- Analyzer-owned recurrence identity attached to an actual pure-program loop trace.  This is
used for requirement programs; unlike a certificate invariant, every semantic field is selected
from the executable root path and the unique analyzer transfer. -/
structure TraceBoundPureSequentialRecurrence
    (analysis : AnalysisResult)
    (stage : StageId)
    (execution : PureProgramExecution)
    (recurrenceInstance : SequentialRecurrenceInstanceRef) where
  transfer : SymbolicRecurrenceTransfer
  uniqueResolution : analysis.symbolicRecurrences.filter
    (fun entry => entry.identity = recurrenceInstance) = [transfer]
  identityMatches : transfer.identity = recurrenceInstance
  root : PureProgramRootExecutionPath execution
  nodeIndex : Nat
  nodeInBounds : nodeIndex < execution.program.root.nodes.length
  view : RootSequentialLoopNodeView execution.program.root.nodes[nodeIndex]
  siteMatches : transfer.source.loop.site = {
    stage
    scope := ⟨[]⟩
    node := ⟨nodeIndex⟩
  }
  countMatches : transfer.source.count = view.count
  carriedArityMatches : transfer.source.carriedArity = view.carriedCount
  iterationSlotMatches : transfer.source.iterationVariable.slot = view.indexSlot
  trace : PureRootSequentialTrace execution root nodeIndex nodeInBounds view

/-- The final runtime value of one carried slot of the actual requirement recurrence. -/
def TraceBoundPureSequentialRecurrence.slotValue
    {analysis : AnalysisResult}
    {stage : StageId}
    {execution : PureProgramExecution}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundPureSequentialRecurrence analysis stage execution recurrenceInstance)
    (slot : Nat) : Option Mxx.Ir.Value :=
  evidence.trace.nodeValues[slot]?

/-- One element selected directly from an actual final carried family of a pure recurrence.
Both lookups are indexed by `evidence.trace.nodeValues`; no `FactEnvironment` binding or
certificate-provided family is involved. -/
structure TraceBoundPureSequentialRecurrence.FinalFamilyElementAt
    {analysis : AnalysisResult}
    {stage : StageId}
    {execution : PureProgramExecution}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundPureSequentialRecurrence analysis stage execution recurrenceInstance)
    (slot index : Nat) : Type where
  family : List Mxx.Ir.Value
  value : Mxx.Ir.Value
  familyFound : evidence.trace.nodeValues[slot]? = some (.family family)
  valueFound : family[index]? = some value

/-- Mechanically project an indexed final family value from the actual pure recurrence trace. -/
def TraceBoundPureSequentialRecurrence.finalFamilyElementAt?
    {analysis : AnalysisResult}
    {stage : StageId}
    {execution : PureProgramExecution}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundPureSequentialRecurrence analysis stage execution recurrenceInstance)
    (slot index : Nat) : Option (evidence.FinalFamilyElementAt slot index) :=
  match familyFound : evidence.trace.nodeValues[slot]? with
  | some (.family family) =>
      match valueFound : family[index]? with
      | some value => some { family, value, familyFound, valueFound }
      | none => none
  | _ => none

/-- The only child runner used by trace-bound recurrence evidence. -/
def StageExecution.rootChildRunner
    {samplers : Mxx.MxxSamplerFamily}
    (execution : StageExecution samplers) : Mxx.Ir.ChildRunner :=
  Mxx.Ir.childRunnerWithFuel samplers execution.stage.program
    execution.stage.program.definitions.length

/-- Root SSA path selected by the existing stage denotation.  This is the minimal execution
witness needed by recurrence evidence and avoids importing unrelated global soundness layers. -/
structure StageRootExecutionPath
    {samplers : Mxx.MxxSamplerFamily}
    (execution : StageExecution samplers) where
  wires : Mxx.Ir.WireEnvironment
  path : Mxx.Ir.EvaluatesNodesPath execution.rootChildRunner samplers execution.params
    execution.inputs 0 execution.stage.program.root.nodes [] wires
  outputEq : Mxx.Ir.collectOutputs execution.stage.program.root.outputs wires = execution.output

theorem StageRootExecutionPath.exists
    {samplers : Mxx.MxxSamplerFamily}
    (execution : StageExecution samplers) : Nonempty (StageRootExecutionPath execution) := by
  obtain ⟨wires, path, outputEq⟩ :=
    (Mxx.Ir.mem_denote_iff_root_path samplers execution.stage.program execution.params
      execution.inputs execution.output).mp execution.outputMember
  exact ⟨{ wires, path, outputEq := outputEq.symm }⟩

private def lookupRootArgumentFact
    (analysis : AnalysisResult)
    (stage : StageId)
    (wire : Mxx.Ir.WireRef) : Option ValueFact :=
  (analysis.facts.find? fun entry ↦ entry.wire = {
    stage
    scope := ⟨[]⟩
    node := ⟨wire.node⟩
    port := wire.port
  }).map (·.fact)

/-- Analyzer facts for the exact root-node arguments, preserving argument order. -/
def AnalysisResult.rootArgumentFacts
    (analysis : AnalysisResult)
    (stage : StageId)
    (arguments : List Mxx.Ir.WireRef) : List ValueFact :=
  arguments.filterMap (lookupRootArgumentFact analysis stage)

/-- Typed analyzer facts for exact root-node arguments. -/
def AnalysisResult.rootArgumentTemplates
    (analysis : AnalysisResult)
    (stage : StageId)
    (arguments : List Mxx.Ir.WireRef) : List ValueFactTemplate :=
  arguments.filterMap fun argument =>
    let wire : CoreWireRef := {
      stage
      scope := ⟨[]⟩
      node := ⟨argument.node⟩
      port := argument.port
    }
    (analysis.facts.find? fun entry ↦ entry.wire = wire).bind ScopedWireFact.toTemplate

/-- Analyzer facts for the exact root-node arguments, retaining the frozen wire identity. -/
def AnalysisResult.rootInvariantInputs
    (analysis : AnalysisResult)
    (stage : StageId)
    (arguments : List Mxx.Ir.WireRef) : List InvariantInputFact :=
  arguments.filterMap fun argument =>
    let wire : CoreWireRef := {
      stage
      scope := ⟨[]⟩
      node := ⟨argument.node⟩
      port := argument.port
    }
    (analysis.facts.find? fun entry => entry.wire = wire).bind fun entry =>
      entry.toTemplate.map fun template => { wire, template }

/-- Analyzer-owned facts at one exact root-node output range. -/
def AnalysisResult.rootNodeOutputFacts
    (analysis : AnalysisResult)
    (stage : StageId)
    (node : Nat)
    (count : Nat) : List ScopedWireFact :=
  analysis.facts.filter fun fact =>
    fact.wire.stage == stage && fact.wire.scope == ⟨[]⟩ &&
      fact.wire.node == ⟨node⟩ && fact.wire.port < count

/-- Frozen identity of one root recurrence output. -/
def rootRecurrenceOutputIdentity
    (stage : StageId)
    (node slot : Nat) : ValueInstanceRef :=
  .ofCoreWire {
    stage
    scope := ⟨[]⟩
    node := ⟨node⟩
    port := slot
  }

/-- Analyzer projection of the symbolic body outputs onto one root recurrence occurrence. -/
def projectRootRecurrenceOutputs
    (transfer : SymbolicRecurrenceTransfer)
    (recurrenceInstance : SequentialRecurrenceInstanceRef)
    (stage : StageId)
    (node : Nat) : List ValueFact :=
  (List.range transfer.source.carriedArity).filterMap fun slot =>
    projectRecurrenceOutput recurrenceInstance transfer.source.bodyOutputs slot
      (rootRecurrenceOutputIdentity stage node slot)

/-- Semantic evidence for an analyzed recurrence at one exact root node of an actual stage
execution.  All runtime data is selected from `StageWireExecution.path`.  Invariant inputs are
`argumentValues.drop view.carriedCount`; there is no field through which a caller can replace
them. -/
structure TraceBoundSequentialRecurrence
    (analysis : AnalysisResult)
    {samplers : Mxx.MxxSamplerFamily}
    (execution : StageExecution samplers)
    (recurrenceInstance : SequentialRecurrenceInstanceRef) where
  transfer : SymbolicRecurrenceTransfer
  uniqueResolution : analysis.symbolicRecurrences.filter
    (fun entry => entry.identity = recurrenceInstance) = [transfer]
  identityMatches : transfer.identity = recurrenceInstance
  stageWires : StageRootExecutionPath execution
  nodeIndex : Nat
  nodeInBounds : nodeIndex < execution.stage.program.root.nodes.length
  view : RootSequentialLoopNodeView execution.stage.program.root.nodes[nodeIndex]
  siteMatches : transfer.source.loop.site = {
    stage := ⟨execution.stage.id⟩
    scope := ⟨[]⟩
    node := ⟨nodeIndex⟩
  }
  countMatches : transfer.source.count = view.count
  carriedArityMatches : transfer.source.carriedArity = view.carriedCount
  iterationSlotMatches : transfer.source.iterationVariable.slot = view.indexSlot
  initialFactsMatch : transfer.source.initial.toList = analysis.rootArgumentTemplates
    ⟨execution.stage.id⟩ (view.argumentRefs.take view.carriedCount)
  invariantFactsMatch : transfer.source.invariantInputs = analysis.rootInvariantInputs
    ⟨execution.stage.id⟩ (view.argumentRefs.drop view.carriedCount)
  projectedFactsMatch :
    (analysis.rootNodeOutputFacts ⟨execution.stage.id⟩ nodeIndex
      transfer.source.carriedArity).map (·.fact) =
    projectRootRecurrenceOutputs transfer recurrenceInstance ⟨execution.stage.id⟩ nodeIndex
  before : Mxx.Ir.WireEnvironment
  prefixPath : Mxx.Ir.EvaluatesNodesPath execution.rootChildRunner samplers execution.params
    execution.inputs 0 (execution.stage.program.root.nodes.take nodeIndex) [] before
  argumentValues : List Mxx.Ir.Value
  evaluatedCount : Int
  countNonnegative : 0 ≤ evaluatedCount
  argumentsEvaluate :
    view.argumentRefs.mapM (fun wire ↦ Mxx.Ir.lookupWire wire before) = some argumentValues
  countEvaluate : view.count.evaluate execution.params = some evaluatedCount
  nodeValues : List Mxx.Ir.Value
  nodeMember : nodeValues ∈ Mxx.Ir.evaluateNode execution.rootChildRunner samplers
    execution.params execution.inputs before execution.stage.program.root.nodes[nodeIndex]
  suffixPath : Mxx.Ir.EvaluatesNodesPath execution.rootChildRunner samplers execution.params
    execution.inputs (nodeIndex + 1) (execution.stage.program.root.nodes.drop (nodeIndex + 1))
    (before ++ Mxx.Ir.bindOutputs nodeIndex nodeValues) stageWires.wires
  executionTrace : Mxx.Ir.SequentialIterationsTrace execution.rootChildRunner view.definition
    execution.params view.indexSlot view.bindings (argumentValues.drop view.carriedCount)
    (List.range evaluatedCount.toNat) (argumentValues.take view.carriedCount) nodeValues

namespace TraceBoundSequentialRecurrence

private theorem final_eq_initial_of_nil_trace
    {runChild : Mxx.Ir.ChildRunner}
    {definition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {indexSlot : Nat}
    {bindings : List (String × Mxx.Ir.IntExpr)}
    {invariantArguments initial final : List Mxx.Ir.Value}
    (trace : Mxx.Ir.SequentialIterationsTrace runChild definition params indexSlot bindings
      invariantArguments [] initial final) :
    final = initial := by
  cases trace
  rfl

/-- Invariant values are always the suffix of the arguments evaluated at the selected loop node. -/
def invariantValues
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance) :
    List Mxx.Ir.Value :=
  evidence.argumentValues.drop evidence.view.carriedCount

/-- Final carried slots are exactly the outputs selected by the executable loop node. -/
def slotValue
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance)
    (slot : Nat) : Option Mxx.Ir.Value :=
  evidence.nodeValues[slot]?

/-- Build trace-bound evidence from the exact root-node support member selected by a stage path.
The sequential trace is obtained by inversion of that support member; the loop is not re-run. -/
def ofRootNodeMember
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (transfer : SymbolicRecurrenceTransfer)
    (uniqueResolution : analysis.symbolicRecurrences.filter
      (fun entry => entry.identity = recurrenceInstance) = [transfer])
    (identityMatches : transfer.identity = recurrenceInstance)
    (stageWires : StageRootExecutionPath execution)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < execution.stage.program.root.nodes.length)
    (view : RootSequentialLoopNodeView execution.stage.program.root.nodes[nodeIndex])
    (siteMatches : transfer.source.loop.site = {
      stage := ⟨execution.stage.id⟩
      scope := ⟨[]⟩
      node := ⟨nodeIndex⟩
    })
    (countMatches : transfer.source.count = view.count)
    (carriedArityMatches : transfer.source.carriedArity = view.carriedCount)
    (iterationSlotMatches : transfer.source.iterationVariable.slot = view.indexSlot)
    (initialFactsMatch : transfer.source.initial.toList = analysis.rootArgumentTemplates
      ⟨execution.stage.id⟩ (view.argumentRefs.take view.carriedCount))
    (invariantFactsMatch : transfer.source.invariantInputs = analysis.rootInvariantInputs
      ⟨execution.stage.id⟩ (view.argumentRefs.drop view.carriedCount))
    (projectedFactsMatch :
      (analysis.rootNodeOutputFacts ⟨execution.stage.id⟩ nodeIndex
        transfer.source.carriedArity).map (·.fact) =
      projectRootRecurrenceOutputs transfer recurrenceInstance ⟨execution.stage.id⟩ nodeIndex)
    (before : Mxx.Ir.WireEnvironment)
    (prefixPath : Mxx.Ir.EvaluatesNodesPath execution.rootChildRunner samplers execution.params
      execution.inputs 0 (execution.stage.program.root.nodes.take nodeIndex) [] before)
    (argumentValues : List Mxx.Ir.Value)
    (evaluatedCount : Int)
    (countNonnegative : 0 ≤ evaluatedCount)
    (argumentsEvaluate :
      view.argumentRefs.mapM (fun wire ↦ Mxx.Ir.lookupWire wire before) = some argumentValues)
    (countEvaluate : view.count.evaluate execution.params = some evaluatedCount)
    (nodeValues : List Mxx.Ir.Value)
    (nodeMember : nodeValues ∈ Mxx.Ir.evaluateNode execution.rootChildRunner samplers
      execution.params execution.inputs before execution.stage.program.root.nodes[nodeIndex])
    (suffixPath : Mxx.Ir.EvaluatesNodesPath execution.rootChildRunner samplers execution.params
      execution.inputs (nodeIndex + 1)
      (execution.stage.program.root.nodes.drop (nodeIndex + 1))
      (before ++ Mxx.Ir.bindOutputs nodeIndex nodeValues) stageWires.wires) :
    TraceBoundSequentialRecurrence analysis execution recurrenceInstance := by
  have member : nodeValues ∈ Mxx.Ir.evaluateNode execution.rootChildRunner samplers
      execution.params execution.inputs before {
        kind := .sequentialLoop view.definition view.count view.indexSlot view.bindings
          view.carriedCount
        arguments := view.argumentRefs
        outputCount := view.outputCount
      } := by
    have typedMember : nodeValues ∈ Mxx.Ir.evaluateNode execution.rootChildRunner samplers
        execution.params execution.inputs before {
          kind := .sequentialLoop view.definition view.count view.indexSlot view.bindings
            view.carriedCount
          arguments := view.argumentRefs
          outputCount := view.outputCount
          outputTypes := view.outputTypes
        } := by
      simpa [view.nodeEq] using nodeMember
    rw [← evaluateNode_outputTypes_irrelevant execution.rootChildRunner samplers execution.params
      execution.inputs before {
        kind := .sequentialLoop view.definition view.count view.indexSlot view.bindings
          view.carriedCount
        arguments := view.argumentRefs
        outputCount := view.outputCount
      } view.outputTypes]
    exact typedMember
  have executionTrace :=
    (Mxx.Ir.mem_evaluateNode_sequentialLoop_iff_trace execution.rootChildRunner samplers
      execution.params execution.inputs before view.definition view.count view.indexSlot
      view.bindings view.carriedCount view.argumentRefs view.outputCount argumentValues
      evaluatedCount argumentsEvaluate countEvaluate nodeValues).mp member
  exact {
    transfer
    uniqueResolution
    identityMatches
    stageWires
    nodeIndex
    nodeInBounds
    view
    siteMatches
    countMatches
    carriedArityMatches
    iterationSlotMatches
    initialFactsMatch
    invariantFactsMatch
    projectedFactsMatch
    before
    prefixPath
    argumentValues
    evaluatedCount
    countNonnegative
    argumentsEvaluate
    countEvaluate
    nodeValues
    nodeMember
    suffixPath
    executionTrace
  }

/-- A zero-iteration loop returns its initial carried values exactly. -/
theorem final_eq_initial_of_count_eq_zero
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance)
    (countEq : evidence.evaluatedCount = 0) :
    evidence.nodeValues = evidence.argumentValues.take evidence.view.carriedCount := by
  have trace := evidence.executionTrace
  simp only [countEq, Int.toNat_zero, List.range_zero] at trace
  exact final_eq_initial_of_nil_trace trace

/-- A one-iteration loop exposes exactly one child-support member using the runner fixed by the
stage execution, with the actual invariant argument suffix. -/
theorem one_step_child
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance)
    (countEq : evidence.evaluatedCount = 1) :
    ∃ evaluatedBindings,
      Mxx.Ir.evaluateBindings
        ((.loopIndex evidence.view.indexSlot, .integer 0) :: execution.params)
        evidence.view.bindings = some evaluatedBindings ∧
      evidence.nodeValues ∈ execution.rootChildRunner evidence.view.definition
        (evaluatedBindings ++
          ((.loopIndex evidence.view.indexSlot, .integer 0) :: execution.params))
        (evidence.argumentValues.take evidence.view.carriedCount ++
          evidence.argumentValues.drop evidence.view.carriedCount) := by
  have trace := evidence.executionTrace
  simp only [countEq, Int.toNat_one, List.range_one] at trace
  cases trace with
  | cons index tail state evaluatedBindings next final bindingsEvaluate childMember rest =>
      cases rest
      exact ⟨evaluatedBindings, bindingsEvaluate, childMember⟩

/-! ## Analyzer-derived recurrence facts

The step relation below is indexed by the binding equality and child-support member stored in the
real `SequentialIterationsTrace`.  It contains no replacement runner, trace, invariant, or
preservation function.  A body soundness proof discharges the complete instantiated output facts
by composing the registered local rules for that exact child member.  Coarse carried schemas are
not semantic evidence. -/

private def templateMatrixType : ValueFactTemplate → Option MatrixTypeExpr
  | { schema := .matrix matrixType _ _ _, .. } => some matrixType
  | _ => none

private def invariantSeedFact?
    (stage : StageId)
    (bodyScope : StaticScopeId)
    (body : Mxx.Ir.Scope)
    (name : String)
    (invariant : InvariantInputFact) : Option ScopedWireFact := do
  if invariant.template.fact.hasCarriedInput then none else
  let destination ← inputNodeWireInScope? stage bodyScope name body.nodes
  return transportFact destination {
    wire := invariant.wire
    matrixType := templateMatrixType invariant.template
    fact := invariant.template.fact
  }

/-- Reconstruct every formal input fact of the frozen sequential body.  Carried inputs are the
exact analyzer-only placeholders retained by the recurrence source; invariant inputs are ordinary
transported facts.  An arity mismatch, a missing input node, or an escaped carried placeholder is
an analysis failure rather than a fallback to a coarser schema. -/
def bodySeedFacts?
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance)
    (body : Mxx.Ir.Scope) : Option ScopedWireFactTable := do
  let carried ← evidence.transfer.source.abstractCarriedSeedFacts?
  let invariantNames := body.inputNames.drop evidence.transfer.source.carriedArity
  let invariants := evidence.transfer.source.invariantInputs
  if body.inputNames.length != evidence.transfer.source.carriedArity + invariants.length then none
  else if invariantNames.length != invariants.length then none
  else
    let invariantFacts ← (invariantNames.zip invariants).mapM fun (name, invariant) =>
      invariantSeedFact? ⟨execution.stage.id⟩ ⟨[evidence.view.definition]⟩ body name invariant
    return carried ++ invariantFacts

/-- Recover body output facts in frozen declaration order from one analyzer fact table. -/
private def bodyOutputFacts?
    (stage : StageId)
    (scope : StaticScopeId)
    (body : Mxx.Ir.Scope)
    (facts : ScopedWireFactTable) : Option (List ScopedWireFact) :=
  body.outputs.mapM fun output ↦ do
    let outputWire : CoreWireRef := {
      stage
      scope
      node := ⟨output.2.node⟩
      port := output.2.port
    }
    facts.find? (fun candidate ↦ candidate.wire = outputWire)

/-- One analyzer-produced body result, tied to one exact executable trace step.

The closed-program theorem currently interprets only concrete root wires; applying it directly to
the template identities in a child scope would be unsound. This evidence therefore records the
frozen definition lookup, exact fail-closed analyzer acceptance seeded by the formal carried and
invariant facts, and its frozen output templates. `AnalyzerDerivedNextFacts` below carries the
substitution-first semantic obligation required before a local-rule composition can become a final
recurrence proof. -/
private structure AnalyzerBodyLocalRuleStep
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance)
    (index : Nat)
    (state next : List Mxx.Ir.Value)
    (evaluatedBindings : Mxx.Ir.ParamEnvironment)
    (bindingsEvaluate : Mxx.Ir.evaluateBindings
      ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params)
      evidence.view.bindings = some evaluatedBindings)
    (childMember : next ∈ execution.rootChildRunner evidence.view.definition
      (evaluatedBindings ++
        ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
      (state ++ evidence.invariantValues)) where
  fuel : Nat
  fuelMatches : execution.stage.program.definitions.length = fuel + 1
  body : Mxx.Ir.Scope
  definitionFound : Mxx.Ir.lookupDefinition evidence.view.definition
    execution.stage.program.definitions = some body
  seedFacts : ScopedWireFactTable
  seedFactsDerived : bodySeedFacts? evidence body = some seedFacts
  analyzerFacts : ScopedWireFactTable
  analyzerAccepted : inferRulesFrom ⟨execution.stage.id⟩
    ⟨[evidence.view.definition]⟩ 0 body.nodes seedFacts = .ok analyzerFacts
  rawOutputs : List ScopedWireFact
  rawOutputsDerived : bodyOutputFacts? ⟨execution.stage.id⟩
    ⟨[evidence.view.definition]⟩ body analyzerFacts = some rawOutputs
  rawOutputTemplates : List ValueFactTemplate
  rawOutputTemplatesDerived : rawOutputs.mapM ScopedWireFact.toTemplate = some rawOutputTemplates
  sourceOutputsMatch : rawOutputTemplates = evidence.transfer.source.bodyOutputs.toList

/-- The complete initial fact table and exact initial carried values used by the abstract body.
It is retained by the recurrence source and real loop trace, never reconstructed from a coarse
schema or supplied by the caller. -/
def initialSubstitution?
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance) :
    Option CarriedFactSubstitution :=
  CarriedFactSubstitution.build evidence.transfer.source.initial.toList
    (evidence.argumentValues.take evidence.view.carriedCount)
    evidence.transfer.source.familyElementTemplates

theorem initialSubstitution_values
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    {evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance}
    {substitution : CarriedFactSubstitution}
    (derived : evidence.initialSubstitution? = some substitution) :
    substitution.values = evidence.argumentValues.take evidence.view.carriedCount := by
  exact CarriedFactSubstitution.build_values derived

/-- Recover the complete body-output facts from the exact analyzer result attached to this step.
The output order is the frozen subgraph output order.  Failure to recover any fact is an analysis
failure, not an opportunity to manufacture one from `CarriedValueSchema`. -/
def AnalyzerBodyLocalRuleStep.outputFacts?
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    {evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance}
    {index : Nat}
    {state next : List Mxx.Ir.Value}
    {evaluatedBindings : Mxx.Ir.ParamEnvironment}
    {bindingsEvaluate : Mxx.Ir.evaluateBindings
      ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params)
      evidence.view.bindings = some evaluatedBindings}
    {childMember : next ∈ execution.rootChildRunner evidence.view.definition
      (evaluatedBindings ++
        ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
      (state ++ evidence.invariantValues)}
    (step : AnalyzerBodyLocalRuleStep evidence index state next evaluatedBindings
      bindingsEvaluate childMember) : Option (List ScopedWireFact) :=
  some step.rawOutputs

/-- Repackage the exact analyzer-produced raw output facts with the actual next carried values.
This is purely structural; semantic validity is stated by `AnalyzerDerivedNextFacts` below. -/
def AnalyzerBodyLocalRuleStep.rawOutputSubstitution?
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    {evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance}
    {index : Nat}
    {state next : List Mxx.Ir.Value}
    {evaluatedBindings : Mxx.Ir.ParamEnvironment}
    {bindingsEvaluate : Mxx.Ir.evaluateBindings
      ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params)
      evidence.view.bindings = some evaluatedBindings}
    {childMember : next ∈ execution.rootChildRunner evidence.view.definition
      (evaluatedBindings ++
        ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
      (state ++ evidence.invariantValues)}
    (step : AnalyzerBodyLocalRuleStep evidence index state next evaluatedBindings
      bindingsEvaluate childMember) : Option CarriedFactSubstitution := do
  CarriedFactSubstitution.build step.rawOutputTemplates next
    evidence.transfer.source.familyElementTemplates

/-- Instantiate the one abstract body-output tuple from the complete immutable carried state at
the start of this exact trace step.  This is distinct from `rawOutputSubstitution?`: the latter
recovers the analyzer's raw output templates for auditing, whereas this operation eliminates all
`carriedInput` placeholders before the next state is exposed to semantics. -/
def AnalyzerBodyLocalRuleStep.instantiatedNextSubstitution?
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    {evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance}
    {index : Nat}
    {state next : List Mxx.Ir.Value}
    {evaluatedBindings : Mxx.Ir.ParamEnvironment}
    {bindingsEvaluate : Mxx.Ir.evaluateBindings
      ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params)
      evidence.view.bindings = some evaluatedBindings}
    {childMember : next ∈ execution.rootChildRunner evidence.view.definition
      (evaluatedBindings ++
        ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
      (state ++ evidence.invariantValues)}
    (step : AnalyzerBodyLocalRuleStep evidence index state next evaluatedBindings
      bindingsEvaluate childMember)
    (previous : CarriedFactSubstitution) : Option CarriedFactSubstitution := do
  let _ ← step.rawOutputSubstitution?
  previous.instantiateBodyOutputs step.rawOutputTemplates next
    evidence.transfer.source.familyElementTemplates

theorem AnalyzerBodyLocalRuleStep.instantiatedNextSubstitution_values
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    {evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance}
    {index : Nat}
    {state next : List Mxx.Ir.Value}
    {evaluatedBindings : Mxx.Ir.ParamEnvironment}
    {bindingsEvaluate : Mxx.Ir.evaluateBindings
      ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params)
      evidence.view.bindings = some evaluatedBindings}
    {childMember : next ∈ execution.rootChildRunner evidence.view.definition
      (evaluatedBindings ++
        ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
      (state ++ evidence.invariantValues)}
    {step : AnalyzerBodyLocalRuleStep evidence index state next evaluatedBindings
      bindingsEvaluate childMember}
    {previous substitution : CarriedFactSubstitution}
    (derived : step.instantiatedNextSubstitution? previous = some substitution) :
    substitution.values = next := by
  unfold AnalyzerBodyLocalRuleStep.instantiatedNextSubstitution? at derived
  cases accepted : step.rawOutputSubstitution? with
  | none => simp [accepted] at derived
  | some outputTemplates =>
      simp only [accepted] at derived
      simp only [CarriedFactSubstitution.instantiateBodyOutputs] at derived
      cases instantiated : previous.instantiateTemplates step.rawOutputTemplates with
      | none => simp [instantiated] at derived
      | some facts =>
          simp only [instantiated] at derived
          exact CarriedFactSubstitution.build_values derived

/-- The child support member stored in a step is exactly an execution path through the frozen
definition accepted by that step. No alternate runner or body can be supplied. -/
theorem AnalyzerBodyLocalRuleStep.bodyExecutionPath
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    {evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance}
    {index : Nat}
    {state next : List Mxx.Ir.Value}
    {evaluatedBindings : Mxx.Ir.ParamEnvironment}
    {bindingsEvaluate : Mxx.Ir.evaluateBindings
      ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params)
      evidence.view.bindings = some evaluatedBindings}
    {childMember : next ∈ execution.rootChildRunner evidence.view.definition
      (evaluatedBindings ++
        ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
      (state ++ evidence.invariantValues)}
    (step : AnalyzerBodyLocalRuleStep evidence index state next evaluatedBindings
      bindingsEvaluate childMember) :
    ∃ wires,
      Mxx.Ir.EvaluatesNodesPath
        (Mxx.Ir.childRunnerWithFuel samplers execution.stage.program step.fuel)
        samplers
        (evaluatedBindings ++
          ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
        (step.body.inputNames.zip (state ++ evidence.invariantValues))
        0 step.body.nodes [] wires ∧
      next = (Mxx.Ir.collectOutputs step.body.outputs wires).map Prod.snd := by
  have member : next ∈ Mxx.Ir.childRunnerWithFuel samplers execution.stage.program
      (step.fuel + 1) evidence.view.definition
      (evaluatedBindings ++
        ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
      (state ++ evidence.invariantValues) := by
    simpa [StageExecution.rootChildRunner, step.fuelMatches] using childMember
  exact (Mxx.Ir.mem_childRunnerWithFuel_succ_iff_path samplers execution.stage.program step.fuel
    evidence.view.definition step.body
    (evaluatedBindings ++
      ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
    (state ++ evidence.invariantValues) next step.definitionFound).mp member

/-- Exact one-step semantic evidence after the previous complete fact state has been substituted
into every carried placeholder. Each body output retains its full analyzer fact and proves that
ordinary semantics holds for the exact value produced by this child execution. -/
private structure AnalyzerDerivedNextFacts
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    {evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance}
    {index : Nat}
    {state next : List Mxx.Ir.Value}
    {evaluatedBindings : Mxx.Ir.ParamEnvironment}
    {bindingsEvaluate : Mxx.Ir.evaluateBindings
      ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params)
      evidence.view.bindings = some evaluatedBindings}
    {childMember : next ∈ execution.rootChildRunner evidence.view.definition
      (evaluatedBindings ++
        ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
      (state ++ evidence.invariantValues)}
    (step : AnalyzerBodyLocalRuleStep evidence index state next evaluatedBindings
      bindingsEvaluate childMember)
    (_contract : Mxx.MxxBoundedSamplerContract samplers)
    (previous : CarriedFactSubstitution) : Type where
  bodyWires : Mxx.Ir.WireEnvironment
  bodyPath :
    Mxx.Ir.EvaluatesNodesPath
      (Mxx.Ir.childRunnerWithFuel samplers execution.stage.program step.fuel)
      samplers
      (evaluatedBindings ++
        ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
      (step.body.inputNames.zip (state ++ evidence.invariantValues))
      0 step.body.nodes [] bodyWires
  outputValues : next = (Mxx.Ir.collectOutputs step.body.outputs bodyWires).map Prod.snd
  instantiatedOutputs : List.Forall₂
    (fun raw actual => InstantiatedScopedFact.Holds previous
      (FactEnvironment.ofWireEnvironment
        (evaluatedBindings ++
          ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
        ⟨execution.stage.id⟩ ⟨[evidence.view.definition]⟩ bodyWires)
      raw actual)
    step.rawOutputs next

/-- Complete recurrence derivation over the exact executable trace.  Unlike the older
coarse-schema helper above, this judgment transports the analyzer-owned full fact templates and
the corresponding immutable carried runtime values at every boundary.  The only substitution
objects it accepts are the closed computations `initialSubstitution?` and
`AnalyzerBodyLocalRuleStep.rawOutputSubstitution?`; no theorem caller can name a replacement fact
table or recurrence invariant. -/
inductive CompleteDerivation
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance)
    (contract : Mxx.MxxBoundedSamplerContract samplers) :
    {indices : List Nat} → {initial final : List Mxx.Ir.Value} →
      (trace : Mxx.Ir.SequentialIterationsTrace execution.rootChildRunner evidence.view.definition
        execution.params evidence.view.indexSlot evidence.view.bindings evidence.invariantValues
        indices initial final) → CarriedFactSubstitution → Type where
  | nil
      {state : List Mxx.Ir.Value}
      (substitution : CarriedFactSubstitution)
      (derived : evidence.initialSubstitution? = some substitution)
      (valuesMatch : substitution.values = state) :
      CompleteDerivation evidence contract (.nil state) substitution
  | cons
      {index : Nat}
      {tail : List Nat}
      {state next final : List Mxx.Ir.Value}
      {evaluatedBindings : Mxx.Ir.ParamEnvironment}
      (bindingsEvaluate : Mxx.Ir.evaluateBindings
        ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params)
        evidence.view.bindings = some evaluatedBindings)
      (childMember : next ∈ execution.rootChildRunner evidence.view.definition
        (evaluatedBindings ++
          ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
        (state ++ evidence.invariantValues))
      (rest : Mxx.Ir.SequentialIterationsTrace execution.rootChildRunner
        evidence.view.definition execution.params evidence.view.indexSlot evidence.view.bindings
        evidence.invariantValues tail next final)
      (step : AnalyzerBodyLocalRuleStep evidence index state next evaluatedBindings
        bindingsEvaluate childMember)
      {before : CarriedFactSubstitution}
      (beforeValues : before.values = state)
      (localFacts : AnalyzerDerivedNextFacts step contract before)
      (nextSubstitution : CarriedFactSubstitution)
      (nextDerived : step.instantiatedNextSubstitution? before = some nextSubstitution)
      (nextValues : nextSubstitution.values = next)
      (restDerivation : CompleteDerivation evidence contract rest nextSubstitution) :
      CompleteDerivation evidence contract
        (.cons index tail state evaluatedBindings next final bindingsEvaluate childMember rest) before

/-- Every complete trace derivation ends in one immutable substitution for the exact final
carried tuple.  This induction follows trace structure only; it never unfolds symbolic forms. -/
theorem CompleteDerivation.existsFinalSubstitution
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    {evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance}
    {contract : Mxx.MxxBoundedSamplerContract samplers}
    {indices : List Nat}
    {initial final : List Mxx.Ir.Value}
    {trace : Mxx.Ir.SequentialIterationsTrace execution.rootChildRunner evidence.view.definition
      execution.params evidence.view.indexSlot evidence.view.bindings evidence.invariantValues
      indices initial final}
    {initialSubstitution : CarriedFactSubstitution}
    (derivation : CompleteDerivation evidence contract trace initialSubstitution) :
    ∃ substitution : CarriedFactSubstitution, substitution.values = final := by
  induction derivation with
  | nil substitution _ valuesMatch => exact ⟨substitution, valuesMatch⟩
  | cons _ _ _ _ _ _ _ _ _ restDerivation induction => exact induction

/-- Extract the final immutable substitution without expanding the loop's symbolic expressions. -/
noncomputable def CompleteDerivation.finalSubstitution
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    {evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance}
    {contract : Mxx.MxxBoundedSamplerContract samplers}
    {indices : List Nat}
    {initial final : List Mxx.Ir.Value}
    {trace : Mxx.Ir.SequentialIterationsTrace execution.rootChildRunner evidence.view.definition
      execution.params evidence.view.indexSlot evidence.view.bindings evidence.invariantValues
      indices initial final}
    {initialSubstitution : CarriedFactSubstitution}
    (derivation : CompleteDerivation evidence contract trace initialSubstitution) : CarriedFactSubstitution :=
  Classical.choose (CompleteDerivation.existsFinalSubstitution derivation)

theorem CompleteDerivation.finalValues
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    {evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance}
    {contract : Mxx.MxxBoundedSamplerContract samplers}
    {indices : List Nat}
    {initial final : List Mxx.Ir.Value}
    {trace : Mxx.Ir.SequentialIterationsTrace execution.rootChildRunner evidence.view.definition
      execution.params evidence.view.indexSlot evidence.view.bindings evidence.invariantValues
      indices initial final}
    {initialSubstitution : CarriedFactSubstitution}
    (derivation : CompleteDerivation evidence contract trace initialSubstitution) :
    derivation.finalSubstitution.values = final := by
  exact Classical.choose_spec (CompleteDerivation.existsFinalSubstitution derivation)

/-- The exact root output identity of one final carried slot. -/
def outputIdentity
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance)
    (slot : Nat) : ValueInstanceRef :=
  rootRecurrenceOutputIdentity ⟨execution.stage.id⟩ evidence.nodeIndex slot

/-- Project the analyzer-produced body templates onto the actual root output identities. -/
def projectedCarriedFacts
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance) :
    List ValueFact :=
  projectRootRecurrenceOutputs evidence.transfer recurrenceInstance
    ⟨execution.stage.id⟩ evidence.nodeIndex

/-- Final recurrence result: analyzer-projected facts paired with the exact final carried values
and the trace-derived complete substitution that owns their fact components. -/
structure FinalProjectedCarriedFacts
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance)
    (environment : FactEnvironment) where
  facts : List ValueFact
  scopedFacts : List ScopedWireFact
  values : List Mxx.Ir.Value
  factsEq : facts = evidence.projectedCarriedFacts
  scopedFactsEq : scopedFacts = analysis.rootNodeOutputFacts ⟨execution.stage.id⟩
    evidence.nodeIndex evidence.transfer.source.carriedArity
  scopedFactsProject : scopedFacts.map (·.fact) = facts
  scopedFactsHold : ∀ fact ∈ scopedFacts, fact.Holds environment
  valuesEq : values = evidence.nodeValues
  finalSubstitution : CarriedFactSubstitution
  substitutionValues : finalSubstitution.values = values

/-- A matrix fact selected from the projected list inherits its semantic proof from the exact
root output fact.  This is a projection of analyzer soundness, not a caller-supplied invariant. -/
theorem FinalProjectedCarriedFacts.matrixFactHolds
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    {evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance}
    {environment : FactEnvironment}
    (projection : FinalProjectedCarriedFacts evidence environment)
    (matrix : MatrixFact)
    (member : ValueFact.matrix matrix ∈ projection.facts) :
    matrix.Holds environment := by
  have mapped : ValueFact.matrix matrix ∈ projection.scopedFacts.map (·.fact) := by
    rw [projection.scopedFactsProject]
    exact member
  obtain ⟨scopedFact, scopedMember, factEq⟩ := List.mem_map.mp mapped
  have holds := projection.scopedFactsHold scopedFact scopedMember
  unfold ScopedWireFact.Holds at holds
  rw [factEq] at holds
  exact holds

noncomputable def CompleteDerivation.finalProjection
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance)
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (environment : FactEnvironment)
    (analysisHolds : BaseAnalysisHolds environment analysis)
    {initialSubstitution : CarriedFactSubstitution}
    (derivation : CompleteDerivation evidence contract evidence.executionTrace initialSubstitution) :
    FinalProjectedCarriedFacts evidence environment := {
  facts := evidence.projectedCarriedFacts
  scopedFacts := analysis.rootNodeOutputFacts ⟨execution.stage.id⟩ evidence.nodeIndex
    evidence.transfer.source.carriedArity
  values := evidence.nodeValues
  factsEq := rfl
  scopedFactsEq := rfl
  scopedFactsProject := evidence.projectedFactsMatch
  scopedFactsHold := by
    intro fact member
    exact analysisHolds.2.2.2 fact (List.mem_of_mem_filter member)
  valuesEq := rfl
  finalSubstitution := derivation.finalSubstitution
  substitutionValues := derivation.finalValues
}

end TraceBoundSequentialRecurrence

end Mxx.Certificate
