import Mxx.Certificate.Execution
import Mxx.Certificate.Analyzer
import Mxx.Certificate.Rules.Family
import Mxx.Certificate.Rules.ClosedProgram
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
    (lookupRootArgumentFact analysis stage argument).map fun fact => { wire, fact }

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
preservation function.  A body soundness proof discharges `nextFacts` by composing the registered
local rules for that exact child member.
-/

/-- Closed runtime check for one analyzer-derived carried schema. The first implementation
deliberately rejects canonical-residue matrix schemas and therefore cannot silently assume a
representation property that the checked child body has not established. -/
private def carriedValueAccepted
    (parameters : Mxx.Ir.ParamEnvironment) : CarriedValueSchema → Mxx.Ir.Value → Bool
  | .matrix matrixType .unknown, .matrix value =>
      match matrixType.evaluate parameters with
      | none => false
      | some evaluated =>
          value.modulus == evaluated.modulus &&
            value.ringDimension == evaluated.ringDimension &&
            value.rows == evaluated.rows && value.columns == evaluated.columns
  | .matrix _ (.canonicalResidues _), _ => false
  | .integer, .integer _ => true
  | .boolean, .boolean _ => true
  | .bytes, .bytes _ => true
  | .family count element, .family values =>
      match evaluateIntExpr parameters count with
      | .error _ => false
      | .ok evaluatedCount =>
          decide (0 ≤ evaluatedCount) && values.length == evaluatedCount.toNat &&
            values.all (carriedValueAccepted parameters element)
  | _, _ => false

/-- Closed slot-by-slot check for the actual carried tuple selected by the executable trace. -/
private def carriedStateAccepted
    (parameters : Mxx.Ir.ParamEnvironment) :
    List CarriedValueSchema → List Mxx.Ir.Value → Bool
  | [], [] => true
  | schema :: schemas, value :: values =>
      carriedValueAccepted parameters schema value &&
        carriedStateAccepted parameters schemas values
  | _, _ => false

private theorem carriedValueAccepted_sound
    (parameters : Mxx.Ir.ParamEnvironment)
    (schema : CarriedValueSchema)
    (value : Mxx.Ir.Value)
    (accepted : carriedValueAccepted parameters schema value = true) :
    schema.Holds parameters value := by
  induction schema generalizing value with
  | matrix matrixType representation =>
      cases representation <;> cases value <;>
        simp [carriedValueAccepted, CarriedValueSchema.Holds] at accepted ⊢
      rename_i matrix
      split at accepted
      · contradiction
      · rename_i evaluated typeEvaluates
        simp only [Bool.and_eq_true, beq_iff_eq] at accepted
        exact ⟨⟨evaluated, typeEvaluates, accepted.1.1.1, accepted.1.1.2,
          accepted.1.2, accepted.2⟩, trivial⟩
  | integer => cases value <;> simp [carriedValueAccepted, CarriedValueSchema.Holds] at accepted ⊢
  | boolean => cases value <;> simp [carriedValueAccepted, CarriedValueSchema.Holds] at accepted ⊢
  | bytes => cases value <;> simp [carriedValueAccepted, CarriedValueSchema.Holds] at accepted ⊢
  | family count element induction =>
      cases value <;> simp [carriedValueAccepted, CarriedValueSchema.Holds] at accepted ⊢
      rename_i values
      split at accepted
      · contradiction
      · rename_i evaluatedCount countEvaluates
        simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq,
          List.all_eq_true] at accepted
        exact ⟨evaluatedCount, countEvaluates, accepted.1.1,
          accepted.1.2, fun value member => induction value (accepted.2 value member)⟩

private theorem carriedStateAccepted_sound
    (parameters : Mxx.Ir.ParamEnvironment)
    (schemas : List CarriedValueSchema)
    (values : List Mxx.Ir.Value)
    (accepted : carriedStateAccepted parameters schemas values = true) :
    CarriedState.Holds parameters schemas values := by
  induction schemas generalizing values with
  | nil =>
      cases values with
      | nil => exact .nil
      | cons value values => simp [carriedStateAccepted] at accepted
  | cons schema schemas induction =>
      cases values with
      | nil => simp [carriedStateAccepted] at accepted
      | cons value values =>
          simp only [carriedStateAccepted, Bool.and_eq_true] at accepted
          exact .cons (carriedValueAccepted_sound parameters schema value accepted.1)
            (induction values accepted.2)

/-- Negative fixture: an unrelated Boolean result cannot satisfy an analyzer-derived integer
carried slot, so it cannot be used to construct a local-rule step. -/
example : carriedStateAccepted [] [.integer] [.boolean false] = false := by
  rfl

/-- One analyzer-produced body result, tied to one exact executable trace step.

The closed-program theorem currently interprets only concrete root wires; applying it directly to
the template identities in a child scope would be unsound. This evidence therefore records the
frozen definition lookup, exact fail-closed analyzer acceptance, and the result of the closed
runtime schema checker. `AnalyzerDerivedNextFacts` below applies generic local-rule soundness to
the exact child path and retains the resulting semantic fact proof; it does not accept one from a
certificate or theorem caller.
-/
structure AnalyzerBodyLocalRuleStep
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
  bodyClosed : ClosedPrimitiveNodes
    (evaluatedBindings ++
      ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
    0 body.nodes
  analyzerFacts : ScopedWireFactTable
  analyzerAccepted : inferRulesFrom ⟨execution.stage.id⟩
    ⟨[evidence.view.definition]⟩ 0 body.nodes [] = .ok analyzerFacts
  schemaAccepted : carriedStateAccepted execution.params evidence.transfer.carriedSchemas next =
    true

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

/-- Evidence derived from the exact child execution and the exact analyzer result for that child.
The caller cannot supply either the body facts or their soundness proof: both are reconstructed
from `childMember`, `definitionFound`, and `analyzerAccepted`.  The coarse carried-state proof is
also the result of the closed runtime schema checker rather than a preservation callback. -/
structure AnalyzerDerivedNextFacts
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
    (contract : Mxx.MxxBoundedSamplerContract samplers) : Prop where
  analyzerSound : ∃ bodyWires : Mxx.Ir.WireEnvironment,
    Mxx.Ir.EvaluatesNodesPath
      (Mxx.Ir.childRunnerWithFuel samplers execution.stage.program step.fuel)
      samplers
      (evaluatedBindings ++
        ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
      (step.body.inputNames.zip (state ++ evidence.invariantValues))
      0 step.body.nodes [] bodyWires ∧
    next = (Mxx.Ir.collectOutputs step.body.outputs bodyWires).map Prod.snd ∧
    step.analyzerFacts.Holds
      (FactEnvironment.ofWireEnvironment
        (evaluatedBindings ++
          ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
        ⟨execution.stage.id⟩ ⟨[evidence.view.definition]⟩ bodyWires)
  carriedFacts : CarriedState.Holds execution.params evidence.transfer.carriedSchemas next

/-- Construct the next-state evidence solely from the frozen child execution and closed checkers. -/
theorem AnalyzerBodyLocalRuleStep.deriveNextFacts
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
    (contract : Mxx.MxxBoundedSamplerContract samplers) :
    AnalyzerDerivedNextFacts step contract := by
  obtain ⟨wires, path, outputValues⟩ := step.bodyExecutionPath
  refine {
    analyzerSound := ⟨wires, path, outputValues, ?_⟩
    carriedFacts := carriedStateAccepted_sound execution.params
      evidence.transfer.carriedSchemas next step.schemaAccepted
  }
  exact inferRulesFrom_sound_closedPrimitives contract path step.bodyClosed
    (by intro target port _; rfl) step.analyzerAccepted
    (ScopedWireFactTable.Holds.nil
      (FactEnvironment.ofWireEnvironment
        (evaluatedBindings ++
          ((.loopIndex evidence.view.indexSlot, .integer index) :: execution.params))
        ⟨execution.stage.id⟩ ⟨[evidence.view.definition]⟩ wires))

/-- Dependent local-rule derivation over the exact executable recurrence trace.  Each `cons`
constructor reuses the binding equality and child-support proof from that trace constructor.
There is no caller-supplied induction predicate or step callback. -/
inductive Derivation
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance)
    (contract : Mxx.MxxBoundedSamplerContract samplers) :
    {indices : List Nat} → {initial final : List Mxx.Ir.Value} →
      (trace : Mxx.Ir.SequentialIterationsTrace execution.rootChildRunner evidence.view.definition
        execution.params evidence.view.indexSlot evidence.view.bindings evidence.invariantValues
        indices initial final) →
      CarriedState.Holds execution.params evidence.transfer.carriedSchemas initial → Type where
  | nil {state : List Mxx.Ir.Value}
      (stateFacts : CarriedState.Holds execution.params evidence.transfer.carriedSchemas state) :
      Derivation evidence contract (.nil state) stateFacts
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
      (stateFacts : CarriedState.Holds execution.params evidence.transfer.carriedSchemas state)
      (derived : AnalyzerDerivedNextFacts step contract)
      (restDerivation : Derivation evidence contract rest
        derived.carriedFacts) :
      Derivation evidence contract
        (.cons index tail state evaluatedBindings next final bindingsEvaluate childMember rest)
        stateFacts

theorem Derivation.initialFacts
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance)
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    {indices : List Nat}
    {initial final : List Mxx.Ir.Value}
    {trace : Mxx.Ir.SequentialIterationsTrace execution.rootChildRunner
      evidence.view.definition
      execution.params evidence.view.indexSlot evidence.view.bindings evidence.invariantValues
      indices initial final}
    {initialFacts : CarriedState.Holds execution.params evidence.transfer.carriedSchemas initial}
    (_derivation : Derivation evidence contract trace initialFacts) :
    CarriedState.Holds execution.params evidence.transfer.carriedSchemas initial := by
  exact initialFacts

theorem Derivation.finalFacts
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance)
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    {indices : List Nat}
    {initial final : List Mxx.Ir.Value}
    {trace : Mxx.Ir.SequentialIterationsTrace execution.rootChildRunner
      evidence.view.definition
      execution.params evidence.view.indexSlot evidence.view.bindings evidence.invariantValues
      indices initial final}
    {initialFacts : CarriedState.Holds execution.params evidence.transfer.carriedSchemas initial}
    (derivation : Derivation evidence contract trace initialFacts) :
    CarriedState.Holds execution.params evidence.transfer.carriedSchemas final := by
  induction derivation with
  | nil stateFacts => exact stateFacts
  | cons _ _ _ _ _ _ restDerivation induction => exact induction

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

/-- Final recurrence result: analyzer-projected facts paired with the actual final carried values
and their closed coarse-schema proof. -/
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
  valuesHold : CarriedState.Holds execution.params evidence.transfer.carriedSchemas values

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
  obtain ⟨scoped, scopedMember, factEq⟩ := List.mem_map.mp mapped
  have holds := projection.scopedFactsHold scoped scopedMember
  rw [← factEq] at holds
  exact holds

def Derivation.finalProjection
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {execution : StageExecution samplers}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundSequentialRecurrence analysis execution recurrenceInstance)
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (environment : FactEnvironment)
    (analysisHolds : BaseAnalysisHolds environment analysis)
    {initialFacts : CarriedState.Holds execution.params evidence.transfer.carriedSchemas
      (evidence.argumentValues.take evidence.view.carriedCount)}
    (derivation : Derivation evidence contract evidence.executionTrace initialFacts) :
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
  valuesHold := derivation.finalFacts
}

end TraceBoundSequentialRecurrence

end Mxx.Certificate
