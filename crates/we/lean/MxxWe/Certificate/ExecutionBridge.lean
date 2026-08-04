import Mxx.Ir.ExecutionFacts
import MxxWe.Certificate.VerifierSound

namespace MxxWe.Certificate

/-- The concrete support member selected at one exact certificate reference.  The certificate
does not supply the node semantics: `resolved` ties the reference to the existing workflow, and
`member` comes from an executable IR path. -/
structure ReferencedNodeExecution
    (workflow : Mxx.Ir.Workflow) (reference : CoreNodeRef)
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment) where
  node : Mxx.Ir.Node
  before : Mxx.Ir.WireEnvironment
  values : List Mxx.Ir.Value
  resolved : resolveNode workflow reference = some node
  member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs before node

/-- The workflow-wide SSA check gives the strict producer-before-consumer fact needed to compose
values recovered from one selected child execution path. -/
theorem verifyScopeSsaOrder_argument_lt
    {scope : Mxx.Ir.Scope} (verified : verifyScopeSsaOrder scope = true)
    (index : Nat) (indexLt : index < scope.nodes.length)
    (argument : Mxx.Ir.WireRef) (member : argument ∈ scope.nodes[index].arguments) :
    argument.node < index := by
  unfold verifyScopeSsaOrder at verified
  simp only [Bool.and_eq_true, List.all_eq_true] at verified
  have nodeChecked :=
    (List.forall_mem_zipIdx'.mp verified.1) index indexLt
  exact of_decide_eq_true (nodeChecked.2 argument member).1

/-- Select one child definition from the workflow-wide SSA-order check. -/
theorem verifyWorkflowSsaOrder_definition
    {workflow : Mxx.Ir.Workflow} (verified : verifyWorkflowSsaOrder workflow = true)
    {stage : Mxx.Ir.Stage} (stageMember : stage ∈ workflow.stages)
    {name : String} {scope : Mxx.Ir.Scope}
    (definitionMember : (name, scope) ∈ stage.program.definitions) :
    verifyScopeSsaOrder scope = true := by
  unfold verifyWorkflowSsaOrder at verified
  simp only [List.all_eq_true, Bool.and_eq_true] at verified
  exact (verified stage stageMember).2 (name, scope) definitionMember

/-- A successful definition lookup identifies an actual member of the program definition list. -/
theorem lookupDefinition_mem
    {name : String} {scope : Mxx.Ir.Scope}
    {definitions : List (String × Mxx.Ir.Scope)}
    (found : Mxx.Ir.lookupDefinition name definitions = some scope) :
    (name, scope) ∈ definitions := by
  induction definitions with
  | nil => simp [Mxx.Ir.lookupDefinition] at found
  | cons head tail induction =>
      rcases head with ⟨candidate, candidateScope⟩
      by_cases same : candidate = name
      · simp [Mxx.Ir.lookupDefinition, same] at found
        subst candidate
        subst candidateScope
        simp
      · simp [Mxx.Ir.lookupDefinition, same] at found
        exact List.mem_cons_of_mem _ (induction found)

/-- A resolved stage is an actual member of the workflow stage list. -/
theorem resolveStage_mem
    {workflow : Mxx.Ir.Workflow} {name : String} {stage : Mxx.Ir.Stage}
    (found : resolveStage workflow name = some stage) :
    stage ∈ workflow.stages := by
  unfold resolveStage at found
  exact List.mem_of_find?_eq_some found

/-- Resolve a node lookup inside an already resolved scope. -/
theorem resolveNode_scopeNode
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {scope : Mxx.Ir.Scope} {node : Mxx.Ir.Node}
    (scopeResolved : resolveScope workflow reference = some scope)
    (nodeResolved : resolveNode workflow reference = some node) :
    scope.nodes[reference.node]? = some node := by
  unfold resolveNode at nodeResolved
  simpa [scopeResolved] using nodeResolved

/-- Every successful node resolution also resolves its enclosing structural scope. -/
theorem resolveScope_of_resolveNode_some
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef} {node : Mxx.Ir.Node}
    (resolved : resolveNode workflow reference = some node) :
    ∃ scope, resolveScope workflow reference = some scope := by
  unfold resolveNode at resolved
  cases scopeResolved : resolveScope workflow reference with
  | none => simp [scopeResolved] at resolved
  | some scope => exact ⟨scope, rfl⟩

/-- One selected Boolean body outcome with its single SSA path retained.  Keeping this path, rather
than only independent node executions, lets later proofs transport an earlier output wire into the
argument environment of every certified consumer. -/
structure ChildExecutionPath
    (stage : Mxx.Ir.Stage) (scope : Mxx.Ir.Scope) (fuel : Nat)
    (samplers : Mxx.MxxSamplerFamily) (params : Mxx.Ir.ParamEnvironment)
    (inputs values : List Mxx.Ir.Value) where
  finalWires : Mxx.Ir.WireEnvironment
  path : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
    (scope.inputNames.zip inputs) 0 scope.nodes [] finalWires
  outputs : values = (Mxx.Ir.collectOutputs scope.outputs finalWires).map Prod.snd

/-- Recover the retained path for one exact child-support member. -/
theorem childExecutionPath_of_outcome
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value} {definition : String}
    (definitionFound :
      Mxx.Ir.lookupDefinition definition stage.program.definitions = some scope)
    (member : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      definition params inputs) :
    Nonempty (ChildExecutionPath stage scope fuel samplers params inputs values) := by
  obtain ⟨finalWires, path, outputs⟩ :=
    (Mxx.Ir.mem_childRunnerWithFuel_succ_iff_path samplers stage.program fuel definition scope
      params inputs values definitionFound).mp member
  exact ⟨{ finalWires, path, outputs }⟩

/-- Extract one certificate-referenced node from the retained child path. -/
theorem ChildExecutionPath.referencedNodeExecution
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value} {node : Mxx.Ir.Node}
    (execution : ChildExecutionPath stage scope fuel samplers params inputs values)
    (scopeResolved : scope.nodes[reference.node]? = some node)
    (workflowResolved : resolveNode workflow reference = some node) :
    Nonempty (ReferencedNodeExecution workflow reference
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs)) := by
  have inBounds : reference.node < scope.nodes.length := by
    by_contra outOfBounds
    rw [List.getElem?_eq_none (Nat.le_of_not_gt outOfBounds)] at scopeResolved
    contradiction
  have nodeEq : scope.nodes[reference.node] = node := by
    rw [List.getElem?_eq_getElem inBounds] at scopeResolved
    exact Option.some.inj scopeResolved
  obtain ⟨before, nodeValues, beforePath, nodeMember, afterPath⟩ :=
    execution.path.atNodeIndex reference.node inBounds
  exact ⟨{
    node
    before
    values := nodeValues
    resolved := workflowResolved
    member := by simpa [nodeEq] using nodeMember
  }⟩

/-- Outputs bound by one node cannot satisfy a lookup for a different node identifier. -/
theorem lookupWire_bindOutputs_of_node_ne
    {nodeId other port : Nat} {values : List Mxx.Ir.Value}
    (different : other ≠ nodeId) :
    Mxx.Ir.lookupWire ⟨other, port⟩ (Mxx.Ir.bindOutputs nodeId values) = none := by
  unfold Mxx.Ir.bindOutputs
  generalize values.zipIdx = entries
  induction entries with
  | nil => rfl
  | cons entry entries induction =>
      rcases entry with ⟨value, outputPort⟩
      simp only [List.map_cons, Mxx.Ir.lookupWire]
      split
      · rename_i equal
        exact (different (congrArg Mxx.Ir.WireRef.node equal).symm).elim
      · exact induction

/-- Nodes with larger SSA identifiers cannot change an earlier wire lookup. -/
theorem Mxx.Ir.EvaluatesNodesPath.pastWire_eq
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    {nodeId : Nat} {nodes : List Mxx.Ir.Node}
    {initial output : Mxx.Ir.WireEnvironment}
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs nodeId nodes initial output)
    (wire : Mxx.Ir.WireRef) (past : wire.node < nodeId) :
    Mxx.Ir.lookupWire wire output = Mxx.Ir.lookupWire wire initial := by
  induction path with
  | nil => rfl
  | cons current node tail state values final member rest induction =>
      rw [induction (by omega)]
      cases lookup : Mxx.Ir.lookupWire wire state with
      | none =>
          rw [Mxx.Ir.lookupWire_append_of_eq_none lookup]
          exact lookupWire_bindOutputs_of_node_ne (by omega)
      | some value => exact Mxx.Ir.lookupWire_append_of_eq_some lookup

/-- Before the next node starts, its output wires are absent from an execution path. -/
theorem evaluatesNodesPath_nextNode_missing
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    {nodeId : Nat} {nodes : List Mxx.Ir.Node}
    {initial output : Mxx.Ir.WireEnvironment}
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs nodeId nodes initial output)
    (port : Nat)
    (initialMissing : Mxx.Ir.lookupWire ⟨nodeId + nodes.length, port⟩ initial = none) :
    Mxx.Ir.lookupWire ⟨nodeId + nodes.length, port⟩ output = none := by
  induction path with
  | nil => exact initialMissing
  | cons current node tail state values final member rest induction =>
      have targetEq : current + (node :: tail).length = current + 1 + tail.length := by
        simp [Nat.add_comm, Nat.add_left_comm]
      rw [targetEq] at initialMissing ⊢
      apply induction
      rw [Mxx.Ir.lookupWire_append_of_eq_none initialMissing]
      exact lookupWire_bindOutputs_of_node_ne (by omega)

/-- A referenced execution retained together with its exact position on the selected child path.
This prevents combining a different nondeterministic support member with the path's outputs. -/
structure ChildPathRootedNodeExecution
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value}
    (path : ChildExecutionPath stage scope fuel samplers params inputs values)
    (execution : ReferencedNodeExecution workflow reference
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs)) : Prop where
  pathPrefix : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
    (scope.inputNames.zip inputs) 0 (scope.nodes.take reference.node) [] execution.before
  pathSuffix : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
    (scope.inputNames.zip inputs) (reference.node + 1)
    (scope.nodes.drop (reference.node + 1))
    (execution.before ++ Mxx.Ir.bindOutputs reference.node execution.values) path.finalWires
  outputFinal : ∀ port, ∀ portValid : port < execution.values.length,
    Mxx.Ir.lookupWire ⟨reference.node, port⟩ path.finalWires =
      some (execution.values.get ⟨port, portValid⟩)

/-- Extract a node execution without forgetting that it belongs to this exact child path. -/
theorem ChildExecutionPath.rootedReferencedNodeExecution
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value} {node : Mxx.Ir.Node}
    (path : ChildExecutionPath stage scope fuel samplers params inputs values)
    (scopeResolved : scope.nodes[reference.node]? = some node)
    (workflowResolved : resolveNode workflow reference = some node) :
    ∃ execution : ReferencedNodeExecution workflow reference
        (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
        (scope.inputNames.zip inputs),
      ChildPathRootedNodeExecution path execution := by
  have inBounds : reference.node < scope.nodes.length := by
    by_contra outOfBounds
    rw [List.getElem?_eq_none (Nat.le_of_not_gt outOfBounds)] at scopeResolved
    contradiction
  have nodeEq : scope.nodes[reference.node] = node := by
    rw [List.getElem?_eq_getElem inBounds] at scopeResolved
    exact Option.some.inj scopeResolved
  obtain ⟨before, nodeValues, beforePath, nodeMember, afterPath⟩ :=
    path.path.atNodeIndex reference.node inBounds
  let execution : ReferencedNodeExecution workflow reference
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs) := {
    node
    before
    values := nodeValues
    resolved := workflowResolved
    member := by simpa [nodeEq] using nodeMember
  }
  refine ⟨execution, {
    pathPrefix := beforePath
    pathSuffix := by simpa [execution] using afterPath
    outputFinal := ?_
  }⟩
  intro port portValid
  apply afterPath.lookupWire_preserved
  have missing : Mxx.Ir.lookupWire ⟨reference.node, port⟩ before = none := by
    have absent := evaluatesNodesPath_nextNode_missing beforePath port
      (by simp [Mxx.Ir.lookupWire])
    simpa [List.length_take, Nat.min_eq_left (Nat.le_of_lt inBounds)] using absent
  simpa [execution] using Mxx.Ir.lookupWire_append_bindOutputs missing portValid

/-- A value visible before a rooted node remains visible in the retained child's final
environment. -/
theorem ChildPathRootedNodeExecution.beforeFinal
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value}
    {path : ChildExecutionPath stage scope fuel samplers params inputs values}
    {execution : ReferencedNodeExecution workflow reference
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs)}
    (rooted : ChildPathRootedNodeExecution path execution)
    {wire : Mxx.Ir.WireRef} {value : Mxx.Ir.Value}
    (resolved : Mxx.Ir.lookupWire wire execution.before = some value) :
    Mxx.Ir.lookupWire wire path.finalWires = some value := by
  apply rooted.pathSuffix.lookupWire_preserved
  exact Mxx.Ir.lookupWire_append_of_eq_some resolved

/-- An earlier SSA wire visible at the end of the retained child was already visible before the
rooted consumer. -/
theorem ChildPathRootedNodeExecution.finalBefore
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value}
    {path : ChildExecutionPath stage scope fuel samplers params inputs values}
    {execution : ReferencedNodeExecution workflow reference
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs)}
    (rooted : ChildPathRootedNodeExecution path execution)
    {wire : Mxx.Ir.WireRef} {value : Mxx.Ir.Value} (past : wire.node < reference.node)
    (resolved : Mxx.Ir.lookupWire wire path.finalWires = some value) :
    Mxx.Ir.lookupWire wire execution.before = some value := by
  have atNode := resolved
  rw [Mxx.Ir.EvaluatesNodesPath.pastWire_eq rooted.pathSuffix wire (by omega)] at atNode
  cases beforeLookup : Mxx.Ir.lookupWire wire execution.before with
  | some found =>
      rw [Mxx.Ir.lookupWire_append_of_eq_some beforeLookup] at atNode
      have foundEq : found = value := Option.some.inj atNode
      subst found
      rfl
  | none =>
      rw [Mxx.Ir.lookupWire_append_of_eq_none beforeLookup,
        lookupWire_bindOutputs_of_node_ne (by omega)] at atNode
      contradiction

/-- The exact output of an earlier rooted producer is the value read by a later rooted
consumer. -/
theorem ChildPathRootedNodeExecution.outputAtConsumer
    {workflow : Mxx.Ir.Workflow} {producerRef consumerRef : CoreNodeRef}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value}
    {path : ChildExecutionPath stage scope fuel samplers params inputs values}
    {producer : ReferencedNodeExecution workflow producerRef
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs)}
    {consumer : ReferencedNodeExecution workflow consumerRef
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs)}
    (producerRooted : ChildPathRootedNodeExecution path producer)
    (consumerRooted : ChildPathRootedNodeExecution path consumer)
    (port : Nat) (portValid : port < producer.values.length)
    (ordered : producerRef.node < consumerRef.node) :
    Mxx.Ir.lookupWire ⟨producerRef.node, port⟩ consumer.before =
      some (producer.values.get ⟨port, portValid⟩) :=
  consumerRooted.finalBefore ordered (producerRooted.outputFinal port portValid)

/-- Any SSA argument whose exact value is known in the retained final environment has that same
value in the rooted consumer's `before` environment. -/
theorem ChildPathRootedNodeExecution.argumentFromFinal
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value}
    {path : ChildExecutionPath stage scope fuel samplers params inputs values}
    {execution : ReferencedNodeExecution workflow reference
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs)}
    (rooted : ChildPathRootedNodeExecution path execution)
    (ssaOrder : verifyScopeSsaOrder scope = true)
    (scopeResolved : resolveScope workflow reference = some scope)
    (wire : Mxx.Ir.WireRef) (argumentMember : wire ∈ execution.node.arguments)
    {value : Mxx.Ir.Value} (resolved : Mxx.Ir.lookupWire wire path.finalWires = some value) :
    Mxx.Ir.lookupWire wire execution.before = some value := by
  have consumerInScope := resolveNode_scopeNode scopeResolved execution.resolved
  have consumerLt : reference.node < scope.nodes.length := by
    by_contra outOfBounds
    rw [List.getElem?_eq_none (Nat.le_of_not_gt outOfBounds)] at consumerInScope
    contradiction
  have consumerNode : scope.nodes[reference.node] = execution.node := by
    rw [List.getElem?_eq_getElem consumerLt] at consumerInScope
    exact Option.some.inj consumerInScope
  have past := verifyScopeSsaOrder_argument_lt ssaOrder reference.node consumerLt wire
    (by simpa [consumerNode] using argumentMember)
  exact rooted.finalBefore past resolved

/-- On one SSA path, every output of an earlier node is the exact value read by any later
consumer.  This is the value-composition fact used for all certified Boolean gate wires. -/
theorem evaluatesNodesPath_orderedNodeOutput
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    {nodes : List Mxx.Ir.Node} {final : Mxx.Ir.WireEnvironment}
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs 0 nodes [] final)
    (producer consumer : Nat) (ordered : producer < consumer)
    (consumerLt : consumer < nodes.length) :
    ∃ producerBefore producerValues consumerBefore consumerValues,
      producerValues ∈ Mxx.Ir.evaluateNode runChild samplers params inputs producerBefore
        nodes[producer] ∧
      consumerValues ∈ Mxx.Ir.evaluateNode runChild samplers params inputs consumerBefore
        nodes[consumer] ∧
      ∀ (port : Nat) (portLt : port < producerValues.length),
        Mxx.Ir.lookupWire ⟨producer, port⟩ consumerBefore = some producerValues[port] := by
  obtain ⟨consumerBefore, consumerValues, beforeConsumer, consumerMember, afterConsumer⟩ :=
    path.atNodeIndex consumer consumerLt
  have producerLt : producer < (nodes.take consumer).length := by
    simp [ordered, Nat.le_of_lt consumerLt]
  obtain ⟨producerBefore, producerValues, beforeProducer, producerMember, afterProducer⟩ :=
    beforeConsumer.atNodeIndex producer producerLt
  refine ⟨producerBefore, producerValues, consumerBefore, consumerValues, ?_, ?_, ?_⟩
  · simpa [List.getElem_take] using producerMember
  · exact consumerMember
  · intro port portLt
    apply afterProducer.lookupWire_preserved
    have missing : Mxx.Ir.lookupWire ⟨producer, port⟩ producerBefore = none := by
      have absent := evaluatesNodesPath_nextNode_missing beforeProducer port
        (by simp [Mxx.Ir.lookupWire])
      have producerLeMin : producer ≤ min consumer nodes.length :=
        Nat.le_min.mpr ⟨Nat.le_of_lt ordered,
          le_trans (Nat.le_of_lt ordered) (Nat.le_of_lt consumerLt)⟩
      simpa [Nat.min_eq_left producerLeMin] using absent
    simpa using Mxx.Ir.lookupWire_append_bindOutputs missing portLt

/-- A successful index lookup supplies the corresponding list bound. -/
theorem list_index_lt_of_getElem?_eq_some
    {α : Type} {values : List α} {index : Nat} {value : α}
    (resolved : values[index]? = some value) :
    index < values.length := by
  by_contra outOfBounds
  rw [List.getElem?_eq_none (Nat.le_of_not_gt outOfBounds)] at resolved
  contradiction

/-- Convert an optional successful list lookup into its bounded `getElem` equality. -/
theorem list_getElem_eq_of_getElem?_eq_some
    {α : Type} {values : List α} {index : Nat} {value : α}
    (resolved : values[index]? = some value) (indexLt : index < values.length) :
    values[index] = value := by
  rw [List.getElem?_eq_getElem indexLt] at resolved
  exact Option.some.inj resolved

/-- Compose a certified producer and consumer directly on the retained child path.  The
SSA checker supplies ordering; the executable path supplies both node support members and the
exact value observed at the consumer. -/
theorem ChildExecutionPath.outputAtConsumer
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value}
    (execution : ChildExecutionPath stage scope fuel samplers params inputs values)
    (ssaOrder : verifyScopeSsaOrder scope = true)
    (producer consumer : Nat) (producerNode consumerNode : Mxx.Ir.Node)
    (producerResolved : scope.nodes[producer]? = some producerNode)
    (consumerResolved : scope.nodes[consumer]? = some consumerNode)
    (argument : Mxx.Ir.WireRef) (argumentNode : argument.node = producer)
    (argumentMember : argument ∈ consumerNode.arguments) :
    ∃ producerBefore producerValues consumerBefore consumerValues,
      producerValues ∈ Mxx.Ir.evaluateNode
        (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
        (scope.inputNames.zip inputs) producerBefore producerNode ∧
      consumerValues ∈ Mxx.Ir.evaluateNode
        (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
        (scope.inputNames.zip inputs) consumerBefore consumerNode ∧
      ∀ (port : Nat) (portLt : port < producerValues.length),
        Mxx.Ir.lookupWire ⟨producer, port⟩ consumerBefore = some producerValues[port] := by
  have producerLt := list_index_lt_of_getElem?_eq_some producerResolved
  have consumerLt := list_index_lt_of_getElem?_eq_some consumerResolved
  have producerNodeEq : scope.nodes[producer] = producerNode := by
    rw [List.getElem?_eq_getElem producerLt] at producerResolved
    exact Option.some.inj producerResolved
  have consumerNodeEq : scope.nodes[consumer] = consumerNode := by
    rw [List.getElem?_eq_getElem consumerLt] at consumerResolved
    exact Option.some.inj consumerResolved
  have ordered : producer < consumer := by
    have argumentOrdered := verifyScopeSsaOrder_argument_lt ssaOrder consumer consumerLt argument
      (by simpa [consumerNodeEq] using argumentMember)
    simpa [argumentNode] using argumentOrdered
  obtain ⟨producerBefore, producerValues, consumerBefore, consumerValues,
      producerMember, consumerMember, observed⟩ :=
    evaluatesNodesPath_orderedNodeOutput execution.path producer consumer ordered consumerLt
  exact ⟨producerBefore, producerValues, consumerBefore, consumerValues,
    by simpa [producerNodeEq] using producerMember,
    by simpa [consumerNodeEq] using consumerMember, observed⟩

/-- Specialize path composition to a producer whose exact executable outcome is one value. -/
theorem ChildExecutionPath.singletonOutputAtConsumer
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value}
    (execution : ChildExecutionPath stage scope fuel samplers params inputs values)
    (ssaOrder : verifyScopeSsaOrder scope = true)
    (producer consumer : Nat) (producerNode consumerNode : Mxx.Ir.Node)
    (producerResolved : scope.nodes[producer]? = some producerNode)
    (consumerResolved : scope.nodes[consumer]? = some consumerNode)
    (argument : Mxx.Ir.WireRef) (argumentNode : argument.node = producer)
    (argumentMember : argument ∈ consumerNode.arguments)
    (value : Mxx.Ir.Value)
    (producerOutcome : ∀ before producerValues,
      producerValues ∈ Mxx.Ir.evaluateNode
        (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
        (scope.inputNames.zip inputs) before producerNode → producerValues = [value]) :
    ∃ consumerBefore consumerValues,
      consumerValues ∈ Mxx.Ir.evaluateNode
        (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
        (scope.inputNames.zip inputs) consumerBefore consumerNode ∧
      Mxx.Ir.lookupWire ⟨producer, 0⟩ consumerBefore = some value := by
  obtain ⟨producerBefore, producerValues, consumerBefore, consumerValues, producerMember,
      consumerMember, observed⟩ :=
    execution.outputAtConsumer ssaOrder producer consumer producerNode consumerNode
      producerResolved consumerResolved argument argumentNode argumentMember
  have valuesEq := producerOutcome producerBefore producerValues producerMember
  subst producerValues
  exact ⟨consumerBefore, consumerValues, consumerMember, observed 0 (by simp)⟩

/-- When `filterMap` preserves length, its value at an index comes from the source value at the
same index. -/
private theorem filterMap_getElem?_of_length_eq
    {α β : Type} (f : α → Option β) (source : List α)
    (lengthEq : (source.filterMap f).length = source.length)
    (index : Nat) (value : α) (sourceAt : source[index]? = some value) :
    ∃ mapped, f value = some mapped ∧ (source.filterMap f)[index]? = some mapped := by
  induction source generalizing index with
  | nil => simp at sourceAt
  | cons head tail induction =>
      cases mappedHead : f head with
      | none =>
          simp only [List.filterMap_cons_none mappedHead] at lengthEq
          have short := List.length_filterMap_le f tail
          simp only [List.length_cons] at lengthEq
          omega
      | some mappedHeadValue =>
          simp only [List.filterMap_cons_some mappedHead] at lengthEq
          simp only [List.length_cons] at lengthEq
          have tailLength : (tail.filterMap f).length = tail.length := by omega
          cases index with
          | zero =>
              simp only [List.getElem?_cons_zero, Option.some.injEq] at sourceAt
              subst value
              exact ⟨mappedHeadValue, mappedHead, by simp [mappedHead]⟩
          | succ index =>
              simp only [List.getElem?_cons_succ] at sourceAt
              obtain ⟨mapped, valueMapped, outputAt⟩ :=
                induction tailLength index sourceAt
              exact ⟨mapped, valueMapped, by simpa [mappedHead] using outputAt⟩

/-- Distinct zipped names map an indexed name to the value at the same index. -/
private theorem lookupEnvironment_zip_getElem?
    (names : List String) (values : List Mxx.Ir.Value)
    (namesNodup : names.Nodup) (lengthEq : names.length = values.length)
    (index : Nat) (name : String) (value : Mxx.Ir.Value)
    (nameAt : names[index]? = some name) (valueAt : values[index]? = some value) :
    Mxx.Ir.lookupEnvironment name (names.zip values) = some value := by
  induction names generalizing values index with
  | nil => simp at nameAt
  | cons head tail induction =>
      cases values with
      | nil => simp at lengthEq
      | cons first rest =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEq
          have tailNodup : tail.Nodup := namesNodup.tail
          cases index with
          | zero =>
              simp only [List.getElem?_cons_zero, Option.some.injEq] at nameAt valueAt
              subst name
              subst value
              simp [Mxx.Ir.lookupEnvironment]
          | succ index =>
              simp only [List.getElem?_cons_succ] at nameAt valueAt
              have nameMem : name ∈ tail := List.mem_of_getElem? nameAt
              have different : head ≠ name := by
                intro equal
                subst name
                exact (List.nodup_cons.mp namesNodup).1 nameMem
              simp [Mxx.Ir.lookupEnvironment, different]
              exact induction rest tailNodup lengthEq index nameAt valueAt

/-- A structural body-input wire identifies the input node for the corresponding input name. -/
theorem scopeInputWire_resolves
    {scope : Mxx.Ir.Scope} (lengthEq : (scopeInputWires scope).length = scope.inputNames.length)
    (index : Nat) (name : String) (wire : Mxx.Ir.WireRef)
    (nameAt : scope.inputNames[index]? = some name)
    (wireAt : (scopeInputWires scope)[index]? = some wire) :
    ∃ entry : Mxx.Ir.Node × Nat,
      scope.nodes.zipIdx.find? (fun candidate => match candidate.1.kind with
        | .input actual => actual = name
        | _ => false) = some entry ∧
      entry.1.kind = .input name ∧ wire = ⟨entry.2, 0⟩ := by
  let findInput := fun inputName : String =>
    (scope.nodes.zipIdx.find? fun entry => match entry.1.kind with
      | .input actual => actual = inputName
      | _ => false).map fun entry => ({ node := entry.2, port := 0 } : Mxx.Ir.WireRef)
  have mapped := filterMap_getElem?_of_length_eq findInput scope.inputNames
    (by
      have filterEq : scope.inputNames.filterMap findInput = scopeInputWires scope := rfl
      rw [filterEq]
      exact lengthEq)
    index name nameAt
  obtain ⟨mappedWire, foundMapped, mappedAt⟩ := mapped
  have wireEq : mappedWire = wire := by
    rw [show scope.inputNames.filterMap findInput = scopeInputWires scope by
      rfl] at mappedAt
    rw [wireAt] at mappedAt
    exact (Option.some.inj mappedAt).symm
  cases found : scope.nodes.zipIdx.find? (fun entry => match entry.1.kind with
      | .input actual => actual = name
      | _ => false) with
  | none => simp [findInput, found] at foundMapped
  | some entry =>
      have entryInput := List.find?_some found
      have kind : entry.1.kind = .input name := by
        cases entryKind : entry.1.kind <;> simp_all
      have mappedWireEq : ({ node := entry.2, port := 0 } : Mxx.Ir.WireRef) = mappedWire := by
        simpa [findInput, found] using foundMapped
      exact ⟨entry, rfl, kind, wireEq.symm.trans mappedWireEq.symm⟩

/-- A checked structural input wire carries the corresponding child argument on the retained
execution path. -/
theorem ChildExecutionPath.inputWireValue
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value}
    (path : ChildExecutionPath stage scope fuel samplers params inputs values)
    (namesNodup : scope.inputNames.Nodup)
    (wireLength : (scopeInputWires scope).length = scope.inputNames.length)
    (inputLength : scope.inputNames.length = inputs.length)
    (index : Nat) (name : String) (wire : Mxx.Ir.WireRef) (value : Mxx.Ir.Value)
    (nameAt : scope.inputNames[index]? = some name)
    (wireAt : (scopeInputWires scope)[index]? = some wire)
    (valueAt : inputs[index]? = some value)
    (wireValid : ∃ node, scope.nodes[wire.node]? = some node ∧ wire.port < node.outputCount) :
    Mxx.Ir.lookupWire wire path.finalWires = some value := by
  obtain ⟨entry, entryFound, entryKind, wireEq⟩ :=
    scopeInputWire_resolves wireLength index name wire nameAt wireAt
  have entryMember := List.mem_of_find?_eq_some entryFound
  have entryFacts := List.mem_zipIdx' entryMember
  have entryLt : entry.2 < scope.nodes.length := entryFacts.1
  have entryNode : scope.nodes[entry.2] = entry.1 := entryFacts.2.symm
  obtain ⟨before, nodeValues, beforePath, nodeMember, afterPath⟩ :=
    path.path.atNodeIndex entry.2 entryLt
  have inputLookup := lookupEnvironment_zip_getElem? scope.inputNames inputs namesNodup
    inputLength index name value nameAt valueAt
  have nodeValuesEq : nodeValues = List.replicate entry.1.outputCount value := by
    simpa [Mxx.Ir.evaluateNode, entryNode, entryKind, inputLookup] using nodeMember
  rcases wireValid with ⟨wireNode, wireNodeAt, portLt⟩
  have wireNodeEq : wireNode = entry.1 := by
    rw [wireEq] at wireNodeAt
    rw [List.getElem?_eq_getElem entryLt] at wireNodeAt
    have exactNode := Option.some.inj wireNodeAt
    simpa [entryNode] using exactNode.symm
  subst wireNode
  rw [wireEq]
  apply afterPath.lookupWire_preserved
  have missing : Mxx.Ir.lookupWire ⟨entry.2, 0⟩ before = none := by
    have absent := evaluatesNodesPath_nextNode_missing beforePath 0
      (by simp [Mxx.Ir.lookupWire])
    have minEq : min entry.2 scope.nodes.length = entry.2 :=
      Nat.min_eq_left (Nat.le_of_lt entryLt)
    simpa only [List.length_take, Nat.zero_add, minEq] using absent
  rw [nodeValuesEq]
  have zeroLt : 0 < entry.1.outputCount := by simpa [wireEq] using portLt
  have repeatedLt : 0 < (List.replicate entry.1.outputCount value).length := by
    simpa using zeroLt
  simpa using Mxx.Ir.lookupWire_append_bindOutputs
    (values := List.replicate entry.1.outputCount value) missing repeatedLt

/-- The corresponding child argument is already present in the exact `before` environment of a
later rooted consumer on the same path. -/
theorem ChildPathRootedNodeExecution.inputWireValue
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value}
    {path : ChildExecutionPath stage scope fuel samplers params inputs values}
    {execution : ReferencedNodeExecution workflow reference
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs)}
    (rooted : ChildPathRootedNodeExecution path execution)
    (ssaOrder : verifyScopeSsaOrder scope = true)
    (scopeResolved : resolveScope workflow reference = some scope)
    (namesNodup : scope.inputNames.Nodup)
    (wireLength : (scopeInputWires scope).length = scope.inputNames.length)
    (inputLength : scope.inputNames.length = inputs.length)
    (index : Nat) (name : String) (wire : Mxx.Ir.WireRef) (value : Mxx.Ir.Value)
    (nameAt : scope.inputNames[index]? = some name)
    (wireAt : (scopeInputWires scope)[index]? = some wire)
    (valueAt : inputs[index]? = some value)
    (argumentMember : wire ∈ execution.node.arguments)
    (wireValid : ∃ node, scope.nodes[wire.node]? = some node ∧ wire.port < node.outputCount) :
    Mxx.Ir.lookupWire wire execution.before = some value := by
  obtain ⟨entry, entryFound, entryKind, wireEq⟩ :=
    scopeInputWire_resolves wireLength index name wire nameAt wireAt
  have entryMember := List.mem_of_find?_eq_some entryFound
  have entryFacts := List.mem_zipIdx' entryMember
  have entryNode : scope.nodes[entry.2] = entry.1 := entryFacts.2.symm
  have consumerInScope := resolveNode_scopeNode scopeResolved execution.resolved
  have consumerLt := list_index_lt_of_getElem?_eq_some consumerInScope
  have consumerNode : scope.nodes[reference.node] = execution.node :=
    list_getElem_eq_of_getElem?_eq_some consumerInScope consumerLt
  have entryBeforeConsumer : entry.2 < reference.node := by
    have ordered := verifyScopeSsaOrder_argument_lt ssaOrder reference.node consumerLt wire
      (by simpa [consumerNode] using argumentMember)
    simpa [wireEq] using ordered
  have entryLt : entry.2 < (scope.nodes.take reference.node).length := by
    simp [entryBeforeConsumer, Nat.le_of_lt consumerLt]
  obtain ⟨before, nodeValues, beforePath, nodeMember, afterPath⟩ :=
    rooted.pathPrefix.atNodeIndex entry.2 entryLt
  have inputLookup := lookupEnvironment_zip_getElem? scope.inputNames inputs namesNodup
    inputLength index name value nameAt valueAt
  have nodeValuesEq : nodeValues = List.replicate entry.1.outputCount value := by
    simpa [List.getElem_take, entryNode, Mxx.Ir.evaluateNode, entryKind, inputLookup] using
      nodeMember
  rcases wireValid with ⟨wireNode, wireNodeAt, portLt⟩
  have wireNodeEq : wireNode = entry.1 := by
    rw [wireEq] at wireNodeAt
    rw [List.getElem?_eq_getElem entryFacts.1] at wireNodeAt
    have exactNode := Option.some.inj wireNodeAt
    simpa [entryNode] using exactNode.symm
  subst wireNode
  rw [wireEq]
  apply afterPath.lookupWire_preserved
  have missing : Mxx.Ir.lookupWire ⟨entry.2, 0⟩ before = none := by
    have absent := evaluatesNodesPath_nextNode_missing beforePath 0
      (by simp [Mxx.Ir.lookupWire])
    have entryLtMin : entry.2 < min reference.node scope.nodes.length := by
      simpa only [List.length_take] using entryLt
    have minEq : min entry.2 (min reference.node scope.nodes.length) = entry.2 :=
      Nat.min_eq_left (Nat.le_of_lt entryLtMin)
    simpa only [List.length_take, Nat.zero_add, minEq] using absent
  rw [nodeValuesEq]
  have zeroLt : 0 < entry.1.outputCount := by simpa [wireEq] using portLt
  have repeatedLt : 0 < (List.replicate entry.1.outputCount value).length := by
    simpa using zeroLt
  simpa using Mxx.Ir.lookupWire_append_bindOutputs
    (values := List.replicate entry.1.outputCount value) missing repeatedLt

/-- A verifier-accepted wire resolves to an in-bounds output port in its resolved scope. -/
theorem verifyWire_scopeValid
    {workflow : Mxx.Ir.Workflow} {reference : CoreWireRef} {scope : Mxx.Ir.Scope}
    (verified : verifyWire workflow reference = true)
    (scopeResolved : resolveScope workflow reference.node = some scope) :
    ∃ node, scope.nodes[reference.node.node]? = some node ∧ reference.port < node.outputCount := by
  unfold verifyWire at verified
  cases nodeResolved : resolveNode workflow reference.node with
  | none => simp [nodeResolved] at verified
  | some node =>
      have inScope := resolveNode_scopeNode scopeResolved nodeResolved
      exact ⟨node, inScope, by simpa [nodeResolved] using verified⟩

/-- A checked parallel boundary identifies its body input and sole output wires exactly. -/
theorem verifyParallelBoundary_bodyBindings
    {workflow : Mxx.Ir.Workflow} {operation : CoreNodeRef} {bodyScope : ScopeRef}
    {outer : List CoreOperandRef} {inner : List CoreWireRef}
    {bodyOutput output : CoreWireRef} {body : Mxx.Ir.Scope}
    (verified : verifyParallelBoundary workflow operation bodyScope outer inner bodyOutput
      output = true)
    (bodyResolved : resolveScope workflow { operation with scope := bodyScope } = some body) :
    body.inputNames.Nodup ∧
      (scopeInputWires body).length = body.inputNames.length ∧
      outer.length = body.inputNames.length ∧
      inner.map wireRef = scopeInputWires body ∧
      body.outputs.map Prod.snd = [wireRef bodyOutput] := by
  unfold verifyParallelBoundary at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  cases nodeResolved : resolveNode workflow operation with
  | none => simp [nodeResolved, bodyResolved] at verified
  | some node =>
      rw [nodeResolved, bodyResolved] at verified
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> simp_all [scopeOutputWires]
      rename_i definition count indexSlot bindings inputModes
      have innerLength : inner.length = body.inputNames.length := by aesop
      aesop

/-- Assemble the interpreter's two-argument lookup from exact per-wire observations. -/
theorem lookupWirePair
    {leftRef rightRef : Mxx.Ir.WireRef} {left right : Mxx.Ir.Value}
    {wires : Mxx.Ir.WireEnvironment}
    (leftLookup : Mxx.Ir.lookupWire leftRef wires = some left)
    (rightLookup : Mxx.Ir.lookupWire rightRef wires = some right) :
    [leftRef, rightRef].mapM (fun wire ↦ Mxx.Ir.lookupWire wire wires) =
      some [left, right] := by
  simp [leftLookup, rightLookup]


/-- Invert an exact root-scope reference from one selected stage outcome.  No generated node
number occurs in the theorem; the index is carried by the checked `CoreNodeRef`. -/
theorem rootNodeExecution_of_stageOutcome
    (workflow : Mxx.Ir.Workflow) (reference : CoreNodeRef) (stage : Mxx.Ir.Stage)
    (samplers : Mxx.MxxSamplerFamily) (params : Mxx.Ir.ParamEnvironment)
    (inputs output : Mxx.Ir.Environment) (node : Mxx.Ir.Node)
    (rootScope : reference.scope = .root)
    (stageResolved : resolveStage workflow reference.stage = some stage)
    (nodeResolved : resolveNode workflow reference = some node)
    (member : output ∈ Mxx.Ir.denote samplers stage.program params inputs) :
    Nonempty (ReferencedNodeExecution workflow reference
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs) := by
  rcases reference with ⟨referenceStage, referenceScope, referenceNode⟩
  dsimp at rootScope
  subst referenceScope
  have rawRootResolved :
      rawScope workflow referenceStage .root = some stage.program.root := by
    simp [rawScope, stageResolved]
  have nodeAt : stage.program.root.nodes[referenceNode]? = some node := by
    simpa [resolveNode, resolveScope, scopeOwnerMatches, rawRootResolved] using nodeResolved
  have inBounds : referenceNode < stage.program.root.nodes.length := by
    by_contra outOfBounds
    have missing : stage.program.root.nodes[referenceNode]? = none := by
      exact List.getElem?_eq_none (Nat.le_of_not_gt outOfBounds)
    rw [missing] at nodeAt
    contradiction
  have nodeEq : stage.program.root.nodes[referenceNode] = node := by
    rw [List.getElem?_eq_getElem inBounds] at nodeAt
    exact Option.some.inj nodeAt
  obtain ⟨before, values, valuesMember⟩ :=
    Mxx.Ir.rootNodeAt_of_mem_denote samplers stage.program params inputs output referenceNode
      inBounds member
  exact ⟨{
    node
    before
    values
    resolved := nodeResolved
    member := by simpa [nodeEq] using valuesMember
  }⟩

/-- Invert an exact node in a structural child scope from one selected child execution.  The
caller obtains `childMember` from the certified parent loop trace; the structural definition name
and node reference are never rediscovered by content search. -/
theorem nestedNodeExecution_of_childOutcome
    (workflow : Mxx.Ir.Workflow) (reference : CoreNodeRef) (stage : Mxx.Ir.Stage)
    (scope : Mxx.Ir.Scope) (fuel : Nat) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs values : List Mxx.Ir.Value)
    (node : Mxx.Ir.Node)
    (definitionResolved :
      Mxx.Ir.lookupDefinition reference.scope.definitionName stage.program.definitions =
        some scope)
    (scopeResolved : resolveScope workflow reference = some scope)
    (nodeResolved : resolveNode workflow reference = some node)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.scope.definitionName params inputs) :
    Nonempty (ReferencedNodeExecution workflow reference
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs)) := by
  have nodeAt : scope.nodes[reference.node]? = some node := by
    simpa [resolveNode, scopeResolved] using nodeResolved
  have inBounds : reference.node < scope.nodes.length := by
    by_contra outOfBounds
    have missing : scope.nodes[reference.node]? = none := by
      exact List.getElem?_eq_none (Nat.le_of_not_gt outOfBounds)
    rw [missing] at nodeAt
    contradiction
  have nodeEq : scope.nodes[reference.node] = node := by
    rw [List.getElem?_eq_getElem inBounds] at nodeAt
    exact Option.some.inj nodeAt
  obtain ⟨wires, path, _⟩ :=
    (Mxx.Ir.mem_childRunnerWithFuel_succ_iff_path samplers stage.program fuel
      reference.scope.definitionName scope params inputs values definitionResolved).mp childMember
  obtain ⟨before, nodeValues, _, nodeMember, _⟩ := path.atNodeIndex reference.node inBounds
  exact ⟨{
    node
    before
    values := nodeValues
    resolved := nodeResolved
    member := by simpa [nodeEq] using nodeMember
  }⟩

/-- Lift one checked single-output parallel body from its executable SSA path to the concrete
child-runner value.  Specialized index and initial-state lemmas discharge `pathValue` using the
fixed body checker; this theorem only connects scope output collection to that path value. -/
theorem parallelLoopSingleBodyOutput_of_childOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef} {bodyOutput : CoreWireRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs values : List Mxx.Ir.Value} {value : Mxx.Ir.Value}
    (loopVerified : verifyParallelLoop workflow reference = true)
    (bodyOutputs : reference.bodyOutputs = [bodyOutput])
    (bodyResolved :
      resolveScope workflow { reference.operation with scope := reference.bodyScope } = some body)
    (definitionFound :
      Mxx.Ir.lookupDefinition reference.bodyScope.definitionName stage.program.definitions =
        some body)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.bodyScope.definitionName params inputs)
    (pathValue : ∀ final,
      Mxx.Ir.EvaluatesNodesPath (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel)
        samplers params (body.inputNames.zip inputs) 0 body.nodes [] final →
      Mxx.Ir.lookupWire (wireRef bodyOutput) final = some value) :
    values = [value] := by
  have outputWires := verifyParallelLoop_bodyOutputs loopVerified bodyResolved
  rw [bodyOutputs] at outputWires
  obtain ⟨final, path, valuesEq⟩ :=
    (Mxx.Ir.mem_childRunnerWithFuel_succ_iff_path samplers stage.program fuel
      reference.bodyScope.definitionName body params inputs values definitionFound).mp childMember
  have outputValue := pathValue final path
  rw [valuesEq]
  cases outputs : body.outputs with
  | nil => simp [outputs] at outputWires
  | cons head tail =>
      cases tail with
      | nil =>
          rcases head with ⟨name, wire⟩
          simp [outputs] at outputWires
          subst wire
          simp [Mxx.Ir.collectOutputs, outputValue]
      | cons next rest => simp [outputs] at outputWires

/-- A one-port parallel loop whose checked child outcome is `valueAt index` produces exactly the
ordered family of those values.  This is the generic family-level lifting used by all five index
loops and by initial-state expansion. -/
theorem parallelIterationsTrace_singlePortValues
    {runChild : Mxx.Ir.ChildRunner} {definition : String}
    {params : Mxx.Ir.ParamEnvironment} {indexSlot : Nat}
    {bindings : List (String × Mxx.Ir.IntExpr)} {modes : List Mxx.Ir.LoopInputMode}
    {arguments : List Mxx.Ir.Value} {indices : List Nat}
    {initial : List (List Mxx.Ir.Value)} {initialValues : List Mxx.Ir.Value}
    {final : List (List Mxx.Ir.Value)}
    (valueAt : Nat → Mxx.Ir.Value)
    (trace : Mxx.Ir.ParallelIterationsTrace runChild definition params indexSlot bindings modes
      arguments indices initial final)
    (initialEq : initial = [initialValues])
    (childExact : ∀ (index : Nat) evaluatedBindings childValues,
      Mxx.Ir.evaluateBindings ((.loopIndex indexSlot, .integer index) :: params) bindings =
          some evaluatedBindings →
      childValues ∈ runChild definition
        (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
        ((modes.zip arguments).map fun (mode, value) => Mxx.Ir.loopArgument mode index value) →
      childValues = [valueAt index]) :
    final = [initialValues ++ indices.map valueAt] := by
  induction trace generalizing initialValues with
  | nil => simp_all
  | cons index tail state evaluatedBindings childValues final bindingsEvaluate childMember rest
      induction =>
      subst state
      have childValuesEq := childExact index evaluatedBindings childValues bindingsEvaluate
        childMember
      subst childValues
      rw [induction (initialValues := initialValues ++ [valueAt index])
        (by simp [Mxx.Ir.appendPortValues])]
      simp [List.append_assoc]

/-- A one-port parallel trace may use hypotheses that are known only for indices actually
executed by that trace. -/
theorem parallelIterationsTrace_singlePortValues_mem
    {runChild : Mxx.Ir.ChildRunner} {definition : String}
    {params : Mxx.Ir.ParamEnvironment} {indexSlot : Nat}
    {bindings : List (String × Mxx.Ir.IntExpr)} {modes : List Mxx.Ir.LoopInputMode}
    {arguments : List Mxx.Ir.Value} {indices : List Nat}
    {initial : List (List Mxx.Ir.Value)} {initialValues : List Mxx.Ir.Value}
    {final : List (List Mxx.Ir.Value)}
    (valueAt : Nat → Mxx.Ir.Value)
    (trace : Mxx.Ir.ParallelIterationsTrace runChild definition params indexSlot bindings modes
      arguments indices initial final)
    (initialEq : initial = [initialValues])
    (childExact : ∀ (index : Nat), index ∈ indices → ∀ evaluatedBindings childValues,
      Mxx.Ir.evaluateBindings ((.loopIndex indexSlot, .integer index) :: params) bindings =
          some evaluatedBindings →
      childValues ∈ runChild definition
        (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
        ((modes.zip arguments).map fun (mode, value) ↦ Mxx.Ir.loopArgument mode index value) →
      childValues = [valueAt index]) :
    final = [initialValues ++ indices.map valueAt] := by
  induction trace generalizing initialValues with
  | nil => simp_all
  | cons index tail state evaluatedBindings childValues final bindingsEvaluate childMember rest
      induction =>
      subst state
      have childValuesEq := childExact index (by simp) evaluatedBindings childValues
        bindingsEvaluate childMember
      subst childValues
      rw [induction (initialValues := initialValues ++ [valueAt index])
        (by simp [Mxx.Ir.appendPortValues])]
      · simp [List.append_assoc]
      · intro queried queriedMember
        exact childExact queried (by simp [queriedMember])

/-- Exact executable parent-loop node recovered from a checked `ParallelLoopRef`. -/
structure CheckedParallelLoopResolution
    (workflow : Mxx.Ir.Workflow) (reference : ParallelLoopRef) where
  resolved : resolveNode workflow reference.operation = some {
    kind := .parallelLoop reference.bodyScope.definitionName reference.count reference.indexSlot
      reference.bindings (reference.inputModes.map CertifiedLoopInputMode.toIr)
    arguments := reference.arguments.map (wireRef ∘ CoreOperandRef.wire)
    outputCount := reference.outputs.length
  }

theorem checkedParallelLoopResolution_of_verified
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef}
    (verified : verifyParallelLoop workflow reference = true) :
    Nonempty (CheckedParallelLoopResolution workflow reference) := by
  unfold verifyParallelLoop at verified
  cases nodeResolved : resolveNode workflow reference.operation with
  | none => simp [nodeResolved] at verified
  | some node =>
      cases bodyResolved :
          resolveScope workflow { reference.operation with scope := reference.bodyScope } with
      | none => simp [nodeResolved, bodyResolved] at verified
      | some body =>
          rcases node with ⟨kind, arguments, outputCount⟩
          cases kind <;> simp_all [Bool.and_eq_true, decide_eq_true_eq]
          exact {
            resolved := by
              have bodyScopeEq : reference.bodyScope =
                  .parallelBody reference.operation.scope reference.operation.node := by aesop
              rw [bodyScopeEq]
              exact nodeResolved
          }

/-- The exact parent parallel-iteration trace selected by one checked loop execution. -/
structure CheckedParallelLoopTrace
    (workflow : Mxx.Ir.Workflow) (reference : ParallelLoopRef)
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs) where
  argumentValues : List Mxx.Ir.Value
  evaluatedCount : Int
  argumentsEvaluate :
    (reference.arguments.map (wireRef ∘ CoreOperandRef.wire)).mapM
      (fun wire => Mxx.Ir.lookupWire wire execution.before) = some argumentValues
  countEvaluate : reference.count.evaluate params = some evaluatedCount
  final : List (List Mxx.Ir.Value)
  iterations : Mxx.Ir.ParallelIterationsTrace runChild reference.bodyScope.definitionName params
    reference.indexSlot reference.bindings
    (reference.inputModes.map CertifiedLoopInputMode.toIr) argumentValues
    (List.range evaluatedCount.toNat) (List.replicate reference.outputs.length []) final
  valuesEq : execution.values = final.map Mxx.Ir.Value.family

theorem checkedParallelLoopTrace_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : CheckedParallelLoopResolution workflow reference)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (argumentValues : List Mxx.Ir.Value) (evaluatedCount : Int)
    (argumentsEvaluate :
      (reference.arguments.map (wireRef ∘ CoreOperandRef.wire)).mapM
        (fun wire => Mxx.Ir.lookupWire wire execution.before) = some argumentValues)
    (countEvaluate : reference.count.evaluate params = some evaluatedCount) :
    Nonempty (CheckedParallelLoopTrace workflow reference runChild samplers params inputs
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

/-- Invert a checked parallel-loop execution once an exact retained output excludes the
interpreter's invalid fallback. -/
theorem checkedParallelLoopTrace_of_nonInvalidOutput
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : CheckedParallelLoopResolution workflow reference)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (nonInvalid : execution.values ≠ [.invalid "ParallelLoop argument mismatch"]) :
    Nonempty (CheckedParallelLoopTrace workflow reference runChild samplers params inputs
      execution) := by
  have executionResolved := execution.resolved
  have resolutionResolved := resolution.resolved
  rw [executionResolved] at resolutionResolved
  have nodeEq := Option.some.inj resolutionResolved
  let argumentRefs := reference.arguments.map (wireRef ∘ CoreOperandRef.wire)
  cases argumentsResult : argumentRefs.mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) with
  | none =>
      have member := execution.member
      simp [nodeEq, Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentRefs, argumentsResult] at member
      exact (nonInvalid member).elim
  | some argumentValues =>
      cases countResult : reference.count.evaluate params with
      | none =>
          have member := execution.member
          simp [nodeEq, Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentRefs, argumentsResult,
            countResult] at member
          exact (nonInvalid member).elim
      | some evaluatedCount =>
          apply checkedParallelLoopTrace_of_execution resolution execution argumentValues
            evaluatedCount
          · simpa [argumentRefs] using argumentsResult
          · exact countResult

/-- Lift exact child semantics through a checked one-port parent parallel loop. -/
theorem checkedParallelLoop_onePortFamily
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    {execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs}
    (trace : CheckedParallelLoopTrace workflow reference runChild samplers params inputs execution)
    (outputsOne : reference.outputs.length = 1)
    (valueAt : Nat → Mxx.Ir.Value)
    (childExact : ∀ (index : Nat) evaluatedBindings childValues,
      Mxx.Ir.evaluateBindings ((.loopIndex reference.indexSlot, .integer index) :: params)
          reference.bindings = some evaluatedBindings →
      childValues ∈ runChild reference.bodyScope.definitionName
        (evaluatedBindings ++ ((.loopIndex reference.indexSlot, .integer index) :: params))
        (((reference.inputModes.map CertifiedLoopInputMode.toIr).zip trace.argumentValues).map
          fun (mode, value) => Mxx.Ir.loopArgument mode index value) →
      childValues = [valueAt index]) :
    execution.values = [.family ((List.range trace.evaluatedCount.toNat).map valueAt)] := by
  have initialEq : List.replicate reference.outputs.length [] =
      ([[]] : List (List Mxx.Ir.Value)) := by simp [outputsOne]
  have finalEq := parallelIterationsTrace_singlePortValues valueAt trace.iterations initialEq
    childExact
  rw [trace.valuesEq, finalEq]
  simp

/-- Membership in a two-stage workflow exposes the exact producer and consumer stage outcomes.
This is the workflow-level bridge used before checking individual artifact edges. -/
theorem twoStageWorkflowOutcome
    (samplers : Mxx.MxxSamplerFamily) (workflow : Mxx.Ir.Workflow)
    (params : Mxx.Ir.ParamEnvironment) (protocolInputs output : Mxx.Ir.Environment)
    (producer consumer : Mxx.Ir.Stage)
    (stages : workflow.stages = [producer, consumer])
    (entrypoint : workflow.entrypoint = consumer.id)
    (stageIdsDistinct : producer.id ≠ consumer.id)
    (member : output ∈ Mxx.Ir.denoteWorkflow samplers workflow params protocolInputs) :
    ∃ produced,
      produced ∈ Mxx.Ir.denote samplers producer.program params
        (Mxx.Ir.stageInputs protocolInputs [] producer) ∧
      output ∈ Mxx.Ir.denote samplers consumer.program params
        (Mxx.Ir.stageInputs protocolInputs [(producer.id, produced)] consumer) := by
  unfold Mxx.Ir.denoteWorkflow at member
  rw [stages, entrypoint] at member
  simpa [Mxx.Ir.evaluateStages, Mxx.Ir.lookupStage, stageIdsDistinct] using member

/-- The workflow interpreter passes an artifact input exactly as the value stored under the
certified producer output name. -/
theorem resolveArtifactInput_eq
    (protocolInputs : Mxx.Ir.Environment) (producerStage producerOutput : String)
    (produced : Mxx.Ir.Environment) (value : Mxx.Ir.Value)
    (outputResolved : Mxx.Ir.lookupEnvironment producerOutput produced = some value) :
    Mxx.Ir.resolveStageInput protocolInputs [(producerStage, produced)]
      (.artifact producerStage producerOutput) = value := by
  simp [Mxx.Ir.resolveStageInput, Mxx.Ir.lookupStage, outputResolved]

private theorem lookupEnvironment_stageInputs_of_binding
    (protocolInputs : Mxx.Ir.Environment) (stages : Mxx.Ir.StageEnvironment)
    (bindings : List (String × Mxx.Ir.InputSource)) (name : String)
    (source : Mxx.Ir.InputSource)
    (namesUnique : (bindings.map Prod.fst).Nodup)
    (bound : (name, source) ∈ bindings) :
    Mxx.Ir.lookupEnvironment name
        (bindings.map fun binding ↦
          (binding.1, Mxx.Ir.resolveStageInput protocolInputs stages binding.2)) =
      some (Mxx.Ir.resolveStageInput protocolInputs stages source) := by
  induction bindings with
  | nil => simp at bound
  | cons head tail induction =>
      rcases head with ⟨headName, headSource⟩
      simp only [List.map_cons, List.nodup_cons, List.mem_map, Prod.exists] at namesUnique
      simp only [List.mem_cons, Prod.mk.injEq] at bound
      rcases bound with ⟨rfl, rfl⟩ | bound
      · simp [Mxx.Ir.lookupEnvironment]
      · have different : headName ≠ name := by
          intro equal
          apply namesUnique.1
          exact ⟨name, source, bound, by simp [equal]⟩
        simp [Mxx.Ir.lookupEnvironment, different]
        exact induction namesUnique.2 bound

/-- An exact artifact binding equates the producer output with the value read by the consumer
input node.  The uniqueness premise is supplied by the verified stage interface. -/
theorem artifactConsumerInput_eq
    (protocolInputs : Mxx.Ir.Environment) (producerStage producerOutput consumerInput : String)
    (produced : Mxx.Ir.Environment) (consumer : Mxx.Ir.Stage) (value : Mxx.Ir.Value)
    (namesUnique : (consumer.inputs.map Prod.fst).Nodup)
    (binding :
      (consumerInput, .artifact producerStage producerOutput) ∈ consumer.inputs)
    (outputResolved : Mxx.Ir.lookupEnvironment producerOutput produced = some value) :
    Mxx.Ir.lookupEnvironment consumerInput
        (Mxx.Ir.stageInputs protocolInputs [(producerStage, produced)] consumer) =
      some value := by
  rw [Mxx.Ir.stageInputs,
    lookupEnvironment_stageInputs_of_binding protocolInputs [(producerStage, produced)]
      consumer.inputs consumerInput (.artifact producerStage producerOutput) namesUnique binding,
    resolveArtifactInput_eq protocolInputs producerStage producerOutput produced value
      outputResolved]

/-- The exact executable fields recovered from a checked input-injection state scan. -/
structure InputInjectionStateScanResolution
    (workflow : Mxx.Ir.Workflow) (layout : InputInjectionLayout) where
  count : Mxx.Ir.IntExpr
  indexSlot : Nat
  bindings : List (String × Mxx.Ir.IntExpr)
  resolved : resolveNode workflow layout.stateScan = some {
    kind := .sequentialLoop layout.bodyScope.definitionName count indexSlot bindings 1
    arguments := [wireRef layout.initialStates.wire, wireRef layout.packedDigits.wire,
      wireRef layout.transitionFamily.wire]
    outputCount := 1
  }

/-- The accepted input-injection layout identifies the existing state-scan node exactly. -/
theorem verifyInputInjection_stateScanResolution
    {workflow : Mxx.Ir.Workflow} {layout : InputInjectionLayout}
    (verified : verifyInputInjection workflow layout = true) :
    Nonempty (InputInjectionStateScanResolution workflow layout) := by
  have scanVerified :
      verifySequentialLoop workflow layout.stateScan layout.bodyScope [layout.initialStates]
        [layout.packedDigits, layout.transitionFamily] [layout.finalStates] = true := by
    rw [verifyInputInjection] at verified
    simp only [Bool.and_eq_true] at verified
    aesop
  unfold verifySequentialLoop at scanVerified
  cases nodeResolved : resolveNode workflow layout.stateScan with
  | none => simp [nodeResolved] at scanVerified
  | some node =>
      cases scopeResolved : resolveScope workflow { layout.stateScan with scope := layout.bodyScope } with
      | none => simp [nodeResolved, scopeResolved] at scanVerified
      | some scope =>
          rcases node with ⟨kind, arguments, outputCount⟩
          cases kind <;> simp_all [wireRef]
          rename_i definition count indexSlot bindings carriedCount
          exact ⟨{
            count
            indexSlot
            bindings
            resolved := by simp_all [wireRef]
          }⟩

/-- `VerifiedDiamondLayout` exposes the same exact state-scan resolution without another
certificate check. -/
theorem VerifiedDiamondLayout.stateScanResolution
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    Nonempty (InputInjectionStateScanResolution workflow certificate.inputInjection) :=
  verifyInputInjection_stateScanResolution verified.inputInjectionMatches

/-- A concrete checked state-scan execution together with the exact sequential trace selected by
the interpreter. -/
structure InputInjectionStateScanTrace
    (workflow : Mxx.Ir.Workflow) (layout : InputInjectionLayout)
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (execution : ReferencedNodeExecution workflow layout.stateScan runChild samplers params inputs)
    where
  resolution : InputInjectionStateScanResolution workflow layout
  count : Mxx.Ir.IntExpr
  indexSlot : Nat
  bindings : List (String × Mxx.Ir.IntExpr)
  countEq : count = resolution.count
  indexSlotEq : indexSlot = resolution.indexSlot
  bindingsEq : bindings = resolution.bindings
  argumentValues : List Mxx.Ir.Value
  evaluatedCount : Int
  argumentsEvaluate :
    [wireRef layout.initialStates.wire, wireRef layout.packedDigits.wire,
      wireRef layout.transitionFamily.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) = some argumentValues
  countEvaluate : count.evaluate params = some evaluatedCount
  iterations :
    Mxx.Ir.SequentialIterationsTrace runChild layout.bodyScope.definitionName params indexSlot
      bindings (argumentValues.drop 1) (List.range evaluatedCount.toNat)
      (argumentValues.take 1) execution.values

/-- Invert a concrete execution of the verified state scan into its exact iteration trace. -/
theorem inputInjectionStateScanTrace_of_resolution
    {workflow : Mxx.Ir.Workflow} {layout : InputInjectionLayout}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : InputInjectionStateScanResolution workflow layout)
    (execution : ReferencedNodeExecution workflow layout.stateScan runChild
      samplers params inputs)
    (argumentValues : List Mxx.Ir.Value) (evaluatedCount : Int)
    (argumentsEvaluate :
      [wireRef layout.initialStates.wire, wireRef layout.packedDigits.wire,
        wireRef layout.transitionFamily.wire].mapM
          (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) = some argumentValues)
    (countEvaluate : resolution.count.evaluate params = some evaluatedCount) :
    Nonempty (InputInjectionStateScanTrace workflow layout runChild samplers
      params inputs execution) := by
  have executionResolved := execution.resolved
  have resolutionResolved := resolution.resolved
  have nodeEq : execution.node = {
      kind := .sequentialLoop layout.bodyScope.definitionName resolution.count
        resolution.indexSlot resolution.bindings 1
      arguments := [wireRef layout.initialStates.wire, wireRef layout.packedDigits.wire,
        wireRef layout.transitionFamily.wire]
      outputCount := 1
    } := by
    rw [executionResolved] at resolutionResolved
    exact Option.some.inj resolutionResolved
  have member : execution.values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs
      execution.before {
        kind := .sequentialLoop layout.bodyScope.definitionName resolution.count
          resolution.indexSlot resolution.bindings 1
        arguments := [wireRef layout.initialStates.wire,
          wireRef layout.packedDigits.wire, wireRef layout.transitionFamily.wire]
        outputCount := 1
      } := by simpa [nodeEq] using execution.member
  have iterations :=
    (Mxx.Ir.mem_evaluateNode_sequentialLoop_iff_trace runChild samplers params inputs
      execution.before layout.bodyScope.definitionName resolution.count
      resolution.indexSlot resolution.bindings 1
      [wireRef layout.initialStates.wire, wireRef layout.packedDigits.wire,
        wireRef layout.transitionFamily.wire]
      1 argumentValues evaluatedCount argumentsEvaluate countEvaluate execution.values).mp member
  exact ⟨{
    resolution
    count := resolution.count
    indexSlot := resolution.indexSlot
    bindings := resolution.bindings
    countEq := rfl
    indexSlotEq := rfl
    bindingsEq := rfl
    argumentValues
    evaluatedCount
    argumentsEvaluate
    countEvaluate
    iterations
  }⟩

/-- Invert a checked state-scan execution without asking the caller to restate executable argument
or parameter lookups.  Successful node membership itself rules out failed argument/count
evaluation. -/
theorem inputInjectionStateScanTrace_of_nonInvalidOutput
    {workflow : Mxx.Ir.Workflow} {layout : InputInjectionLayout}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : InputInjectionStateScanResolution workflow layout)
    (execution : ReferencedNodeExecution workflow layout.stateScan runChild samplers params inputs)
    (nonInvalid : execution.values ≠ [.invalid "SequentialLoop argument mismatch"]) :
    Nonempty (InputInjectionStateScanTrace workflow layout runChild samplers params inputs
      execution) := by
  have executionResolved := execution.resolved
  have resolutionResolved := resolution.resolved
  rw [executionResolved] at resolutionResolved
  have nodeEq := Option.some.inj resolutionResolved
  let argumentRefs := [wireRef layout.initialStates.wire, wireRef layout.packedDigits.wire,
    wireRef layout.transitionFamily.wire]
  cases argumentsResult : argumentRefs.mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) with
  | none =>
      have member := execution.member
      simp [nodeEq, Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentRefs, argumentsResult] at member
      exact (nonInvalid member).elim
  | some argumentValues =>
      cases countResult : resolution.count.evaluate params with
      | none =>
          have member := execution.member
          simp [nodeEq, Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentRefs, argumentsResult,
            countResult] at member
          exact (nonInvalid member).elim
      | some evaluatedCount =>
          apply inputInjectionStateScanTrace_of_resolution resolution execution argumentValues
            evaluatedCount
          · simpa [argumentRefs] using argumentsResult
          · exact countResult

/-- One referenced node together with the prefix of the single retained child path that reaches
its `before` environment. -/
structure ReferencedNodeExecutionOnSharedPath
    (workflow : Mxx.Ir.Workflow) (reference : CoreNodeRef)
    (stage : Mxx.Ir.Stage) (scope : Mxx.Ir.Scope) (fuel : Nat)
    (samplers : Mxx.MxxSamplerFamily) (params : Mxx.Ir.ParamEnvironment)
    (inputs : List Mxx.Ir.Value) (final : Mxx.Ir.WireEnvironment) where
  referenceInBounds : reference.node < scope.nodes.length
  execution : ReferencedNodeExecution workflow reference
    (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
    (scope.inputNames.zip inputs)
  pathPrefix : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
    (scope.inputNames.zip inputs) 0 (scope.nodes.take reference.node) [] execution.before
  pathSuffix : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
    (scope.inputNames.zip inputs) (reference.node + 1) (scope.nodes.drop (reference.node + 1))
    (execution.before ++ Mxx.Ir.bindOutputs reference.node execution.values) final

/-- Every value already visible before a retained child node remains visible in the shared final
environment. -/
theorem ReferencedNodeExecutionOnSharedPath.beforeFinal
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : List Mxx.Ir.Value} {final : Mxx.Ir.WireEnvironment}
    (shared : ReferencedNodeExecutionOnSharedPath workflow reference stage scope fuel samplers
      params inputs final)
    {wire : Mxx.Ir.WireRef} {value : Mxx.Ir.Value}
    (resolved : Mxx.Ir.lookupWire wire shared.execution.before = some value) :
    Mxx.Ir.lookupWire wire final = some value := by
  apply shared.pathSuffix.lookupWire_preserved
  exact Mxx.Ir.lookupWire_append_of_eq_some resolved

/-- Every output port of a retained child node denotes its exact value in the shared final
environment. -/
theorem ReferencedNodeExecutionOnSharedPath.outputFinal
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : List Mxx.Ir.Value} {final : Mxx.Ir.WireEnvironment}
    (shared : ReferencedNodeExecutionOnSharedPath workflow reference stage scope fuel samplers
      params inputs final)
    (port : Nat) (portValid : port < shared.execution.values.length) :
    Mxx.Ir.lookupWire ⟨reference.node, port⟩ final =
      some (shared.execution.values.get ⟨port, portValid⟩) := by
  apply shared.pathSuffix.lookupWire_preserved
  have missing : Mxx.Ir.lookupWire ⟨reference.node, port⟩ shared.execution.before = none := by
    have absent := evaluatesNodesPath_nextNode_missing shared.pathPrefix port
      (by simp [Mxx.Ir.lookupWire])
    have minEq : min reference.node scope.nodes.length = reference.node :=
      Nat.min_eq_left (Nat.le_of_lt shared.referenceInBounds)
    simpa only [List.length_take, Nat.zero_add, minEq] using absent
  simpa using Mxx.Ir.lookupWire_append_bindOutputs missing portValid

/-- An earlier SSA wire present in the shared final environment was already present before this
retained child node. -/
theorem ReferencedNodeExecutionOnSharedPath.finalBefore
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : List Mxx.Ir.Value} {final : Mxx.Ir.WireEnvironment}
    (shared : ReferencedNodeExecutionOnSharedPath workflow reference stage scope fuel samplers
      params inputs final)
    {wire : Mxx.Ir.WireRef} {value : Mxx.Ir.Value} (past : wire.node < reference.node)
    (resolved : Mxx.Ir.lookupWire wire final = some value) :
    Mxx.Ir.lookupWire wire shared.execution.before = some value := by
  have atNode := resolved
  rw [Mxx.Ir.EvaluatesNodesPath.pastWire_eq shared.pathSuffix wire (by omega)] at atNode
  cases beforeLookup : Mxx.Ir.lookupWire wire shared.execution.before with
  | some found =>
      rw [Mxx.Ir.lookupWire_append_of_eq_some beforeLookup] at atNode
      have foundEq : found = value := Option.some.inj atNode
      subst found
      rfl
  | none =>
      rw [Mxx.Ir.lookupWire_append_of_eq_none beforeLookup,
        lookupWire_bindOutputs_of_node_ne (by omega)] at atNode
      contradiction

/-- The exact outer operations executed during one input-injection iteration.  All six witnesses
are prefixes of the same retained executable child path; no matrix equation is asserted here. -/
structure InputInjectionIterationExecutions
    (workflow : Mxx.Ir.Workflow) (layout : InputInjectionLayout)
    (stage : Mxx.Ir.Stage) (scope : Mxx.Ir.Scope) (fuel : Nat)
    (samplers : Mxx.MxxSamplerFamily) (params : Mxx.Ir.ParamEnvironment)
    (inputs : List Mxx.Ir.Value) where
  outputs : List Mxx.Ir.Value
  final : Mxx.Ir.WireEnvironment
  path : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
    (scope.inputNames.zip inputs) 0 scope.nodes [] final
  outputsEq : outputs = (Mxx.Ir.collectOutputs scope.outputs final).map Prod.snd
  selectedDigit : ReferencedNodeExecutionOnSharedPath workflow layout.selectedDigit.operation
    stage scope fuel samplers params inputs final
  sourceIndices : ReferencedNodeExecutionOnSharedPath workflow
    layout.sourceIndices.parallelLoop.operation stage scope fuel samplers params inputs
      final
  sourceStates : ReferencedNodeExecutionOnSharedPath workflow
    layout.sourceStates.parallelLoop.operation stage scope fuel samplers params inputs
      final
  transitionIndices : ReferencedNodeExecutionOnSharedPath workflow
    layout.transitionIndices.parallelLoop.operation stage scope fuel samplers params inputs
      final
  selectedTransitions : ReferencedNodeExecutionOnSharedPath workflow
    layout.selectedTransitions.parallelLoop.operation stage scope fuel samplers params inputs
      final
  stateProduct : ReferencedNodeExecutionOnSharedPath workflow layout.stateProduct.parallelLoop
    stage scope fuel samplers params inputs final

/-- Extract one referenced body node from a retained shared child path.  Unlike repeatedly
inverting child-runner membership, this theorem cannot select different nondeterministic paths for
different references. -/
private theorem nestedNodeExecution_of_sharedPath
    {workflow : Mxx.Ir.Workflow} (reference : CoreNodeRef) (stage : Mxx.Ir.Stage)
    (scope : Mxx.Ir.Scope) (fuel : Nat) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : List Mxx.Ir.Value)
    (final : Mxx.Ir.WireEnvironment)
    (path : Mxx.Ir.EvaluatesNodesPath
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs) 0 scope.nodes [] final)
    (node : Mxx.Ir.Node)
    (scopeResolved : resolveScope workflow reference = some scope)
    (nodeResolved : resolveNode workflow reference = some node) :
    Nonempty (ReferencedNodeExecutionOnSharedPath workflow reference stage scope fuel samplers
      params inputs final) := by
  have nodeAt : scope.nodes[reference.node]? = some node := by
    simpa [resolveNode, scopeResolved] using nodeResolved
  have inBounds : reference.node < scope.nodes.length := by
    by_contra outOfBounds
    have missing : scope.nodes[reference.node]? = none :=
      List.getElem?_eq_none (Nat.le_of_not_gt outOfBounds)
    rw [missing] at nodeAt
    contradiction
  have nodeEq : scope.nodes[reference.node] = node := by
    rw [List.getElem?_eq_getElem inBounds] at nodeAt
    exact Option.some.inj nodeAt
  obtain ⟨before, values, pathPrefix, member, pathSuffix⟩ :=
    path.atNodeIndex reference.node inBounds
  let execution : ReferencedNodeExecution workflow reference
      (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers params
      (scope.inputNames.zip inputs) := {
    node
    before
    values
    resolved := nodeResolved
    member := by simpa [nodeEq] using member
  }
  exact ⟨{
    referenceInBounds := inBounds
    execution
    pathPrefix
    pathSuffix := by simpa [execution] using pathSuffix
  }⟩

private theorem resolveNode_of_verifyDynamicGet
    {workflow : Mxx.Ir.Workflow} {reference : DynamicFamilyGetRef} {family : CoreWireRef}
    (verified : verifyDynamicGet workflow reference family = true) :
    ∃ node, resolveNode workflow reference.operation = some node := by
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [verifyDynamicGet, verifyOperationOutput, verifyWire, resolved] at verified
  | some node => exact ⟨node, rfl⟩

private theorem resolveNode_of_verifyParallelLoop
    {workflow : Mxx.Ir.Workflow} {reference : ParallelLoopRef}
    (verified : verifyParallelLoop workflow reference = true) :
    ∃ node, resolveNode workflow reference.operation = some node := by
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [verifyParallelLoop, resolved] at verified
  | some node => exact ⟨node, rfl⟩

private theorem parallelLoop_of_verifyIndexFormula
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyParallelIndexFormulaRef workflow reference = true) :
    verifyParallelLoop workflow reference.parallelLoop = true := by
  unfold verifyParallelIndexFormulaRef at verified
  simp only [Bool.and_eq_true] at verified
  exact verified.1.1.1.1

private theorem indexFormula_of_verifyOnlineSource
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyOnlineSourceIndexFormula workflow reference = true) :
    verifyParallelIndexFormulaRef workflow reference = true := by
  unfold verifyOnlineSourceIndexFormula at verified
  simp only [Bool.and_eq_true] at verified
  aesop

private theorem indexFormula_of_verifyOnlineTransition
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    (verified : verifyOnlineTransitionIndexFormula workflow reference = true) :
    verifyParallelIndexFormulaRef workflow reference = true := by
  unfold verifyOnlineTransitionIndexFormula at verified
  simp only [Bool.and_eq_true] at verified
  aesop

private theorem resolveNode_of_verifyParallelFamilyGet
    {workflow : Mxx.Ir.Workflow} {reference : ParallelFamilyGetRef}
    (verified : verifyParallelFamilyGet workflow reference = true) :
    ∃ node, resolveNode workflow reference.parallelLoop.operation = some node := by
  have loopVerified : verifyParallelLoop workflow reference.parallelLoop = true := by
    rw [verifyParallelFamilyGet] at verified
    simp only [Bool.and_eq_true] at verified
    aesop
  exact resolveNode_of_verifyParallelLoop loopVerified

private theorem resolveNode_of_verifyParallelMatrixBinary
    {workflow : Mxx.Ir.Workflow} {reference : ParallelMatrixBinaryRef}
    {expected : Mxx.Ir.NodeKind}
    (verified : verifyParallelMatrixBinary workflow reference expected = true) :
    ∃ node, resolveNode workflow reference.parallelLoop = some node := by
  have boundaryVerified : verifyParallelBoundary workflow reference.parallelLoop
      reference.bodyScope [reference.leftFamily, reference.rightFamily]
      [reference.bodyLeft, reference.bodyRight] reference.bodyOutput reference.outputFamily =
        true := by
    rw [verifyParallelMatrixBinary] at verified
    simp only [Bool.and_eq_true] at verified
    aesop
  cases resolved : resolveNode workflow reference.parallelLoop with
  | none => simp [verifyParallelBoundary, resolved] at boundaryVerified
  | some node => exact ⟨node, rfl⟩

/-- A successful input-injection check exposes the six exact outer nodes executed by each
state-scan body invocation. -/
theorem inputInjectionIterationExecutions_of_childOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs outputs : List Mxx.Ir.Value}
    (verified : VerifiedDiamondLayout workflow certificate)
    (definitionResolved :
      Mxx.Ir.lookupDefinition certificate.inputInjection.bodyScope.definitionName
          stage.program.definitions = some scope)
    (scopeResolved :
      resolveScope workflow
          { certificate.inputInjection.stateScan with
            scope := certificate.inputInjection.bodyScope } = some scope)
    (childMember : outputs ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      certificate.inputInjection.bodyScope.definitionName params inputs) :
    Nonempty (InputInjectionIterationExecutions workflow certificate.inputInjection stage scope
      fuel samplers params inputs) := by
  let layout := certificate.inputInjection
  have checked := verified.inputInjectionMatches
  rw [verifyInputInjection] at checked
  simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
  rcases checked with ⟨checked, _externalSources⟩
  rcases checked with ⟨checked, _selectedSlotMatches⟩
  rcases checked with ⟨checked, selectedTransitionsScope⟩
  rcases checked with ⟨checked, transitionIndicesScope⟩
  rcases checked with ⟨checked, sourceStatesScope⟩
  rcases checked with ⟨checked, sourceIndicesScope⟩
  rcases checked with ⟨checked, selectedDigitScope⟩
  rcases checked with ⟨checked, stagesAligned⟩
  simp only [List.all_cons, List.all_nil, Bool.and_eq_true, decide_eq_true_eq,
    and_true] at stagesAligned
  have selectedDigitStage :
      layout.selectedDigit.operation.stage = layout.stateScan.stage := stagesAligned.1
  have sourceIndicesStage :
      layout.sourceIndices.parallelLoop.operation.stage = layout.stateScan.stage :=
    stagesAligned.2.1
  have sourceStatesStage :
      layout.sourceStates.parallelLoop.operation.stage = layout.stateScan.stage :=
    stagesAligned.2.2.1
  have transitionIndicesStage :
      layout.transitionIndices.parallelLoop.operation.stage = layout.stateScan.stage :=
    stagesAligned.2.2.2.1
  have selectedTransitionsStage :
      layout.selectedTransitions.parallelLoop.operation.stage = layout.stateScan.stage :=
    stagesAligned.2.2.2.2.1
  have stateProductStage :
      layout.stateProduct.parallelLoop.stage = layout.stateScan.stage :=
    stagesAligned.2.2.2.2.2
  rcases checked with ⟨checked, _stateProductRightFamily⟩
  rcases checked with ⟨checked, _stateProductLeftFamily⟩
  rcases checked with ⟨checked, _selectedTransitionsSource⟩
  rcases checked with ⟨checked, _transitionIndicesOutput⟩
  rcases checked with ⟨checked, _sourceStatesSource⟩
  rcases checked with ⟨checked, _sourceIndicesOutput⟩
  rcases checked with ⟨checked, _transitionIndicesOutputCount⟩
  rcases checked with ⟨checked, _sourceIndicesOutputCount⟩
  rcases checked with ⟨checked, _bodyFinalStates⟩
  rcases checked with ⟨checked, stateProductScope⟩
  have selectedDigitVerified :
      verifyDynamicGet workflow layout.selectedDigit layout.bodyPackedDigits = true := by aesop
  have sourceIndicesExact :
      verifyOnlineSourceIndexFormula workflow layout.sourceIndices = true := by aesop
  have sourceIndicesFormulaVerified :
      verifyParallelIndexFormulaRef workflow layout.sourceIndices = true :=
    indexFormula_of_verifyOnlineSource sourceIndicesExact
  have sourceIndicesVerified :
      verifyParallelLoop workflow layout.sourceIndices.parallelLoop = true := by
    exact parallelLoop_of_verifyIndexFormula sourceIndicesFormulaVerified
  have sourceStatesVerified : verifyParallelFamilyGet workflow layout.sourceStates = true := by aesop
  have transitionIndicesExact :
      verifyOnlineTransitionIndexFormula workflow layout.transitionIndices = true := by aesop
  have transitionIndicesFormulaVerified :
      verifyParallelIndexFormulaRef workflow layout.transitionIndices = true :=
    indexFormula_of_verifyOnlineTransition transitionIndicesExact
  have transitionIndicesVerified :
      verifyParallelLoop workflow layout.transitionIndices.parallelLoop = true := by
    exact parallelLoop_of_verifyIndexFormula transitionIndicesFormulaVerified
  have selectedTransitionsVerified :
      verifyParallelFamilyGet workflow layout.selectedTransitions = true := by aesop
  have stateProductVerified :
      verifyParallelMatrixBinary workflow layout.stateProduct .matrixMultiply = true := by aesop
  obtain ⟨selectedDigitNode, selectedDigitResolved⟩ :=
    resolveNode_of_verifyDynamicGet selectedDigitVerified
  obtain ⟨sourceIndicesNode, sourceIndicesResolved⟩ :=
    resolveNode_of_verifyParallelLoop sourceIndicesVerified
  obtain ⟨sourceStatesNode, sourceStatesResolved⟩ :=
    resolveNode_of_verifyParallelFamilyGet sourceStatesVerified
  obtain ⟨transitionIndicesNode, transitionIndicesResolved⟩ :=
    resolveNode_of_verifyParallelLoop transitionIndicesVerified
  obtain ⟨selectedTransitionsNode, selectedTransitionsResolved⟩ :=
    resolveNode_of_verifyParallelFamilyGet selectedTransitionsVerified
  obtain ⟨stateProductNode, stateProductResolved⟩ :=
    resolveNode_of_verifyParallelMatrixBinary stateProductVerified
  have selectedDigitScopeResolved :
      resolveScope workflow layout.selectedDigit.operation = some scope := by
    simpa [layout, resolveScope, selectedDigitScope, selectedDigitStage] using scopeResolved
  have sourceIndicesScopeResolved :
      resolveScope workflow layout.sourceIndices.parallelLoop.operation = some scope := by
    simpa [layout, resolveScope, sourceIndicesScope, sourceIndicesStage] using scopeResolved
  have sourceStatesScopeResolved :
      resolveScope workflow layout.sourceStates.parallelLoop.operation = some scope := by
    simpa [layout, resolveScope, sourceStatesScope, sourceStatesStage] using scopeResolved
  have transitionIndicesScopeResolved :
      resolveScope workflow layout.transitionIndices.parallelLoop.operation = some scope := by
    simpa [layout, resolveScope, transitionIndicesScope, transitionIndicesStage] using scopeResolved
  have selectedTransitionsScopeResolved :
      resolveScope workflow layout.selectedTransitions.parallelLoop.operation = some scope := by
    simpa [layout, resolveScope, selectedTransitionsScope, selectedTransitionsStage] using scopeResolved
  have stateProductScopeResolved :
      resolveScope workflow layout.stateProduct.parallelLoop = some scope := by
    simpa [layout, resolveScope, stateProductScope, stateProductStage] using scopeResolved
  obtain ⟨final, path, outputsEq⟩ :=
    (Mxx.Ir.mem_childRunnerWithFuel_succ_iff_path samplers stage.program fuel
      layout.bodyScope.definitionName scope params inputs outputs definitionResolved).mp childMember
  obtain ⟨selectedDigit⟩ := nestedNodeExecution_of_sharedPath
    layout.selectedDigit.operation stage scope fuel samplers params inputs final path
    selectedDigitNode selectedDigitScopeResolved selectedDigitResolved
  obtain ⟨sourceIndices⟩ := nestedNodeExecution_of_sharedPath
    layout.sourceIndices.parallelLoop.operation stage scope fuel samplers params inputs final path
    sourceIndicesNode sourceIndicesScopeResolved sourceIndicesResolved
  obtain ⟨sourceStates⟩ := nestedNodeExecution_of_sharedPath
    layout.sourceStates.parallelLoop.operation stage scope fuel samplers params inputs final path
    sourceStatesNode sourceStatesScopeResolved sourceStatesResolved
  obtain ⟨transitionIndices⟩ := nestedNodeExecution_of_sharedPath
    layout.transitionIndices.parallelLoop.operation stage scope fuel samplers params inputs final
    path transitionIndicesNode transitionIndicesScopeResolved transitionIndicesResolved
  obtain ⟨selectedTransitions⟩ := nestedNodeExecution_of_sharedPath
    layout.selectedTransitions.parallelLoop.operation stage scope fuel samplers params inputs final
    path selectedTransitionsNode selectedTransitionsScopeResolved selectedTransitionsResolved
  obtain ⟨stateProduct⟩ := nestedNodeExecution_of_sharedPath
    layout.stateProduct.parallelLoop stage scope fuel samplers params inputs final path
    stateProductNode stateProductScopeResolved stateProductResolved
  exact ⟨{
    outputs
    final
    path
    outputsEq
    selectedDigit
    sourceIndices
    sourceStates
    transitionIndices
    selectedTransitions
    stateProduct
  }⟩

/-- Every selected state-scan iteration has exact executions for the typed input-injection body
references.  This is universal over iterations; it does not select one representative state. -/
theorem InputInjectionStateScanTrace.everyIterationExecution
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {execution : ReferencedNodeExecution workflow certificate.inputInjection.stateScan
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs}
    (verified : VerifiedDiamondLayout workflow certificate)
    (trace : InputInjectionStateScanTrace workflow certificate.inputInjection
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs execution)
    (definitionResolved :
      Mxx.Ir.lookupDefinition certificate.inputInjection.bodyScope.definitionName
          stage.program.definitions = some scope)
    (scopeResolved :
      resolveScope workflow
          { certificate.inputInjection.stateScan with
            scope := certificate.inputInjection.bodyScope } = some scope) :
    ∀ index ∈ List.range trace.evaluatedCount.toNat,
      ∃ childParams childInputs childOutputs,
        childOutputs ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
          certificate.inputInjection.bodyScope.definitionName childParams childInputs ∧
        Nonempty (InputInjectionIterationExecutions workflow certificate.inputInjection stage
          scope fuel samplers childParams childInputs) := by
  apply trace.iterations.everyChild
  intro index evaluatedBindings state next _ childMember
  exact inputInjectionIterationExecutions_of_childOutcome verified definitionResolved scopeResolved
    childMember

/-- Exact executable fields of a checked parallel matrix operation. -/
structure ParallelMatrixBinaryResolution
    (workflow : Mxx.Ir.Workflow) (reference : ParallelMatrixBinaryRef)
    (expected : Mxx.Ir.NodeKind) where
  count : Mxx.Ir.IntExpr
  indexSlot : Nat
  bindings : List (String × Mxx.Ir.IntExpr)
  modes : List Mxx.Ir.LoopInputMode
  loopResolved : resolveNode workflow reference.parallelLoop = some {
    kind := .parallelLoop reference.bodyScope.definitionName count indexSlot bindings modes
    arguments := [wireRef reference.leftFamily.wire, wireRef reference.rightFamily.wire]
    outputCount := 1
  }
  operationResolved : resolveNode workflow reference.operation.operation = some {
    kind := expected
    arguments := [wireRef reference.bodyLeft, wireRef reference.bodyRight]
    outputCount := 1
  }

private theorem verifyMatrixBinary_exactNode
    {workflow : Mxx.Ir.Workflow} {reference : MatrixBinaryRef}
    {expected : Mxx.Ir.NodeKind}
    (verified : verifyMatrixBinary workflow reference expected = true) :
    resolveNode workflow reference.operation = some {
      kind := expected
      arguments := [wireRef reference.left.wire, wireRef reference.right.wire]
      outputCount := 1
    } := by
  cases resolved : resolveNode workflow reference.operation with
  | none =>
      simp [verifyMatrixBinary, verifyBinaryNode, verifyOperationOutput, verifyWire, resolved]
        at verified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      simp [verifyMatrixBinary, verifyBinaryNode, verifyOperationOutput, verifyWire, resolved]
        at verified
      simp_all [wireRef]

/-- A checked parallel matrix operation resolves to the exact existing loop and body operation. -/
theorem verifyParallelMatrixBinary_resolution
    {workflow : Mxx.Ir.Workflow} {reference : ParallelMatrixBinaryRef}
    {expected : Mxx.Ir.NodeKind}
    (verified : verifyParallelMatrixBinary workflow reference expected = true) :
    Nonempty (ParallelMatrixBinaryResolution workflow reference expected) := by
  have checked := verified
  rw [verifyParallelMatrixBinary] at checked
  simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
  have boundaryVerified : verifyParallelBoundary workflow reference.parallelLoop
      reference.bodyScope [reference.leftFamily, reference.rightFamily]
      [reference.bodyLeft, reference.bodyRight] reference.bodyOutput reference.outputFamily =
        true := by aesop
  have operationVerified : verifyMatrixBinary workflow reference.operation expected = true := by
    aesop
  have operationLeftEq : reference.operation.left.wire = reference.bodyLeft := by aesop
  have operationRightEq : reference.operation.right.wire = reference.bodyRight := by aesop
  have operationExact := verifyMatrixBinary_exactNode operationVerified
  unfold verifyParallelBoundary at boundaryVerified
  cases loopResolved : resolveNode workflow reference.parallelLoop with
  | none => simp [loopResolved] at boundaryVerified
  | some loopNode =>
      cases bodyResolved :
          resolveScope workflow { reference.parallelLoop with scope := reference.bodyScope } with
      | none => simp [loopResolved, bodyResolved] at boundaryVerified
      | some body =>
          rcases loopNode with ⟨loopKind, loopArguments, loopOutputCount⟩
          cases loopKind <;> simp_all [wireRef]
          rename_i definition count indexSlot bindings modes
          exact ⟨{
            count
            indexSlot
            bindings
            modes
            loopResolved := by simp_all [wireRef]
            operationResolved := by
              have bodyInputsEq :
                  [wireRef reference.bodyLeft, wireRef reference.bodyRight] =
                    scopeInputWires body := by aesop
              simpa [bodyInputsEq] using operationExact
          }⟩

/-- Exact parallel-iteration trace selected by one checked matrix-family operation. -/
structure ParallelMatrixBinaryTrace
    (workflow : Mxx.Ir.Workflow) (reference : ParallelMatrixBinaryRef)
    (expected : Mxx.Ir.NodeKind)
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (execution : ReferencedNodeExecution workflow reference.parallelLoop runChild samplers params
      inputs) where
  resolution : ParallelMatrixBinaryResolution workflow reference expected
  argumentValues : List Mxx.Ir.Value
  evaluatedCount : Int
  argumentsEvaluate :
    [wireRef reference.leftFamily.wire, wireRef reference.rightFamily.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) = some argumentValues
  countEvaluate : resolution.count.evaluate params = some evaluatedCount
  final : List (List Mxx.Ir.Value)
  iterations : Mxx.Ir.ParallelIterationsTrace runChild reference.bodyScope.definitionName params
    resolution.indexSlot resolution.bindings resolution.modes argumentValues
    (List.range evaluatedCount.toNat) (List.replicate 1 []) final
  valuesEq : execution.values = final.map Mxx.Ir.Value.family

/-- Invert one concrete checked parallel matrix-family execution into its exact loop trace. -/
theorem parallelMatrixBinaryTrace_of_resolution
    {workflow : Mxx.Ir.Workflow} {reference : ParallelMatrixBinaryRef}
    {expected : Mxx.Ir.NodeKind}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : ParallelMatrixBinaryResolution workflow reference expected)
    (execution : ReferencedNodeExecution workflow reference.parallelLoop runChild samplers params
      inputs)
    (argumentValues : List Mxx.Ir.Value) (evaluatedCount : Int)
    (argumentsEvaluate :
      [wireRef reference.leftFamily.wire, wireRef reference.rightFamily.wire].mapM
          (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) = some argumentValues)
    (countEvaluate : resolution.count.evaluate params = some evaluatedCount) :
    Nonempty (ParallelMatrixBinaryTrace workflow reference expected runChild samplers params inputs
      execution) := by
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
      } := by simpa [nodeEq] using execution.member
  obtain ⟨final, iterations, valuesEq⟩ :=
    (Mxx.Ir.mem_evaluateNode_parallelLoop_iff_trace runChild samplers params inputs
      execution.before reference.bodyScope.definitionName resolution.count
      resolution.indexSlot resolution.bindings resolution.modes
      [wireRef reference.leftFamily.wire, wireRef reference.rightFamily.wire]
      1 argumentValues evaluatedCount argumentsEvaluate countEvaluate execution.values).mp member
  exact ⟨{
    resolution
    argumentValues
    evaluatedCount
    argumentsEvaluate
    countEvaluate
    final
    iterations
    valuesEq
  }⟩

private theorem verifyOperand_owner
    {workflow : Mxx.Ir.Workflow} {reference : CoreOperandRef}
    (verified : verifyOperand workflow reference = true) :
    reference.node.stage = reference.wire.node.stage ∧
      reference.node.scope = reference.wire.node.scope := by
  unfold verifyOperand at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  exact verified.1.1

/-- Every point produced by a checked parallel matrix-family operation comes from an exact
execution of the certified body operation.  The statement is universal over the loop indices,
which is the foundation for a pointwise invariant over every output-family element. -/
theorem ParallelMatrixBinaryTrace.everyOperationExecution
    {workflow : Mxx.Ir.Workflow} {reference : ParallelMatrixBinaryRef}
    {expected : Mxx.Ir.NodeKind} {stage : Mxx.Ir.Stage} {scope : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {execution : ReferencedNodeExecution workflow reference.parallelLoop
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs}
    (verified : verifyParallelMatrixBinary workflow reference expected = true)
    (trace : ParallelMatrixBinaryTrace workflow reference expected
      (Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)) samplers params inputs
      execution)
    (definitionResolved :
      Mxx.Ir.lookupDefinition reference.bodyScope.definitionName stage.program.definitions =
        some scope)
    (scopeResolved :
      resolveScope workflow { reference.parallelLoop with scope := reference.bodyScope } =
        some scope) :
    ∀ index ∈ List.range trace.evaluatedCount.toNat,
      ∃ childParams childInputs childOutputs,
        childOutputs ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
          reference.bodyScope.definitionName childParams childInputs ∧
        Nonempty (ReferencedNodeExecution workflow reference.operation.operation
          (Mxx.Ir.childRunnerWithFuel samplers stage.program fuel) samplers childParams
          (scope.inputNames.zip childInputs)) := by
  have checked := verified
  rw [verifyParallelMatrixBinary] at checked
  simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
  have boundaryVerified : verifyParallelBoundary workflow reference.parallelLoop
      reference.bodyScope [reference.leftFamily, reference.rightFamily]
      [reference.bodyLeft, reference.bodyRight] reference.bodyOutput reference.outputFamily =
        true := by aesop
  have operationVerified : verifyMatrixBinary workflow reference.operation expected = true := by
    aesop
  have operationLeftEq : reference.operation.left.wire = reference.bodyLeft := by aesop
  have operationLeftNode : reference.operation.left.node = reference.operation.operation := by
    have operationChecked := operationVerified
    rw [verifyMatrixBinary, verifyBinaryNode] at operationChecked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at operationChecked
    aesop
  have leftVerified : verifyOperand workflow reference.operation.left = true := by
    rw [verifyMatrixBinary, verifyBinaryNode] at operationVerified
    simp only [Bool.and_eq_true] at operationVerified
    aesop
  have leftOwner := verifyOperand_owner leftVerified
  have bodyLeftStage : reference.bodyLeft.node.stage = reference.parallelLoop.stage := by
    rw [verifyParallelBoundary] at boundaryVerified
    simp only [Bool.and_eq_true, decide_eq_true_eq] at boundaryVerified
    aesop
  have bodyLeftScope : reference.bodyLeft.node.scope = reference.bodyScope := by
    rw [verifyParallelBoundary] at boundaryVerified
    simp only [Bool.and_eq_true, decide_eq_true_eq] at boundaryVerified
    aesop
  have operationStage :
      reference.operation.operation.stage = reference.parallelLoop.stage := by
    calc
      reference.operation.operation.stage = reference.operation.left.node.stage :=
        congrArg CoreNodeRef.stage operationLeftNode.symm
      _ = reference.operation.left.wire.node.stage := leftOwner.1
      _ = reference.bodyLeft.node.stage := congrArg (fun wire : CoreWireRef ↦ wire.node.stage)
        operationLeftEq
      _ = reference.parallelLoop.stage := bodyLeftStage
  have operationScope : reference.operation.operation.scope = reference.bodyScope := by
    calc
      reference.operation.operation.scope = reference.operation.left.node.scope :=
        congrArg CoreNodeRef.scope operationLeftNode.symm
      _ = reference.operation.left.wire.node.scope := leftOwner.2
      _ = reference.bodyLeft.node.scope := congrArg (fun wire : CoreWireRef ↦ wire.node.scope)
        operationLeftEq
      _ = reference.bodyScope := bodyLeftScope
  have operationScopeResolved :
      resolveScope workflow reference.operation.operation = some scope := by
    simpa [resolveScope, operationStage, operationScope] using scopeResolved
  have operationDefinitionResolved :
      Mxx.Ir.lookupDefinition reference.operation.operation.scope.definitionName
          stage.program.definitions = some scope := by
    simpa [operationScope] using definitionResolved
  have operationNodeResolved := trace.resolution.operationResolved
  apply trace.iterations.everyChild
  intro index evaluatedBindings childValues _ childMember
  have operationChild : childValues ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program
      (fuel + 1) reference.operation.operation.scope.definitionName
      (evaluatedBindings ++ ((.loopIndex trace.resolution.indexSlot, .integer index) :: params))
      ((trace.resolution.modes.zip trace.argumentValues).map fun (mode, value) ↦
        Mxx.Ir.loopArgument mode index value) := by
    simpa [operationScope] using childMember
  exact nestedNodeExecution_of_childOutcome workflow reference.operation.operation stage scope
    fuel samplers
    (evaluatedBindings ++ ((.loopIndex trace.resolution.indexSlot, .integer index) :: params))
    ((trace.resolution.modes.zip trace.argumentValues).map fun (mode, value) ↦
      Mxx.Ir.loopArgument mode index value)
    childValues {
      kind := expected
      arguments := [wireRef reference.bodyLeft, wireRef reference.bodyRight]
      outputCount := 1
    } operationDefinitionResolved operationScopeResolved operationNodeResolved operationChild

/-- A predicate holds pointwise for every value of every accumulated output port. -/
def EveryPortValue (predicate : Mxx.Ir.Value → Prop)
    (ports : List (List Mxx.Ir.Value)) : Prop :=
  ∀ port ∈ ports, ∀ value ∈ port, predicate value

private theorem everyPortValue_append
    (predicate : Mxx.Ir.Value → Prop) (ports : List (List Mxx.Ir.Value))
    (values : List Mxx.Ir.Value)
    (portsHold : EveryPortValue predicate ports)
    (valuesHold : ∀ value ∈ values, predicate value) :
    EveryPortValue predicate (Mxx.Ir.appendPortValues ports values) := by
  induction ports generalizing values with
  | nil => cases values <;> simp [EveryPortValue, Mxx.Ir.appendPortValues]
  | cons port ports induction =>
      cases values with
      | nil => simp [EveryPortValue, Mxx.Ir.appendPortValues]
      | cons value values =>
          simp only [Mxx.Ir.appendPortValues]
          intro output outputMember candidate candidateMember
          rcases List.mem_cons.mp outputMember with rfl | outputMember
          · rcases List.mem_append.mp candidateMember with candidateMember | candidateMember
            · exact portsHold port (by simp) candidate candidateMember
            · simp only [List.mem_singleton] at candidateMember
              subst candidate
              exact valuesHold value (by simp)
          · exact induction values
              (fun nested nestedMember ↦ portsHold nested (by simp [nestedMember]))
              (fun item itemMember ↦ valuesHold item (by simp [itemMember]))
              output outputMember candidate candidateMember

/-- Pointwise lifting for a parallel matrix-family trace.  Unlike an existential representative
matrix, the conclusion covers every value accumulated in every final output family. -/
theorem ParallelMatrixBinaryTrace.everyOutputValue
    {workflow : Mxx.Ir.Workflow} {reference : ParallelMatrixBinaryRef}
    {expected : Mxx.Ir.NodeKind} {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {execution : ReferencedNodeExecution workflow reference.parallelLoop runChild samplers params
      inputs}
    (trace : ParallelMatrixBinaryTrace workflow reference expected runChild samplers params inputs
      execution)
    (predicate : Mxx.Ir.Value → Prop)
    (childValuesHold : ∀ (index : Nat) evaluatedBindings childValues,
      Mxx.Ir.evaluateBindings
          ((.loopIndex trace.resolution.indexSlot, .integer index) :: params)
          trace.resolution.bindings = some evaluatedBindings →
      childValues ∈ runChild reference.bodyScope.definitionName
        (evaluatedBindings ++
          ((.loopIndex trace.resolution.indexSlot, .integer index) :: params))
        ((trace.resolution.modes.zip trace.argumentValues).map fun (mode, value) ↦
          Mxx.Ir.loopArgument mode index value) →
      ∀ value ∈ childValues, predicate value) :
    EveryPortValue predicate trace.final := by
  apply trace.iterations.invariant (EveryPortValue predicate)
  · intro index evaluatedBindings state childValues bindingsEvaluate childMember stateHold
    exact everyPortValue_append predicate state childValues stateHold
      (childValuesHold index evaluatedBindings childValues bindingsEvaluate childMember)
  · simp [EveryPortValue]

/-- An exact matrix-add node selected by a checked operation reference. -/
structure MatrixAddResolution
    (workflow : Mxx.Ir.Workflow) (reference : OperationRef) where
  resolved : resolveNode workflow reference.operation = some {
    kind := .matrixAdd
    arguments := reference.inputs.map (wireRef ∘ CoreOperandRef.wire)
    outputCount := reference.outputs.length
  }

theorem verifyMatrixAdd_resolution
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    (verified : verifyOperationKind workflow reference (fun kind => match kind with
      | .matrixAdd => true
      | _ => false) = true) :
    Nonempty (MatrixAddResolution workflow reference) := by
  unfold verifyOperationKind at verified
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [verifyOperation, resolved] at verified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> simp_all [verifyOperation]
      exact @MatrixAddResolution.mk workflow reference (by simp_all)

/-- Concrete semantics of an exact checked matrix-add node. -/
theorem matrixAddOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : MatrixAddResolution workflow reference)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (leftRef rightRef : Mxx.Ir.WireRef) (left right : Mxx.Matrix)
    (inputWires : reference.inputs.map (wireRef ∘ CoreOperandRef.wire) =
      [leftRef, rightRef])
    (outputsOne : reference.outputs.length = 1)
    (argumentsEvaluate :
      [leftRef, rightRef].mapM (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
        some [.matrix left, .matrix right]) :
    execution.values = [.matrix (Mxx.matrixAdd left right)] := by
  have executionResolved := execution.resolved
  have nodeResolved := resolution.resolved
  rw [executionResolved] at nodeResolved
  have nodeEq := Option.some.inj nodeResolved
  apply Mxx.Ir.mem_evaluateNode_matrixAdd_of_arguments runChild samplers params inputs
    execution.before leftRef rightRef left right 1 argumentsEvaluate
  simpa [nodeEq, inputWires, outputsOne] using execution.member

/-- An exact matrix-multiply node selected by a checked operation reference. -/
structure MatrixMultiplyResolution
    (workflow : Mxx.Ir.Workflow) (reference : OperationRef) where
  resolved : resolveNode workflow reference.operation = some {
    kind := .matrixMultiply
    arguments := reference.inputs.map (wireRef ∘ CoreOperandRef.wire)
    outputCount := reference.outputs.length
  }

theorem verifyMatrixMultiply_resolution
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    (verified : verifyOperationKind workflow reference (fun kind => match kind with
      | .matrixMultiply => true
      | _ => false) = true) :
    Nonempty (MatrixMultiplyResolution workflow reference) := by
  unfold verifyOperationKind at verified
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [verifyOperation, resolved] at verified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> simp_all [verifyOperation]
      exact @MatrixMultiplyResolution.mk workflow reference (by simp_all)

/-- Concrete semantics of an exact checked matrix-multiply node. -/
theorem matrixMultiplyOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : MatrixMultiplyResolution workflow reference)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (leftRef rightRef : Mxx.Ir.WireRef) (left right : Mxx.Matrix)
    (inputWires : reference.inputs.map (wireRef ∘ CoreOperandRef.wire) =
      [leftRef, rightRef])
    (outputsOne : reference.outputs.length = 1)
    (argumentsEvaluate :
      [leftRef, rightRef].mapM (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
        some [.matrix left, .matrix right]) :
    execution.values = [.matrix (Mxx.matrixMultiply left right)] := by
  have executionResolved := execution.resolved
  have nodeResolved := resolution.resolved
  rw [executionResolved] at nodeResolved
  have nodeEq := Option.some.inj nodeResolved
  apply Mxx.Ir.mem_evaluateNode_matrixMultiply_of_arguments runChild samplers params inputs
    execution.before leftRef rightRef left right 1 argumentsEvaluate
  simpa [nodeEq, inputWires, outputsOne] using execution.member

/-- Exact Gaussian sampler node recovered from an accepted operation reference. -/
structure GaussianSampleResolution
    (workflow : Mxx.Ir.Workflow) (reference : OperationRef) where
  matrixType : Mxx.Ir.MatrixTypeExpr
  cutoff : Mxx.Ir.IntExpr
  resolved : resolveNode workflow reference.operation = some {
    kind := .gaussianSample matrixType cutoff
    arguments := reference.inputs.map (wireRef ∘ CoreOperandRef.wire)
    outputCount := reference.outputs.length
  }

theorem verifyGaussianSample_resolution
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    (verified : verifyOperationKind workflow reference (fun kind => match kind with
      | .gaussianSample _ _ => true
      | _ => false) = true) :
    Nonempty (GaussianSampleResolution workflow reference) := by
  unfold verifyOperationKind at verified
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [verifyOperation, resolved] at verified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> simp_all [verifyOperation]
      rename_i matrixType cutoff
      exact ⟨{ matrixType, cutoff, resolved := by simp_all }⟩

theorem verifyExactGaussianSample_resolution
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    {expectedType : Mxx.Ir.MatrixTypeExpr} {expectedCutoff : Mxx.Ir.IntExpr}
    (verified : verifyOperationKind workflow reference (fun kind => match kind with
      | .gaussianSample matrixType cutoff =>
          decide (matrixType = expectedType) && decide (cutoff = expectedCutoff)
      | _ => false) = true) :
    Nonempty (GaussianSampleResolution workflow reference) := by
  unfold verifyOperationKind at verified
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [verifyOperation, resolved] at verified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> simp_all [verifyOperation]
      rename_i matrixType cutoff
      exact ⟨{ matrixType, cutoff, resolved := by simp_all }⟩

/-- Concrete Gaussian outcome with the sampler contract's deterministic hard-support bound. -/
structure GaussianSampleOutcome
    (workflow : Mxx.Ir.Workflow) (reference : OperationRef)
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs) where
  matrixParams : Mxx.SamplerParams
  sample : Mxx.Matrix
  valuesEq : execution.values = [.matrix sample]
  shape : Mxx.Toolkit.MatrixShape sample matrixParams.modulus matrixParams.ringDimension
    matrixParams.rows matrixParams.columns
  norm : Mxx.maxCenteredCoefficientNorm sample ≤ matrixParams.maxCoefficientBound

theorem gaussianSampleOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (resolution : GaussianSampleResolution workflow reference)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (inputsEmpty : reference.inputs = []) (outputsOne : reference.outputs.length = 1)
    (matrixParams : Mxx.SamplerParams)
    (matrixTypeEvaluate : resolution.matrixType.evaluate params resolution.cutoff =
      some matrixParams) :
    Nonempty (GaussianSampleOutcome workflow reference runChild samplers params inputs
      execution) := by
  have executionResolved := execution.resolved
  have nodeResolved := resolution.resolved
  rw [executionResolved] at nodeResolved
  have nodeEq := Option.some.inj nodeResolved
  have member : execution.values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs
      execution.before {
        kind := .gaussianSample resolution.matrixType resolution.cutoff
        arguments := []
        outputCount := 1
      } := by simpa [nodeEq, inputsEmpty, outputsOne] using execution.member
  obtain ⟨rawSample, sampleMember, valuesEq⟩ :=
    Mxx.Ir.mem_evaluateNode_gaussianSample runChild samplers params inputs execution.before
      resolution.matrixType resolution.cutoff matrixParams 1 matrixTypeEvaluate member
  let sample := rawSample.withSamplerParams matrixParams
  have norm : Mxx.maxCenteredCoefficientNorm sample ≤ matrixParams.maxCoefficientBound :=
    contract.gaussianHardSupport matrixParams rawSample sampleMember
  exact ⟨{
    matrixParams
    sample
    valuesEq := by simpa [sample] using valuesEq
    shape := Mxx.Toolkit.withSamplerParams_shape rawSample matrixParams
    norm
  }⟩

/-- Exact preimage sampler node recovered from an accepted operation reference. -/
structure PreimageSampleResolution
    (workflow : Mxx.Ir.Workflow) (reference : OperationRef) where
  matrixType : Mxx.Ir.MatrixTypeExpr
  cutoff : Mxx.Ir.IntExpr
  resolved : resolveNode workflow reference.operation = some {
    kind := .preimageSample matrixType cutoff
    arguments := reference.inputs.map (wireRef ∘ CoreOperandRef.wire)
    outputCount := reference.outputs.length
  }

theorem verifyPreimageSample_resolution
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    (verified : verifyOperationKind workflow reference (fun kind => match kind with
      | .preimageSample _ _ => true
      | _ => false) = true) :
    Nonempty (PreimageSampleResolution workflow reference) := by
  unfold verifyOperationKind at verified
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [verifyOperation, resolved] at verified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> simp_all [verifyOperation]
      rename_i matrixType cutoff
      exact ⟨{ matrixType, cutoff, resolved := by simp_all }⟩

theorem verifyExactPreimageSample_resolution
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    {expectedType : Mxx.Ir.MatrixTypeExpr} {expectedCutoff : Mxx.Ir.IntExpr}
    (verified : verifyOperationKind workflow reference (fun kind => match kind with
      | .preimageSample matrixType cutoff =>
          decide (matrixType = expectedType) && decide (cutoff = expectedCutoff)
      | _ => false) = true) :
    Nonempty (PreimageSampleResolution workflow reference) := by
  unfold verifyOperationKind at verified
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [verifyOperation, resolved] at verified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> simp_all [verifyOperation]
      rename_i matrixType cutoff
      exact ⟨{ matrixType, cutoff, resolved := by simp_all }⟩

/-- Concrete preimage outcome with its exact `B * K = P` equation and hard-support bound. -/
structure PreimageSampleOutcome
    (workflow : Mxx.Ir.Workflow) (reference : OperationRef)
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs) where
  matrixParams : Mxx.SamplerParams
  publicMatrix : Mxx.Matrix
  target : Mxx.Matrix
  preimage : Mxx.Matrix
  valuesEq : execution.values = [.matrix preimage]
  shape : Mxx.Toolkit.MatrixShape preimage matrixParams.modulus matrixParams.ringDimension
    matrixParams.rows matrixParams.columns
  equation : Mxx.matrixMul publicMatrix preimage = target
  norm : Mxx.maxCenteredCoefficientNorm preimage ≤ matrixParams.maxCoefficientBound

theorem preimageSampleOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (resolution : PreimageSampleResolution workflow reference)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (publicRef trapdoorRef targetRef : Mxx.Ir.WireRef)
    (publicMatrix target : Mxx.Matrix)
    (inputWires : reference.inputs.map (wireRef ∘ CoreOperandRef.wire) =
      [publicRef, trapdoorRef, targetRef])
    (outputsOne : reference.outputs.length = 1)
    (argumentsEvaluate :
      [publicRef, trapdoorRef, targetRef].mapM
          (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
        some [.matrix publicMatrix, .trapdoor publicMatrix, .matrix target])
    (matrixParams : Mxx.SamplerParams)
    (matrixTypeEvaluate : resolution.matrixType.evaluate params resolution.cutoff =
      some matrixParams) :
    Nonempty (PreimageSampleOutcome workflow reference runChild samplers params inputs
      execution) := by
  have executionResolved := execution.resolved
  have nodeResolved := resolution.resolved
  rw [executionResolved] at nodeResolved
  have nodeEq := Option.some.inj nodeResolved
  have member : execution.values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs
      execution.before {
        kind := .preimageSample resolution.matrixType resolution.cutoff
        arguments := [publicRef, trapdoorRef, targetRef]
        outputCount := 1
      } := by simpa [nodeEq, inputWires, outputsOne] using execution.member
  obtain ⟨rawPreimage, preimageMember, valuesEq⟩ :=
    Mxx.Ir.mem_evaluateNode_preimageSample_of_arguments runChild samplers params inputs
      execution.before publicRef trapdoorRef targetRef publicMatrix target resolution.matrixType
      resolution.cutoff matrixParams 1 argumentsEvaluate matrixTypeEvaluate member
  let preimage := rawPreimage.withSamplerParams matrixParams
  have contractFacts := contract.preimageContract matrixParams publicMatrix target rawPreimage
    preimageMember
  exact ⟨{
    matrixParams
    publicMatrix
    target
    preimage
    valuesEq := by simpa [preimage] using valuesEq
    shape := Mxx.Toolkit.withSamplerParams_shape rawPreimage matrixParams
    equation := by simpa [preimage] using contractFacts.1
    norm := by simpa [preimage] using contractFacts.2
  }⟩

/-- The accepted preprocessing layout uses the exact certified transition-source index formula. -/
theorem VerifiedDiamondLayout.preprocessingSourceIndexFormulaMatches
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyPreprocessingSourceIndexFormula workflow
      certificate.inputPreprocessing.transitionSourceIndices = true := by
  have checked := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at checked
  simp only [Bool.and_eq_true] at checked
  aesop

/-- The accepted preprocessing layout uses the exact certified transition-target index formula. -/
theorem VerifiedDiamondLayout.preprocessingTargetIndexFormulaMatches
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyPreprocessingTargetIndexFormula workflow
      certificate.inputPreprocessing.transitionTargetIndices = true := by
  have checked := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at checked
  simp only [Bool.and_eq_true] at checked
  aesop

/-- The accepted preprocessing layout uses the exact certified digit-secret index formula. -/
theorem VerifiedDiamondLayout.preprocessingDigitSecretIndexFormulaMatches
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyPreprocessingDigitSecretIndexFormula workflow
      certificate.inputPreprocessing.digitSecretIndices = true := by
  have checked := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at checked
  simp only [Bool.and_eq_true] at checked
  aesop

/-- The accepted decryption layout expands the initial state with the exact certified selector
body before entering the state scan. -/
theorem VerifiedDiamondLayout.initialStateExpansionMatches
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyInitialStateExpansionRef workflow
      certificate.inputInjection.initialStatesExpansion = true := by
  have checked := verified.inputInjectionMatches
  unfold verifyInputInjection at checked
  simp only [Bool.and_eq_true] at checked
  aesop

/-- The accepted state-scan body uses the exact certified source-index formula. -/
theorem VerifiedDiamondLayout.onlineSourceIndexFormulaMatches
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyOnlineSourceIndexFormula workflow certificate.inputInjection.sourceIndices = true := by
  have checked := verified.inputInjectionMatches
  unfold verifyInputInjection at checked
  simp only [Bool.and_eq_true] at checked
  aesop

/-- The accepted state-scan body uses the exact certified transition-index formula. -/
theorem VerifiedDiamondLayout.onlineTransitionIndexFormulaMatches
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    verifyOnlineTransitionIndexFormula workflow
      certificate.inputInjection.transitionIndices = true := by
  have checked := verified.inputInjectionMatches
  unfold verifyInputInjection at checked
  simp only [Bool.and_eq_true] at checked
  aesop

/-- The accepted Diamond preprocessing layout resolves its initial error sampler directly. -/
theorem VerifiedDiamondLayout.initialErrorSampleResolution
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    Nonempty (GaussianSampleResolution workflow
      certificate.inputPreprocessing.initialErrorSample) := by
  have checked := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at checked
  simp only [Bool.and_eq_true] at checked
  have sampleChecked : verifyOperationKind workflow
      certificate.inputPreprocessing.initialErrorSample (fun kind => match kind with
      | .gaussianSample matrixType cutoff =>
          decide (matrixType = {
            modulus := .parameter "diamond_modulus"
            ringDimension := .parameter "diamond_ring_dimension"
            rows := .constant 1
            columns := .add (.constant 4) (.multiply (.constant 2)
              (.parameter "diamond_digit_count"))
          }) && decide (cutoff = .parameter "diamond_error_max_coefficient_bound")
      | _ => false) = true := by aesop
  exact verifyExactGaussianSample_resolution sampleChecked

/-- The accepted Diamond preprocessing layout resolves its exact initial matrix product. -/
theorem VerifiedDiamondLayout.initialPublicProductResolution
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    Nonempty (MatrixMultiplyResolution workflow
      certificate.inputPreprocessing.initialPublicProduct) := by
  apply verifyMatrixMultiply_resolution
  have checked := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at checked
  simp only [Bool.and_eq_true] at checked
  aesop

/-- The accepted Diamond preprocessing layout resolves its exact initial-state addition. -/
theorem VerifiedDiamondLayout.initialStateResolution
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    Nonempty (MatrixAddResolution workflow certificate.inputPreprocessing.initialState) := by
  apply verifyMatrixAdd_resolution
  have checked := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at checked
  simp only [Bool.and_eq_true] at checked
  aesop

/-- The accepted Diamond preprocessing layout resolves the per-transition Gaussian sampler. -/
theorem VerifiedDiamondLayout.transitionErrorSampleResolution
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    Nonempty (GaussianSampleResolution workflow
      certificate.inputPreprocessing.transitionTargets.body.errorSample) := by
  have checked := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at checked
  simp only [Bool.and_eq_true] at checked
  have targetsChecked : verifyParallelTransitionTarget workflow
      certificate.inputPreprocessing.transitionTargets = true := by aesop
  unfold verifyParallelTransitionTarget at targetsChecked
  simp only [Bool.and_eq_true] at targetsChecked
  have sampleChecked : verifyOperationKind workflow
      certificate.inputPreprocessing.transitionTargets.body.errorSample
      (fun kind => match kind with
      | .gaussianSample matrixType cutoff =>
          decide (matrixType = {
            modulus := .parameter "diamond_modulus"
            ringDimension := .parameter "diamond_ring_dimension"
            rows := .constant 2
            columns := .add (.constant 4) (.multiply (.constant 2)
              (.parameter "diamond_digit_count"))
          }) && decide (cutoff = .parameter "diamond_error_max_coefficient_bound")
      | _ => false) = true := by aesop
  exact verifyExactGaussianSample_resolution sampleChecked

/-- The accepted Diamond preprocessing layout resolves the exact transition target product. -/
theorem VerifiedDiamondLayout.transitionSelectorProductResolution
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    Nonempty (MatrixMultiplyResolution workflow
      certificate.inputPreprocessing.transitionTargets.body.selectorProduct) := by
  apply verifyMatrixMultiply_resolution
  have checked := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at checked
  simp only [Bool.and_eq_true] at checked
  have targetsChecked : verifyParallelTransitionTarget workflow
      certificate.inputPreprocessing.transitionTargets = true := by aesop
  unfold verifyParallelTransitionTarget at targetsChecked
  simp only [Bool.and_eq_true] at targetsChecked
  aesop

/-- The accepted Diamond preprocessing layout resolves the exact transition-target addition. -/
theorem VerifiedDiamondLayout.transitionTargetSumResolution
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    Nonempty (MatrixAddResolution workflow
      certificate.inputPreprocessing.transitionTargets.body.targetSum) := by
  apply verifyMatrixAdd_resolution
  have checked := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at checked
  simp only [Bool.and_eq_true] at checked
  have targetsChecked : verifyParallelTransitionTarget workflow
      certificate.inputPreprocessing.transitionTargets = true := by aesop
  unfold verifyParallelTransitionTarget at targetsChecked
  simp only [Bool.and_eq_true] at targetsChecked
  aesop

/-- The accepted Diamond preprocessing layout resolves the exact per-transition preimage sampler. -/
theorem VerifiedDiamondLayout.transitionPreimageSampleResolution
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    Nonempty (PreimageSampleResolution workflow
      certificate.inputPreprocessing.transitionPreimages.body.sample) := by
  have checked := verified.inputPreprocessingMatches
  unfold verifyInputPreprocessing at checked
  simp only [Bool.and_eq_true] at checked
  have preimagesChecked : verifyParallelPreimage workflow
      certificate.inputPreprocessing.transitionPreimages
      (.add (.constant 4) (.multiply (.constant 2)
        (.parameter "diamond_digit_count"))) = true := by aesop
  unfold verifyParallelPreimage at preimagesChecked
  simp only [Bool.and_eq_true] at preimagesChecked
  have sampleChecked : verifyPreimage workflow
      certificate.inputPreprocessing.transitionPreimages.body
      (.add (.constant 4) (.multiply (.constant 2)
        (.parameter "diamond_digit_count"))) = true := by aesop
  unfold verifyPreimage at sampleChecked
  simp only [Bool.and_eq_true] at sampleChecked
  have exactSampleChecked : verifyOperationKind workflow
      certificate.inputPreprocessing.transitionPreimages.body.sample
      (fun kind => match kind with
      | .preimageSample matrixType cutoff =>
          decide (matrixType = {
            modulus := .parameter "diamond_modulus"
            ringDimension := .parameter "diamond_ring_dimension"
            rows := .add (.constant 4) (.multiply (.constant 2)
              (.parameter "diamond_digit_count"))
            columns := .add (.constant 4) (.multiply (.constant 2)
              (.parameter "diamond_digit_count"))
          }) && decide (cutoff = .parameter "diamond_preimage_max_coefficient_bound")
      | _ => false) = true := by aesop
  exact verifyExactPreimageSample_resolution exactSampleChecked

/-- Construct the generic input-injection step from the exact raw matrix equations recovered from
the executable add/multiply nodes and the preimage sampler contract. -/
theorem inputInjectionStep_of_rawEquations
    {q ringDimension stateColumns : Nat} [NeZero q] [NeZero ringDimension] [Fact (1 < q)]
    (state signal base stateError transition selector nextBase transitionError : Mxx.Matrix)
    (stateShape : Mxx.Toolkit.MatrixShape state q ringDimension 1 stateColumns)
    (signalShape : Mxx.Toolkit.MatrixShape signal q ringDimension 1 2)
    (baseShape : Mxx.Toolkit.MatrixShape base q ringDimension 2 stateColumns)
    (stateErrorShape : Mxx.Toolkit.MatrixShape stateError q ringDimension 1 stateColumns)
    (transitionShape :
      Mxx.Toolkit.MatrixShape transition q ringDimension stateColumns stateColumns)
    (selectorShape : Mxx.Toolkit.MatrixShape selector q ringDimension 2 2)
    (nextBaseShape : Mxx.Toolkit.MatrixShape nextBase q ringDimension 2 stateColumns)
    (transitionErrorShape :
      Mxx.Toolkit.MatrixShape transitionError q ringDimension 2 stateColumns)
    (stateRaw : state = Mxx.matrixAdd (Mxx.matrixMul signal base) stateError)
    (transitionRaw : Mxx.matrixMul base transition =
      Mxx.matrixAdd (Mxx.matrixMul selector nextBase) transitionError) :
    Nonempty (MxxWe.InputInjectionStep q ringDimension stateColumns) := by
  have signalProductShape :=
    Mxx.Toolkit.matrixMul_shape signal base signalShape baseShape
  have selectorProductShape :=
    Mxx.Toolkit.matrixMul_shape selector nextBase selectorShape nextBaseShape
  refine ⟨{
    state
    signal
    base
    stateError
    transition
    selector
    nextBase
    transitionError
    stateShape
    signalShape
    baseShape
    stateErrorShape
    transitionShape
    selectorShape
    nextBaseShape
    transitionErrorShape
    stateEquation := ?_
    transitionEquation := ?_
  }⟩
  · calc
      Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns state =
          Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns
            (Mxx.matrixAdd (Mxx.matrixMul signal base) stateError) := by rw [stateRaw]
      _ = Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns
              (Mxx.matrixMul signal base) +
            Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns stateError :=
        Mxx.Toolkit.matrixValue_add q ringDimension 1 stateColumns _ _
          ⟨signalProductShape.modulus, signalProductShape.ringDimension,
            signalProductShape.rows, signalProductShape.columns⟩
          ⟨stateErrorShape.modulus, stateErrorShape.ringDimension,
            stateErrorShape.rows, stateErrorShape.columns⟩
      _ = Mxx.Toolkit.matrixValue q ringDimension 1 2 signal *
              Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns base +
            Mxx.Toolkit.matrixValue q ringDimension 1 stateColumns stateError := by
        rw [Mxx.Toolkit.matrixValue_mul q ringDimension 1 2 stateColumns signal base
          ⟨signalShape.modulus, signalShape.ringDimension, signalShape.rows, signalShape.columns⟩
          ⟨baseShape.modulus, baseShape.ringDimension, baseShape.rows, baseShape.columns⟩]
  · calc
      Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns base *
          Mxx.Toolkit.matrixValue q ringDimension stateColumns stateColumns transition =
          Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns
            (Mxx.matrixMul base transition) := by
        rw [Mxx.Toolkit.matrixValue_mul q ringDimension 2 stateColumns stateColumns base transition
          ⟨baseShape.modulus, baseShape.ringDimension, baseShape.rows, baseShape.columns⟩
          ⟨transitionShape.modulus, transitionShape.ringDimension,
            transitionShape.rows, transitionShape.columns⟩]
      _ =
          Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns
            (Mxx.matrixAdd (Mxx.matrixMul selector nextBase) transitionError) := by
        rw [transitionRaw]
      _ = Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns
              (Mxx.matrixMul selector nextBase) +
            Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns transitionError :=
        Mxx.Toolkit.matrixValue_add q ringDimension 2 stateColumns _ _
          ⟨selectorProductShape.modulus, selectorProductShape.ringDimension,
            selectorProductShape.rows, selectorProductShape.columns⟩
          ⟨transitionErrorShape.modulus, transitionErrorShape.ringDimension,
            transitionErrorShape.rows, transitionErrorShape.columns⟩
      _ = Mxx.Toolkit.matrixValue q ringDimension 2 2 selector *
              Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns nextBase +
            Mxx.Toolkit.matrixValue q ringDimension 2 stateColumns transitionError := by
        rw [Mxx.Toolkit.matrixValue_mul q ringDimension 2 2 stateColumns selector nextBase
          ⟨selectorShape.modulus, selectorShape.ringDimension,
            selectorShape.rows, selectorShape.columns⟩
          ⟨nextBaseShape.modulus, nextBaseShape.ringDimension,
            nextBaseShape.rows, nextBaseShape.columns⟩]

/-- Two local gadget-decomposition nodes with the same evaluated RHS public key and parameters
produce the same value.  The executions may belong to different stages, scopes, and input
environments.  No ciphertext field or artifact identity occurs in the statement. -/
theorem deterministicDecompositionValuesEqual
    (leftRunChild rightRunChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily) (contract : Mxx.MxxBoundedSamplerContract samplers)
    (params : Mxx.Ir.ParamEnvironment)
    (leftInputs rightInputs : Mxx.Ir.Environment)
    (leftWires rightWires : Mxx.Ir.WireEnvironment)
    (leftRef rightRef : Mxx.Ir.WireRef) (publicKey : Mxx.Matrix)
    (matrixType : Mxx.Ir.MatrixTypeExpr) (base digitCount : Mxx.Ir.IntExpr)
    (matrixParams : Mxx.SamplerParams) (evaluatedBase evaluatedDigitCount : Int)
    (outputCount : Nat)
    (leftArguments : [leftRef].mapM (fun wire => Mxx.Ir.lookupWire wire leftWires) =
      some [.matrix publicKey])
    (rightArguments : [rightRef].mapM (fun wire => Mxx.Ir.lookupWire wire rightWires) =
      some [.matrix publicKey])
    (matrixTypeEvaluate : matrixType.evaluate params (.constant 0) = some matrixParams)
    (baseEvaluate : base.evaluate params = some evaluatedBase)
    (digitCountEvaluate : digitCount.evaluate params = some evaluatedDigitCount)
    {leftValues rightValues : List Mxx.Ir.Value}
    (leftMember : leftValues ∈ Mxx.Ir.evaluateNode leftRunChild samplers params leftInputs
      leftWires {
        kind := .gadgetDecompose matrixType base digitCount
        arguments := [leftRef]
        outputCount
      })
    (rightMember : rightValues ∈ Mxx.Ir.evaluateNode rightRunChild samplers params rightInputs
      rightWires {
        kind := .gadgetDecompose matrixType base digitCount
        arguments := [rightRef]
        outputCount
      }) :
    leftValues = rightValues := by
  obtain ⟨left, leftSupport, rfl⟩ :=
    Mxx.Ir.mem_evaluateNode_gadgetDecompose_of_arguments leftRunChild samplers params
      leftInputs leftWires leftRef publicKey matrixType base digitCount matrixParams
      evaluatedBase evaluatedDigitCount outputCount leftArguments matrixTypeEvaluate
      baseEvaluate digitCountEvaluate leftMember
  obtain ⟨right, rightSupport, rfl⟩ :=
    Mxx.Ir.mem_evaluateNode_gadgetDecompose_of_arguments rightRunChild samplers params
      rightInputs rightWires rightRef publicKey matrixType base digitCount matrixParams
      evaluatedBase evaluatedDigitCount outputCount rightArguments matrixTypeEvaluate
      baseEvaluate digitCountEvaluate rightMember
  rw [contract.gadgetDecomposeUnique matrixParams evaluatedBase evaluatedDigitCount.toNat
    publicKey left right leftSupport rightSupport]

structure CheckedDynamicFamilyGetResolution
    (workflow : Mxx.Ir.Workflow) (reference : DynamicFamilyGetRef) where
  resolved : resolveNode workflow reference.operation = some ({
    kind := .familyGetDynamic
    arguments := [wireRef reference.family.wire, wireRef reference.index.wire]
    outputCount := 1
  } : Mxx.Ir.Node)

theorem checkedDynamicFamilyGetResolution_of_verified
    {workflow : Mxx.Ir.Workflow} {reference : DynamicFamilyGetRef} {family : CoreWireRef}
    (verified : verifyDynamicGet workflow reference family = true) :
    Nonempty (CheckedDynamicFamilyGetResolution workflow reference) := by
  unfold verifyDynamicGet at verified
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [resolved] at verified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> simp_all [Bool.and_eq_true, decide_eq_true_eq]
      constructor
      simp_all

/-- Concrete semantics of one exact dynamic-family lookup execution. -/
theorem checkedDynamicFamilyGetOutcome
    {workflow : Mxx.Ir.Workflow} {reference : DynamicFamilyGetRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : CheckedDynamicFamilyGetResolution workflow reference)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (family : List Mxx.Ir.Value) (index : Int)
    (argumentsEvaluate :
      [wireRef reference.family.wire, wireRef reference.index.wire].mapM
        (fun wire => Mxx.Ir.lookupWire wire execution.before) =
          some [.family family, .integer index]) :
    execution.values =
      [family[index.toNat]?.getD (.invalid "FamilyGetDynamic index out of range")] := by
  have executionResolved := execution.resolved
  have checkedResolved := resolution.resolved
  rw [executionResolved] at checkedResolved
  have nodeEq := Option.some.inj checkedResolved
  apply Mxx.Ir.mem_evaluateNode_familyGetDynamic_of_arguments runChild samplers params inputs
    execution.before (wireRef reference.family.wire) (wireRef reference.index.wire)
    family index 1 argumentsEvaluate
  simpa [nodeEq] using execution.member

theorem oneSourceParallelGatherFacts
    {workflow : Mxx.Ir.Workflow} {reference : ParallelGatherRef}
    {source : CoreOperandRef} {bodySource output : CoreWireRef}
    {get : DynamicFamilyGetRef}
    (verified : verifyParallelGather workflow reference = true)
    (sources : reference.sourceFamilies = [source])
    (bodySources : reference.bodySources = [bodySource])
    (gets : reference.gets = [get])
    (outputs : reference.outputFamilies = [output]) :
    verifyParallelLoop workflow reference.parallelLoop = true ∧
      verifyDynamicGet workflow get bodySource = true ∧
      reference.parallelLoop.arguments = [reference.indexFamily, source] ∧
      reference.parallelLoop.bodyInputs = [reference.bodyIndex, bodySource] ∧
      reference.parallelLoop.bodyOutputs = [get.output] ∧
      reference.parallelLoop.outputs = [output] ∧
      get.index.wire = reference.bodyIndex ∧ get.family.wire = bodySource := by
  unfold verifyParallelGather at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  simp [sources, bodySources, gets, outputs] at verified
  have getFamilyEq : get.family.wire = bodySource := by
    have getVerified : verifyDynamicGet workflow get bodySource = true := by aesop
    unfold verifyDynamicGet at getVerified
    simp only [Bool.and_eq_true, decide_eq_true_eq] at getVerified
    aesop
  aesop

theorem oneSourceParallelGather_childOutcome
    {workflow : Mxx.Ir.Workflow} {reference : ParallelGatherRef}
    {source : CoreOperandRef} {bodySource output : CoreWireRef}
    {get : DynamicFamilyGetRef}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {childInputs childValues : List Mxx.Ir.Value} {index : Nat}
    {sourceValues : List Mxx.Ir.Value}
    (verified : verifyParallelGather workflow reference = true)
    (sources : reference.sourceFamilies = [source])
    (bodySources : reference.bodySources = [bodySource])
    (gets : reference.gets = [get])
    (outputs : reference.outputFamilies = [output])
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (definitionFound : Mxx.Ir.lookupDefinition
      reference.parallelLoop.bodyScope.definitionName stage.program.definitions = some body)
    (ssaOrder : verifyScopeSsaOrder body = true)
    (childInputsEq : childInputs = [.integer (Int.ofNat index), .family sourceValues])
    (childMember : childValues ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.parallelLoop.bodyScope.definitionName params childInputs) :
    childValues = [sourceValues[index]?.getD
      (.invalid "FamilyGetDynamic index out of range")] := by
  obtain ⟨loopVerified, getVerified, argumentsEq, bodyInputsEq, bodyOutputsEq,
      _, getIndexEq, getFamilyEq⟩ :=
    oneSourceParallelGatherFacts verified sources bodySources gets outputs
  obtain ⟨path⟩ := childExecutionPath_of_outcome definitionFound childMember
  obtain ⟨getResolution⟩ := checkedDynamicFamilyGetResolution_of_verified getVerified
  have getOperationEq : get.operation = get.output.node := by
    have checked := getVerified
    unfold verifyDynamicGet verifyOperationOutput at checked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
    aesop
  have getScopeResolved : resolveScope workflow get.operation = some body := by
    have outputStage : get.output.node.stage = reference.parallelLoop.operation.stage := by
      have checked := loopVerified
      unfold verifyParallelLoop at checked
      simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
      simp [bodyOutputsEq] at checked
      aesop
    have outputScope : get.output.node.scope = reference.parallelLoop.bodyScope := by
      have checked := loopVerified
      unfold verifyParallelLoop at checked
      simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
      simp [bodyOutputsEq] at checked
      aesop
    unfold resolveScope at bodyResolved ⊢
    simpa [getOperationEq, outputStage, outputScope] using bodyResolved
  have getNodeInScope := resolveNode_scopeNode getScopeResolved getResolution.resolved
  obtain ⟨getExecution, getRooted⟩ :=
    path.rootedReferencedNodeExecution getNodeInScope getResolution.resolved
  have getNodeEq : getExecution.node = {
      kind := .familyGetDynamic
      arguments := [wireRef get.family.wire, wireRef get.index.wire]
      outputCount := 1
    } := by
    have checkedResolved := getResolution.resolved
    rw [getExecution.resolved] at checkedResolved
    exact Option.some.inj checkedResolved
  obtain ⟨namesNodup, wireLength, argumentLength, bodyInputWires⟩ :=
    verifyParallelLoop_bodyInputBindings loopVerified bodyResolved
  have inputLength : body.inputNames.length = childInputs.length := by
    calc
      body.inputNames.length = reference.parallelLoop.arguments.length := argumentLength.symm
      _ = 2 := by simp only [argumentsEq, List.length_cons, List.length_nil]
      _ = childInputs.length := by
        simp only [childInputsEq, List.length_cons, List.length_nil]
  have exactInputWires : scopeInputWires body =
      [wireRef reference.bodyIndex, wireRef bodySource] := by
    calc
      scopeInputWires body = reference.parallelLoop.bodyInputs.map wireRef :=
        bodyInputWires.symm
      _ = [wireRef reference.bodyIndex, wireRef bodySource] := by
        simp only [bodyInputsEq, List.map_cons, List.map_nil]
  have inputLocations :
      reference.bodyIndex.node.stage = reference.parallelLoop.operation.stage ∧
      reference.bodyIndex.node.scope = reference.parallelLoop.bodyScope ∧
      bodySource.node.stage = reference.parallelLoop.operation.stage ∧
      bodySource.node.scope = reference.parallelLoop.bodyScope := by
    have checked := loopVerified
    unfold verifyParallelLoop at checked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
    simp [bodyInputsEq] at checked
    aesop
  have indexScopeResolved : resolveScope workflow reference.bodyIndex.node = some body := by
    simpa [resolveScope, inputLocations.1, inputLocations.2.1] using bodyResolved
  have sourceScopeResolved : resolveScope workflow bodySource.node = some body := by
    simpa [resolveScope, inputLocations.2.2.1, inputLocations.2.2.2] using bodyResolved
  have indexValid : ∃ node, body.nodes[(wireRef reference.bodyIndex).node]? = some node ∧
      (wireRef reference.bodyIndex).port < node.outputCount := by
    have indexChecked : verifyWire workflow reference.bodyIndex = true := by
      have checked := loopVerified
      unfold verifyParallelLoop at checked
      simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
      simp [bodyInputsEq] at checked
      aesop
    exact verifyWire_scopeValid indexChecked indexScopeResolved
  have sourceValid : ∃ node, body.nodes[(wireRef bodySource).node]? = some node ∧
      (wireRef bodySource).port < node.outputCount := by
    have sourceChecked : verifyWire workflow bodySource = true := by
      have checked := loopVerified
      unfold verifyParallelLoop at checked
      simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
      simp [bodyInputsEq] at checked
      aesop
    exact verifyWire_scopeValid sourceChecked sourceScopeResolved
  cases names : body.inputNames with
  | nil => simp [names, childInputsEq] at inputLength
  | cons indexName tail =>
    cases tail with
    | nil => simp [names, childInputsEq] at inputLength
    | cons sourceName rest =>
      have restEmpty : rest = [] := by simpa [names, childInputsEq] using inputLength
      subst rest
      have indexLookup : Mxx.Ir.lookupWire (wireRef reference.bodyIndex)
          getExecution.before = some (.integer (Int.ofNat index)) := by
        apply getRooted.inputWireValue ssaOrder getScopeResolved namesNodup wireLength inputLength
          0 indexName (wireRef reference.bodyIndex) (.integer (Int.ofNat index))
        · simp [names]
        · simp [exactInputWires]
        · simp [childInputsEq]
        · simp [getNodeEq, getIndexEq]
        · exact indexValid
      have sourceLookup : Mxx.Ir.lookupWire (wireRef bodySource) getExecution.before =
          some (.family sourceValues) := by
        apply getRooted.inputWireValue ssaOrder getScopeResolved namesNodup wireLength inputLength
          1 sourceName (wireRef bodySource) (.family sourceValues)
        · simp [names]
        · simp [exactInputWires]
        · simp [childInputsEq]
        · simp [getNodeEq, getFamilyEq]
        · exact sourceValid
      have getArguments :
          [wireRef get.family.wire, wireRef get.index.wire].mapM
              (fun wire => Mxx.Ir.lookupWire wire getExecution.before) =
            some [.family sourceValues, .integer (Int.ofNat index)] := by
        rw [getFamilyEq, getIndexEq]
        exact lookupWirePair sourceLookup indexLookup
      have getValues := checkedDynamicFamilyGetOutcome getResolution getExecution sourceValues
        (Int.ofNat index) getArguments
      have getOutputPort : get.output.port = 0 := by
        have checked := getVerified
        unfold verifyDynamicGet verifyOperationOutput verifyWire at checked
        simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
        rw [← getOperationEq, getResolution.resolved] at checked
        simp at checked
        omega
      have getOutput := getRooted.outputFinal 0 (by simp [getValues])
      have scopeOutputs := verifyParallelLoop_bodyOutputs loopVerified bodyResolved
      rw [bodyOutputsEq] at scopeOutputs
      rw [path.outputs]
      cases bodyOutputs : body.outputs with
      | nil => simp [bodyOutputs] at scopeOutputs
      | cons head tail =>
        cases tail with
        | nil =>
          rcases head with ⟨name, wire⟩
          simp [bodyOutputs] at scopeOutputs
          subst wire
          have outputNodeEq : get.operation.node = get.output.node.node :=
            congrArg CoreNodeRef.node getOperationEq
          simp only [Mxx.Ir.collectOutputs, List.map_cons, List.map_nil, wireRef,
            getOutputPort]
          rw [← outputNodeEq]
          rw [getOutput]
          simp [getValues]
        | cons next rest => simp [bodyOutputs] at scopeOutputs

end MxxWe.Certificate
