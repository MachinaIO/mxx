import Mxx.Certificate.Workflow
import Mxx.Ir.ExecutionFacts

namespace Mxx.Certificate

/-- One selected execution of a program scope.  The membership proof is the semantic content;
this is not a runtime log and does not define a second evaluator.  Root-scope membership exposes
nested scope and loop executions through the path and loop inversion theorems in
`Mxx.Ir.ExecutionFacts`. -/
structure ScopeExecution (samplers : MxxSamplerFamily) where
  program : Mxx.Ir.Prog
  fuel : Nat
  scope : Mxx.Ir.Scope
  params : Mxx.Ir.ParamEnvironment
  inputs : Mxx.Ir.Environment
  output : Mxx.Ir.Environment
  outputMember : output ∈ Mxx.Ir.denoteScopeWithFuel samplers program fuel scope params inputs

/-- One selected execution of a workflow stage, witnessed directly by the existing program
denotation. -/
structure StageExecution (samplers : MxxSamplerFamily) where
  stage : Mxx.Ir.Stage
  params : Mxx.Ir.ParamEnvironment
  protocolInputs : Mxx.Ir.Environment
  priorStages : Mxx.Ir.StageEnvironment
  inputs : Mxx.Ir.Environment
  output : Mxx.Ir.Environment
  inputsEq : inputs = Mxx.Ir.stageInputs protocolInputs priorStages stage
  outputMember : output ∈ Mxx.Ir.denote samplers stage.program params inputs

/-- The value transported by one artifact edge.  Both equalities refer to the environments of
the corresponding semantic stage executions. -/
structure ArtifactBindingExecution where
  sourceStage : String
  sourceOutput : String
  destinationStage : String
  destinationInput : String
  value : Mxx.Ir.Value
  protocolInputs : Mxx.Ir.Environment
  sourceStages : Mxx.Ir.StageEnvironment
  destinationEnvironment : Mxx.Ir.Environment
  destination : Mxx.Ir.Stage
  valueEq : value = Mxx.Ir.resolveStageInput protocolInputs sourceStages
    (.artifact sourceStage sourceOutput)
  destinationEnvironmentEq : destinationEnvironment =
    Mxx.Ir.stageInputs protocolInputs sourceStages destination

/-- One selected branch through `evaluateStages`.  Each constructor stores membership in the
existing stage denotation, so this relation witnesses the evaluator rather than reimplementing
it. -/
inductive StageExecutions
    (samplers : MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (protocolInputs : Mxx.Ir.Environment) :
    List Mxx.Ir.Stage → Mxx.Ir.StageEnvironment → Mxx.Ir.StageEnvironment →
      List (StageExecution samplers) → Prop where
  | nil (state : Mxx.Ir.StageEnvironment) :
      StageExecutions samplers params protocolInputs [] state state []
  | cons
      (stage : Mxx.Ir.Stage)
      (tail : List Mxx.Ir.Stage)
      (state final : Mxx.Ir.StageEnvironment)
      (output : Mxx.Ir.Environment)
      (executions : List (StageExecution samplers))
      (outputMember : output ∈ Mxx.Ir.denote samplers stage.program params
        (Mxx.Ir.stageInputs protocolInputs state stage))
      (tailExecution : StageExecutions samplers params protocolInputs tail
        (state ++ [(stage.id, output)]) final executions) :
      StageExecutions samplers params protocolInputs (stage :: tail) state final
        ({
          stage
          params
          protocolInputs
          priorStages := state
          inputs := Mxx.Ir.stageInputs protocolInputs state stage
          output
          inputsEq := rfl
          outputMember
        } :: executions)

private theorem mem_evaluateStages_iff_stageExecutions
    (samplers : MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (protocolInputs : Mxx.Ir.Environment)
    (stages : List Mxx.Ir.Stage)
    (states : List Mxx.Ir.StageEnvironment)
    (final : Mxx.Ir.StageEnvironment) :
    final ∈ Mxx.Ir.evaluateStages samplers params protocolInputs stages states ↔
      ∃ initial ∈ states, ∃ executions,
        StageExecutions samplers params protocolInputs stages initial final executions := by
  induction stages generalizing states final with
  | nil =>
      simp only [Mxx.Ir.evaluateStages]
      constructor
      · intro member
        exact ⟨final, member, [], .nil final⟩
      · rintro ⟨initial, member, executions, execution⟩
        cases execution
        exact member
  | cons stage tail induction =>
      rw [Mxx.Ir.evaluateStages, induction]
      constructor
      · rintro ⟨next, nextMember, executions, tailExecution⟩
        simp only [List.mem_flatMap, List.mem_map] at nextMember
        obtain ⟨state, stateMember, output, outputMember, rfl⟩ := nextMember
        exact ⟨state, stateMember, _ :: executions,
          .cons _ _ _ _ _ _ outputMember tailExecution⟩
      · rintro ⟨state, stateMember, executions, execution⟩
        cases execution with
        | cons _ _ _ _ output tailExecutions outputMember tailExecution =>
            refine ⟨_, ?_, tailExecutions, tailExecution⟩
            simp only [List.mem_flatMap, List.mem_map]
            exact ⟨state, stateMember, output, outputMember, rfl⟩

def StageExecution.rootScope
    {samplers : MxxSamplerFamily}
    (execution : StageExecution samplers) : ScopeExecution samplers := {
  program := execution.stage.program
  fuel := execution.stage.program.definitions.length + 1
  scope := execution.stage.program.root
  params := execution.params
  inputs := execution.inputs
  output := execution.output
  outputMember := by simpa [Mxx.Ir.denote] using execution.outputMember
}

def StageExecution.artifactBindings
    {samplers : MxxSamplerFamily}
    (execution : StageExecution samplers) : List ArtifactBindingExecution :=
  execution.stage.inputs.filterMap fun (destinationInput, source) =>
    match source with
    | .protocol _ => none
    | .artifact sourceStage sourceOutput => some {
        sourceStage
        sourceOutput
        destinationStage := execution.stage.id
        destinationInput
        value := Mxx.Ir.resolveStageInput execution.protocolInputs execution.priorStages
          (.artifact sourceStage sourceOutput)
        protocolInputs := execution.protocolInputs
        sourceStages := execution.priorStages
        destinationEnvironment := execution.inputs
        destination := execution.stage
        valueEq := rfl
        destinationEnvironmentEq := execution.inputsEq
      }

private structure WorkflowExecutionWitness
    (samplers : MxxSamplerFamily)
    (workflow : Mxx.Ir.Workflow)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs output : Mxx.Ir.Environment) where
  final : Mxx.Ir.StageEnvironment
  executions : List (StageExecution samplers)
  execution : StageExecutions samplers params inputs workflow.stages [] final executions
  finalMember : final ∈ Mxx.Ir.evaluateStages samplers params inputs workflow.stages [[]]
  outputEq :
    (Mxx.Ir.lookupStage workflow.entrypoint final).getD
      [("__workflow_error", .invalid "entrypoint did not execute")] = output

private theorem workflowExecutionWitness_exists
    {samplers : MxxSamplerFamily}
    {workflow : Mxx.Ir.Workflow}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs output : Mxx.Ir.Environment}
    (member : output ∈ Mxx.Ir.denoteWorkflow samplers workflow params inputs) :
    Nonempty (WorkflowExecutionWitness samplers workflow params inputs output) := by
  unfold Mxx.Ir.denoteWorkflow at member
  simp only [List.mem_map] at member
  obtain ⟨final, finalMember, outputEq⟩ := member
  obtain ⟨initial, initialMember, executions, execution⟩ :=
    (mem_evaluateStages_iff_stageExecutions samplers params inputs workflow.stages [[]] final).mp
      finalMember
  simp only [List.mem_singleton] at initialMember
  subst initial
  exact ⟨{
    final
    executions
    execution
    finalMember
    outputEq
  }⟩

/-- A semantic workflow witness.  Its authoritative execution claim is membership in the
existing `denoteWorkflow`; the other fields expose internal executions when a soundness proof
needs them. -/
structure WorkflowExecutionTrace
    (samplers : MxxSamplerFamily)
    (workflow : Mxx.Ir.Workflow)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment) where
  stageExecutions : List (StageExecution samplers)
  scopeExecutions : List (ScopeExecution samplers)
  artifactBindings : List ArtifactBindingExecution
  finalStages : Mxx.Ir.StageEnvironment
  entrypointOutput : Mxx.Ir.Environment
  stageExecutionWitness : StageExecutions samplers params inputs workflow.stages [] finalStages
    stageExecutions
  scopeExecutionsEq : scopeExecutions = stageExecutions.map StageExecution.rootScope
  artifactBindingsEq : artifactBindings = stageExecutions.flatMap StageExecution.artifactBindings
  entrypointEq :
    (Mxx.Ir.lookupStage workflow.entrypoint finalStages).getD
      [("__workflow_error", .invalid "entrypoint did not execute")] = entrypointOutput
  entrypointMember : entrypointOutput ∈ Mxx.Ir.denoteWorkflow samplers workflow params inputs

def WorkflowExecutionTrace.erase
    {samplers : MxxSamplerFamily}
    {workflow : Mxx.Ir.Workflow}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (trace : WorkflowExecutionTrace samplers workflow params inputs) : Mxx.Ir.Environment :=
  trace.entrypointOutput

private noncomputable def WorkflowExecutionTrace.canonical
    {samplers : MxxSamplerFamily}
    {workflow : Mxx.Ir.Workflow}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs output : Mxx.Ir.Environment}
    (member : output ∈ Mxx.Ir.denoteWorkflow samplers workflow params inputs) :
    WorkflowExecutionTrace samplers workflow params inputs :=
  let witness := Classical.choice (workflowExecutionWitness_exists member)
  {
    stageExecutions := witness.executions
    scopeExecutions := witness.executions.map StageExecution.rootScope
    artifactBindings := witness.executions.flatMap StageExecution.artifactBindings
    finalStages := witness.final
    entrypointOutput := output
    stageExecutionWitness := witness.execution
    scopeExecutionsEq := rfl
    artifactBindingsEq := rfl
    entrypointEq := witness.outputEq
    entrypointMember := by
      unfold Mxx.Ir.denoteWorkflow
      simp only [List.mem_map]
      exact ⟨witness.final, witness.finalMember, witness.outputEq⟩
  }

@[simp] private theorem WorkflowExecutionTrace.erase_canonical
    {samplers : MxxSamplerFamily}
    {workflow : Mxx.Ir.Workflow}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs output : Mxx.Ir.Environment}
    (member : output ∈ Mxx.Ir.denoteWorkflow samplers workflow params inputs) :
    (WorkflowExecutionTrace.canonical member).erase = output := by
  rfl

/-- Trace-producing view of `denoteWorkflow`.  It attaches proof witnesses to the existing
support; it does not duplicate or replace workflow execution. -/
noncomputable def denoteWorkflowTraces
    (samplers : MxxSamplerFamily)
    (workflow : Mxx.Ir.Workflow)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment) :
    List (WorkflowExecutionTrace samplers workflow params inputs) :=
  (Mxx.Ir.denoteWorkflow samplers workflow params inputs).attach.map fun output =>
    WorkflowExecutionTrace.canonical output.property

/-- Erasing the proof witnesses gives exactly the pre-existing workflow denotation. -/
@[simp] theorem erase_denoteWorkflowTraces
    (samplers : MxxSamplerFamily)
    (workflow : Mxx.Ir.Workflow)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment) :
    (denoteWorkflowTraces samplers workflow params inputs).map
        WorkflowExecutionTrace.erase =
      Mxx.Ir.denoteWorkflow samplers workflow params inputs := by
  simp [denoteWorkflowTraces, WorkflowExecutionTrace.erase_canonical]

/-- A pure-program outcome paired with the exact existing `denotePure` computation.  `none` is
retained, rather than silently dropping an invalid or nondeterministic pure execution. -/
structure PureProgramExecution where
  program : Mxx.Ir.Prog
  params : Mxx.Ir.ParamEnvironment
  inputs : Mxx.Ir.Environment
  output : Option Mxx.Ir.Environment
  outputEq : output = Mxx.Ir.denotePure program params inputs

def PureProgramExecution.canonical
    (program : Mxx.Ir.Prog)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment) : PureProgramExecution := {
  program
  params
  inputs
  output := Mxx.Ir.denotePure program params inputs
  outputEq := rfl
}

/-- Concrete comparator data, including the output environment of an optional comparator map and
the polarity-normalized failure bit. -/
structure ComparatorOutcome where
  inputs : Mxx.Ir.Environment
  output : Option Mxx.Ir.Environment
  failure : Bool

structure ComparatorExecution where
  specification : ComparatorSpec
  workflowOutput : Mxx.Ir.Environment
  idealOutput : Option Mxx.Ir.Environment
  outcome : ComparatorOutcome

private def endpointFor
    (bundle : ClosedProtocolBundle)
    (specification : EndpointSpecId) : Option EndpointAnchor :=
  bundle.endpoints.entries.find? fun endpoint => endpoint.specification = specification

private def comparatorBindings : ComparatorSpec → List ComparatorEndpointBinding
  | .equality bindings | .equalityAfterMap _ bindings => bindings

private def endpointValues
    (bundle : ClosedProtocolBundle)
    (workflowOutput idealOutput : Mxx.Ir.Environment)
    (binding : ComparatorEndpointBinding) : Option (Mxx.Ir.Value × Mxx.Ir.Value) := do
  let endpoint ← endpointFor bundle binding.endpoint
  let actual ← Mxx.Ir.lookupEnvironment endpoint.workflowOutput workflowOutput
  let ideal ← Mxx.Ir.lookupEnvironment endpoint.idealOutput idealOutput
  return (actual, ideal)

private def comparatorInputEnvironment
    (bundle : ClosedProtocolBundle)
    (workflowOutput idealOutput : Mxx.Ir.Environment) : Mxx.Ir.Environment :=
  (comparatorBindings bundle.comparator).flatMap fun binding =>
    match endpointValues bundle workflowOutput idealOutput binding with
    | none =>
        let actual := (binding.actualInput, .invalid "missing endpoint actual value")
        if binding.idealInput.isEmpty then [actual]
        else [actual, (binding.idealInput, .invalid "missing endpoint ideal value")]
    | some (actual, ideal) =>
        let actualBinding := (binding.actualInput, actual)
        if binding.idealInput.isEmpty then [actualBinding]
        else [actualBinding, (binding.idealInput, ideal)]

private def resultIsFailure
    (output : Mxx.Ir.Environment)
    (binding : ComparatorEndpointBinding) : Bool :=
  match Mxx.Ir.lookupEnvironment binding.resultOutput output with
  | some (.boolean value) => value == binding.failureValue
  | _ => true

private def equalityComparatorOutput
    (bundle : ClosedProtocolBundle)
    (workflowOutput idealOutput : Mxx.Ir.Environment)
    (bindings : List ComparatorEndpointBinding) : Mxx.Ir.Environment :=
  bindings.map fun binding =>
    let failed := match endpointValues bundle workflowOutput idealOutput binding with
      | some (actual, ideal) => !actual.equal ideal
      | none => true
    let result := if failed then binding.failureValue else !binding.failureValue
    (binding.resultOutput, .boolean result)

/-- Existing IR primitives give the complete comparator meaning.  Equality synthesizes its
boolean result directly; equality-after-map delegates to `denotePure`. -/
def denoteComparator
    (bundle : ClosedProtocolBundle)
    (params : Mxx.Ir.ParamEnvironment)
    (workflowOutput : Mxx.Ir.Environment)
    (idealOutput : Option Mxx.Ir.Environment) : ComparatorOutcome :=
  match idealOutput with
  | none => { inputs := [], output := none, failure := true }
  | some ideal =>
      let inputs := comparatorInputEnvironment bundle workflowOutput ideal
      match bundle.comparator with
      | .equality bindings =>
          let output := equalityComparatorOutput bundle workflowOutput ideal bindings
          {
            inputs
            output := some output
            failure := bindings.any (resultIsFailure output)
          }
      | .equalityAfterMap program bindings =>
          let output := Mxx.Ir.denotePure program params inputs
          {
            inputs
            output
            failure := match output with
              | none => true
              | some environment => bindings.any (resultIsFailure environment)
          }

/-- Public semantic reduction for the single-endpoint equality comparator used by the initial
closed protocols. Bundle verification establishes the endpoint lookup and unique result name;
the theorem exposes only the equality fact needed by correctness proofs. -/
theorem denoteComparator_singleEquality_success
    (bundle : ClosedProtocolBundle)
    (params : Mxx.Ir.ParamEnvironment)
    (workflowOutput idealOutput : Mxx.Ir.Environment)
    (binding : ComparatorEndpointBinding)
    (endpoint : EndpointAnchor)
    (value : Bool)
    (comparator : bundle.comparator = .equality [binding])
    (endpointFound : bundle.endpoints.entries.find? (fun candidate =>
      candidate.specification = binding.endpoint) = some endpoint)
    (actualLookup : Mxx.Ir.lookupEnvironment endpoint.workflowOutput workflowOutput =
      some (.boolean value))
    (idealLookup : Mxx.Ir.lookupEnvironment endpoint.idealOutput idealOutput =
      some (.boolean value)) :
    (denoteComparator bundle params workflowOutput (some idealOutput)).failure = false := by
  unfold denoteComparator
  rw [comparator]
  simp only [equalityComparatorOutput, List.map_singleton]
  have endpointEqual : endpointValues bundle workflowOutput idealOutput binding =
      some (.boolean value, .boolean value) := by
    simp [endpointValues, endpointFor, endpointFound, actualLookup, idealLookup]
  rw [endpointEqual]
  simp [Mxx.Ir.Value.equal, resultIsFailure, Mxx.Ir.lookupEnvironment]

def protocolInputName
    (bundle : ClosedProtocolBundle)
    (input : ProtocolInputId) : Option String :=
  (bundle.inputContract.inputs.find? fun entry => entry.1 = input).map fun entry => entry.2.1

def protocolInputBoundTo
    (bundle : ClosedProtocolBundle)
    (destination : ProtocolInputDestination) : Option ProtocolInputId :=
  (bundle.inputBindings.find? fun binding => binding.destinations.contains destination).map
    (·.input)

def protocolDestinationValue
    (bundle : ClosedProtocolBundle)
    (protocolInputs : Mxx.Ir.Environment)
    (destination : ProtocolInputDestination) : Mxx.Ir.Value :=
  match protocolInputBoundTo bundle destination >>= protocolInputName bundle with
  | some name => Mxx.Ir.lookupEnvironment name protocolInputs |>.getD
      (.invalid s!"missing protocol input {name}")
  | none => .invalid "unbound protocol input destination"

def requirementInputEnvironment
    (bundle : ClosedProtocolBundle)
    (protocolInputs : Mxx.Ir.Environment)
    (index : Nat)
    (program : Mxx.Ir.Prog) : Mxx.Ir.Environment :=
  program.root.inputNames.map fun name =>
    (name, protocolDestinationValue bundle protocolInputs (.requirement index name))

def idealInputEnvironment
    (bundle : ClosedProtocolBundle)
    (protocolInputs : Mxx.Ir.Environment) : Mxx.Ir.Environment :=
  bundle.ideal.root.inputNames.map fun name =>
    (name, protocolDestinationValue bundle protocolInputs (.ideal name))

structure ClosedProtocolOutcome where
  workflowOutput : Mxx.Ir.Environment
  requirementOutputs : List (Option Mxx.Ir.Environment)
  idealOutput : Option Mxx.Ir.Environment
  comparator : ComparatorOutcome

def denoteClosedProtocolOutcome
    (bundle : ClosedProtocolBundle)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs workflowOutput : Mxx.Ir.Environment) : ClosedProtocolOutcome :=
  let requirements := bundle.requirements.mapIdx fun index program =>
    Mxx.Ir.denotePure program params (requirementInputEnvironment bundle inputs index program)
  let ideal := Mxx.Ir.denotePure bundle.ideal params (idealInputEnvironment bundle inputs)
  {
    workflowOutput
    requirementOutputs := requirements
    idealOutput := ideal
    comparator := denoteComparator bundle params workflowOutput ideal
  }

/-- Bundle-level semantic witness.  Every component is tied to the same workflow output and the
same concrete parameter/input environments; no certificate-provided semantic fact occurs here. -/
structure ClosedProtocolExecutionTrace
    (samplers : MxxSamplerFamily)
    (bundle : ClosedProtocolBundle)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment) where
  workflow : WorkflowExecutionTrace samplers bundle.workflow params inputs
  requirements : List PureProgramExecution
  ideal : PureProgramExecution
  comparator : ComparatorExecution
  requirementPrograms : requirements.map (·.program) = bundle.requirements
  requirementParams : requirements.map (·.params) = bundle.requirements.map fun _ => params
  requirementInputsEq : requirements.map (·.inputs) =
    bundle.requirements.mapIdx fun index program =>
      requirementInputEnvironment bundle inputs index program
  idealProgram : ideal.program = bundle.ideal
  idealParams : ideal.params = params
  idealInputsEq : ideal.inputs = idealInputEnvironment bundle inputs
  comparatorSpec : comparator.specification = bundle.comparator
  comparatorWorkflowOutput : comparator.workflowOutput = workflow.entrypointOutput
  comparatorIdealOutput : comparator.idealOutput = ideal.output
  comparatorOutcomeEq : comparator.outcome =
    denoteComparator bundle params workflow.entrypointOutput ideal.output

def ClosedProtocolExecutionTrace.erase
    {samplers : MxxSamplerFamily}
    {bundle : ClosedProtocolBundle}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (trace : ClosedProtocolExecutionTrace samplers bundle params inputs) :
    ClosedProtocolOutcome := {
  workflowOutput := trace.workflow.entrypointOutput
  requirementOutputs := trace.requirements.map (·.output)
  idealOutput := trace.ideal.output
  comparator := trace.comparator.outcome
}

private theorem map_fst_zipIdx {α : Type} (values : List α) (start : Nat) :
    (values.zipIdx start).map Prod.fst = values := by
  induction values generalizing start with
  | nil => rfl
  | cons head tail induction =>
      simp only [List.zipIdx, List.map_cons]
      rw [induction]

private theorem map_const_zipIdx {α β : Type} (values : List α) (start : Nat) (value : β) :
    (values.zipIdx start).map (fun _ => value) = values.map fun _ => value := by
  change List.map (Function.const (α × Nat) value) (values.zipIdx start) =
    List.map (Function.const α value) values
  rw [List.map_const, List.map_const, List.length_zipIdx]

private def ClosedProtocolExecutionTrace.canonical
    {samplers : MxxSamplerFamily}
    {bundle : ClosedProtocolBundle}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (workflow : WorkflowExecutionTrace samplers bundle.workflow params inputs) :
    ClosedProtocolExecutionTrace samplers bundle params inputs :=
  let requirements := bundle.requirements.mapIdx fun index program =>
    PureProgramExecution.canonical program params
      (requirementInputEnvironment bundle inputs index program)
  let ideal := PureProgramExecution.canonical bundle.ideal params
    (idealInputEnvironment bundle inputs)
  let comparatorOutcome := denoteComparator bundle params workflow.entrypointOutput ideal.output
  {
    workflow
    requirements
    ideal
    comparator := {
      specification := bundle.comparator
      workflowOutput := workflow.entrypointOutput
      idealOutput := ideal.output
      outcome := comparatorOutcome
    }
    requirementPrograms := by
      simp only [requirements, List.mapIdx_eq_zipIdx_map, List.map_map,
        PureProgramExecution.canonical]
      exact map_fst_zipIdx bundle.requirements 0
    requirementParams := by
      simp only [requirements, List.mapIdx_eq_zipIdx_map, List.map_map,
        PureProgramExecution.canonical]
      calc
        _ = bundle.requirements.zipIdx.map (fun _ => params) := by
          apply List.map_congr_left
          rintro ⟨program, index⟩ member
          rfl
        _ = bundle.requirements.map (fun _ => params) :=
          map_const_zipIdx bundle.requirements 0 params
    requirementInputsEq := by
      simp [requirements, PureProgramExecution.canonical, List.mapIdx_eq_zipIdx_map]
    idealProgram := rfl
    idealParams := rfl
    idealInputsEq := rfl
    comparatorSpec := rfl
    comparatorWorkflowOutput := rfl
    comparatorIdealOutput := rfl
    comparatorOutcomeEq := rfl
  }

noncomputable def denoteProtocolBundleTraces
    (samplers : MxxSamplerFamily)
    (bundle : ClosedProtocolBundle)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment) :
    List (ClosedProtocolExecutionTrace samplers bundle params inputs) :=
  (denoteWorkflowTraces samplers bundle.workflow params inputs).map
    ClosedProtocolExecutionTrace.canonical

def denoteProtocolBundleOutcomes
    (samplers : MxxSamplerFamily)
    (bundle : ClosedProtocolBundle)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment) : List ClosedProtocolOutcome :=
  (Mxx.Ir.denoteWorkflow samplers bundle.workflow params inputs).map
    (denoteClosedProtocolOutcome bundle params inputs)

private theorem erase_canonicalClosedProtocolExecutionTrace
    {samplers : MxxSamplerFamily}
    {bundle : ClosedProtocolBundle}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (workflow : WorkflowExecutionTrace samplers bundle.workflow params inputs) :
    (ClosedProtocolExecutionTrace.canonical workflow).erase =
      denoteClosedProtocolOutcome bundle params inputs workflow.entrypointOutput := by
  simp [ClosedProtocolExecutionTrace.canonical, ClosedProtocolExecutionTrace.erase,
    denoteClosedProtocolOutcome, PureProgramExecution.canonical, List.mapIdx_eq_zipIdx_map]

/-- Erasing every bundle trace gives exactly the existing workflow, pure-program, and comparator
outcomes. -/
@[simp] theorem erase_denoteProtocolBundleTraces
    (samplers : MxxSamplerFamily)
    (bundle : ClosedProtocolBundle)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment) :
    (denoteProtocolBundleTraces samplers bundle params inputs).map
        ClosedProtocolExecutionTrace.erase =
      denoteProtocolBundleOutcomes samplers bundle params inputs := by
  simp [denoteProtocolBundleTraces, denoteProtocolBundleOutcomes,
    erase_canonicalClosedProtocolExecutionTrace, denoteWorkflowTraces,
    WorkflowExecutionTrace.canonical]

end Mxx.Certificate
