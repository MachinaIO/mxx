import Mxx.Certificate.Analyzer
import Mxx.Certificate.Execution
import Mxx.Certificate.Preconditions
import Mxx.Certificate.Rules.DeterministicLeaves
import Mxx.Certificate.Rules.ScalarWorkflow
import Mxx.Certificate.Semantics

namespace Mxx.Certificate

def ScopedWireFactTable.Holds
    (environment : FactEnvironment)
    (facts : ScopedWireFactTable) : Prop :=
  ∀ fact ∈ facts, fact.Holds environment

/-- Structural invariant of analyzer-produced matrix facts. Symbolic expressions and relations
retain producer identities, while the fact subject is always the concrete wrapper wire consumed
by the current executable scope. -/
def ScopedWireFact.SubjectMatches (fact : ScopedWireFact) : Prop :=
  match fact.fact with
  | .matrix matrix => matrix.subject = .ofCoreWire fact.wire
  | _ => True

def ScopedWireFactTable.SubjectsMatch (facts : ScopedWireFactTable) : Prop :=
  ∀ fact ∈ facts, fact.SubjectMatches

theorem ScopedWireFact.matrixLookup_of_holds
    {environment : FactEnvironment}
    {fact : ScopedWireFact}
    {matrix : MatrixFact}
    (kind : fact.fact = .matrix matrix)
    (subjectMatches : fact.SubjectMatches)
    (holds : fact.Holds environment) :
    ∃ value, environment.values (.ofCoreWire fact.wire) = some (.matrix value) := by
  simp only [ScopedWireFact.SubjectMatches, kind] at subjectMatches
  simp only [ScopedWireFact.Holds, kind] at holds
  obtain ⟨value, _, lookup, _⟩ := holds
  change matrix.subject = .ofCoreWire fact.wire at subjectMatches
  refine ⟨value, ?_⟩
  rw [← subjectMatches]
  exact lookup

theorem transportFact_subjectMatches
    (wire : CoreWireRef)
    (source : ScopedWireFact) :
    (transportFact wire source).SubjectMatches := by
  rcases source with ⟨sourceWire, matrixType, fact⟩
  cases fact <;> simp [transportFact, ScopedWireFact.SubjectMatches]

theorem ScopedWireFactTable.Holds.nil (environment : FactEnvironment) :
    ScopedWireFactTable.Holds environment [] := by
  simp [ScopedWireFactTable.Holds]

theorem ScopedWireFactTable.Holds.append
    {environment : FactEnvironment}
    {left right : ScopedWireFactTable}
    (leftHolds : left.Holds environment)
    (rightHolds : right.Holds environment) :
    (left ++ right).Holds environment := by
  intro fact member
  rw [List.mem_append] at member
  exact member.elim (leftHolds fact) (rightHolds fact)

theorem ScopedWireFactTable.Holds.singleton
    {environment : FactEnvironment}
    {fact : ScopedWireFact}
    (factHolds : fact.Holds environment) :
    ScopedWireFactTable.Holds environment [fact] := by
  simpa [ScopedWireFactTable.Holds] using factHolds

theorem ScopedWireFactTable.Holds.append_singleton
    {environment : FactEnvironment}
    {facts : ScopedWireFactTable}
    {fact : ScopedWireFact}
    (factsHold : facts.Holds environment)
    (factHolds : fact.Holds environment) :
    (facts ++ [fact]).Holds environment :=
  factsHold.append (.singleton factHolds)

theorem AnalysisHolds.of_fact_table
    {environment : FactEnvironment}
    {analysis : AnalysisResult}
    (analysisOwned : environment.analysis = some analysis)
    (arenaOwned : environment.expressionArena = analysis.expressionArena)
    (arenaWellFormed : analysis.expressionArena.WF = true)
    (factsHold : analysis.facts.Holds environment) :
    AnalysisHolds environment analysis :=
  ⟨analysisOwned, arenaOwned, arenaWellFormed, factsHold⟩

/-- Internal wire state selected by an existing root-scope denotation. It is proof data extracted
from `evaluateNodes`, not an independently executed trace. -/
structure StageWireExecution
    {samplers : MxxSamplerFamily}
    (execution : StageExecution samplers) where
  wires : Mxx.Ir.WireEnvironment
  member : wires ∈ Mxx.Ir.evaluateNodes
    (Mxx.Ir.childRunnerWithFuel samplers execution.stage.program
      execution.stage.program.definitions.length)
    samplers execution.params execution.inputs execution.stage.program.root.nodes 0 [[]]
  path : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel samplers execution.stage.program
      execution.stage.program.definitions.length)
    samplers execution.params execution.inputs 0 execution.stage.program.root.nodes [] wires
  outputEq : Mxx.Ir.collectOutputs execution.stage.program.root.outputs wires = execution.output

theorem StageWireExecution.exists
    {samplers : MxxSamplerFamily}
    (execution : StageExecution samplers) :
    Nonempty (StageWireExecution execution) := by
  have outputMember := execution.outputMember
  unfold Mxx.Ir.denote at outputMember
  simp only [Mxx.Ir.denoteScopeWithFuel, List.mem_map] at outputMember
  obtain ⟨wires, member, outputEq⟩ := outputMember
  obtain ⟨initial, initialMember, path⟩ :=
    (Mxx.Ir.mem_evaluateNodes_iff_exists_path _ _ _ _ _ _ _ _).mp member
  simp only [List.mem_singleton] at initialMember
  subst initial
  exact ⟨⟨wires, member, path, outputEq⟩⟩

/-- One executable head step, lifted to the final SSA environment of the selected stage path.
The returned membership is the real `evaluateNode` support member and the lookup equality follows
from SSA freshness plus preservation through every later node. -/
theorem StageWireExecution.headOutputLookup
    {samplers : MxxSamplerFamily}
    {execution : StageExecution samplers}
    (witness : StageWireExecution execution)
    (node : Mxx.Ir.Node)
    (tail : List Mxx.Ir.Node)
    (nodes : execution.stage.program.root.nodes = node :: tail)
    (port : Nat)
    (fresh : Mxx.Ir.lookupWire ⟨0, port⟩ [] = none) :
    ∃ values,
      values ∈ Mxx.Ir.evaluateNode
        (Mxx.Ir.childRunnerWithFuel samplers execution.stage.program
          execution.stage.program.definitions.length)
        samplers execution.params execution.inputs [] node ∧
      ∀ portValid : port < values.length,
        Mxx.Ir.lookupWire ⟨0, port⟩ witness.wires = some values[port] := by
  have path := witness.path
  rw [nodes] at path
  exact path.outputAtHead port fresh

/-- Indexed executable step used by generated programs. The theorem exposes the exact prefix
state passed to the selected node, its real support member, and the selected output in the final
SSA environment. -/
theorem StageWireExecution.nodeOutputLookup
    {samplers : MxxSamplerFamily}
    {execution : StageExecution samplers}
    (witness : StageWireExecution execution)
    (index port : Nat)
    (inBounds : index < execution.stage.program.root.nodes.length)
    (fresh : ∀ before values,
      Mxx.Ir.EvaluatesNodesPath
        (Mxx.Ir.childRunnerWithFuel samplers execution.stage.program
          execution.stage.program.definitions.length)
        samplers execution.params execution.inputs 0
        (execution.stage.program.root.nodes.take index) [] before →
      values ∈ Mxx.Ir.evaluateNode
        (Mxx.Ir.childRunnerWithFuel samplers execution.stage.program
          execution.stage.program.definitions.length)
        samplers execution.params execution.inputs before
        execution.stage.program.root.nodes[index] →
      Mxx.Ir.lookupWire ⟨index, port⟩ before = none) :
    ∃ before values,
      values ∈ Mxx.Ir.evaluateNode
        (Mxx.Ir.childRunnerWithFuel samplers execution.stage.program
          execution.stage.program.definitions.length)
        samplers execution.params execution.inputs before
        execution.stage.program.root.nodes[index] ∧
      ∀ portValid : port < values.length,
        Mxx.Ir.lookupWire ⟨index, port⟩ witness.wires = some values[port] := by
  obtain ⟨before, values, prefixPath, member, suffixPath⟩ :=
    witness.path.atNodeIndex index inBounds
  refine ⟨before, values, member, fun portValid => ?_⟩
  apply suffixPath.lookupWire_preserved
  simpa using Mxx.Ir.lookupWire_append_bindOutputs
    (fresh before values prefixPath member) portValid

/-- Closed form of `nodeOutputLookup`. SSA freshness follows from the prefix path itself and is
not accepted from a certificate or theorem caller. -/
theorem StageWireExecution.nodeOutputLookupClosed
    {samplers : MxxSamplerFamily}
    {execution : StageExecution samplers}
    (witness : StageWireExecution execution)
    (index port : Nat)
    (inBounds : index < execution.stage.program.root.nodes.length) :
    ∃ before values,
      values ∈ Mxx.Ir.evaluateNode
        (Mxx.Ir.childRunnerWithFuel samplers execution.stage.program
          execution.stage.program.definitions.length)
        samplers execution.params execution.inputs before
        execution.stage.program.root.nodes[index] ∧
      ∀ portValid : port < values.length,
        Mxx.Ir.lookupWire ⟨index, port⟩ witness.wires = some values[port] := by
  apply witness.nodeOutputLookup index port inBounds
  intro before values prefixPath _
  apply prefixPath.lookupWire_after_end index port
  · simp [List.length_take, Nat.min_eq_left (Nat.le_of_lt inBounds)]
  · rfl

def StageWireExecution.factEnvironment
    {samplers : MxxSamplerFamily}
    {execution : StageExecution samplers}
    (witness : StageWireExecution execution) : FactEnvironment :=
  FactEnvironment.ofWireEnvironment execution.params ⟨execution.stage.id⟩ ⟨[]⟩ witness.wires

/-- Choose the internal SSA state already witnessed by a stage denotation. This is proof-only
choice over `StageWireExecution.exists`; it never re-executes the program. -/
noncomputable def StageExecution.wireExecution
    {samplers : MxxSamplerFamily}
    (execution : StageExecution samplers) : StageWireExecution execution :=
  Classical.choice (StageWireExecution.exists execution)

private def protocolInputValue
    (protocol : ClosedProtocolDecl)
    (inputs : Mxx.Ir.Environment)
    (id : ProtocolInputId) : Option Mxx.Ir.Value := do
  let entry ← protocol.bundle.inputContract.inputs.find? fun entry => entry.1 = id
  Mxx.Ir.lookupEnvironment entry.2.1 inputs

/-- A single identity environment for an entire selected workflow execution. Concrete wire
identities are read from the internal SSA state extracted from each existing stage denotation;
protocol-input identities are read from the closed input contract. This preserves producer wire
identity when an artifact fact is transported into a later stage. -/
noncomputable def WorkflowExecutionTrace.factEnvironment
    {samplers : MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (trace : WorkflowExecutionTrace samplers protocol.bundle.workflow parameters inputs) :
    FactEnvironment := {
  parameters
  values := fun reference =>
    match reference with
    | .protocolInput id => protocolInputValue protocol inputs id
    | .concrete wire => do
        let execution ← trace.stageExecutions.find? fun execution =>
          execution.stage.id = wire.stage.name
        if wire.scope.path.isEmpty then
          Mxx.Ir.lookupWire ⟨wire.node.value, wire.port⟩ execution.wireExecution.wires
        else none
    | _ => none
}

theorem WorkflowExecutionTrace.factEnvironment_protocolInput
    {samplers : MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (trace : WorkflowExecutionTrace samplers protocol.bundle.workflow parameters inputs)
    (id : ProtocolInputId)
    (entry : ProtocolInputId × String × InputValueContract)
    (found : protocol.bundle.inputContract.inputs.find? (fun candidate => candidate.1 = id) =
      some entry) :
    trace.factEnvironment.values (.protocolInput id) =
      Mxx.Ir.lookupEnvironment entry.2.1 inputs := by
  simp [WorkflowExecutionTrace.factEnvironment, protocolInputValue, found]

theorem WorkflowExecutionTrace.factEnvironment_concrete
    {samplers : MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (trace : WorkflowExecutionTrace samplers protocol.bundle.workflow parameters inputs)
    (wire : CoreWireRef)
    (execution : StageExecution samplers)
    (root : wire.scope.path = [])
    (found : trace.stageExecutions.find? (fun candidate =>
      candidate.stage.id = wire.stage.name) = some execution) :
    trace.factEnvironment.values (.concrete wire) =
      Mxx.Ir.lookupWire ⟨wire.node.value, wire.port⟩ execution.wireExecution.wires := by
  simp [WorkflowExecutionTrace.factEnvironment, found, root]

/-- Soundness of the Gaussian `applyRule` branch used by the canonical toy protocol. The output
lookup and hard-support bound are supplied by the selected SSA node execution through
`gaussianNode_local_sound`; this theorem only connects those executable facts to the analyzer's
derived `MatrixFact`. -/
theorem applyRule_sound
    {environment : FactEnvironment}
    {facts result : ScopedWireFactTable}
    {stage : StageId}
    {nodeId : Nat}
    {matrixType : MatrixTypeExpr}
    {cutoff : IntExpr}
    {outputTypes : List Mxx.Ir.WireTypeExpr}
    {output : Mxx.Matrix}
    {bound : Nat}
    {obligations : DerivedObligations}
    {endpoints : List EndpointFact}
    (accepted : applyRule facts {
      stage
      nodeId
      node := {
        kind := .gaussianSample matrixType cutoff
        arguments := []
        outputCount := 1
        outputTypes
      }
      rule := .introduceGaussian
    } = .ok (result, obligations, endpoints))
    (factsHold : facts.Holds environment)
    (outputLookup : environment.values (.concrete {
      stage
      scope := ⟨[]⟩
      node := ⟨nodeId⟩
      port := 0
    }) = some (.matrix output))
    (boundEvaluates : (BoundExpr.parameter cutoff).evaluate environment.parameters = .ok bound)
    (outputNorm : Mxx.maxCenteredCoefficientNorm output ≤ bound) :
    result.Holds environment := by
  have enabled : isInitialRuleEnabled .introduceGaussian = true := by decide
  simp [applyRule, enabled] at accepted
  obtain ⟨rfl, rfl, rfl⟩ := accepted
  apply factsHold.append_singleton
  exact boundedMatrixFact_holds environment _ output (.parameter cutoff) bound
    outputLookup boundEvaluates outputNorm

theorem applyRule_sound_zero
    {environment : FactEnvironment}
    {facts result : ScopedWireFactTable}
    {stage : StageId}
    {nodeId : Nat}
    {matrixType : MatrixTypeExpr}
    {outputTypes : List Mxx.Ir.WireTypeExpr}
    {parameters : Mxx.SamplerParams}
    {obligations : DerivedObligations}
    {endpoints : List EndpointFact}
    (accepted : applyRule facts {
      stage
      nodeId
      node := {
        kind := .zeroMatrix matrixType
        arguments := []
        outputCount := 1
        outputTypes
      }
      rule := .introduceExactConstant
    } = .ok (result, obligations, endpoints))
    (factsHold : facts.Holds environment)
    (typeEvaluates : matrixType.evaluate environment.parameters = some parameters)
    (outputLookup : environment.values (.concrete {
      stage
      scope := ⟨[]⟩
      node := ⟨nodeId⟩
      port := 0
    }) = some (.matrix (zeroConstantOutput parameters))) :
    result.Holds environment := by
  have enabled : isInitialRuleEnabled .introduceExactConstant = true := by decide
  simp [applyRule, enabled] at accepted
  obtain ⟨rfl, rfl, rfl⟩ := accepted
  apply factsHold.append_singleton
  exact exactMatrixFact_holds environment _ (.zero matrixType)
    (zeroConstantOutput parameters) (.constant 0) 0 outputLookup
    (.zero typeEvaluates) rfl (zeroConstant_norm_eq_zero parameters).le

theorem applyRule_sound_identity
    {environment : FactEnvironment}
    {facts result : ScopedWireFactTable}
    {stage : StageId}
    {nodeId : Nat}
    {matrixType : MatrixTypeExpr}
    {outputTypes : List Mxx.Ir.WireTypeExpr}
    {output : Mxx.Matrix}
    {obligations : DerivedObligations}
    {endpoints : List EndpointFact}
    (accepted : applyRule facts {
      stage
      nodeId
      node := {
        kind := .identityMatrix matrixType
        arguments := []
        outputCount := 1
        outputTypes
      }
      rule := .introduceExactConstant
    } = .ok (result, obligations, endpoints))
    (factsHold : facts.Holds environment)
    (outputLookup : environment.values (.concrete {
      stage
      scope := ⟨[]⟩
      node := ⟨nodeId⟩
      port := 0
    }) = some (.matrix output))
    (outputNorm : Mxx.maxCenteredCoefficientNorm output ≤ 1) :
    result.Holds environment := by
  have enabled : isInitialRuleEnabled .introduceExactConstant = true := by decide
  simp [applyRule, enabled] at accepted
  obtain ⟨rfl, rfl, rfl⟩ := accepted
  apply factsHold.append_singleton
  exact exactMatrixFact_holds environment _ (.wire {
      value := .concrete {
        stage
        scope := ⟨[]⟩
        node := ⟨nodeId⟩
        port := 0
      }
      type := matrixType
    }) output (.constant 1) 1 outputLookup (.wire outputLookup) rfl outputNorm

theorem applyRule_sound_constantMatrix
    {environment : FactEnvironment}
    {facts result : ScopedWireFactTable}
    {stage : StageId}
    {nodeId : Nat}
    {matrixType : MatrixTypeExpr}
    {coefficients : List IntExpr}
    {outputTypes : List Mxx.Ir.WireTypeExpr}
    {output : Mxx.Matrix}
    {bound : Nat}
    {obligations : DerivedObligations}
    {endpoints : List EndpointFact}
    (accepted : applyRule facts {
      stage
      nodeId
      node := {
        kind := .constantMatrix matrixType coefficients
        arguments := []
        outputCount := 1
        outputTypes
      }
      rule := .introduceExactConstant
    } = .ok (result, obligations, endpoints))
    (factsHold : facts.Holds environment)
    (outputLookup : environment.values (.concrete {
      stage
      scope := ⟨[]⟩
      node := ⟨nodeId⟩
      port := 0
    }) = some (.matrix output))
    (boundEvaluates :
      (BoundExpr.floorDivide (.absolute matrixType.modulus) 2).evaluate
        environment.parameters = .ok bound)
    (outputNorm : Mxx.maxCenteredCoefficientNorm output ≤ bound) :
    result.Holds environment := by
  have enabled : isInitialRuleEnabled .introduceExactConstant = true := by decide
  simp [applyRule, enabled] at accepted
  obtain ⟨rfl, rfl, rfl⟩ := accepted
  apply factsHold.append_singleton
  exact exactMatrixFact_holds environment _ (.wire {
      value := .concrete {
        stage
        scope := ⟨[]⟩
        node := ⟨nodeId⟩
        port := 0
      }
      type := matrixType
    }) output _ bound outputLookup (.wire outputLookup) boundEvaluates outputNorm

theorem applyRule_sound_gadgetMatrix
    {environment : FactEnvironment}
    {facts result : ScopedWireFactTable}
    {stage : StageId}
    {nodeId : Nat}
    {matrixType : MatrixTypeExpr}
    {base : IntExpr}
    {outputTypes : List Mxx.Ir.WireTypeExpr}
    {parameters : Mxx.SamplerParams}
    {evaluatedBase : Int}
    {bound : Nat}
    {obligations : DerivedObligations}
    {endpoints : List EndpointFact}
    (accepted : applyRule facts {
      stage
      nodeId
      node := {
        kind := .gadgetMatrix matrixType base
        arguments := []
        outputCount := 1
        outputTypes
      }
      rule := .introduceExactConstant
    } = .ok (result, obligations, endpoints))
    (factsHold : facts.Holds environment)
    (typeEvaluates : matrixType.evaluate environment.parameters = some parameters)
    (baseEvaluates : base.evaluate environment.parameters = some evaluatedBase)
    (outputLookup : environment.values (.concrete {
      stage
      scope := ⟨[]⟩
      node := ⟨nodeId⟩
      port := 0
    }) = some (.matrix (Mxx.gadgetMatrix parameters evaluatedBase
      (if parameters.rows = 0 then 0 else parameters.columns / parameters.rows))))
    (boundEvaluates :
      (BoundExpr.floorDivide (.absolute matrixType.modulus) 2).evaluate
        environment.parameters = .ok bound)
    (outputNorm : Mxx.maxCenteredCoefficientNorm
      (Mxx.gadgetMatrix parameters evaluatedBase
        (if parameters.rows = 0 then 0 else parameters.columns / parameters.rows)) ≤ bound) :
    result.Holds environment := by
  have enabled : isInitialRuleEnabled .introduceExactConstant = true := by decide
  simp [applyRule, enabled] at accepted
  obtain ⟨rfl, rfl, rfl⟩ := accepted
  apply factsHold.append_singleton
  exact exactMatrixFact_holds environment _ (.gadget matrixType base) _ _ bound outputLookup
    (.gadget typeEvaluates baseEvaluates) boundEvaluates outputNorm

/-- Soundness lift for the plain deterministic hash constructor. The analyzer does not trust a
declared hash bound: the selected executable output is bounded by the centered modulus radius. -/
theorem applyRule_sound_plainHash
    {environment : FactEnvironment}
    {facts result : ScopedWireFactTable}
    {stage : StageId}
    {nodeId : Nat}
    {matrixType : MatrixTypeExpr}
    {tagPrefix : List Nat}
    {tagExpressions tagDecimalExpressions tagU64LeExpressions : List IntExpr}
    {outputTypes : List Mxx.Ir.WireTypeExpr}
    {output : Mxx.Matrix}
    {bound : Nat}
    {obligations : DerivedObligations}
    {endpoints : List EndpointFact}
    (accepted : applyRule facts {
      stage
      nodeId
      node := {
        kind := .hashSample matrixType .plain tagPrefix tagExpressions tagDecimalExpressions
          tagU64LeExpressions none none
        arguments := []
        outputCount := 1
        outputTypes
      }
      rule := .introduceHash
    } = .ok (result, obligations, endpoints))
    (factsHold : facts.Holds environment)
    (outputLookup : environment.values (.concrete {
      stage
      scope := ⟨[]⟩
      node := ⟨nodeId⟩
      port := 0
    }) = some (.matrix output))
    (boundEvaluates :
      (BoundExpr.floorDivide (.absolute matrixType.modulus) 2).evaluate
        environment.parameters = .ok bound)
    (outputNorm : Mxx.maxCenteredCoefficientNorm output ≤ bound) :
    result.Holds environment := by
  have enabled : isInitialRuleEnabled .introduceHash = true := by decide
  simp [applyRule, enabled] at accepted
  obtain ⟨rfl, rfl, rfl⟩ := accepted
  apply factsHold.append_singleton
  exact exactMatrixFact_holds environment _ (.wire {
      value := .concrete {
        stage
        scope := ⟨[]⟩
        node := ⟨nodeId⟩
        port := 0
      }
      type := matrixType
    }) output _ bound outputLookup (.wire outputLookup) boundEvaluates outputNorm

theorem booleanProtocolInputFact_holds
    {environment : FactEnvironment}
    {wire : CoreWireRef}
    {input : ProtocolInputId}
    {value : Bool}
    (wireLookup : environment.values (.ofCoreWire wire) = some (.boolean value))
    (inputLookup : environment.values (.protocolInput input) = some (.boolean value)) :
    ScopedWireFact.Holds environment {
      wire
      matrixType := none
      fact := .boolean { expression := .boolWire (.protocolInput input) }
    } := by
  exact ⟨value, wireLookup, .boolWire inputLookup⟩

theorem boolToIntFact_holds
    {environment : FactEnvironment}
    {wire : CoreWireRef}
    {input : RuntimeExpr .boolean}
    {value : Bool}
    (inputDenotes : RuntimeBoolExpr.Denotes environment input value)
    (wireLookup : environment.values (.ofCoreWire wire) =
      some (.integer (if value then 1 else 0))) :
    ScopedWireFact.Holds environment {
      wire
      matrixType := none
      fact := .integer {
        expression := .boolToInt input
        lower := .integer (.constant 0)
        upper := .integer (.constant 1)
      }
    } := by
  refine ⟨_, 0, 1, wireLookup, .boolToInt inputDenotes, rfl, rfl, ?_, ?_⟩
  · cases value <;> decide
  · cases value <;> decide

theorem selectExactFact_holds
    {environment : FactEnvironment}
    {wire : CoreWireRef}
    {matrixType : MatrixTypeExpr}
    {index : RuntimeExpr .integer}
    {branches : List MatrixExpr}
    {indexValue : Int}
    {branch : MatrixExpr}
    {value : Mxx.Matrix}
    {boundExpression : BoundExpr}
    {bound : Nat}
    (indexDenotes : RuntimeIntExpr.Denotes environment index indexValue)
    (nonnegative : 0 ≤ indexValue)
    (selected : branches[indexValue.toNat]? = some branch)
    (branchDenotes : MatrixExpr.Denotes environment branch value)
    (wireLookup : environment.values (.concrete wire) = some (.matrix value))
    (boundEvaluates : boundExpression.evaluate environment.parameters = .ok bound)
    (normBound : Mxx.maxCenteredCoefficientNorm value ≤ bound) :
    ScopedWireFact.Holds environment {
      wire
      matrixType := some matrixType
      fact := .matrix {
        subject := .concrete wire
        primary := .exact (.select index branches)
        relations := []
        totalNormBound := boundExpression
      }
    } := by
  exact exactMatrixFact_holds environment _ _ value boundExpression bound wireLookup
    (.select indexDenotes nonnegative selected branchDenotes) boundEvaluates normBound

theorem thresholdDecodeBoolFact_holds
    {environment : FactEnvironment}
    {wire : CoreWireRef}
    {matrix : ValueInstanceRef}
    {matrixValue : Mxx.Matrix}
    {ciphertextModulus plaintextModulus position : IntExpr}
    {q p index coefficient : Int}
    (matrixLookup : environment.values matrix = some (.matrix matrixValue))
    (qEvaluates : evaluateIntExpr environment.parameters ciphertextModulus = .ok q)
    (pEvaluates : evaluateIntExpr environment.parameters plaintextModulus = .ok p)
    (positionEvaluates : evaluateIntExpr environment.parameters position = .ok index)
    (nonnegative : 0 ≤ index)
    (coefficientLookup : matrixValue.coefficients[index.toNat]? = some coefficient)
    (wireLookup : environment.values (.ofCoreWire wire) =
      some (.boolean (Mxx.Ir.thresholdDecodeBool q p coefficient))) :
    ScopedWireFact.Holds environment {
      wire
      matrixType := none
      fact := .boolean {
        expression := .thresholdDecodeBool matrix ciphertextModulus plaintextModulus position
      }
    } := by
  exact ⟨_, wireLookup, .thresholdDecodeBool matrixLookup qEvaluates pEvaluates
    positionEvaluates nonnegative coefficientLookup⟩

/-- Semantic leaf used by the toy `matrixAdd`: an exact selected carrier plus a bounded Gaussian
becomes the one-term affine form emitted by `addOrSubtractFact`. -/
theorem addExactBoundedFact_holds
    {environment : FactEnvironment}
    {wire : CoreWireRef}
    {matrixType : MatrixTypeExpr}
    {identityType : MatrixTypeExpr}
    {signalExpression : MatrixExpr}
    {signal noise : Mxx.Matrix}
    {noiseBound totalBound : BoundExpr}
    {noiseBoundValue totalBoundValue : Nat}
    (wireLookup : environment.values (.concrete wire) =
      some (.matrix (Mxx.matrixAdd signal noise)))
    (signalDenotes : MatrixExpr.Denotes environment signalExpression signal)
    (noiseBoundEvaluates : noiseBound.evaluate environment.parameters = .ok noiseBoundValue)
    (noiseNorm : Mxx.maxCenteredCoefficientNorm noise ≤ noiseBoundValue)
    (totalBoundEvaluates : totalBound.evaluate environment.parameters = .ok totalBoundValue)
    (totalNorm : Mxx.maxCenteredCoefficientNorm (Mxx.matrixAdd signal noise) ≤ totalBoundValue) :
    ScopedWireFact.Holds environment {
      wire
      matrixType := some matrixType
      fact := .matrix {
        subject := .concrete wire
        primary := .affine {
          terms := [{
            coefficient := {
              expression := .identity identityType
              normBound := .constant 1
            }
            basis := signalExpression
            mode := .ordinaryMatrixProduct
          }]
          noiseBound
        }
        relations := []
        totalNormBound := totalBound
      }
    } :=
  identitySignalNoiseMatrixFact_holds environment _ identityType signalExpression signal noise
    noiseBound totalBound noiseBoundValue totalBoundValue wireLookup signalDenotes
    noiseBoundEvaluates noiseNorm totalBoundEvaluates totalNorm

/-- Base case of the executable-path induction used by program soundness. An empty root scope
cannot manufacture a fact: successful analysis returns exactly the already-sound initial table. -/
theorem analyzeProgram_sound
    {environment : FactEnvironment}
    {stage : StageId}
    {program : Mxx.Ir.Prog}
    {initial result : ScopedWireFactTable}
    (definitionsEmpty : program.definitions = [])
    (nodesEmpty : program.root.nodes = [])
    (accepted : analyzeProgram stage program initial = .ok result)
    (initialHolds : initial.Holds environment) :
    result.Holds environment := by
  simp [analyzeProgram, definitionsEmpty, nodesEmpty, inferRules, inferRulesFrom] at accepted
  change Except.ok initial = Except.ok result at accepted
  injection accepted with equality
  rw [← equality]
  exact initialHolds

/-- First nonempty induction case. An executable input node only binds the supplied runtime value;
the analyzer intentionally preserves the already-verified initial protocol-input fact table. -/
theorem analyzeProgram_sound_input
    {samplers : MxxSamplerFamily}
    {environment : FactEnvironment}
    {execution : StageExecution samplers}
    {initial result : ScopedWireFactTable}
    {name : String}
    {outputTypes : List Mxx.Ir.WireTypeExpr}
    (witness : StageWireExecution execution)
    (definitionsEmpty : execution.stage.program.definitions = [])
    (nodes : execution.stage.program.root.nodes = [{
      kind := .input name
      arguments := []
      outputCount := 1
      outputTypes
    }])
    (accepted : analyzeProgram ⟨execution.stage.id⟩ execution.stage.program initial =
      .ok result)
    (initialHolds : initial.Holds environment) :
    result.Holds environment := by
  obtain ⟨values, _, _⟩ := witness.headOutputLookup _ [] nodes 0 (by rfl)
  simp [analyzeProgram, definitionsEmpty, nodes, inferRules, inferRulesFrom,
    inferNodeFacts] at accepted
  change Except.ok initial = Except.ok result at accepted
  injection accepted with equality
  rw [← equality]
  exact initialHolds

/-- General input-head induction step. The input branch emits no new analyzer fact; the suffix is
analyzed from absolute node identifier one and discharged by the induction hypothesis. -/
theorem analyzeProgram_sound_input_cons
    {environment : FactEnvironment}
    {stage : StageId}
    {program : Mxx.Ir.Prog}
    {initial result : ScopedWireFactTable}
    {name : String}
    {outputTypes : List Mxx.Ir.WireTypeExpr}
    {tail : List Mxx.Ir.Node}
    (definitionsEmpty : program.definitions = [])
    (nodes : program.root.nodes = {
      kind := .input name
      arguments := []
      outputCount := 1
      outputTypes
    } :: tail)
    (accepted : analyzeProgram stage program initial = .ok result)
    (suffixSound : ∀ suffixResult,
      inferRulesFrom stage ⟨[]⟩ 1 tail initial = .ok suffixResult →
      suffixResult.Holds environment) :
    result.Holds environment := by
  simp [analyzeProgram, definitionsEmpty, nodes, inferRules, inferRulesFrom,
    inferNodeFacts] at accepted
  exact suffixSound result accepted

/-- Common executable/analyzer induction step. Per-kind theorems prove `headSound` from the
selected node execution; the suffix hypothesis composes it at the next absolute SSA identifier. -/
private theorem analyzeProgram_sound_cons
    {environment : FactEnvironment}
    {stage : StageId}
    {scope : StaticScopeId}
    {nodeId : Nat}
    {node : Mxx.Ir.Node}
    {tail : List Mxx.Ir.Node}
    {initial result : ScopedWireFactTable}
    (accepted : inferRulesFrom stage scope nodeId (node :: tail) initial = .ok result)
    (headSound : ∀ next,
      inferNodeFacts stage scope nodeId node initial = .ok next →
      next.Holds environment)
    (suffixSound : ∀ next suffixResult,
      next.Holds environment →
      inferRulesFrom stage scope (nodeId + 1) tail next = .ok suffixResult →
      suffixResult.Holds environment) :
    result.Holds environment := by
  simp only [inferRulesFrom] at accepted
  cases headResult : inferNodeFacts stage scope nodeId node initial with
  | error error =>
      rw [headResult] at accepted
      contradiction
  | ok next =>
      rw [headResult] at accepted
      exact suffixSound next result (headSound next headResult) accepted

end Mxx.Certificate
