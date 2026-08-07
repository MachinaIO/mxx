import Mxx.Certificate.Preconditions
import Mxx.Certificate.Rules.RequirementAcceptance

namespace Mxx.Certificate

/-!
# Requirement execution selected from one closed protocol trace

The lemmas in this file eliminate the existential in `ProtocolPreconditions` against the exact
`PureProgramExecution` already stored in the same `ClosedProtocolExecutionTrace`.  Determinism of
`denotePure` makes the two outputs equal; no second runner or caller-provided requirement trace is
accepted.
-/

/-- The accepted output of one actual requirement execution in a closed protocol trace. -/
structure ClosedRequirementAcceptedExecution
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs)
    (index : Nat)
    (program : Mxx.Ir.Prog)
    (outputName : String) where
  execution : PureProgramExecution
  executionAt : trace.requirements[index]? = some execution
  programMatches : execution.program = program
  parametersMatch : execution.params = parameters
  inputsMatch : execution.inputs =
    requirementInputEnvironment protocol.bundle inputs index program
  output : Mxx.Ir.Environment
  outputSome : execution.output = some output
  acceptedLookup : Mxx.Ir.lookupEnvironment outputName output = some (.boolean true)

/-- Select the exact requirement execution certified by the closed precondition. The proof uses
the trace's program/parameter/input map equalities and the existing deterministic `denotePure`;
it does not re-run the program. -/
theorem ClosedRequirementAcceptedExecution.exists
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs)
    (preconditions : protocol.ProtocolPreconditions parameters inputs)
    (index : Nat)
    (program : Mxx.Ir.Prog)
    (outputName : String)
    (programAt : protocol.bundle.requirements[index]? = some program)
    (outputNameAt :
      protocol.bundle.preconditionSpec.requirementOutputs[index]? = some outputName) :
    Nonempty (ClosedRequirementAcceptedExecution trace index program outputName) := by
  obtain ⟨output, denoteEq, acceptedLookup⟩ :=
    preconditions index program outputName programAt outputNameAt
  have mappedProgramAt : (trace.requirements.map (·.program))[index]? = some program := by
    rw [trace.requirementPrograms]
    exact programAt
  rw [List.getElem?_map] at mappedProgramAt
  cases executionAt : trace.requirements[index]? with
  | none => simp [executionAt] at mappedProgramAt
  | some execution =>
      have programMatches : execution.program = program := by
        simp [executionAt] at mappedProgramAt
        exact mappedProgramAt
      have mappedParamsAt : (trace.requirements.map (·.params))[index]? = some parameters := by
        rw [trace.requirementParams, List.getElem?_map, programAt]
        rfl
      rw [List.getElem?_map, executionAt] at mappedParamsAt
      have parametersMatch : execution.params = parameters := by simpa using mappedParamsAt
      have mappedInputsAt : (trace.requirements.map (·.inputs))[index]? =
          some (requirementInputEnvironment protocol.bundle inputs index program) := by
        rw [trace.requirementInputsEq]
        simp [List.mapIdx_eq_zipIdx_map, programAt]
      rw [List.getElem?_map, executionAt] at mappedInputsAt
      have inputsMatch : execution.inputs =
          requirementInputEnvironment protocol.bundle inputs index program := by
        simpa using mappedInputsAt
      have outputSome : execution.output = some output := by
        rw [execution.outputEq, programMatches, parametersMatch, inputsMatch]
        exact denoteEq
      exact ⟨{
        execution
        executionAt
        programMatches
        parametersMatch
        inputsMatch
        output
        outputSome
        acceptedLookup
      }⟩

/-- The accepted execution exposes the exact root SSA path used by the closed requirement. -/
theorem ClosedRequirementAcceptedExecution.rootPath
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {index : Nat}
    {program : Mxx.Ir.Prog}
    {outputName : String}
    (execution : ClosedRequirementAcceptedExecution trace index program outputName) :
    Nonempty (PureProgramRootExecutionPath execution.execution) :=
  PureProgramRootExecutionPath.exists execution.execution execution.output execution.outputSome

end Mxx.Certificate
