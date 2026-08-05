import Mxx.Certificate.Checker
import Mxx.Certificate.Execution
import Mxx.Certificate.Preconditions

namespace Mxx.Certificate

/-! # Closed-bundle soundness composition

This module contains the protocol-independent final composition step.  It does not inspect or
re-execute a protocol graph.  A registered endpoint rule establishes `EndpointAgreement` from
the analyzer's facts and the selected execution trace; this module then connects that result to
the existing ideal program and comparator denotations.
-/

/-- End-to-end correctness of a closed protocol declaration.  The proposition quantifies over
the existing bundle denotation, so it introduces neither a second evaluator nor a certificate
chosen execution. -/
def ClosedProtocolDecl.Correct
    (protocol : ClosedProtocolDecl)
    (samplers : MxxSamplerFamily)
    (parameters : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment) : Prop :=
  ∀ outcome ∈ denoteProtocolBundleOutcomes samplers protocol.bundle parameters inputs,
    outcome.comparator.failure = false

/-- Semantic result of a registered Boolean endpoint rule for one selected bundle trace.

The equality comparator, endpoint registration, ideal termination, and both output lookups are
all explicit.  Consequently the bundle composition theorem below needs no protocol-specific
execution expansion.  This structure is proof output of the closed endpoint-soundness layer; it
is not certificate data and is not accepted by either verifier phase. -/
structure EndpointAgreement
    {samplers : MxxSamplerFamily}
    {bundle : ClosedProtocolBundle}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (trace : ClosedProtocolExecutionTrace samplers bundle parameters inputs) where
  binding : ComparatorEndpointBinding
  endpoint : EndpointAnchor
  value : Bool
  idealOutput : Mxx.Ir.Environment
  comparator : bundle.comparator = .equality [binding]
  endpointFound : bundle.endpoints.entries.find? (fun candidate =>
    candidate.specification = binding.endpoint) = some endpoint
  idealOutputEq : trace.ideal.output = some idealOutput
  actualLookup : Mxx.Ir.lookupEnvironment endpoint.workflowOutput
    trace.workflow.entrypointOutput = some (.boolean value)
  idealLookup : Mxx.Ir.lookupEnvironment endpoint.idealOutput idealOutput =
    some (.boolean value)

/-- A registered endpoint agreement forces the comparator of the same execution trace to
succeed.  The proof delegates comparator semantics to the public generic equality theorem. -/
theorem EndpointAgreement.comparator_succeeds
    {samplers : MxxSamplerFamily}
    {bundle : ClosedProtocolBundle}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {trace : ClosedProtocolExecutionTrace samplers bundle parameters inputs}
    (agreement : EndpointAgreement trace) :
    trace.comparator.outcome.failure = false := by
  rw [trace.comparatorOutcomeEq, agreement.idealOutputEq]
  exact denoteComparator_singleEquality_success bundle parameters
    trace.workflow.entrypointOutput agreement.idealOutput agreement.binding agreement.endpoint
    agreement.value agreement.comparator agreement.endpointFound agreement.actualLookup
    agreement.idealLookup

/-- Contract implemented by the closed analyzer/endpoint soundness induction.  All premises are
derived from runtime inputs, verifier acceptance, or the sampler contract; no field is supplied
by a sparse certificate. -/
def ClosedEndpointSoundness
    (protocol : ClosedProtocolDecl)
    (analysis : AnalysisResult)
    (checked : CheckedStaticObligations) : Prop :=
  ∀ (samplers : MxxSamplerFamily),
    MxxBoundedSamplerContract samplers →
    ∀ (parameters : Mxx.Ir.ParamEnvironment)
      (inputs : Mxx.Ir.Environment),
    protocol.ParamsWF parameters →
    protocol.InputsWF parameters inputs →
    protocol.ProtocolPreconditions parameters inputs →
    analyzeProtocol protocol { overrides := [] } = .ok analysis →
    checkStaticParameters analysis parameters = .ok checked →
    ∀ trace ∈ denoteProtocolBundleTraces samplers protocol.bundle parameters inputs,
      Nonempty (EndpointAgreement trace)

/-- Generic Phase-A/Phase-B-to-bundle composition for the single Boolean equality endpoint used
by the initial closed protocols.

`endpointSound` is the output theorem of the registered endpoint-soundness layer.  It receives
the exact Phase A and Phase B acceptance equalities and must derive agreement for the actual
trace.  Thus this theorem cannot be used with a self-reported symbolic fact, a caller-selected
execution, or a manually unfolded protocol.  Once the closed analyzer soundness induction
constructs `endpointSound`, application to every protocol graph is mechanical. -/
theorem ClosedProtocolDecl.correct_of_analyzer_checker_endpoint
    {protocol : ClosedProtocolDecl}
    {samplers : MxxSamplerFamily}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {checked : CheckedStaticObligations}
    (contract : MxxBoundedSamplerContract samplers)
    (paramsWF : protocol.ParamsWF parameters)
    (inputsWF : protocol.InputsWF parameters inputs)
    (preconditions : protocol.ProtocolPreconditions parameters inputs)
    (phaseA : analyzeProtocol protocol { overrides := [] } = .ok analysis)
    (phaseB : checkStaticParameters analysis parameters = .ok checked)
    (endpointSound : ClosedEndpointSoundness protocol analysis checked) :
    protocol.Correct samplers parameters inputs := by
  intro outcome outcomeMember
  have tracedMember : outcome ∈
      (denoteProtocolBundleTraces samplers protocol.bundle parameters inputs).map
        ClosedProtocolExecutionTrace.erase := by
    rw [erase_denoteProtocolBundleTraces]
    exact outcomeMember
  obtain ⟨trace, traceMember, rfl⟩ := List.mem_map.mp tracedMember
  obtain ⟨agreement⟩ := endpointSound samplers contract parameters inputs paramsWF inputsWF
    preconditions phaseA phaseB trace traceMember
  exact agreement.comparator_succeeds

end Mxx.Certificate
