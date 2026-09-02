import MxxWe.DiamondWE.Decoder
import MxxWe.DiamondWE.Noise
import MxxIrCore.Eval
import MxxRuntime

namespace Mxx.We.DiamondWE

open Mxx.IR
open Mxx.Primitives

structure CandidateValidity
    (candidate : Candidate)
    (view : Candidate.HasDiamondGraphShape candidate) : Prop where
  parameters_match : CandidateParametersMatch candidate
  bound_expression_eq :
    candidate.bound.expression =
      deriveOutputNoiseBound candidate.parameters candidate.circuitShape.depth
  bound_environment_eq :
    ∀ parameter,
      candidate.bound.environment parameter =
        deriveBoundEnvironment candidate.parameters candidate.circuitShape.depth parameter
  decoder_safe : candidate.bound.value < decoderNoiseThreshold candidate.parameters.modulus

/- This is the only evaluator bridge currently available: it is the actual typed IR evaluator,
   with its environment and trace.  No output equation is assumed here. -/
structure DiamondEvalView (oracle : Mxx.Runtime.RuntimeGadgetOracle) (candidate : Candidate) where
  env : RuntimeEvalEnv oracle candidate.program.data
  trace : RuntimeTrace oracle
  evaluated : Mxx.IR.eval (RuntimeBackend oracle) candidate.program env = .ok trace

/- Diamond's externally visible outputs are top-level stage outputs.  Fixing the empty occurrence
   path here prevents a generated proof from selecting a convenient child/grid occurrence. -/
structure DiamondEvalObservation (oracle : Mxx.Runtime.RuntimeGadgetOracle) (candidate : Candidate)
    (view : DiamondEvalView oracle candidate) where
  decoded : Bool
  noisyPlaintext : RuntimeMatrixValue candidate.plaintextMatrixType
  decodedTrace :
    traceTypedValueAt view.trace #[] candidate.refs.decodedOutput = some decoded
  noisyPlaintextTrace :
    traceTypedValueAt view.trace #[] candidate.refs.noisyPlaintextOutput = some noisyPlaintext

/- This is the generated boundary for one reached primitive output.  It contains only coordinates,
   the stored payload/type witnesses, and the actual trace value.  Application algebra is proved by
   `input_injector_step_with_bound`, `bgg_layer_with_bound`, and `radix_step`; no final coefficient,
   distance, noise, or decode conclusion is accepted as a field. -/
structure ReachedPrimitiveOutput (oracle : Mxx.Runtime.RuntimeGadgetOracle) (candidate : Candidate)
    (view : DiamondEvalView oracle candidate) where
  occurrence : WireOccurrence
  payload : NodePayload
  outputType : WireType
  output : RuntimeDynamicValue oracle
  occurrenceValid : Mxx.IR.occurrenceValid candidate.program.data occurrence
  storedPayload : ∃ stage scope node,
    candidate.program.data.stages[occurrence.stage]? = some stage ∧
    scopeAt stage occurrence.wire.scope = some scope ∧
    nodeAt scope occurrence.wire.node = some node ∧ node.payload = payload
  storedOutput : ∃ stage scope node,
    candidate.program.data.stages[occurrence.stage]? = some stage ∧
    scopeAt stage occurrence.wire.scope = some scope ∧
    nodeAt scope occurrence.wire.node = some node ∧
    node.outputs[occurrence.wire.port]? = some outputType
  outputTypeStored : output.1 = outputType
  outputTrace : traceValueAt view.trace occurrence = some output

private def anyTypedReferenceOccurrence {program : Program}
    (reference : AnyTypedWireRef program) : WireOccurrence :=
  occurrenceOf reference.2.stage #[] reference.2.wire

/- The four output roles are fixed by `Candidate.Refs`; they are not selected by a caller-provided
   predicate.  Requiring one reached output for every named role is mechanically complete at the
   Diamond boundary and cannot be discharged by an empty or irrelevant generic-node list.  Each
   internal injector/BGG/radix derivation will instead reference its own reached node directly. -/
structure GeneratedPrimitivePlan (oracle : Mxx.Runtime.RuntimeGadgetOracle) (candidate : Candidate)
    (view : DiamondEvalView oracle candidate) where
  encryptionCircuitOutput : ReachedPrimitiveOutput oracle candidate view
  encryptionCircuitOccurrence :
    encryptionCircuitOutput.occurrence =
      anyTypedReferenceOccurrence candidate.refs.encryptionCircuitOutput
  encryptionCircuitType :
    encryptionCircuitOutput.outputType = candidate.refs.encryptionCircuitOutput.1
  decryptionCircuitOutput : ReachedPrimitiveOutput oracle candidate view
  decryptionCircuitOccurrence :
    decryptionCircuitOutput.occurrence =
      anyTypedReferenceOccurrence candidate.refs.decryptionCircuitOutput
  decryptionCircuitType :
    decryptionCircuitOutput.outputType = candidate.refs.decryptionCircuitOutput.1
  noisyPlaintextOutput : ReachedPrimitiveOutput oracle candidate view
  noisyPlaintextOccurrence :
    noisyPlaintextOutput.occurrence = occurrenceOf
      candidate.refs.noisyPlaintextOutput.stage #[] candidate.refs.noisyPlaintextOutput.wire
  noisyPlaintextType :
    noisyPlaintextOutput.outputType = .matrix candidate.plaintextMatrixType
  decodedOutput : ReachedPrimitiveOutput oracle candidate view
  decodedOccurrence :
    decodedOutput.occurrence =
      occurrenceOf candidate.refs.decodedOutput.stage #[] candidate.refs.decodedOutput.wire
  decodedType : decodedOutput.outputType = .bool
  goodSamples : GoodSamples oracle candidate view.env view.trace

/- The final claim consumes one successful good run and reads both observable values from that
   run's exact trace through the generated typed references.  `exactNoisyPlaintext` is not a
   separately chosen protocol value:
   the heterogeneous equality below identifies it with the runtime matrix stored at the generated
   output
   wire.  The `ApproxWithin` witness is part of the existential conclusion, so a caller cannot make
   the theorem vacuous by supplying an output equation or a final noise assumption. -/
def CorrectnessClaim (candidate : Candidate) : Prop :=
  ∀ (oracle : Mxx.Runtime.RuntimeGadgetOracle) (message : Bool)
    (env : RuntimeEvalEnv oracle candidate.program.data)
    (run : GoodRunPromise oracle candidate env),
    ValidExternalInputs oracle candidate env message →
    BooleanCircuitEvaluatesToOne oracle candidate env →
    ∃ (noisyPlaintext : RuntimeMatrixValue candidate.plaintextMatrixType)
      (decoded : Bool)
      (exactNoisyPlaintext : ExactMatrix candidate.parameters.modulus
        candidate.parameters.ringDimension 1 1),
      traceTypedValueAt run.trace #[] candidate.refs.noisyPlaintextOutput =
        some noisyPlaintext ∧
      traceTypedValueAt run.trace #[] candidate.refs.decodedOutput = some decoded ∧
      HEq noisyPlaintext exactNoisyPlaintext ∧
      Nonempty (ApproxWithin exactNoisyPlaintext
        (idealPlaintext candidate.parameters message) candidate.bound.value) ∧
        candidate.bound.value < decoderNoiseThreshold candidate.parameters.modulus ∧
          decoded = message

theorem CorrectnessClaim.evaluation_observation {candidate : Candidate}
    (claim : CorrectnessClaim candidate) (oracle : Mxx.Runtime.RuntimeGadgetOracle)
    (message : Bool)
    (env : RuntimeEvalEnv oracle candidate.program.data)
    (run : GoodRunPromise oracle candidate env)
    (inputs : ValidExternalInputs oracle candidate env message)
    (circuit : BooleanCircuitEvaluatesToOne oracle candidate env) :
    ∃ view : DiamondEvalView oracle candidate,
      Nonempty (DiamondEvalObservation oracle candidate view) ∧
        GoodSamples oracle candidate view.env view.trace := by
  obtain ⟨noisyPlaintext, decoded, _, noisyTrace, decodedTrace, _⟩ :=
    claim oracle message env run inputs circuit
  let view : DiamondEvalView oracle candidate := {
    env := env
    trace := run.trace
    evaluated := run.evaluated
  }
  refine ⟨view, ⟨{
    decoded := decoded
    noisyPlaintext := noisyPlaintext
    decodedTrace := decodedTrace
    noisyPlaintextTrace := noisyTrace
  }⟩, ?_⟩
  exact run.goodSamples

/- The approximation contained in the public claim is exactly the handoff consumed by Noise.lean.
   This is a consequence of a proved claim, not an alternative assumption from which correctness is
   assembled. -/
theorem CorrectnessClaim.output_noise_certificate {candidate : Candidate}
    (claim : CorrectnessClaim candidate) (oracle : Mxx.Runtime.RuntimeGadgetOracle)
    (message : Bool)
    (env : RuntimeEvalEnv oracle candidate.program.data)
    (run : GoodRunPromise oracle candidate env)
    (inputs : ValidExternalInputs oracle candidate env message)
    (circuit : BooleanCircuitEvaluatesToOne oracle candidate env) :
    ∃ certificate : OutputNoiseCertificate candidate candidate.bound.value,
      certificate.ideal = idealPlaintext candidate.parameters message := by
  obtain ⟨_, _, exactNoisyPlaintext, _, _, _, ⟨approximation⟩, _, _⟩ :=
    claim oracle message env run inputs circuit
  exact ⟨{
    noisyPlaintext := exactNoisyPlaintext
    ideal := idealPlaintext candidate.parameters message
    approximation := approximation
  }, rfl⟩

/- Independent liveness for every valid external environment is intentionally deferred.  A
   theorem named `correct` still needs generated reached-node equations that identify the noisy
   output with the composition of the injector, BGG, and radix lemmas, and a threshold-decode
   equation for the decoded output.  Keeping those obligations out of this file prevents an
   evaluator assumption or a caller-chosen final equation from entering the correctness boundary.
-/

end Mxx.We.DiamondWE
