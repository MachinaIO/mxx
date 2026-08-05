import MxxCorrectness.Generated.ToyExample.Ir
import Mathlib.Tactic

namespace MxxCorrectness.Proofs.ToyExample

open Mxx
open Mxx.Certificate
open MxxCorrectness.Generated.ToyExample

def parameters (cutoff : Nat) : Mxx.Ir.ParamEnvironment :=
  [("cutoff", .integer cutoff)]

def inputs (message : Bool) : Mxx.Ir.Environment :=
  [("message", .boolean message)]

def samplerParams (cutoff : Nat) : Mxx.SamplerParams := {
  maxCoefficientBound := cutoff
  modulus := 256
  ringDimension := 1
  rows := 1
  columns := 1
}

private def ciphertext (cutoff : Nat) (message : Bool) (sample : Mxx.Matrix) : Mxx.Matrix := {
  coefficients := List.map (Mxx.reduceCoefficient 256)
    (Mxx.addCoefficients (if message then [128] else [0])
      (sample.withSamplerParams (samplerParams cutoff)).coefficients)
  modulus := 256
  ringDimension := 1
  rows := 1
  columns := 1
}

private theorem encryptOutcome
    (samplers : MxxSamplerFamily)
    (cutoff : Nat)
    (message : Bool)
    (output : Mxx.Ir.Environment)
    (member : output ∈ Mxx.Ir.denote samplers ToyExample_stage_encrypt (parameters cutoff)
      (Mxx.Ir.stageInputs (inputs message) []
        { id := "encrypt", program := ToyExample_stage_encrypt,
          inputs := [("message", .protocol "message")] })) :
    ∃ sample ∈ samplers.gaussianSample (samplerParams cutoff),
      [("ciphertext", .matrix (ciphertext cutoff message sample))] = output := by
  have cutoffNonnegative : ¬ ((cutoff : Int) < 0) := by omega
  cases message <;>
    simp [ToyExample_stage_encrypt, parameters, inputs, Mxx.Ir.stageInputs,
      Mxx.Ir.resolveStageInput, Mxx.Ir.lookupEnvironment, Mxx.Ir.denote,
      Mxx.Ir.denoteScopeWithFuel, Mxx.Ir.evaluateNodes, Mxx.Ir.evaluateNode,
      Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs, Mxx.Ir.collectOutputs,
      Mxx.Ir.IntExpr.evaluate, Mxx.Ir.MatrixTypeExpr.evaluate, Mxx.Matrix.withSamplerParams,
      samplerParams, Mxx.Ir.lookupParam, Mxx.matrixAdd, Mxx.reduceCoefficient,
      ciphertext, cutoffNonnegative] at member ⊢
  all_goals exact member

private theorem decryptOutcome
    (samplers : MxxSamplerFamily)
    (cutoff : Nat)
    (message : Bool)
    (sample : Mxx.Matrix)
    (output : Mxx.Ir.Environment)
    (member : output ∈ Mxx.Ir.denote samplers ToyExample_stage_decrypt (parameters cutoff)
      (Mxx.Ir.stageInputs (inputs message)
        [("encrypt", [("ciphertext", .matrix (ciphertext cutoff message sample))])]
        { id := "decrypt", program := ToyExample_stage_decrypt,
          inputs := [("ciphertext", .artifact "encrypt" "ciphertext")] })) :
    [("decoded", .boolean (Mxx.Ir.thresholdDecodeBool 256 2
      ((ciphertext cutoff message sample).coefficients.headD 0)))] = output := by
  simp [ToyExample_stage_decrypt, parameters, inputs, Mxx.Ir.stageInputs,
    Mxx.Ir.resolveStageInput, Mxx.Ir.lookupStage, Mxx.Ir.lookupEnvironment, Mxx.Ir.denote,
    Mxx.Ir.denoteScopeWithFuel, Mxx.Ir.evaluateNodes, Mxx.Ir.evaluateNode,
    Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs, Mxx.Ir.collectOutputs,
    Mxx.Ir.IntExpr.evaluate] at member
  cases message <;> cases sample with
  | mk coefficients modulus ringDimension rows columns =>
      cases coefficients <;>
        simpa [ciphertext, Mxx.Matrix.withSamplerParams, samplerParams,
          Mxx.addCoefficients, Mxx.Ir.lookupWire] using member.symm

private theorem stageOutputs
    (samplers : MxxSamplerFamily)
    (cutoff : Nat)
    (message : Bool)
    {final : Mxx.Ir.StageEnvironment}
    {executions : List (StageExecution samplers)}
    (execution : StageExecutions samplers (parameters cutoff) (inputs message)
      ToyExample_protocol.bundle.workflow.stages [] final executions) :
    ∃ encryptOutput decryptOutput,
      encryptOutput ∈ Mxx.Ir.denote samplers ToyExample_stage_encrypt (parameters cutoff)
        (Mxx.Ir.stageInputs (inputs message) []
          { id := "encrypt", program := ToyExample_stage_encrypt,
            inputs := [("message", .protocol "message")] }) ∧
      decryptOutput ∈ Mxx.Ir.denote samplers ToyExample_stage_decrypt (parameters cutoff)
        (Mxx.Ir.stageInputs (inputs message) [("encrypt", encryptOutput)]
          { id := "decrypt", program := ToyExample_stage_decrypt,
            inputs := [("ciphertext", .artifact "encrypt" "ciphertext")] }) ∧
      final = [("encrypt", encryptOutput), ("decrypt", decryptOutput)] := by
  cases execution with
  | cons _ _ _ _ encryptOutput _ encryptMember tailExecution =>
      cases tailExecution with
      | cons _ _ _ _ decryptOutput _ decryptMember tailExecution =>
          cases tailExecution
          exact ⟨encryptOutput, decryptOutput, encryptMember, decryptMember, rfl⟩

private theorem traceOutcome
    (samplers : MxxSamplerFamily)
    (cutoff : Nat)
    (message : Bool)
    (trace : WorkflowExecutionTrace samplers ToyExample_protocol.bundle.workflow
      (parameters cutoff) (inputs message)) :
    ∃ sample ∈ samplers.gaussianSample (samplerParams cutoff),
      [("decoded", .boolean (Mxx.Ir.thresholdDecodeBool 256 2
        (Mxx.reduceCoefficient 256
          ((if message then 128 else 0) +
            (sample.withSamplerParams (samplerParams cutoff)).coefficients.headD 0))))] =
        trace.entrypointOutput := by
  obtain ⟨encryptOutput, decryptOutput, encryptMember, decryptMember, finalOutput⟩ :=
    stageOutputs samplers cutoff message trace.stageExecutionWitness
  obtain ⟨sample, sampleMember, encryptOutputEq⟩ :=
    encryptOutcome samplers cutoff message encryptOutput encryptMember
  rw [← encryptOutputEq] at decryptMember
  have decryptOutputEq :=
    decryptOutcome samplers cutoff message sample decryptOutput decryptMember
  have entrypointEq := trace.entrypointEq
  rw [finalOutput] at entrypointEq
  have decryptIsEntrypoint : decryptOutput = trace.entrypointOutput := by
    simpa [ToyExample_protocol, Mxx.Ir.lookupStage] using entrypointEq
  refine ⟨sample, sampleMember, ?_⟩
  rw [← decryptIsEntrypoint, ← decryptOutputEq]
  cases message <;> cases sample with
  | mk coefficients modulus ringDimension rows columns =>
      cases coefficients <;>
        simp [ciphertext, Mxx.Matrix.withSamplerParams, samplerParams, Mxx.addCoefficients]

/-- The concrete toy workflow outcome, derived from the generic workflow trace and its erasure
theorem. Only the two stage-local denotations are unfolded above. -/
theorem workflowOutcome
    (samplers : MxxSamplerFamily)
    (cutoff : Nat)
    (message : Bool)
    (output : Mxx.Ir.Environment)
    (member : output ∈ Mxx.Ir.denoteWorkflow samplers ToyExample_protocol.bundle.workflow
      (parameters cutoff) (inputs message)) :
    ∃ sample ∈ samplers.gaussianSample (samplerParams cutoff),
      [("decoded", .boolean (Mxx.Ir.thresholdDecodeBool 256 2
        (Mxx.reduceCoefficient 256
          ((if message then 128 else 0) +
            (sample.withSamplerParams (samplerParams cutoff)).coefficients.headD 0))))] =
        output := by
  have erasedMember : output ∈
      (denoteWorkflowTraces samplers ToyExample_protocol.bundle.workflow
        (parameters cutoff) (inputs message)).map WorkflowExecutionTrace.erase := by
    rw [erase_denoteWorkflowTraces]
    exact member
  obtain ⟨trace, _, traceErases⟩ := List.mem_map.mp erasedMember
  obtain ⟨sample, sampleMember, traceOutput⟩ := traceOutcome samplers cutoff message trace
  exact ⟨sample, sampleMember, traceOutput.trans traceErases⟩

end MxxCorrectness.Proofs.ToyExample
