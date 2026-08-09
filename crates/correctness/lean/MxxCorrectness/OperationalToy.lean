import MxxCorrectness.Generated.ToyExample.Ir
import Mxx.Certificate.OperationalBounds

namespace MxxCorrectness.OperationalToy

open MxxCorrectness.Generated.ToyExample

private def environment : Mxx.Ir.ParamEnvironment :=
  [(.parameter "cutoff", .integer 10)]

private def report :
    Except Mxx.Certificate.OperationalError Mxx.Certificate.OperationalNoiseCheckReport := do
  let workflow : Mxx.Certificate.OperationalWorkflowSpec := {
    workflow := ToyExample_protocol.bundle.workflow
    inputContract := ToyExample_protocol.bundle.inputContract
  }
  let results ← Mxx.Certificate.evaluateWorkflowOperational workflow
    [("encrypt", ToyExample_stage_encrypt_derivation),
     ("decrypt", ToyExample_stage_decrypt_derivation)] environment []
  let binding ← match ToyExample_protocol.bundle.anchorBindings.find? fun binding =>
      binding.anchor.label == "toy.decoder.residual" with
    | some value => pure value
    | none => throw (.missingStageResult "anchor" "toy.decoder.residual")
  let wire ← match binding.wires with
    | [value] => pure value
    | _ => throw (.missingStageResult "anchor-wire" "toy.decoder.residual")
  let stage ← match results.find? fun result => result.stage == wire.stage.name with
    | some value => pure value
    | none => throw (.missingStageResult "stage" wire.stage.name)
  let residual ← Mxx.Certificate.lookupFact (wire.node.value + 1) stage.facts
    { node := wire.node.value, port := wire.port }
  let residual ← match residual with
    | .matrix value => pure value
    | _ => throw (.missingStageResult "matrix" "toy.decoder.residual")
  Mxx.Certificate.decoderNoiseCheckReport results residual environment 2 256

/-- The generated Toy workflow and derivation reach the same generic operational report used by
application parameter search. This is a closed checker-evaluation fixture, not a runtime-soundness
theorem. -/
example : report.map (fun result => result.accepted) = .ok true := by
  native_decide

/-- The report obtains the cutoff from the generated Gaussian node and applies the generic strict
decoder inequality, rather than a Toy-specific Rust formula. -/
example : report.map (fun result => result.obligations) =
    .ok [.decoderThreshold 2 256 10] := by
  native_decide

private def corruptedCarrierDerivation : Mxx.Certificate.ProgramDerivation := {
  ToyExample_stage_encrypt_derivation with
  root := {
    ToyExample_stage_encrypt_derivation.root with
    attachments := [{
      ownerNamespace := "mxx-correctness"
      ruleName := "protocol-boolean-signal-grouping"
      roles := [
        ("value", { node := 4, port := 0 }),
        ("selector", { node := 0, port := 0 }),
        ("zero", { node := 2, port := 0 }),
        ("carrier", { node := 2, port := 0 })
      ]
    }]
  }
}

/-- The attachment is only an untrusted structural hint. Replacing the carrier by the zero branch
is rejected before it can assign a Large role. -/
example : (match Mxx.Certificate.evaluateProgramOperationalWithLayouts
    ToyExample_stage_encrypt corruptedCarrierDerivation environment [] with
  | .error (.invalidDerivationAttachment "mxx-correctness"
      "protocol-boolean-signal-grouping") => true
  | _ => false) = true := by
  native_decide

end MxxCorrectness.OperationalToy
