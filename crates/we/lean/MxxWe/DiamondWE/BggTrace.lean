import MxxBgg.Invariant

namespace Mxx.We.DiamondWE

open Mxx.Bgg
open Mxx.Gadgets
open Mxx.Primitives

/-
  BGG layer bounds are uniform over the active slots.  This is the numerical
  recurrence used by the Diamond circuit: an AND gate first consumes one
  preimage and an XOR gate subsequently combines that product with two linear
  terms.  The factor `n` in each matrix product is the one-output-coefficient
  negacyclic convolution bound; no `n^2` factor is introduced here.
-/
def bggLayerBound (ringDimension gadgetColumns preimageBound previousBound : Nat) : Nat :=
  Mxx.Bgg.booleanSixGateUniformBound ringDimension gadgetColumns preimageBound previousBound

/- The bound after `layers` applications of the fixed BGG layer recurrence. -/
def bggBoundAfter (layers ringDimension gadgetColumns preimageBound baseBound : Nat) : Nat :=
  Nat.rec baseBound
    (fun _ previous => bggLayerBound ringDimension gadgetColumns preimageBound previous)
    layers

@[simp] theorem bggBoundAfter_zero
    (ringDimension gadgetColumns preimageBound baseBound : Nat) :
    bggBoundAfter 0 ringDimension gadgetColumns preimageBound baseBound = baseBound := rfl

@[simp] theorem bggBoundAfter_succ
    (layers ringDimension gadgetColumns preimageBound baseBound : Nat) :
    bggBoundAfter (layers + 1) ringDimension gadgetColumns preimageBound baseBound =
      bggLayerBound ringDimension gadgetColumns preimageBound
        (bggBoundAfter layers ringDimension gadgetColumns preimageBound baseBound) := rfl

/- A single layer certificate contains the actual certificate for every active
   gate.  There is no default witness: callers must provide the certificate
   emitted by the corresponding runtime operation for every slot. -/
structure BggLayerCertificate
    {shape : LayeredBoolCircuitShape}
    (circuit : LayeredBoolCircuit shape)
    (valid : circuit.Valid)
    {q n secretColumns gadgetColumns : Nat}
    (mask : ExactMatrix q n 1 secretColumns)
    (gadget : ExactMatrix q n secretColumns gadgetColumns)
    (oneMessage : ExactPoly q n)
    (one : BooleanEncodingValue mask gadget)
    (layer : Fin shape.depth)
    (previous : ExactLayerState mask gadget oneMessage
      (circuit.previousNatWidth layer)) where
  preimageBound : Nat
  nextBound : Nat
  oneCarries : EncodingCarriesBool one.encoding oneMessage true
  oneMessageIdempotent : oneMessage * oneMessage = oneMessage
  witnesses : ∀ slot, CertifiedGateWitness mask gadget previous.values one
    (activeGateSpec circuit valid layer slot) preimageBound nextBound
  output : ExactLayerState mask gadget oneMessage (circuit.activeWidth layer)
  output_eq : output = exactAdvance previous one
    (fun slot => activeGateSpec circuit valid layer slot)
    witnesses oneCarries oneMessageIdempotent

/- The output state of a layer certificate is definitionally the state made by
   applying the supplied gate certificates.  This theorem is kept explicit so
   generated application proofs can rewrite without unfolding all gate cases. -/
theorem BggLayerCertificate.output_noiseBound
    {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape}
    {valid : circuit.Valid}
    {q n secretColumns gadgetColumns : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    {one : BooleanEncodingValue mask gadget}
    {layer : Fin shape.depth}
    {previous : ExactLayerState mask gadget oneMessage
      (circuit.previousNatWidth layer)}
    (certificate : BggLayerCertificate circuit valid mask gadget oneMessage one layer previous) :
    certificate.output.noiseBound = certificate.nextBound := by
  rw [certificate.output_eq]
  rfl

/- The executable Boolean layer and the certified BGG layer consume the same
   typed gate specifications.  Thus the layer result is recovered from the
   circuit evaluator, not supplied as a semantic equation by the caller. -/
theorem BggLayerCertificate.runtime
    {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape}
    {valid : circuit.Valid}
    {q n secretColumns gadgetColumns : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    {one : BooleanEncodingValue mask gadget}
    {layer : Fin shape.depth}
    {previous : ExactLayerState mask gadget oneMessage
      (circuit.previousNatWidth layer)}
    (certificate : BggLayerCertificate circuit valid mask gadget oneMessage one layer previous) :
    circuit.evaluateLayer? layer.val (Array.ofFn previous.bits) =
      some (Array.ofFn certificate.output.bits) := by
  rw [certificate.output_eq]
  exact Mxx.Bgg.exactAdvance_matches_runtimeLayer valid layer previous one
    certificate.witnesses one.carries certificate.oneMessageIdempotent

/- A uniform BGG trace fixes the numerical recurrence at every step.  Its
   constructors still require concrete `CertifiedGateWitness` values, so this
   package cannot silently replace a failed preimage/product certificate. -/
inductive UniformBggTrace
    {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape}
    {q n secretColumns gadgetColumns : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    (valid : circuit.Valid)
    (one : BooleanEncodingValue mask gadget)
    (oneCarries : EncodingCarriesBool one.encoding oneMessage true)
    (oneMessageIdempotent : oneMessage * oneMessage = oneMessage)
    (initial : ExactLayerState mask gadget oneMessage shape.inputWidth)
    (ringDimension gadgetColumns preimageBound baseBound : Nat) :
    Nat → ExactLayerSigma mask gadget oneMessage → Prop
  | initial (initial_bound : initial.noiseBound = baseBound) :
      UniformBggTrace valid one oneCarries oneMessageIdempotent initial
        ringDimension gadgetColumns preimageBound baseBound 0
        ⟨shape.inputWidth, initial⟩
  | step {completed width}
      {state : ExactLayerState mask gadget oneMessage width}
      (run : UniformBggTrace valid one oneCarries oneMessageIdempotent initial
        ringDimension gadgetColumns preimageBound baseBound completed ⟨width, state⟩)
      (position : Fin shape.depth)
      (position_eq : position.val = completed)
      (width_eq : width = circuit.previousNatWidth position)
      (witnesses : ∀ slot, CertifiedGateWitness mask gadget
        (Mxx.Bgg.castExactLayerState width_eq state).values one
        (activeGateSpec circuit valid position slot) preimageBound
        (bggBoundAfter (completed + 1) ringDimension gadgetColumns preimageBound baseBound)) :
      UniformBggTrace valid one oneCarries oneMessageIdempotent initial
        ringDimension gadgetColumns preimageBound baseBound (completed + 1)
        ⟨circuit.activeWidth position,
          exactAdvance (Mxx.Bgg.castExactLayerState width_eq state) one
            (fun slot => activeGateSpec circuit valid position slot)
            witnesses oneCarries oneMessageIdempotent⟩

/- Uniform traces are accepted by the generic certified-run API used by the
   rest of the application.  The conversion only changes the proposition's
   wrapper; it never manufactures a layer witness. -/
def UniformBggTrace.toCertified
    {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape}
    {q n secretColumns gadgetColumns : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    {valid : circuit.Valid}
    {one : BooleanEncodingValue mask gadget}
    {oneCarries : EncodingCarriesBool one.encoding oneMessage true}
    {oneMessageIdempotent : oneMessage * oneMessage = oneMessage}
    {initial : ExactLayerState mask gadget oneMessage shape.inputWidth}
    {ringDimension gadgetColumns preimageBound baseBound completed : Nat}
    {terminal : ExactLayerSigma mask gadget oneMessage}
    (run : UniformBggTrace valid one oneCarries oneMessageIdempotent initial
      ringDimension gadgetColumns preimageBound baseBound completed terminal) :
    CertifiedLayeredRun valid one oneCarries oneMessageIdempotent initial completed terminal := by
  induction run with
  | initial initial_bound => exact .initial
  | @step completed width state run position position_eq width_eq witnesses ih =>
      exact .step ih position position_eq width_eq witnesses

/- The recurrence is proved by induction over the trace.  The terminal state's
   bound is therefore the recurrence result, rather than a caller-provided
   final bound. -/
theorem UniformBggTrace.noiseBound_eq
    {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape}
    {q n secretColumns gadgetColumns : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    {valid : circuit.Valid}
    {one : BooleanEncodingValue mask gadget}
    {oneCarries : EncodingCarriesBool one.encoding oneMessage true}
    {oneMessageIdempotent : oneMessage * oneMessage = oneMessage}
    {initial : ExactLayerState mask gadget oneMessage shape.inputWidth}
    {ringDimension gadgetColumns preimageBound baseBound completed : Nat}
    {terminal : ExactLayerSigma mask gadget oneMessage}
    (run : UniformBggTrace valid one oneCarries oneMessageIdempotent initial
      ringDimension gadgetColumns preimageBound baseBound completed terminal) :
    match terminal with
    | ⟨_, state⟩ => state.noiseBound =
      bggBoundAfter completed ringDimension gadgetColumns preimageBound baseBound := by
  induction run with
  | initial initial_bound =>
      simp [initial_bound]
  | @step completed width state run position position_eq width_eq witnesses ih =>
      simp [bggBoundAfter, ih]

/- The final active slot remains the evaluator's selected slot.  This helper
   packages both the evaluator replay and the certified final BGG facts for an
   accepting Boolean circuit instance. -/
theorem UniformBggTrace.acceptingOutput
    {shape : LayeredBoolCircuitShape}
    {circuit : LayeredBoolCircuit shape}
    {q n secretColumns gadgetColumns : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    {valid : circuit.Valid}
    {one : BooleanEncodingValue mask gadget}
    {oneCarries : EncodingCarriesBool one.encoding oneMessage true}
    {oneMessageIdempotent : oneMessage * oneMessage = oneMessage}
    {initial : ExactLayerState mask gadget oneMessage shape.inputWidth}
    {ringDimension gadgetColumns preimageBound baseBound : Nat}
    (instanceBits : Fin shape.instanceWidth → Bool)
    (witnessBits : Fin shape.witnessWidth → Bool)
    (initialRuntime : Array.ofFn initial.bits =
      (Array.ofFn instanceBits).append (Array.ofFn witnessBits))
    (final : ExactLayerState mask gadget oneMessage
      (circuit.activeWidth (Mxx.Bgg.finalLayer valid)))
    (run : UniformBggTrace valid one oneCarries oneMessageIdempotent initial
      ringDimension gadgetColumns preimageBound baseBound shape.depth ⟨_, final⟩)
    (accepted : circuit.evaluate valid instanceBits witnessBits = some true) :
    EncodingCarriesBool (final.values (Mxx.Bgg.outputIndex circuit valid)).encoding
      oneMessage true ∧
      EncodingErrorWithin (final.values (Mxx.Bgg.outputIndex circuit valid)).encoding
        final.noiseBound := by
  exact Mxx.Bgg.acceptingCertifiedBggOutput instanceBits witnessBits initialRuntime final
    run.toCertified accepted

end Mxx.We.DiamondWE
