import Mxx.Certificate.OperationalNoise.ToyGenerated.Proof

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.ToyGenerated

open Mxx.Certificate.OperationalNoise.ToyABI

/-- Kernel-checked acceptance of the exact generated toy certificate and proof event sequence. -/
theorem accepted :
    ∀ witness : ToyReplayWitness events, ToyOperationalClaim events witness :=
  operationalProof proofValid

/-- The existing toy claim is the specialization of the shared factor-generic value claim. -/
theorem acceptedAsGeneric :
    ∀ witness : ToyReplayWitness events,
      TallSemantics.ValueClaim.Interprets 257 witness.env (ToyResidual events witness.env)
          (finalValue events).toSemanticClaim ∧
        2 * 2 * centeredNorm 257 (ToyResidual events witness.env) < 257 := by
  intro witness
  simpa [ToyOperationalClaim, ToyValue.Interprets] using accepted witness

#print axioms accepted
#print axioms acceptedAsGeneric

end Mxx.Certificate.OperationalNoise.ToyGenerated
