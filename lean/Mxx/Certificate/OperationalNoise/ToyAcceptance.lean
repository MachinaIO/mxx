import Mxx.Certificate.OperationalNoise.ToyGenerated.Proof

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.ToyGenerated

open Mxx.Certificate.OperationalNoise.ToyABI

/-- Kernel-checked acceptance of the exact generated toy certificate and proof event sequence. -/
theorem accepted :
    ∀ witness : ToyReplayWitness events, ToyOperationalClaim events witness :=
  operationalProof proofValid

#print axioms accepted

end Mxx.Certificate.OperationalNoise.ToyGenerated
