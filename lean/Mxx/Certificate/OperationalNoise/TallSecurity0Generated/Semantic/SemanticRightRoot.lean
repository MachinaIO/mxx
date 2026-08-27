import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRootShard035

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRoot

open Mxx.Certificate.OperationalNoise
open TallSecurity0ABI
open TallSemantics

/-- The generated theorem application for the reached right exact-zero root. -/
theorem rightRootClaimSound (selector : Nat) (selectorLower : 0 ≤ selector)
    (selectorUpper : selector < 32)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticRightRootResult6275.resultEvent SemanticRightRootResult6275.owner
      (SemanticRightRootResult6275.actual selector witness)
      SemanticRightRootResult6275.rawTerms SemanticRightRootResult6275.summary := by
  exact SemanticRightRootResult6275.claimSound selector selectorLower selectorUpper witness


end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRoot
