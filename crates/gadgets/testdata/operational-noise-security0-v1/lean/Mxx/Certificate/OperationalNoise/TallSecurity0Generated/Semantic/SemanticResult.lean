import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultImport02_000

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResult

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

theorem resultClaimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env
      SemanticResult107564.resultEvent SemanticResult107564.owner
      (SemanticResult107564.actual selector witness)
      SemanticResult107564.rawTerms SemanticResult107564.summary := by
  exact SemanticResult107564.claimSound selector selectorLower selectorUpper witness

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResult
