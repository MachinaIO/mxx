import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultImport02_000

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResult

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

theorem resultClaimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env
      SemanticResult308622.resultEvent SemanticResult308622.owner
      (SemanticResult308622.actual selector witness)
      SemanticResult308622.rawTerms SemanticResult308622.summary := by
  exact SemanticResult308622.claimSound selector selectorLower selectorUpper witness

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResult
