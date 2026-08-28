import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard2161
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard142
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard2160

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult308622
def owner : Owner := ⟨.program ⟨257⟩, ⟨71547⟩⟩
def rawTerms : List Term := Proof.Events1205.exact308622RawTerms
def summary : Bound := (.finite 146340160251294585514145619529726732840708448851938772156114662662487408692)
def resultEvent : Nat := 308622
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult308622.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteLeftMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge308204.frameStart)
    (coefficientBound := .large) (coefficientTransfer := 308203) (resultEvent := resultEvent)
    (owner := owner) (leftOwner := SemanticResult308200.owner)
    (rightOwner := SemanticResult16883.owner)
    (leftResult := 308200) (rightResult := 16883)
    (leftActual := SemanticResult308200.actual selector witness)
    (rightActual := SemanticResult16883.actual selector witness)
    (leftRaw := SemanticResult308200.rawTerms)
    (rightRaw := SemanticResult16883.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 146340160251294585514145619529726732840708448851938772156114662662487408692)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 308201) (rightBinding := 308202)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71546⟩) (rightExpression := ⟨67657⟩)
    (base := LeftOperatorMerge308204.base)
    (reconstruction := LeftOperatorMerge308204.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult308200.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16883.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge308204.operationAgreement
  · rfl
  · decide
end SemanticResult308622

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
