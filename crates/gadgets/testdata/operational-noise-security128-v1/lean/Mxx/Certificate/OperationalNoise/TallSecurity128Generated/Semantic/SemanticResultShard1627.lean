import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1627
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1625
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1626

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult232024
def owner : Owner := ⟨.program ⟨257⟩, ⟨18848⟩⟩
def rawTerms : List Term := Proof.Events906.exact232024RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232024
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232024.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232021) (rightBinding := 232022)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16019⟩) (rightExpression := ⟨18847⟩)
    (transferEvent := 232023)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232020.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231997.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232024

namespace SemanticResult232028
def owner : Owner := ⟨.program ⟨257⟩, ⟨22068⟩⟩
def rawTerms : List Term := Proof.Events906.exact232028RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232028
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232028.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232025) (rightBinding := 232026)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18848⟩) (rightExpression := ⟨22067⟩)
    (transferEvent := 232027)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232024.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231974.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232028

namespace SemanticResult232032
def owner : Owner := ⟨.program ⟨257⟩, ⟨32088⟩⟩
def rawTerms : List Term := Proof.Events906.exact232032RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232032
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232032.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232029) (rightBinding := 232030)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22068⟩) (rightExpression := ⟨32087⟩)
    (transferEvent := 232031)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232028.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231951.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232032

namespace SemanticResult232036
def owner : Owner := ⟨.program ⟨257⟩, ⟨51143⟩⟩
def rawTerms : List Term := Proof.Events906.exact232036RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232036
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232036.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232033) (rightBinding := 232034)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32088⟩) (rightExpression := ⟨51142⟩)
    (transferEvent := 232035)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232032.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231928.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232036

namespace SemanticResult232040
def owner : Owner := ⟨.program ⟨257⟩, ⟨54123⟩⟩
def rawTerms : List Term := Proof.Events906.exact232040RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232040
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232040.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232037) (rightBinding := 232038)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51143⟩) (rightExpression := ⟨54122⟩)
    (transferEvent := 232039)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232036.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231905.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232040

namespace SemanticResult232044
def owner : Owner := ⟨.program ⟨257⟩, ⟨57103⟩⟩
def rawTerms : List Term := Proof.Events906.exact232044RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232044
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232044.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232041) (rightBinding := 232042)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54123⟩) (rightExpression := ⟨57102⟩)
    (transferEvent := 232043)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232040.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231882.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232044

namespace SemanticResult232048
def owner : Owner := ⟨.program ⟨257⟩, ⟨60083⟩⟩
def rawTerms : List Term := Proof.Events906.exact232048RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232048
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232048.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232045) (rightBinding := 232046)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57103⟩) (rightExpression := ⟨60082⟩)
    (transferEvent := 232047)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232044.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231859.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232048

namespace SemanticResult232052
def owner : Owner := ⟨.program ⟨257⟩, ⟨63063⟩⟩
def rawTerms : List Term := Proof.Events906.exact232052RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232052
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232052.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232049) (rightBinding := 232050)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60083⟩) (rightExpression := ⟨63062⟩)
    (transferEvent := 232051)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232048.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231836.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232052

namespace SemanticResult232056
def owner : Owner := ⟨.program ⟨257⟩, ⟨66532⟩⟩
def rawTerms : List Term := Proof.Events906.exact232056RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232056
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232056.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232053) (rightBinding := 232054)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63063⟩) (rightExpression := ⟨66531⟩)
    (transferEvent := 232055)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232052.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231813.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232056

namespace SemanticResult232060
def owner : Owner := ⟨.program ⟨257⟩, ⟨66533⟩⟩
def rawTerms : List Term := Proof.Events906.exact232060RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232060
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232060.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232057) (rightBinding := 232058)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66532⟩) (rightExpression := ⟨26606⟩)
    (transferEvent := 232059)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232056.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231790.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232060

namespace SemanticResult232064
def owner : Owner := ⟨.program ⟨257⟩, ⟨66534⟩⟩
def rawTerms : List Term := Proof.Events906.exact232064RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232064
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232064.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232061) (rightBinding := 232062)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66533⟩) (rightExpression := ⟨29286⟩)
    (transferEvent := 232063)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232060.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231767.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232064

namespace SemanticResult232068
def owner : Owner := ⟨.program ⟨257⟩, ⟨66535⟩⟩
def rawTerms : List Term := Proof.Events906.exact232068RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232068
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232068.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232065) (rightBinding := 232066)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66534⟩) (rightExpression := ⟨34950⟩)
    (transferEvent := 232067)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232064.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231744.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232068

namespace SemanticResult232072
def owner : Owner := ⟨.program ⟨257⟩, ⟨66536⟩⟩
def rawTerms : List Term := Proof.Events906.exact232072RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232072
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232072.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232069) (rightBinding := 232070)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66535⟩) (rightExpression := ⟨37630⟩)
    (transferEvent := 232071)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232068.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231721.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232072

namespace SemanticResult232076
def owner : Owner := ⟨.program ⟨257⟩, ⟨66537⟩⟩
def rawTerms : List Term := Proof.Events906.exact232076RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232076
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232076.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232073) (rightBinding := 232074)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66536⟩) (rightExpression := ⟨40306⟩)
    (transferEvent := 232075)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232072.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231698.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232076

namespace SemanticResult232080
def owner : Owner := ⟨.program ⟨257⟩, ⟨66538⟩⟩
def rawTerms : List Term := Proof.Events906.exact232080RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232080
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232080.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232077) (rightBinding := 232078)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66537⟩) (rightExpression := ⟨42986⟩)
    (transferEvent := 232079)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232076.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231675.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232080

namespace SemanticResult232084
def owner : Owner := ⟨.program ⟨257⟩, ⟨66539⟩⟩
def rawTerms : List Term := Proof.Events906.exact232084RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232084
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232084.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232081) (rightBinding := 232082)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66538⟩) (rightExpression := ⟨45670⟩)
    (transferEvent := 232083)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232080.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult231652.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232084

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
