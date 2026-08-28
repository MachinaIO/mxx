import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard048
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard045
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard046
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard047

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult5980
def owner : Owner := ⟨.program ⟨257⟩, ⟨32027⟩⟩
def rawTerms : List Term := Proof.Events023.exact5980RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5980
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5980.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5977) (rightBinding := 5978)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22007⟩) (rightExpression := ⟨32026⟩)
    (transferEvent := 5979)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5976.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5940.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5980

namespace SemanticResult5984
def owner : Owner := ⟨.program ⟨257⟩, ⟨51091⟩⟩
def rawTerms : List Term := Proof.Events023.exact5984RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5984
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5984.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5981) (rightBinding := 5982)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32027⟩) (rightExpression := ⟨51090⟩)
    (transferEvent := 5983)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5980.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5932.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5984

namespace SemanticResult5988
def owner : Owner := ⟨.program ⟨257⟩, ⟨54071⟩⟩
def rawTerms : List Term := Proof.Events023.exact5988RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5988
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5988.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5985) (rightBinding := 5986)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51091⟩) (rightExpression := ⟨54070⟩)
    (transferEvent := 5987)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5984.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5924.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5988

namespace SemanticResult5992
def owner : Owner := ⟨.program ⟨257⟩, ⟨57051⟩⟩
def rawTerms : List Term := Proof.Events023.exact5992RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5992
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5992.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5989) (rightBinding := 5990)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54071⟩) (rightExpression := ⟨57050⟩)
    (transferEvent := 5991)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5988.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5916.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5992

namespace SemanticResult5996
def owner : Owner := ⟨.program ⟨257⟩, ⟨60031⟩⟩
def rawTerms : List Term := Proof.Events023.exact5996RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 5996
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult5996.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5993) (rightBinding := 5994)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57051⟩) (rightExpression := ⟨60030⟩)
    (transferEvent := 5995)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5992.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5908.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult5996

namespace SemanticResult6000
def owner : Owner := ⟨.program ⟨257⟩, ⟨63011⟩⟩
def rawTerms : List Term := Proof.Events023.exact6000RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6000
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6000.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 5997) (rightBinding := 5998)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60031⟩) (rightExpression := ⟨63010⟩)
    (transferEvent := 5999)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult5996.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5900.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6000

namespace SemanticResult6004
def owner : Owner := ⟨.program ⟨257⟩, ⟨66310⟩⟩
def rawTerms : List Term := Proof.Events023.exact6004RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6004
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6004.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6001) (rightBinding := 6002)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63011⟩) (rightExpression := ⟨66309⟩)
    (transferEvent := 6003)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6000.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5892.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6004

namespace SemanticResult6008
def owner : Owner := ⟨.program ⟨257⟩, ⟨66311⟩⟩
def rawTerms : List Term := Proof.Events023.exact6008RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6008
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6008.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6005) (rightBinding := 6006)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66310⟩) (rightExpression := ⟨26571⟩)
    (transferEvent := 6007)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6004.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5884.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6008

namespace SemanticResult6012
def owner : Owner := ⟨.program ⟨257⟩, ⟨66312⟩⟩
def rawTerms : List Term := Proof.Events023.exact6012RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6012
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6012.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6009) (rightBinding := 6010)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66311⟩) (rightExpression := ⟨29251⟩)
    (transferEvent := 6011)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6008.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5876.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6012

namespace SemanticResult6016
def owner : Owner := ⟨.program ⟨257⟩, ⟨66313⟩⟩
def rawTerms : List Term := Proof.Events023.exact6016RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6016
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6016.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6013) (rightBinding := 6014)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66312⟩) (rightExpression := ⟨34908⟩)
    (transferEvent := 6015)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6012.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5868.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6016

namespace SemanticResult6020
def owner : Owner := ⟨.program ⟨257⟩, ⟨66314⟩⟩
def rawTerms : List Term := Proof.Events023.exact6020RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6020
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6020.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6017) (rightBinding := 6018)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66313⟩) (rightExpression := ⟨37588⟩)
    (transferEvent := 6019)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6016.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5860.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6020

namespace SemanticResult6024
def owner : Owner := ⟨.program ⟨257⟩, ⟨66315⟩⟩
def rawTerms : List Term := Proof.Events023.exact6024RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6024
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6024.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6021) (rightBinding := 6022)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66314⟩) (rightExpression := ⟨40271⟩)
    (transferEvent := 6023)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6020.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5852.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6024

namespace SemanticResult6028
def owner : Owner := ⟨.program ⟨257⟩, ⟨66316⟩⟩
def rawTerms : List Term := Proof.Events023.exact6028RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6028
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6028.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6025) (rightBinding := 6026)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66315⟩) (rightExpression := ⟨42951⟩)
    (transferEvent := 6027)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6024.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5844.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6028

namespace SemanticResult6032
def owner : Owner := ⟨.program ⟨257⟩, ⟨66317⟩⟩
def rawTerms : List Term := Proof.Events023.exact6032RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6032
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6032.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6029) (rightBinding := 6030)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66316⟩) (rightExpression := ⟨45628⟩)
    (transferEvent := 6031)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6028.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5836.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6032

namespace SemanticResult6036
def owner : Owner := ⟨.program ⟨257⟩, ⟨66318⟩⟩
def rawTerms : List Term := Proof.Events023.exact6036RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6036
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6036.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6033) (rightBinding := 6034)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66317⟩) (rightExpression := ⟨48308⟩)
    (transferEvent := 6035)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6032.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5828.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6036

namespace SemanticResult6040
def owner : Owner := ⟨.program ⟨257⟩, ⟨67385⟩⟩
def rawTerms : List Term := Proof.Events023.exact6040RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6040
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6040.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6037) (rightBinding := 6038)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66318⟩) (rightExpression := ⟨67383⟩)
    (transferEvent := 6039)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6036.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5820.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6040

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
