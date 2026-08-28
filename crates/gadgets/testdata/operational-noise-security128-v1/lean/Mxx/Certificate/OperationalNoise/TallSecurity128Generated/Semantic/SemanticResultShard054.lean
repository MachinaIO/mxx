import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard054
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard049
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard053

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult6732
def owner : Owner := ⟨.program ⟨257⟩, ⟨51034⟩⟩
def rawTerms : List Term := Proof.Events026.exact6732RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6732
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6732.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6729) (rightBinding := 6730)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨31970⟩) (rightExpression := ⟨51033⟩)
    (transferEvent := 6731)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6680.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6732

namespace SemanticResult6736
def owner : Owner := ⟨.program ⟨257⟩, ⟨54014⟩⟩
def rawTerms : List Term := Proof.Events026.exact6736RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6736
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6736.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6733) (rightBinding := 6734)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51034⟩) (rightExpression := ⟨54013⟩)
    (transferEvent := 6735)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6732.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6672.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6736

namespace SemanticResult6740
def owner : Owner := ⟨.program ⟨257⟩, ⟨56994⟩⟩
def rawTerms : List Term := Proof.Events026.exact6740RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6740
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6740.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6737) (rightBinding := 6738)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54014⟩) (rightExpression := ⟨56993⟩)
    (transferEvent := 6739)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6736.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6664.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6740

namespace SemanticResult6744
def owner : Owner := ⟨.program ⟨257⟩, ⟨59974⟩⟩
def rawTerms : List Term := Proof.Events026.exact6744RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6744
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6744.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6741) (rightBinding := 6742)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56994⟩) (rightExpression := ⟨59973⟩)
    (transferEvent := 6743)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6740.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6656.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6744

namespace SemanticResult6748
def owner : Owner := ⟨.program ⟨257⟩, ⟨62954⟩⟩
def rawTerms : List Term := Proof.Events026.exact6748RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6748
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6748.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6745) (rightBinding := 6746)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59974⟩) (rightExpression := ⟨62953⟩)
    (transferEvent := 6747)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6744.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6648.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6748

namespace SemanticResult6752
def owner : Owner := ⟨.program ⟨257⟩, ⟨66100⟩⟩
def rawTerms : List Term := Proof.Events026.exact6752RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6752
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6752.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6749) (rightBinding := 6750)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨62954⟩) (rightExpression := ⟨66099⟩)
    (transferEvent := 6751)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6748.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6640.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6752

namespace SemanticResult6756
def owner : Owner := ⟨.program ⟨257⟩, ⟨66101⟩⟩
def rawTerms : List Term := Proof.Events026.exact6756RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6756
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6756.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6753) (rightBinding := 6754)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66100⟩) (rightExpression := ⟨26532⟩)
    (transferEvent := 6755)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6752.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6632.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6756

namespace SemanticResult6760
def owner : Owner := ⟨.program ⟨257⟩, ⟨66102⟩⟩
def rawTerms : List Term := Proof.Events026.exact6760RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6760
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6760.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6757) (rightBinding := 6758)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66101⟩) (rightExpression := ⟨29212⟩)
    (transferEvent := 6759)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6756.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6624.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6760

namespace SemanticResult6764
def owner : Owner := ⟨.program ⟨257⟩, ⟨66103⟩⟩
def rawTerms : List Term := Proof.Events026.exact6764RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6764
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6764.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6761) (rightBinding := 6762)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66102⟩) (rightExpression := ⟨34869⟩)
    (transferEvent := 6763)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6760.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6616.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6764

namespace SemanticResult6768
def owner : Owner := ⟨.program ⟨257⟩, ⟨66104⟩⟩
def rawTerms : List Term := Proof.Events026.exact6768RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6768
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6768.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6765) (rightBinding := 6766)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66103⟩) (rightExpression := ⟨37549⟩)
    (transferEvent := 6767)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6764.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6608.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6768

namespace SemanticResult6772
def owner : Owner := ⟨.program ⟨257⟩, ⟨66105⟩⟩
def rawTerms : List Term := Proof.Events026.exact6772RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6772
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6772.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6769) (rightBinding := 6770)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66104⟩) (rightExpression := ⟨40232⟩)
    (transferEvent := 6771)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6768.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6600.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6772

namespace SemanticResult6776
def owner : Owner := ⟨.program ⟨257⟩, ⟨66106⟩⟩
def rawTerms : List Term := Proof.Events026.exact6776RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6776
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6776.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6773) (rightBinding := 6774)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66105⟩) (rightExpression := ⟨42912⟩)
    (transferEvent := 6775)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6772.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6592.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6776

namespace SemanticResult6780
def owner : Owner := ⟨.program ⟨257⟩, ⟨66107⟩⟩
def rawTerms : List Term := Proof.Events026.exact6780RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6780
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6780.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6777) (rightBinding := 6778)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66106⟩) (rightExpression := ⟨45589⟩)
    (transferEvent := 6779)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6776.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6584.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6780

namespace SemanticResult6784
def owner : Owner := ⟨.program ⟨257⟩, ⟨66108⟩⟩
def rawTerms : List Term := Proof.Events026.exact6784RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6784
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6784.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6781) (rightBinding := 6782)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66107⟩) (rightExpression := ⟨48269⟩)
    (transferEvent := 6783)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6780.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6576.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6784

namespace SemanticResult6788
def owner : Owner := ⟨.program ⟨257⟩, ⟨67325⟩⟩
def rawTerms : List Term := Proof.Events026.exact6788RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6788
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6788.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 6785) (rightBinding := 6786)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66108⟩) (rightExpression := ⟨67323⟩)
    (transferEvent := 6787)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6784.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6568.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult6788

namespace SemanticResult6811
def owner : Owner := ⟨.program ⟨257⟩, ⟨67326⟩⟩
def rawTerms : List Term := Proof.Events026.exact6811RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 6811
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult6811.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge6792.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge6792.frameStart)
    (transferEvent := 6791) (owner := owner)
    (leftResult := 6788) (rightResult := 6065)
    (working := LeftOperatorMerge6792.working)
    (reconstruction := LeftOperatorMerge6792.reconstruction)
    (leftReference := .predecessor 0 6789 .coefficient) (rightReference := .predecessor 1 6790 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult6788.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6065.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge6792.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult6811

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
