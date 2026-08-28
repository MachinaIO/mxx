import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard141
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard129
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard134
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard135
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard136
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard137
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard138
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard139
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard140

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult16730
def owner : Owner := ⟨.program ⟨257⟩, ⟨7109⟩⟩
def rawTerms : List Term := Proof.Events065.exact16730RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16730
def producerEvent : Nat := 16729
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16730.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult16730

namespace SemanticResult16734
def owner : Owner := ⟨.program ⟨257⟩, ⟨7110⟩⟩
def rawTerms : List Term := Proof.Events065.exact16734RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16734
def producerEvent : Nat := 16733
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16734.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 16731 .coefficient) (.value (.predecessor 1 16732 .coefficient)), 0, .finite 8192, .scale (.predecessor 0 16731 .coefficient) (.value (.predecessor 1 16732 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult16734

namespace SemanticResult16757
def owner : Owner := ⟨.program ⟨257⟩, ⟨9444⟩⟩
def rawTerms : List Term := Proof.Events065.exact16757RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16757
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16757.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge16738.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge16738.frameStart)
    (transferEvent := 16737) (owner := owner)
    (leftResult := 15977) (rightResult := 16734)
    (working := LeftOperatorMerge16738.working)
    (reconstruction := LeftOperatorMerge16738.reconstruction)
    (leftReference := .predecessor 0 16735 .coefficient) (rightReference := .predecessor 1 16736 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult15977.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16734.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge16738.operationAgreement
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
end SemanticResult16757

namespace SemanticResult16761
def owner : Owner := ⟨.program ⟨257⟩, ⟨9445⟩⟩
def rawTerms : List Term := Proof.Events065.exact16761RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16761
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16761.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16758) (rightBinding := 16759)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9128⟩) (rightExpression := ⟨9444⟩)
    (transferEvent := 16760)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15901.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16757.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16761

namespace SemanticResult16765
def owner : Owner := ⟨.program ⟨257⟩, ⟨9684⟩⟩
def rawTerms : List Term := Proof.Events065.exact16765RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16765
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16765.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16762) (rightBinding := 16763)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9445⟩) (rightExpression := ⟨9683⟩)
    (transferEvent := 16764)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16761.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16722.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16765

namespace SemanticResult16769
def owner : Owner := ⟨.program ⟨257⟩, ⟨9685⟩⟩
def rawTerms : List Term := Proof.Events065.exact16769RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16769
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16769.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16766) (rightBinding := 16767)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9684⟩) (rightExpression := ⟨9682⟩)
    (transferEvent := 16768)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16765.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16682.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16769

namespace SemanticResult16773
def owner : Owner := ⟨.program ⟨257⟩, ⟨9686⟩⟩
def rawTerms : List Term := Proof.Events065.exact16773RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16773
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16773.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16770) (rightBinding := 16771)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9685⟩) (rightExpression := ⟨9681⟩)
    (transferEvent := 16772)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16769.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16642.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16773

namespace SemanticResult16777
def owner : Owner := ⟨.program ⟨257⟩, ⟨9687⟩⟩
def rawTerms : List Term := Proof.Events065.exact16777RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16777
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16777.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16774) (rightBinding := 16775)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9686⟩) (rightExpression := ⟨9680⟩)
    (transferEvent := 16776)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16773.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16602.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16777

namespace SemanticResult16781
def owner : Owner := ⟨.program ⟨257⟩, ⟨9688⟩⟩
def rawTerms : List Term := Proof.Events065.exact16781RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16781
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16781.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16778) (rightBinding := 16779)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9687⟩) (rightExpression := ⟨9679⟩)
    (transferEvent := 16780)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16777.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16562.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16781

namespace SemanticResult16785
def owner : Owner := ⟨.program ⟨257⟩, ⟨9689⟩⟩
def rawTerms : List Term := Proof.Events065.exact16785RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16785
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16785.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16782) (rightBinding := 16783)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9688⟩) (rightExpression := ⟨9678⟩)
    (transferEvent := 16784)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16781.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16522.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16785

namespace SemanticResult16789
def owner : Owner := ⟨.program ⟨257⟩, ⟨9690⟩⟩
def rawTerms : List Term := Proof.Events065.exact16789RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16789
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16789.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16786) (rightBinding := 16787)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9689⟩) (rightExpression := ⟨9677⟩)
    (transferEvent := 16788)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16785.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16482.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16789

namespace SemanticResult16793
def owner : Owner := ⟨.program ⟨257⟩, ⟨9691⟩⟩
def rawTerms : List Term := Proof.Events065.exact16793RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16793
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16793.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16790) (rightBinding := 16791)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9690⟩) (rightExpression := ⟨9676⟩)
    (transferEvent := 16792)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16789.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16442.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16793

namespace SemanticResult16797
def owner : Owner := ⟨.program ⟨257⟩, ⟨9692⟩⟩
def rawTerms : List Term := Proof.Events065.exact16797RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16797
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16797.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16794) (rightBinding := 16795)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9691⟩) (rightExpression := ⟨9675⟩)
    (transferEvent := 16796)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16793.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16402.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16797

namespace SemanticResult16801
def owner : Owner := ⟨.program ⟨257⟩, ⟨9693⟩⟩
def rawTerms : List Term := Proof.Events065.exact16801RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16801
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16801.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16798) (rightBinding := 16799)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9692⟩) (rightExpression := ⟨9674⟩)
    (transferEvent := 16800)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16797.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16362.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16801

namespace SemanticResult16805
def owner : Owner := ⟨.program ⟨257⟩, ⟨9694⟩⟩
def rawTerms : List Term := Proof.Events065.exact16805RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16805
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16805.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16802) (rightBinding := 16803)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9693⟩) (rightExpression := ⟨9673⟩)
    (transferEvent := 16804)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16801.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16322.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16805

namespace SemanticResult16809
def owner : Owner := ⟨.program ⟨257⟩, ⟨9695⟩⟩
def rawTerms : List Term := Proof.Events065.exact16809RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16809
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16809.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16806) (rightBinding := 16807)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9694⟩) (rightExpression := ⟨9672⟩)
    (transferEvent := 16808)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16805.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16282.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16809

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
