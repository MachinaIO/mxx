import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard142
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard121
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard129
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard130
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard131
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard132
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard133
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard141

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult16813
def owner : Owner := ⟨.program ⟨257⟩, ⟨9696⟩⟩
def rawTerms : List Term := Proof.Events065.exact16813RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16813
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16813.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16810) (rightBinding := 16811)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9695⟩) (rightExpression := ⟨9671⟩)
    (transferEvent := 16812)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16809.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16242.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16813

namespace SemanticResult16817
def owner : Owner := ⟨.program ⟨257⟩, ⟨9697⟩⟩
def rawTerms : List Term := Proof.Events065.exact16817RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16817
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16817.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16814) (rightBinding := 16815)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9696⟩) (rightExpression := ⟨9670⟩)
    (transferEvent := 16816)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16813.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16202.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16817

namespace SemanticResult16821
def owner : Owner := ⟨.program ⟨257⟩, ⟨9698⟩⟩
def rawTerms : List Term := Proof.Events065.exact16821RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16821
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16821.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16818) (rightBinding := 16819)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9697⟩) (rightExpression := ⟨9669⟩)
    (transferEvent := 16820)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16817.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16162.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16821

namespace SemanticResult16825
def owner : Owner := ⟨.program ⟨257⟩, ⟨9699⟩⟩
def rawTerms : List Term := Proof.Events065.exact16825RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16825
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16825.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16822) (rightBinding := 16823)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9698⟩) (rightExpression := ⟨9668⟩)
    (transferEvent := 16824)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16821.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16122.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16825

namespace SemanticResult16829
def owner : Owner := ⟨.program ⟨257⟩, ⟨9700⟩⟩
def rawTerms : List Term := Proof.Events065.exact16829RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16829
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16829.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16826) (rightBinding := 16827)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9699⟩) (rightExpression := ⟨9667⟩)
    (transferEvent := 16828)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16825.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16082.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16829

namespace SemanticResult16833
def owner : Owner := ⟨.program ⟨257⟩, ⟨9701⟩⟩
def rawTerms : List Term := Proof.Events065.exact16833RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16833
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16833.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16830) (rightBinding := 16831)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9700⟩) (rightExpression := ⟨9666⟩)
    (transferEvent := 16832)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16829.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16042.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16833

namespace SemanticResult16837
def owner : Owner := ⟨.program ⟨257⟩, ⟨9702⟩⟩
def rawTerms : List Term := Proof.Events065.exact16837RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16837
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16837.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16834) (rightBinding := 16835)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9701⟩) (rightExpression := ⟨9665⟩)
    (transferEvent := 16836)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16833.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16002.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16837

namespace SemanticResult16879
def owner : Owner := ⟨.program ⟨257⟩, ⟨9703⟩⟩
def rawTerms : List Term := Proof.Events065.exact16879RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16879
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16879.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge16841.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge16841.frameStart)
    (transferEvent := 16840) (owner := owner)
    (leftResult := 27) (rightResult := 16837)
    (working := LeftOperatorMerge16841.working)
    (reconstruction := LeftOperatorMerge16841.reconstruction)
    (leftReference := .predecessor 0 16838 .coefficient) (rightReference := .predecessor 1 16839 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16837.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge16841.operationAgreement
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
end SemanticResult16879

namespace SemanticResult16883
def owner : Owner := ⟨.program ⟨257⟩, ⟨67657⟩⟩
def rawTerms : List Term := Proof.Events065.exact16883RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16883
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16883.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16880) (rightBinding := 16881)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9703⟩) (rightExpression := ⟨67655⟩)
    (transferEvent := 16882)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16879.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15487.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16883

namespace SemanticResult16885
def owner : Owner := ⟨.program ⟨257⟩, ⟨34⟩⟩
def rawTerms : List Term := Proof.Events065.exact16885RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16885
def producerEvent : Nat := 16884
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16885.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .finite 26, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult16885

namespace SemanticResult16906
def owner : Owner := ⟨.program ⟨257⟩, ⟨5672⟩⟩
def rawTerms : List Term := Proof.Events066.exact16906RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16906
def producerEvent : Nat := 16905
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16906.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 16900 .coefficient), 0, .finite 1, .identity (.predecessor 0 16900 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult16906

namespace SemanticResult16911
def owner : Owner := ⟨.program ⟨257⟩, ⟨6963⟩⟩
def rawTerms : List Term := Proof.Events066.exact16911RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16911
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16911.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge16910.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge16910.frameStart)
    (transferEvent := 16909) (owner := owner)
    (leftResult := 16906) (rightResult := 2)
    (working := LeftOperatorMerge16910.working)
    (reconstruction := LeftOperatorMerge16910.reconstruction)
    (leftReference := .predecessor 0 16907 .coefficient) (rightReference := .predecessor 1 16908 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult16906.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge16910.operationAgreement
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
end SemanticResult16911

namespace SemanticResult16922
def owner : Owner := ⟨.program ⟨257⟩, ⟨5441⟩⟩
def rawTerms : List Term := Proof.Events066.exact16922RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16922
def producerEvent : Nat := 16921
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16922.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 16916 .coefficient), 0, .finite 1, .identity (.predecessor 0 16916 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult16922

namespace SemanticResult16927
def owner : Owner := ⟨.program ⟨257⟩, ⟨7589⟩⟩
def rawTerms : List Term := Proof.Events066.exact16927RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16927
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16927.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge16926.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge16926.frameStart)
    (transferEvent := 16925) (owner := owner)
    (leftResult := 16922) (rightResult := 15503)
    (working := LeftOperatorMerge16926.working)
    (reconstruction := LeftOperatorMerge16926.reconstruction)
    (leftReference := .predecessor 0 16923 .coefficient) (rightReference := .predecessor 1 16924 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult16922.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15503.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge16926.operationAgreement
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
end SemanticResult16927

namespace SemanticResult16931
def owner : Owner := ⟨.program ⟨257⟩, ⟨9283⟩⟩
def rawTerms : List Term := Proof.Events066.exact16931RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 16931
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16931.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 16928) (rightBinding := 16929)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7589⟩) (rightExpression := ⟨6963⟩)
    (transferEvent := 16930)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16927.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16911.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult16931

namespace SemanticResult16937
def owner : Owner := ⟨.program ⟨257⟩, ⟨9284⟩⟩
def rawTerms : List Term := Proof.Events066.exact16937RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 16937
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult16937.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 16934) (survivorTransfer := 16935)
    (survivorEvent := 16936) (resultEvent := resultEvent)
    (rightCoefficientProducer := 16884)
    (owner := owner) (leftOwner := SemanticResult16931.owner)
    (rightOwner := SemanticResult16885.owner)
    (leftResult := 16931) (rightResult := 16885)
    (leftBinding := 16932) (rightBinding := 16933)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9283⟩) (rightExpression := ⟨34⟩)
    (leftActual := SemanticResult16931.actual selector witness)
    (rightActual := SemanticResult16885.actual selector witness)
    (leftRaw := SemanticResult16931.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨34⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftAuthority16884.actual selector witness)
    (survivorMagnitude := LeftBound16935.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult16931.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult16885.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16884.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16884.derived selector witness)
  · exact LeftBound16935.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult16937

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
