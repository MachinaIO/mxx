import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1882
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard103
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard168
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1859
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1860
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1880
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1881

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult268911
def owner : Owner := ⟨.program ⟨257⟩, ⟨36425⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268911RawTerms
def summary : Bound := (.finite 32192539770951767057087530795008)
def resultEvent : Nat := 268911
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268911.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge268908.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult268904.owner)
    (rightOwner := SemanticResult268726.owner)
    (leftResult := 268904) (rightResult := 268726)
    (leftActual := SemanticResult268904.actual selector witness)
    (rightActual := SemanticResult268726.actual selector witness)
    (leftRaw := SemanticResult268904.rawTerms)
    (rightRaw := SemanticResult268726.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32192539770951564984245676933120) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 268905) (rightBinding := 268906)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨35333⟩) (rightExpression := ⟨36424⟩)
    (coefficientTransfer := 268907) (summaryTransfer := 268910)
    (base := LeftOperatorMerge268908.base)
    (reconstruction := LeftOperatorMerge268908.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult268904.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult268726.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge268908.operationAgreement
  · rfl
  · decide
end SemanticResult268911

namespace SemanticResult268918
def owner : Owner := ⟨.program ⟨257⟩, ⟨30166⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268918RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 268918
def producerEvent : Nat := 268917
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268918.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult268918

namespace SemanticResult268921
def owner : Owner := ⟨.program ⟨257⟩, ⟨30762⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268921RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 268921
def producerEvent : Nat := 268920
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268921.actual selector witness
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
end SemanticResult268921

namespace SemanticResult268928
def owner : Owner := ⟨.program ⟨257⟩, ⟨30039⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268928RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 268928
def producerEvent : Nat := 268927
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268928.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult268928

namespace SemanticResult268931
def owner : Owner := ⟨.program ⟨257⟩, ⟨30508⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268931RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 268931
def producerEvent : Nat := 268930
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268931.actual selector witness
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
end SemanticResult268931

namespace SemanticResult268936
def owner : Owner := ⟨.program ⟨257⟩, ⟨28577⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268936RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 268936
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268936.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge268935.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge268935.frameStart)
    (transferEvent := 268934) (owner := owner)
    (leftResult := 12948) (rightResult := 266028)
    (working := LeftOperatorMerge268935.working)
    (reconstruction := LeftOperatorMerge268935.reconstruction)
    (leftReference := .predecessor 0 268932 .coefficient) (rightReference := .predecessor 1 268933 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult12948.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult266028.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge268935.operationAgreement
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
end SemanticResult268936

namespace SemanticResult268941
def owner : Owner := ⟨.program ⟨257⟩, ⟨7635⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268941RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 268941
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268941.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge268940.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge268940.frameStart)
    (transferEvent := 268939) (owner := owner)
    (leftResult := 265898) (rightResult := 20086)
    (working := LeftOperatorMerge268940.working)
    (reconstruction := LeftOperatorMerge268940.reconstruction)
    (leftReference := .predecessor 0 268937 .coefficient) (rightReference := .predecessor 1 268938 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult265898.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20086.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge268940.operationAgreement
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
end SemanticResult268941

namespace SemanticResult268945
def owner : Owner := ⟨.program ⟨257⟩, ⟨28578⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268945RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 268945
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268945.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 268942) (rightBinding := 268943)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7635⟩) (rightExpression := ⟨28577⟩)
    (transferEvent := 268944)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult268941.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult268936.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult268945

namespace SemanticResult268951
def owner : Owner := ⟨.program ⟨257⟩, ⟨28579⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268951RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 268951
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268951.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 268948) (survivorTransfer := 268949)
    (survivorEvent := 268950) (resultEvent := resultEvent)
    (rightCoefficientProducer := 20077)
    (owner := owner) (leftOwner := SemanticResult268945.owner)
    (rightOwner := SemanticResult20078.owner)
    (leftResult := 268945) (rightResult := 20078)
    (leftBinding := 268946) (rightBinding := 268947)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28578⟩) (rightExpression := ⟨105⟩)
    (leftActual := SemanticResult268945.actual selector witness)
    (rightActual := SemanticResult20078.actual selector witness)
    (leftRaw := SemanticResult268945.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨105⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound20077.actual selector witness)
    (survivorMagnitude := LeftBound268949.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult268945.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20078.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20077.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20077.derived selector witness)
  · exact LeftBound268949.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult268951

namespace SemanticResult268959
def owner : Owner := ⟨.program ⟨257⟩, ⟨28580⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268959RawTerms
def summary : Bound := (.finite 30670848)
def resultEvent : Nat := 268959
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268959.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32768
      (.finite ⟨26, by decide⟩)
      (.finite ⟨36, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge268957.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge268957.frameStart)
    (owner := owner) (leftOwner := SemanticResult268951.owner)
    (rightOwner := SemanticResult12951.owner)
    (leftResult := 268951) (rightResult := 12951)
    (leftActual := SemanticResult268951.actual selector witness)
    (rightActual := SemanticResult12951.actual selector witness)
    (leftRaw := SemanticResult268951.rawTerms)
    (rightRaw := SemanticResult12951.rawTerms)
    (working := LeftOperatorMerge268957.working)
    (leftBinding := 268952) (rightBinding := 268953)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28579⟩) (rightExpression := ⟨13156⟩)
    (coefficientTransfer := 268954) (summaryTransfer := 268956)
    (rightCoefficientProducer := 12950)
    (rightSummaryTransfer := 268955)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨36, by decide⟩)
    (rightRecordedMaximum := 36)
    (rightSummaryMaximum := ⟨36, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32768)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge268957.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority12950.actual selector witness)
    (summaryMagnitude := LeftBound268956.actual selector witness)
    (reconstruction := LeftOperatorMerge268957.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult268951.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12951.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12950.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority12950.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge268957.operationAgreement
  · exact LeftBound268956.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge268957.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult268959

namespace SemanticResult268964
def owner : Owner := ⟨.program ⟨257⟩, ⟨13157⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268964RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 268964
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268964.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge268963.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge268963.frameStart)
    (transferEvent := 268962) (owner := owner)
    (leftResult := 12951) (rightResult := 266028)
    (working := LeftOperatorMerge268963.working)
    (reconstruction := LeftOperatorMerge268963.reconstruction)
    (leftReference := .predecessor 0 268960 .coefficient) (rightReference := .predecessor 1 268961 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult12951.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult266028.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge268963.operationAgreement
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
end SemanticResult268964

namespace SemanticResult268969
def owner : Owner := ⟨.program ⟨257⟩, ⟨7652⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268969RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 268969
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268969.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge268968.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge268968.frameStart)
    (transferEvent := 268967) (owner := owner)
    (leftResult := 265898) (rightResult := 20127)
    (working := LeftOperatorMerge268968.working)
    (reconstruction := LeftOperatorMerge268968.reconstruction)
    (leftReference := .predecessor 0 268965 .coefficient) (rightReference := .predecessor 1 268966 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult265898.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20127.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge268968.operationAgreement
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
end SemanticResult268969

namespace SemanticResult268973
def owner : Owner := ⟨.program ⟨257⟩, ⟨13158⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268973RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 268973
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268973.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 268970) (rightBinding := 268971)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7652⟩) (rightExpression := ⟨13157⟩)
    (transferEvent := 268972)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult268969.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult268964.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult268973

namespace SemanticResult268979
def owner : Owner := ⟨.program ⟨257⟩, ⟨13159⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268979RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 268979
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268979.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 268976) (survivorTransfer := 268977)
    (survivorEvent := 268978) (resultEvent := resultEvent)
    (rightCoefficientProducer := 20118)
    (owner := owner) (leftOwner := SemanticResult268973.owner)
    (rightOwner := SemanticResult20119.owner)
    (leftResult := 268973) (rightResult := 20119)
    (leftBinding := 268974) (rightBinding := 268975)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13158⟩) (rightExpression := ⟨122⟩)
    (leftActual := SemanticResult268973.actual selector witness)
    (rightActual := SemanticResult20119.actual selector witness)
    (leftRaw := SemanticResult268973.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨122⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound20118.actual selector witness)
    (survivorMagnitude := LeftBound268977.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult268973.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20119.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20118.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20118.derived selector witness)
  · exact LeftBound268977.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult268979

namespace SemanticResult268989
def owner : Owner := ⟨.program ⟨257⟩, ⟨13160⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268989RawTerms
def summary : Bound := (.finite 279172874240)
def resultEvent : Nat := 268989
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268989.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1310720
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge268985.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge268985.frameStart)
    (owner := owner) (leftOwner := SemanticResult268979.owner)
    (rightOwner := SemanticResult20116.owner)
    (leftResult := 268979) (rightResult := 20116)
    (leftActual := SemanticResult268979.actual selector witness)
    (rightActual := SemanticResult20116.actual selector witness)
    (leftRaw := SemanticResult268979.rawTerms)
    (rightRaw := SemanticResult20116.rawTerms)
    (working := LeftOperatorMerge268985.working)
    (leftBinding := 268980) (rightBinding := 268981)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13159⟩) (rightExpression := ⟨9548⟩)
    (coefficientTransfer := 268982) (summaryTransfer := 268984)
    (rightCoefficientProducer := 20115)
    (rightSummaryTransfer := 268983)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1310720)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge268985.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound20115.actual selector witness)
    (summaryMagnitude := LeftBound268984.actual selector witness)
    (reconstruction := LeftOperatorMerge268985.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult268979.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20116.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20115.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound20115.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge268985.operationAgreement
  · exact LeftBound268984.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge268985.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 268986 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge268985.working
    [{ coefficient := (-1), key := LeftRelationMerge268986.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge268986.frameStart
      LeftRelationMerge268986.owner (.relation 268986) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge268986.deltas
    rows := LeftRelationMerge268986.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge268985.working LeftRelationMerge268986.source
        (relationContext LeftRelationMerge268986.source
          LeftRelationMerge268986.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge268985.working, LeftRelationMerge268986.deltas,
    LeftRelationMerge268986.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 268986)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨13160⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨13156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge268985.working) (working := relationWorking0)
    (reconstruction := relationReconstruction0)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt0 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact mergeClaim selector selectorLower selectorUpper witness
  · exact relationAgreement0
  · decide +kernel
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (relationClaim0 selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult268989

namespace SemanticResult268995
def owner : Owner := ⟨.program ⟨257⟩, ⟨28581⟩⟩
def rawTerms : List Term := Proof.Events1050.exact268995RawTerms
def summary : Bound := (.finite 279203545088)
def resultEvent : Nat := 268995
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult268995.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge268993.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult268989.owner)
    (rightOwner := SemanticResult268959.owner)
    (leftResult := 268989) (rightResult := 268959)
    (leftActual := SemanticResult268989.actual selector witness)
    (rightActual := SemanticResult268959.actual selector witness)
    (leftRaw := SemanticResult268989.rawTerms)
    (rightRaw := SemanticResult268959.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 279172874240)
    (rightMaximum := 30670848) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 268990) (rightBinding := 268991)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13160⟩) (rightExpression := ⟨28580⟩)
    (coefficientTransfer := 268992) (summaryTransfer := 268994)
    (base := LeftOperatorMerge268993.base)
    (reconstruction := LeftOperatorMerge268993.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult268989.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult268959.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge268993.operationAgreement
  · rfl
  · decide
end SemanticResult268995

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
