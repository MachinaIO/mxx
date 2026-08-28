import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard809
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard039
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard204
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard205
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard753
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard807
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard808

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult112374
def owner : Owner := ⟨.program ⟨257⟩, ⟨33926⟩⟩
def rawTerms : List Term := Proof.Events438.exact112374RawTerms
def summary : Bound := (.finite 32189200113375081643992404983808)
def resultEvent : Nat := 112374
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112374.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge112371.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult112367.owner)
    (rightOwner := SemanticResult112189.owner)
    (leftResult := 112367) (rightResult := 112189)
    (leftActual := SemanticResult112367.actual selector witness)
    (rightActual := SemanticResult112189.actual selector witness)
    (leftRaw := SemanticResult112367.rawTerms)
    (rightRaw := SemanticResult112189.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32189200113374879571150551121920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 112368) (rightBinding := 112369)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32719⟩) (rightExpression := ⟨33925⟩)
    (coefficientTransfer := 112370) (summaryTransfer := 112373)
    (base := LeftOperatorMerge112371.base)
    (reconstruction := LeftOperatorMerge112371.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult112367.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult112189.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge112371.operationAgreement
  · rfl
  · decide
end SemanticResult112374

namespace SemanticResult112381
def owner : Owner := ⟨.program ⟨257⟩, ⟨23090⟩⟩
def rawTerms : List Term := Proof.Events438.exact112381RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 112381
def producerEvent : Nat := 112380
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112381.actual selector witness
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
end SemanticResult112381

namespace SemanticResult112384
def owner : Owner := ⟨.program ⟨257⟩, ⟨23903⟩⟩
def rawTerms : List Term := Proof.Events439.exact112384RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 112384
def producerEvent : Nat := 112383
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112384.actual selector witness
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
end SemanticResult112384

namespace SemanticResult112391
def owner : Owner := ⟨.program ⟨257⟩, ⟨22935⟩⟩
def rawTerms : List Term := Proof.Events439.exact112391RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 112391
def producerEvent : Nat := 112390
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112391.actual selector witness
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
end SemanticResult112391

namespace SemanticResult112394
def owner : Owner := ⟨.program ⟨257⟩, ⟨23450⟩⟩
def rawTerms : List Term := Proof.Events439.exact112394RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 112394
def producerEvent : Nat := 112393
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112394.actual selector witness
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
end SemanticResult112394

namespace SemanticResult112399
def owner : Owner := ⟨.program ⟨257⟩, ⟨21521⟩⟩
def rawTerms : List Term := Proof.Events439.exact112399RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 112399
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112399.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge112398.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge112398.frameStart)
    (transferEvent := 112397) (owner := owner)
    (leftResult := 4927) (rightResult := 105153)
    (working := LeftOperatorMerge112398.working)
    (reconstruction := LeftOperatorMerge112398.reconstruction)
    (leftReference := .predecessor 0 112395 .coefficient) (rightReference := .predecessor 1 112396 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4927.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult105153.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge112398.operationAgreement
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
end SemanticResult112399

namespace SemanticResult112404
def owner : Owner := ⟨.program ⟨257⟩, ⟨8726⟩⟩
def rawTerms : List Term := Proof.Events439.exact112404RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 112404
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112404.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge112403.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge112403.frameStart)
    (transferEvent := 112402) (owner := owner)
    (leftResult := 105023) (rightResult := 24595)
    (working := LeftOperatorMerge112403.working)
    (reconstruction := LeftOperatorMerge112403.reconstruction)
    (leftReference := .predecessor 0 112400 .coefficient) (rightReference := .predecessor 1 112401 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult105023.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult24595.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge112403.operationAgreement
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
end SemanticResult112404

namespace SemanticResult112408
def owner : Owner := ⟨.program ⟨257⟩, ⟨21522⟩⟩
def rawTerms : List Term := Proof.Events439.exact112408RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 112408
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112408.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 112405) (rightBinding := 112406)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨8726⟩) (rightExpression := ⟨21521⟩)
    (transferEvent := 112407)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult112404.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult112399.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult112408

namespace SemanticResult112414
def owner : Owner := ⟨.program ⟨257⟩, ⟨21523⟩⟩
def rawTerms : List Term := Proof.Events439.exact112414RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 112414
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112414.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 112411) (survivorTransfer := 112412)
    (survivorEvent := 112413) (resultEvent := resultEvent)
    (rightCoefficientProducer := 24586)
    (owner := owner) (leftOwner := SemanticResult112408.owner)
    (rightOwner := SemanticResult24587.owner)
    (leftResult := 112408) (rightResult := 24587)
    (leftBinding := 112409) (rightBinding := 112410)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21522⟩) (rightExpression := ⟨132⟩)
    (leftActual := SemanticResult112408.actual selector witness)
    (rightActual := SemanticResult24587.actual selector witness)
    (leftRaw := SemanticResult112408.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨132⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound24586.actual selector witness)
    (survivorMagnitude := LeftBound112412.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult112408.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult24587.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24586.derived selector witness)
  · exact LeftBound112412.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult112414

namespace SemanticResult112422
def owner : Owner := ⟨.program ⟨257⟩, ⟨21524⟩⟩
def rawTerms : List Term := Proof.Events439.exact112422RawTerms
def summary : Bound := (.finite 3407872)
def resultEvent : Nat := 112422
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112422.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32768
      (.finite ⟨26, by decide⟩)
      (.finite ⟨4, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge112420.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge112420.frameStart)
    (owner := owner) (leftOwner := SemanticResult112414.owner)
    (rightOwner := SemanticResult4930.owner)
    (leftResult := 112414) (rightResult := 4930)
    (leftActual := SemanticResult112414.actual selector witness)
    (rightActual := SemanticResult4930.actual selector witness)
    (leftRaw := SemanticResult112414.rawTerms)
    (rightRaw := SemanticResult4930.rawTerms)
    (working := LeftOperatorMerge112420.working)
    (leftBinding := 112415) (rightBinding := 112416)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21523⟩) (rightExpression := ⟨21116⟩)
    (coefficientTransfer := 112417) (summaryTransfer := 112419)
    (rightCoefficientProducer := 4929)
    (rightSummaryTransfer := 112418)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨4, by decide⟩)
    (rightRecordedMaximum := 4)
    (rightSummaryMaximum := ⟨4, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32768)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge112420.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority4929.actual selector witness)
    (summaryMagnitude := LeftBound112419.actual selector witness)
    (reconstruction := LeftOperatorMerge112420.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult112414.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4930.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4929.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority4929.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge112420.operationAgreement
  · exact LeftBound112419.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge112420.working summary) := by
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
end SemanticResult112422

namespace SemanticResult112427
def owner : Owner := ⟨.program ⟨257⟩, ⟨21117⟩⟩
def rawTerms : List Term := Proof.Events439.exact112427RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 112427
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112427.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge112426.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge112426.frameStart)
    (transferEvent := 112425) (owner := owner)
    (leftResult := 4930) (rightResult := 105153)
    (working := LeftOperatorMerge112426.working)
    (reconstruction := LeftOperatorMerge112426.reconstruction)
    (leftReference := .predecessor 0 112423 .coefficient) (rightReference := .predecessor 1 112424 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4930.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult105153.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge112426.operationAgreement
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
end SemanticResult112427

namespace SemanticResult112432
def owner : Owner := ⟨.program ⟨257⟩, ⟨8706⟩⟩
def rawTerms : List Term := Proof.Events439.exact112432RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 112432
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112432.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge112431.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge112431.frameStart)
    (transferEvent := 112430) (owner := owner)
    (leftResult := 105023) (rightResult := 24636)
    (working := LeftOperatorMerge112431.working)
    (reconstruction := LeftOperatorMerge112431.reconstruction)
    (leftReference := .predecessor 0 112428 .coefficient) (rightReference := .predecessor 1 112429 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult105023.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult24636.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge112431.operationAgreement
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
end SemanticResult112432

namespace SemanticResult112436
def owner : Owner := ⟨.program ⟨257⟩, ⟨21118⟩⟩
def rawTerms : List Term := Proof.Events439.exact112436RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 112436
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112436.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 112433) (rightBinding := 112434)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨8706⟩) (rightExpression := ⟨21117⟩)
    (transferEvent := 112435)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult112432.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult112427.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult112436

namespace SemanticResult112442
def owner : Owner := ⟨.program ⟨257⟩, ⟨21119⟩⟩
def rawTerms : List Term := Proof.Events439.exact112442RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 112442
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112442.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 112439) (survivorTransfer := 112440)
    (survivorEvent := 112441) (resultEvent := resultEvent)
    (rightCoefficientProducer := 24627)
    (owner := owner) (leftOwner := SemanticResult112436.owner)
    (rightOwner := SemanticResult24628.owner)
    (leftResult := 112436) (rightResult := 24628)
    (leftBinding := 112437) (rightBinding := 112438)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21118⟩) (rightExpression := ⟨112⟩)
    (leftActual := SemanticResult112436.actual selector witness)
    (rightActual := SemanticResult24628.actual selector witness)
    (leftRaw := SemanticResult112436.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨112⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound24627.actual selector witness)
    (survivorMagnitude := LeftBound112440.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult112436.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult24628.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24627.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24627.derived selector witness)
  · exact LeftBound112440.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult112442

namespace SemanticResult112452
def owner : Owner := ⟨.program ⟨257⟩, ⟨21120⟩⟩
def rawTerms : List Term := Proof.Events439.exact112452RawTerms
def summary : Bound := (.finite 279172874240)
def resultEvent : Nat := 112452
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112452.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1310720
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge112448.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge112448.frameStart)
    (owner := owner) (leftOwner := SemanticResult112442.owner)
    (rightOwner := SemanticResult24625.owner)
    (leftResult := 112442) (rightResult := 24625)
    (leftActual := SemanticResult112442.actual selector witness)
    (rightActual := SemanticResult24625.actual selector witness)
    (leftRaw := SemanticResult112442.rawTerms)
    (rightRaw := SemanticResult24625.rawTerms)
    (working := LeftOperatorMerge112448.working)
    (leftBinding := 112443) (rightBinding := 112444)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21119⟩) (rightExpression := ⟨9575⟩)
    (coefficientTransfer := 112445) (summaryTransfer := 112447)
    (rightCoefficientProducer := 24624)
    (rightSummaryTransfer := 112446)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1310720)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge112448.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound24624.actual selector witness)
    (summaryMagnitude := LeftBound112447.actual selector witness)
    (reconstruction := LeftOperatorMerge112448.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult112442.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult24625.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24624.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound24624.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge112448.operationAgreement
  · exact LeftBound112447.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge112448.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 112449 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨21116⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge112448.working
    [{ coefficient := (-1), key := LeftRelationMerge112449.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge112449.frameStart
      LeftRelationMerge112449.owner (.relation 112449) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge112449.deltas
    rows := LeftRelationMerge112449.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge112448.working LeftRelationMerge112449.source
        (relationContext LeftRelationMerge112449.source
          LeftRelationMerge112449.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge112448.working, LeftRelationMerge112449.deltas,
    LeftRelationMerge112449.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 112449)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨21120⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨21116⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge112448.working) (working := relationWorking0)
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
end SemanticResult112452

namespace SemanticResult112458
def owner : Owner := ⟨.program ⟨257⟩, ⟨21525⟩⟩
def rawTerms : List Term := Proof.Events439.exact112458RawTerms
def summary : Bound := (.finite 279176282112)
def resultEvent : Nat := 112458
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult112458.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge112456.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult112452.owner)
    (rightOwner := SemanticResult112422.owner)
    (leftResult := 112452) (rightResult := 112422)
    (leftActual := SemanticResult112452.actual selector witness)
    (rightActual := SemanticResult112422.actual selector witness)
    (leftRaw := SemanticResult112452.rawTerms)
    (rightRaw := SemanticResult112422.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 279172874240)
    (rightMaximum := 3407872) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 112453) (rightBinding := 112454)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21120⟩) (rightExpression := ⟨21524⟩)
    (coefficientTransfer := 112455) (summaryTransfer := 112457)
    (base := LeftOperatorMerge112456.base)
    (reconstruction := LeftOperatorMerge112456.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult112452.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult112422.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge112456.operationAgreement
  · rfl
  · decide
end SemanticResult112458

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
