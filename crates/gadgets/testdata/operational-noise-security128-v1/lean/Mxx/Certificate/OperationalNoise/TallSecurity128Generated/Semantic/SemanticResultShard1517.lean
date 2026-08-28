import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1517
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard080
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard208
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard209
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1457
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1516

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult215261
def owner : Owner := ⟨.program ⟨257⟩, ⟨8611⟩⟩
def rawTerms : List Term := Proof.Events840.exact215261RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 215261
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215261.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge215260.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge215260.frameStart)
    (transferEvent := 215259) (owner := owner)
    (leftResult := 207398) (rightResult := 25096)
    (working := LeftOperatorMerge215260.working)
    (reconstruction := LeftOperatorMerge215260.reconstruction)
    (leftReference := .predecessor 0 215257 .coefficient) (rightReference := .predecessor 1 215258 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult207398.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult25096.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge215260.operationAgreement
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
end SemanticResult215261

namespace SemanticResult215265
def owner : Owner := ⟨.program ⟨257⟩, ⟨18278⟩⟩
def rawTerms : List Term := Proof.Events840.exact215265RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 215265
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215265.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 215262) (rightBinding := 215263)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨8611⟩) (rightExpression := ⟨18277⟩)
    (transferEvent := 215264)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult215261.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult215256.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult215265

namespace SemanticResult215271
def owner : Owner := ⟨.program ⟨257⟩, ⟨18279⟩⟩
def rawTerms : List Term := Proof.Events840.exact215271RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 215271
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215271.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 215268) (survivorTransfer := 215269)
    (survivorEvent := 215270) (resultEvent := resultEvent)
    (rightCoefficientProducer := 25087)
    (owner := owner) (leftOwner := SemanticResult215265.owner)
    (rightOwner := SemanticResult25088.owner)
    (leftResult := 215265) (rightResult := 25088)
    (leftBinding := 215266) (rightBinding := 215267)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18278⟩) (rightExpression := ⟨131⟩)
    (leftActual := SemanticResult215265.actual selector witness)
    (rightActual := SemanticResult25088.actual selector witness)
    (leftRaw := SemanticResult215265.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨131⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound25087.actual selector witness)
    (survivorMagnitude := LeftBound215269.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult215265.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult25088.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25087.derived selector witness)
  · exact LeftBound215269.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult215271

namespace SemanticResult215279
def owner : Owner := ⟨.program ⟨257⟩, ⟨18280⟩⟩
def rawTerms : List Term := Proof.Events840.exact215279RawTerms
def summary : Bound := (.finite 2555904)
def resultEvent : Nat := 215279
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215279.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32768
      (.finite ⟨26, by decide⟩)
      (.finite ⟨3, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge215277.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge215277.frameStart)
    (owner := owner) (leftOwner := SemanticResult215271.owner)
    (rightOwner := SemanticResult10189.owner)
    (leftResult := 215271) (rightResult := 10189)
    (leftActual := SemanticResult215271.actual selector witness)
    (rightActual := SemanticResult10189.actual selector witness)
    (leftRaw := SemanticResult215271.rawTerms)
    (rightRaw := SemanticResult10189.rawTerms)
    (working := LeftOperatorMerge215277.working)
    (leftBinding := 215272) (rightBinding := 215273)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18279⟩) (rightExpression := ⟨12681⟩)
    (coefficientTransfer := 215274) (summaryTransfer := 215276)
    (rightCoefficientProducer := 10188)
    (rightSummaryTransfer := 215275)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨3, by decide⟩)
    (rightRecordedMaximum := 3)
    (rightSummaryMaximum := ⟨3, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32768)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge215277.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority10188.actual selector witness)
    (summaryMagnitude := LeftBound215276.actual selector witness)
    (reconstruction := LeftOperatorMerge215277.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult215271.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10189.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10188.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority10188.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge215277.operationAgreement
  · exact LeftBound215276.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge215277.working summary) := by
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
end SemanticResult215279

namespace SemanticResult215284
def owner : Owner := ⟨.program ⟨257⟩, ⟨12682⟩⟩
def rawTerms : List Term := Proof.Events840.exact215284RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 215284
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215284.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge215283.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge215283.frameStart)
    (transferEvent := 215282) (owner := owner)
    (leftResult := 10189) (rightResult := 207528)
    (working := LeftOperatorMerge215283.working)
    (reconstruction := LeftOperatorMerge215283.reconstruction)
    (leftReference := .predecessor 0 215280 .coefficient) (rightReference := .predecessor 1 215281 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult10189.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult207528.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge215283.operationAgreement
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
end SemanticResult215284

namespace SemanticResult215289
def owner : Owner := ⟨.program ⟨257⟩, ⟨8583⟩⟩
def rawTerms : List Term := Proof.Events840.exact215289RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 215289
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215289.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge215288.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge215288.frameStart)
    (transferEvent := 215287) (owner := owner)
    (leftResult := 207398) (rightResult := 25137)
    (working := LeftOperatorMerge215288.working)
    (reconstruction := LeftOperatorMerge215288.reconstruction)
    (leftReference := .predecessor 0 215285 .coefficient) (rightReference := .predecessor 1 215286 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult207398.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult25137.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge215288.operationAgreement
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
end SemanticResult215289

namespace SemanticResult215293
def owner : Owner := ⟨.program ⟨257⟩, ⟨12683⟩⟩
def rawTerms : List Term := Proof.Events840.exact215293RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 215293
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215293.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 215290) (rightBinding := 215291)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨8583⟩) (rightExpression := ⟨12682⟩)
    (transferEvent := 215292)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult215289.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult215284.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult215293

namespace SemanticResult215299
def owner : Owner := ⟨.program ⟨257⟩, ⟨12684⟩⟩
def rawTerms : List Term := Proof.Events841.exact215299RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 215299
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215299.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 215296) (survivorTransfer := 215297)
    (survivorEvent := 215298) (resultEvent := resultEvent)
    (rightCoefficientProducer := 25128)
    (owner := owner) (leftOwner := SemanticResult215293.owner)
    (rightOwner := SemanticResult25129.owner)
    (leftResult := 215293) (rightResult := 25129)
    (leftBinding := 215294) (rightBinding := 215295)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12683⟩) (rightExpression := ⟨103⟩)
    (leftActual := SemanticResult215293.actual selector witness)
    (rightActual := SemanticResult25129.actual selector witness)
    (leftRaw := SemanticResult215293.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨103⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound25128.actual selector witness)
    (survivorMagnitude := LeftBound215297.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult215293.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult25129.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25128.derived selector witness)
  · exact LeftBound215297.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult215299

namespace SemanticResult215309
def owner : Owner := ⟨.program ⟨257⟩, ⟨12685⟩⟩
def rawTerms : List Term := Proof.Events841.exact215309RawTerms
def summary : Bound := (.finite 279172874240)
def resultEvent : Nat := 215309
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215309.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1310720
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge215305.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge215305.frameStart)
    (owner := owner) (leftOwner := SemanticResult215299.owner)
    (rightOwner := SemanticResult25126.owner)
    (leftResult := 215299) (rightResult := 25126)
    (leftActual := SemanticResult215299.actual selector witness)
    (rightActual := SemanticResult25126.actual selector witness)
    (leftRaw := SemanticResult215299.rawTerms)
    (rightRaw := SemanticResult25126.rawTerms)
    (working := LeftOperatorMerge215305.working)
    (leftBinding := 215300) (rightBinding := 215301)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12684⟩) (rightExpression := ⟨9572⟩)
    (coefficientTransfer := 215302) (summaryTransfer := 215304)
    (rightCoefficientProducer := 25125)
    (rightSummaryTransfer := 215303)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1310720)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge215305.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound25125.actual selector witness)
    (summaryMagnitude := LeftBound215304.actual selector witness)
    (reconstruction := LeftOperatorMerge215305.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult215299.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult25126.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25125.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound25125.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge215305.operationAgreement
  · exact LeftBound215304.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge215305.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 215306 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge215305.working
    [{ coefficient := (-1), key := LeftRelationMerge215306.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge215306.frameStart
      LeftRelationMerge215306.owner (.relation 215306) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge215306.deltas
    rows := LeftRelationMerge215306.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge215305.working LeftRelationMerge215306.source
        (relationContext LeftRelationMerge215306.source
          LeftRelationMerge215306.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge215305.working, LeftRelationMerge215306.deltas,
    LeftRelationMerge215306.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 215306)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨12685⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge215305.working) (working := relationWorking0)
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
end SemanticResult215309

namespace SemanticResult215315
def owner : Owner := ⟨.program ⟨257⟩, ⟨18281⟩⟩
def rawTerms : List Term := Proof.Events841.exact215315RawTerms
def summary : Bound := (.finite 279175430144)
def resultEvent : Nat := 215315
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215315.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge215313.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult215309.owner)
    (rightOwner := SemanticResult215279.owner)
    (leftResult := 215309) (rightResult := 215279)
    (leftActual := SemanticResult215309.actual selector witness)
    (rightActual := SemanticResult215279.actual selector witness)
    (leftRaw := SemanticResult215309.rawTerms)
    (rightRaw := SemanticResult215279.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 279172874240)
    (rightMaximum := 2555904) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 215310) (rightBinding := 215311)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12685⟩) (rightExpression := ⟨18280⟩)
    (coefficientTransfer := 215312) (summaryTransfer := 215314)
    (base := LeftOperatorMerge215313.base)
    (reconstruction := LeftOperatorMerge215313.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult215309.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult215279.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge215313.operationAgreement
  · rfl
  · decide
end SemanticResult215315

namespace SemanticResult215325
def owner : Owner := ⟨.program ⟨257⟩, ⟨20220⟩⟩
def rawTerms : List Term := Proof.Events841.exact215325RawTerms
def summary : Bound := (.finite 2997623355788031426560)
def resultEvent : Nat := 215325
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215325.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1310720
      (.finite ⟨279175430144, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge215321.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge215321.frameStart)
    (owner := owner) (leftOwner := SemanticResult215315.owner)
    (rightOwner := SemanticResult215251.owner)
    (leftResult := 215315) (rightResult := 215251)
    (leftActual := SemanticResult215315.actual selector witness)
    (rightActual := SemanticResult215251.actual selector witness)
    (leftRaw := SemanticResult215315.rawTerms)
    (rightRaw := SemanticResult215251.rawTerms)
    (working := LeftOperatorMerge215321.working)
    (leftBinding := 215316) (rightBinding := 215317)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18281⟩) (rightExpression := ⟨20219⟩)
    (coefficientTransfer := 215318) (summaryTransfer := 215320)
    (rightCoefficientProducer := 215250)
    (rightSummaryTransfer := 215319)
    (leftMaximum := ⟨279175430144, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1310720)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge215321.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority215250.actual selector witness)
    (summaryMagnitude := LeftBound215320.actual selector witness)
    (reconstruction := LeftOperatorMerge215321.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult215315.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult215251.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority215250.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority215250.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge215321.operationAgreement
  · exact LeftBound215320.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge215321.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 215322 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19709⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19709⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge215321.working
    [{ coefficient := (-1), key := LeftRelationMerge215322.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge215322.frameStart
      LeftRelationMerge215322.owner (.relation 215322) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge215322.deltas
    rows := LeftRelationMerge215322.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge215321.working LeftRelationMerge215322.source
        (relationContext LeftRelationMerge215322.source
          LeftRelationMerge215322.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge215321.working, LeftRelationMerge215322.deltas,
    LeftRelationMerge215322.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 215322)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨20220⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨12681⟩⟩, ⟨.program ⟨257⟩, ⟨18274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20219⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge215321.working) (working := relationWorking0)
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
end SemanticResult215325

namespace SemanticResult215328
def owner : Owner := ⟨.program ⟨257⟩, ⟨19149⟩⟩
def rawTerms : List Term := Proof.Events841.exact215328RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 215328
def producerEvent : Nat := 215327
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215328.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨37⟩), 0, .finite 5647228698, .authorityRelationPreimageSource ⟨37⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult215328

namespace SemanticResult215332
def owner : Owner := ⟨.program ⟨257⟩, ⟨19151⟩⟩
def rawTerms : List Term := Proof.Events841.exact215332RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 215332
def producerEvent : Nat := 215331
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215332.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 215329 .coefficient) (.value (.predecessor 1 215330 .coefficient)), 0, .finite 5647228698, .scale (.predecessor 0 215329 .coefficient) (.value (.predecessor 1 215330 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult215332

namespace SemanticResult215410
def owner : Owner := ⟨.program ⟨257⟩, ⟨18274⟩⟩
def rawTerms : List Term := Proof.Events841.exact215410RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 215410
def producerEvent : Nat := 215409
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215410.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 215387, .finite 3, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult215410

namespace SemanticResult215413
def owner : Owner := ⟨.program ⟨257⟩, ⟨12681⟩⟩
def rawTerms : List Term := Proof.Events841.exact215413RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 215413
def producerEvent : Nat := 215412
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215413.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 215387, .finite 3, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult215413

namespace SemanticResult215418
def owner : Owner := ⟨.program ⟨257⟩, ⟨18275⟩⟩
def rawTerms : List Term := Proof.Events841.exact215418RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 215418
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult215418.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge215417.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge215417.frameStart)
    (transferEvent := 215416) (owner := owner)
    (leftResult := 215413) (rightResult := 215410)
    (working := LeftOperatorMerge215417.working)
    (reconstruction := LeftOperatorMerge215417.reconstruction)
    (leftReference := .predecessor 0 215414 .coefficient) (rightReference := .predecessor 1 215415 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult215413.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult215410.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge215417.operationAgreement
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
end SemanticResult215418

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
