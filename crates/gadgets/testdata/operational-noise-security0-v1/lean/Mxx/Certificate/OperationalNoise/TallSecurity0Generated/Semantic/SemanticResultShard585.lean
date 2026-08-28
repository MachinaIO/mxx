import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard585
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard032
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard077
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard566
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard584

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult82341
def owner : Owner := ⟨.program ⟨214⟩, ⟨7241⟩⟩
def rawTerms : List Term := Proof.Events321.exact82341RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82341
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82341.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82340.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge82340.frameStart)
    (transferEvent := 82339) (owner := owner)
    (leftResult := 79790) (rightResult := 8977)
    (working := LeftOperatorMerge82340.working)
    (reconstruction := LeftOperatorMerge82340.reconstruction)
    (leftReference := .predecessor 0 82337 .coefficient) (rightReference := .predecessor 1 82338 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8977.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge82340.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult82341

namespace SemanticResult82345
def owner : Owner := ⟨.program ⟨214⟩, ⟨12374⟩⟩
def rawTerms : List Term := Proof.Events321.exact82345RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82345
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82345.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 82342) (rightBinding := 82343)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7241⟩) (rightExpression := ⟨12373⟩)
    (transferEvent := 82344)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult82341.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult82336.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult82345

namespace SemanticResult82351
def owner : Owner := ⟨.program ⟨214⟩, ⟨12375⟩⟩
def rawTerms : List Term := Proof.Events321.exact82351RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 82351
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82351.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 82348) (survivorTransfer := 82349)
    (survivorEvent := 82350) (resultEvent := resultEvent)
    (rightCoefficientProducer := 8968)
    (owner := owner) (leftOwner := SemanticResult82345.owner)
    (rightOwner := SemanticResult8969.owner)
    (leftResult := 82345) (rightResult := 8969)
    (leftBinding := 82346) (rightBinding := 82347)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12374⟩) (rightExpression := ⟨99⟩)
    (leftActual := SemanticResult82345.actual selector witness)
    (rightActual := SemanticResult8969.actual selector witness)
    (leftRaw := SemanticResult82345.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨99⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound8968.actual selector witness)
    (survivorMagnitude := LeftBound82349.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult82345.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8969.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8968.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8968.derived selector witness)
  · exact LeftBound82349.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult82351

namespace SemanticResult82359
def owner : Owner := ⟨.program ⟨214⟩, ⟨12376⟩⟩
def rawTerms : List Term := Proof.Events321.exact82359RawTerms
def summary : Bound := (.finite 33280)
def resultEvent : Nat := 82359
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82359.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨40, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82357.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge82357.frameStart)
    (owner := owner) (leftOwner := SemanticResult82351.owner)
    (rightOwner := SemanticResult3946.owner)
    (leftResult := 82351) (rightResult := 3946)
    (leftActual := SemanticResult82351.actual selector witness)
    (rightActual := SemanticResult3946.actual selector witness)
    (leftRaw := SemanticResult82351.rawTerms)
    (rightRaw := SemanticResult3946.rawTerms)
    (working := LeftOperatorMerge82357.working)
    (leftBinding := 82352) (rightBinding := 82353)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12375⟩) (rightExpression := ⟨9820⟩)
    (coefficientTransfer := 82354) (summaryTransfer := 82356)
    (rightCoefficientProducer := 3945)
    (rightSummaryTransfer := 82355)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨40, by decide⟩)
    (rightRecordedMaximum := 40)
    (rightSummaryMaximum := ⟨40, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge82357.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority3945.actual selector witness)
    (summaryMagnitude := LeftBound82356.actual selector witness)
    (reconstruction := LeftOperatorMerge82357.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult82351.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3946.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3945.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority3945.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge82357.operationAgreement
  · exact LeftBound82356.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82357.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult82359

namespace SemanticResult82364
def owner : Owner := ⟨.program ⟨214⟩, ⟨9821⟩⟩
def rawTerms : List Term := Proof.Events321.exact82364RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82364
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82364.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82363.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge82363.frameStart)
    (transferEvent := 82362) (owner := owner)
    (leftResult := 3946) (rightResult := 79920)
    (working := LeftOperatorMerge82363.working)
    (reconstruction := LeftOperatorMerge82363.reconstruction)
    (leftReference := .predecessor 0 82360 .coefficient) (rightReference := .predecessor 1 82361 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3946.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79920.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge82363.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult82364

namespace SemanticResult82369
def owner : Owner := ⟨.program ⟨214⟩, ⟨7221⟩⟩
def rawTerms : List Term := Proof.Events321.exact82369RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82369
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82369.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82368.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge82368.frameStart)
    (transferEvent := 82367) (owner := owner)
    (leftResult := 79790) (rightResult := 9018)
    (working := LeftOperatorMerge82368.working)
    (reconstruction := LeftOperatorMerge82368.reconstruction)
    (leftReference := .predecessor 0 82365 .coefficient) (rightReference := .predecessor 1 82366 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9018.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge82368.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult82369

namespace SemanticResult82373
def owner : Owner := ⟨.program ⟨214⟩, ⟨9822⟩⟩
def rawTerms : List Term := Proof.Events321.exact82373RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82373
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82373.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 82370) (rightBinding := 82371)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7221⟩) (rightExpression := ⟨9821⟩)
    (transferEvent := 82372)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult82369.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult82364.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult82373

namespace SemanticResult82379
def owner : Owner := ⟨.program ⟨214⟩, ⟨9823⟩⟩
def rawTerms : List Term := Proof.Events321.exact82379RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 82379
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82379.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 82376) (survivorTransfer := 82377)
    (survivorEvent := 82378) (resultEvent := resultEvent)
    (rightCoefficientProducer := 9009)
    (owner := owner) (leftOwner := SemanticResult82373.owner)
    (rightOwner := SemanticResult9010.owner)
    (leftResult := 82373) (rightResult := 9010)
    (leftBinding := 82374) (rightBinding := 82375)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9822⟩) (rightExpression := ⟨79⟩)
    (leftActual := SemanticResult82373.actual selector witness)
    (rightActual := SemanticResult9010.actual selector witness)
    (leftRaw := SemanticResult82373.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨79⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound9009.actual selector witness)
    (survivorMagnitude := LeftBound82377.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult82373.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9010.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9009.derived selector witness)
  · exact LeftBound82377.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult82379

namespace SemanticResult82389
def owner : Owner := ⟨.program ⟨214⟩, ⟨9824⟩⟩
def rawTerms : List Term := Proof.Events321.exact82389RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 82389
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82389.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82385.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge82385.frameStart)
    (owner := owner) (leftOwner := SemanticResult82379.owner)
    (rightOwner := SemanticResult9007.owner)
    (leftResult := 82379) (rightResult := 9007)
    (leftActual := SemanticResult82379.actual selector witness)
    (rightActual := SemanticResult9007.actual selector witness)
    (leftRaw := SemanticResult82379.rawTerms)
    (rightRaw := SemanticResult9007.rawTerms)
    (working := LeftOperatorMerge82385.working)
    (leftBinding := 82380) (rightBinding := 82381)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9823⟩) (rightExpression := ⟨7868⟩)
    (coefficientTransfer := 82382) (summaryTransfer := 82384)
    (rightCoefficientProducer := 9006)
    (rightSummaryTransfer := 82383)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge82385.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound9006.actual selector witness)
    (summaryMagnitude := LeftBound82384.actual selector witness)
    (reconstruction := LeftOperatorMerge82385.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult82379.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9007.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9006.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound9006.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge82385.operationAgreement
  · exact LeftBound82384.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82385.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 82386 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge82385.working
    [{ coefficient := (-1), key := LeftRelationMerge82386.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge82386.frameStart
      LeftRelationMerge82386.owner (.relation 82386) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge82386.deltas
    rows := LeftRelationMerge82386.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge82385.working LeftRelationMerge82386.source
        (relationContext LeftRelationMerge82386.source
          LeftRelationMerge82386.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge82385.working, LeftRelationMerge82386.deltas,
    LeftRelationMerge82386.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 82386)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9824⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge82385.working) (working := relationWorking0)
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
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (relationClaim0 selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult82389

namespace SemanticResult82395
def owner : Owner := ⟨.program ⟨214⟩, ⟨12377⟩⟩
def rawTerms : List Term := Proof.Events321.exact82395RawTerms
def summary : Bound := (.finite 95453696)
def resultEvent : Nat := 82395
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82395.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge82393.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult82389.owner)
    (rightOwner := SemanticResult82359.owner)
    (leftResult := 82389) (rightResult := 82359)
    (leftActual := SemanticResult82389.actual selector witness)
    (rightActual := SemanticResult82359.actual selector witness)
    (leftRaw := SemanticResult82389.rawTerms)
    (rightRaw := SemanticResult82359.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 33280) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 82390) (rightBinding := 82391)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9824⟩) (rightExpression := ⟨12376⟩)
    (coefficientTransfer := 82392) (summaryTransfer := 82394)
    (base := LeftOperatorMerge82393.base)
    (reconstruction := LeftOperatorMerge82393.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult82389.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult82359.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge82393.operationAgreement
  · rfl
  · decide
end SemanticResult82395

namespace SemanticResult82405
def owner : Owner := ⟨.program ⟨214⟩, ⟨25374⟩⟩
def rawTerms : List Term := Proof.Events321.exact82405RawTerms
def summary : Bound := (.finite 350316591579136)
def resultEvent : Nat := 82405
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82405.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95453696, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82401.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge82401.frameStart)
    (owner := owner) (leftOwner := SemanticResult82395.owner)
    (rightOwner := SemanticResult82331.owner)
    (leftResult := 82395) (rightResult := 82331)
    (leftActual := SemanticResult82395.actual selector witness)
    (rightActual := SemanticResult82331.actual selector witness)
    (leftRaw := SemanticResult82395.rawTerms)
    (rightRaw := SemanticResult82331.rawTerms)
    (working := LeftOperatorMerge82401.working)
    (leftBinding := 82396) (rightBinding := 82397)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12377⟩) (rightExpression := ⟨25373⟩)
    (coefficientTransfer := 82398) (summaryTransfer := 82400)
    (rightCoefficientProducer := 82330)
    (rightSummaryTransfer := 82399)
    (leftMaximum := ⟨95453696, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge82401.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority82330.actual selector witness)
    (summaryMagnitude := LeftBound82400.actual selector witness)
    (reconstruction := LeftOperatorMerge82401.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult82395.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult82331.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82330.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority82330.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge82401.operationAgreement
  · exact LeftBound82400.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82401.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 82402 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25373⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩, ⟨.program ⟨214⟩, ⟨12370⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23206⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23206⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge82401.working
    [{ coefficient := (-1), key := LeftRelationMerge82402.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge82402.frameStart
      LeftRelationMerge82402.owner (.relation 82402) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge82402.deltas
    rows := LeftRelationMerge82402.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge82401.working LeftRelationMerge82402.source
        (relationContext LeftRelationMerge82402.source
          LeftRelationMerge82402.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge82401.working, LeftRelationMerge82402.deltas,
    LeftRelationMerge82402.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 82402)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25374⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9820⟩⟩, ⟨.program ⟨214⟩, ⟨12370⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25373⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25373⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge82401.working) (working := relationWorking0)
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
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (relationClaim0 selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult82405

namespace SemanticResult82408
def owner : Owner := ⟨.program ⟨214⟩, ⟨19888⟩⟩
def rawTerms : List Term := Proof.Events321.exact82408RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82408
def producerEvent : Nat := 82407
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82408.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨20⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨20⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult82408

namespace SemanticResult82412
def owner : Owner := ⟨.program ⟨214⟩, ⟨19890⟩⟩
def rawTerms : List Term := Proof.Events321.exact82412RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82412
def producerEvent : Nat := 82411
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82412.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 82409 .coefficient) (.value (.predecessor 1 82410 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 82409 .coefficient) (.value (.predecessor 1 82410 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult82412

namespace SemanticResult82490
def owner : Owner := ⟨.program ⟨214⟩, ⟨12370⟩⟩
def rawTerms : List Term := Proof.Events322.exact82490RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82490
def producerEvent : Nat := 82489
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82490.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 82467, .finite 40, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult82490

namespace SemanticResult82493
def owner : Owner := ⟨.program ⟨214⟩, ⟨9820⟩⟩
def rawTerms : List Term := Proof.Events322.exact82493RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82493
def producerEvent : Nat := 82492
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82493.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 82467, .finite 40, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult82493

namespace SemanticResult82498
def owner : Owner := ⟨.program ⟨214⟩, ⟨12371⟩⟩
def rawTerms : List Term := Proof.Events322.exact82498RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82498
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82498.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82497.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge82497.frameStart)
    (transferEvent := 82496) (owner := owner)
    (leftResult := 82493) (rightResult := 82490)
    (working := LeftOperatorMerge82497.working)
    (reconstruction := LeftOperatorMerge82497.reconstruction)
    (leftReference := .predecessor 0 82494 .coefficient) (rightReference := .predecessor 1 82495 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult82493.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult82490.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge82497.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult82498

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
