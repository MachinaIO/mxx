import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard406
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard101
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard102
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard405

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult55993
def owner : Owner := ⟨.program ⟨214⟩, ⟨7272⟩⟩
def rawTerms : List Term := Proof.Events218.exact55993RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55993
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55993.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55992.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge55992.frameStart)
    (transferEvent := 55991) (owner := owner)
    (leftResult := 50540) (rightResult := 11983)
    (working := LeftOperatorMerge55992.working)
    (reconstruction := LeftOperatorMerge55992.reconstruction)
    (leftReference := .predecessor 0 55989 .coefficient) (rightReference := .predecessor 1 55990 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11983.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge55992.operationAgreement
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
end SemanticResult55993

namespace SemanticResult55997
def owner : Owner := ⟨.program ⟨214⟩, ⟨11391⟩⟩
def rawTerms : List Term := Proof.Events218.exact55997RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55997
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55997.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 55994) (rightBinding := 55995)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7272⟩) (rightExpression := ⟨11390⟩)
    (transferEvent := 55996)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult55993.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult55988.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult55997

namespace SemanticResult56003
def owner : Owner := ⟨.program ⟨214⟩, ⟨11392⟩⟩
def rawTerms : List Term := Proof.Events218.exact56003RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 56003
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56003.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 56000) (survivorTransfer := 56001)
    (survivorEvent := 56002) (resultEvent := resultEvent)
    (rightCoefficientProducer := 11974)
    (owner := owner) (leftOwner := SemanticResult55997.owner)
    (rightOwner := SemanticResult11975.owner)
    (leftResult := 55997) (rightResult := 11975)
    (leftBinding := 55998) (rightBinding := 55999)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11391⟩) (rightExpression := ⟨92⟩)
    (leftActual := SemanticResult55997.actual selector witness)
    (rightActual := SemanticResult11975.actual selector witness)
    (leftRaw := SemanticResult55997.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨92⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound11974.actual selector witness)
    (survivorMagnitude := LeftBound56001.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult55997.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11975.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11974.derived selector witness)
  · exact LeftBound56001.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult56003

namespace SemanticResult56011
def owner : Owner := ⟨.program ⟨214⟩, ⟨14002⟩⟩
def rawTerms : List Term := Proof.Events218.exact56011RawTerms
def summary : Bound := (.finite 13312)
def resultEvent : Nat := 56011
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56011.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨16, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge56009.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge56009.frameStart)
    (owner := owner) (leftOwner := SemanticResult56003.owner)
    (rightOwner := SemanticResult2594.owner)
    (leftResult := 56003) (rightResult := 2594)
    (leftActual := SemanticResult56003.actual selector witness)
    (rightActual := SemanticResult2594.actual selector witness)
    (leftRaw := SemanticResult56003.rawTerms)
    (rightRaw := SemanticResult2594.rawTerms)
    (working := LeftOperatorMerge56009.working)
    (leftBinding := 56004) (rightBinding := 56005)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11392⟩) (rightExpression := ⟨13999⟩)
    (coefficientTransfer := 56006) (summaryTransfer := 56008)
    (rightCoefficientProducer := 2593)
    (rightSummaryTransfer := 56007)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨16, by decide⟩)
    (rightRecordedMaximum := 16)
    (rightSummaryMaximum := ⟨16, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge56009.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority2593.actual selector witness)
    (summaryMagnitude := LeftBound56008.actual selector witness)
    (reconstruction := LeftOperatorMerge56009.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56003.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2594.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2593.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority2593.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge56009.operationAgreement
  · exact LeftBound56008.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge56009.working summary) := by
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
end SemanticResult56011

namespace SemanticResult56016
def owner : Owner := ⟨.program ⟨214⟩, ⟨14003⟩⟩
def rawTerms : List Term := Proof.Events218.exact56016RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56016
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56016.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge56015.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge56015.frameStart)
    (transferEvent := 56014) (owner := owner)
    (leftResult := 2594) (rightResult := 50670)
    (working := LeftOperatorMerge56015.working)
    (reconstruction := LeftOperatorMerge56015.reconstruction)
    (leftReference := .predecessor 0 56012 .coefficient) (rightReference := .predecessor 1 56013 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2594.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge56015.operationAgreement
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
end SemanticResult56016

namespace SemanticResult56021
def owner : Owner := ⟨.program ⟨214⟩, ⟨7252⟩⟩
def rawTerms : List Term := Proof.Events218.exact56021RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56021
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56021.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge56020.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge56020.frameStart)
    (transferEvent := 56019) (owner := owner)
    (leftResult := 50540) (rightResult := 12024)
    (working := LeftOperatorMerge56020.working)
    (reconstruction := LeftOperatorMerge56020.reconstruction)
    (leftReference := .predecessor 0 56017 .coefficient) (rightReference := .predecessor 1 56018 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12024.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge56020.operationAgreement
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
end SemanticResult56021

namespace SemanticResult56025
def owner : Owner := ⟨.program ⟨214⟩, ⟨14004⟩⟩
def rawTerms : List Term := Proof.Events218.exact56025RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56025
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56025.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 56022) (rightBinding := 56023)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7252⟩) (rightExpression := ⟨14003⟩)
    (transferEvent := 56024)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56021.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56016.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult56025

namespace SemanticResult56031
def owner : Owner := ⟨.program ⟨214⟩, ⟨14005⟩⟩
def rawTerms : List Term := Proof.Events218.exact56031RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 56031
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56031.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 56028) (survivorTransfer := 56029)
    (survivorEvent := 56030) (resultEvent := resultEvent)
    (rightCoefficientProducer := 12015)
    (owner := owner) (leftOwner := SemanticResult56025.owner)
    (rightOwner := SemanticResult12016.owner)
    (leftResult := 56025) (rightResult := 12016)
    (leftBinding := 56026) (rightBinding := 56027)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14004⟩) (rightExpression := ⟨72⟩)
    (leftActual := SemanticResult56025.actual selector witness)
    (rightActual := SemanticResult12016.actual selector witness)
    (leftRaw := SemanticResult56025.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨72⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound12015.actual selector witness)
    (survivorMagnitude := LeftBound56029.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56025.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12016.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12015.derived selector witness)
  · exact LeftBound56029.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult56031

namespace SemanticResult56041
def owner : Owner := ⟨.program ⟨214⟩, ⟨14006⟩⟩
def rawTerms : List Term := Proof.Events218.exact56041RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 56041
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56041.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge56037.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge56037.frameStart)
    (owner := owner) (leftOwner := SemanticResult56031.owner)
    (rightOwner := SemanticResult12013.owner)
    (leftResult := 56031) (rightResult := 12013)
    (leftActual := SemanticResult56031.actual selector witness)
    (rightActual := SemanticResult12013.actual selector witness)
    (leftRaw := SemanticResult56031.rawTerms)
    (rightRaw := SemanticResult12013.rawTerms)
    (working := LeftOperatorMerge56037.working)
    (leftBinding := 56032) (rightBinding := 56033)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14005⟩) (rightExpression := ⟨7850⟩)
    (coefficientTransfer := 56034) (summaryTransfer := 56036)
    (rightCoefficientProducer := 12012)
    (rightSummaryTransfer := 56035)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge56037.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound12012.actual selector witness)
    (summaryMagnitude := LeftBound56036.actual selector witness)
    (reconstruction := LeftOperatorMerge56037.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56031.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12013.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12012.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound12012.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge56037.operationAgreement
  · exact LeftBound56036.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge56037.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 56038 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge56037.working
    [{ coefficient := (-1), key := LeftRelationMerge56038.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge56038.frameStart
      LeftRelationMerge56038.owner (.relation 56038) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge56038.deltas
    rows := LeftRelationMerge56038.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge56037.working LeftRelationMerge56038.source
        (relationContext LeftRelationMerge56038.source
          LeftRelationMerge56038.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge56037.working, LeftRelationMerge56038.deltas,
    LeftRelationMerge56038.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 56038)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨14006⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge56037.working) (working := relationWorking0)
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
end SemanticResult56041

namespace SemanticResult56047
def owner : Owner := ⟨.program ⟨214⟩, ⟨14007⟩⟩
def rawTerms : List Term := Proof.Events218.exact56047RawTerms
def summary : Bound := (.finite 95433728)
def resultEvent : Nat := 56047
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56047.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge56045.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult56041.owner)
    (rightOwner := SemanticResult56011.owner)
    (leftResult := 56041) (rightResult := 56011)
    (leftActual := SemanticResult56041.actual selector witness)
    (rightActual := SemanticResult56011.actual selector witness)
    (leftRaw := SemanticResult56041.rawTerms)
    (rightRaw := SemanticResult56011.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 13312) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 56042) (rightBinding := 56043)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14006⟩) (rightExpression := ⟨14002⟩)
    (coefficientTransfer := 56044) (summaryTransfer := 56046)
    (base := LeftOperatorMerge56045.base)
    (reconstruction := LeftOperatorMerge56045.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56041.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56011.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge56045.operationAgreement
  · rfl
  · decide
end SemanticResult56047

namespace SemanticResult56057
def owner : Owner := ⟨.program ⟨214⟩, ⟨25995⟩⟩
def rawTerms : List Term := Proof.Events218.exact56057RawTerms
def summary : Bound := (.finite 350243308699648)
def resultEvent : Nat := 56057
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56057.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95433728, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge56053.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge56053.frameStart)
    (owner := owner) (leftOwner := SemanticResult56047.owner)
    (rightOwner := SemanticResult55983.owner)
    (leftResult := 56047) (rightResult := 55983)
    (leftActual := SemanticResult56047.actual selector witness)
    (rightActual := SemanticResult55983.actual selector witness)
    (leftRaw := SemanticResult56047.rawTerms)
    (rightRaw := SemanticResult55983.rawTerms)
    (working := LeftOperatorMerge56053.working)
    (leftBinding := 56048) (rightBinding := 56049)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14007⟩) (rightExpression := ⟨25994⟩)
    (coefficientTransfer := 56050) (summaryTransfer := 56052)
    (rightCoefficientProducer := 55982)
    (rightSummaryTransfer := 56051)
    (leftMaximum := ⟨95433728, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge56053.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority55982.actual selector witness)
    (summaryMagnitude := LeftBound56052.actual selector witness)
    (reconstruction := LeftOperatorMerge56053.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56047.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult55983.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55982.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority55982.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge56053.operationAgreement
  · exact LeftBound56052.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge56053.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 56054 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge56053.working
    [{ coefficient := (-1), key := LeftRelationMerge56054.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge56054.frameStart
      LeftRelationMerge56054.owner (.relation 56054) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge56054.deltas
    rows := LeftRelationMerge56054.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge56053.working LeftRelationMerge56054.source
        (relationContext LeftRelationMerge56054.source
          LeftRelationMerge56054.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge56053.working, LeftRelationMerge56054.deltas,
    LeftRelationMerge56054.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 56054)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25995⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11389⟩⟩, ⟨.program ⟨214⟩, ⟨13999⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25994⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge56053.working) (working := relationWorking0)
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
end SemanticResult56057

namespace SemanticResult56060
def owner : Owner := ⟨.program ⟨214⟩, ⟨19460⟩⟩
def rawTerms : List Term := Proof.Events218.exact56060RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56060
def producerEvent : Nat := 56059
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56060.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨14⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨14⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult56060

namespace SemanticResult56064
def owner : Owner := ⟨.program ⟨214⟩, ⟨19462⟩⟩
def rawTerms : List Term := Proof.Events219.exact56064RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56064
def producerEvent : Nat := 56063
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56064.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 56061 .coefficient) (.value (.predecessor 1 56062 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 56061 .coefficient) (.value (.predecessor 1 56062 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult56064

namespace SemanticResult56142
def owner : Owner := ⟨.program ⟨214⟩, ⟨11389⟩⟩
def rawTerms : List Term := Proof.Events219.exact56142RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56142
def producerEvent : Nat := 56141
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56142.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 56119, .finite 16, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult56142

namespace SemanticResult56145
def owner : Owner := ⟨.program ⟨214⟩, ⟨13999⟩⟩
def rawTerms : List Term := Proof.Events219.exact56145RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56145
def producerEvent : Nat := 56144
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56145.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 56119, .finite 16, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult56145

namespace SemanticResult56150
def owner : Owner := ⟨.program ⟨214⟩, ⟨14000⟩⟩
def rawTerms : List Term := Proof.Events219.exact56150RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56150
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56150.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge56149.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge56149.frameStart)
    (transferEvent := 56148) (owner := owner)
    (leftResult := 56145) (rightResult := 56142)
    (working := LeftOperatorMerge56149.working)
    (reconstruction := LeftOperatorMerge56149.reconstruction)
    (leftReference := .predecessor 0 56146 .coefficient) (rightReference := .predecessor 1 56147 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult56145.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56142.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge56149.operationAgreement
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
end SemanticResult56150

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
