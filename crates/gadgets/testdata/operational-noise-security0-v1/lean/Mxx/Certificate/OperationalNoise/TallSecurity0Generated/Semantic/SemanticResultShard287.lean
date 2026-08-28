import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard287
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard014
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard081
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard286

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult38958
def owner : Owner := ⟨.program ⟨214⟩, ⟨7316⟩⟩
def rawTerms : List Term := Proof.Events152.exact38958RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 38958
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult38958.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge38957.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge38957.frameStart)
    (transferEvent := 38956) (owner := owner)
    (leftResult := 35915) (rightResult := 9478)
    (working := LeftOperatorMerge38957.working)
    (reconstruction := LeftOperatorMerge38957.reconstruction)
    (leftReference := .predecessor 0 38954 .coefficient) (rightReference := .predecessor 1 38955 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9478.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge38957.operationAgreement
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
end SemanticResult38958

namespace SemanticResult38962
def owner : Owner := ⟨.program ⟨214⟩, ⟨11977⟩⟩
def rawTerms : List Term := Proof.Events152.exact38962RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 38962
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult38962.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 38959) (rightBinding := 38960)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7316⟩) (rightExpression := ⟨11976⟩)
    (transferEvent := 38961)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult38958.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult38953.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult38962

namespace SemanticResult38968
def owner : Owner := ⟨.program ⟨214⟩, ⟨11978⟩⟩
def rawTerms : List Term := Proof.Events152.exact38968RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 38968
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult38968.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 38965) (survivorTransfer := 38966)
    (survivorEvent := 38967) (resultEvent := resultEvent)
    (rightCoefficientProducer := 9469)
    (owner := owner) (leftOwner := SemanticResult38962.owner)
    (rightOwner := SemanticResult9470.owner)
    (leftResult := 38962) (rightResult := 9470)
    (leftBinding := 38963) (rightBinding := 38964)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11977⟩) (rightExpression := ⟨98⟩)
    (leftActual := SemanticResult38962.actual selector witness)
    (rightActual := SemanticResult9470.actual selector witness)
    (leftRaw := SemanticResult38962.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨98⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound9469.actual selector witness)
    (survivorMagnitude := LeftBound38966.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult38962.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9470.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9469.derived selector witness)
  · exact LeftBound38966.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult38968

namespace SemanticResult38976
def owner : Owner := ⟨.program ⟨214⟩, ⟨11979⟩⟩
def rawTerms : List Term := Proof.Events152.exact38976RawTerms
def summary : Bound := (.finite 29952)
def resultEvent : Nat := 38976
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult38976.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨36, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge38974.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge38974.frameStart)
    (owner := owner) (leftOwner := SemanticResult38968.owner)
    (rightOwner := SemanticResult1731.owner)
    (leftResult := 38968) (rightResult := 1731)
    (leftActual := SemanticResult38968.actual selector witness)
    (rightActual := SemanticResult1731.actual selector witness)
    (leftRaw := SemanticResult38968.rawTerms)
    (rightRaw := SemanticResult1731.rawTerms)
    (working := LeftOperatorMerge38974.working)
    (leftBinding := 38969) (rightBinding := 38970)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11978⟩) (rightExpression := ⟨9725⟩)
    (coefficientTransfer := 38971) (summaryTransfer := 38973)
    (rightCoefficientProducer := 1730)
    (rightSummaryTransfer := 38972)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨36, by decide⟩)
    (rightRecordedMaximum := 36)
    (rightSummaryMaximum := ⟨36, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge38974.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1730.actual selector witness)
    (summaryMagnitude := LeftBound38973.actual selector witness)
    (reconstruction := LeftOperatorMerge38974.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult38968.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1731.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1730.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1730.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge38974.operationAgreement
  · exact LeftBound38973.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge38974.working summary) := by
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
end SemanticResult38976

namespace SemanticResult38981
def owner : Owner := ⟨.program ⟨214⟩, ⟨9726⟩⟩
def rawTerms : List Term := Proof.Events152.exact38981RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 38981
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult38981.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge38980.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge38980.frameStart)
    (transferEvent := 38979) (owner := owner)
    (leftResult := 1731) (rightResult := 36045)
    (working := LeftOperatorMerge38980.working)
    (reconstruction := LeftOperatorMerge38980.reconstruction)
    (leftReference := .predecessor 0 38977 .coefficient) (rightReference := .predecessor 1 38978 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1731.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge38980.operationAgreement
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
end SemanticResult38981

namespace SemanticResult38986
def owner : Owner := ⟨.program ⟨214⟩, ⟨7296⟩⟩
def rawTerms : List Term := Proof.Events152.exact38986RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 38986
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult38986.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge38985.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge38985.frameStart)
    (transferEvent := 38984) (owner := owner)
    (leftResult := 35915) (rightResult := 9519)
    (working := LeftOperatorMerge38985.working)
    (reconstruction := LeftOperatorMerge38985.reconstruction)
    (leftReference := .predecessor 0 38982 .coefficient) (rightReference := .predecessor 1 38983 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9519.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge38985.operationAgreement
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
end SemanticResult38986

namespace SemanticResult38990
def owner : Owner := ⟨.program ⟨214⟩, ⟨9727⟩⟩
def rawTerms : List Term := Proof.Events152.exact38990RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 38990
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult38990.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 38987) (rightBinding := 38988)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7296⟩) (rightExpression := ⟨9726⟩)
    (transferEvent := 38989)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult38986.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult38981.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult38990

namespace SemanticResult38996
def owner : Owner := ⟨.program ⟨214⟩, ⟨9728⟩⟩
def rawTerms : List Term := Proof.Events152.exact38996RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 38996
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult38996.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 38993) (survivorTransfer := 38994)
    (survivorEvent := 38995) (resultEvent := resultEvent)
    (rightCoefficientProducer := 9510)
    (owner := owner) (leftOwner := SemanticResult38990.owner)
    (rightOwner := SemanticResult9511.owner)
    (leftResult := 38990) (rightResult := 9511)
    (leftBinding := 38991) (rightBinding := 38992)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9727⟩) (rightExpression := ⟨78⟩)
    (leftActual := SemanticResult38990.actual selector witness)
    (rightActual := SemanticResult9511.actual selector witness)
    (leftRaw := SemanticResult38990.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨78⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound9510.actual selector witness)
    (survivorMagnitude := LeftBound38994.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult38990.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9511.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9510.derived selector witness)
  · exact LeftBound38994.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult38996

namespace SemanticResult39006
def owner : Owner := ⟨.program ⟨214⟩, ⟨9729⟩⟩
def rawTerms : List Term := Proof.Events152.exact39006RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 39006
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39006.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39002.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge39002.frameStart)
    (owner := owner) (leftOwner := SemanticResult38996.owner)
    (rightOwner := SemanticResult9508.owner)
    (leftResult := 38996) (rightResult := 9508)
    (leftActual := SemanticResult38996.actual selector witness)
    (rightActual := SemanticResult9508.actual selector witness)
    (leftRaw := SemanticResult38996.rawTerms)
    (rightRaw := SemanticResult9508.rawTerms)
    (working := LeftOperatorMerge39002.working)
    (leftBinding := 38997) (rightBinding := 38998)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9728⟩) (rightExpression := ⟨7865⟩)
    (coefficientTransfer := 38999) (summaryTransfer := 39001)
    (rightCoefficientProducer := 9507)
    (rightSummaryTransfer := 39000)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge39002.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound9507.actual selector witness)
    (summaryMagnitude := LeftBound39001.actual selector witness)
    (reconstruction := LeftOperatorMerge39002.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult38996.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9508.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9507.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound9507.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge39002.operationAgreement
  · exact LeftBound39001.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39002.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 39003 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9725⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge39002.working
    [{ coefficient := (-1), key := LeftRelationMerge39003.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge39003.frameStart
      LeftRelationMerge39003.owner (.relation 39003) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge39003.deltas
    rows := LeftRelationMerge39003.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge39002.working LeftRelationMerge39003.source
        (relationContext LeftRelationMerge39003.source
          LeftRelationMerge39003.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge39002.working, LeftRelationMerge39003.deltas,
    LeftRelationMerge39003.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 39003)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9729⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9725⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge39002.working) (working := relationWorking0)
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
end SemanticResult39006

namespace SemanticResult39012
def owner : Owner := ⟨.program ⟨214⟩, ⟨11980⟩⟩
def rawTerms : List Term := Proof.Events152.exact39012RawTerms
def summary : Bound := (.finite 95450368)
def resultEvent : Nat := 39012
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39012.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge39010.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult39006.owner)
    (rightOwner := SemanticResult38976.owner)
    (leftResult := 39006) (rightResult := 38976)
    (leftActual := SemanticResult39006.actual selector witness)
    (rightActual := SemanticResult38976.actual selector witness)
    (leftRaw := SemanticResult39006.rawTerms)
    (rightRaw := SemanticResult38976.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 29952) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 39007) (rightBinding := 39008)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9729⟩) (rightExpression := ⟨11979⟩)
    (coefficientTransfer := 39009) (summaryTransfer := 39011)
    (base := LeftOperatorMerge39010.base)
    (reconstruction := LeftOperatorMerge39010.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult39006.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult38976.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge39010.operationAgreement
  · rfl
  · decide
end SemanticResult39012

namespace SemanticResult39022
def owner : Owner := ⟨.program ⟨214⟩, ⟨25230⟩⟩
def rawTerms : List Term := Proof.Events152.exact39022RawTerms
def summary : Bound := (.finite 350304377765888)
def resultEvent : Nat := 39022
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39022.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95450368, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39018.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge39018.frameStart)
    (owner := owner) (leftOwner := SemanticResult39012.owner)
    (rightOwner := SemanticResult38948.owner)
    (leftResult := 39012) (rightResult := 38948)
    (leftActual := SemanticResult39012.actual selector witness)
    (rightActual := SemanticResult38948.actual selector witness)
    (leftRaw := SemanticResult39012.rawTerms)
    (rightRaw := SemanticResult38948.rawTerms)
    (working := LeftOperatorMerge39018.working)
    (leftBinding := 39013) (rightBinding := 39014)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11980⟩) (rightExpression := ⟨25229⟩)
    (coefficientTransfer := 39015) (summaryTransfer := 39017)
    (rightCoefficientProducer := 38947)
    (rightSummaryTransfer := 39016)
    (leftMaximum := ⟨95450368, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge39018.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority38947.actual selector witness)
    (summaryMagnitude := LeftBound39017.actual selector witness)
    (reconstruction := LeftOperatorMerge39018.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult39012.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult38948.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38947.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority38947.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge39018.operationAgreement
  · exact LeftBound39017.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39018.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 39019 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25229⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9725⟩⟩, ⟨.program ⟨214⟩, ⟨11973⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23126⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23126⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge39018.working
    [{ coefficient := (-1), key := LeftRelationMerge39019.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge39019.frameStart
      LeftRelationMerge39019.owner (.relation 39019) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge39019.deltas
    rows := LeftRelationMerge39019.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge39018.working LeftRelationMerge39019.source
        (relationContext LeftRelationMerge39019.source
          LeftRelationMerge39019.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge39018.working, LeftRelationMerge39019.deltas,
    LeftRelationMerge39019.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 39019)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25230⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9725⟩⟩, ⟨.program ⟨214⟩, ⟨11973⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25229⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25229⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge39018.working) (working := relationWorking0)
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
end SemanticResult39022

namespace SemanticResult39025
def owner : Owner := ⟨.program ⟨214⟩, ⟨19824⟩⟩
def rawTerms : List Term := Proof.Events152.exact39025RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39025
def producerEvent : Nat := 39024
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39025.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨19⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨19⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult39025

namespace SemanticResult39029
def owner : Owner := ⟨.program ⟨214⟩, ⟨19826⟩⟩
def rawTerms : List Term := Proof.Events152.exact39029RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39029
def producerEvent : Nat := 39028
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39029.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 39026 .coefficient) (.value (.predecessor 1 39027 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 39026 .coefficient) (.value (.predecessor 1 39027 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult39029

namespace SemanticResult39107
def owner : Owner := ⟨.program ⟨214⟩, ⟨11973⟩⟩
def rawTerms : List Term := Proof.Events152.exact39107RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39107
def producerEvent : Nat := 39106
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39107.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 39084, .finite 36, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult39107

namespace SemanticResult39110
def owner : Owner := ⟨.program ⟨214⟩, ⟨9725⟩⟩
def rawTerms : List Term := Proof.Events152.exact39110RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39110
def producerEvent : Nat := 39109
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39110.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 39084, .finite 36, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult39110

namespace SemanticResult39115
def owner : Owner := ⟨.program ⟨214⟩, ⟨11974⟩⟩
def rawTerms : List Term := Proof.Events152.exact39115RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39115
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39115.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39114.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge39114.frameStart)
    (transferEvent := 39113) (owner := owner)
    (leftResult := 39110) (rightResult := 39107)
    (working := LeftOperatorMerge39114.working)
    (reconstruction := LeftOperatorMerge39114.reconstruction)
    (leftReference := .predecessor 0 39111 .coefficient) (rightReference := .predecessor 1 39112 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult39110.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39107.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge39114.operationAgreement
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
end SemanticResult39115

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
