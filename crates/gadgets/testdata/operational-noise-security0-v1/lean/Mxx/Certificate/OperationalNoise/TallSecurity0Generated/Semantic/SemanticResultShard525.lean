import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard525
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard027
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard121
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard122
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard524

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult73028
def owner : Owner := ⟨.program ⟨214⟩, ⟨7191⟩⟩
def rawTerms : List Term := Proof.Events285.exact73028RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73028
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73028.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73027.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge73027.frameStart)
    (transferEvent := 73026) (owner := owner)
    (leftResult := 65165) (rightResult := 14488)
    (working := LeftOperatorMerge73027.working)
    (reconstruction := LeftOperatorMerge73027.reconstruction)
    (leftReference := .predecessor 0 73024 .coefficient) (rightReference := .predecessor 1 73025 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14488.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge73027.operationAgreement
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
end SemanticResult73028

namespace SemanticResult73032
def owner : Owner := ⟨.program ⟨214⟩, ⟨10672⟩⟩
def rawTerms : List Term := Proof.Events285.exact73032RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73032
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73032.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 73029) (rightBinding := 73030)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7191⟩) (rightExpression := ⟨10671⟩)
    (transferEvent := 73031)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73028.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73023.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult73032

namespace SemanticResult73038
def owner : Owner := ⟨.program ⟨214⟩, ⟨10673⟩⟩
def rawTerms : List Term := Proof.Events285.exact73038RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 73038
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73038.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 73035) (survivorTransfer := 73036)
    (survivorEvent := 73037) (resultEvent := resultEvent)
    (rightCoefficientProducer := 14479)
    (owner := owner) (leftOwner := SemanticResult73032.owner)
    (rightOwner := SemanticResult14480.owner)
    (leftResult := 73032) (rightResult := 14480)
    (leftBinding := 73033) (rightBinding := 73034)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10672⟩) (rightExpression := ⟨87⟩)
    (leftActual := SemanticResult73032.actual selector witness)
    (rightActual := SemanticResult14480.actual selector witness)
    (leftRaw := SemanticResult73032.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨87⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound14479.actual selector witness)
    (survivorMagnitude := LeftBound73036.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73032.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14480.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14479.derived selector witness)
  · exact LeftBound73036.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult73038

namespace SemanticResult73046
def owner : Owner := ⟨.program ⟨214⟩, ⟨10674⟩⟩
def rawTerms : List Term := Proof.Events285.exact73046RawTerms
def summary : Bound := (.finite 2496)
def resultEvent : Nat := 73046
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73046.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨3, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73044.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge73044.frameStart)
    (owner := owner) (leftOwner := SemanticResult73038.owner)
    (rightOwner := SemanticResult3457.owner)
    (leftResult := 73038) (rightResult := 3457)
    (leftActual := SemanticResult73038.actual selector witness)
    (rightActual := SemanticResult3457.actual selector witness)
    (leftRaw := SemanticResult73038.rawTerms)
    (rightRaw := SemanticResult3457.rawTerms)
    (working := LeftOperatorMerge73044.working)
    (leftBinding := 73039) (rightBinding := 73040)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10673⟩) (rightExpression := ⟨9500⟩)
    (coefficientTransfer := 73041) (summaryTransfer := 73043)
    (rightCoefficientProducer := 3456)
    (rightSummaryTransfer := 73042)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨3, by decide⟩)
    (rightRecordedMaximum := 3)
    (rightSummaryMaximum := ⟨3, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge73044.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority3456.actual selector witness)
    (summaryMagnitude := LeftBound73043.actual selector witness)
    (reconstruction := LeftOperatorMerge73044.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73038.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3457.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3456.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority3456.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge73044.operationAgreement
  · exact LeftBound73043.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73044.working summary) := by
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
end SemanticResult73046

namespace SemanticResult73051
def owner : Owner := ⟨.program ⟨214⟩, ⟨9501⟩⟩
def rawTerms : List Term := Proof.Events285.exact73051RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73051
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73051.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73050.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge73050.frameStart)
    (transferEvent := 73049) (owner := owner)
    (leftResult := 3457) (rightResult := 65295)
    (working := LeftOperatorMerge73050.working)
    (reconstruction := LeftOperatorMerge73050.reconstruction)
    (leftReference := .predecessor 0 73047 .coefficient) (rightReference := .predecessor 1 73048 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3457.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge73050.operationAgreement
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
end SemanticResult73051

namespace SemanticResult73056
def owner : Owner := ⟨.program ⟨214⟩, ⟨7200⟩⟩
def rawTerms : List Term := Proof.Events285.exact73056RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73056
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73056.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73055.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge73055.frameStart)
    (transferEvent := 73054) (owner := owner)
    (leftResult := 65165) (rightResult := 14529)
    (working := LeftOperatorMerge73055.working)
    (reconstruction := LeftOperatorMerge73055.reconstruction)
    (leftReference := .predecessor 0 73052 .coefficient) (rightReference := .predecessor 1 73053 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14529.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge73055.operationAgreement
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
end SemanticResult73056

namespace SemanticResult73060
def owner : Owner := ⟨.program ⟨214⟩, ⟨9502⟩⟩
def rawTerms : List Term := Proof.Events285.exact73060RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73060
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73060.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 73057) (rightBinding := 73058)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7200⟩) (rightExpression := ⟨9501⟩)
    (transferEvent := 73059)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73056.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73051.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult73060

namespace SemanticResult73066
def owner : Owner := ⟨.program ⟨214⟩, ⟨9503⟩⟩
def rawTerms : List Term := Proof.Events285.exact73066RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 73066
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73066.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 73063) (survivorTransfer := 73064)
    (survivorEvent := 73065) (resultEvent := resultEvent)
    (rightCoefficientProducer := 14520)
    (owner := owner) (leftOwner := SemanticResult73060.owner)
    (rightOwner := SemanticResult14521.owner)
    (leftResult := 73060) (rightResult := 14521)
    (leftBinding := 73061) (rightBinding := 73062)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9502⟩) (rightExpression := ⟨96⟩)
    (leftActual := SemanticResult73060.actual selector witness)
    (rightActual := SemanticResult14521.actual selector witness)
    (leftRaw := SemanticResult73060.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨96⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound14520.actual selector witness)
    (survivorMagnitude := LeftBound73064.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73060.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14521.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14520.derived selector witness)
  · exact LeftBound73064.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult73066

namespace SemanticResult73076
def owner : Owner := ⟨.program ⟨214⟩, ⟨9504⟩⟩
def rawTerms : List Term := Proof.Events285.exact73076RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 73076
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73076.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73072.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge73072.frameStart)
    (owner := owner) (leftOwner := SemanticResult73066.owner)
    (rightOwner := SemanticResult14518.owner)
    (leftResult := 73066) (rightResult := 14518)
    (leftActual := SemanticResult73066.actual selector witness)
    (rightActual := SemanticResult14518.actual selector witness)
    (leftRaw := SemanticResult73066.rawTerms)
    (rightRaw := SemanticResult14518.rawTerms)
    (working := LeftOperatorMerge73072.working)
    (leftBinding := 73067) (rightBinding := 73068)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9503⟩) (rightExpression := ⟨7835⟩)
    (coefficientTransfer := 73069) (summaryTransfer := 73071)
    (rightCoefficientProducer := 14517)
    (rightSummaryTransfer := 73070)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge73072.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound14517.actual selector witness)
    (summaryMagnitude := LeftBound73071.actual selector witness)
    (reconstruction := LeftOperatorMerge73072.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73066.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14518.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14517.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound14517.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge73072.operationAgreement
  · exact LeftBound73071.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73072.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 73073 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge73072.working
    [{ coefficient := (-1), key := LeftRelationMerge73073.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge73073.frameStart
      LeftRelationMerge73073.owner (.relation 73073) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge73073.deltas
    rows := LeftRelationMerge73073.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge73072.working LeftRelationMerge73073.source
        (relationContext LeftRelationMerge73073.source
          LeftRelationMerge73073.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge73072.working, LeftRelationMerge73073.deltas,
    LeftRelationMerge73073.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 73073)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9504⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge73072.working) (working := relationWorking0)
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
end SemanticResult73076

namespace SemanticResult73082
def owner : Owner := ⟨.program ⟨214⟩, ⟨10675⟩⟩
def rawTerms : List Term := Proof.Events285.exact73082RawTerms
def summary : Bound := (.finite 95422912)
def resultEvent : Nat := 73082
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73082.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge73080.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult73076.owner)
    (rightOwner := SemanticResult73046.owner)
    (leftResult := 73076) (rightResult := 73046)
    (leftActual := SemanticResult73076.actual selector witness)
    (rightActual := SemanticResult73046.actual selector witness)
    (leftRaw := SemanticResult73076.rawTerms)
    (rightRaw := SemanticResult73046.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 2496) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 73077) (rightBinding := 73078)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9504⟩) (rightExpression := ⟨10674⟩)
    (coefficientTransfer := 73079) (summaryTransfer := 73081)
    (base := LeftOperatorMerge73080.base)
    (reconstruction := LeftOperatorMerge73080.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73076.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73046.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge73080.operationAgreement
  · rfl
  · decide
end SemanticResult73082

namespace SemanticResult73092
def owner : Owner := ⟨.program ⟨214⟩, ⟨24984⟩⟩
def rawTerms : List Term := Proof.Events285.exact73092RawTerms
def summary : Bound := (.finite 350203613806592)
def resultEvent : Nat := 73092
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73092.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95422912, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73088.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge73088.frameStart)
    (owner := owner) (leftOwner := SemanticResult73082.owner)
    (rightOwner := SemanticResult73018.owner)
    (leftResult := 73082) (rightResult := 73018)
    (leftActual := SemanticResult73082.actual selector witness)
    (rightActual := SemanticResult73018.actual selector witness)
    (leftRaw := SemanticResult73082.rawTerms)
    (rightRaw := SemanticResult73018.rawTerms)
    (working := LeftOperatorMerge73088.working)
    (leftBinding := 73083) (rightBinding := 73084)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10675⟩) (rightExpression := ⟨24983⟩)
    (coefficientTransfer := 73085) (summaryTransfer := 73087)
    (rightCoefficientProducer := 73017)
    (rightSummaryTransfer := 73086)
    (leftMaximum := ⟨95422912, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge73088.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority73017.actual selector witness)
    (summaryMagnitude := LeftBound73087.actual selector witness)
    (reconstruction := LeftOperatorMerge73088.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73082.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73018.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73017.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority73017.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge73088.operationAgreement
  · exact LeftBound73087.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73088.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 73089 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22994⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22994⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge73088.working
    [{ coefficient := (-1), key := LeftRelationMerge73089.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge73089.frameStart
      LeftRelationMerge73089.owner (.relation 73089) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge73089.deltas
    rows := LeftRelationMerge73089.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge73088.working LeftRelationMerge73089.source
        (relationContext LeftRelationMerge73089.source
          LeftRelationMerge73089.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge73088.working, LeftRelationMerge73089.deltas,
    LeftRelationMerge73089.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 73089)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨24984⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9500⟩⟩, ⟨.program ⟨214⟩, ⟨10668⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24983⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge73088.working) (working := relationWorking0)
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
end SemanticResult73092

namespace SemanticResult73095
def owner : Owner := ⟨.program ⟨214⟩, ⟨19092⟩⟩
def rawTerms : List Term := Proof.Events285.exact73095RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73095
def producerEvent : Nat := 73094
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73095.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨8⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨8⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult73095

namespace SemanticResult73099
def owner : Owner := ⟨.program ⟨214⟩, ⟨19094⟩⟩
def rawTerms : List Term := Proof.Events285.exact73099RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73099
def producerEvent : Nat := 73098
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73099.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 73096 .coefficient) (.value (.predecessor 1 73097 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 73096 .coefficient) (.value (.predecessor 1 73097 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult73099

namespace SemanticResult73177
def owner : Owner := ⟨.program ⟨214⟩, ⟨10668⟩⟩
def rawTerms : List Term := Proof.Events285.exact73177RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73177
def producerEvent : Nat := 73176
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73177.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 73154, .finite 3, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult73177

namespace SemanticResult73180
def owner : Owner := ⟨.program ⟨214⟩, ⟨9500⟩⟩
def rawTerms : List Term := Proof.Events285.exact73180RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73180
def producerEvent : Nat := 73179
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73180.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 73154, .finite 3, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult73180

namespace SemanticResult73185
def owner : Owner := ⟨.program ⟨214⟩, ⟨10669⟩⟩
def rawTerms : List Term := Proof.Events285.exact73185RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73185
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73185.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73184.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge73184.frameStart)
    (transferEvent := 73183) (owner := owner)
    (leftResult := 73180) (rightResult := 73177)
    (working := LeftOperatorMerge73184.working)
    (reconstruction := LeftOperatorMerge73184.reconstruction)
    (leftReference := .predecessor 0 73181 .coefficient) (rightReference := .predecessor 1 73182 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult73180.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73177.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge73184.operationAgreement
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
end SemanticResult73185

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
