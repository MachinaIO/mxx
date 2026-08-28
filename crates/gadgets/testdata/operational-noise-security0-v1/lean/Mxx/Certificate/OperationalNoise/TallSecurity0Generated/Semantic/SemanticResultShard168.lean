import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard168
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard007
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard061
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard167

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult21923
def owner : Owner := ⟨.program ⟨214⟩, ⟨7359⟩⟩
def rawTerms : List Term := Proof.Events085.exact21923RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21923
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21923.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21922.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge21922.frameStart)
    (transferEvent := 21921) (owner := owner)
    (leftResult := 21290) (rightResult := 6973)
    (working := LeftOperatorMerge21922.working)
    (reconstruction := LeftOperatorMerge21922.reconstruction)
    (leftReference := .predecessor 0 21919 .coefficient) (rightReference := .predecessor 1 21920 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6973.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge21922.operationAgreement
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
end SemanticResult21923

namespace SemanticResult21927
def owner : Owner := ⟨.program ⟨214⟩, ⟨13182⟩⟩
def rawTerms : List Term := Proof.Events085.exact21927RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21927
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21927.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 21924) (rightBinding := 21925)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7359⟩) (rightExpression := ⟨13181⟩)
    (transferEvent := 21926)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21923.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21918.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult21927

namespace SemanticResult21933
def owner : Owner := ⟨.program ⟨214⟩, ⟨13183⟩⟩
def rawTerms : List Term := Proof.Events085.exact21933RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 21933
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21933.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 21930) (survivorTransfer := 21931)
    (survivorEvent := 21932) (resultEvent := resultEvent)
    (rightCoefficientProducer := 6964)
    (owner := owner) (leftOwner := SemanticResult21927.owner)
    (rightOwner := SemanticResult6965.owner)
    (leftResult := 21927) (rightResult := 6965)
    (leftBinding := 21928) (rightBinding := 21929)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13182⟩) (rightExpression := ⟨103⟩)
    (leftActual := SemanticResult21927.actual selector witness)
    (rightActual := SemanticResult6965.actual selector witness)
    (leftRaw := SemanticResult21927.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨103⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound6964.actual selector witness)
    (survivorMagnitude := LeftBound21931.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21927.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6965.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6964.derived selector witness)
  · exact LeftBound21931.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult21933

namespace SemanticResult21941
def owner : Owner := ⟨.program ⟨214⟩, ⟨13184⟩⟩
def rawTerms : List Term := Proof.Events085.exact21941RawTerms
def summary : Bound := (.finite 48256)
def resultEvent : Nat := 21941
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21941.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨58, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21939.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge21939.frameStart)
    (owner := owner) (leftOwner := SemanticResult21933.owner)
    (rightOwner := SemanticResult868.owner)
    (leftResult := 21933) (rightResult := 868)
    (leftActual := SemanticResult21933.actual selector witness)
    (rightActual := SemanticResult868.actual selector witness)
    (leftRaw := SemanticResult21933.rawTerms)
    (rightRaw := SemanticResult868.rawTerms)
    (working := LeftOperatorMerge21939.working)
    (leftBinding := 21934) (rightBinding := 21935)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13183⟩) (rightExpression := ⟨10255⟩)
    (coefficientTransfer := 21936) (summaryTransfer := 21938)
    (rightCoefficientProducer := 867)
    (rightSummaryTransfer := 21937)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨58, by decide⟩)
    (rightRecordedMaximum := 58)
    (rightSummaryMaximum := ⟨58, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge21939.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority867.actual selector witness)
    (summaryMagnitude := LeftBound21938.actual selector witness)
    (reconstruction := LeftOperatorMerge21939.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21933.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult868.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority867.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority867.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge21939.operationAgreement
  · exact LeftBound21938.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21939.working summary) := by
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
end SemanticResult21941

namespace SemanticResult21946
def owner : Owner := ⟨.program ⟨214⟩, ⟨10256⟩⟩
def rawTerms : List Term := Proof.Events085.exact21946RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21946
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21946.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21945.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge21945.frameStart)
    (transferEvent := 21944) (owner := owner)
    (leftResult := 868) (rightResult := 21420)
    (working := LeftOperatorMerge21945.working)
    (reconstruction := LeftOperatorMerge21945.reconstruction)
    (leftReference := .predecessor 0 21942 .coefficient) (rightReference := .predecessor 1 21943 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult868.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge21945.operationAgreement
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
end SemanticResult21946

namespace SemanticResult21951
def owner : Owner := ⟨.program ⟨214⟩, ⟨7339⟩⟩
def rawTerms : List Term := Proof.Events085.exact21951RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21951
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21951.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21950.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge21950.frameStart)
    (transferEvent := 21949) (owner := owner)
    (leftResult := 21290) (rightResult := 7014)
    (working := LeftOperatorMerge21950.working)
    (reconstruction := LeftOperatorMerge21950.reconstruction)
    (leftReference := .predecessor 0 21947 .coefficient) (rightReference := .predecessor 1 21948 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7014.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge21950.operationAgreement
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
end SemanticResult21951

namespace SemanticResult21955
def owner : Owner := ⟨.program ⟨214⟩, ⟨10257⟩⟩
def rawTerms : List Term := Proof.Events085.exact21955RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21955
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21955.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 21952) (rightBinding := 21953)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7339⟩) (rightExpression := ⟨10256⟩)
    (transferEvent := 21954)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21951.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21946.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult21955

namespace SemanticResult21961
def owner : Owner := ⟨.program ⟨214⟩, ⟨10258⟩⟩
def rawTerms : List Term := Proof.Events085.exact21961RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 21961
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21961.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 21958) (survivorTransfer := 21959)
    (survivorEvent := 21960) (resultEvent := resultEvent)
    (rightCoefficientProducer := 7005)
    (owner := owner) (leftOwner := SemanticResult21955.owner)
    (rightOwner := SemanticResult7006.owner)
    (leftResult := 21955) (rightResult := 7006)
    (leftBinding := 21956) (rightBinding := 21957)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10257⟩) (rightExpression := ⟨83⟩)
    (leftActual := SemanticResult21955.actual selector witness)
    (rightActual := SemanticResult7006.actual selector witness)
    (leftRaw := SemanticResult21955.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨83⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound7005.actual selector witness)
    (survivorMagnitude := LeftBound21959.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21955.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7006.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7005.derived selector witness)
  · exact LeftBound21959.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult21961

namespace SemanticResult21971
def owner : Owner := ⟨.program ⟨214⟩, ⟨10259⟩⟩
def rawTerms : List Term := Proof.Events085.exact21971RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 21971
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21971.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21967.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge21967.frameStart)
    (owner := owner) (leftOwner := SemanticResult21961.owner)
    (rightOwner := SemanticResult7003.owner)
    (leftResult := 21961) (rightResult := 7003)
    (leftActual := SemanticResult21961.actual selector witness)
    (rightActual := SemanticResult7003.actual selector witness)
    (leftRaw := SemanticResult21961.rawTerms)
    (rightRaw := SemanticResult7003.rawTerms)
    (working := LeftOperatorMerge21967.working)
    (leftBinding := 21962) (rightBinding := 21963)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10258⟩) (rightExpression := ⟨7880⟩)
    (coefficientTransfer := 21964) (summaryTransfer := 21966)
    (rightCoefficientProducer := 7002)
    (rightSummaryTransfer := 21965)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge21967.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound7002.actual selector witness)
    (summaryMagnitude := LeftBound21966.actual selector witness)
    (reconstruction := LeftOperatorMerge21967.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21961.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7003.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7002.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound7002.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge21967.operationAgreement
  · exact LeftBound21966.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21967.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 21968 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge21967.working
    [{ coefficient := (-1), key := LeftRelationMerge21968.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge21968.frameStart
      LeftRelationMerge21968.owner (.relation 21968) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge21968.deltas
    rows := LeftRelationMerge21968.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge21967.working LeftRelationMerge21968.source
        (relationContext LeftRelationMerge21968.source
          LeftRelationMerge21968.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge21967.working, LeftRelationMerge21968.deltas,
    LeftRelationMerge21968.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 21968)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10259⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge21967.working) (working := relationWorking0)
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
end SemanticResult21971

namespace SemanticResult21977
def owner : Owner := ⟨.program ⟨214⟩, ⟨13185⟩⟩
def rawTerms : List Term := Proof.Events085.exact21977RawTerms
def summary : Bound := (.finite 95468672)
def resultEvent : Nat := 21977
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21977.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge21975.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult21971.owner)
    (rightOwner := SemanticResult21941.owner)
    (leftResult := 21971) (rightResult := 21941)
    (leftActual := SemanticResult21971.actual selector witness)
    (rightActual := SemanticResult21941.actual selector witness)
    (leftRaw := SemanticResult21971.rawTerms)
    (rightRaw := SemanticResult21941.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 48256) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 21972) (rightBinding := 21973)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10259⟩) (rightExpression := ⟨13184⟩)
    (coefficientTransfer := 21974) (summaryTransfer := 21976)
    (base := LeftOperatorMerge21975.base)
    (reconstruction := LeftOperatorMerge21975.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21971.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21941.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge21975.operationAgreement
  · rfl
  · decide
end SemanticResult21977

namespace SemanticResult21987
def owner : Owner := ⟨.program ⟨214⟩, ⟨25697⟩⟩
def rawTerms : List Term := Proof.Events085.exact21987RawTerms
def summary : Bound := (.finite 350371553738752)
def resultEvent : Nat := 21987
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21987.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95468672, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21983.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge21983.frameStart)
    (owner := owner) (leftOwner := SemanticResult21977.owner)
    (rightOwner := SemanticResult21913.owner)
    (leftResult := 21977) (rightResult := 21913)
    (leftActual := SemanticResult21977.actual selector witness)
    (rightActual := SemanticResult21913.actual selector witness)
    (leftRaw := SemanticResult21977.rawTerms)
    (rightRaw := SemanticResult21913.rawTerms)
    (working := LeftOperatorMerge21983.working)
    (leftBinding := 21978) (rightBinding := 21979)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13185⟩) (rightExpression := ⟨25696⟩)
    (coefficientTransfer := 21980) (summaryTransfer := 21982)
    (rightCoefficientProducer := 21912)
    (rightSummaryTransfer := 21981)
    (leftMaximum := ⟨95468672, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge21983.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority21912.actual selector witness)
    (summaryMagnitude := LeftBound21982.actual selector witness)
    (reconstruction := LeftOperatorMerge21983.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21977.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21913.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21912.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority21912.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge21983.operationAgreement
  · exact LeftBound21982.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21983.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 21984 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25696⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23380⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23380⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge21983.working
    [{ coefficient := (-1), key := LeftRelationMerge21984.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge21984.frameStart
      LeftRelationMerge21984.owner (.relation 21984) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge21984.deltas
    rows := LeftRelationMerge21984.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge21983.working LeftRelationMerge21984.source
        (relationContext LeftRelationMerge21984.source
          LeftRelationMerge21984.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge21983.working, LeftRelationMerge21984.deltas,
    LeftRelationMerge21984.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 21984)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25697⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10255⟩⟩, ⟨.program ⟨214⟩, ⟨13178⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25696⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25696⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge21983.working) (working := relationWorking0)
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
end SemanticResult21987

namespace SemanticResult21990
def owner : Owner := ⟨.program ⟨214⟩, ⟨20188⟩⟩
def rawTerms : List Term := Proof.Events085.exact21990RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21990
def producerEvent : Nat := 21989
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21990.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨25⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨25⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult21990

namespace SemanticResult21994
def owner : Owner := ⟨.program ⟨214⟩, ⟨20190⟩⟩
def rawTerms : List Term := Proof.Events085.exact21994RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21994
def producerEvent : Nat := 21993
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21994.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 21991 .coefficient) (.value (.predecessor 1 21992 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 21991 .coefficient) (.value (.predecessor 1 21992 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult21994

namespace SemanticResult22072
def owner : Owner := ⟨.program ⟨214⟩, ⟨13178⟩⟩
def rawTerms : List Term := Proof.Events086.exact22072RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 22072
def producerEvent : Nat := 22071
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22072.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 22049, .finite 58, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult22072

namespace SemanticResult22075
def owner : Owner := ⟨.program ⟨214⟩, ⟨10255⟩⟩
def rawTerms : List Term := Proof.Events086.exact22075RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 22075
def producerEvent : Nat := 22074
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22075.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 22049, .finite 58, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult22075

namespace SemanticResult22080
def owner : Owner := ⟨.program ⟨214⟩, ⟨13179⟩⟩
def rawTerms : List Term := Proof.Events086.exact22080RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 22080
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22080.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge22079.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge22079.frameStart)
    (transferEvent := 22078) (owner := owner)
    (leftResult := 22075) (rightResult := 22072)
    (working := LeftOperatorMerge22079.working)
    (reconstruction := LeftOperatorMerge22079.reconstruction)
    (leftReference := .predecessor 0 22076 .coefficient) (rightReference := .predecessor 1 22077 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult22075.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult22072.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge22079.operationAgreement
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
end SemanticResult22080

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
