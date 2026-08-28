import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard685
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard037
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard077
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard684

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult96561
def owner : Owner := ⟨.program ⟨214⟩, ⟨7122⟩⟩
def rawTerms : List Term := Proof.Events377.exact96561RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96561
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96561.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96560.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge96560.frameStart)
    (transferEvent := 96559) (owner := owner)
    (leftResult := 27) (rightResult := 8977)
    (working := LeftOperatorMerge96560.working)
    (reconstruction := LeftOperatorMerge96560.reconstruction)
    (leftReference := .predecessor 0 96557 .coefficient) (rightReference := .predecessor 1 96558 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8977.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge96560.operationAgreement
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
end SemanticResult96561

namespace SemanticResult96565
def owner : Owner := ⟨.program ⟨214⟩, ⟨12350⟩⟩
def rawTerms : List Term := Proof.Events377.exact96565RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96565
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96565.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 96562) (rightBinding := 96563)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7122⟩) (rightExpression := ⟨12349⟩)
    (transferEvent := 96564)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96561.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult96556.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult96565

namespace SemanticResult96571
def owner : Owner := ⟨.program ⟨214⟩, ⟨12351⟩⟩
def rawTerms : List Term := Proof.Events377.exact96571RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 96571
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96571.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 96568) (survivorTransfer := 96569)
    (survivorEvent := 96570) (resultEvent := resultEvent)
    (rightCoefficientProducer := 8968)
    (owner := owner) (leftOwner := SemanticResult96565.owner)
    (rightOwner := SemanticResult8969.owner)
    (leftResult := 96565) (rightResult := 8969)
    (leftBinding := 96566) (rightBinding := 96567)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12350⟩) (rightExpression := ⟨99⟩)
    (leftActual := SemanticResult96565.actual selector witness)
    (rightActual := SemanticResult8969.actual selector witness)
    (leftRaw := SemanticResult96565.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨99⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound8968.actual selector witness)
    (survivorMagnitude := LeftBound96569.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96565.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8969.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8968.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8968.derived selector witness)
  · exact LeftBound96569.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult96571

namespace SemanticResult96579
def owner : Owner := ⟨.program ⟨214⟩, ⟨12352⟩⟩
def rawTerms : List Term := Proof.Events377.exact96579RawTerms
def summary : Bound := (.finite 33280)
def resultEvent : Nat := 96579
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96579.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨40, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96577.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge96577.frameStart)
    (owner := owner) (leftOwner := SemanticResult96571.owner)
    (rightOwner := SemanticResult4684.owner)
    (leftResult := 96571) (rightResult := 4684)
    (leftActual := SemanticResult96571.actual selector witness)
    (rightActual := SemanticResult4684.actual selector witness)
    (leftRaw := SemanticResult96571.rawTerms)
    (rightRaw := SemanticResult4684.rawTerms)
    (working := LeftOperatorMerge96577.working)
    (leftBinding := 96572) (rightBinding := 96573)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12351⟩) (rightExpression := ⟨9805⟩)
    (coefficientTransfer := 96574) (summaryTransfer := 96576)
    (rightCoefficientProducer := 4683)
    (rightSummaryTransfer := 96575)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨40, by decide⟩)
    (rightRecordedMaximum := 40)
    (rightSummaryMaximum := ⟨40, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge96577.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority4683.actual selector witness)
    (summaryMagnitude := LeftBound96576.actual selector witness)
    (reconstruction := LeftOperatorMerge96577.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96571.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4684.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4683.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority4683.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge96577.operationAgreement
  · exact LeftBound96576.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96577.working summary) := by
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
end SemanticResult96579

namespace SemanticResult96584
def owner : Owner := ⟨.program ⟨214⟩, ⟨9806⟩⟩
def rawTerms : List Term := Proof.Events377.exact96584RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96584
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96584.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96583.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge96583.frameStart)
    (transferEvent := 96582) (owner := owner)
    (leftResult := 4684) (rightResult := 32)
    (working := LeftOperatorMerge96583.working)
    (reconstruction := LeftOperatorMerge96583.reconstruction)
    (leftReference := .predecessor 0 96580 .coefficient) (rightReference := .predecessor 1 96581 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4684.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge96583.operationAgreement
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
end SemanticResult96584

namespace SemanticResult96589
def owner : Owner := ⟨.program ⟨214⟩, ⟨7102⟩⟩
def rawTerms : List Term := Proof.Events377.exact96589RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96589
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96589.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96588.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge96588.frameStart)
    (transferEvent := 96587) (owner := owner)
    (leftResult := 27) (rightResult := 9018)
    (working := LeftOperatorMerge96588.working)
    (reconstruction := LeftOperatorMerge96588.reconstruction)
    (leftReference := .predecessor 0 96585 .coefficient) (rightReference := .predecessor 1 96586 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9018.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge96588.operationAgreement
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
end SemanticResult96589

namespace SemanticResult96593
def owner : Owner := ⟨.program ⟨214⟩, ⟨9807⟩⟩
def rawTerms : List Term := Proof.Events377.exact96593RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96593
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96593.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 96590) (rightBinding := 96591)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7102⟩) (rightExpression := ⟨9806⟩)
    (transferEvent := 96592)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96589.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult96584.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult96593

namespace SemanticResult96599
def owner : Owner := ⟨.program ⟨214⟩, ⟨9808⟩⟩
def rawTerms : List Term := Proof.Events377.exact96599RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 96599
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96599.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 96596) (survivorTransfer := 96597)
    (survivorEvent := 96598) (resultEvent := resultEvent)
    (rightCoefficientProducer := 9009)
    (owner := owner) (leftOwner := SemanticResult96593.owner)
    (rightOwner := SemanticResult9010.owner)
    (leftResult := 96593) (rightResult := 9010)
    (leftBinding := 96594) (rightBinding := 96595)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9807⟩) (rightExpression := ⟨79⟩)
    (leftActual := SemanticResult96593.actual selector witness)
    (rightActual := SemanticResult9010.actual selector witness)
    (leftRaw := SemanticResult96593.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨79⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound9009.actual selector witness)
    (survivorMagnitude := LeftBound96597.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96593.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9010.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9009.derived selector witness)
  · exact LeftBound96597.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult96599

namespace SemanticResult96609
def owner : Owner := ⟨.program ⟨214⟩, ⟨9809⟩⟩
def rawTerms : List Term := Proof.Events377.exact96609RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 96609
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96609.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96605.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge96605.frameStart)
    (owner := owner) (leftOwner := SemanticResult96599.owner)
    (rightOwner := SemanticResult9007.owner)
    (leftResult := 96599) (rightResult := 9007)
    (leftActual := SemanticResult96599.actual selector witness)
    (rightActual := SemanticResult9007.actual selector witness)
    (leftRaw := SemanticResult96599.rawTerms)
    (rightRaw := SemanticResult9007.rawTerms)
    (working := LeftOperatorMerge96605.working)
    (leftBinding := 96600) (rightBinding := 96601)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9808⟩) (rightExpression := ⟨7868⟩)
    (coefficientTransfer := 96602) (summaryTransfer := 96604)
    (rightCoefficientProducer := 9006)
    (rightSummaryTransfer := 96603)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge96605.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound9006.actual selector witness)
    (summaryMagnitude := LeftBound96604.actual selector witness)
    (reconstruction := LeftOperatorMerge96605.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96599.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9007.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9006.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound9006.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge96605.operationAgreement
  · exact LeftBound96604.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96605.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 96606 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9805⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge96605.working
    [{ coefficient := (-1), key := LeftRelationMerge96606.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge96606.frameStart
      LeftRelationMerge96606.owner (.relation 96606) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge96606.deltas
    rows := LeftRelationMerge96606.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge96605.working LeftRelationMerge96606.source
        (relationContext LeftRelationMerge96606.source
          LeftRelationMerge96606.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge96605.working, LeftRelationMerge96606.deltas,
    LeftRelationMerge96606.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 96606)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9809⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9805⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge96605.working) (working := relationWorking0)
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
end SemanticResult96609

namespace SemanticResult96615
def owner : Owner := ⟨.program ⟨214⟩, ⟨12353⟩⟩
def rawTerms : List Term := Proof.Events377.exact96615RawTerms
def summary : Bound := (.finite 95453696)
def resultEvent : Nat := 96615
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96615.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge96613.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult96609.owner)
    (rightOwner := SemanticResult96579.owner)
    (leftResult := 96609) (rightResult := 96579)
    (leftActual := SemanticResult96609.actual selector witness)
    (rightActual := SemanticResult96579.actual selector witness)
    (leftRaw := SemanticResult96609.rawTerms)
    (rightRaw := SemanticResult96579.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 33280) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 96610) (rightBinding := 96611)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9809⟩) (rightExpression := ⟨12352⟩)
    (coefficientTransfer := 96612) (summaryTransfer := 96614)
    (base := LeftOperatorMerge96613.base)
    (reconstruction := LeftOperatorMerge96613.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96609.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult96579.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge96613.operationAgreement
  · rfl
  · decide
end SemanticResult96615

namespace SemanticResult96625
def owner : Owner := ⟨.program ⟨214⟩, ⟨25361⟩⟩
def rawTerms : List Term := Proof.Events377.exact96625RawTerms
def summary : Bound := (.finite 350316591579136)
def resultEvent : Nat := 96625
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96625.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95453696, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96621.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge96621.frameStart)
    (owner := owner) (leftOwner := SemanticResult96615.owner)
    (rightOwner := SemanticResult96551.owner)
    (leftResult := 96615) (rightResult := 96551)
    (leftActual := SemanticResult96615.actual selector witness)
    (rightActual := SemanticResult96551.actual selector witness)
    (leftRaw := SemanticResult96615.rawTerms)
    (rightRaw := SemanticResult96551.rawTerms)
    (working := LeftOperatorMerge96621.working)
    (leftBinding := 96616) (rightBinding := 96617)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12353⟩) (rightExpression := ⟨25360⟩)
    (coefficientTransfer := 96618) (summaryTransfer := 96620)
    (rightCoefficientProducer := 96550)
    (rightSummaryTransfer := 96619)
    (leftMaximum := ⟨95453696, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge96621.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority96550.actual selector witness)
    (summaryMagnitude := LeftBound96620.actual selector witness)
    (reconstruction := LeftOperatorMerge96621.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96615.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult96551.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96550.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority96550.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge96621.operationAgreement
  · exact LeftBound96620.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96621.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 96622 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23200⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23200⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge96621.working
    [{ coefficient := (-1), key := LeftRelationMerge96622.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge96622.frameStart
      LeftRelationMerge96622.owner (.relation 96622) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge96622.deltas
    rows := LeftRelationMerge96622.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge96621.working LeftRelationMerge96622.source
        (relationContext LeftRelationMerge96622.source
          LeftRelationMerge96622.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge96621.working, LeftRelationMerge96622.deltas,
    LeftRelationMerge96622.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 96622)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25361⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge96621.working) (working := relationWorking0)
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
end SemanticResult96625

namespace SemanticResult96628
def owner : Owner := ⟨.program ⟨214⟩, ⟨19877⟩⟩
def rawTerms : List Term := Proof.Events377.exact96628RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96628
def producerEvent : Nat := 96627
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96628.actual selector witness
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
end SemanticResult96628

namespace SemanticResult96632
def owner : Owner := ⟨.program ⟨214⟩, ⟨19879⟩⟩
def rawTerms : List Term := Proof.Events377.exact96632RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96632
def producerEvent : Nat := 96631
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96632.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 96629 .coefficient) (.value (.predecessor 1 96630 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 96629 .coefficient) (.value (.predecessor 1 96630 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult96632

namespace SemanticResult96686
def owner : Owner := ⟨.program ⟨214⟩, ⟨12346⟩⟩
def rawTerms : List Term := Proof.Events377.exact96686RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96686
def producerEvent : Nat := 96685
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96686.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 96675, .finite 40, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult96686

namespace SemanticResult96689
def owner : Owner := ⟨.program ⟨214⟩, ⟨9805⟩⟩
def rawTerms : List Term := Proof.Events377.exact96689RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96689
def producerEvent : Nat := 96688
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96689.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 96675, .finite 40, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult96689

namespace SemanticResult96694
def owner : Owner := ⟨.program ⟨214⟩, ⟨12347⟩⟩
def rawTerms : List Term := Proof.Events377.exact96694RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96694
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96694.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96693.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge96693.frameStart)
    (transferEvent := 96692) (owner := owner)
    (leftResult := 96689) (rightResult := 96686)
    (working := LeftOperatorMerge96693.working)
    (reconstruction := LeftOperatorMerge96693.reconstruction)
    (leftReference := .predecessor 0 96690 .coefficient) (rightReference := .predecessor 1 96691 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult96689.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult96686.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge96693.operationAgreement
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
end SemanticResult96694

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
