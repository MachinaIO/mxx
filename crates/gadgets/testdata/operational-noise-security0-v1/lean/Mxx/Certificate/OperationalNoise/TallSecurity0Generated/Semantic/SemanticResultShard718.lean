import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard718
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard039
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard113
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard114

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult100454
def owner : Owner := ⟨.program ⟨214⟩, ⟨23158⟩⟩
def rawTerms : List Term := Proof.Events392.exact100454RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100454
def producerEvent : Nat := 100453
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100454.actual selector witness
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
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult100454

namespace SemanticResult100457
def owner : Owner := ⟨.program ⟨214⟩, ⟨25283⟩⟩
def rawTerms : List Term := Proof.Events392.exact100457RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100457
def producerEvent : Nat := 100456
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100457.actual selector witness
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
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult100457

namespace SemanticResult100462
def owner : Owner := ⟨.program ⟨214⟩, ⟨11122⟩⟩
def rawTerms : List Term := Proof.Events392.exact100462RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100462
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100462.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100461.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge100461.frameStart)
    (transferEvent := 100460) (owner := owner)
    (leftResult := 4888) (rightResult := 32)
    (working := LeftOperatorMerge100461.working)
    (reconstruction := LeftOperatorMerge100461.reconstruction)
    (leftReference := .predecessor 0 100458 .coefficient) (rightReference := .predecessor 1 100459 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4888.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge100461.operationAgreement
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
end SemanticResult100462

namespace SemanticResult100467
def owner : Owner := ⟨.program ⟨214⟩, ⟨7112⟩⟩
def rawTerms : List Term := Proof.Events392.exact100467RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100467
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100467.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100466.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge100466.frameStart)
    (transferEvent := 100465) (owner := owner)
    (leftResult := 27) (rightResult := 13486)
    (working := LeftOperatorMerge100466.working)
    (reconstruction := LeftOperatorMerge100466.reconstruction)
    (leftReference := .predecessor 0 100463 .coefficient) (rightReference := .predecessor 1 100464 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13486.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge100466.operationAgreement
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
end SemanticResult100467

namespace SemanticResult100471
def owner : Owner := ⟨.program ⟨214⟩, ⟨11123⟩⟩
def rawTerms : List Term := Proof.Events392.exact100471RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100471
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100471.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100468) (rightBinding := 100469)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7112⟩) (rightExpression := ⟨11122⟩)
    (transferEvent := 100470)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100467.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100462.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100471

namespace SemanticResult100477
def owner : Owner := ⟨.program ⟨214⟩, ⟨11124⟩⟩
def rawTerms : List Term := Proof.Events392.exact100477RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 100477
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100477.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 100474) (survivorTransfer := 100475)
    (survivorEvent := 100476) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13477)
    (owner := owner) (leftOwner := SemanticResult100471.owner)
    (rightOwner := SemanticResult13478.owner)
    (leftResult := 100471) (rightResult := 13478)
    (leftBinding := 100472) (rightBinding := 100473)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11123⟩) (rightExpression := ⟨89⟩)
    (leftActual := SemanticResult100471.actual selector witness)
    (rightActual := SemanticResult13478.actual selector witness)
    (leftRaw := SemanticResult100471.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨89⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13477.actual selector witness)
    (survivorMagnitude := LeftBound100475.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100471.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13478.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13477.derived selector witness)
  · exact LeftBound100475.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult100477

namespace SemanticResult100485
def owner : Owner := ⟨.program ⟨214⟩, ⟨12139⟩⟩
def rawTerms : List Term := Proof.Events392.exact100485RawTerms
def summary : Bound := (.finite 4992)
def resultEvent : Nat := 100485
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100485.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨6, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100483.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge100483.frameStart)
    (owner := owner) (leftOwner := SemanticResult100477.owner)
    (rightOwner := SemanticResult4891.owner)
    (leftResult := 100477) (rightResult := 4891)
    (leftActual := SemanticResult100477.actual selector witness)
    (rightActual := SemanticResult4891.actual selector witness)
    (leftRaw := SemanticResult100477.rawTerms)
    (rightRaw := SemanticResult4891.rawTerms)
    (working := LeftOperatorMerge100483.working)
    (leftBinding := 100478) (rightBinding := 100479)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11124⟩) (rightExpression := ⟨12136⟩)
    (coefficientTransfer := 100480) (summaryTransfer := 100482)
    (rightCoefficientProducer := 4890)
    (rightSummaryTransfer := 100481)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨6, by decide⟩)
    (rightRecordedMaximum := 6)
    (rightSummaryMaximum := ⟨6, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge100483.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority4890.actual selector witness)
    (summaryMagnitude := LeftBound100482.actual selector witness)
    (reconstruction := LeftOperatorMerge100483.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100477.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4891.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4890.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority4890.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge100483.operationAgreement
  · exact LeftBound100482.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100483.working summary) := by
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
end SemanticResult100485

namespace SemanticResult100490
def owner : Owner := ⟨.program ⟨214⟩, ⟨12140⟩⟩
def rawTerms : List Term := Proof.Events392.exact100490RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100490
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100490.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100489.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge100489.frameStart)
    (transferEvent := 100488) (owner := owner)
    (leftResult := 4891) (rightResult := 32)
    (working := LeftOperatorMerge100489.working)
    (reconstruction := LeftOperatorMerge100489.reconstruction)
    (leftReference := .predecessor 0 100486 .coefficient) (rightReference := .predecessor 1 100487 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4891.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge100489.operationAgreement
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
end SemanticResult100490

namespace SemanticResult100495
def owner : Owner := ⟨.program ⟨214⟩, ⟨7129⟩⟩
def rawTerms : List Term := Proof.Events392.exact100495RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100495
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100495.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100494.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge100494.frameStart)
    (transferEvent := 100493) (owner := owner)
    (leftResult := 27) (rightResult := 13527)
    (working := LeftOperatorMerge100494.working)
    (reconstruction := LeftOperatorMerge100494.reconstruction)
    (leftReference := .predecessor 0 100491 .coefficient) (rightReference := .predecessor 1 100492 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13527.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge100494.operationAgreement
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
end SemanticResult100495

namespace SemanticResult100499
def owner : Owner := ⟨.program ⟨214⟩, ⟨12141⟩⟩
def rawTerms : List Term := Proof.Events392.exact100499RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100499
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100499.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100496) (rightBinding := 100497)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7129⟩) (rightExpression := ⟨12140⟩)
    (transferEvent := 100498)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100495.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100490.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100499

namespace SemanticResult100505
def owner : Owner := ⟨.program ⟨214⟩, ⟨12142⟩⟩
def rawTerms : List Term := Proof.Events392.exact100505RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 100505
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100505.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 100502) (survivorTransfer := 100503)
    (survivorEvent := 100504) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13518)
    (owner := owner) (leftOwner := SemanticResult100499.owner)
    (rightOwner := SemanticResult13519.owner)
    (leftResult := 100499) (rightResult := 13519)
    (leftBinding := 100500) (rightBinding := 100501)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12141⟩) (rightExpression := ⟨106⟩)
    (leftActual := SemanticResult100499.actual selector witness)
    (rightActual := SemanticResult13519.actual selector witness)
    (leftRaw := SemanticResult100499.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨106⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13518.actual selector witness)
    (survivorMagnitude := LeftBound100503.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100499.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13519.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13518.derived selector witness)
  · exact LeftBound100503.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult100505

namespace SemanticResult100515
def owner : Owner := ⟨.program ⟨214⟩, ⟨12143⟩⟩
def rawTerms : List Term := Proof.Events392.exact100515RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 100515
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100515.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100511.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge100511.frameStart)
    (owner := owner) (leftOwner := SemanticResult100505.owner)
    (rightOwner := SemanticResult13516.owner)
    (leftResult := 100505) (rightResult := 13516)
    (leftActual := SemanticResult100505.actual selector witness)
    (rightActual := SemanticResult13516.actual selector witness)
    (leftRaw := SemanticResult100505.rawTerms)
    (rightRaw := SemanticResult13516.rawTerms)
    (working := LeftOperatorMerge100511.working)
    (leftBinding := 100506) (rightBinding := 100507)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12142⟩) (rightExpression := ⟨7841⟩)
    (coefficientTransfer := 100508) (summaryTransfer := 100510)
    (rightCoefficientProducer := 13515)
    (rightSummaryTransfer := 100509)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge100511.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound13515.actual selector witness)
    (summaryMagnitude := LeftBound100510.actual selector witness)
    (reconstruction := LeftOperatorMerge100511.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100505.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13516.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13515.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound13515.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge100511.operationAgreement
  · exact LeftBound100510.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100511.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 100512 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge100511.working
    [{ coefficient := (-1), key := LeftRelationMerge100512.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge100512.frameStart
      LeftRelationMerge100512.owner (.relation 100512) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge100512.deltas
    rows := LeftRelationMerge100512.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge100511.working LeftRelationMerge100512.source
        (relationContext LeftRelationMerge100512.source
          LeftRelationMerge100512.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge100511.working, LeftRelationMerge100512.deltas,
    LeftRelationMerge100512.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 100512)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨12143⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge100511.working) (working := relationWorking0)
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
end SemanticResult100515

namespace SemanticResult100521
def owner : Owner := ⟨.program ⟨214⟩, ⟨12144⟩⟩
def rawTerms : List Term := Proof.Events392.exact100521RawTerms
def summary : Bound := (.finite 95425408)
def resultEvent : Nat := 100521
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100521.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge100519.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult100515.owner)
    (rightOwner := SemanticResult100485.owner)
    (leftResult := 100515) (rightResult := 100485)
    (leftActual := SemanticResult100515.actual selector witness)
    (rightActual := SemanticResult100485.actual selector witness)
    (leftRaw := SemanticResult100515.rawTerms)
    (rightRaw := SemanticResult100485.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 4992) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 100516) (rightBinding := 100517)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12143⟩) (rightExpression := ⟨12139⟩)
    (coefficientTransfer := 100518) (summaryTransfer := 100520)
    (base := LeftOperatorMerge100519.base)
    (reconstruction := LeftOperatorMerge100519.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100515.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100485.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge100519.operationAgreement
  · rfl
  · decide
end SemanticResult100521

namespace SemanticResult100531
def owner : Owner := ⟨.program ⟨214⟩, ⟨25284⟩⟩
def rawTerms : List Term := Proof.Events392.exact100531RawTerms
def summary : Bound := (.finite 350212774166528)
def resultEvent : Nat := 100531
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100531.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95425408, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100527.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge100527.frameStart)
    (owner := owner) (leftOwner := SemanticResult100521.owner)
    (rightOwner := SemanticResult100457.owner)
    (leftResult := 100521) (rightResult := 100457)
    (leftActual := SemanticResult100521.actual selector witness)
    (rightActual := SemanticResult100457.actual selector witness)
    (leftRaw := SemanticResult100521.rawTerms)
    (rightRaw := SemanticResult100457.rawTerms)
    (working := LeftOperatorMerge100527.working)
    (leftBinding := 100522) (rightBinding := 100523)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12144⟩) (rightExpression := ⟨25283⟩)
    (coefficientTransfer := 100524) (summaryTransfer := 100526)
    (rightCoefficientProducer := 100456)
    (rightSummaryTransfer := 100525)
    (leftMaximum := ⟨95425408, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge100527.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority100456.actual selector witness)
    (summaryMagnitude := LeftBound100526.actual selector witness)
    (reconstruction := LeftOperatorMerge100527.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100521.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100457.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100456.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority100456.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge100527.operationAgreement
  · exact LeftBound100526.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100527.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 100528 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23158⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23158⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge100527.working
    [{ coefficient := (-1), key := LeftRelationMerge100528.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge100528.frameStart
      LeftRelationMerge100528.owner (.relation 100528) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge100528.deltas
    rows := LeftRelationMerge100528.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge100527.working LeftRelationMerge100528.source
        (relationContext LeftRelationMerge100528.source
          LeftRelationMerge100528.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge100527.working, LeftRelationMerge100528.deltas,
    LeftRelationMerge100528.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 100528)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25284⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨11121⟩⟩, ⟨.program ⟨214⟩, ⟨12136⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25283⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge100527.working) (working := relationWorking0)
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
end SemanticResult100531

namespace SemanticResult100534
def owner : Owner := ⟨.program ⟨214⟩, ⟨19229⟩⟩
def rawTerms : List Term := Proof.Events392.exact100534RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100534
def producerEvent : Nat := 100533
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100534.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨10⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨10⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult100534

namespace SemanticResult100538
def owner : Owner := ⟨.program ⟨214⟩, ⟨19231⟩⟩
def rawTerms : List Term := Proof.Events392.exact100538RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100538
def producerEvent : Nat := 100537
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100538.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 100535 .coefficient) (.value (.predecessor 1 100536 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 100535 .coefficient) (.value (.predecessor 1 100536 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult100538

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
