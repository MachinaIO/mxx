import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard618
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard033
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard113
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard114
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard566

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult86648
def owner : Owner := ⟨.program ⟨214⟩, ⟨23164⟩⟩
def rawTerms : List Term := Proof.Events338.exact86648RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 86648
def producerEvent : Nat := 86647
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86648.actual selector witness
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
end SemanticResult86648

namespace SemanticResult86651
def owner : Owner := ⟨.program ⟨214⟩, ⟨25296⟩⟩
def rawTerms : List Term := Proof.Events338.exact86651RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 86651
def producerEvent : Nat := 86650
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86651.actual selector witness
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
end SemanticResult86651

namespace SemanticResult86656
def owner : Owner := ⟨.program ⟨214⟩, ⟨11134⟩⟩
def rawTerms : List Term := Proof.Events338.exact86656RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 86656
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86656.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge86655.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge86655.frameStart)
    (transferEvent := 86654) (owner := owner)
    (leftResult := 4150) (rightResult := 79920)
    (working := LeftOperatorMerge86655.working)
    (reconstruction := LeftOperatorMerge86655.reconstruction)
    (leftReference := .predecessor 0 86652 .coefficient) (rightReference := .predecessor 1 86653 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4150.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79920.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge86655.operationAgreement
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
end SemanticResult86656

namespace SemanticResult86661
def owner : Owner := ⟨.program ⟨214⟩, ⟨7231⟩⟩
def rawTerms : List Term := Proof.Events338.exact86661RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 86661
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86661.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge86660.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge86660.frameStart)
    (transferEvent := 86659) (owner := owner)
    (leftResult := 79790) (rightResult := 13486)
    (working := LeftOperatorMerge86660.working)
    (reconstruction := LeftOperatorMerge86660.reconstruction)
    (leftReference := .predecessor 0 86657 .coefficient) (rightReference := .predecessor 1 86658 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13486.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge86660.operationAgreement
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
end SemanticResult86661

namespace SemanticResult86665
def owner : Owner := ⟨.program ⟨214⟩, ⟨11135⟩⟩
def rawTerms : List Term := Proof.Events338.exact86665RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 86665
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86665.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 86662) (rightBinding := 86663)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7231⟩) (rightExpression := ⟨11134⟩)
    (transferEvent := 86664)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult86661.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult86656.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult86665

namespace SemanticResult86671
def owner : Owner := ⟨.program ⟨214⟩, ⟨11136⟩⟩
def rawTerms : List Term := Proof.Events338.exact86671RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 86671
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86671.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 86668) (survivorTransfer := 86669)
    (survivorEvent := 86670) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13477)
    (owner := owner) (leftOwner := SemanticResult86665.owner)
    (rightOwner := SemanticResult13478.owner)
    (leftResult := 86665) (rightResult := 13478)
    (leftBinding := 86666) (rightBinding := 86667)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11135⟩) (rightExpression := ⟨89⟩)
    (leftActual := SemanticResult86665.actual selector witness)
    (rightActual := SemanticResult13478.actual selector witness)
    (leftRaw := SemanticResult86665.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨89⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13477.actual selector witness)
    (survivorMagnitude := LeftBound86669.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult86665.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13478.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13477.derived selector witness)
  · exact LeftBound86669.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult86671

namespace SemanticResult86679
def owner : Owner := ⟨.program ⟨214⟩, ⟨12166⟩⟩
def rawTerms : List Term := Proof.Events338.exact86679RawTerms
def summary : Bound := (.finite 4992)
def resultEvent : Nat := 86679
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86679.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨6, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge86677.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge86677.frameStart)
    (owner := owner) (leftOwner := SemanticResult86671.owner)
    (rightOwner := SemanticResult4153.owner)
    (leftResult := 86671) (rightResult := 4153)
    (leftActual := SemanticResult86671.actual selector witness)
    (rightActual := SemanticResult4153.actual selector witness)
    (leftRaw := SemanticResult86671.rawTerms)
    (rightRaw := SemanticResult4153.rawTerms)
    (working := LeftOperatorMerge86677.working)
    (leftBinding := 86672) (rightBinding := 86673)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11136⟩) (rightExpression := ⟨12163⟩)
    (coefficientTransfer := 86674) (summaryTransfer := 86676)
    (rightCoefficientProducer := 4152)
    (rightSummaryTransfer := 86675)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨6, by decide⟩)
    (rightRecordedMaximum := 6)
    (rightSummaryMaximum := ⟨6, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge86677.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority4152.actual selector witness)
    (summaryMagnitude := LeftBound86676.actual selector witness)
    (reconstruction := LeftOperatorMerge86677.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult86671.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4153.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4152.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority4152.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge86677.operationAgreement
  · exact LeftBound86676.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge86677.working summary) := by
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
end SemanticResult86679

namespace SemanticResult86684
def owner : Owner := ⟨.program ⟨214⟩, ⟨12167⟩⟩
def rawTerms : List Term := Proof.Events338.exact86684RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 86684
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86684.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge86683.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge86683.frameStart)
    (transferEvent := 86682) (owner := owner)
    (leftResult := 4153) (rightResult := 79920)
    (working := LeftOperatorMerge86683.working)
    (reconstruction := LeftOperatorMerge86683.reconstruction)
    (leftReference := .predecessor 0 86680 .coefficient) (rightReference := .predecessor 1 86681 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4153.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79920.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge86683.operationAgreement
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
end SemanticResult86684

namespace SemanticResult86689
def owner : Owner := ⟨.program ⟨214⟩, ⟨7248⟩⟩
def rawTerms : List Term := Proof.Events338.exact86689RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 86689
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86689.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge86688.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge86688.frameStart)
    (transferEvent := 86687) (owner := owner)
    (leftResult := 79790) (rightResult := 13527)
    (working := LeftOperatorMerge86688.working)
    (reconstruction := LeftOperatorMerge86688.reconstruction)
    (leftReference := .predecessor 0 86685 .coefficient) (rightReference := .predecessor 1 86686 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13527.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge86688.operationAgreement
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
end SemanticResult86689

namespace SemanticResult86693
def owner : Owner := ⟨.program ⟨214⟩, ⟨12168⟩⟩
def rawTerms : List Term := Proof.Events338.exact86693RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 86693
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86693.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 86690) (rightBinding := 86691)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7248⟩) (rightExpression := ⟨12167⟩)
    (transferEvent := 86692)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult86689.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult86684.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult86693

namespace SemanticResult86699
def owner : Owner := ⟨.program ⟨214⟩, ⟨12169⟩⟩
def rawTerms : List Term := Proof.Events338.exact86699RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 86699
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86699.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 86696) (survivorTransfer := 86697)
    (survivorEvent := 86698) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13518)
    (owner := owner) (leftOwner := SemanticResult86693.owner)
    (rightOwner := SemanticResult13519.owner)
    (leftResult := 86693) (rightResult := 13519)
    (leftBinding := 86694) (rightBinding := 86695)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12168⟩) (rightExpression := ⟨106⟩)
    (leftActual := SemanticResult86693.actual selector witness)
    (rightActual := SemanticResult13519.actual selector witness)
    (leftRaw := SemanticResult86693.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨106⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13518.actual selector witness)
    (survivorMagnitude := LeftBound86697.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult86693.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13519.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13518.derived selector witness)
  · exact LeftBound86697.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult86699

namespace SemanticResult86709
def owner : Owner := ⟨.program ⟨214⟩, ⟨12170⟩⟩
def rawTerms : List Term := Proof.Events338.exact86709RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 86709
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86709.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge86705.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge86705.frameStart)
    (owner := owner) (leftOwner := SemanticResult86699.owner)
    (rightOwner := SemanticResult13516.owner)
    (leftResult := 86699) (rightResult := 13516)
    (leftActual := SemanticResult86699.actual selector witness)
    (rightActual := SemanticResult13516.actual selector witness)
    (leftRaw := SemanticResult86699.rawTerms)
    (rightRaw := SemanticResult13516.rawTerms)
    (working := LeftOperatorMerge86705.working)
    (leftBinding := 86700) (rightBinding := 86701)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12169⟩) (rightExpression := ⟨7841⟩)
    (coefficientTransfer := 86702) (summaryTransfer := 86704)
    (rightCoefficientProducer := 13515)
    (rightSummaryTransfer := 86703)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge86705.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound13515.actual selector witness)
    (summaryMagnitude := LeftBound86704.actual selector witness)
    (reconstruction := LeftOperatorMerge86705.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult86699.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13516.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13515.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound13515.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge86705.operationAgreement
  · exact LeftBound86704.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge86705.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 86706 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge86705.working
    [{ coefficient := (-1), key := LeftRelationMerge86706.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge86706.frameStart
      LeftRelationMerge86706.owner (.relation 86706) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge86706.deltas
    rows := LeftRelationMerge86706.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge86705.working LeftRelationMerge86706.source
        (relationContext LeftRelationMerge86706.source
          LeftRelationMerge86706.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge86705.working, LeftRelationMerge86706.deltas,
    LeftRelationMerge86706.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 86706)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨12170⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge86705.working) (working := relationWorking0)
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
end SemanticResult86709

namespace SemanticResult86715
def owner : Owner := ⟨.program ⟨214⟩, ⟨12171⟩⟩
def rawTerms : List Term := Proof.Events338.exact86715RawTerms
def summary : Bound := (.finite 95425408)
def resultEvent : Nat := 86715
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86715.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge86713.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult86709.owner)
    (rightOwner := SemanticResult86679.owner)
    (leftResult := 86709) (rightResult := 86679)
    (leftActual := SemanticResult86709.actual selector witness)
    (rightActual := SemanticResult86679.actual selector witness)
    (leftRaw := SemanticResult86709.rawTerms)
    (rightRaw := SemanticResult86679.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 4992) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 86710) (rightBinding := 86711)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12170⟩) (rightExpression := ⟨12166⟩)
    (coefficientTransfer := 86712) (summaryTransfer := 86714)
    (base := LeftOperatorMerge86713.base)
    (reconstruction := LeftOperatorMerge86713.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult86709.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult86679.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge86713.operationAgreement
  · rfl
  · decide
end SemanticResult86715

namespace SemanticResult86725
def owner : Owner := ⟨.program ⟨214⟩, ⟨25297⟩⟩
def rawTerms : List Term := Proof.Events338.exact86725RawTerms
def summary : Bound := (.finite 350212774166528)
def resultEvent : Nat := 86725
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86725.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95425408, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge86721.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge86721.frameStart)
    (owner := owner) (leftOwner := SemanticResult86715.owner)
    (rightOwner := SemanticResult86651.owner)
    (leftResult := 86715) (rightResult := 86651)
    (leftActual := SemanticResult86715.actual selector witness)
    (rightActual := SemanticResult86651.actual selector witness)
    (leftRaw := SemanticResult86715.rawTerms)
    (rightRaw := SemanticResult86651.rawTerms)
    (working := LeftOperatorMerge86721.working)
    (leftBinding := 86716) (rightBinding := 86717)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12171⟩) (rightExpression := ⟨25296⟩)
    (coefficientTransfer := 86718) (summaryTransfer := 86720)
    (rightCoefficientProducer := 86650)
    (rightSummaryTransfer := 86719)
    (leftMaximum := ⟨95425408, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge86721.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority86650.actual selector witness)
    (summaryMagnitude := LeftBound86720.actual selector witness)
    (reconstruction := LeftOperatorMerge86721.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult86715.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult86651.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86650.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority86650.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge86721.operationAgreement
  · exact LeftBound86720.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge86721.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 86722 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23164⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23164⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge86721.working
    [{ coefficient := (-1), key := LeftRelationMerge86722.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge86722.frameStart
      LeftRelationMerge86722.owner (.relation 86722) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge86722.deltas
    rows := LeftRelationMerge86722.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge86721.working LeftRelationMerge86722.source
        (relationContext LeftRelationMerge86722.source
          LeftRelationMerge86722.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge86721.working, LeftRelationMerge86722.deltas,
    LeftRelationMerge86722.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 86722)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25297⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11133⟩⟩, ⟨.program ⟨214⟩, ⟨12163⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25296⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge86721.working) (working := relationWorking0)
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
end SemanticResult86725

namespace SemanticResult86728
def owner : Owner := ⟨.program ⟨214⟩, ⟨19240⟩⟩
def rawTerms : List Term := Proof.Events338.exact86728RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 86728
def producerEvent : Nat := 86727
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86728.actual selector witness
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
end SemanticResult86728

namespace SemanticResult86732
def owner : Owner := ⟨.program ⟨214⟩, ⟨19242⟩⟩
def rawTerms : List Term := Proof.Events338.exact86732RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 86732
def producerEvent : Nat := 86731
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult86732.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 86729 .coefficient) (.value (.predecessor 1 86730 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 86729 .coefficient) (.value (.predecessor 1 86730 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult86732

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
