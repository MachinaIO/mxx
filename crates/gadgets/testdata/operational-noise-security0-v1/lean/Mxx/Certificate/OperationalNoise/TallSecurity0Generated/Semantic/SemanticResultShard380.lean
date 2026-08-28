import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard380
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard073
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard365

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult52606
def owner : Owner := ⟨.program ⟨214⟩, ⟨23250⟩⟩
def rawTerms : List Term := Proof.Events205.exact52606RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 52606
def producerEvent : Nat := 52605
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52606.actual selector witness
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
end SemanticResult52606

namespace SemanticResult52609
def owner : Owner := ⟨.program ⟨214⟩, ⟨25455⟩⟩
def rawTerms : List Term := Proof.Events205.exact52609RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 52609
def producerEvent : Nat := 52608
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52609.actual selector witness
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
end SemanticResult52609

namespace SemanticResult52614
def owner : Owner := ⟨.program ⟨214⟩, ⟨12577⟩⟩
def rawTerms : List Term := Proof.Events205.exact52614RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 52614
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52614.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge52613.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge52613.frameStart)
    (transferEvent := 52612) (owner := owner)
    (leftResult := 2430) (rightResult := 50670)
    (working := LeftOperatorMerge52613.working)
    (reconstruction := LeftOperatorMerge52613.reconstruction)
    (leftReference := .predecessor 0 52610 .coefficient) (rightReference := .predecessor 1 52611 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2430.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge52613.operationAgreement
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
end SemanticResult52614

namespace SemanticResult52619
def owner : Owner := ⟨.program ⟨214⟩, ⟨7280⟩⟩
def rawTerms : List Term := Proof.Events205.exact52619RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 52619
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52619.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge52618.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge52618.frameStart)
    (transferEvent := 52617) (owner := owner)
    (leftResult := 50540) (rightResult := 8476)
    (working := LeftOperatorMerge52618.working)
    (reconstruction := LeftOperatorMerge52618.reconstruction)
    (leftReference := .predecessor 0 52615 .coefficient) (rightReference := .predecessor 1 52616 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8476.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge52618.operationAgreement
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
end SemanticResult52619

namespace SemanticResult52623
def owner : Owner := ⟨.program ⟨214⟩, ⟨12578⟩⟩
def rawTerms : List Term := Proof.Events205.exact52623RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 52623
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52623.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 52620) (rightBinding := 52621)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7280⟩) (rightExpression := ⟨12577⟩)
    (transferEvent := 52622)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult52619.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult52614.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult52623

namespace SemanticResult52629
def owner : Owner := ⟨.program ⟨214⟩, ⟨12579⟩⟩
def rawTerms : List Term := Proof.Events205.exact52629RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 52629
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52629.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 52626) (survivorTransfer := 52627)
    (survivorEvent := 52628) (resultEvent := resultEvent)
    (rightCoefficientProducer := 8467)
    (owner := owner) (leftOwner := SemanticResult52623.owner)
    (rightOwner := SemanticResult8468.owner)
    (leftResult := 52623) (rightResult := 8468)
    (leftBinding := 52624) (rightBinding := 52625)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12578⟩) (rightExpression := ⟨100⟩)
    (leftActual := SemanticResult52623.actual selector witness)
    (rightActual := SemanticResult8468.actual selector witness)
    (leftRaw := SemanticResult52623.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨100⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound8467.actual selector witness)
    (survivorMagnitude := LeftBound52627.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult52623.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8468.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8467.derived selector witness)
  · exact LeftBound52627.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult52629

namespace SemanticResult52637
def owner : Owner := ⟨.program ⟨214⟩, ⟨12580⟩⟩
def rawTerms : List Term := Proof.Events205.exact52637RawTerms
def summary : Bound := (.finite 34944)
def resultEvent : Nat := 52637
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52637.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨42, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge52635.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge52635.frameStart)
    (owner := owner) (leftOwner := SemanticResult52629.owner)
    (rightOwner := SemanticResult2433.owner)
    (leftResult := 52629) (rightResult := 2433)
    (leftActual := SemanticResult52629.actual selector witness)
    (rightActual := SemanticResult2433.actual selector witness)
    (leftRaw := SemanticResult52629.rawTerms)
    (rightRaw := SemanticResult2433.rawTerms)
    (working := LeftOperatorMerge52635.working)
    (leftBinding := 52630) (rightBinding := 52631)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12579⟩) (rightExpression := ⟨9930⟩)
    (coefficientTransfer := 52632) (summaryTransfer := 52634)
    (rightCoefficientProducer := 2432)
    (rightSummaryTransfer := 52633)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨42, by decide⟩)
    (rightRecordedMaximum := 42)
    (rightSummaryMaximum := ⟨42, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge52635.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority2432.actual selector witness)
    (summaryMagnitude := LeftBound52634.actual selector witness)
    (reconstruction := LeftOperatorMerge52635.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult52629.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2433.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2432.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority2432.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge52635.operationAgreement
  · exact LeftBound52634.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge52635.working summary) := by
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
end SemanticResult52637

namespace SemanticResult52642
def owner : Owner := ⟨.program ⟨214⟩, ⟨9931⟩⟩
def rawTerms : List Term := Proof.Events205.exact52642RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 52642
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52642.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge52641.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge52641.frameStart)
    (transferEvent := 52640) (owner := owner)
    (leftResult := 2433) (rightResult := 50670)
    (working := LeftOperatorMerge52641.working)
    (reconstruction := LeftOperatorMerge52641.reconstruction)
    (leftReference := .predecessor 0 52638 .coefficient) (rightReference := .predecessor 1 52639 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2433.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge52641.operationAgreement
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
end SemanticResult52642

namespace SemanticResult52647
def owner : Owner := ⟨.program ⟨214⟩, ⟨7260⟩⟩
def rawTerms : List Term := Proof.Events205.exact52647RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 52647
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52647.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge52646.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge52646.frameStart)
    (transferEvent := 52645) (owner := owner)
    (leftResult := 50540) (rightResult := 8517)
    (working := LeftOperatorMerge52646.working)
    (reconstruction := LeftOperatorMerge52646.reconstruction)
    (leftReference := .predecessor 0 52643 .coefficient) (rightReference := .predecessor 1 52644 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8517.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge52646.operationAgreement
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
end SemanticResult52647

namespace SemanticResult52651
def owner : Owner := ⟨.program ⟨214⟩, ⟨9932⟩⟩
def rawTerms : List Term := Proof.Events205.exact52651RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 52651
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52651.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 52648) (rightBinding := 52649)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7260⟩) (rightExpression := ⟨9931⟩)
    (transferEvent := 52650)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult52647.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult52642.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult52651

namespace SemanticResult52657
def owner : Owner := ⟨.program ⟨214⟩, ⟨9933⟩⟩
def rawTerms : List Term := Proof.Events205.exact52657RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 52657
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52657.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 52654) (survivorTransfer := 52655)
    (survivorEvent := 52656) (resultEvent := resultEvent)
    (rightCoefficientProducer := 8508)
    (owner := owner) (leftOwner := SemanticResult52651.owner)
    (rightOwner := SemanticResult8509.owner)
    (leftResult := 52651) (rightResult := 8509)
    (leftBinding := 52652) (rightBinding := 52653)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9932⟩) (rightExpression := ⟨80⟩)
    (leftActual := SemanticResult52651.actual selector witness)
    (rightActual := SemanticResult8509.actual selector witness)
    (leftRaw := SemanticResult52651.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨80⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound8508.actual selector witness)
    (survivorMagnitude := LeftBound52655.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult52651.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8509.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8508.derived selector witness)
  · exact LeftBound52655.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult52657

namespace SemanticResult52667
def owner : Owner := ⟨.program ⟨214⟩, ⟨9934⟩⟩
def rawTerms : List Term := Proof.Events205.exact52667RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 52667
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52667.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge52663.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge52663.frameStart)
    (owner := owner) (leftOwner := SemanticResult52657.owner)
    (rightOwner := SemanticResult8506.owner)
    (leftResult := 52657) (rightResult := 8506)
    (leftActual := SemanticResult52657.actual selector witness)
    (rightActual := SemanticResult8506.actual selector witness)
    (leftRaw := SemanticResult52657.rawTerms)
    (rightRaw := SemanticResult8506.rawTerms)
    (working := LeftOperatorMerge52663.working)
    (leftBinding := 52658) (rightBinding := 52659)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9933⟩) (rightExpression := ⟨7871⟩)
    (coefficientTransfer := 52660) (summaryTransfer := 52662)
    (rightCoefficientProducer := 8505)
    (rightSummaryTransfer := 52661)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge52663.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound8505.actual selector witness)
    (summaryMagnitude := LeftBound52662.actual selector witness)
    (reconstruction := LeftOperatorMerge52663.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult52657.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8506.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8505.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound8505.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge52663.operationAgreement
  · exact LeftBound52662.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge52663.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 52664 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge52663.working
    [{ coefficient := (-1), key := LeftRelationMerge52664.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge52664.frameStart
      LeftRelationMerge52664.owner (.relation 52664) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge52664.deltas
    rows := LeftRelationMerge52664.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge52663.working LeftRelationMerge52664.source
        (relationContext LeftRelationMerge52664.source
          LeftRelationMerge52664.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge52663.working, LeftRelationMerge52664.deltas,
    LeftRelationMerge52664.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 52664)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9934⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge52663.working) (working := relationWorking0)
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
end SemanticResult52667

namespace SemanticResult52673
def owner : Owner := ⟨.program ⟨214⟩, ⟨12581⟩⟩
def rawTerms : List Term := Proof.Events205.exact52673RawTerms
def summary : Bound := (.finite 95455360)
def resultEvent : Nat := 52673
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52673.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge52671.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult52667.owner)
    (rightOwner := SemanticResult52637.owner)
    (leftResult := 52667) (rightResult := 52637)
    (leftActual := SemanticResult52667.actual selector witness)
    (rightActual := SemanticResult52637.actual selector witness)
    (leftRaw := SemanticResult52667.rawTerms)
    (rightRaw := SemanticResult52637.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 34944) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 52668) (rightBinding := 52669)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9934⟩) (rightExpression := ⟨12580⟩)
    (coefficientTransfer := 52670) (summaryTransfer := 52672)
    (base := LeftOperatorMerge52671.base)
    (reconstruction := LeftOperatorMerge52671.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult52667.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult52637.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge52671.operationAgreement
  · rfl
  · decide
end SemanticResult52673

namespace SemanticResult52683
def owner : Owner := ⟨.program ⟨214⟩, ⟨25456⟩⟩
def rawTerms : List Term := Proof.Events205.exact52683RawTerms
def summary : Bound := (.finite 350322698485760)
def resultEvent : Nat := 52683
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52683.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95455360, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge52679.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge52679.frameStart)
    (owner := owner) (leftOwner := SemanticResult52673.owner)
    (rightOwner := SemanticResult52609.owner)
    (leftResult := 52673) (rightResult := 52609)
    (leftActual := SemanticResult52673.actual selector witness)
    (rightActual := SemanticResult52609.actual selector witness)
    (leftRaw := SemanticResult52673.rawTerms)
    (rightRaw := SemanticResult52609.rawTerms)
    (working := LeftOperatorMerge52679.working)
    (leftBinding := 52674) (rightBinding := 52675)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12581⟩) (rightExpression := ⟨25455⟩)
    (coefficientTransfer := 52676) (summaryTransfer := 52678)
    (rightCoefficientProducer := 52608)
    (rightSummaryTransfer := 52677)
    (leftMaximum := ⟨95455360, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge52679.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority52608.actual selector witness)
    (summaryMagnitude := LeftBound52678.actual selector witness)
    (reconstruction := LeftOperatorMerge52679.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult52673.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult52609.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52608.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority52608.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge52679.operationAgreement
  · exact LeftBound52678.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge52679.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 52680 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23250⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23250⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge52679.working
    [{ coefficient := (-1), key := LeftRelationMerge52680.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge52680.frameStart
      LeftRelationMerge52680.owner (.relation 52680) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge52680.deltas
    rows := LeftRelationMerge52680.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge52679.working LeftRelationMerge52680.source
        (relationContext LeftRelationMerge52680.source
          LeftRelationMerge52680.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge52679.working, LeftRelationMerge52680.deltas,
    LeftRelationMerge52680.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 52680)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25456⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9930⟩⟩, ⟨.program ⟨214⟩, ⟨12574⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25455⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge52679.working) (working := relationWorking0)
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
end SemanticResult52683

namespace SemanticResult52686
def owner : Owner := ⟨.program ⟨214⟩, ⟨19964⟩⟩
def rawTerms : List Term := Proof.Events205.exact52686RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 52686
def producerEvent : Nat := 52685
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52686.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨21⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨21⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult52686

namespace SemanticResult52690
def owner : Owner := ⟨.program ⟨214⟩, ⟨19966⟩⟩
def rawTerms : List Term := Proof.Events205.exact52690RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 52690
def producerEvent : Nat := 52689
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult52690.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 52687 .coefficient) (.value (.predecessor 1 52688 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 52687 .coefficient) (.value (.predecessor 1 52688 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult52690

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
