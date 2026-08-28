import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard402
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard097
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard098
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard365

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult55488
def owner : Owner := ⟨.program ⟨214⟩, ⟨24165⟩⟩
def rawTerms : List Term := Proof.Events216.exact55488RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55488
def producerEvent : Nat := 55487
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55488.actual selector witness
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
end SemanticResult55488

namespace SemanticResult55491
def owner : Owner := ⟨.program ⟨214⟩, ⟨27879⟩⟩
def rawTerms : List Term := Proof.Events216.exact55491RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55491
def producerEvent : Nat := 55490
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55491.actual selector witness
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
end SemanticResult55491

namespace SemanticResult55498
def owner : Owner := ⟨.program ⟨214⟩, ⟨23586⟩⟩
def rawTerms : List Term := Proof.Events216.exact55498RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55498
def producerEvent : Nat := 55497
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55498.actual selector witness
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
end SemanticResult55498

namespace SemanticResult55501
def owner : Owner := ⟨.program ⟨214⟩, ⟨26071⟩⟩
def rawTerms : List Term := Proof.Events216.exact55501RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55501
def producerEvent : Nat := 55500
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55501.actual selector witness
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
end SemanticResult55501

namespace SemanticResult55506
def owner : Owner := ⟨.program ⟨214⟩, ⟨11474⟩⟩
def rawTerms : List Term := Proof.Events216.exact55506RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55506
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55506.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55505.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge55505.frameStart)
    (transferEvent := 55504) (owner := owner)
    (leftResult := 2568) (rightResult := 50670)
    (working := LeftOperatorMerge55505.working)
    (reconstruction := LeftOperatorMerge55505.reconstruction)
    (leftReference := .predecessor 0 55502 .coefficient) (rightReference := .predecessor 1 55503 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2568.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge55505.operationAgreement
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
end SemanticResult55506

namespace SemanticResult55511
def owner : Owner := ⟨.program ⟨214⟩, ⟨7273⟩⟩
def rawTerms : List Term := Proof.Events216.exact55511RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55511
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55511.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55510.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge55510.frameStart)
    (transferEvent := 55509) (owner := owner)
    (leftResult := 50540) (rightResult := 11482)
    (working := LeftOperatorMerge55510.working)
    (reconstruction := LeftOperatorMerge55510.reconstruction)
    (leftReference := .predecessor 0 55507 .coefficient) (rightReference := .predecessor 1 55508 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11482.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge55510.operationAgreement
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
end SemanticResult55511

namespace SemanticResult55515
def owner : Owner := ⟨.program ⟨214⟩, ⟨11475⟩⟩
def rawTerms : List Term := Proof.Events216.exact55515RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55515
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55515.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 55512) (rightBinding := 55513)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7273⟩) (rightExpression := ⟨11474⟩)
    (transferEvent := 55514)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult55511.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult55506.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult55515

namespace SemanticResult55521
def owner : Owner := ⟨.program ⟨214⟩, ⟨11476⟩⟩
def rawTerms : List Term := Proof.Events216.exact55521RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 55521
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55521.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 55518) (survivorTransfer := 55519)
    (survivorEvent := 55520) (resultEvent := resultEvent)
    (rightCoefficientProducer := 11473)
    (owner := owner) (leftOwner := SemanticResult55515.owner)
    (rightOwner := SemanticResult11474.owner)
    (leftResult := 55515) (rightResult := 11474)
    (leftBinding := 55516) (rightBinding := 55517)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11475⟩) (rightExpression := ⟨93⟩)
    (leftActual := SemanticResult55515.actual selector witness)
    (rightActual := SemanticResult11474.actual selector witness)
    (leftRaw := SemanticResult55515.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨93⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound11473.actual selector witness)
    (survivorMagnitude := LeftBound55519.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult55515.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11474.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)
  · exact LeftBound55519.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult55521

namespace SemanticResult55529
def owner : Owner := ⟨.program ⟨214⟩, ⟨14219⟩⟩
def rawTerms : List Term := Proof.Events216.exact55529RawTerms
def summary : Bound := (.finite 14976)
def resultEvent : Nat := 55529
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55529.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨18, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55527.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge55527.frameStart)
    (owner := owner) (leftOwner := SemanticResult55521.owner)
    (rightOwner := SemanticResult2571.owner)
    (leftResult := 55521) (rightResult := 2571)
    (leftActual := SemanticResult55521.actual selector witness)
    (rightActual := SemanticResult2571.actual selector witness)
    (leftRaw := SemanticResult55521.rawTerms)
    (rightRaw := SemanticResult2571.rawTerms)
    (working := LeftOperatorMerge55527.working)
    (leftBinding := 55522) (rightBinding := 55523)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11476⟩) (rightExpression := ⟨14216⟩)
    (coefficientTransfer := 55524) (summaryTransfer := 55526)
    (rightCoefficientProducer := 2570)
    (rightSummaryTransfer := 55525)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨18, by decide⟩)
    (rightRecordedMaximum := 18)
    (rightSummaryMaximum := ⟨18, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge55527.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority2570.actual selector witness)
    (summaryMagnitude := LeftBound55526.actual selector witness)
    (reconstruction := LeftOperatorMerge55527.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult55521.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2571.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2570.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority2570.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge55527.operationAgreement
  · exact LeftBound55526.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55527.working summary) := by
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
end SemanticResult55529

namespace SemanticResult55534
def owner : Owner := ⟨.program ⟨214⟩, ⟨14220⟩⟩
def rawTerms : List Term := Proof.Events216.exact55534RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55534
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55534.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55533.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge55533.frameStart)
    (transferEvent := 55532) (owner := owner)
    (leftResult := 2571) (rightResult := 50670)
    (working := LeftOperatorMerge55533.working)
    (reconstruction := LeftOperatorMerge55533.reconstruction)
    (leftReference := .predecessor 0 55530 .coefficient) (rightReference := .predecessor 1 55531 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2571.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge55533.operationAgreement
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
end SemanticResult55534

namespace SemanticResult55539
def owner : Owner := ⟨.program ⟨214⟩, ⟨7253⟩⟩
def rawTerms : List Term := Proof.Events216.exact55539RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55539
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55539.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55538.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge55538.frameStart)
    (transferEvent := 55537) (owner := owner)
    (leftResult := 50540) (rightResult := 11523)
    (working := LeftOperatorMerge55538.working)
    (reconstruction := LeftOperatorMerge55538.reconstruction)
    (leftReference := .predecessor 0 55535 .coefficient) (rightReference := .predecessor 1 55536 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11523.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge55538.operationAgreement
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
end SemanticResult55539

namespace SemanticResult55543
def owner : Owner := ⟨.program ⟨214⟩, ⟨14221⟩⟩
def rawTerms : List Term := Proof.Events216.exact55543RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55543
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55543.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 55540) (rightBinding := 55541)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7253⟩) (rightExpression := ⟨14220⟩)
    (transferEvent := 55542)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult55539.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult55534.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult55543

namespace SemanticResult55549
def owner : Owner := ⟨.program ⟨214⟩, ⟨14222⟩⟩
def rawTerms : List Term := Proof.Events216.exact55549RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 55549
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55549.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 55546) (survivorTransfer := 55547)
    (survivorEvent := 55548) (resultEvent := resultEvent)
    (rightCoefficientProducer := 11514)
    (owner := owner) (leftOwner := SemanticResult55543.owner)
    (rightOwner := SemanticResult11515.owner)
    (leftResult := 55543) (rightResult := 11515)
    (leftBinding := 55544) (rightBinding := 55545)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14221⟩) (rightExpression := ⟨73⟩)
    (leftActual := SemanticResult55543.actual selector witness)
    (rightActual := SemanticResult11515.actual selector witness)
    (leftRaw := SemanticResult55543.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨73⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound11514.actual selector witness)
    (survivorMagnitude := LeftBound55547.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult55543.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11515.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11514.derived selector witness)
  · exact LeftBound55547.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult55549

namespace SemanticResult55559
def owner : Owner := ⟨.program ⟨214⟩, ⟨14223⟩⟩
def rawTerms : List Term := Proof.Events217.exact55559RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 55559
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55559.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55555.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge55555.frameStart)
    (owner := owner) (leftOwner := SemanticResult55549.owner)
    (rightOwner := SemanticResult11512.owner)
    (leftResult := 55549) (rightResult := 11512)
    (leftActual := SemanticResult55549.actual selector witness)
    (rightActual := SemanticResult11512.actual selector witness)
    (leftRaw := SemanticResult55549.rawTerms)
    (rightRaw := SemanticResult11512.rawTerms)
    (working := LeftOperatorMerge55555.working)
    (leftBinding := 55550) (rightBinding := 55551)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14222⟩) (rightExpression := ⟨7853⟩)
    (coefficientTransfer := 55552) (summaryTransfer := 55554)
    (rightCoefficientProducer := 11511)
    (rightSummaryTransfer := 55553)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge55555.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound11511.actual selector witness)
    (summaryMagnitude := LeftBound55554.actual selector witness)
    (reconstruction := LeftOperatorMerge55555.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult55549.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11512.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11511.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound11511.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge55555.operationAgreement
  · exact LeftBound55554.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55555.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 55556 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6779⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6779⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge55555.working
    [{ coefficient := (-1), key := LeftRelationMerge55556.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge55556.frameStart
      LeftRelationMerge55556.owner (.relation 55556) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge55556.deltas
    rows := LeftRelationMerge55556.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge55555.working LeftRelationMerge55556.source
        (relationContext LeftRelationMerge55556.source
          LeftRelationMerge55556.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge55555.working, LeftRelationMerge55556.deltas,
    LeftRelationMerge55556.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 55556)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨14223⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge55555.working) (working := relationWorking0)
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
end SemanticResult55559

namespace SemanticResult55565
def owner : Owner := ⟨.program ⟨214⟩, ⟨14224⟩⟩
def rawTerms : List Term := Proof.Events217.exact55565RawTerms
def summary : Bound := (.finite 95435392)
def resultEvent : Nat := 55565
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55565.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge55563.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult55559.owner)
    (rightOwner := SemanticResult55529.owner)
    (leftResult := 55559) (rightResult := 55529)
    (leftActual := SemanticResult55559.actual selector witness)
    (rightActual := SemanticResult55529.actual selector witness)
    (leftRaw := SemanticResult55559.rawTerms)
    (rightRaw := SemanticResult55529.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 14976) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 55560) (rightBinding := 55561)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14223⟩) (rightExpression := ⟨14219⟩)
    (coefficientTransfer := 55562) (summaryTransfer := 55564)
    (base := LeftOperatorMerge55563.base)
    (reconstruction := LeftOperatorMerge55563.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult55559.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult55529.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge55563.operationAgreement
  · rfl
  · decide
end SemanticResult55565

namespace SemanticResult55575
def owner : Owner := ⟨.program ⟨214⟩, ⟨26072⟩⟩
def rawTerms : List Term := Proof.Events217.exact55575RawTerms
def summary : Bound := (.finite 350249415606272)
def resultEvent : Nat := 55575
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55575.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95435392, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55571.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge55571.frameStart)
    (owner := owner) (leftOwner := SemanticResult55565.owner)
    (rightOwner := SemanticResult55501.owner)
    (leftResult := 55565) (rightResult := 55501)
    (leftActual := SemanticResult55565.actual selector witness)
    (rightActual := SemanticResult55501.actual selector witness)
    (leftRaw := SemanticResult55565.rawTerms)
    (rightRaw := SemanticResult55501.rawTerms)
    (working := LeftOperatorMerge55571.working)
    (leftBinding := 55566) (rightBinding := 55567)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14224⟩) (rightExpression := ⟨26071⟩)
    (coefficientTransfer := 55568) (summaryTransfer := 55570)
    (rightCoefficientProducer := 55500)
    (rightSummaryTransfer := 55569)
    (leftMaximum := ⟨95435392, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge55571.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority55500.actual selector witness)
    (summaryMagnitude := LeftBound55570.actual selector witness)
    (reconstruction := LeftOperatorMerge55571.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult55565.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult55501.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55500.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority55500.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge55571.operationAgreement
  · exact LeftBound55570.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55571.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 55572 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23586⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23586⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge55571.working
    [{ coefficient := (-1), key := LeftRelationMerge55572.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge55572.frameStart
      LeftRelationMerge55572.owner (.relation 55572) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge55572.deltas
    rows := LeftRelationMerge55572.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge55571.working LeftRelationMerge55572.source
        (relationContext LeftRelationMerge55572.source
          LeftRelationMerge55572.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge55571.working, LeftRelationMerge55572.deltas,
    LeftRelationMerge55572.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 55572)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨26072⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11473⟩⟩, ⟨.program ⟨214⟩, ⟨14216⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge55571.working) (working := relationWorking0)
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
end SemanticResult55575

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
