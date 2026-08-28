import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard057
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard364

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult50655
def owner : Owner := ⟨.program ⟨214⟩, ⟨30139⟩⟩
def rawTerms : List Term := Proof.Events197.exact50655RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50655
def producerEvent : Nat := 50654
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50655.actual selector witness
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
end SemanticResult50655

namespace SemanticResult50662
def owner : Owner := ⟨.program ⟨214⟩, ⟨23418⟩⟩
def rawTerms : List Term := Proof.Events197.exact50662RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50662
def producerEvent : Nat := 50661
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50662.actual selector witness
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
end SemanticResult50662

namespace SemanticResult50665
def owner : Owner := ⟨.program ⟨214⟩, ⟨25763⟩⟩
def rawTerms : List Term := Proof.Events197.exact50665RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50665
def producerEvent : Nat := 50664
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50665.actual selector witness
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
end SemanticResult50665

namespace SemanticResult50670
def owner : Owner := ⟨.program ⟨214⟩, ⟨6568⟩⟩
def rawTerms : List Term := Proof.Events197.exact50670RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50670
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50670.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50669.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge50669.frameStart)
    (transferEvent := 50668) (owner := owner)
    (leftResult := 50540) (rightResult := 2)
    (working := LeftOperatorMerge50669.working)
    (reconstruction := LeftOperatorMerge50669.reconstruction)
    (leftReference := .predecessor 0 50666 .coefficient) (rightReference := .predecessor 1 50667 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge50669.operationAgreement
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
end SemanticResult50670

namespace SemanticResult50675
def owner : Owner := ⟨.program ⟨214⟩, ⟨13361⟩⟩
def rawTerms : List Term := Proof.Events197.exact50675RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50675
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50675.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50674.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge50674.frameStart)
    (transferEvent := 50673) (owner := owner)
    (leftResult := 2338) (rightResult := 50670)
    (working := LeftOperatorMerge50674.working)
    (reconstruction := LeftOperatorMerge50674.reconstruction)
    (leftReference := .predecessor 0 50671 .coefficient) (rightReference := .predecessor 1 50672 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2338.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge50674.operationAgreement
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
end SemanticResult50675

namespace SemanticResult50680
def owner : Owner := ⟨.program ⟨214⟩, ⟨7284⟩⟩
def rawTerms : List Term := Proof.Events197.exact50680RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50680
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50680.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50679.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge50679.frameStart)
    (transferEvent := 50678) (owner := owner)
    (leftResult := 50540) (rightResult := 6457)
    (working := LeftOperatorMerge50679.working)
    (reconstruction := LeftOperatorMerge50679.reconstruction)
    (leftReference := .predecessor 0 50676 .coefficient) (rightReference := .predecessor 1 50677 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6457.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge50679.operationAgreement
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
end SemanticResult50680

namespace SemanticResult50684
def owner : Owner := ⟨.program ⟨214⟩, ⟨13362⟩⟩
def rawTerms : List Term := Proof.Events197.exact50684RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50684
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50684.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 50681) (rightBinding := 50682)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7284⟩) (rightExpression := ⟨13361⟩)
    (transferEvent := 50683)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50680.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50675.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50684

namespace SemanticResult50690
def owner : Owner := ⟨.program ⟨214⟩, ⟨13363⟩⟩
def rawTerms : List Term := Proof.Events198.exact50690RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 50690
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50690.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 50687) (survivorTransfer := 50688)
    (survivorEvent := 50689) (resultEvent := resultEvent)
    (rightCoefficientProducer := 6443)
    (owner := owner) (leftOwner := SemanticResult50684.owner)
    (rightOwner := SemanticResult6444.owner)
    (leftResult := 50684) (rightResult := 6444)
    (leftBinding := 50685) (rightBinding := 50686)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13362⟩) (rightExpression := ⟨104⟩)
    (leftActual := SemanticResult50684.actual selector witness)
    (rightActual := SemanticResult6444.actual selector witness)
    (leftRaw := SemanticResult50684.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨104⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound6443.actual selector witness)
    (survivorMagnitude := LeftBound50688.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50684.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6444.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6443.derived selector witness)
  · exact LeftBound50688.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult50690

namespace SemanticResult50698
def owner : Owner := ⟨.program ⟨214⟩, ⟨13364⟩⟩
def rawTerms : List Term := Proof.Events198.exact50698RawTerms
def summary : Bound := (.finite 49920)
def resultEvent : Nat := 50698
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50698.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨60, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50696.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge50696.frameStart)
    (owner := owner) (leftOwner := SemanticResult50690.owner)
    (rightOwner := SemanticResult2341.owner)
    (leftResult := 50690) (rightResult := 2341)
    (leftActual := SemanticResult50690.actual selector witness)
    (rightActual := SemanticResult2341.actual selector witness)
    (leftRaw := SemanticResult50690.rawTerms)
    (rightRaw := SemanticResult2341.rawTerms)
    (working := LeftOperatorMerge50696.working)
    (leftBinding := 50691) (rightBinding := 50692)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13363⟩) (rightExpression := ⟨10350⟩)
    (coefficientTransfer := 50693) (summaryTransfer := 50695)
    (rightCoefficientProducer := 2340)
    (rightSummaryTransfer := 50694)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨60, by decide⟩)
    (rightRecordedMaximum := 60)
    (rightSummaryMaximum := ⟨60, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge50696.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority2340.actual selector witness)
    (summaryMagnitude := LeftBound50695.actual selector witness)
    (reconstruction := LeftOperatorMerge50696.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50690.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2341.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2340.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority2340.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge50696.operationAgreement
  · exact LeftBound50695.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50696.working summary) := by
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
end SemanticResult50698

namespace SemanticResult50703
def owner : Owner := ⟨.program ⟨214⟩, ⟨10351⟩⟩
def rawTerms : List Term := Proof.Events198.exact50703RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50703
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50703.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50702.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge50702.frameStart)
    (transferEvent := 50701) (owner := owner)
    (leftResult := 2341) (rightResult := 50670)
    (working := LeftOperatorMerge50702.working)
    (reconstruction := LeftOperatorMerge50702.reconstruction)
    (leftReference := .predecessor 0 50699 .coefficient) (rightReference := .predecessor 1 50700 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2341.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge50702.operationAgreement
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
end SemanticResult50703

namespace SemanticResult50708
def owner : Owner := ⟨.program ⟨214⟩, ⟨7264⟩⟩
def rawTerms : List Term := Proof.Events198.exact50708RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50708
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50708.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50707.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge50707.frameStart)
    (transferEvent := 50706) (owner := owner)
    (leftResult := 50540) (rightResult := 6498)
    (working := LeftOperatorMerge50707.working)
    (reconstruction := LeftOperatorMerge50707.reconstruction)
    (leftReference := .predecessor 0 50704 .coefficient) (rightReference := .predecessor 1 50705 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6498.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge50707.operationAgreement
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
end SemanticResult50708

namespace SemanticResult50712
def owner : Owner := ⟨.program ⟨214⟩, ⟨10352⟩⟩
def rawTerms : List Term := Proof.Events198.exact50712RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50712
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50712.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 50709) (rightBinding := 50710)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7264⟩) (rightExpression := ⟨10351⟩)
    (transferEvent := 50711)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50708.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50703.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50712

namespace SemanticResult50718
def owner : Owner := ⟨.program ⟨214⟩, ⟨10353⟩⟩
def rawTerms : List Term := Proof.Events198.exact50718RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 50718
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50718.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 50715) (survivorTransfer := 50716)
    (survivorEvent := 50717) (resultEvent := resultEvent)
    (rightCoefficientProducer := 6489)
    (owner := owner) (leftOwner := SemanticResult50712.owner)
    (rightOwner := SemanticResult6490.owner)
    (leftResult := 50712) (rightResult := 6490)
    (leftBinding := 50713) (rightBinding := 50714)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10352⟩) (rightExpression := ⟨84⟩)
    (leftActual := SemanticResult50712.actual selector witness)
    (rightActual := SemanticResult6490.actual selector witness)
    (leftRaw := SemanticResult50712.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨84⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound6489.actual selector witness)
    (survivorMagnitude := LeftBound50716.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50712.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6490.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6489.derived selector witness)
  · exact LeftBound50716.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult50718

namespace SemanticResult50728
def owner : Owner := ⟨.program ⟨214⟩, ⟨10354⟩⟩
def rawTerms : List Term := Proof.Events198.exact50728RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 50728
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50728.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50724.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge50724.frameStart)
    (owner := owner) (leftOwner := SemanticResult50718.owner)
    (rightOwner := SemanticResult6487.owner)
    (leftResult := 50718) (rightResult := 6487)
    (leftActual := SemanticResult50718.actual selector witness)
    (rightActual := SemanticResult6487.actual selector witness)
    (leftRaw := SemanticResult50718.rawTerms)
    (rightRaw := SemanticResult6487.rawTerms)
    (working := LeftOperatorMerge50724.working)
    (leftBinding := 50719) (rightBinding := 50720)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10353⟩) (rightExpression := ⟨7883⟩)
    (coefficientTransfer := 50721) (summaryTransfer := 50723)
    (rightCoefficientProducer := 6486)
    (rightSummaryTransfer := 50722)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge50724.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound6486.actual selector witness)
    (summaryMagnitude := LeftBound50723.actual selector witness)
    (reconstruction := LeftOperatorMerge50724.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50718.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6487.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6486.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound6486.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge50724.operationAgreement
  · exact LeftBound50723.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50724.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 50725 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10350⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge50724.working
    [{ coefficient := (-1), key := LeftRelationMerge50725.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge50725.frameStart
      LeftRelationMerge50725.owner (.relation 50725) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge50725.deltas
    rows := LeftRelationMerge50725.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge50724.working LeftRelationMerge50725.source
        (relationContext LeftRelationMerge50725.source
          LeftRelationMerge50725.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge50724.working, LeftRelationMerge50725.deltas,
    LeftRelationMerge50725.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 50725)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10354⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10350⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge50724.working) (working := relationWorking0)
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
end SemanticResult50728

namespace SemanticResult50734
def owner : Owner := ⟨.program ⟨214⟩, ⟨13365⟩⟩
def rawTerms : List Term := Proof.Events198.exact50734RawTerms
def summary : Bound := (.finite 95470336)
def resultEvent : Nat := 50734
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50734.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge50732.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50728.owner)
    (rightOwner := SemanticResult50698.owner)
    (leftResult := 50728) (rightResult := 50698)
    (leftActual := SemanticResult50728.actual selector witness)
    (rightActual := SemanticResult50698.actual selector witness)
    (leftRaw := SemanticResult50728.rawTerms)
    (rightRaw := SemanticResult50698.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 49920) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50729) (rightBinding := 50730)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10354⟩) (rightExpression := ⟨13364⟩)
    (coefficientTransfer := 50731) (summaryTransfer := 50733)
    (base := LeftOperatorMerge50732.base)
    (reconstruction := LeftOperatorMerge50732.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50698.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge50732.operationAgreement
  · rfl
  · decide
end SemanticResult50734

namespace SemanticResult50744
def owner : Owner := ⟨.program ⟨214⟩, ⟨25764⟩⟩
def rawTerms : List Term := Proof.Events198.exact50744RawTerms
def summary : Bound := (.finite 350377660645376)
def resultEvent : Nat := 50744
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50744.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95470336, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50740.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge50740.frameStart)
    (owner := owner) (leftOwner := SemanticResult50734.owner)
    (rightOwner := SemanticResult50665.owner)
    (leftResult := 50734) (rightResult := 50665)
    (leftActual := SemanticResult50734.actual selector witness)
    (rightActual := SemanticResult50665.actual selector witness)
    (leftRaw := SemanticResult50734.rawTerms)
    (rightRaw := SemanticResult50665.rawTerms)
    (working := LeftOperatorMerge50740.working)
    (leftBinding := 50735) (rightBinding := 50736)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13365⟩) (rightExpression := ⟨25763⟩)
    (coefficientTransfer := 50737) (summaryTransfer := 50739)
    (rightCoefficientProducer := 50664)
    (rightSummaryTransfer := 50738)
    (leftMaximum := ⟨95470336, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge50740.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority50664.actual selector witness)
    (summaryMagnitude := LeftBound50739.actual selector witness)
    (reconstruction := LeftOperatorMerge50740.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50734.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50665.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50664.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority50664.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge50740.operationAgreement
  · exact LeftBound50739.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50740.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 50741 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23418⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23418⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge50740.working
    [{ coefficient := (-1), key := LeftRelationMerge50741.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge50741.frameStart
      LeftRelationMerge50741.owner (.relation 50741) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge50741.deltas
    rows := LeftRelationMerge50741.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge50740.working LeftRelationMerge50741.source
        (relationContext LeftRelationMerge50741.source
          LeftRelationMerge50741.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge50740.working, LeftRelationMerge50741.deltas,
    LeftRelationMerge50741.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 50741)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25764⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge50740.working) (working := relationWorking0)
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
end SemanticResult50744

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
