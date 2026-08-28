import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard212
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard009
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard109
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard110
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard164

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult27687
def owner : Owner := ⟨.program ⟨214⟩, ⟨27254⟩⟩
def rawTerms : List Term := Proof.Events108.exact27687RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27687
def producerEvent : Nat := 27686
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27687.actual selector witness
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
end SemanticResult27687

namespace SemanticResult27694
def owner : Owner := ⟨.program ⟨214⟩, ⟨23464⟩⟩
def rawTerms : List Term := Proof.Events108.exact27694RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27694
def producerEvent : Nat := 27693
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27694.actual selector witness
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
end SemanticResult27694

namespace SemanticResult27697
def owner : Owner := ⟨.program ⟨214⟩, ⟨25850⟩⟩
def rawTerms : List Term := Proof.Events108.exact27697RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27697
def producerEvent : Nat := 27696
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27697.actual selector witness
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
end SemanticResult27697

namespace SemanticResult27702
def owner : Owner := ⟨.program ⟨214⟩, ⟨11230⟩⟩
def rawTerms : List Term := Proof.Events108.exact27702RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27702
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27702.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge27701.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge27701.frameStart)
    (transferEvent := 27700) (owner := owner)
    (leftResult := 1141) (rightResult := 21420)
    (working := LeftOperatorMerge27701.working)
    (reconstruction := LeftOperatorMerge27701.reconstruction)
    (leftReference := .predecessor 0 27698 .coefficient) (rightReference := .predecessor 1 27699 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1141.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge27701.operationAgreement
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
end SemanticResult27702

namespace SemanticResult27707
def owner : Owner := ⟨.program ⟨214⟩, ⟨7346⟩⟩
def rawTerms : List Term := Proof.Events108.exact27707RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27707
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27707.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge27706.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge27706.frameStart)
    (transferEvent := 27705) (owner := owner)
    (leftResult := 21290) (rightResult := 12985)
    (working := LeftOperatorMerge27706.working)
    (reconstruction := LeftOperatorMerge27706.reconstruction)
    (leftReference := .predecessor 0 27703 .coefficient) (rightReference := .predecessor 1 27704 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12985.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge27706.operationAgreement
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
end SemanticResult27707

namespace SemanticResult27711
def owner : Owner := ⟨.program ⟨214⟩, ⟨11231⟩⟩
def rawTerms : List Term := Proof.Events108.exact27711RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27711
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27711.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27708) (rightBinding := 27709)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7346⟩) (rightExpression := ⟨11230⟩)
    (transferEvent := 27710)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27707.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27702.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27711

namespace SemanticResult27717
def owner : Owner := ⟨.program ⟨214⟩, ⟨11232⟩⟩
def rawTerms : List Term := Proof.Events108.exact27717RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 27717
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27717.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 27714) (survivorTransfer := 27715)
    (survivorEvent := 27716) (resultEvent := resultEvent)
    (rightCoefficientProducer := 12976)
    (owner := owner) (leftOwner := SemanticResult27711.owner)
    (rightOwner := SemanticResult12977.owner)
    (leftResult := 27711) (rightResult := 12977)
    (leftBinding := 27712) (rightBinding := 27713)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11231⟩) (rightExpression := ⟨90⟩)
    (leftActual := SemanticResult27711.actual selector witness)
    (rightActual := SemanticResult12977.actual selector witness)
    (leftRaw := SemanticResult27711.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨90⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound12976.actual selector witness)
    (survivorMagnitude := LeftBound27715.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27711.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12977.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12976.derived selector witness)
  · exact LeftBound27715.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult27717

namespace SemanticResult27725
def owner : Owner := ⟨.program ⟨214⟩, ⟨13586⟩⟩
def rawTerms : List Term := Proof.Events108.exact27725RawTerms
def summary : Bound := (.finite 8320)
def resultEvent : Nat := 27725
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27725.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨10, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge27723.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge27723.frameStart)
    (owner := owner) (leftOwner := SemanticResult27717.owner)
    (rightOwner := SemanticResult1144.owner)
    (leftResult := 27717) (rightResult := 1144)
    (leftActual := SemanticResult27717.actual selector witness)
    (rightActual := SemanticResult1144.actual selector witness)
    (leftRaw := SemanticResult27717.rawTerms)
    (rightRaw := SemanticResult1144.rawTerms)
    (working := LeftOperatorMerge27723.working)
    (leftBinding := 27718) (rightBinding := 27719)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11232⟩) (rightExpression := ⟨13583⟩)
    (coefficientTransfer := 27720) (summaryTransfer := 27722)
    (rightCoefficientProducer := 1143)
    (rightSummaryTransfer := 27721)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨10, by decide⟩)
    (rightRecordedMaximum := 10)
    (rightSummaryMaximum := ⟨10, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge27723.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1143.actual selector witness)
    (summaryMagnitude := LeftBound27722.actual selector witness)
    (reconstruction := LeftOperatorMerge27723.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27717.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1144.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1143.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1143.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge27723.operationAgreement
  · exact LeftBound27722.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge27723.working summary) := by
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
end SemanticResult27725

namespace SemanticResult27730
def owner : Owner := ⟨.program ⟨214⟩, ⟨13587⟩⟩
def rawTerms : List Term := Proof.Events108.exact27730RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27730
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27730.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge27729.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge27729.frameStart)
    (transferEvent := 27728) (owner := owner)
    (leftResult := 1144) (rightResult := 21420)
    (working := LeftOperatorMerge27729.working)
    (reconstruction := LeftOperatorMerge27729.reconstruction)
    (leftReference := .predecessor 0 27726 .coefficient) (rightReference := .predecessor 1 27727 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1144.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge27729.operationAgreement
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
end SemanticResult27730

namespace SemanticResult27735
def owner : Owner := ⟨.program ⟨214⟩, ⟨7363⟩⟩
def rawTerms : List Term := Proof.Events108.exact27735RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27735
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27735.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge27734.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge27734.frameStart)
    (transferEvent := 27733) (owner := owner)
    (leftResult := 21290) (rightResult := 13026)
    (working := LeftOperatorMerge27734.working)
    (reconstruction := LeftOperatorMerge27734.reconstruction)
    (leftReference := .predecessor 0 27731 .coefficient) (rightReference := .predecessor 1 27732 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13026.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge27734.operationAgreement
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
end SemanticResult27735

namespace SemanticResult27739
def owner : Owner := ⟨.program ⟨214⟩, ⟨13588⟩⟩
def rawTerms : List Term := Proof.Events108.exact27739RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27739
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27739.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 27736) (rightBinding := 27737)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7363⟩) (rightExpression := ⟨13587⟩)
    (transferEvent := 27738)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27735.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27730.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult27739

namespace SemanticResult27745
def owner : Owner := ⟨.program ⟨214⟩, ⟨13589⟩⟩
def rawTerms : List Term := Proof.Events108.exact27745RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 27745
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27745.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 27742) (survivorTransfer := 27743)
    (survivorEvent := 27744) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13017)
    (owner := owner) (leftOwner := SemanticResult27739.owner)
    (rightOwner := SemanticResult13018.owner)
    (leftResult := 27739) (rightResult := 13018)
    (leftBinding := 27740) (rightBinding := 27741)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13588⟩) (rightExpression := ⟨107⟩)
    (leftActual := SemanticResult27739.actual selector witness)
    (rightActual := SemanticResult13018.actual selector witness)
    (leftRaw := SemanticResult27739.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨107⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13017.actual selector witness)
    (survivorMagnitude := LeftBound27743.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27739.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13018.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13017.derived selector witness)
  · exact LeftBound27743.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult27745

namespace SemanticResult27755
def owner : Owner := ⟨.program ⟨214⟩, ⟨13590⟩⟩
def rawTerms : List Term := Proof.Events108.exact27755RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 27755
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27755.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge27751.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge27751.frameStart)
    (owner := owner) (leftOwner := SemanticResult27745.owner)
    (rightOwner := SemanticResult13015.owner)
    (leftResult := 27745) (rightResult := 13015)
    (leftActual := SemanticResult27745.actual selector witness)
    (rightActual := SemanticResult13015.actual selector witness)
    (leftRaw := SemanticResult27745.rawTerms)
    (rightRaw := SemanticResult13015.rawTerms)
    (working := LeftOperatorMerge27751.working)
    (leftBinding := 27746) (rightBinding := 27747)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13589⟩) (rightExpression := ⟨7844⟩)
    (coefficientTransfer := 27748) (summaryTransfer := 27750)
    (rightCoefficientProducer := 13014)
    (rightSummaryTransfer := 27749)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge27751.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound13014.actual selector witness)
    (summaryMagnitude := LeftBound27750.actual selector witness)
    (reconstruction := LeftOperatorMerge27751.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27745.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13015.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13014.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound13014.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge27751.operationAgreement
  · exact LeftBound27750.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge27751.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 27752 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge27751.working
    [{ coefficient := (-1), key := LeftRelationMerge27752.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge27752.frameStart
      LeftRelationMerge27752.owner (.relation 27752) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge27752.deltas
    rows := LeftRelationMerge27752.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge27751.working LeftRelationMerge27752.source
        (relationContext LeftRelationMerge27752.source
          LeftRelationMerge27752.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge27751.working, LeftRelationMerge27752.deltas,
    LeftRelationMerge27752.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 27752)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨13590⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge27751.working) (working := relationWorking0)
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
end SemanticResult27755

namespace SemanticResult27761
def owner : Owner := ⟨.program ⟨214⟩, ⟨13591⟩⟩
def rawTerms : List Term := Proof.Events108.exact27761RawTerms
def summary : Bound := (.finite 95428736)
def resultEvent : Nat := 27761
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27761.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge27759.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult27755.owner)
    (rightOwner := SemanticResult27725.owner)
    (leftResult := 27755) (rightResult := 27725)
    (leftActual := SemanticResult27755.actual selector witness)
    (rightActual := SemanticResult27725.actual selector witness)
    (leftRaw := SemanticResult27755.rawTerms)
    (rightRaw := SemanticResult27725.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 8320) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 27756) (rightBinding := 27757)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13590⟩) (rightExpression := ⟨13586⟩)
    (coefficientTransfer := 27758) (summaryTransfer := 27760)
    (base := LeftOperatorMerge27759.base)
    (reconstruction := LeftOperatorMerge27759.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27755.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27725.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge27759.operationAgreement
  · rfl
  · decide
end SemanticResult27761

namespace SemanticResult27771
def owner : Owner := ⟨.program ⟨214⟩, ⟨25851⟩⟩
def rawTerms : List Term := Proof.Events108.exact27771RawTerms
def summary : Bound := (.finite 350224987979776)
def resultEvent : Nat := 27771
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27771.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95428736, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge27767.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge27767.frameStart)
    (owner := owner) (leftOwner := SemanticResult27761.owner)
    (rightOwner := SemanticResult27697.owner)
    (leftResult := 27761) (rightResult := 27697)
    (leftActual := SemanticResult27761.actual selector witness)
    (rightActual := SemanticResult27697.actual selector witness)
    (leftRaw := SemanticResult27761.rawTerms)
    (rightRaw := SemanticResult27697.rawTerms)
    (working := LeftOperatorMerge27767.working)
    (leftBinding := 27762) (rightBinding := 27763)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13591⟩) (rightExpression := ⟨25850⟩)
    (coefficientTransfer := 27764) (summaryTransfer := 27766)
    (rightCoefficientProducer := 27696)
    (rightSummaryTransfer := 27765)
    (leftMaximum := ⟨95428736, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge27767.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority27696.actual selector witness)
    (summaryMagnitude := LeftBound27766.actual selector witness)
    (reconstruction := LeftOperatorMerge27767.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult27761.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27697.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27696.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority27696.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge27767.operationAgreement
  · exact LeftBound27766.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge27767.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 27768 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23464⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23464⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge27767.working
    [{ coefficient := (-1), key := LeftRelationMerge27768.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge27768.frameStart
      LeftRelationMerge27768.owner (.relation 27768) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge27768.deltas
    rows := LeftRelationMerge27768.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge27767.working LeftRelationMerge27768.source
        (relationContext LeftRelationMerge27768.source
          LeftRelationMerge27768.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge27767.working, LeftRelationMerge27768.deltas,
    LeftRelationMerge27768.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 27768)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25851⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11229⟩⟩, ⟨.program ⟨214⟩, ⟨13583⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge27767.working) (working := relationWorking0)
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
end SemanticResult27771

namespace SemanticResult27774
def owner : Owner := ⟨.program ⟨214⟩, ⟨19324⟩⟩
def rawTerms : List Term := Proof.Events108.exact27774RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 27774
def producerEvent : Nat := 27773
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult27774.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨12⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨12⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult27774

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
