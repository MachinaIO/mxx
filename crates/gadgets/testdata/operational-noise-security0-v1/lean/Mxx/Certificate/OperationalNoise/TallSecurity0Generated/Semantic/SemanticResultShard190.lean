import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard190
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard008
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard085
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard086
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard164

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult24805
def owner : Owner := ⟨.program ⟨214⟩, ⟨25157⟩⟩
def rawTerms : List Term := Proof.Events096.exact24805RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24805
def producerEvent : Nat := 24804
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24805.actual selector witness
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
end SemanticResult24805

namespace SemanticResult24810
def owner : Owner := ⟨.program ⟨214⟩, ⟨11788⟩⟩
def rawTerms : List Term := Proof.Events096.exact24810RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24810
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24810.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge24809.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge24809.frameStart)
    (transferEvent := 24808) (owner := owner)
    (leftResult := 1003) (rightResult := 21420)
    (working := LeftOperatorMerge24809.working)
    (reconstruction := LeftOperatorMerge24809.reconstruction)
    (leftReference := .predecessor 0 24806 .coefficient) (rightReference := .predecessor 1 24807 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1003.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge24809.operationAgreement
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
end SemanticResult24810

namespace SemanticResult24815
def owner : Owner := ⟨.program ⟨214⟩, ⟨7353⟩⟩
def rawTerms : List Term := Proof.Events096.exact24815RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24815
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24815.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge24814.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge24814.frameStart)
    (transferEvent := 24813) (owner := owner)
    (leftResult := 21290) (rightResult := 9979)
    (working := LeftOperatorMerge24814.working)
    (reconstruction := LeftOperatorMerge24814.reconstruction)
    (leftReference := .predecessor 0 24811 .coefficient) (rightReference := .predecessor 1 24812 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9979.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge24814.operationAgreement
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
end SemanticResult24815

namespace SemanticResult24819
def owner : Owner := ⟨.program ⟨214⟩, ⟨11789⟩⟩
def rawTerms : List Term := Proof.Events096.exact24819RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24819
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24819.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 24816) (rightBinding := 24817)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7353⟩) (rightExpression := ⟨11788⟩)
    (transferEvent := 24818)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult24815.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult24810.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult24819

namespace SemanticResult24825
def owner : Owner := ⟨.program ⟨214⟩, ⟨11790⟩⟩
def rawTerms : List Term := Proof.Events096.exact24825RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 24825
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24825.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 24822) (survivorTransfer := 24823)
    (survivorEvent := 24824) (resultEvent := resultEvent)
    (rightCoefficientProducer := 9970)
    (owner := owner) (leftOwner := SemanticResult24819.owner)
    (rightOwner := SemanticResult9971.owner)
    (leftResult := 24819) (rightResult := 9971)
    (leftBinding := 24820) (rightBinding := 24821)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11789⟩) (rightExpression := ⟨97⟩)
    (leftActual := SemanticResult24819.actual selector witness)
    (rightActual := SemanticResult9971.actual selector witness)
    (leftRaw := SemanticResult24819.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨97⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound9970.actual selector witness)
    (survivorMagnitude := LeftBound24823.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult24819.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9971.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9970.derived selector witness)
  · exact LeftBound24823.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult24825

namespace SemanticResult24833
def owner : Owner := ⟨.program ⟨214⟩, ⟨11791⟩⟩
def rawTerms : List Term := Proof.Events097.exact24833RawTerms
def summary : Bound := (.finite 24960)
def resultEvent : Nat := 24833
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24833.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨30, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge24831.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge24831.frameStart)
    (owner := owner) (leftOwner := SemanticResult24825.owner)
    (rightOwner := SemanticResult1006.owner)
    (leftResult := 24825) (rightResult := 1006)
    (leftActual := SemanticResult24825.actual selector witness)
    (rightActual := SemanticResult1006.actual selector witness)
    (leftRaw := SemanticResult24825.rawTerms)
    (rightRaw := SemanticResult1006.rawTerms)
    (working := LeftOperatorMerge24831.working)
    (leftBinding := 24826) (rightBinding := 24827)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11790⟩) (rightExpression := ⟨9625⟩)
    (coefficientTransfer := 24828) (summaryTransfer := 24830)
    (rightCoefficientProducer := 1005)
    (rightSummaryTransfer := 24829)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨30, by decide⟩)
    (rightRecordedMaximum := 30)
    (rightSummaryMaximum := ⟨30, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge24831.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1005.actual selector witness)
    (summaryMagnitude := LeftBound24830.actual selector witness)
    (reconstruction := LeftOperatorMerge24831.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult24825.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1006.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1005.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1005.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge24831.operationAgreement
  · exact LeftBound24830.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge24831.working summary) := by
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
end SemanticResult24833

namespace SemanticResult24838
def owner : Owner := ⟨.program ⟨214⟩, ⟨9626⟩⟩
def rawTerms : List Term := Proof.Events097.exact24838RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24838
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24838.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge24837.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge24837.frameStart)
    (transferEvent := 24836) (owner := owner)
    (leftResult := 1006) (rightResult := 21420)
    (working := LeftOperatorMerge24837.working)
    (reconstruction := LeftOperatorMerge24837.reconstruction)
    (leftReference := .predecessor 0 24834 .coefficient) (rightReference := .predecessor 1 24835 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1006.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge24837.operationAgreement
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
end SemanticResult24838

namespace SemanticResult24843
def owner : Owner := ⟨.program ⟨214⟩, ⟨7333⟩⟩
def rawTerms : List Term := Proof.Events097.exact24843RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24843
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24843.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge24842.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge24842.frameStart)
    (transferEvent := 24841) (owner := owner)
    (leftResult := 21290) (rightResult := 10020)
    (working := LeftOperatorMerge24842.working)
    (reconstruction := LeftOperatorMerge24842.reconstruction)
    (leftReference := .predecessor 0 24839 .coefficient) (rightReference := .predecessor 1 24840 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10020.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge24842.operationAgreement
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
end SemanticResult24843

namespace SemanticResult24847
def owner : Owner := ⟨.program ⟨214⟩, ⟨9627⟩⟩
def rawTerms : List Term := Proof.Events097.exact24847RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24847
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24847.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 24844) (rightBinding := 24845)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7333⟩) (rightExpression := ⟨9626⟩)
    (transferEvent := 24846)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult24843.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult24838.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult24847

namespace SemanticResult24853
def owner : Owner := ⟨.program ⟨214⟩, ⟨9628⟩⟩
def rawTerms : List Term := Proof.Events097.exact24853RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 24853
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24853.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 24850) (survivorTransfer := 24851)
    (survivorEvent := 24852) (resultEvent := resultEvent)
    (rightCoefficientProducer := 10011)
    (owner := owner) (leftOwner := SemanticResult24847.owner)
    (rightOwner := SemanticResult10012.owner)
    (leftResult := 24847) (rightResult := 10012)
    (leftBinding := 24848) (rightBinding := 24849)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9627⟩) (rightExpression := ⟨77⟩)
    (leftActual := SemanticResult24847.actual selector witness)
    (rightActual := SemanticResult10012.actual selector witness)
    (leftRaw := SemanticResult24847.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨77⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound10011.actual selector witness)
    (survivorMagnitude := LeftBound24851.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult24847.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10012.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10011.derived selector witness)
  · exact LeftBound24851.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult24853

namespace SemanticResult24863
def owner : Owner := ⟨.program ⟨214⟩, ⟨9629⟩⟩
def rawTerms : List Term := Proof.Events097.exact24863RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 24863
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24863.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge24859.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge24859.frameStart)
    (owner := owner) (leftOwner := SemanticResult24853.owner)
    (rightOwner := SemanticResult10009.owner)
    (leftResult := 24853) (rightResult := 10009)
    (leftActual := SemanticResult24853.actual selector witness)
    (rightActual := SemanticResult10009.actual selector witness)
    (leftRaw := SemanticResult24853.rawTerms)
    (rightRaw := SemanticResult10009.rawTerms)
    (working := LeftOperatorMerge24859.working)
    (leftBinding := 24854) (rightBinding := 24855)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9628⟩) (rightExpression := ⟨7862⟩)
    (coefficientTransfer := 24856) (summaryTransfer := 24858)
    (rightCoefficientProducer := 10008)
    (rightSummaryTransfer := 24857)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge24859.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound10008.actual selector witness)
    (summaryMagnitude := LeftBound24858.actual selector witness)
    (reconstruction := LeftOperatorMerge24859.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult24853.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10009.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10008.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound10008.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge24859.operationAgreement
  · exact LeftBound24858.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge24859.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 24860 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge24859.working
    [{ coefficient := (-1), key := LeftRelationMerge24860.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge24860.frameStart
      LeftRelationMerge24860.owner (.relation 24860) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge24860.deltas
    rows := LeftRelationMerge24860.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge24859.working LeftRelationMerge24860.source
        (relationContext LeftRelationMerge24860.source
          LeftRelationMerge24860.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge24859.working, LeftRelationMerge24860.deltas,
    LeftRelationMerge24860.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 24860)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9629⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge24859.working) (working := relationWorking0)
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
end SemanticResult24863

namespace SemanticResult24869
def owner : Owner := ⟨.program ⟨214⟩, ⟨11792⟩⟩
def rawTerms : List Term := Proof.Events097.exact24869RawTerms
def summary : Bound := (.finite 95445376)
def resultEvent : Nat := 24869
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24869.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge24867.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult24863.owner)
    (rightOwner := SemanticResult24833.owner)
    (leftResult := 24863) (rightResult := 24833)
    (leftActual := SemanticResult24863.actual selector witness)
    (rightActual := SemanticResult24833.actual selector witness)
    (leftRaw := SemanticResult24863.rawTerms)
    (rightRaw := SemanticResult24833.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 24960) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 24864) (rightBinding := 24865)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9629⟩) (rightExpression := ⟨11791⟩)
    (coefficientTransfer := 24866) (summaryTransfer := 24868)
    (base := LeftOperatorMerge24867.base)
    (reconstruction := LeftOperatorMerge24867.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult24863.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult24833.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge24867.operationAgreement
  · rfl
  · decide
end SemanticResult24869

namespace SemanticResult24879
def owner : Owner := ⟨.program ⟨214⟩, ⟨25158⟩⟩
def rawTerms : List Term := Proof.Events097.exact24879RawTerms
def summary : Bound := (.finite 350286057046016)
def resultEvent : Nat := 24879
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24879.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95445376, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge24875.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge24875.frameStart)
    (owner := owner) (leftOwner := SemanticResult24869.owner)
    (rightOwner := SemanticResult24805.owner)
    (leftResult := 24869) (rightResult := 24805)
    (leftActual := SemanticResult24869.actual selector witness)
    (rightActual := SemanticResult24805.actual selector witness)
    (leftRaw := SemanticResult24869.rawTerms)
    (rightRaw := SemanticResult24805.rawTerms)
    (working := LeftOperatorMerge24875.working)
    (leftBinding := 24870) (rightBinding := 24871)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11792⟩) (rightExpression := ⟨25157⟩)
    (coefficientTransfer := 24872) (summaryTransfer := 24874)
    (rightCoefficientProducer := 24804)
    (rightSummaryTransfer := 24873)
    (leftMaximum := ⟨95445376, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge24875.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority24804.actual selector witness)
    (summaryMagnitude := LeftBound24874.actual selector witness)
    (reconstruction := LeftOperatorMerge24875.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult24869.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult24805.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24804.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority24804.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge24875.operationAgreement
  · exact LeftBound24874.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge24875.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 24876 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23086⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23086⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge24875.working
    [{ coefficient := (-1), key := LeftRelationMerge24876.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge24876.frameStart
      LeftRelationMerge24876.owner (.relation 24876) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge24876.deltas
    rows := LeftRelationMerge24876.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge24875.working LeftRelationMerge24876.source
        (relationContext LeftRelationMerge24876.source
          LeftRelationMerge24876.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge24875.working, LeftRelationMerge24876.deltas,
    LeftRelationMerge24876.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 24876)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25158⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25157⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge24875.working) (working := relationWorking0)
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
end SemanticResult24879

namespace SemanticResult24882
def owner : Owner := ⟨.program ⟨214⟩, ⟨19756⟩⟩
def rawTerms : List Term := Proof.Events097.exact24882RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24882
def producerEvent : Nat := 24881
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24882.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨18⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨18⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult24882

namespace SemanticResult24886
def owner : Owner := ⟨.program ⟨214⟩, ⟨19758⟩⟩
def rawTerms : List Term := Proof.Events097.exact24886RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24886
def producerEvent : Nat := 24885
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24886.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 24883 .coefficient) (.value (.predecessor 1 24884 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 24883 .coefficient) (.value (.predecessor 1 24884 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult24886

namespace SemanticResult24964
def owner : Owner := ⟨.program ⟨214⟩, ⟨11785⟩⟩
def rawTerms : List Term := Proof.Events097.exact24964RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24964
def producerEvent : Nat := 24963
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24964.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 24941, .finite 30, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult24964

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
