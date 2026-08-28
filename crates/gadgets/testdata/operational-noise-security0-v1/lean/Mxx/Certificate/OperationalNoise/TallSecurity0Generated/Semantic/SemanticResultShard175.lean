import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard175
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard007
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard008
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard069
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard173
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard174

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult22857
def owner : Owner := ⟨.program ⟨214⟩, ⟨29644⟩⟩
def rawTerms : List Term := Proof.Events089.exact22857RawTerms
def summary : Bound := (.finite 1292449485504936292352)
def resultEvent : Nat := 22857
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22857.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge22854.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult22850.owner)
    (rightOwner := SemanticResult22672.owner)
    (leftResult := 22850) (rightResult := 22672)
    (leftActual := SemanticResult22850.actual selector witness)
    (rightActual := SemanticResult22672.actual selector witness)
    (leftRaw := SemanticResult22850.rawTerms)
    (rightRaw := SemanticResult22672.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292449483693632782336) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 22851) (rightBinding := 22852)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22567⟩) (rightExpression := ⟨29643⟩)
    (coefficientTransfer := 22853) (summaryTransfer := 22856)
    (base := LeftOperatorMerge22854.base)
    (reconstruction := LeftOperatorMerge22854.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult22850.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult22672.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge22854.operationAgreement
  · rfl
  · decide
end SemanticResult22857

namespace SemanticResult22864
def owner : Owner := ⟨.program ⟨214⟩, ⟨24612⟩⟩
def rawTerms : List Term := Proof.Events089.exact22864RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 22864
def producerEvent : Nat := 22863
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22864.actual selector witness
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
end SemanticResult22864

namespace SemanticResult22867
def owner : Owner := ⟨.program ⟨214⟩, ⟨29424⟩⟩
def rawTerms : List Term := Proof.Events089.exact22867RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 22867
def producerEvent : Nat := 22866
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22867.actual selector witness
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
end SemanticResult22867

namespace SemanticResult22874
def owner : Owner := ⟨.program ⟨214⟩, ⟨23296⟩⟩
def rawTerms : List Term := Proof.Events089.exact22874RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 22874
def producerEvent : Nat := 22873
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22874.actual selector witness
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
end SemanticResult22874

namespace SemanticResult22877
def owner : Owner := ⟨.program ⟨214⟩, ⟨25542⟩⟩
def rawTerms : List Term := Proof.Events089.exact22877RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 22877
def producerEvent : Nat := 22876
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22877.actual selector witness
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
end SemanticResult22877

namespace SemanticResult22882
def owner : Owner := ⟨.program ⟨214⟩, ⟨12789⟩⟩
def rawTerms : List Term := Proof.Events089.exact22882RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 22882
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22882.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge22881.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge22881.frameStart)
    (transferEvent := 22880) (owner := owner)
    (leftResult := 911) (rightResult := 21420)
    (working := LeftOperatorMerge22881.working)
    (reconstruction := LeftOperatorMerge22881.reconstruction)
    (leftReference := .predecessor 0 22878 .coefficient) (rightReference := .predecessor 1 22879 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult911.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge22881.operationAgreement
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
end SemanticResult22882

namespace SemanticResult22887
def owner : Owner := ⟨.program ⟨214⟩, ⟨7357⟩⟩
def rawTerms : List Term := Proof.Events089.exact22887RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 22887
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22887.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge22886.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge22886.frameStart)
    (transferEvent := 22885) (owner := owner)
    (leftResult := 21290) (rightResult := 7975)
    (working := LeftOperatorMerge22886.working)
    (reconstruction := LeftOperatorMerge22886.reconstruction)
    (leftReference := .predecessor 0 22883 .coefficient) (rightReference := .predecessor 1 22884 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7975.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge22886.operationAgreement
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
end SemanticResult22887

namespace SemanticResult22891
def owner : Owner := ⟨.program ⟨214⟩, ⟨12790⟩⟩
def rawTerms : List Term := Proof.Events089.exact22891RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 22891
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22891.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 22888) (rightBinding := 22889)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7357⟩) (rightExpression := ⟨12789⟩)
    (transferEvent := 22890)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult22887.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult22882.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult22891

namespace SemanticResult22897
def owner : Owner := ⟨.program ⟨214⟩, ⟨12791⟩⟩
def rawTerms : List Term := Proof.Events089.exact22897RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 22897
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22897.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 22894) (survivorTransfer := 22895)
    (survivorEvent := 22896) (resultEvent := resultEvent)
    (rightCoefficientProducer := 7966)
    (owner := owner) (leftOwner := SemanticResult22891.owner)
    (rightOwner := SemanticResult7967.owner)
    (leftResult := 22891) (rightResult := 7967)
    (leftBinding := 22892) (rightBinding := 22893)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12790⟩) (rightExpression := ⟨101⟩)
    (leftActual := SemanticResult22891.actual selector witness)
    (rightActual := SemanticResult7967.actual selector witness)
    (leftRaw := SemanticResult22891.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨101⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound7966.actual selector witness)
    (survivorMagnitude := LeftBound22895.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult22891.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7967.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7966.derived selector witness)
  · exact LeftBound22895.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult22897

namespace SemanticResult22905
def owner : Owner := ⟨.program ⟨214⟩, ⟨12792⟩⟩
def rawTerms : List Term := Proof.Events089.exact22905RawTerms
def summary : Bound := (.finite 38272)
def resultEvent : Nat := 22905
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22905.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨46, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge22903.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge22903.frameStart)
    (owner := owner) (leftOwner := SemanticResult22897.owner)
    (rightOwner := SemanticResult914.owner)
    (leftResult := 22897) (rightResult := 914)
    (leftActual := SemanticResult22897.actual selector witness)
    (rightActual := SemanticResult914.actual selector witness)
    (leftRaw := SemanticResult22897.rawTerms)
    (rightRaw := SemanticResult914.rawTerms)
    (working := LeftOperatorMerge22903.working)
    (leftBinding := 22898) (rightBinding := 22899)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12791⟩) (rightExpression := ⟨10045⟩)
    (coefficientTransfer := 22900) (summaryTransfer := 22902)
    (rightCoefficientProducer := 913)
    (rightSummaryTransfer := 22901)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨46, by decide⟩)
    (rightRecordedMaximum := 46)
    (rightSummaryMaximum := ⟨46, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge22903.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority913.actual selector witness)
    (summaryMagnitude := LeftBound22902.actual selector witness)
    (reconstruction := LeftOperatorMerge22903.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult22897.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult914.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority913.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority913.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge22903.operationAgreement
  · exact LeftBound22902.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge22903.working summary) := by
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
end SemanticResult22905

namespace SemanticResult22910
def owner : Owner := ⟨.program ⟨214⟩, ⟨10046⟩⟩
def rawTerms : List Term := Proof.Events089.exact22910RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 22910
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22910.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge22909.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge22909.frameStart)
    (transferEvent := 22908) (owner := owner)
    (leftResult := 914) (rightResult := 21420)
    (working := LeftOperatorMerge22909.working)
    (reconstruction := LeftOperatorMerge22909.reconstruction)
    (leftReference := .predecessor 0 22906 .coefficient) (rightReference := .predecessor 1 22907 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult914.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge22909.operationAgreement
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
end SemanticResult22910

namespace SemanticResult22915
def owner : Owner := ⟨.program ⟨214⟩, ⟨7337⟩⟩
def rawTerms : List Term := Proof.Events089.exact22915RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 22915
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22915.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge22914.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge22914.frameStart)
    (transferEvent := 22913) (owner := owner)
    (leftResult := 21290) (rightResult := 8016)
    (working := LeftOperatorMerge22914.working)
    (reconstruction := LeftOperatorMerge22914.reconstruction)
    (leftReference := .predecessor 0 22911 .coefficient) (rightReference := .predecessor 1 22912 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8016.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge22914.operationAgreement
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
end SemanticResult22915

namespace SemanticResult22919
def owner : Owner := ⟨.program ⟨214⟩, ⟨10047⟩⟩
def rawTerms : List Term := Proof.Events089.exact22919RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 22919
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22919.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 22916) (rightBinding := 22917)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7337⟩) (rightExpression := ⟨10046⟩)
    (transferEvent := 22918)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult22915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult22910.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult22919

namespace SemanticResult22925
def owner : Owner := ⟨.program ⟨214⟩, ⟨10048⟩⟩
def rawTerms : List Term := Proof.Events089.exact22925RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 22925
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22925.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 22922) (survivorTransfer := 22923)
    (survivorEvent := 22924) (resultEvent := resultEvent)
    (rightCoefficientProducer := 8007)
    (owner := owner) (leftOwner := SemanticResult22919.owner)
    (rightOwner := SemanticResult8008.owner)
    (leftResult := 22919) (rightResult := 8008)
    (leftBinding := 22920) (rightBinding := 22921)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10047⟩) (rightExpression := ⟨81⟩)
    (leftActual := SemanticResult22919.actual selector witness)
    (rightActual := SemanticResult8008.actual selector witness)
    (leftRaw := SemanticResult22919.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨81⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound8007.actual selector witness)
    (survivorMagnitude := LeftBound22923.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult22919.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8008.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8007.derived selector witness)
  · exact LeftBound22923.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult22925

namespace SemanticResult22935
def owner : Owner := ⟨.program ⟨214⟩, ⟨10049⟩⟩
def rawTerms : List Term := Proof.Events089.exact22935RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 22935
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22935.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge22931.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge22931.frameStart)
    (owner := owner) (leftOwner := SemanticResult22925.owner)
    (rightOwner := SemanticResult8005.owner)
    (leftResult := 22925) (rightResult := 8005)
    (leftActual := SemanticResult22925.actual selector witness)
    (rightActual := SemanticResult8005.actual selector witness)
    (leftRaw := SemanticResult22925.rawTerms)
    (rightRaw := SemanticResult8005.rawTerms)
    (working := LeftOperatorMerge22931.working)
    (leftBinding := 22926) (rightBinding := 22927)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10048⟩) (rightExpression := ⟨7874⟩)
    (coefficientTransfer := 22928) (summaryTransfer := 22930)
    (rightCoefficientProducer := 8004)
    (rightSummaryTransfer := 22929)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge22931.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound8004.actual selector witness)
    (summaryMagnitude := LeftBound22930.actual selector witness)
    (reconstruction := LeftOperatorMerge22931.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult22925.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8005.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8004.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound8004.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge22931.operationAgreement
  · exact LeftBound22930.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge22931.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 22932 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10045⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge22931.working
    [{ coefficient := (-1), key := LeftRelationMerge22932.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge22932.frameStart
      LeftRelationMerge22932.owner (.relation 22932) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge22932.deltas
    rows := LeftRelationMerge22932.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge22931.working LeftRelationMerge22932.source
        (relationContext LeftRelationMerge22932.source
          LeftRelationMerge22932.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge22931.working, LeftRelationMerge22932.deltas,
    LeftRelationMerge22932.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 22932)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10049⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10045⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge22931.working) (working := relationWorking0)
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
end SemanticResult22935

namespace SemanticResult22941
def owner : Owner := ⟨.program ⟨214⟩, ⟨12793⟩⟩
def rawTerms : List Term := Proof.Events089.exact22941RawTerms
def summary : Bound := (.finite 95458688)
def resultEvent : Nat := 22941
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult22941.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge22939.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult22935.owner)
    (rightOwner := SemanticResult22905.owner)
    (leftResult := 22935) (rightResult := 22905)
    (leftActual := SemanticResult22935.actual selector witness)
    (rightActual := SemanticResult22905.actual selector witness)
    (leftRaw := SemanticResult22935.rawTerms)
    (rightRaw := SemanticResult22905.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 38272) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 22936) (rightBinding := 22937)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10049⟩) (rightExpression := ⟨12792⟩)
    (coefficientTransfer := 22938) (summaryTransfer := 22940)
    (base := LeftOperatorMerge22939.base)
    (reconstruction := LeftOperatorMerge22939.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult22935.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult22905.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge22939.operationAgreement
  · rfl
  · decide
end SemanticResult22941

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
