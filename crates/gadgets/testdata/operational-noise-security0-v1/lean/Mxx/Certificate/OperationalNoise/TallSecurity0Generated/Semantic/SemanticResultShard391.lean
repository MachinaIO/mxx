import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard391
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard085
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard086
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard365

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult54045
def owner : Owner := ⟨.program ⟨214⟩, ⟨28530⟩⟩
def rawTerms : List Term := Proof.Events211.exact54045RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54045
def producerEvent : Nat := 54044
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54045.actual selector witness
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
end SemanticResult54045

namespace SemanticResult54052
def owner : Owner := ⟨.program ⟨214⟩, ⟨23082⟩⟩
def rawTerms : List Term := Proof.Events211.exact54052RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54052
def producerEvent : Nat := 54051
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54052.actual selector witness
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
end SemanticResult54052

namespace SemanticResult54055
def owner : Owner := ⟨.program ⟨214⟩, ⟨25147⟩⟩
def rawTerms : List Term := Proof.Events211.exact54055RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54055
def producerEvent : Nat := 54054
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54055.actual selector witness
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
end SemanticResult54055

namespace SemanticResult54060
def owner : Owner := ⟨.program ⟨214⟩, ⟨11772⟩⟩
def rawTerms : List Term := Proof.Events211.exact54060RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54060
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54060.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54059.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge54059.frameStart)
    (transferEvent := 54058) (owner := owner)
    (leftResult := 2499) (rightResult := 50670)
    (working := LeftOperatorMerge54059.working)
    (reconstruction := LeftOperatorMerge54059.reconstruction)
    (leftReference := .predecessor 0 54056 .coefficient) (rightReference := .predecessor 1 54057 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2499.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge54059.operationAgreement
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
end SemanticResult54060

namespace SemanticResult54065
def owner : Owner := ⟨.program ⟨214⟩, ⟨7277⟩⟩
def rawTerms : List Term := Proof.Events211.exact54065RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54065
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54065.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54064.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge54064.frameStart)
    (transferEvent := 54063) (owner := owner)
    (leftResult := 50540) (rightResult := 9979)
    (working := LeftOperatorMerge54064.working)
    (reconstruction := LeftOperatorMerge54064.reconstruction)
    (leftReference := .predecessor 0 54061 .coefficient) (rightReference := .predecessor 1 54062 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9979.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge54064.operationAgreement
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
end SemanticResult54065

namespace SemanticResult54069
def owner : Owner := ⟨.program ⟨214⟩, ⟨11773⟩⟩
def rawTerms : List Term := Proof.Events211.exact54069RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54069
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54069.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 54066) (rightBinding := 54067)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7277⟩) (rightExpression := ⟨11772⟩)
    (transferEvent := 54068)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54065.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult54060.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult54069

namespace SemanticResult54075
def owner : Owner := ⟨.program ⟨214⟩, ⟨11774⟩⟩
def rawTerms : List Term := Proof.Events211.exact54075RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 54075
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54075.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 54072) (survivorTransfer := 54073)
    (survivorEvent := 54074) (resultEvent := resultEvent)
    (rightCoefficientProducer := 9970)
    (owner := owner) (leftOwner := SemanticResult54069.owner)
    (rightOwner := SemanticResult9971.owner)
    (leftResult := 54069) (rightResult := 9971)
    (leftBinding := 54070) (rightBinding := 54071)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11773⟩) (rightExpression := ⟨97⟩)
    (leftActual := SemanticResult54069.actual selector witness)
    (rightActual := SemanticResult9971.actual selector witness)
    (leftRaw := SemanticResult54069.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨97⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound9970.actual selector witness)
    (survivorMagnitude := LeftBound54073.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54069.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9971.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9970.derived selector witness)
  · exact LeftBound54073.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult54075

namespace SemanticResult54083
def owner : Owner := ⟨.program ⟨214⟩, ⟨11775⟩⟩
def rawTerms : List Term := Proof.Events211.exact54083RawTerms
def summary : Bound := (.finite 24960)
def resultEvent : Nat := 54083
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54083.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨30, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54081.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge54081.frameStart)
    (owner := owner) (leftOwner := SemanticResult54075.owner)
    (rightOwner := SemanticResult2502.owner)
    (leftResult := 54075) (rightResult := 2502)
    (leftActual := SemanticResult54075.actual selector witness)
    (rightActual := SemanticResult2502.actual selector witness)
    (leftRaw := SemanticResult54075.rawTerms)
    (rightRaw := SemanticResult2502.rawTerms)
    (working := LeftOperatorMerge54081.working)
    (leftBinding := 54076) (rightBinding := 54077)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11774⟩) (rightExpression := ⟨9615⟩)
    (coefficientTransfer := 54078) (summaryTransfer := 54080)
    (rightCoefficientProducer := 2501)
    (rightSummaryTransfer := 54079)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨30, by decide⟩)
    (rightRecordedMaximum := 30)
    (rightSummaryMaximum := ⟨30, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge54081.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority2501.actual selector witness)
    (summaryMagnitude := LeftBound54080.actual selector witness)
    (reconstruction := LeftOperatorMerge54081.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54075.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2502.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2501.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority2501.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge54081.operationAgreement
  · exact LeftBound54080.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54081.working summary) := by
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
end SemanticResult54083

namespace SemanticResult54088
def owner : Owner := ⟨.program ⟨214⟩, ⟨9616⟩⟩
def rawTerms : List Term := Proof.Events211.exact54088RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54088
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54088.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54087.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge54087.frameStart)
    (transferEvent := 54086) (owner := owner)
    (leftResult := 2502) (rightResult := 50670)
    (working := LeftOperatorMerge54087.working)
    (reconstruction := LeftOperatorMerge54087.reconstruction)
    (leftReference := .predecessor 0 54084 .coefficient) (rightReference := .predecessor 1 54085 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2502.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge54087.operationAgreement
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
end SemanticResult54088

namespace SemanticResult54093
def owner : Owner := ⟨.program ⟨214⟩, ⟨7257⟩⟩
def rawTerms : List Term := Proof.Events211.exact54093RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54093
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54093.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54092.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge54092.frameStart)
    (transferEvent := 54091) (owner := owner)
    (leftResult := 50540) (rightResult := 10020)
    (working := LeftOperatorMerge54092.working)
    (reconstruction := LeftOperatorMerge54092.reconstruction)
    (leftReference := .predecessor 0 54089 .coefficient) (rightReference := .predecessor 1 54090 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10020.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge54092.operationAgreement
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
end SemanticResult54093

namespace SemanticResult54097
def owner : Owner := ⟨.program ⟨214⟩, ⟨9617⟩⟩
def rawTerms : List Term := Proof.Events211.exact54097RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54097
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54097.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 54094) (rightBinding := 54095)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7257⟩) (rightExpression := ⟨9616⟩)
    (transferEvent := 54096)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54093.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult54088.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult54097

namespace SemanticResult54103
def owner : Owner := ⟨.program ⟨214⟩, ⟨9618⟩⟩
def rawTerms : List Term := Proof.Events211.exact54103RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 54103
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54103.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 54100) (survivorTransfer := 54101)
    (survivorEvent := 54102) (resultEvent := resultEvent)
    (rightCoefficientProducer := 10011)
    (owner := owner) (leftOwner := SemanticResult54097.owner)
    (rightOwner := SemanticResult10012.owner)
    (leftResult := 54097) (rightResult := 10012)
    (leftBinding := 54098) (rightBinding := 54099)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9617⟩) (rightExpression := ⟨77⟩)
    (leftActual := SemanticResult54097.actual selector witness)
    (rightActual := SemanticResult10012.actual selector witness)
    (leftRaw := SemanticResult54097.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨77⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound10011.actual selector witness)
    (survivorMagnitude := LeftBound54101.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54097.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10012.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10011.derived selector witness)
  · exact LeftBound54101.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult54103

namespace SemanticResult54113
def owner : Owner := ⟨.program ⟨214⟩, ⟨9619⟩⟩
def rawTerms : List Term := Proof.Events211.exact54113RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 54113
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54113.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54109.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge54109.frameStart)
    (owner := owner) (leftOwner := SemanticResult54103.owner)
    (rightOwner := SemanticResult10009.owner)
    (leftResult := 54103) (rightResult := 10009)
    (leftActual := SemanticResult54103.actual selector witness)
    (rightActual := SemanticResult10009.actual selector witness)
    (leftRaw := SemanticResult54103.rawTerms)
    (rightRaw := SemanticResult10009.rawTerms)
    (working := LeftOperatorMerge54109.working)
    (leftBinding := 54104) (rightBinding := 54105)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9618⟩) (rightExpression := ⟨7862⟩)
    (coefficientTransfer := 54106) (summaryTransfer := 54108)
    (rightCoefficientProducer := 10008)
    (rightSummaryTransfer := 54107)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge54109.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound10008.actual selector witness)
    (summaryMagnitude := LeftBound54108.actual selector witness)
    (reconstruction := LeftOperatorMerge54109.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54103.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10009.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10008.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound10008.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge54109.operationAgreement
  · exact LeftBound54108.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54109.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 54110 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9615⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge54109.working
    [{ coefficient := (-1), key := LeftRelationMerge54110.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge54110.frameStart
      LeftRelationMerge54110.owner (.relation 54110) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge54110.deltas
    rows := LeftRelationMerge54110.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge54109.working LeftRelationMerge54110.source
        (relationContext LeftRelationMerge54110.source
          LeftRelationMerge54110.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge54109.working, LeftRelationMerge54110.deltas,
    LeftRelationMerge54110.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 54110)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9619⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9615⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge54109.working) (working := relationWorking0)
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
end SemanticResult54113

namespace SemanticResult54119
def owner : Owner := ⟨.program ⟨214⟩, ⟨11776⟩⟩
def rawTerms : List Term := Proof.Events211.exact54119RawTerms
def summary : Bound := (.finite 95445376)
def resultEvent : Nat := 54119
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54119.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge54117.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult54113.owner)
    (rightOwner := SemanticResult54083.owner)
    (leftResult := 54113) (rightResult := 54083)
    (leftActual := SemanticResult54113.actual selector witness)
    (rightActual := SemanticResult54083.actual selector witness)
    (leftRaw := SemanticResult54113.rawTerms)
    (rightRaw := SemanticResult54083.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 24960) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 54114) (rightBinding := 54115)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9619⟩) (rightExpression := ⟨11775⟩)
    (coefficientTransfer := 54116) (summaryTransfer := 54118)
    (base := LeftOperatorMerge54117.base)
    (reconstruction := LeftOperatorMerge54117.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54113.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult54083.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge54117.operationAgreement
  · rfl
  · decide
end SemanticResult54119

namespace SemanticResult54129
def owner : Owner := ⟨.program ⟨214⟩, ⟨25148⟩⟩
def rawTerms : List Term := Proof.Events211.exact54129RawTerms
def summary : Bound := (.finite 350286057046016)
def resultEvent : Nat := 54129
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54129.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95445376, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54125.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge54125.frameStart)
    (owner := owner) (leftOwner := SemanticResult54119.owner)
    (rightOwner := SemanticResult54055.owner)
    (leftResult := 54119) (rightResult := 54055)
    (leftActual := SemanticResult54119.actual selector witness)
    (rightActual := SemanticResult54055.actual selector witness)
    (leftRaw := SemanticResult54119.rawTerms)
    (rightRaw := SemanticResult54055.rawTerms)
    (working := LeftOperatorMerge54125.working)
    (leftBinding := 54120) (rightBinding := 54121)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11776⟩) (rightExpression := ⟨25147⟩)
    (coefficientTransfer := 54122) (summaryTransfer := 54124)
    (rightCoefficientProducer := 54054)
    (rightSummaryTransfer := 54123)
    (leftMaximum := ⟨95445376, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge54125.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority54054.actual selector witness)
    (summaryMagnitude := LeftBound54124.actual selector witness)
    (reconstruction := LeftOperatorMerge54125.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54119.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult54055.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54054.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority54054.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge54125.operationAgreement
  · exact LeftBound54124.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54125.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 54126 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25147⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9615⟩⟩, ⟨.program ⟨214⟩, ⟨11769⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23082⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23082⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge54125.working
    [{ coefficient := (-1), key := LeftRelationMerge54126.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge54126.frameStart
      LeftRelationMerge54126.owner (.relation 54126) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge54126.deltas
    rows := LeftRelationMerge54126.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge54125.working LeftRelationMerge54126.source
        (relationContext LeftRelationMerge54126.source
          LeftRelationMerge54126.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge54125.working, LeftRelationMerge54126.deltas,
    LeftRelationMerge54126.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 54126)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25148⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9615⟩⟩, ⟨.program ⟨214⟩, ⟨11769⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25147⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25147⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge54125.working) (working := relationWorking0)
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
end SemanticResult54129

namespace SemanticResult54132
def owner : Owner := ⟨.program ⟨214⟩, ⟨19748⟩⟩
def rawTerms : List Term := Proof.Events211.exact54132RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54132
def producerEvent : Nat := 54131
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54132.actual selector witness
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
end SemanticResult54132

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
