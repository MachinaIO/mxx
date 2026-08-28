import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard065
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard055
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard056

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult7466
def owner : Owner := ⟨.program ⟨214⟩, ⟨102⟩⟩
def rawTerms : List Term := Proof.Events029.exact7466RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7466
def producerEvent : Nat := 7465
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7466.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 7464 .coefficient), 0, .finite 26, .identity (.predecessor 0 7464 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult7466

namespace SemanticResult7471
def owner : Owner := ⟨.program ⟨214⟩, ⟨12993⟩⟩
def rawTerms : List Term := Proof.Events029.exact7471RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7471
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7471.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge7470.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge7470.frameStart)
    (transferEvent := 7469) (owner := owner)
    (leftResult := 97) (rightResult := 6449)
    (working := LeftOperatorMerge7470.working)
    (reconstruction := LeftOperatorMerge7470.reconstruction)
    (leftReference := .predecessor 0 7467 .coefficient) (rightReference := .predecessor 1 7468 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult97.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6449.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge7470.operationAgreement
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
end SemanticResult7471

namespace SemanticResult7474
def owner : Owner := ⟨.program ⟨214⟩, ⟨6788⟩⟩
def rawTerms : List Term := Proof.Events029.exact7474RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7474
def producerEvent : Nat := 7473
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7474.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 7472 .coefficient), 0, .large, .identity (.predecessor 0 7472 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult7474

namespace SemanticResult7479
def owner : Owner := ⟨.program ⟨214⟩, ⟨7396⟩⟩
def rawTerms : List Term := Proof.Events029.exact7479RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7479
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7479.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge7478.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge7478.frameStart)
    (transferEvent := 7477) (owner := owner)
    (leftResult := 6314) (rightResult := 7474)
    (working := LeftOperatorMerge7478.working)
    (reconstruction := LeftOperatorMerge7478.reconstruction)
    (leftReference := .predecessor 0 7475 .coefficient) (rightReference := .predecessor 1 7476 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult6314.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7474.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge7478.operationAgreement
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
end SemanticResult7479

namespace SemanticResult7483
def owner : Owner := ⟨.program ⟨214⟩, ⟨12994⟩⟩
def rawTerms : List Term := Proof.Events029.exact7483RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7483
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7483.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7480) (rightBinding := 7481)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7396⟩) (rightExpression := ⟨12993⟩)
    (transferEvent := 7482)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7479.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7471.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7483

namespace SemanticResult7489
def owner : Owner := ⟨.program ⟨214⟩, ⟨12995⟩⟩
def rawTerms : List Term := Proof.Events029.exact7489RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 7489
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7489.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 7486) (survivorTransfer := 7487)
    (survivorEvent := 7488) (resultEvent := resultEvent)
    (rightCoefficientProducer := 7465)
    (owner := owner) (leftOwner := SemanticResult7483.owner)
    (rightOwner := SemanticResult7466.owner)
    (leftResult := 7483) (rightResult := 7466)
    (leftBinding := 7484) (rightBinding := 7485)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12994⟩) (rightExpression := ⟨102⟩)
    (leftActual := SemanticResult7483.actual selector witness)
    (rightActual := SemanticResult7466.actual selector witness)
    (leftRaw := SemanticResult7483.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨102⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound7465.actual selector witness)
    (survivorMagnitude := LeftBound7487.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7483.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7466.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7465.derived selector witness)
  · exact LeftBound7487.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult7489

namespace SemanticResult7497
def owner : Owner := ⟨.program ⟨214⟩, ⟨12996⟩⟩
def rawTerms : List Term := Proof.Events029.exact7497RawTerms
def summary : Bound := (.finite 43264)
def resultEvent : Nat := 7497
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7497.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨52, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge7495.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge7495.frameStart)
    (owner := owner) (leftOwner := SemanticResult7489.owner)
    (rightOwner := SemanticResult100.owner)
    (leftResult := 7489) (rightResult := 100)
    (leftActual := SemanticResult7489.actual selector witness)
    (rightActual := SemanticResult100.actual selector witness)
    (leftRaw := SemanticResult7489.rawTerms)
    (rightRaw := SemanticResult100.rawTerms)
    (working := LeftOperatorMerge7495.working)
    (leftBinding := 7490) (rightBinding := 7491)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12995⟩) (rightExpression := ⟨10155⟩)
    (coefficientTransfer := 7492) (summaryTransfer := 7494)
    (rightCoefficientProducer := 99)
    (rightSummaryTransfer := 7493)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨52, by decide⟩)
    (rightRecordedMaximum := 52)
    (rightSummaryMaximum := ⟨52, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge7495.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority99.actual selector witness)
    (summaryMagnitude := LeftBound7494.actual selector witness)
    (reconstruction := LeftOperatorMerge7495.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7489.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority99.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge7495.operationAgreement
  · exact LeftBound7494.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge7495.working summary) := by
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
end SemanticResult7497

namespace SemanticResult7500
def owner : Owner := ⟨.program ⟨214⟩, ⟨7876⟩⟩
def rawTerms : List Term := Proof.Events029.exact7500RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7500
def producerEvent : Nat := 7499
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7500.actual selector witness
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
end SemanticResult7500

namespace SemanticResult7504
def owner : Owner := ⟨.program ⟨214⟩, ⟨7877⟩⟩
def rawTerms : List Term := Proof.Events029.exact7504RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7504
def producerEvent : Nat := 7503
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7504.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 7501 .coefficient) (.value (.predecessor 1 7502 .coefficient)), 0, .finite 8192, .scale (.predecessor 0 7501 .coefficient) (.value (.predecessor 1 7502 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult7504

namespace SemanticResult7507
def owner : Owner := ⟨.program ⟨214⟩, ⟨82⟩⟩
def rawTerms : List Term := Proof.Events029.exact7507RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7507
def producerEvent : Nat := 7506
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7507.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 7505 .coefficient), 0, .finite 26, .identity (.predecessor 0 7505 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult7507

namespace SemanticResult7512
def owner : Owner := ⟨.program ⟨214⟩, ⟨10156⟩⟩
def rawTerms : List Term := Proof.Events029.exact7512RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7512
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7512.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge7511.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge7511.frameStart)
    (transferEvent := 7510) (owner := owner)
    (leftResult := 100) (rightResult := 6449)
    (working := LeftOperatorMerge7511.working)
    (reconstruction := LeftOperatorMerge7511.reconstruction)
    (leftReference := .predecessor 0 7508 .coefficient) (rightReference := .predecessor 1 7509 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult100.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6449.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge7511.operationAgreement
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
end SemanticResult7512

namespace SemanticResult7515
def owner : Owner := ⟨.program ⟨214⟩, ⟨6768⟩⟩
def rawTerms : List Term := Proof.Events029.exact7515RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7515
def producerEvent : Nat := 7514
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7515.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 7513 .coefficient), 0, .large, .identity (.predecessor 0 7513 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult7515

namespace SemanticResult7520
def owner : Owner := ⟨.program ⟨214⟩, ⟨7376⟩⟩
def rawTerms : List Term := Proof.Events029.exact7520RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7520
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7520.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge7519.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge7519.frameStart)
    (transferEvent := 7518) (owner := owner)
    (leftResult := 6314) (rightResult := 7515)
    (working := LeftOperatorMerge7519.working)
    (reconstruction := LeftOperatorMerge7519.reconstruction)
    (leftReference := .predecessor 0 7516 .coefficient) (rightReference := .predecessor 1 7517 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult6314.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7515.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge7519.operationAgreement
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
end SemanticResult7520

namespace SemanticResult7524
def owner : Owner := ⟨.program ⟨214⟩, ⟨10157⟩⟩
def rawTerms : List Term := Proof.Events029.exact7524RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 7524
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7524.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 7521) (rightBinding := 7522)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7376⟩) (rightExpression := ⟨10156⟩)
    (transferEvent := 7523)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7520.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7512.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult7524

namespace SemanticResult7530
def owner : Owner := ⟨.program ⟨214⟩, ⟨10158⟩⟩
def rawTerms : List Term := Proof.Events029.exact7530RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 7530
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7530.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 7527) (survivorTransfer := 7528)
    (survivorEvent := 7529) (resultEvent := resultEvent)
    (rightCoefficientProducer := 7506)
    (owner := owner) (leftOwner := SemanticResult7524.owner)
    (rightOwner := SemanticResult7507.owner)
    (leftResult := 7524) (rightResult := 7507)
    (leftBinding := 7525) (rightBinding := 7526)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10157⟩) (rightExpression := ⟨82⟩)
    (leftActual := SemanticResult7524.actual selector witness)
    (rightActual := SemanticResult7507.actual selector witness)
    (leftRaw := SemanticResult7524.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨82⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound7506.actual selector witness)
    (survivorMagnitude := LeftBound7528.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7524.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7507.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7506.derived selector witness)
  · exact LeftBound7528.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult7530

namespace SemanticResult7540
def owner : Owner := ⟨.program ⟨214⟩, ⟨10159⟩⟩
def rawTerms : List Term := Proof.Events029.exact7540RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 7540
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult7540.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge7536.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge7536.frameStart)
    (owner := owner) (leftOwner := SemanticResult7530.owner)
    (rightOwner := SemanticResult7504.owner)
    (leftResult := 7530) (rightResult := 7504)
    (leftActual := SemanticResult7530.actual selector witness)
    (rightActual := SemanticResult7504.actual selector witness)
    (leftRaw := SemanticResult7530.rawTerms)
    (rightRaw := SemanticResult7504.rawTerms)
    (working := LeftOperatorMerge7536.working)
    (leftBinding := 7531) (rightBinding := 7532)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10158⟩) (rightExpression := ⟨7877⟩)
    (coefficientTransfer := 7533) (summaryTransfer := 7535)
    (rightCoefficientProducer := 7503)
    (rightSummaryTransfer := 7534)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge7536.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound7503.actual selector witness)
    (summaryMagnitude := LeftBound7535.actual selector witness)
    (reconstruction := LeftOperatorMerge7536.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult7530.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7504.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7503.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound7503.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge7536.operationAgreement
  · exact LeftBound7535.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge7536.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 7537 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge7536.working
    [{ coefficient := (-1), key := LeftRelationMerge7537.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge7537.frameStart
      LeftRelationMerge7537.owner (.relation 7537) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge7537.deltas
    rows := LeftRelationMerge7537.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge7536.working LeftRelationMerge7537.source
        (relationContext LeftRelationMerge7537.source
          LeftRelationMerge7537.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge7536.working, LeftRelationMerge7537.deltas,
    LeftRelationMerge7537.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 7537)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10159⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10155⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge7536.working) (working := relationWorking0)
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
end SemanticResult7540

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
