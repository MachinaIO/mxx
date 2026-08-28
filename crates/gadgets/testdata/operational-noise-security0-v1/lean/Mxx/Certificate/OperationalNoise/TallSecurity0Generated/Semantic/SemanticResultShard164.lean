import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard007
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard057
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard163

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult21415
def owner : Owner := ⟨.program ⟨214⟩, ⟨25773⟩⟩
def rawTerms : List Term := Proof.Events083.exact21415RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21415
def producerEvent : Nat := 21414
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21415.actual selector witness
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
end SemanticResult21415

namespace SemanticResult21420
def owner : Owner := ⟨.program ⟨214⟩, ⟨6570⟩⟩
def rawTerms : List Term := Proof.Events083.exact21420RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21420
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21420.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21419.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge21419.frameStart)
    (transferEvent := 21418) (owner := owner)
    (leftResult := 21290) (rightResult := 2)
    (working := LeftOperatorMerge21419.working)
    (reconstruction := LeftOperatorMerge21419.reconstruction)
    (leftReference := .predecessor 0 21416 .coefficient) (rightReference := .predecessor 1 21417 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge21419.operationAgreement
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
end SemanticResult21420

namespace SemanticResult21425
def owner : Owner := ⟨.program ⟨214⟩, ⟨13377⟩⟩
def rawTerms : List Term := Proof.Events083.exact21425RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21425
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21425.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21424.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge21424.frameStart)
    (transferEvent := 21423) (owner := owner)
    (leftResult := 842) (rightResult := 21420)
    (working := LeftOperatorMerge21424.working)
    (reconstruction := LeftOperatorMerge21424.reconstruction)
    (leftReference := .predecessor 0 21421 .coefficient) (rightReference := .predecessor 1 21422 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult842.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge21424.operationAgreement
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
end SemanticResult21425

namespace SemanticResult21430
def owner : Owner := ⟨.program ⟨214⟩, ⟨7360⟩⟩
def rawTerms : List Term := Proof.Events083.exact21430RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21430
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21430.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21429.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge21429.frameStart)
    (transferEvent := 21428) (owner := owner)
    (leftResult := 21290) (rightResult := 6457)
    (working := LeftOperatorMerge21429.working)
    (reconstruction := LeftOperatorMerge21429.reconstruction)
    (leftReference := .predecessor 0 21426 .coefficient) (rightReference := .predecessor 1 21427 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6457.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge21429.operationAgreement
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
end SemanticResult21430

namespace SemanticResult21434
def owner : Owner := ⟨.program ⟨214⟩, ⟨13378⟩⟩
def rawTerms : List Term := Proof.Events083.exact21434RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21434
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21434.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 21431) (rightBinding := 21432)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7360⟩) (rightExpression := ⟨13377⟩)
    (transferEvent := 21433)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21430.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21425.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult21434

namespace SemanticResult21440
def owner : Owner := ⟨.program ⟨214⟩, ⟨13379⟩⟩
def rawTerms : List Term := Proof.Events083.exact21440RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 21440
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21440.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 21437) (survivorTransfer := 21438)
    (survivorEvent := 21439) (resultEvent := resultEvent)
    (rightCoefficientProducer := 6443)
    (owner := owner) (leftOwner := SemanticResult21434.owner)
    (rightOwner := SemanticResult6444.owner)
    (leftResult := 21434) (rightResult := 6444)
    (leftBinding := 21435) (rightBinding := 21436)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13378⟩) (rightExpression := ⟨104⟩)
    (leftActual := SemanticResult21434.actual selector witness)
    (rightActual := SemanticResult6444.actual selector witness)
    (leftRaw := SemanticResult21434.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨104⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound6443.actual selector witness)
    (survivorMagnitude := LeftBound21438.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21434.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6444.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6443.derived selector witness)
  · exact LeftBound21438.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult21440

namespace SemanticResult21448
def owner : Owner := ⟨.program ⟨214⟩, ⟨13380⟩⟩
def rawTerms : List Term := Proof.Events083.exact21448RawTerms
def summary : Bound := (.finite 49920)
def resultEvent : Nat := 21448
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21448.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨60, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21446.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge21446.frameStart)
    (owner := owner) (leftOwner := SemanticResult21440.owner)
    (rightOwner := SemanticResult845.owner)
    (leftResult := 21440) (rightResult := 845)
    (leftActual := SemanticResult21440.actual selector witness)
    (rightActual := SemanticResult845.actual selector witness)
    (leftRaw := SemanticResult21440.rawTerms)
    (rightRaw := SemanticResult845.rawTerms)
    (working := LeftOperatorMerge21446.working)
    (leftBinding := 21441) (rightBinding := 21442)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13379⟩) (rightExpression := ⟨10360⟩)
    (coefficientTransfer := 21443) (summaryTransfer := 21445)
    (rightCoefficientProducer := 844)
    (rightSummaryTransfer := 21444)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨60, by decide⟩)
    (rightRecordedMaximum := 60)
    (rightSummaryMaximum := ⟨60, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge21446.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority844.actual selector witness)
    (summaryMagnitude := LeftBound21445.actual selector witness)
    (reconstruction := LeftOperatorMerge21446.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21440.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult845.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority844.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority844.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge21446.operationAgreement
  · exact LeftBound21445.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21446.working summary) := by
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
end SemanticResult21448

namespace SemanticResult21453
def owner : Owner := ⟨.program ⟨214⟩, ⟨10361⟩⟩
def rawTerms : List Term := Proof.Events083.exact21453RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21453
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21453.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21452.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge21452.frameStart)
    (transferEvent := 21451) (owner := owner)
    (leftResult := 845) (rightResult := 21420)
    (working := LeftOperatorMerge21452.working)
    (reconstruction := LeftOperatorMerge21452.reconstruction)
    (leftReference := .predecessor 0 21449 .coefficient) (rightReference := .predecessor 1 21450 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult845.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge21452.operationAgreement
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
end SemanticResult21453

namespace SemanticResult21458
def owner : Owner := ⟨.program ⟨214⟩, ⟨7340⟩⟩
def rawTerms : List Term := Proof.Events083.exact21458RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21458
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21458.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21457.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge21457.frameStart)
    (transferEvent := 21456) (owner := owner)
    (leftResult := 21290) (rightResult := 6498)
    (working := LeftOperatorMerge21457.working)
    (reconstruction := LeftOperatorMerge21457.reconstruction)
    (leftReference := .predecessor 0 21454 .coefficient) (rightReference := .predecessor 1 21455 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6498.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge21457.operationAgreement
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
end SemanticResult21458

namespace SemanticResult21462
def owner : Owner := ⟨.program ⟨214⟩, ⟨10362⟩⟩
def rawTerms : List Term := Proof.Events083.exact21462RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21462
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21462.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 21459) (rightBinding := 21460)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7340⟩) (rightExpression := ⟨10361⟩)
    (transferEvent := 21461)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21458.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21453.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult21462

namespace SemanticResult21468
def owner : Owner := ⟨.program ⟨214⟩, ⟨10363⟩⟩
def rawTerms : List Term := Proof.Events083.exact21468RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 21468
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21468.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 21465) (survivorTransfer := 21466)
    (survivorEvent := 21467) (resultEvent := resultEvent)
    (rightCoefficientProducer := 6489)
    (owner := owner) (leftOwner := SemanticResult21462.owner)
    (rightOwner := SemanticResult6490.owner)
    (leftResult := 21462) (rightResult := 6490)
    (leftBinding := 21463) (rightBinding := 21464)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10362⟩) (rightExpression := ⟨84⟩)
    (leftActual := SemanticResult21462.actual selector witness)
    (rightActual := SemanticResult6490.actual selector witness)
    (leftRaw := SemanticResult21462.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨84⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound6489.actual selector witness)
    (survivorMagnitude := LeftBound21466.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21462.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6490.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6489.derived selector witness)
  · exact LeftBound21466.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult21468

namespace SemanticResult21478
def owner : Owner := ⟨.program ⟨214⟩, ⟨10364⟩⟩
def rawTerms : List Term := Proof.Events083.exact21478RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 21478
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21478.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21474.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge21474.frameStart)
    (owner := owner) (leftOwner := SemanticResult21468.owner)
    (rightOwner := SemanticResult6487.owner)
    (leftResult := 21468) (rightResult := 6487)
    (leftActual := SemanticResult21468.actual selector witness)
    (rightActual := SemanticResult6487.actual selector witness)
    (leftRaw := SemanticResult21468.rawTerms)
    (rightRaw := SemanticResult6487.rawTerms)
    (working := LeftOperatorMerge21474.working)
    (leftBinding := 21469) (rightBinding := 21470)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10363⟩) (rightExpression := ⟨7883⟩)
    (coefficientTransfer := 21471) (summaryTransfer := 21473)
    (rightCoefficientProducer := 6486)
    (rightSummaryTransfer := 21472)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge21474.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound6486.actual selector witness)
    (summaryMagnitude := LeftBound21473.actual selector witness)
    (reconstruction := LeftOperatorMerge21474.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21468.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6487.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6486.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound6486.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge21474.operationAgreement
  · exact LeftBound21473.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21474.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 21475 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10360⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge21474.working
    [{ coefficient := (-1), key := LeftRelationMerge21475.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge21475.frameStart
      LeftRelationMerge21475.owner (.relation 21475) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge21475.deltas
    rows := LeftRelationMerge21475.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge21474.working LeftRelationMerge21475.source
        (relationContext LeftRelationMerge21475.source
          LeftRelationMerge21475.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge21474.working, LeftRelationMerge21475.deltas,
    LeftRelationMerge21475.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 21475)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10364⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10360⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge21474.working) (working := relationWorking0)
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
end SemanticResult21478

namespace SemanticResult21484
def owner : Owner := ⟨.program ⟨214⟩, ⟨13381⟩⟩
def rawTerms : List Term := Proof.Events083.exact21484RawTerms
def summary : Bound := (.finite 95470336)
def resultEvent : Nat := 21484
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21484.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge21482.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult21478.owner)
    (rightOwner := SemanticResult21448.owner)
    (leftResult := 21478) (rightResult := 21448)
    (leftActual := SemanticResult21478.actual selector witness)
    (rightActual := SemanticResult21448.actual selector witness)
    (leftRaw := SemanticResult21478.rawTerms)
    (rightRaw := SemanticResult21448.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 49920) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 21479) (rightBinding := 21480)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10364⟩) (rightExpression := ⟨13380⟩)
    (coefficientTransfer := 21481) (summaryTransfer := 21483)
    (base := LeftOperatorMerge21482.base)
    (reconstruction := LeftOperatorMerge21482.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21478.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21448.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge21482.operationAgreement
  · rfl
  · decide
end SemanticResult21484

namespace SemanticResult21494
def owner : Owner := ⟨.program ⟨214⟩, ⟨25774⟩⟩
def rawTerms : List Term := Proof.Events083.exact21494RawTerms
def summary : Bound := (.finite 350377660645376)
def resultEvent : Nat := 21494
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21494.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95470336, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21490.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge21490.frameStart)
    (owner := owner) (leftOwner := SemanticResult21484.owner)
    (rightOwner := SemanticResult21415.owner)
    (leftResult := 21484) (rightResult := 21415)
    (leftActual := SemanticResult21484.actual selector witness)
    (rightActual := SemanticResult21415.actual selector witness)
    (leftRaw := SemanticResult21484.rawTerms)
    (rightRaw := SemanticResult21415.rawTerms)
    (working := LeftOperatorMerge21490.working)
    (leftBinding := 21485) (rightBinding := 21486)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13381⟩) (rightExpression := ⟨25773⟩)
    (coefficientTransfer := 21487) (summaryTransfer := 21489)
    (rightCoefficientProducer := 21414)
    (rightSummaryTransfer := 21488)
    (leftMaximum := ⟨95470336, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge21490.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority21414.actual selector witness)
    (summaryMagnitude := LeftBound21489.actual selector witness)
    (reconstruction := LeftOperatorMerge21490.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21484.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21415.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21414.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority21414.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge21490.operationAgreement
  · exact LeftBound21489.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge21490.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 21491 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23422⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23422⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge21490.working
    [{ coefficient := (-1), key := LeftRelationMerge21491.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge21491.frameStart
      LeftRelationMerge21491.owner (.relation 21491) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge21491.deltas
    rows := LeftRelationMerge21491.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge21490.working LeftRelationMerge21491.source
        (relationContext LeftRelationMerge21491.source
          LeftRelationMerge21491.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge21490.working, LeftRelationMerge21491.deltas,
    LeftRelationMerge21491.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 21491)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25774⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10360⟩⟩, ⟨.program ⟨214⟩, ⟨13374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge21490.working) (working := relationWorking0)
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
end SemanticResult21494

namespace SemanticResult21497
def owner : Owner := ⟨.program ⟨214⟩, ⟨20260⟩⟩
def rawTerms : List Term := Proof.Events083.exact21497RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21497
def producerEvent : Nat := 21496
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21497.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨26⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨26⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult21497

namespace SemanticResult21501
def owner : Owner := ⟨.program ⟨214⟩, ⟨20262⟩⟩
def rawTerms : List Term := Proof.Events083.exact21501RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 21501
def producerEvent : Nat := 21500
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult21501.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 21498 .coefficient) (.value (.predecessor 1 21499 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 21498 .coefficient) (.value (.predecessor 1 21499 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult21501

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
