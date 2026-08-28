import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard570
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard031
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard061
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard566

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult80401
def owner : Owner := ⟨.program ⟨214⟩, ⟨29819⟩⟩
def rawTerms : List Term := Proof.Events314.exact80401RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 80401
def producerEvent : Nat := 80400
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80401.actual selector witness
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
end SemanticResult80401

namespace SemanticResult80408
def owner : Owner := ⟨.program ⟨214⟩, ⟨23374⟩⟩
def rawTerms : List Term := Proof.Events314.exact80408RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 80408
def producerEvent : Nat := 80407
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80408.actual selector witness
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
end SemanticResult80408

namespace SemanticResult80411
def owner : Owner := ⟨.program ⟨214⟩, ⟨25681⟩⟩
def rawTerms : List Term := Proof.Events314.exact80411RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 80411
def producerEvent : Nat := 80410
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80411.actual selector witness
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
end SemanticResult80411

namespace SemanticResult80416
def owner : Owner := ⟨.program ⟨214⟩, ⟨13157⟩⟩
def rawTerms : List Term := Proof.Events314.exact80416RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 80416
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80416.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge80415.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge80415.frameStart)
    (transferEvent := 80414) (owner := owner)
    (leftResult := 3851) (rightResult := 79920)
    (working := LeftOperatorMerge80415.working)
    (reconstruction := LeftOperatorMerge80415.reconstruction)
    (leftReference := .predecessor 0 80412 .coefficient) (rightReference := .predecessor 1 80413 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3851.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79920.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge80415.operationAgreement
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
end SemanticResult80416

namespace SemanticResult80421
def owner : Owner := ⟨.program ⟨214⟩, ⟨7245⟩⟩
def rawTerms : List Term := Proof.Events314.exact80421RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 80421
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80421.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge80420.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge80420.frameStart)
    (transferEvent := 80419) (owner := owner)
    (leftResult := 79790) (rightResult := 6973)
    (working := LeftOperatorMerge80420.working)
    (reconstruction := LeftOperatorMerge80420.reconstruction)
    (leftReference := .predecessor 0 80417 .coefficient) (rightReference := .predecessor 1 80418 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6973.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge80420.operationAgreement
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
end SemanticResult80421

namespace SemanticResult80425
def owner : Owner := ⟨.program ⟨214⟩, ⟨13158⟩⟩
def rawTerms : List Term := Proof.Events314.exact80425RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 80425
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80425.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 80422) (rightBinding := 80423)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7245⟩) (rightExpression := ⟨13157⟩)
    (transferEvent := 80424)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult80421.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult80416.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult80425

namespace SemanticResult80431
def owner : Owner := ⟨.program ⟨214⟩, ⟨13159⟩⟩
def rawTerms : List Term := Proof.Events314.exact80431RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 80431
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80431.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 80428) (survivorTransfer := 80429)
    (survivorEvent := 80430) (resultEvent := resultEvent)
    (rightCoefficientProducer := 6964)
    (owner := owner) (leftOwner := SemanticResult80425.owner)
    (rightOwner := SemanticResult6965.owner)
    (leftResult := 80425) (rightResult := 6965)
    (leftBinding := 80426) (rightBinding := 80427)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13158⟩) (rightExpression := ⟨103⟩)
    (leftActual := SemanticResult80425.actual selector witness)
    (rightActual := SemanticResult6965.actual selector witness)
    (leftRaw := SemanticResult80425.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨103⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound6964.actual selector witness)
    (survivorMagnitude := LeftBound80429.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult80425.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6965.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6964.derived selector witness)
  · exact LeftBound80429.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult80431

namespace SemanticResult80439
def owner : Owner := ⟨.program ⟨214⟩, ⟨13160⟩⟩
def rawTerms : List Term := Proof.Events314.exact80439RawTerms
def summary : Bound := (.finite 48256)
def resultEvent : Nat := 80439
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80439.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨58, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge80437.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge80437.frameStart)
    (owner := owner) (leftOwner := SemanticResult80431.owner)
    (rightOwner := SemanticResult3854.owner)
    (leftResult := 80431) (rightResult := 3854)
    (leftActual := SemanticResult80431.actual selector witness)
    (rightActual := SemanticResult3854.actual selector witness)
    (leftRaw := SemanticResult80431.rawTerms)
    (rightRaw := SemanticResult3854.rawTerms)
    (working := LeftOperatorMerge80437.working)
    (leftBinding := 80432) (rightBinding := 80433)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13159⟩) (rightExpression := ⟨10240⟩)
    (coefficientTransfer := 80434) (summaryTransfer := 80436)
    (rightCoefficientProducer := 3853)
    (rightSummaryTransfer := 80435)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨58, by decide⟩)
    (rightRecordedMaximum := 58)
    (rightSummaryMaximum := ⟨58, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge80437.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority3853.actual selector witness)
    (summaryMagnitude := LeftBound80436.actual selector witness)
    (reconstruction := LeftOperatorMerge80437.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult80431.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3854.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3853.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority3853.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge80437.operationAgreement
  · exact LeftBound80436.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge80437.working summary) := by
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
end SemanticResult80439

namespace SemanticResult80444
def owner : Owner := ⟨.program ⟨214⟩, ⟨10241⟩⟩
def rawTerms : List Term := Proof.Events314.exact80444RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 80444
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80444.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge80443.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge80443.frameStart)
    (transferEvent := 80442) (owner := owner)
    (leftResult := 3854) (rightResult := 79920)
    (working := LeftOperatorMerge80443.working)
    (reconstruction := LeftOperatorMerge80443.reconstruction)
    (leftReference := .predecessor 0 80440 .coefficient) (rightReference := .predecessor 1 80441 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3854.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79920.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge80443.operationAgreement
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
end SemanticResult80444

namespace SemanticResult80449
def owner : Owner := ⟨.program ⟨214⟩, ⟨7225⟩⟩
def rawTerms : List Term := Proof.Events314.exact80449RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 80449
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80449.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge80448.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge80448.frameStart)
    (transferEvent := 80447) (owner := owner)
    (leftResult := 79790) (rightResult := 7014)
    (working := LeftOperatorMerge80448.working)
    (reconstruction := LeftOperatorMerge80448.reconstruction)
    (leftReference := .predecessor 0 80445 .coefficient) (rightReference := .predecessor 1 80446 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7014.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge80448.operationAgreement
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
end SemanticResult80449

namespace SemanticResult80453
def owner : Owner := ⟨.program ⟨214⟩, ⟨10242⟩⟩
def rawTerms : List Term := Proof.Events314.exact80453RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 80453
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80453.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 80450) (rightBinding := 80451)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7225⟩) (rightExpression := ⟨10241⟩)
    (transferEvent := 80452)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult80449.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult80444.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult80453

namespace SemanticResult80459
def owner : Owner := ⟨.program ⟨214⟩, ⟨10243⟩⟩
def rawTerms : List Term := Proof.Events314.exact80459RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 80459
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80459.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 80456) (survivorTransfer := 80457)
    (survivorEvent := 80458) (resultEvent := resultEvent)
    (rightCoefficientProducer := 7005)
    (owner := owner) (leftOwner := SemanticResult80453.owner)
    (rightOwner := SemanticResult7006.owner)
    (leftResult := 80453) (rightResult := 7006)
    (leftBinding := 80454) (rightBinding := 80455)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10242⟩) (rightExpression := ⟨83⟩)
    (leftActual := SemanticResult80453.actual selector witness)
    (rightActual := SemanticResult7006.actual selector witness)
    (leftRaw := SemanticResult80453.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨83⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound7005.actual selector witness)
    (survivorMagnitude := LeftBound80457.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult80453.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7006.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7005.derived selector witness)
  · exact LeftBound80457.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult80459

namespace SemanticResult80469
def owner : Owner := ⟨.program ⟨214⟩, ⟨10244⟩⟩
def rawTerms : List Term := Proof.Events314.exact80469RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 80469
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80469.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge80465.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge80465.frameStart)
    (owner := owner) (leftOwner := SemanticResult80459.owner)
    (rightOwner := SemanticResult7003.owner)
    (leftResult := 80459) (rightResult := 7003)
    (leftActual := SemanticResult80459.actual selector witness)
    (rightActual := SemanticResult7003.actual selector witness)
    (leftRaw := SemanticResult80459.rawTerms)
    (rightRaw := SemanticResult7003.rawTerms)
    (working := LeftOperatorMerge80465.working)
    (leftBinding := 80460) (rightBinding := 80461)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10243⟩) (rightExpression := ⟨7880⟩)
    (coefficientTransfer := 80462) (summaryTransfer := 80464)
    (rightCoefficientProducer := 7002)
    (rightSummaryTransfer := 80463)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge80465.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound7002.actual selector witness)
    (summaryMagnitude := LeftBound80464.actual selector witness)
    (reconstruction := LeftOperatorMerge80465.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult80459.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7003.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7002.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound7002.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge80465.operationAgreement
  · exact LeftBound80464.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge80465.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 80466 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge80465.working
    [{ coefficient := (-1), key := LeftRelationMerge80466.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge80466.frameStart
      LeftRelationMerge80466.owner (.relation 80466) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge80466.deltas
    rows := LeftRelationMerge80466.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge80465.working LeftRelationMerge80466.source
        (relationContext LeftRelationMerge80466.source
          LeftRelationMerge80466.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge80465.working, LeftRelationMerge80466.deltas,
    LeftRelationMerge80466.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 80466)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10244⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge80465.working) (working := relationWorking0)
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
end SemanticResult80469

namespace SemanticResult80475
def owner : Owner := ⟨.program ⟨214⟩, ⟨13161⟩⟩
def rawTerms : List Term := Proof.Events314.exact80475RawTerms
def summary : Bound := (.finite 95468672)
def resultEvent : Nat := 80475
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80475.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge80473.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult80469.owner)
    (rightOwner := SemanticResult80439.owner)
    (leftResult := 80469) (rightResult := 80439)
    (leftActual := SemanticResult80469.actual selector witness)
    (rightActual := SemanticResult80439.actual selector witness)
    (leftRaw := SemanticResult80469.rawTerms)
    (rightRaw := SemanticResult80439.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 48256) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 80470) (rightBinding := 80471)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10244⟩) (rightExpression := ⟨13160⟩)
    (coefficientTransfer := 80472) (summaryTransfer := 80474)
    (base := LeftOperatorMerge80473.base)
    (reconstruction := LeftOperatorMerge80473.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult80469.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult80439.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge80473.operationAgreement
  · rfl
  · decide
end SemanticResult80475

namespace SemanticResult80485
def owner : Owner := ⟨.program ⟨214⟩, ⟨25682⟩⟩
def rawTerms : List Term := Proof.Events314.exact80485RawTerms
def summary : Bound := (.finite 350371553738752)
def resultEvent : Nat := 80485
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80485.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95468672, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge80481.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge80481.frameStart)
    (owner := owner) (leftOwner := SemanticResult80475.owner)
    (rightOwner := SemanticResult80411.owner)
    (leftResult := 80475) (rightResult := 80411)
    (leftActual := SemanticResult80475.actual selector witness)
    (rightActual := SemanticResult80411.actual selector witness)
    (leftRaw := SemanticResult80475.rawTerms)
    (rightRaw := SemanticResult80411.rawTerms)
    (working := LeftOperatorMerge80481.working)
    (leftBinding := 80476) (rightBinding := 80477)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13161⟩) (rightExpression := ⟨25681⟩)
    (coefficientTransfer := 80478) (summaryTransfer := 80480)
    (rightCoefficientProducer := 80410)
    (rightSummaryTransfer := 80479)
    (leftMaximum := ⟨95468672, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge80481.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority80410.actual selector witness)
    (summaryMagnitude := LeftBound80480.actual selector witness)
    (reconstruction := LeftOperatorMerge80481.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult80475.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult80411.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80410.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority80410.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge80481.operationAgreement
  · exact LeftBound80480.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge80481.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 80482 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23374⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23374⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge80481.working
    [{ coefficient := (-1), key := LeftRelationMerge80482.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge80482.frameStart
      LeftRelationMerge80482.owner (.relation 80482) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge80482.deltas
    rows := LeftRelationMerge80482.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge80481.working LeftRelationMerge80482.source
        (relationContext LeftRelationMerge80482.source
          LeftRelationMerge80482.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge80481.working, LeftRelationMerge80482.deltas,
    LeftRelationMerge80482.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 80482)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25682⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10240⟩⟩, ⟨.program ⟨214⟩, ⟨13154⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25681⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge80481.working) (working := relationWorking0)
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
end SemanticResult80485

namespace SemanticResult80488
def owner : Owner := ⟨.program ⟨214⟩, ⟨20176⟩⟩
def rawTerms : List Term := Proof.Events314.exact80488RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 80488
def producerEvent : Nat := 80487
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult80488.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨25⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨25⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult80488

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
