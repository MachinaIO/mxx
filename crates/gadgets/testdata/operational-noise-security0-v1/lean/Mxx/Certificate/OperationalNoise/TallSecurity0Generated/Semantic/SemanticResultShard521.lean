import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard521
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard027
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard117
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard118
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard465

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult72523
def owner : Owner := ⟨.program ⟨214⟩, ⟨23844⟩⟩
def rawTerms : List Term := Proof.Events283.exact72523RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 72523
def producerEvent : Nat := 72522
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72523.actual selector witness
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
end SemanticResult72523

namespace SemanticResult72526
def owner : Owner := ⟨.program ⟨214⟩, ⟨26768⟩⟩
def rawTerms : List Term := Proof.Events283.exact72526RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 72526
def producerEvent : Nat := 72525
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72526.actual selector witness
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
end SemanticResult72526

namespace SemanticResult72533
def owner : Owner := ⟨.program ⟨214⟩, ⟨23036⟩⟩
def rawTerms : List Term := Proof.Events283.exact72533RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 72533
def producerEvent : Nat := 72532
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72533.actual selector witness
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
end SemanticResult72533

namespace SemanticResult72536
def owner : Owner := ⟨.program ⟨214⟩, ⟨25060⟩⟩
def rawTerms : List Term := Proof.Events283.exact72536RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 72536
def producerEvent : Nat := 72535
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72536.actual selector witness
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
end SemanticResult72536

namespace SemanticResult72541
def owner : Owner := ⟨.program ⟨214⟩, ⟨10972⟩⟩
def rawTerms : List Term := Proof.Events283.exact72541RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 72541
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72541.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge72540.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge72540.frameStart)
    (transferEvent := 72539) (owner := owner)
    (leftResult := 3431) (rightResult := 65295)
    (working := LeftOperatorMerge72540.working)
    (reconstruction := LeftOperatorMerge72540.reconstruction)
    (leftReference := .predecessor 0 72537 .coefficient) (rightReference := .predecessor 1 72538 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3431.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge72540.operationAgreement
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
end SemanticResult72541

namespace SemanticResult72546
def owner : Owner := ⟨.program ⟨214⟩, ⟨7192⟩⟩
def rawTerms : List Term := Proof.Events283.exact72546RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 72546
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72546.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge72545.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge72545.frameStart)
    (transferEvent := 72544) (owner := owner)
    (leftResult := 65165) (rightResult := 13987)
    (working := LeftOperatorMerge72545.working)
    (reconstruction := LeftOperatorMerge72545.reconstruction)
    (leftReference := .predecessor 0 72542 .coefficient) (rightReference := .predecessor 1 72543 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13987.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge72545.operationAgreement
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
end SemanticResult72546

namespace SemanticResult72550
def owner : Owner := ⟨.program ⟨214⟩, ⟨10973⟩⟩
def rawTerms : List Term := Proof.Events283.exact72550RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 72550
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72550.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 72547) (rightBinding := 72548)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7192⟩) (rightExpression := ⟨10972⟩)
    (transferEvent := 72549)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult72546.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult72541.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult72550

namespace SemanticResult72556
def owner : Owner := ⟨.program ⟨214⟩, ⟨10974⟩⟩
def rawTerms : List Term := Proof.Events283.exact72556RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 72556
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72556.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 72553) (survivorTransfer := 72554)
    (survivorEvent := 72555) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13978)
    (owner := owner) (leftOwner := SemanticResult72550.owner)
    (rightOwner := SemanticResult13979.owner)
    (leftResult := 72550) (rightResult := 13979)
    (leftBinding := 72551) (rightBinding := 72552)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10973⟩) (rightExpression := ⟨88⟩)
    (leftActual := SemanticResult72550.actual selector witness)
    (rightActual := SemanticResult13979.actual selector witness)
    (leftRaw := SemanticResult72550.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨88⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13978.actual selector witness)
    (survivorMagnitude := LeftBound72554.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult72550.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13979.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13978.derived selector witness)
  · exact LeftBound72554.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult72556

namespace SemanticResult72564
def owner : Owner := ⟨.program ⟨214⟩, ⟨10975⟩⟩
def rawTerms : List Term := Proof.Events283.exact72564RawTerms
def summary : Bound := (.finite 3328)
def resultEvent : Nat := 72564
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72564.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨4, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge72562.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge72562.frameStart)
    (owner := owner) (leftOwner := SemanticResult72556.owner)
    (rightOwner := SemanticResult3434.owner)
    (leftResult := 72556) (rightResult := 3434)
    (leftActual := SemanticResult72556.actual selector witness)
    (rightActual := SemanticResult3434.actual selector witness)
    (leftRaw := SemanticResult72556.rawTerms)
    (rightRaw := SemanticResult3434.rawTerms)
    (working := LeftOperatorMerge72562.working)
    (leftBinding := 72557) (rightBinding := 72558)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10974⟩) (rightExpression := ⟨10837⟩)
    (coefficientTransfer := 72559) (summaryTransfer := 72561)
    (rightCoefficientProducer := 3433)
    (rightSummaryTransfer := 72560)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨4, by decide⟩)
    (rightRecordedMaximum := 4)
    (rightSummaryMaximum := ⟨4, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge72562.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority3433.actual selector witness)
    (summaryMagnitude := LeftBound72561.actual selector witness)
    (reconstruction := LeftOperatorMerge72562.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult72556.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3434.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3433.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority3433.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge72562.operationAgreement
  · exact LeftBound72561.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge72562.working summary) := by
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
end SemanticResult72564

namespace SemanticResult72569
def owner : Owner := ⟨.program ⟨214⟩, ⟨10838⟩⟩
def rawTerms : List Term := Proof.Events283.exact72569RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 72569
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72569.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge72568.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge72568.frameStart)
    (transferEvent := 72567) (owner := owner)
    (leftResult := 3434) (rightResult := 65295)
    (working := LeftOperatorMerge72568.working)
    (reconstruction := LeftOperatorMerge72568.reconstruction)
    (leftReference := .predecessor 0 72565 .coefficient) (rightReference := .predecessor 1 72566 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3434.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge72568.operationAgreement
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
end SemanticResult72569

namespace SemanticResult72574
def owner : Owner := ⟨.program ⟨214⟩, ⟨7209⟩⟩
def rawTerms : List Term := Proof.Events283.exact72574RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 72574
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72574.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge72573.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge72573.frameStart)
    (transferEvent := 72572) (owner := owner)
    (leftResult := 65165) (rightResult := 14028)
    (working := LeftOperatorMerge72573.working)
    (reconstruction := LeftOperatorMerge72573.reconstruction)
    (leftReference := .predecessor 0 72570 .coefficient) (rightReference := .predecessor 1 72571 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14028.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge72573.operationAgreement
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
end SemanticResult72574

namespace SemanticResult72578
def owner : Owner := ⟨.program ⟨214⟩, ⟨10839⟩⟩
def rawTerms : List Term := Proof.Events283.exact72578RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 72578
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72578.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 72575) (rightBinding := 72576)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7209⟩) (rightExpression := ⟨10838⟩)
    (transferEvent := 72577)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult72574.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult72569.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult72578

namespace SemanticResult72584
def owner : Owner := ⟨.program ⟨214⟩, ⟨10840⟩⟩
def rawTerms : List Term := Proof.Events283.exact72584RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 72584
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72584.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 72581) (survivorTransfer := 72582)
    (survivorEvent := 72583) (resultEvent := resultEvent)
    (rightCoefficientProducer := 14019)
    (owner := owner) (leftOwner := SemanticResult72578.owner)
    (rightOwner := SemanticResult14020.owner)
    (leftResult := 72578) (rightResult := 14020)
    (leftBinding := 72579) (rightBinding := 72580)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10839⟩) (rightExpression := ⟨105⟩)
    (leftActual := SemanticResult72578.actual selector witness)
    (rightActual := SemanticResult14020.actual selector witness)
    (leftRaw := SemanticResult72578.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨105⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound14019.actual selector witness)
    (survivorMagnitude := LeftBound72582.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult72578.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14020.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14019.derived selector witness)
  · exact LeftBound72582.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult72584

namespace SemanticResult72594
def owner : Owner := ⟨.program ⟨214⟩, ⟨10841⟩⟩
def rawTerms : List Term := Proof.Events283.exact72594RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 72594
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72594.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge72590.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge72590.frameStart)
    (owner := owner) (leftOwner := SemanticResult72584.owner)
    (rightOwner := SemanticResult14017.owner)
    (leftResult := 72584) (rightResult := 14017)
    (leftActual := SemanticResult72584.actual selector witness)
    (rightActual := SemanticResult14017.actual selector witness)
    (leftRaw := SemanticResult72584.rawTerms)
    (rightRaw := SemanticResult14017.rawTerms)
    (working := LeftOperatorMerge72590.working)
    (leftBinding := 72585) (rightBinding := 72586)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10840⟩) (rightExpression := ⟨7838⟩)
    (coefficientTransfer := 72587) (summaryTransfer := 72589)
    (rightCoefficientProducer := 14016)
    (rightSummaryTransfer := 72588)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge72590.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound14016.actual selector witness)
    (summaryMagnitude := LeftBound72589.actual selector witness)
    (reconstruction := LeftOperatorMerge72590.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult72584.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14017.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14016.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound14016.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge72590.operationAgreement
  · exact LeftBound72589.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge72590.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 72591 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10837⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge72590.working
    [{ coefficient := (-1), key := LeftRelationMerge72591.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge72591.frameStart
      LeftRelationMerge72591.owner (.relation 72591) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge72591.deltas
    rows := LeftRelationMerge72591.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge72590.working LeftRelationMerge72591.source
        (relationContext LeftRelationMerge72591.source
          LeftRelationMerge72591.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge72590.working, LeftRelationMerge72591.deltas,
    LeftRelationMerge72591.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 72591)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10841⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10837⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge72590.working) (working := relationWorking0)
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
end SemanticResult72594

namespace SemanticResult72600
def owner : Owner := ⟨.program ⟨214⟩, ⟨10976⟩⟩
def rawTerms : List Term := Proof.Events283.exact72600RawTerms
def summary : Bound := (.finite 95423744)
def resultEvent : Nat := 72600
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72600.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge72598.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult72594.owner)
    (rightOwner := SemanticResult72564.owner)
    (leftResult := 72594) (rightResult := 72564)
    (leftActual := SemanticResult72594.actual selector witness)
    (rightActual := SemanticResult72564.actual selector witness)
    (leftRaw := SemanticResult72594.rawTerms)
    (rightRaw := SemanticResult72564.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 3328) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 72595) (rightBinding := 72596)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10841⟩) (rightExpression := ⟨10975⟩)
    (coefficientTransfer := 72597) (summaryTransfer := 72599)
    (base := LeftOperatorMerge72598.base)
    (reconstruction := LeftOperatorMerge72598.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult72594.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult72564.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge72598.operationAgreement
  · rfl
  · decide
end SemanticResult72600

namespace SemanticResult72610
def owner : Owner := ⟨.program ⟨214⟩, ⟨25061⟩⟩
def rawTerms : List Term := Proof.Events283.exact72610RawTerms
def summary : Bound := (.finite 350206667259904)
def resultEvent : Nat := 72610
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult72610.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95423744, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge72606.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge72606.frameStart)
    (owner := owner) (leftOwner := SemanticResult72600.owner)
    (rightOwner := SemanticResult72536.owner)
    (leftResult := 72600) (rightResult := 72536)
    (leftActual := SemanticResult72600.actual selector witness)
    (rightActual := SemanticResult72536.actual selector witness)
    (leftRaw := SemanticResult72600.rawTerms)
    (rightRaw := SemanticResult72536.rawTerms)
    (working := LeftOperatorMerge72606.working)
    (leftBinding := 72601) (rightBinding := 72602)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10976⟩) (rightExpression := ⟨25060⟩)
    (coefficientTransfer := 72603) (summaryTransfer := 72605)
    (rightCoefficientProducer := 72535)
    (rightSummaryTransfer := 72604)
    (leftMaximum := ⟨95423744, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge72606.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority72535.actual selector witness)
    (summaryMagnitude := LeftBound72605.actual selector witness)
    (reconstruction := LeftOperatorMerge72606.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult72600.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult72536.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72535.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority72535.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge72606.operationAgreement
  · exact LeftBound72605.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge72606.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 72607 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25060⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23036⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23036⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge72606.working
    [{ coefficient := (-1), key := LeftRelationMerge72607.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge72607.frameStart
      LeftRelationMerge72607.owner (.relation 72607) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge72607.deltas
    rows := LeftRelationMerge72607.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge72606.working LeftRelationMerge72607.source
        (relationContext LeftRelationMerge72607.source
          LeftRelationMerge72607.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge72606.working, LeftRelationMerge72607.deltas,
    LeftRelationMerge72607.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 72607)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25061⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10837⟩⟩, ⟨.program ⟨214⟩, ⟨10969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25060⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25060⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge72606.working) (working := relationWorking0)
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
end SemanticResult72610

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
