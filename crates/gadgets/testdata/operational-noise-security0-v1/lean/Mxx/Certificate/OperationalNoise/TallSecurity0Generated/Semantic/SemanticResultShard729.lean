import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard729
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard039
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard125
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard126

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult101749
def owner : Owner := ⟨.program ⟨214⟩, ⟨26326⟩⟩
def rawTerms : List Term := Proof.Events397.exact101749RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 101749
def producerEvent : Nat := 101748
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101749.actual selector witness
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
end SemanticResult101749

namespace SemanticResult101756
def owner : Owner := ⟨.program ⟨214⟩, ⟨22948⟩⟩
def rawTerms : List Term := Proof.Events397.exact101756RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 101756
def producerEvent : Nat := 101755
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101756.actual selector witness
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
end SemanticResult101756

namespace SemanticResult101759
def owner : Owner := ⟨.program ⟨214⟩, ⟨24898⟩⟩
def rawTerms : List Term := Proof.Events397.exact101759RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 101759
def producerEvent : Nat := 101758
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101759.actual selector witness
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
end SemanticResult101759

namespace SemanticResult101764
def owner : Owner := ⟨.program ⟨214⟩, ⟨10459⟩⟩
def rawTerms : List Term := Proof.Events397.exact101764RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 101764
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101764.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge101763.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge101763.frameStart)
    (transferEvent := 101762) (owner := owner)
    (leftResult := 4957) (rightResult := 32)
    (working := LeftOperatorMerge101763.working)
    (reconstruction := LeftOperatorMerge101763.reconstruction)
    (leftReference := .predecessor 0 101760 .coefficient) (rightReference := .predecessor 1 101761 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4957.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge101763.operationAgreement
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
end SemanticResult101764

namespace SemanticResult101769
def owner : Owner := ⟨.program ⟨214⟩, ⟨7109⟩⟩
def rawTerms : List Term := Proof.Events397.exact101769RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 101769
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101769.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge101768.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge101768.frameStart)
    (transferEvent := 101767) (owner := owner)
    (leftResult := 27) (rightResult := 14989)
    (working := LeftOperatorMerge101768.working)
    (reconstruction := LeftOperatorMerge101768.reconstruction)
    (leftReference := .predecessor 0 101765 .coefficient) (rightReference := .predecessor 1 101766 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14989.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge101768.operationAgreement
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
end SemanticResult101769

namespace SemanticResult101773
def owner : Owner := ⟨.program ⟨214⟩, ⟨10460⟩⟩
def rawTerms : List Term := Proof.Events397.exact101773RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 101773
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101773.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 101770) (rightBinding := 101771)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7109⟩) (rightExpression := ⟨10459⟩)
    (transferEvent := 101772)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult101769.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult101764.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult101773

namespace SemanticResult101779
def owner : Owner := ⟨.program ⟨214⟩, ⟨10461⟩⟩
def rawTerms : List Term := Proof.Events397.exact101779RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 101779
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101779.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 101776) (survivorTransfer := 101777)
    (survivorEvent := 101778) (resultEvent := resultEvent)
    (rightCoefficientProducer := 14980)
    (owner := owner) (leftOwner := SemanticResult101773.owner)
    (rightOwner := SemanticResult14981.owner)
    (leftResult := 101773) (rightResult := 14981)
    (leftBinding := 101774) (rightBinding := 101775)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10460⟩) (rightExpression := ⟨86⟩)
    (leftActual := SemanticResult101773.actual selector witness)
    (rightActual := SemanticResult14981.actual selector witness)
    (leftRaw := SemanticResult101773.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨86⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound14980.actual selector witness)
    (survivorMagnitude := LeftBound101777.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult101773.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14981.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14980.derived selector witness)
  · exact LeftBound101777.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult101779

namespace SemanticResult101787
def owner : Owner := ⟨.program ⟨214⟩, ⟨10462⟩⟩
def rawTerms : List Term := Proof.Events397.exact101787RawTerms
def summary : Bound := (.finite 1664)
def resultEvent : Nat := 101787
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101787.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨2, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge101785.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge101785.frameStart)
    (owner := owner) (leftOwner := SemanticResult101779.owner)
    (rightOwner := SemanticResult4960.owner)
    (leftResult := 101779) (rightResult := 4960)
    (leftActual := SemanticResult101779.actual selector witness)
    (rightActual := SemanticResult4960.actual selector witness)
    (leftRaw := SemanticResult101779.rawTerms)
    (rightRaw := SemanticResult4960.rawTerms)
    (working := LeftOperatorMerge101785.working)
    (leftBinding := 101780) (rightBinding := 101781)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10461⟩) (rightExpression := ⟨9385⟩)
    (coefficientTransfer := 101782) (summaryTransfer := 101784)
    (rightCoefficientProducer := 4959)
    (rightSummaryTransfer := 101783)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨2, by decide⟩)
    (rightRecordedMaximum := 2)
    (rightSummaryMaximum := ⟨2, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge101785.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority4959.actual selector witness)
    (summaryMagnitude := LeftBound101784.actual selector witness)
    (reconstruction := LeftOperatorMerge101785.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult101779.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4960.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4959.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority4959.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge101785.operationAgreement
  · exact LeftBound101784.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge101785.working summary) := by
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
end SemanticResult101787

namespace SemanticResult101792
def owner : Owner := ⟨.program ⟨214⟩, ⟨9386⟩⟩
def rawTerms : List Term := Proof.Events397.exact101792RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 101792
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101792.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge101791.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge101791.frameStart)
    (transferEvent := 101790) (owner := owner)
    (leftResult := 4960) (rightResult := 32)
    (working := LeftOperatorMerge101791.working)
    (reconstruction := LeftOperatorMerge101791.reconstruction)
    (leftReference := .predecessor 0 101788 .coefficient) (rightReference := .predecessor 1 101789 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4960.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge101791.operationAgreement
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
end SemanticResult101792

namespace SemanticResult101797
def owner : Owner := ⟨.program ⟨214⟩, ⟨7108⟩⟩
def rawTerms : List Term := Proof.Events397.exact101797RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 101797
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101797.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge101796.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge101796.frameStart)
    (transferEvent := 101795) (owner := owner)
    (leftResult := 27) (rightResult := 15030)
    (working := LeftOperatorMerge101796.working)
    (reconstruction := LeftOperatorMerge101796.reconstruction)
    (leftReference := .predecessor 0 101793 .coefficient) (rightReference := .predecessor 1 101794 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15030.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge101796.operationAgreement
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
end SemanticResult101797

namespace SemanticResult101801
def owner : Owner := ⟨.program ⟨214⟩, ⟨9387⟩⟩
def rawTerms : List Term := Proof.Events397.exact101801RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 101801
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101801.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 101798) (rightBinding := 101799)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7108⟩) (rightExpression := ⟨9386⟩)
    (transferEvent := 101800)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult101797.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult101792.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult101801

namespace SemanticResult101807
def owner : Owner := ⟨.program ⟨214⟩, ⟨9388⟩⟩
def rawTerms : List Term := Proof.Events397.exact101807RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 101807
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101807.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 101804) (survivorTransfer := 101805)
    (survivorEvent := 101806) (resultEvent := resultEvent)
    (rightCoefficientProducer := 15021)
    (owner := owner) (leftOwner := SemanticResult101801.owner)
    (rightOwner := SemanticResult15022.owner)
    (leftResult := 101801) (rightResult := 15022)
    (leftBinding := 101802) (rightBinding := 101803)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9387⟩) (rightExpression := ⟨85⟩)
    (leftActual := SemanticResult101801.actual selector witness)
    (rightActual := SemanticResult15022.actual selector witness)
    (leftRaw := SemanticResult101801.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨85⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound15021.actual selector witness)
    (survivorMagnitude := LeftBound101805.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult101801.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15022.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15021.derived selector witness)
  · exact LeftBound101805.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult101807

namespace SemanticResult101817
def owner : Owner := ⟨.program ⟨214⟩, ⟨9389⟩⟩
def rawTerms : List Term := Proof.Events397.exact101817RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 101817
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101817.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge101813.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge101813.frameStart)
    (owner := owner) (leftOwner := SemanticResult101807.owner)
    (rightOwner := SemanticResult15019.owner)
    (leftResult := 101807) (rightResult := 15019)
    (leftActual := SemanticResult101807.actual selector witness)
    (rightActual := SemanticResult15019.actual selector witness)
    (leftRaw := SemanticResult101807.rawTerms)
    (rightRaw := SemanticResult15019.rawTerms)
    (working := LeftOperatorMerge101813.working)
    (leftBinding := 101808) (rightBinding := 101809)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9388⟩) (rightExpression := ⟨7832⟩)
    (coefficientTransfer := 101810) (summaryTransfer := 101812)
    (rightCoefficientProducer := 15018)
    (rightSummaryTransfer := 101811)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge101813.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound15018.actual selector witness)
    (summaryMagnitude := LeftBound101812.actual selector witness)
    (reconstruction := LeftOperatorMerge101813.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult101807.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15019.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15018.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound15018.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge101813.operationAgreement
  · exact LeftBound101812.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge101813.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 101814 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge101813.working
    [{ coefficient := (-1), key := LeftRelationMerge101814.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge101814.frameStart
      LeftRelationMerge101814.owner (.relation 101814) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge101814.deltas
    rows := LeftRelationMerge101814.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge101813.working LeftRelationMerge101814.source
        (relationContext LeftRelationMerge101814.source
          LeftRelationMerge101814.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge101813.working, LeftRelationMerge101814.deltas,
    LeftRelationMerge101814.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 101814)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9389⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge101813.working) (working := relationWorking0)
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
end SemanticResult101817

namespace SemanticResult101823
def owner : Owner := ⟨.program ⟨214⟩, ⟨10463⟩⟩
def rawTerms : List Term := Proof.Events397.exact101823RawTerms
def summary : Bound := (.finite 95422080)
def resultEvent : Nat := 101823
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101823.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge101821.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult101817.owner)
    (rightOwner := SemanticResult101787.owner)
    (leftResult := 101817) (rightResult := 101787)
    (leftActual := SemanticResult101817.actual selector witness)
    (rightActual := SemanticResult101787.actual selector witness)
    (leftRaw := SemanticResult101817.rawTerms)
    (rightRaw := SemanticResult101787.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 1664) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 101818) (rightBinding := 101819)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9389⟩) (rightExpression := ⟨10462⟩)
    (coefficientTransfer := 101820) (summaryTransfer := 101822)
    (base := LeftOperatorMerge101821.base)
    (reconstruction := LeftOperatorMerge101821.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult101817.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult101787.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge101821.operationAgreement
  · rfl
  · decide
end SemanticResult101823

namespace SemanticResult101833
def owner : Owner := ⟨.program ⟨214⟩, ⟨24899⟩⟩
def rawTerms : List Term := Proof.Events397.exact101833RawTerms
def summary : Bound := (.finite 350200560353280)
def resultEvent : Nat := 101833
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101833.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95422080, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge101829.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge101829.frameStart)
    (owner := owner) (leftOwner := SemanticResult101823.owner)
    (rightOwner := SemanticResult101759.owner)
    (leftResult := 101823) (rightResult := 101759)
    (leftActual := SemanticResult101823.actual selector witness)
    (rightActual := SemanticResult101759.actual selector witness)
    (leftRaw := SemanticResult101823.rawTerms)
    (rightRaw := SemanticResult101759.rawTerms)
    (working := LeftOperatorMerge101829.working)
    (leftBinding := 101824) (rightBinding := 101825)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10463⟩) (rightExpression := ⟨24898⟩)
    (coefficientTransfer := 101826) (summaryTransfer := 101828)
    (rightCoefficientProducer := 101758)
    (rightSummaryTransfer := 101827)
    (leftMaximum := ⟨95422080, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge101829.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority101758.actual selector witness)
    (summaryMagnitude := LeftBound101828.actual selector witness)
    (reconstruction := LeftOperatorMerge101829.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult101823.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult101759.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101758.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority101758.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge101829.operationAgreement
  · exact LeftBound101828.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge101829.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 101830 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22948⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22948⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge101829.working
    [{ coefficient := (-1), key := LeftRelationMerge101830.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge101830.frameStart
      LeftRelationMerge101830.owner (.relation 101830) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge101830.deltas
    rows := LeftRelationMerge101830.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge101829.working LeftRelationMerge101830.source
        (relationContext LeftRelationMerge101830.source
          LeftRelationMerge101830.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge101829.working, LeftRelationMerge101830.deltas,
    LeftRelationMerge101830.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 101830)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨24899⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9385⟩⟩, ⟨.program ⟨214⟩, ⟨10456⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24898⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge101829.working) (working := relationWorking0)
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
end SemanticResult101833

namespace SemanticResult101836
def owner : Owner := ⟨.program ⟨214⟩, ⟨19013⟩⟩
def rawTerms : List Term := Proof.Events397.exact101836RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 101836
def producerEvent : Nat := 101835
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult101836.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨7⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨7⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult101836

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
