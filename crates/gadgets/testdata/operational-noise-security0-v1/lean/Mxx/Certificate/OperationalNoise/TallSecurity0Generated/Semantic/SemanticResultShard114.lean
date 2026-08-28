import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard114
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard002
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard055
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard113

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult13495
def owner : Owner := ⟨.program ⟨214⟩, ⟨11151⟩⟩
def rawTerms : List Term := Proof.Events052.exact13495RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13495
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13495.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13492) (rightBinding := 13493)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7383⟩) (rightExpression := ⟨11150⟩)
    (transferEvent := 13494)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13491.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13483.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13495

namespace SemanticResult13501
def owner : Owner := ⟨.program ⟨214⟩, ⟨11152⟩⟩
def rawTerms : List Term := Proof.Events052.exact13501RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 13501
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13501.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 13498) (survivorTransfer := 13499)
    (survivorEvent := 13500) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13477)
    (owner := owner) (leftOwner := SemanticResult13495.owner)
    (rightOwner := SemanticResult13478.owner)
    (leftResult := 13495) (rightResult := 13478)
    (leftBinding := 13496) (rightBinding := 13497)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11151⟩) (rightExpression := ⟨89⟩)
    (leftActual := SemanticResult13495.actual selector witness)
    (rightActual := SemanticResult13478.actual selector witness)
    (leftRaw := SemanticResult13495.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨89⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13477.actual selector witness)
    (survivorMagnitude := LeftBound13499.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13495.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13478.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13477.derived selector witness)
  · exact LeftBound13499.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult13501

namespace SemanticResult13509
def owner : Owner := ⟨.program ⟨214⟩, ⟨12202⟩⟩
def rawTerms : List Term := Proof.Events052.exact13509RawTerms
def summary : Bound := (.finite 4992)
def resultEvent : Nat := 13509
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13509.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨6, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge13507.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge13507.frameStart)
    (owner := owner) (leftOwner := SemanticResult13501.owner)
    (rightOwner := SemanticResult376.owner)
    (leftResult := 13501) (rightResult := 376)
    (leftActual := SemanticResult13501.actual selector witness)
    (rightActual := SemanticResult376.actual selector witness)
    (leftRaw := SemanticResult13501.rawTerms)
    (rightRaw := SemanticResult376.rawTerms)
    (working := LeftOperatorMerge13507.working)
    (leftBinding := 13502) (rightBinding := 13503)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11152⟩) (rightExpression := ⟨12199⟩)
    (coefficientTransfer := 13504) (summaryTransfer := 13506)
    (rightCoefficientProducer := 375)
    (rightSummaryTransfer := 13505)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨6, by decide⟩)
    (rightRecordedMaximum := 6)
    (rightSummaryMaximum := ⟨6, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge13507.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority375.actual selector witness)
    (summaryMagnitude := LeftBound13506.actual selector witness)
    (reconstruction := LeftOperatorMerge13507.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13501.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult376.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority375.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority375.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge13507.operationAgreement
  · exact LeftBound13506.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge13507.working summary) := by
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
end SemanticResult13509

namespace SemanticResult13512
def owner : Owner := ⟨.program ⟨214⟩, ⟨7840⟩⟩
def rawTerms : List Term := Proof.Events052.exact13512RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13512
def producerEvent : Nat := 13511
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13512.actual selector witness
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
end SemanticResult13512

namespace SemanticResult13516
def owner : Owner := ⟨.program ⟨214⟩, ⟨7841⟩⟩
def rawTerms : List Term := Proof.Events052.exact13516RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13516
def producerEvent : Nat := 13515
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13516.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 13513 .coefficient) (.value (.predecessor 1 13514 .coefficient)), 0, .finite 8192, .scale (.predecessor 0 13513 .coefficient) (.value (.predecessor 1 13514 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult13516

namespace SemanticResult13519
def owner : Owner := ⟨.program ⟨214⟩, ⟨106⟩⟩
def rawTerms : List Term := Proof.Events052.exact13519RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13519
def producerEvent : Nat := 13518
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13519.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 13517 .coefficient), 0, .finite 26, .identity (.predecessor 0 13517 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult13519

namespace SemanticResult13524
def owner : Owner := ⟨.program ⟨214⟩, ⟨12203⟩⟩
def rawTerms : List Term := Proof.Events052.exact13524RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13524
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13524.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge13523.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge13523.frameStart)
    (transferEvent := 13522) (owner := owner)
    (leftResult := 376) (rightResult := 6449)
    (working := LeftOperatorMerge13523.working)
    (reconstruction := LeftOperatorMerge13523.reconstruction)
    (leftReference := .predecessor 0 13520 .coefficient) (rightReference := .predecessor 1 13521 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult376.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6449.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge13523.operationAgreement
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
end SemanticResult13524

namespace SemanticResult13527
def owner : Owner := ⟨.program ⟨214⟩, ⟨6792⟩⟩
def rawTerms : List Term := Proof.Events052.exact13527RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13527
def producerEvent : Nat := 13526
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13527.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 13525 .coefficient), 0, .large, .identity (.predecessor 0 13525 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult13527

namespace SemanticResult13532
def owner : Owner := ⟨.program ⟨214⟩, ⟨7400⟩⟩
def rawTerms : List Term := Proof.Events052.exact13532RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13532
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13532.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge13531.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge13531.frameStart)
    (transferEvent := 13530) (owner := owner)
    (leftResult := 6314) (rightResult := 13527)
    (working := LeftOperatorMerge13531.working)
    (reconstruction := LeftOperatorMerge13531.reconstruction)
    (leftReference := .predecessor 0 13528 .coefficient) (rightReference := .predecessor 1 13529 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult6314.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13527.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge13531.operationAgreement
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
end SemanticResult13532

namespace SemanticResult13536
def owner : Owner := ⟨.program ⟨214⟩, ⟨12204⟩⟩
def rawTerms : List Term := Proof.Events052.exact13536RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13536
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13536.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13533) (rightBinding := 13534)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7400⟩) (rightExpression := ⟨12203⟩)
    (transferEvent := 13535)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13532.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13524.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13536

namespace SemanticResult13542
def owner : Owner := ⟨.program ⟨214⟩, ⟨12205⟩⟩
def rawTerms : List Term := Proof.Events052.exact13542RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 13542
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13542.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 13539) (survivorTransfer := 13540)
    (survivorEvent := 13541) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13518)
    (owner := owner) (leftOwner := SemanticResult13536.owner)
    (rightOwner := SemanticResult13519.owner)
    (leftResult := 13536) (rightResult := 13519)
    (leftBinding := 13537) (rightBinding := 13538)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12204⟩) (rightExpression := ⟨106⟩)
    (leftActual := SemanticResult13536.actual selector witness)
    (rightActual := SemanticResult13519.actual selector witness)
    (leftRaw := SemanticResult13536.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨106⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13518.actual selector witness)
    (survivorMagnitude := LeftBound13540.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13536.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13519.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13518.derived selector witness)
  · exact LeftBound13540.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult13542

namespace SemanticResult13552
def owner : Owner := ⟨.program ⟨214⟩, ⟨12206⟩⟩
def rawTerms : List Term := Proof.Events052.exact13552RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 13552
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13552.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge13548.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge13548.frameStart)
    (owner := owner) (leftOwner := SemanticResult13542.owner)
    (rightOwner := SemanticResult13516.owner)
    (leftResult := 13542) (rightResult := 13516)
    (leftActual := SemanticResult13542.actual selector witness)
    (rightActual := SemanticResult13516.actual selector witness)
    (leftRaw := SemanticResult13542.rawTerms)
    (rightRaw := SemanticResult13516.rawTerms)
    (working := LeftOperatorMerge13548.working)
    (leftBinding := 13543) (rightBinding := 13544)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12205⟩) (rightExpression := ⟨7841⟩)
    (coefficientTransfer := 13545) (summaryTransfer := 13547)
    (rightCoefficientProducer := 13515)
    (rightSummaryTransfer := 13546)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge13548.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound13515.actual selector witness)
    (summaryMagnitude := LeftBound13547.actual selector witness)
    (reconstruction := LeftOperatorMerge13548.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13542.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13516.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13515.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound13515.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge13548.operationAgreement
  · exact LeftBound13547.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge13548.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 13549 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge13548.working
    [{ coefficient := (-1), key := LeftRelationMerge13549.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge13549.frameStart
      LeftRelationMerge13549.owner (.relation 13549) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge13549.deltas
    rows := LeftRelationMerge13549.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge13548.working LeftRelationMerge13549.source
        (relationContext LeftRelationMerge13549.source
          LeftRelationMerge13549.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge13548.working, LeftRelationMerge13549.deltas,
    LeftRelationMerge13549.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 13549)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨12206⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge13548.working) (working := relationWorking0)
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
end SemanticResult13552

namespace SemanticResult13558
def owner : Owner := ⟨.program ⟨214⟩, ⟨12207⟩⟩
def rawTerms : List Term := Proof.Events052.exact13558RawTerms
def summary : Bound := (.finite 95425408)
def resultEvent : Nat := 13558
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13558.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge13556.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult13552.owner)
    (rightOwner := SemanticResult13509.owner)
    (leftResult := 13552) (rightResult := 13509)
    (leftActual := SemanticResult13552.actual selector witness)
    (rightActual := SemanticResult13509.actual selector witness)
    (leftRaw := SemanticResult13552.rawTerms)
    (rightRaw := SemanticResult13509.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 4992) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 13553) (rightBinding := 13554)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12206⟩) (rightExpression := ⟨12202⟩)
    (coefficientTransfer := 13555) (summaryTransfer := 13557)
    (base := LeftOperatorMerge13556.base)
    (reconstruction := LeftOperatorMerge13556.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13552.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13509.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge13556.operationAgreement
  · rfl
  · decide
end SemanticResult13558

namespace SemanticResult13568
def owner : Owner := ⟨.program ⟨214⟩, ⟨25317⟩⟩
def rawTerms : List Term := Proof.Events053.exact13568RawTerms
def summary : Bound := (.finite 350212774166528)
def resultEvent : Nat := 13568
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13568.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95425408, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge13564.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge13564.frameStart)
    (owner := owner) (leftOwner := SemanticResult13558.owner)
    (rightOwner := SemanticResult13475.owner)
    (leftResult := 13558) (rightResult := 13475)
    (leftActual := SemanticResult13558.actual selector witness)
    (rightActual := SemanticResult13475.actual selector witness)
    (leftRaw := SemanticResult13558.rawTerms)
    (rightRaw := SemanticResult13475.rawTerms)
    (working := LeftOperatorMerge13564.working)
    (leftBinding := 13559) (rightBinding := 13560)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12207⟩) (rightExpression := ⟨25316⟩)
    (coefficientTransfer := 13561) (summaryTransfer := 13563)
    (rightCoefficientProducer := 13474)
    (rightSummaryTransfer := 13562)
    (leftMaximum := ⟨95425408, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge13564.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority13474.actual selector witness)
    (summaryMagnitude := LeftBound13563.actual selector witness)
    (reconstruction := LeftOperatorMerge13564.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13558.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13475.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13474.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority13474.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge13564.operationAgreement
  · exact LeftBound13563.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge13564.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 13565 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25316⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23172⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23172⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge13564.working
    [{ coefficient := (-1), key := LeftRelationMerge13565.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge13565.frameStart
      LeftRelationMerge13565.owner (.relation 13565) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge13565.deltas
    rows := LeftRelationMerge13565.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge13564.working LeftRelationMerge13565.source
        (relationContext LeftRelationMerge13565.source
          LeftRelationMerge13565.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge13564.working, LeftRelationMerge13565.deltas,
    LeftRelationMerge13565.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 13565)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25317⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11149⟩⟩, ⟨.program ⟨214⟩, ⟨12199⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25316⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25316⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge13564.working) (working := relationWorking0)
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
end SemanticResult13568

namespace SemanticResult13571
def owner : Owner := ⟨.program ⟨214⟩, ⟨19256⟩⟩
def rawTerms : List Term := Proof.Events053.exact13571RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13571
def producerEvent : Nat := 13570
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13571.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨10⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨10⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult13571

namespace SemanticResult13575
def owner : Owner := ⟨.program ⟨214⟩, ⟨19258⟩⟩
def rawTerms : List Term := Proof.Events053.exact13575RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13575
def producerEvent : Nat := 13574
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13575.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 13572 .coefficient) (.value (.predecessor 1 13573 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 13572 .coefficient) (.value (.predecessor 1 13573 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult13575

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
