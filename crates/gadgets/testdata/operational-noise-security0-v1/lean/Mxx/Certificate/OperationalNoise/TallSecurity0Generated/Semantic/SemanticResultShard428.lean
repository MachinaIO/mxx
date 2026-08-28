import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard428
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard125
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard126
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard365

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult58875
def owner : Owner := ⟨.program ⟨214⟩, ⟨24916⟩⟩
def rawTerms : List Term := Proof.Events229.exact58875RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 58875
def producerEvent : Nat := 58874
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult58875.actual selector witness
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
end SemanticResult58875

namespace SemanticResult58880
def owner : Owner := ⟨.program ⟨214⟩, ⟨10491⟩⟩
def rawTerms : List Term := Proof.Events230.exact58880RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 58880
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult58880.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge58879.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge58879.frameStart)
    (transferEvent := 58878) (owner := owner)
    (leftResult := 2729) (rightResult := 50670)
    (working := LeftOperatorMerge58879.working)
    (reconstruction := LeftOperatorMerge58879.reconstruction)
    (leftReference := .predecessor 0 58876 .coefficient) (rightReference := .predecessor 1 58877 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2729.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge58879.operationAgreement
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
end SemanticResult58880

namespace SemanticResult58885
def owner : Owner := ⟨.program ⟨214⟩, ⟨7266⟩⟩
def rawTerms : List Term := Proof.Events230.exact58885RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 58885
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult58885.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge58884.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge58884.frameStart)
    (transferEvent := 58883) (owner := owner)
    (leftResult := 50540) (rightResult := 14989)
    (working := LeftOperatorMerge58884.working)
    (reconstruction := LeftOperatorMerge58884.reconstruction)
    (leftReference := .predecessor 0 58881 .coefficient) (rightReference := .predecessor 1 58882 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14989.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge58884.operationAgreement
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
end SemanticResult58885

namespace SemanticResult58889
def owner : Owner := ⟨.program ⟨214⟩, ⟨10492⟩⟩
def rawTerms : List Term := Proof.Events230.exact58889RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 58889
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult58889.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 58886) (rightBinding := 58887)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7266⟩) (rightExpression := ⟨10491⟩)
    (transferEvent := 58888)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult58885.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult58880.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult58889

namespace SemanticResult58895
def owner : Owner := ⟨.program ⟨214⟩, ⟨10493⟩⟩
def rawTerms : List Term := Proof.Events230.exact58895RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 58895
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult58895.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 58892) (survivorTransfer := 58893)
    (survivorEvent := 58894) (resultEvent := resultEvent)
    (rightCoefficientProducer := 14980)
    (owner := owner) (leftOwner := SemanticResult58889.owner)
    (rightOwner := SemanticResult14981.owner)
    (leftResult := 58889) (rightResult := 14981)
    (leftBinding := 58890) (rightBinding := 58891)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10492⟩) (rightExpression := ⟨86⟩)
    (leftActual := SemanticResult58889.actual selector witness)
    (rightActual := SemanticResult14981.actual selector witness)
    (leftRaw := SemanticResult58889.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨86⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound14980.actual selector witness)
    (survivorMagnitude := LeftBound58893.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult58889.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14981.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14980.derived selector witness)
  · exact LeftBound58893.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult58895

namespace SemanticResult58903
def owner : Owner := ⟨.program ⟨214⟩, ⟨10494⟩⟩
def rawTerms : List Term := Proof.Events230.exact58903RawTerms
def summary : Bound := (.finite 1664)
def resultEvent : Nat := 58903
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult58903.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨2, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge58901.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge58901.frameStart)
    (owner := owner) (leftOwner := SemanticResult58895.owner)
    (rightOwner := SemanticResult2732.owner)
    (leftResult := 58895) (rightResult := 2732)
    (leftActual := SemanticResult58895.actual selector witness)
    (rightActual := SemanticResult2732.actual selector witness)
    (leftRaw := SemanticResult58895.rawTerms)
    (rightRaw := SemanticResult2732.rawTerms)
    (working := LeftOperatorMerge58901.working)
    (leftBinding := 58896) (rightBinding := 58897)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10493⟩) (rightExpression := ⟨9405⟩)
    (coefficientTransfer := 58898) (summaryTransfer := 58900)
    (rightCoefficientProducer := 2731)
    (rightSummaryTransfer := 58899)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨2, by decide⟩)
    (rightRecordedMaximum := 2)
    (rightSummaryMaximum := ⟨2, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge58901.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority2731.actual selector witness)
    (summaryMagnitude := LeftBound58900.actual selector witness)
    (reconstruction := LeftOperatorMerge58901.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult58895.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2732.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2731.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority2731.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge58901.operationAgreement
  · exact LeftBound58900.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge58901.working summary) := by
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
end SemanticResult58903

namespace SemanticResult58908
def owner : Owner := ⟨.program ⟨214⟩, ⟨9406⟩⟩
def rawTerms : List Term := Proof.Events230.exact58908RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 58908
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult58908.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge58907.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge58907.frameStart)
    (transferEvent := 58906) (owner := owner)
    (leftResult := 2732) (rightResult := 50670)
    (working := LeftOperatorMerge58907.working)
    (reconstruction := LeftOperatorMerge58907.reconstruction)
    (leftReference := .predecessor 0 58904 .coefficient) (rightReference := .predecessor 1 58905 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2732.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge58907.operationAgreement
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
end SemanticResult58908

namespace SemanticResult58913
def owner : Owner := ⟨.program ⟨214⟩, ⟨7265⟩⟩
def rawTerms : List Term := Proof.Events230.exact58913RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 58913
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult58913.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge58912.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge58912.frameStart)
    (transferEvent := 58911) (owner := owner)
    (leftResult := 50540) (rightResult := 15030)
    (working := LeftOperatorMerge58912.working)
    (reconstruction := LeftOperatorMerge58912.reconstruction)
    (leftReference := .predecessor 0 58909 .coefficient) (rightReference := .predecessor 1 58910 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15030.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge58912.operationAgreement
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
end SemanticResult58913

namespace SemanticResult58917
def owner : Owner := ⟨.program ⟨214⟩, ⟨9407⟩⟩
def rawTerms : List Term := Proof.Events230.exact58917RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 58917
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult58917.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 58914) (rightBinding := 58915)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7265⟩) (rightExpression := ⟨9406⟩)
    (transferEvent := 58916)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult58913.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult58908.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult58917

namespace SemanticResult58923
def owner : Owner := ⟨.program ⟨214⟩, ⟨9408⟩⟩
def rawTerms : List Term := Proof.Events230.exact58923RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 58923
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult58923.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 58920) (survivorTransfer := 58921)
    (survivorEvent := 58922) (resultEvent := resultEvent)
    (rightCoefficientProducer := 15021)
    (owner := owner) (leftOwner := SemanticResult58917.owner)
    (rightOwner := SemanticResult15022.owner)
    (leftResult := 58917) (rightResult := 15022)
    (leftBinding := 58918) (rightBinding := 58919)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9407⟩) (rightExpression := ⟨85⟩)
    (leftActual := SemanticResult58917.actual selector witness)
    (rightActual := SemanticResult15022.actual selector witness)
    (leftRaw := SemanticResult58917.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨85⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound15021.actual selector witness)
    (survivorMagnitude := LeftBound58921.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult58917.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15022.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15021.derived selector witness)
  · exact LeftBound58921.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult58923

namespace SemanticResult58933
def owner : Owner := ⟨.program ⟨214⟩, ⟨9409⟩⟩
def rawTerms : List Term := Proof.Events230.exact58933RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 58933
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult58933.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge58929.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge58929.frameStart)
    (owner := owner) (leftOwner := SemanticResult58923.owner)
    (rightOwner := SemanticResult15019.owner)
    (leftResult := 58923) (rightResult := 15019)
    (leftActual := SemanticResult58923.actual selector witness)
    (rightActual := SemanticResult15019.actual selector witness)
    (leftRaw := SemanticResult58923.rawTerms)
    (rightRaw := SemanticResult15019.rawTerms)
    (working := LeftOperatorMerge58929.working)
    (leftBinding := 58924) (rightBinding := 58925)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9408⟩) (rightExpression := ⟨7832⟩)
    (coefficientTransfer := 58926) (summaryTransfer := 58928)
    (rightCoefficientProducer := 15018)
    (rightSummaryTransfer := 58927)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge58929.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound15018.actual selector witness)
    (summaryMagnitude := LeftBound58928.actual selector witness)
    (reconstruction := LeftOperatorMerge58929.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult58923.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15019.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15018.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound15018.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge58929.operationAgreement
  · exact LeftBound58928.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge58929.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 58930 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9405⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge58929.working
    [{ coefficient := (-1), key := LeftRelationMerge58930.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge58930.frameStart
      LeftRelationMerge58930.owner (.relation 58930) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge58930.deltas
    rows := LeftRelationMerge58930.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge58929.working LeftRelationMerge58930.source
        (relationContext LeftRelationMerge58930.source
          LeftRelationMerge58930.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge58929.working, LeftRelationMerge58930.deltas,
    LeftRelationMerge58930.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 58930)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9409⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9405⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge58929.working) (working := relationWorking0)
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
end SemanticResult58933

namespace SemanticResult58939
def owner : Owner := ⟨.program ⟨214⟩, ⟨10495⟩⟩
def rawTerms : List Term := Proof.Events230.exact58939RawTerms
def summary : Bound := (.finite 95422080)
def resultEvent : Nat := 58939
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult58939.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge58937.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult58933.owner)
    (rightOwner := SemanticResult58903.owner)
    (leftResult := 58933) (rightResult := 58903)
    (leftActual := SemanticResult58933.actual selector witness)
    (rightActual := SemanticResult58903.actual selector witness)
    (leftRaw := SemanticResult58933.rawTerms)
    (rightRaw := SemanticResult58903.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 1664) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 58934) (rightBinding := 58935)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9409⟩) (rightExpression := ⟨10494⟩)
    (coefficientTransfer := 58936) (summaryTransfer := 58938)
    (base := LeftOperatorMerge58937.base)
    (reconstruction := LeftOperatorMerge58937.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult58933.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult58903.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge58937.operationAgreement
  · rfl
  · decide
end SemanticResult58939

namespace SemanticResult58949
def owner : Owner := ⟨.program ⟨214⟩, ⟨24917⟩⟩
def rawTerms : List Term := Proof.Events230.exact58949RawTerms
def summary : Bound := (.finite 350200560353280)
def resultEvent : Nat := 58949
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult58949.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95422080, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge58945.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge58945.frameStart)
    (owner := owner) (leftOwner := SemanticResult58939.owner)
    (rightOwner := SemanticResult58875.owner)
    (leftResult := 58939) (rightResult := 58875)
    (leftActual := SemanticResult58939.actual selector witness)
    (rightActual := SemanticResult58875.actual selector witness)
    (leftRaw := SemanticResult58939.rawTerms)
    (rightRaw := SemanticResult58875.rawTerms)
    (working := LeftOperatorMerge58945.working)
    (leftBinding := 58940) (rightBinding := 58941)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10495⟩) (rightExpression := ⟨24916⟩)
    (coefficientTransfer := 58942) (summaryTransfer := 58944)
    (rightCoefficientProducer := 58874)
    (rightSummaryTransfer := 58943)
    (leftMaximum := ⟨95422080, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge58945.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority58874.actual selector witness)
    (summaryMagnitude := LeftBound58944.actual selector witness)
    (reconstruction := LeftOperatorMerge58945.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult58939.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult58875.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58874.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority58874.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge58945.operationAgreement
  · exact LeftBound58944.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge58945.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 58946 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24916⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9405⟩⟩, ⟨.program ⟨214⟩, ⟨10488⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22956⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22956⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge58945.working
    [{ coefficient := (-1), key := LeftRelationMerge58946.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge58946.frameStart
      LeftRelationMerge58946.owner (.relation 58946) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge58946.deltas
    rows := LeftRelationMerge58946.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge58945.working LeftRelationMerge58946.source
        (relationContext LeftRelationMerge58946.source
          LeftRelationMerge58946.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge58945.working, LeftRelationMerge58946.deltas,
    LeftRelationMerge58946.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 58946)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨24917⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9405⟩⟩, ⟨.program ⟨214⟩, ⟨10488⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24916⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24916⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge58945.working) (working := relationWorking0)
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
end SemanticResult58949

namespace SemanticResult58952
def owner : Owner := ⟨.program ⟨214⟩, ⟨19028⟩⟩
def rawTerms : List Term := Proof.Events230.exact58952RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 58952
def producerEvent : Nat := 58951
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult58952.actual selector witness
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
end SemanticResult58952

namespace SemanticResult58956
def owner : Owner := ⟨.program ⟨214⟩, ⟨19030⟩⟩
def rawTerms : List Term := Proof.Events230.exact58956RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 58956
def producerEvent : Nat := 58955
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult58956.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 58953 .coefficient) (.value (.predecessor 1 58954 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 58953 .coefficient) (.value (.predecessor 1 58954 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult58956

namespace SemanticResult59034
def owner : Owner := ⟨.program ⟨214⟩, ⟨10488⟩⟩
def rawTerms : List Term := Proof.Events230.exact59034RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 59034
def producerEvent : Nat := 59033
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult59034.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 59011, .finite 2, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult59034

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
