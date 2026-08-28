import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard126
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard002
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard055
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard125

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult14986
def owner : Owner := ⟨.program ⟨214⟩, ⟨10515⟩⟩
def rawTerms : List Term := Proof.Events058.exact14986RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14986
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14986.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge14985.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge14985.frameStart)
    (transferEvent := 14984) (owner := owner)
    (leftResult := 442) (rightResult := 6449)
    (working := LeftOperatorMerge14985.working)
    (reconstruction := LeftOperatorMerge14985.reconstruction)
    (leftReference := .predecessor 0 14982 .coefficient) (rightReference := .predecessor 1 14983 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult442.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6449.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge14985.operationAgreement
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
end SemanticResult14986

namespace SemanticResult14989
def owner : Owner := ⟨.program ⟨214⟩, ⟨6772⟩⟩
def rawTerms : List Term := Proof.Events058.exact14989RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14989
def producerEvent : Nat := 14988
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14989.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 14987 .coefficient), 0, .large, .identity (.predecessor 0 14987 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult14989

namespace SemanticResult14994
def owner : Owner := ⟨.program ⟨214⟩, ⟨7380⟩⟩
def rawTerms : List Term := Proof.Events058.exact14994RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14994
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14994.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge14993.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge14993.frameStart)
    (transferEvent := 14992) (owner := owner)
    (leftResult := 6314) (rightResult := 14989)
    (working := LeftOperatorMerge14993.working)
    (reconstruction := LeftOperatorMerge14993.reconstruction)
    (leftReference := .predecessor 0 14990 .coefficient) (rightReference := .predecessor 1 14991 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult6314.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14989.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge14993.operationAgreement
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
end SemanticResult14994

namespace SemanticResult14998
def owner : Owner := ⟨.program ⟨214⟩, ⟨10516⟩⟩
def rawTerms : List Term := Proof.Events058.exact14998RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14998
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14998.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14995) (rightBinding := 14996)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7380⟩) (rightExpression := ⟨10515⟩)
    (transferEvent := 14997)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14994.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14986.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14998

namespace SemanticResult15004
def owner : Owner := ⟨.program ⟨214⟩, ⟨10517⟩⟩
def rawTerms : List Term := Proof.Events058.exact15004RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 15004
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15004.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 15001) (survivorTransfer := 15002)
    (survivorEvent := 15003) (resultEvent := resultEvent)
    (rightCoefficientProducer := 14980)
    (owner := owner) (leftOwner := SemanticResult14998.owner)
    (rightOwner := SemanticResult14981.owner)
    (leftResult := 14998) (rightResult := 14981)
    (leftBinding := 14999) (rightBinding := 15000)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10516⟩) (rightExpression := ⟨86⟩)
    (leftActual := SemanticResult14998.actual selector witness)
    (rightActual := SemanticResult14981.actual selector witness)
    (leftRaw := SemanticResult14998.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨86⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound14980.actual selector witness)
    (survivorMagnitude := LeftBound15002.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14998.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14981.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14980.derived selector witness)
  · exact LeftBound15002.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult15004

namespace SemanticResult15012
def owner : Owner := ⟨.program ⟨214⟩, ⟨10518⟩⟩
def rawTerms : List Term := Proof.Events058.exact15012RawTerms
def summary : Bound := (.finite 1664)
def resultEvent : Nat := 15012
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15012.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨2, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge15010.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge15010.frameStart)
    (owner := owner) (leftOwner := SemanticResult15004.owner)
    (rightOwner := SemanticResult445.owner)
    (leftResult := 15004) (rightResult := 445)
    (leftActual := SemanticResult15004.actual selector witness)
    (rightActual := SemanticResult445.actual selector witness)
    (leftRaw := SemanticResult15004.rawTerms)
    (rightRaw := SemanticResult445.rawTerms)
    (working := LeftOperatorMerge15010.working)
    (leftBinding := 15005) (rightBinding := 15006)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10517⟩) (rightExpression := ⟨9420⟩)
    (coefficientTransfer := 15007) (summaryTransfer := 15009)
    (rightCoefficientProducer := 444)
    (rightSummaryTransfer := 15008)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨2, by decide⟩)
    (rightRecordedMaximum := 2)
    (rightSummaryMaximum := ⟨2, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge15010.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority444.actual selector witness)
    (summaryMagnitude := LeftBound15009.actual selector witness)
    (reconstruction := LeftOperatorMerge15010.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15004.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult445.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority444.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority444.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge15010.operationAgreement
  · exact LeftBound15009.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge15010.working summary) := by
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
end SemanticResult15012

namespace SemanticResult15015
def owner : Owner := ⟨.program ⟨214⟩, ⟨7831⟩⟩
def rawTerms : List Term := Proof.Events058.exact15015RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15015
def producerEvent : Nat := 15014
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15015.actual selector witness
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
end SemanticResult15015

namespace SemanticResult15019
def owner : Owner := ⟨.program ⟨214⟩, ⟨7832⟩⟩
def rawTerms : List Term := Proof.Events058.exact15019RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15019
def producerEvent : Nat := 15018
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15019.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 15016 .coefficient) (.value (.predecessor 1 15017 .coefficient)), 0, .finite 8192, .scale (.predecessor 0 15016 .coefficient) (.value (.predecessor 1 15017 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult15019

namespace SemanticResult15022
def owner : Owner := ⟨.program ⟨214⟩, ⟨85⟩⟩
def rawTerms : List Term := Proof.Events058.exact15022RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15022
def producerEvent : Nat := 15021
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15022.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 15020 .coefficient), 0, .finite 26, .identity (.predecessor 0 15020 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult15022

namespace SemanticResult15027
def owner : Owner := ⟨.program ⟨214⟩, ⟨9421⟩⟩
def rawTerms : List Term := Proof.Events058.exact15027RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15027
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15027.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge15026.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge15026.frameStart)
    (transferEvent := 15025) (owner := owner)
    (leftResult := 445) (rightResult := 6449)
    (working := LeftOperatorMerge15026.working)
    (reconstruction := LeftOperatorMerge15026.reconstruction)
    (leftReference := .predecessor 0 15023 .coefficient) (rightReference := .predecessor 1 15024 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult445.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6449.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge15026.operationAgreement
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
end SemanticResult15027

namespace SemanticResult15030
def owner : Owner := ⟨.program ⟨214⟩, ⟨6771⟩⟩
def rawTerms : List Term := Proof.Events058.exact15030RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15030
def producerEvent : Nat := 15029
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15030.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 15028 .coefficient), 0, .large, .identity (.predecessor 0 15028 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult15030

namespace SemanticResult15035
def owner : Owner := ⟨.program ⟨214⟩, ⟨7379⟩⟩
def rawTerms : List Term := Proof.Events058.exact15035RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15035
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15035.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge15034.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge15034.frameStart)
    (transferEvent := 15033) (owner := owner)
    (leftResult := 6314) (rightResult := 15030)
    (working := LeftOperatorMerge15034.working)
    (reconstruction := LeftOperatorMerge15034.reconstruction)
    (leftReference := .predecessor 0 15031 .coefficient) (rightReference := .predecessor 1 15032 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult6314.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15030.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge15034.operationAgreement
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
end SemanticResult15035

namespace SemanticResult15039
def owner : Owner := ⟨.program ⟨214⟩, ⟨9422⟩⟩
def rawTerms : List Term := Proof.Events058.exact15039RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15039
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15039.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15036) (rightBinding := 15037)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7379⟩) (rightExpression := ⟨9421⟩)
    (transferEvent := 15038)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15035.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15027.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15039

namespace SemanticResult15045
def owner : Owner := ⟨.program ⟨214⟩, ⟨9423⟩⟩
def rawTerms : List Term := Proof.Events058.exact15045RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 15045
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15045.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 15042) (survivorTransfer := 15043)
    (survivorEvent := 15044) (resultEvent := resultEvent)
    (rightCoefficientProducer := 15021)
    (owner := owner) (leftOwner := SemanticResult15039.owner)
    (rightOwner := SemanticResult15022.owner)
    (leftResult := 15039) (rightResult := 15022)
    (leftBinding := 15040) (rightBinding := 15041)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9422⟩) (rightExpression := ⟨85⟩)
    (leftActual := SemanticResult15039.actual selector witness)
    (rightActual := SemanticResult15022.actual selector witness)
    (leftRaw := SemanticResult15039.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨85⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound15021.actual selector witness)
    (survivorMagnitude := LeftBound15043.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15039.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15022.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15021.derived selector witness)
  · exact LeftBound15043.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult15045

namespace SemanticResult15055
def owner : Owner := ⟨.program ⟨214⟩, ⟨9424⟩⟩
def rawTerms : List Term := Proof.Events058.exact15055RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 15055
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15055.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge15051.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge15051.frameStart)
    (owner := owner) (leftOwner := SemanticResult15045.owner)
    (rightOwner := SemanticResult15019.owner)
    (leftResult := 15045) (rightResult := 15019)
    (leftActual := SemanticResult15045.actual selector witness)
    (rightActual := SemanticResult15019.actual selector witness)
    (leftRaw := SemanticResult15045.rawTerms)
    (rightRaw := SemanticResult15019.rawTerms)
    (working := LeftOperatorMerge15051.working)
    (leftBinding := 15046) (rightBinding := 15047)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9423⟩) (rightExpression := ⟨7832⟩)
    (coefficientTransfer := 15048) (summaryTransfer := 15050)
    (rightCoefficientProducer := 15018)
    (rightSummaryTransfer := 15049)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge15051.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound15018.actual selector witness)
    (summaryMagnitude := LeftBound15050.actual selector witness)
    (reconstruction := LeftOperatorMerge15051.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15045.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15019.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15018.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound15018.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge15051.operationAgreement
  · exact LeftBound15050.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge15051.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 15052 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9420⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge15051.working
    [{ coefficient := (-1), key := LeftRelationMerge15052.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge15052.frameStart
      LeftRelationMerge15052.owner (.relation 15052) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge15052.deltas
    rows := LeftRelationMerge15052.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge15051.working LeftRelationMerge15052.source
        (relationContext LeftRelationMerge15052.source
          LeftRelationMerge15052.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge15051.working, LeftRelationMerge15052.deltas,
    LeftRelationMerge15052.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 15052)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9424⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9420⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge15051.working) (working := relationWorking0)
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
end SemanticResult15055

namespace SemanticResult15061
def owner : Owner := ⟨.program ⟨214⟩, ⟨10519⟩⟩
def rawTerms : List Term := Proof.Events058.exact15061RawTerms
def summary : Bound := (.finite 95422080)
def resultEvent : Nat := 15061
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15061.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge15059.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult15055.owner)
    (rightOwner := SemanticResult15012.owner)
    (leftResult := 15055) (rightResult := 15012)
    (leftActual := SemanticResult15055.actual selector witness)
    (rightActual := SemanticResult15012.actual selector witness)
    (leftRaw := SemanticResult15055.rawTerms)
    (rightRaw := SemanticResult15012.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 1664) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 15056) (rightBinding := 15057)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9424⟩) (rightExpression := ⟨10518⟩)
    (coefficientTransfer := 15058) (summaryTransfer := 15060)
    (base := LeftOperatorMerge15059.base)
    (reconstruction := LeftOperatorMerge15059.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15055.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15012.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge15059.operationAgreement
  · rfl
  · decide
end SemanticResult15061

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
