import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard413
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard109
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard110
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard411
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard412

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult56927
def owner : Owner := ⟨.program ⟨214⟩, ⟨27448⟩⟩
def rawTerms : List Term := Proof.Events222.exact56927RawTerms
def summary : Bound := (.finite 1292001236604524572672)
def resultEvent : Nat := 56927
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56927.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge56924.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult56920.owner)
    (rightOwner := SemanticResult56742.owner)
    (leftResult := 56920) (rightResult := 56742)
    (leftActual := SemanticResult56920.actual selector witness)
    (rightActual := SemanticResult56742.actual selector witness)
    (leftRaw := SemanticResult56920.rawTerms)
    (rightRaw := SemanticResult56742.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292001234793221062656) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 56921) (rightBinding := 56922)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21119⟩) (rightExpression := ⟨27447⟩)
    (coefficientTransfer := 56923) (summaryTransfer := 56926)
    (base := LeftOperatorMerge56924.base)
    (reconstruction := LeftOperatorMerge56924.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56920.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56742.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge56924.operationAgreement
  · rfl
  · decide
end SemanticResult56927

namespace SemanticResult56934
def owner : Owner := ⟨.program ⟨214⟩, ⟨23976⟩⟩
def rawTerms : List Term := Proof.Events222.exact56934RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56934
def producerEvent : Nat := 56933
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56934.actual selector witness
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
end SemanticResult56934

namespace SemanticResult56937
def owner : Owner := ⟨.program ⟨214⟩, ⟨27228⟩⟩
def rawTerms : List Term := Proof.Events222.exact56937RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56937
def producerEvent : Nat := 56936
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56937.actual selector witness
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
end SemanticResult56937

namespace SemanticResult56944
def owner : Owner := ⟨.program ⟨214⟩, ⟨23460⟩⟩
def rawTerms : List Term := Proof.Events222.exact56944RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56944
def producerEvent : Nat := 56943
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56944.actual selector witness
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
end SemanticResult56944

namespace SemanticResult56947
def owner : Owner := ⟨.program ⟨214⟩, ⟨25840⟩⟩
def rawTerms : List Term := Proof.Events222.exact56947RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56947
def producerEvent : Nat := 56946
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56947.actual selector witness
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
end SemanticResult56947

namespace SemanticResult56952
def owner : Owner := ⟨.program ⟨214⟩, ⟨11222⟩⟩
def rawTerms : List Term := Proof.Events222.exact56952RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56952
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56952.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge56951.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge56951.frameStart)
    (transferEvent := 56950) (owner := owner)
    (leftResult := 2637) (rightResult := 50670)
    (working := LeftOperatorMerge56951.working)
    (reconstruction := LeftOperatorMerge56951.reconstruction)
    (leftReference := .predecessor 0 56948 .coefficient) (rightReference := .predecessor 1 56949 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2637.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge56951.operationAgreement
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
end SemanticResult56952

namespace SemanticResult56957
def owner : Owner := ⟨.program ⟨214⟩, ⟨7270⟩⟩
def rawTerms : List Term := Proof.Events222.exact56957RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56957
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56957.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge56956.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge56956.frameStart)
    (transferEvent := 56955) (owner := owner)
    (leftResult := 50540) (rightResult := 12985)
    (working := LeftOperatorMerge56956.working)
    (reconstruction := LeftOperatorMerge56956.reconstruction)
    (leftReference := .predecessor 0 56953 .coefficient) (rightReference := .predecessor 1 56954 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12985.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge56956.operationAgreement
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
end SemanticResult56957

namespace SemanticResult56961
def owner : Owner := ⟨.program ⟨214⟩, ⟨11223⟩⟩
def rawTerms : List Term := Proof.Events222.exact56961RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56961
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56961.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 56958) (rightBinding := 56959)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7270⟩) (rightExpression := ⟨11222⟩)
    (transferEvent := 56960)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56957.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56952.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult56961

namespace SemanticResult56967
def owner : Owner := ⟨.program ⟨214⟩, ⟨11224⟩⟩
def rawTerms : List Term := Proof.Events222.exact56967RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 56967
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56967.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 56964) (survivorTransfer := 56965)
    (survivorEvent := 56966) (resultEvent := resultEvent)
    (rightCoefficientProducer := 12976)
    (owner := owner) (leftOwner := SemanticResult56961.owner)
    (rightOwner := SemanticResult12977.owner)
    (leftResult := 56961) (rightResult := 12977)
    (leftBinding := 56962) (rightBinding := 56963)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11223⟩) (rightExpression := ⟨90⟩)
    (leftActual := SemanticResult56961.actual selector witness)
    (rightActual := SemanticResult12977.actual selector witness)
    (leftRaw := SemanticResult56961.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨90⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound12976.actual selector witness)
    (survivorMagnitude := LeftBound56965.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56961.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12977.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12976.derived selector witness)
  · exact LeftBound56965.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult56967

namespace SemanticResult56975
def owner : Owner := ⟨.program ⟨214⟩, ⟨13568⟩⟩
def rawTerms : List Term := Proof.Events222.exact56975RawTerms
def summary : Bound := (.finite 8320)
def resultEvent : Nat := 56975
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56975.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨10, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge56973.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge56973.frameStart)
    (owner := owner) (leftOwner := SemanticResult56967.owner)
    (rightOwner := SemanticResult2640.owner)
    (leftResult := 56967) (rightResult := 2640)
    (leftActual := SemanticResult56967.actual selector witness)
    (rightActual := SemanticResult2640.actual selector witness)
    (leftRaw := SemanticResult56967.rawTerms)
    (rightRaw := SemanticResult2640.rawTerms)
    (working := LeftOperatorMerge56973.working)
    (leftBinding := 56968) (rightBinding := 56969)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11224⟩) (rightExpression := ⟨13565⟩)
    (coefficientTransfer := 56970) (summaryTransfer := 56972)
    (rightCoefficientProducer := 2639)
    (rightSummaryTransfer := 56971)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨10, by decide⟩)
    (rightRecordedMaximum := 10)
    (rightSummaryMaximum := ⟨10, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge56973.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority2639.actual selector witness)
    (summaryMagnitude := LeftBound56972.actual selector witness)
    (reconstruction := LeftOperatorMerge56973.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56967.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2640.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2639.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority2639.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge56973.operationAgreement
  · exact LeftBound56972.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge56973.working summary) := by
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
end SemanticResult56975

namespace SemanticResult56980
def owner : Owner := ⟨.program ⟨214⟩, ⟨13569⟩⟩
def rawTerms : List Term := Proof.Events222.exact56980RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56980
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56980.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge56979.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge56979.frameStart)
    (transferEvent := 56978) (owner := owner)
    (leftResult := 2640) (rightResult := 50670)
    (working := LeftOperatorMerge56979.working)
    (reconstruction := LeftOperatorMerge56979.reconstruction)
    (leftReference := .predecessor 0 56976 .coefficient) (rightReference := .predecessor 1 56977 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2640.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge56979.operationAgreement
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
end SemanticResult56980

namespace SemanticResult56985
def owner : Owner := ⟨.program ⟨214⟩, ⟨7287⟩⟩
def rawTerms : List Term := Proof.Events222.exact56985RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56985
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56985.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge56984.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge56984.frameStart)
    (transferEvent := 56983) (owner := owner)
    (leftResult := 50540) (rightResult := 13026)
    (working := LeftOperatorMerge56984.working)
    (reconstruction := LeftOperatorMerge56984.reconstruction)
    (leftReference := .predecessor 0 56981 .coefficient) (rightReference := .predecessor 1 56982 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13026.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge56984.operationAgreement
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
end SemanticResult56985

namespace SemanticResult56989
def owner : Owner := ⟨.program ⟨214⟩, ⟨13570⟩⟩
def rawTerms : List Term := Proof.Events222.exact56989RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 56989
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56989.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 56986) (rightBinding := 56987)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7287⟩) (rightExpression := ⟨13569⟩)
    (transferEvent := 56988)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56985.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56980.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult56989

namespace SemanticResult56995
def owner : Owner := ⟨.program ⟨214⟩, ⟨13571⟩⟩
def rawTerms : List Term := Proof.Events222.exact56995RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 56995
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult56995.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 56992) (survivorTransfer := 56993)
    (survivorEvent := 56994) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13017)
    (owner := owner) (leftOwner := SemanticResult56989.owner)
    (rightOwner := SemanticResult13018.owner)
    (leftResult := 56989) (rightResult := 13018)
    (leftBinding := 56990) (rightBinding := 56991)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13570⟩) (rightExpression := ⟨107⟩)
    (leftActual := SemanticResult56989.actual selector witness)
    (rightActual := SemanticResult13018.actual selector witness)
    (leftRaw := SemanticResult56989.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨107⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13017.actual selector witness)
    (survivorMagnitude := LeftBound56993.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56989.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13018.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13017.derived selector witness)
  · exact LeftBound56993.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult56995

namespace SemanticResult57005
def owner : Owner := ⟨.program ⟨214⟩, ⟨13572⟩⟩
def rawTerms : List Term := Proof.Events222.exact57005RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 57005
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57005.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge57001.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge57001.frameStart)
    (owner := owner) (leftOwner := SemanticResult56995.owner)
    (rightOwner := SemanticResult13015.owner)
    (leftResult := 56995) (rightResult := 13015)
    (leftActual := SemanticResult56995.actual selector witness)
    (rightActual := SemanticResult13015.actual selector witness)
    (leftRaw := SemanticResult56995.rawTerms)
    (rightRaw := SemanticResult13015.rawTerms)
    (working := LeftOperatorMerge57001.working)
    (leftBinding := 56996) (rightBinding := 56997)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13571⟩) (rightExpression := ⟨7844⟩)
    (coefficientTransfer := 56998) (summaryTransfer := 57000)
    (rightCoefficientProducer := 13014)
    (rightSummaryTransfer := 56999)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge57001.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound13014.actual selector witness)
    (summaryMagnitude := LeftBound57000.actual selector witness)
    (reconstruction := LeftOperatorMerge57001.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult56995.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13015.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13014.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound13014.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge57001.operationAgreement
  · exact LeftBound57000.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge57001.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 57002 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge57001.working
    [{ coefficient := (-1), key := LeftRelationMerge57002.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge57002.frameStart
      LeftRelationMerge57002.owner (.relation 57002) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge57002.deltas
    rows := LeftRelationMerge57002.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge57001.working LeftRelationMerge57002.source
        (relationContext LeftRelationMerge57002.source
          LeftRelationMerge57002.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge57001.working, LeftRelationMerge57002.deltas,
    LeftRelationMerge57002.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 57002)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨13572⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨13565⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge57001.working) (working := relationWorking0)
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
end SemanticResult57005

namespace SemanticResult57011
def owner : Owner := ⟨.program ⟨214⟩, ⟨13573⟩⟩
def rawTerms : List Term := Proof.Events222.exact57011RawTerms
def summary : Bound := (.finite 95428736)
def resultEvent : Nat := 57011
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57011.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge57009.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult57005.owner)
    (rightOwner := SemanticResult56975.owner)
    (leftResult := 57005) (rightResult := 56975)
    (leftActual := SemanticResult57005.actual selector witness)
    (rightActual := SemanticResult56975.actual selector witness)
    (leftRaw := SemanticResult57005.rawTerms)
    (rightRaw := SemanticResult56975.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 8320) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 57006) (rightBinding := 57007)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13572⟩) (rightExpression := ⟨13568⟩)
    (coefficientTransfer := 57008) (summaryTransfer := 57010)
    (base := LeftOperatorMerge57009.base)
    (reconstruction := LeftOperatorMerge57009.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult57005.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult56975.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge57009.operationAgreement
  · rfl
  · decide
end SemanticResult57011

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
