import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard369
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard061
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard365

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult51163
def owner : Owner := ⟨.program ⟨214⟩, ⟨25686⟩⟩
def rawTerms : List Term := Proof.Events199.exact51163RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51163
def producerEvent : Nat := 51162
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51163.actual selector witness
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
end SemanticResult51163

namespace SemanticResult51168
def owner : Owner := ⟨.program ⟨214⟩, ⟨13165⟩⟩
def rawTerms : List Term := Proof.Events199.exact51168RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51168
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51168.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge51167.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge51167.frameStart)
    (transferEvent := 51166) (owner := owner)
    (leftResult := 2361) (rightResult := 50670)
    (working := LeftOperatorMerge51167.working)
    (reconstruction := LeftOperatorMerge51167.reconstruction)
    (leftReference := .predecessor 0 51164 .coefficient) (rightReference := .predecessor 1 51165 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2361.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge51167.operationAgreement
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
end SemanticResult51168

namespace SemanticResult51173
def owner : Owner := ⟨.program ⟨214⟩, ⟨7283⟩⟩
def rawTerms : List Term := Proof.Events199.exact51173RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51173
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51173.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge51172.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge51172.frameStart)
    (transferEvent := 51171) (owner := owner)
    (leftResult := 50540) (rightResult := 6973)
    (working := LeftOperatorMerge51172.working)
    (reconstruction := LeftOperatorMerge51172.reconstruction)
    (leftReference := .predecessor 0 51169 .coefficient) (rightReference := .predecessor 1 51170 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6973.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge51172.operationAgreement
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
end SemanticResult51173

namespace SemanticResult51177
def owner : Owner := ⟨.program ⟨214⟩, ⟨13166⟩⟩
def rawTerms : List Term := Proof.Events199.exact51177RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51177
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51177.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 51174) (rightBinding := 51175)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7283⟩) (rightExpression := ⟨13165⟩)
    (transferEvent := 51176)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult51173.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult51168.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult51177

namespace SemanticResult51183
def owner : Owner := ⟨.program ⟨214⟩, ⟨13167⟩⟩
def rawTerms : List Term := Proof.Events199.exact51183RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 51183
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51183.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 51180) (survivorTransfer := 51181)
    (survivorEvent := 51182) (resultEvent := resultEvent)
    (rightCoefficientProducer := 6964)
    (owner := owner) (leftOwner := SemanticResult51177.owner)
    (rightOwner := SemanticResult6965.owner)
    (leftResult := 51177) (rightResult := 6965)
    (leftBinding := 51178) (rightBinding := 51179)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13166⟩) (rightExpression := ⟨103⟩)
    (leftActual := SemanticResult51177.actual selector witness)
    (rightActual := SemanticResult6965.actual selector witness)
    (leftRaw := SemanticResult51177.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨103⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound6964.actual selector witness)
    (survivorMagnitude := LeftBound51181.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult51177.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6965.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6964.derived selector witness)
  · exact LeftBound51181.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult51183

namespace SemanticResult51191
def owner : Owner := ⟨.program ⟨214⟩, ⟨13168⟩⟩
def rawTerms : List Term := Proof.Events199.exact51191RawTerms
def summary : Bound := (.finite 48256)
def resultEvent : Nat := 51191
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51191.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨58, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge51189.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge51189.frameStart)
    (owner := owner) (leftOwner := SemanticResult51183.owner)
    (rightOwner := SemanticResult2364.owner)
    (leftResult := 51183) (rightResult := 2364)
    (leftActual := SemanticResult51183.actual selector witness)
    (rightActual := SemanticResult2364.actual selector witness)
    (leftRaw := SemanticResult51183.rawTerms)
    (rightRaw := SemanticResult2364.rawTerms)
    (working := LeftOperatorMerge51189.working)
    (leftBinding := 51184) (rightBinding := 51185)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13167⟩) (rightExpression := ⟨10245⟩)
    (coefficientTransfer := 51186) (summaryTransfer := 51188)
    (rightCoefficientProducer := 2363)
    (rightSummaryTransfer := 51187)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨58, by decide⟩)
    (rightRecordedMaximum := 58)
    (rightSummaryMaximum := ⟨58, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge51189.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority2363.actual selector witness)
    (summaryMagnitude := LeftBound51188.actual selector witness)
    (reconstruction := LeftOperatorMerge51189.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult51183.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2364.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2363.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority2363.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge51189.operationAgreement
  · exact LeftBound51188.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge51189.working summary) := by
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
end SemanticResult51191

namespace SemanticResult51196
def owner : Owner := ⟨.program ⟨214⟩, ⟨10246⟩⟩
def rawTerms : List Term := Proof.Events199.exact51196RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51196
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51196.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge51195.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge51195.frameStart)
    (transferEvent := 51194) (owner := owner)
    (leftResult := 2364) (rightResult := 50670)
    (working := LeftOperatorMerge51195.working)
    (reconstruction := LeftOperatorMerge51195.reconstruction)
    (leftReference := .predecessor 0 51192 .coefficient) (rightReference := .predecessor 1 51193 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2364.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge51195.operationAgreement
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
end SemanticResult51196

namespace SemanticResult51201
def owner : Owner := ⟨.program ⟨214⟩, ⟨7263⟩⟩
def rawTerms : List Term := Proof.Events200.exact51201RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51201
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51201.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge51200.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge51200.frameStart)
    (transferEvent := 51199) (owner := owner)
    (leftResult := 50540) (rightResult := 7014)
    (working := LeftOperatorMerge51200.working)
    (reconstruction := LeftOperatorMerge51200.reconstruction)
    (leftReference := .predecessor 0 51197 .coefficient) (rightReference := .predecessor 1 51198 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7014.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge51200.operationAgreement
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
end SemanticResult51201

namespace SemanticResult51205
def owner : Owner := ⟨.program ⟨214⟩, ⟨10247⟩⟩
def rawTerms : List Term := Proof.Events200.exact51205RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51205
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51205.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 51202) (rightBinding := 51203)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7263⟩) (rightExpression := ⟨10246⟩)
    (transferEvent := 51204)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult51201.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult51196.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult51205

namespace SemanticResult51211
def owner : Owner := ⟨.program ⟨214⟩, ⟨10248⟩⟩
def rawTerms : List Term := Proof.Events200.exact51211RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 51211
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51211.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 51208) (survivorTransfer := 51209)
    (survivorEvent := 51210) (resultEvent := resultEvent)
    (rightCoefficientProducer := 7005)
    (owner := owner) (leftOwner := SemanticResult51205.owner)
    (rightOwner := SemanticResult7006.owner)
    (leftResult := 51205) (rightResult := 7006)
    (leftBinding := 51206) (rightBinding := 51207)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10247⟩) (rightExpression := ⟨83⟩)
    (leftActual := SemanticResult51205.actual selector witness)
    (rightActual := SemanticResult7006.actual selector witness)
    (leftRaw := SemanticResult51205.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨83⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound7005.actual selector witness)
    (survivorMagnitude := LeftBound51209.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult51205.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7006.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7005.derived selector witness)
  · exact LeftBound51209.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult51211

namespace SemanticResult51221
def owner : Owner := ⟨.program ⟨214⟩, ⟨10249⟩⟩
def rawTerms : List Term := Proof.Events200.exact51221RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 51221
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51221.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge51217.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge51217.frameStart)
    (owner := owner) (leftOwner := SemanticResult51211.owner)
    (rightOwner := SemanticResult7003.owner)
    (leftResult := 51211) (rightResult := 7003)
    (leftActual := SemanticResult51211.actual selector witness)
    (rightActual := SemanticResult7003.actual selector witness)
    (leftRaw := SemanticResult51211.rawTerms)
    (rightRaw := SemanticResult7003.rawTerms)
    (working := LeftOperatorMerge51217.working)
    (leftBinding := 51212) (rightBinding := 51213)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10248⟩) (rightExpression := ⟨7880⟩)
    (coefficientTransfer := 51214) (summaryTransfer := 51216)
    (rightCoefficientProducer := 7002)
    (rightSummaryTransfer := 51215)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge51217.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound7002.actual selector witness)
    (summaryMagnitude := LeftBound51216.actual selector witness)
    (reconstruction := LeftOperatorMerge51217.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult51211.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7003.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7002.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound7002.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge51217.operationAgreement
  · exact LeftBound51216.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge51217.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 51218 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10245⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge51217.working
    [{ coefficient := (-1), key := LeftRelationMerge51218.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge51218.frameStart
      LeftRelationMerge51218.owner (.relation 51218) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge51218.deltas
    rows := LeftRelationMerge51218.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge51217.working LeftRelationMerge51218.source
        (relationContext LeftRelationMerge51218.source
          LeftRelationMerge51218.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge51217.working, LeftRelationMerge51218.deltas,
    LeftRelationMerge51218.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 51218)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10249⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10245⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge51217.working) (working := relationWorking0)
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
end SemanticResult51221

namespace SemanticResult51227
def owner : Owner := ⟨.program ⟨214⟩, ⟨13169⟩⟩
def rawTerms : List Term := Proof.Events200.exact51227RawTerms
def summary : Bound := (.finite 95468672)
def resultEvent : Nat := 51227
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51227.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge51225.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult51221.owner)
    (rightOwner := SemanticResult51191.owner)
    (leftResult := 51221) (rightResult := 51191)
    (leftActual := SemanticResult51221.actual selector witness)
    (rightActual := SemanticResult51191.actual selector witness)
    (leftRaw := SemanticResult51221.rawTerms)
    (rightRaw := SemanticResult51191.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 48256) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 51222) (rightBinding := 51223)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10249⟩) (rightExpression := ⟨13168⟩)
    (coefficientTransfer := 51224) (summaryTransfer := 51226)
    (base := LeftOperatorMerge51225.base)
    (reconstruction := LeftOperatorMerge51225.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult51221.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult51191.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge51225.operationAgreement
  · rfl
  · decide
end SemanticResult51227

namespace SemanticResult51237
def owner : Owner := ⟨.program ⟨214⟩, ⟨25687⟩⟩
def rawTerms : List Term := Proof.Events200.exact51237RawTerms
def summary : Bound := (.finite 350371553738752)
def resultEvent : Nat := 51237
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51237.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95468672, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge51233.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge51233.frameStart)
    (owner := owner) (leftOwner := SemanticResult51227.owner)
    (rightOwner := SemanticResult51163.owner)
    (leftResult := 51227) (rightResult := 51163)
    (leftActual := SemanticResult51227.actual selector witness)
    (rightActual := SemanticResult51163.actual selector witness)
    (leftRaw := SemanticResult51227.rawTerms)
    (rightRaw := SemanticResult51163.rawTerms)
    (working := LeftOperatorMerge51233.working)
    (leftBinding := 51228) (rightBinding := 51229)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13169⟩) (rightExpression := ⟨25686⟩)
    (coefficientTransfer := 51230) (summaryTransfer := 51232)
    (rightCoefficientProducer := 51162)
    (rightSummaryTransfer := 51231)
    (leftMaximum := ⟨95468672, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge51233.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority51162.actual selector witness)
    (summaryMagnitude := LeftBound51232.actual selector witness)
    (reconstruction := LeftOperatorMerge51233.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult51227.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult51163.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51162.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority51162.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge51233.operationAgreement
  · exact LeftBound51232.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge51233.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 51234 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25686⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10245⟩⟩, ⟨.program ⟨214⟩, ⟨13162⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23376⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23376⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge51233.working
    [{ coefficient := (-1), key := LeftRelationMerge51234.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge51234.frameStart
      LeftRelationMerge51234.owner (.relation 51234) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge51234.deltas
    rows := LeftRelationMerge51234.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge51233.working LeftRelationMerge51234.source
        (relationContext LeftRelationMerge51234.source
          LeftRelationMerge51234.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge51233.working, LeftRelationMerge51234.deltas,
    LeftRelationMerge51234.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 51234)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25687⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10245⟩⟩, ⟨.program ⟨214⟩, ⟨13162⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25686⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25686⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge51233.working) (working := relationWorking0)
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
end SemanticResult51237

namespace SemanticResult51240
def owner : Owner := ⟨.program ⟨214⟩, ⟨20180⟩⟩
def rawTerms : List Term := Proof.Events200.exact51240RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51240
def producerEvent : Nat := 51239
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51240.actual selector witness
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
end SemanticResult51240

namespace SemanticResult51244
def owner : Owner := ⟨.program ⟨214⟩, ⟨20182⟩⟩
def rawTerms : List Term := Proof.Events200.exact51244RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51244
def producerEvent : Nat := 51243
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51244.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 51241 .coefficient) (.value (.predecessor 1 51242 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 51241 .coefficient) (.value (.predecessor 1 51242 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult51244

namespace SemanticResult51322
def owner : Owner := ⟨.program ⟨214⟩, ⟨13162⟩⟩
def rawTerms : List Term := Proof.Events200.exact51322RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51322
def producerEvent : Nat := 51321
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51322.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 51299, .finite 58, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult51322

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
