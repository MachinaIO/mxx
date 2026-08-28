import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard488
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard026
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard081
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard465

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult68198
def owner : Owner := ⟨.program ⟨214⟩, ⟨25214⟩⟩
def rawTerms : List Term := Proof.Events266.exact68198RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 68198
def producerEvent : Nat := 68197
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68198.actual selector witness
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
end SemanticResult68198

namespace SemanticResult68203
def owner : Owner := ⟨.program ⟨214⟩, ⟨11952⟩⟩
def rawTerms : List Term := Proof.Events266.exact68203RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 68203
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68203.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge68202.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge68202.frameStart)
    (transferEvent := 68201) (owner := owner)
    (leftResult := 3224) (rightResult := 65295)
    (working := LeftOperatorMerge68202.working)
    (reconstruction := LeftOperatorMerge68202.reconstruction)
    (leftReference := .predecessor 0 68199 .coefficient) (rightReference := .predecessor 1 68200 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3224.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge68202.operationAgreement
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
end SemanticResult68203

namespace SemanticResult68208
def owner : Owner := ⟨.program ⟨214⟩, ⟨7202⟩⟩
def rawTerms : List Term := Proof.Events266.exact68208RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 68208
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68208.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge68207.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge68207.frameStart)
    (transferEvent := 68206) (owner := owner)
    (leftResult := 65165) (rightResult := 9478)
    (working := LeftOperatorMerge68207.working)
    (reconstruction := LeftOperatorMerge68207.reconstruction)
    (leftReference := .predecessor 0 68204 .coefficient) (rightReference := .predecessor 1 68205 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9478.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge68207.operationAgreement
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
end SemanticResult68208

namespace SemanticResult68212
def owner : Owner := ⟨.program ⟨214⟩, ⟨11953⟩⟩
def rawTerms : List Term := Proof.Events266.exact68212RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 68212
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68212.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 68209) (rightBinding := 68210)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7202⟩) (rightExpression := ⟨11952⟩)
    (transferEvent := 68211)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult68208.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult68203.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult68212

namespace SemanticResult68218
def owner : Owner := ⟨.program ⟨214⟩, ⟨11954⟩⟩
def rawTerms : List Term := Proof.Events266.exact68218RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 68218
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68218.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 68215) (survivorTransfer := 68216)
    (survivorEvent := 68217) (resultEvent := resultEvent)
    (rightCoefficientProducer := 9469)
    (owner := owner) (leftOwner := SemanticResult68212.owner)
    (rightOwner := SemanticResult9470.owner)
    (leftResult := 68212) (rightResult := 9470)
    (leftBinding := 68213) (rightBinding := 68214)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11953⟩) (rightExpression := ⟨98⟩)
    (leftActual := SemanticResult68212.actual selector witness)
    (rightActual := SemanticResult9470.actual selector witness)
    (leftRaw := SemanticResult68212.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨98⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound9469.actual selector witness)
    (survivorMagnitude := LeftBound68216.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult68212.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9470.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9469.derived selector witness)
  · exact LeftBound68216.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult68218

namespace SemanticResult68226
def owner : Owner := ⟨.program ⟨214⟩, ⟨11955⟩⟩
def rawTerms : List Term := Proof.Events266.exact68226RawTerms
def summary : Bound := (.finite 29952)
def resultEvent : Nat := 68226
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68226.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨36, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge68224.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge68224.frameStart)
    (owner := owner) (leftOwner := SemanticResult68218.owner)
    (rightOwner := SemanticResult3227.owner)
    (leftResult := 68218) (rightResult := 3227)
    (leftActual := SemanticResult68218.actual selector witness)
    (rightActual := SemanticResult3227.actual selector witness)
    (leftRaw := SemanticResult68218.rawTerms)
    (rightRaw := SemanticResult3227.rawTerms)
    (working := LeftOperatorMerge68224.working)
    (leftBinding := 68219) (rightBinding := 68220)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11954⟩) (rightExpression := ⟨9710⟩)
    (coefficientTransfer := 68221) (summaryTransfer := 68223)
    (rightCoefficientProducer := 3226)
    (rightSummaryTransfer := 68222)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨36, by decide⟩)
    (rightRecordedMaximum := 36)
    (rightSummaryMaximum := ⟨36, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge68224.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority3226.actual selector witness)
    (summaryMagnitude := LeftBound68223.actual selector witness)
    (reconstruction := LeftOperatorMerge68224.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult68218.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3227.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3226.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority3226.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge68224.operationAgreement
  · exact LeftBound68223.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge68224.working summary) := by
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
end SemanticResult68226

namespace SemanticResult68231
def owner : Owner := ⟨.program ⟨214⟩, ⟨9711⟩⟩
def rawTerms : List Term := Proof.Events266.exact68231RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 68231
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68231.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge68230.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge68230.frameStart)
    (transferEvent := 68229) (owner := owner)
    (leftResult := 3227) (rightResult := 65295)
    (working := LeftOperatorMerge68230.working)
    (reconstruction := LeftOperatorMerge68230.reconstruction)
    (leftReference := .predecessor 0 68227 .coefficient) (rightReference := .predecessor 1 68228 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3227.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge68230.operationAgreement
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
end SemanticResult68231

namespace SemanticResult68236
def owner : Owner := ⟨.program ⟨214⟩, ⟨7182⟩⟩
def rawTerms : List Term := Proof.Events266.exact68236RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 68236
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68236.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge68235.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge68235.frameStart)
    (transferEvent := 68234) (owner := owner)
    (leftResult := 65165) (rightResult := 9519)
    (working := LeftOperatorMerge68235.working)
    (reconstruction := LeftOperatorMerge68235.reconstruction)
    (leftReference := .predecessor 0 68232 .coefficient) (rightReference := .predecessor 1 68233 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9519.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge68235.operationAgreement
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
end SemanticResult68236

namespace SemanticResult68240
def owner : Owner := ⟨.program ⟨214⟩, ⟨9712⟩⟩
def rawTerms : List Term := Proof.Events266.exact68240RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 68240
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68240.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 68237) (rightBinding := 68238)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7182⟩) (rightExpression := ⟨9711⟩)
    (transferEvent := 68239)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult68236.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult68231.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult68240

namespace SemanticResult68246
def owner : Owner := ⟨.program ⟨214⟩, ⟨9713⟩⟩
def rawTerms : List Term := Proof.Events266.exact68246RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 68246
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68246.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 68243) (survivorTransfer := 68244)
    (survivorEvent := 68245) (resultEvent := resultEvent)
    (rightCoefficientProducer := 9510)
    (owner := owner) (leftOwner := SemanticResult68240.owner)
    (rightOwner := SemanticResult9511.owner)
    (leftResult := 68240) (rightResult := 9511)
    (leftBinding := 68241) (rightBinding := 68242)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9712⟩) (rightExpression := ⟨78⟩)
    (leftActual := SemanticResult68240.actual selector witness)
    (rightActual := SemanticResult9511.actual selector witness)
    (leftRaw := SemanticResult68240.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨78⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound9510.actual selector witness)
    (survivorMagnitude := LeftBound68244.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult68240.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9511.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9510.derived selector witness)
  · exact LeftBound68244.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult68246

namespace SemanticResult68256
def owner : Owner := ⟨.program ⟨214⟩, ⟨9714⟩⟩
def rawTerms : List Term := Proof.Events266.exact68256RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 68256
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68256.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge68252.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge68252.frameStart)
    (owner := owner) (leftOwner := SemanticResult68246.owner)
    (rightOwner := SemanticResult9508.owner)
    (leftResult := 68246) (rightResult := 9508)
    (leftActual := SemanticResult68246.actual selector witness)
    (rightActual := SemanticResult9508.actual selector witness)
    (leftRaw := SemanticResult68246.rawTerms)
    (rightRaw := SemanticResult9508.rawTerms)
    (working := LeftOperatorMerge68252.working)
    (leftBinding := 68247) (rightBinding := 68248)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9713⟩) (rightExpression := ⟨7865⟩)
    (coefficientTransfer := 68249) (summaryTransfer := 68251)
    (rightCoefficientProducer := 9507)
    (rightSummaryTransfer := 68250)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge68252.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound9507.actual selector witness)
    (summaryMagnitude := LeftBound68251.actual selector witness)
    (reconstruction := LeftOperatorMerge68252.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult68246.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9508.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9507.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound9507.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge68252.operationAgreement
  · exact LeftBound68251.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge68252.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 68253 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge68252.working
    [{ coefficient := (-1), key := LeftRelationMerge68253.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge68253.frameStart
      LeftRelationMerge68253.owner (.relation 68253) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge68253.deltas
    rows := LeftRelationMerge68253.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge68252.working LeftRelationMerge68253.source
        (relationContext LeftRelationMerge68253.source
          LeftRelationMerge68253.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge68252.working, LeftRelationMerge68253.deltas,
    LeftRelationMerge68253.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 68253)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9714⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge68252.working) (working := relationWorking0)
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
end SemanticResult68256

namespace SemanticResult68262
def owner : Owner := ⟨.program ⟨214⟩, ⟨11956⟩⟩
def rawTerms : List Term := Proof.Events266.exact68262RawTerms
def summary : Bound := (.finite 95450368)
def resultEvent : Nat := 68262
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68262.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge68260.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult68256.owner)
    (rightOwner := SemanticResult68226.owner)
    (leftResult := 68256) (rightResult := 68226)
    (leftActual := SemanticResult68256.actual selector witness)
    (rightActual := SemanticResult68226.actual selector witness)
    (leftRaw := SemanticResult68256.rawTerms)
    (rightRaw := SemanticResult68226.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 29952) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 68257) (rightBinding := 68258)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9714⟩) (rightExpression := ⟨11955⟩)
    (coefficientTransfer := 68259) (summaryTransfer := 68261)
    (base := LeftOperatorMerge68260.base)
    (reconstruction := LeftOperatorMerge68260.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult68256.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult68226.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge68260.operationAgreement
  · rfl
  · decide
end SemanticResult68262

namespace SemanticResult68272
def owner : Owner := ⟨.program ⟨214⟩, ⟨25215⟩⟩
def rawTerms : List Term := Proof.Events266.exact68272RawTerms
def summary : Bound := (.finite 350304377765888)
def resultEvent : Nat := 68272
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68272.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95450368, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge68268.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge68268.frameStart)
    (owner := owner) (leftOwner := SemanticResult68262.owner)
    (rightOwner := SemanticResult68198.owner)
    (leftResult := 68262) (rightResult := 68198)
    (leftActual := SemanticResult68262.actual selector witness)
    (rightActual := SemanticResult68198.actual selector witness)
    (leftRaw := SemanticResult68262.rawTerms)
    (rightRaw := SemanticResult68198.rawTerms)
    (working := LeftOperatorMerge68268.working)
    (leftBinding := 68263) (rightBinding := 68264)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11956⟩) (rightExpression := ⟨25214⟩)
    (coefficientTransfer := 68265) (summaryTransfer := 68267)
    (rightCoefficientProducer := 68197)
    (rightSummaryTransfer := 68266)
    (leftMaximum := ⟨95450368, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge68268.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority68197.actual selector witness)
    (summaryMagnitude := LeftBound68267.actual selector witness)
    (reconstruction := LeftOperatorMerge68268.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult68262.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult68198.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68197.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority68197.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge68268.operationAgreement
  · exact LeftBound68267.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge68268.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 68269 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23120⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23120⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge68268.working
    [{ coefficient := (-1), key := LeftRelationMerge68269.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge68269.frameStart
      LeftRelationMerge68269.owner (.relation 68269) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge68269.deltas
    rows := LeftRelationMerge68269.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge68268.working LeftRelationMerge68269.source
        (relationContext LeftRelationMerge68269.source
          LeftRelationMerge68269.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge68268.working, LeftRelationMerge68269.deltas,
    LeftRelationMerge68269.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 68269)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25215⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9710⟩⟩, ⟨.program ⟨214⟩, ⟨11949⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25214⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge68268.working) (working := relationWorking0)
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
end SemanticResult68272

namespace SemanticResult68275
def owner : Owner := ⟨.program ⟨214⟩, ⟨19812⟩⟩
def rawTerms : List Term := Proof.Events266.exact68275RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 68275
def producerEvent : Nat := 68274
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68275.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨19⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨19⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult68275

namespace SemanticResult68279
def owner : Owner := ⟨.program ⟨214⟩, ⟨19814⟩⟩
def rawTerms : List Term := Proof.Events266.exact68279RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 68279
def producerEvent : Nat := 68278
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68279.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 68276 .coefficient) (.value (.predecessor 1 68277 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 68276 .coefficient) (.value (.predecessor 1 68277 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult68279

namespace SemanticResult68357
def owner : Owner := ⟨.program ⟨214⟩, ⟨11949⟩⟩
def rawTerms : List Term := Proof.Events267.exact68357RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 68357
def producerEvent : Nat := 68356
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult68357.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 68334, .finite 36, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult68357

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
