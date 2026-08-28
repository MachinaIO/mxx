import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard473
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard065
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard471
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard472

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult66250
def owner : Owner := ⟨.program ⟨214⟩, ⟨29809⟩⟩
def rawTerms : List Term := Proof.Events258.exact66250RawTerms
def summary : Bound := (.finite 1292516722839998050304)
def resultEvent : Nat := 66250
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66250.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge66247.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult66243.owner)
    (rightOwner := SemanticResult66065.owner)
    (leftResult := 66243) (rightResult := 66065)
    (leftActual := SemanticResult66243.actual selector witness)
    (rightActual := SemanticResult66065.actual selector witness)
    (leftRaw := SemanticResult66243.rawTerms)
    (rightRaw := SemanticResult66065.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292516721028694540288) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 66244) (rightBinding := 66245)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22695⟩) (rightExpression := ⟨29808⟩)
    (coefficientTransfer := 66246) (summaryTransfer := 66249)
    (base := LeftOperatorMerge66247.base)
    (reconstruction := LeftOperatorMerge66247.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult66243.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult66065.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge66247.operationAgreement
  · rfl
  · decide
end SemanticResult66250

namespace SemanticResult66257
def owner : Owner := ⟨.program ⟨214⟩, ⟨24663⟩⟩
def rawTerms : List Term := Proof.Events258.exact66257RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 66257
def producerEvent : Nat := 66256
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66257.actual selector witness
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
end SemanticResult66257

namespace SemanticResult66260
def owner : Owner := ⟨.program ⟨214⟩, ⟨29589⟩⟩
def rawTerms : List Term := Proof.Events258.exact66260RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 66260
def producerEvent : Nat := 66259
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66260.actual selector witness
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
end SemanticResult66260

namespace SemanticResult66267
def owner : Owner := ⟨.program ⟨214⟩, ⟨23330⟩⟩
def rawTerms : List Term := Proof.Events258.exact66267RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 66267
def producerEvent : Nat := 66266
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66267.actual selector witness
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
end SemanticResult66267

namespace SemanticResult66270
def owner : Owner := ⟨.program ⟨214⟩, ⟨25599⟩⟩
def rawTerms : List Term := Proof.Events258.exact66270RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 66270
def producerEvent : Nat := 66269
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66270.actual selector witness
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
end SemanticResult66270

namespace SemanticResult66275
def owner : Owner := ⟨.program ⟨214⟩, ⟨12953⟩⟩
def rawTerms : List Term := Proof.Events258.exact66275RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 66275
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66275.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge66274.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge66274.frameStart)
    (transferEvent := 66273) (owner := owner)
    (leftResult := 3132) (rightResult := 65295)
    (working := LeftOperatorMerge66274.working)
    (reconstruction := LeftOperatorMerge66274.reconstruction)
    (leftReference := .predecessor 0 66271 .coefficient) (rightReference := .predecessor 1 66272 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3132.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge66274.operationAgreement
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
end SemanticResult66275

namespace SemanticResult66280
def owner : Owner := ⟨.program ⟨214⟩, ⟨7206⟩⟩
def rawTerms : List Term := Proof.Events258.exact66280RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 66280
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66280.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge66279.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge66279.frameStart)
    (transferEvent := 66278) (owner := owner)
    (leftResult := 65165) (rightResult := 7474)
    (working := LeftOperatorMerge66279.working)
    (reconstruction := LeftOperatorMerge66279.reconstruction)
    (leftReference := .predecessor 0 66276 .coefficient) (rightReference := .predecessor 1 66277 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7474.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge66279.operationAgreement
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
end SemanticResult66280

namespace SemanticResult66284
def owner : Owner := ⟨.program ⟨214⟩, ⟨12954⟩⟩
def rawTerms : List Term := Proof.Events258.exact66284RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 66284
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66284.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 66281) (rightBinding := 66282)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7206⟩) (rightExpression := ⟨12953⟩)
    (transferEvent := 66283)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult66280.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult66275.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult66284

namespace SemanticResult66290
def owner : Owner := ⟨.program ⟨214⟩, ⟨12955⟩⟩
def rawTerms : List Term := Proof.Events258.exact66290RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 66290
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66290.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 66287) (survivorTransfer := 66288)
    (survivorEvent := 66289) (resultEvent := resultEvent)
    (rightCoefficientProducer := 7465)
    (owner := owner) (leftOwner := SemanticResult66284.owner)
    (rightOwner := SemanticResult7466.owner)
    (leftResult := 66284) (rightResult := 7466)
    (leftBinding := 66285) (rightBinding := 66286)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12954⟩) (rightExpression := ⟨102⟩)
    (leftActual := SemanticResult66284.actual selector witness)
    (rightActual := SemanticResult7466.actual selector witness)
    (leftRaw := SemanticResult66284.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨102⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound7465.actual selector witness)
    (survivorMagnitude := LeftBound66288.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult66284.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7466.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7465.derived selector witness)
  · exact LeftBound66288.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult66290

namespace SemanticResult66298
def owner : Owner := ⟨.program ⟨214⟩, ⟨12956⟩⟩
def rawTerms : List Term := Proof.Events258.exact66298RawTerms
def summary : Bound := (.finite 43264)
def resultEvent : Nat := 66298
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66298.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨52, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge66296.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge66296.frameStart)
    (owner := owner) (leftOwner := SemanticResult66290.owner)
    (rightOwner := SemanticResult3135.owner)
    (leftResult := 66290) (rightResult := 3135)
    (leftActual := SemanticResult66290.actual selector witness)
    (rightActual := SemanticResult3135.actual selector witness)
    (leftRaw := SemanticResult66290.rawTerms)
    (rightRaw := SemanticResult3135.rawTerms)
    (working := LeftOperatorMerge66296.working)
    (leftBinding := 66291) (rightBinding := 66292)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12955⟩) (rightExpression := ⟨10130⟩)
    (coefficientTransfer := 66293) (summaryTransfer := 66295)
    (rightCoefficientProducer := 3134)
    (rightSummaryTransfer := 66294)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨52, by decide⟩)
    (rightRecordedMaximum := 52)
    (rightSummaryMaximum := ⟨52, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge66296.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority3134.actual selector witness)
    (summaryMagnitude := LeftBound66295.actual selector witness)
    (reconstruction := LeftOperatorMerge66296.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult66290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3135.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3134.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority3134.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge66296.operationAgreement
  · exact LeftBound66295.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge66296.working summary) := by
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
end SemanticResult66298

namespace SemanticResult66303
def owner : Owner := ⟨.program ⟨214⟩, ⟨10131⟩⟩
def rawTerms : List Term := Proof.Events258.exact66303RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 66303
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66303.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge66302.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge66302.frameStart)
    (transferEvent := 66301) (owner := owner)
    (leftResult := 3135) (rightResult := 65295)
    (working := LeftOperatorMerge66302.working)
    (reconstruction := LeftOperatorMerge66302.reconstruction)
    (leftReference := .predecessor 0 66299 .coefficient) (rightReference := .predecessor 1 66300 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3135.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge66302.operationAgreement
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
end SemanticResult66303

namespace SemanticResult66308
def owner : Owner := ⟨.program ⟨214⟩, ⟨7186⟩⟩
def rawTerms : List Term := Proof.Events259.exact66308RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 66308
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66308.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge66307.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge66307.frameStart)
    (transferEvent := 66306) (owner := owner)
    (leftResult := 65165) (rightResult := 7515)
    (working := LeftOperatorMerge66307.working)
    (reconstruction := LeftOperatorMerge66307.reconstruction)
    (leftReference := .predecessor 0 66304 .coefficient) (rightReference := .predecessor 1 66305 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7515.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge66307.operationAgreement
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
end SemanticResult66308

namespace SemanticResult66312
def owner : Owner := ⟨.program ⟨214⟩, ⟨10132⟩⟩
def rawTerms : List Term := Proof.Events259.exact66312RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 66312
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66312.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 66309) (rightBinding := 66310)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7186⟩) (rightExpression := ⟨10131⟩)
    (transferEvent := 66311)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult66308.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult66303.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult66312

namespace SemanticResult66318
def owner : Owner := ⟨.program ⟨214⟩, ⟨10133⟩⟩
def rawTerms : List Term := Proof.Events259.exact66318RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 66318
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66318.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 66315) (survivorTransfer := 66316)
    (survivorEvent := 66317) (resultEvent := resultEvent)
    (rightCoefficientProducer := 7506)
    (owner := owner) (leftOwner := SemanticResult66312.owner)
    (rightOwner := SemanticResult7507.owner)
    (leftResult := 66312) (rightResult := 7507)
    (leftBinding := 66313) (rightBinding := 66314)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10132⟩) (rightExpression := ⟨82⟩)
    (leftActual := SemanticResult66312.actual selector witness)
    (rightActual := SemanticResult7507.actual selector witness)
    (leftRaw := SemanticResult66312.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨82⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound7506.actual selector witness)
    (survivorMagnitude := LeftBound66316.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult66312.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7507.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7506.derived selector witness)
  · exact LeftBound66316.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult66318

namespace SemanticResult66328
def owner : Owner := ⟨.program ⟨214⟩, ⟨10134⟩⟩
def rawTerms : List Term := Proof.Events259.exact66328RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 66328
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66328.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge66324.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge66324.frameStart)
    (owner := owner) (leftOwner := SemanticResult66318.owner)
    (rightOwner := SemanticResult7504.owner)
    (leftResult := 66318) (rightResult := 7504)
    (leftActual := SemanticResult66318.actual selector witness)
    (rightActual := SemanticResult7504.actual selector witness)
    (leftRaw := SemanticResult66318.rawTerms)
    (rightRaw := SemanticResult7504.rawTerms)
    (working := LeftOperatorMerge66324.working)
    (leftBinding := 66319) (rightBinding := 66320)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10133⟩) (rightExpression := ⟨7877⟩)
    (coefficientTransfer := 66321) (summaryTransfer := 66323)
    (rightCoefficientProducer := 7503)
    (rightSummaryTransfer := 66322)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge66324.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound7503.actual selector witness)
    (summaryMagnitude := LeftBound66323.actual selector witness)
    (reconstruction := LeftOperatorMerge66324.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult66318.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7504.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7503.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound7503.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge66324.operationAgreement
  · exact LeftBound66323.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge66324.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 66325 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10130⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge66324.working
    [{ coefficient := (-1), key := LeftRelationMerge66325.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge66325.frameStart
      LeftRelationMerge66325.owner (.relation 66325) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge66325.deltas
    rows := LeftRelationMerge66325.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge66324.working LeftRelationMerge66325.source
        (relationContext LeftRelationMerge66325.source
          LeftRelationMerge66325.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge66324.working, LeftRelationMerge66325.deltas,
    LeftRelationMerge66325.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 66325)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10134⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10130⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge66324.working) (working := relationWorking0)
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
end SemanticResult66328

namespace SemanticResult66334
def owner : Owner := ⟨.program ⟨214⟩, ⟨12957⟩⟩
def rawTerms : List Term := Proof.Events259.exact66334RawTerms
def summary : Bound := (.finite 95463680)
def resultEvent : Nat := 66334
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult66334.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge66332.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult66328.owner)
    (rightOwner := SemanticResult66298.owner)
    (leftResult := 66328) (rightResult := 66298)
    (leftActual := SemanticResult66328.actual selector witness)
    (rightActual := SemanticResult66298.actual selector witness)
    (leftRaw := SemanticResult66328.rawTerms)
    (rightRaw := SemanticResult66298.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 43264) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 66329) (rightBinding := 66330)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10134⟩) (rightExpression := ⟨12956⟩)
    (coefficientTransfer := 66331) (summaryTransfer := 66333)
    (base := LeftOperatorMerge66332.base)
    (reconstruction := LeftOperatorMerge66332.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult66328.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult66298.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge66332.operationAgreement
  · rfl
  · decide
end SemanticResult66334

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
