import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard201
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard008
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard097
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard098
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard164

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult26248
def owner : Owner := ⟨.program ⟨214⟩, ⟨23590⟩⟩
def rawTerms : List Term := Proof.Events102.exact26248RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 26248
def producerEvent : Nat := 26247
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26248.actual selector witness
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
end SemanticResult26248

namespace SemanticResult26251
def owner : Owner := ⟨.program ⟨214⟩, ⟨26081⟩⟩
def rawTerms : List Term := Proof.Events102.exact26251RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 26251
def producerEvent : Nat := 26250
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26251.actual selector witness
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
end SemanticResult26251

namespace SemanticResult26256
def owner : Owner := ⟨.program ⟨214⟩, ⟨11482⟩⟩
def rawTerms : List Term := Proof.Events102.exact26256RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 26256
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26256.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge26255.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge26255.frameStart)
    (transferEvent := 26254) (owner := owner)
    (leftResult := 1072) (rightResult := 21420)
    (working := LeftOperatorMerge26255.working)
    (reconstruction := LeftOperatorMerge26255.reconstruction)
    (leftReference := .predecessor 0 26252 .coefficient) (rightReference := .predecessor 1 26253 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1072.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge26255.operationAgreement
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
end SemanticResult26256

namespace SemanticResult26261
def owner : Owner := ⟨.program ⟨214⟩, ⟨7349⟩⟩
def rawTerms : List Term := Proof.Events102.exact26261RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 26261
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26261.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge26260.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge26260.frameStart)
    (transferEvent := 26259) (owner := owner)
    (leftResult := 21290) (rightResult := 11482)
    (working := LeftOperatorMerge26260.working)
    (reconstruction := LeftOperatorMerge26260.reconstruction)
    (leftReference := .predecessor 0 26257 .coefficient) (rightReference := .predecessor 1 26258 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11482.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge26260.operationAgreement
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
end SemanticResult26261

namespace SemanticResult26265
def owner : Owner := ⟨.program ⟨214⟩, ⟨11483⟩⟩
def rawTerms : List Term := Proof.Events102.exact26265RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 26265
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26265.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 26262) (rightBinding := 26263)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7349⟩) (rightExpression := ⟨11482⟩)
    (transferEvent := 26264)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult26261.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult26256.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult26265

namespace SemanticResult26271
def owner : Owner := ⟨.program ⟨214⟩, ⟨11484⟩⟩
def rawTerms : List Term := Proof.Events102.exact26271RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 26271
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26271.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 26268) (survivorTransfer := 26269)
    (survivorEvent := 26270) (resultEvent := resultEvent)
    (rightCoefficientProducer := 11473)
    (owner := owner) (leftOwner := SemanticResult26265.owner)
    (rightOwner := SemanticResult11474.owner)
    (leftResult := 26265) (rightResult := 11474)
    (leftBinding := 26266) (rightBinding := 26267)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11483⟩) (rightExpression := ⟨93⟩)
    (leftActual := SemanticResult26265.actual selector witness)
    (rightActual := SemanticResult11474.actual selector witness)
    (leftRaw := SemanticResult26265.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨93⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound11473.actual selector witness)
    (survivorMagnitude := LeftBound26269.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult26265.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11474.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)
  · exact LeftBound26269.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult26271

namespace SemanticResult26279
def owner : Owner := ⟨.program ⟨214⟩, ⟨14237⟩⟩
def rawTerms : List Term := Proof.Events102.exact26279RawTerms
def summary : Bound := (.finite 14976)
def resultEvent : Nat := 26279
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26279.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨18, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge26277.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge26277.frameStart)
    (owner := owner) (leftOwner := SemanticResult26271.owner)
    (rightOwner := SemanticResult1075.owner)
    (leftResult := 26271) (rightResult := 1075)
    (leftActual := SemanticResult26271.actual selector witness)
    (rightActual := SemanticResult1075.actual selector witness)
    (leftRaw := SemanticResult26271.rawTerms)
    (rightRaw := SemanticResult1075.rawTerms)
    (working := LeftOperatorMerge26277.working)
    (leftBinding := 26272) (rightBinding := 26273)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11484⟩) (rightExpression := ⟨14234⟩)
    (coefficientTransfer := 26274) (summaryTransfer := 26276)
    (rightCoefficientProducer := 1074)
    (rightSummaryTransfer := 26275)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨18, by decide⟩)
    (rightRecordedMaximum := 18)
    (rightSummaryMaximum := ⟨18, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge26277.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1074.actual selector witness)
    (summaryMagnitude := LeftBound26276.actual selector witness)
    (reconstruction := LeftOperatorMerge26277.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult26271.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1075.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1074.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1074.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge26277.operationAgreement
  · exact LeftBound26276.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge26277.working summary) := by
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
end SemanticResult26279

namespace SemanticResult26284
def owner : Owner := ⟨.program ⟨214⟩, ⟨14238⟩⟩
def rawTerms : List Term := Proof.Events102.exact26284RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 26284
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26284.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge26283.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge26283.frameStart)
    (transferEvent := 26282) (owner := owner)
    (leftResult := 1075) (rightResult := 21420)
    (working := LeftOperatorMerge26283.working)
    (reconstruction := LeftOperatorMerge26283.reconstruction)
    (leftReference := .predecessor 0 26280 .coefficient) (rightReference := .predecessor 1 26281 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1075.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge26283.operationAgreement
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
end SemanticResult26284

namespace SemanticResult26289
def owner : Owner := ⟨.program ⟨214⟩, ⟨7329⟩⟩
def rawTerms : List Term := Proof.Events102.exact26289RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 26289
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26289.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge26288.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge26288.frameStart)
    (transferEvent := 26287) (owner := owner)
    (leftResult := 21290) (rightResult := 11523)
    (working := LeftOperatorMerge26288.working)
    (reconstruction := LeftOperatorMerge26288.reconstruction)
    (leftReference := .predecessor 0 26285 .coefficient) (rightReference := .predecessor 1 26286 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11523.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge26288.operationAgreement
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
end SemanticResult26289

namespace SemanticResult26293
def owner : Owner := ⟨.program ⟨214⟩, ⟨14239⟩⟩
def rawTerms : List Term := Proof.Events102.exact26293RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 26293
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26293.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 26290) (rightBinding := 26291)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7329⟩) (rightExpression := ⟨14238⟩)
    (transferEvent := 26292)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult26289.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult26284.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult26293

namespace SemanticResult26299
def owner : Owner := ⟨.program ⟨214⟩, ⟨14240⟩⟩
def rawTerms : List Term := Proof.Events102.exact26299RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 26299
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26299.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 26296) (survivorTransfer := 26297)
    (survivorEvent := 26298) (resultEvent := resultEvent)
    (rightCoefficientProducer := 11514)
    (owner := owner) (leftOwner := SemanticResult26293.owner)
    (rightOwner := SemanticResult11515.owner)
    (leftResult := 26293) (rightResult := 11515)
    (leftBinding := 26294) (rightBinding := 26295)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14239⟩) (rightExpression := ⟨73⟩)
    (leftActual := SemanticResult26293.actual selector witness)
    (rightActual := SemanticResult11515.actual selector witness)
    (leftRaw := SemanticResult26293.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨73⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound11514.actual selector witness)
    (survivorMagnitude := LeftBound26297.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult26293.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11515.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11514.derived selector witness)
  · exact LeftBound26297.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult26299

namespace SemanticResult26309
def owner : Owner := ⟨.program ⟨214⟩, ⟨14241⟩⟩
def rawTerms : List Term := Proof.Events102.exact26309RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 26309
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26309.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge26305.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge26305.frameStart)
    (owner := owner) (leftOwner := SemanticResult26299.owner)
    (rightOwner := SemanticResult11512.owner)
    (leftResult := 26299) (rightResult := 11512)
    (leftActual := SemanticResult26299.actual selector witness)
    (rightActual := SemanticResult11512.actual selector witness)
    (leftRaw := SemanticResult26299.rawTerms)
    (rightRaw := SemanticResult11512.rawTerms)
    (working := LeftOperatorMerge26305.working)
    (leftBinding := 26300) (rightBinding := 26301)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14240⟩) (rightExpression := ⟨7853⟩)
    (coefficientTransfer := 26302) (summaryTransfer := 26304)
    (rightCoefficientProducer := 11511)
    (rightSummaryTransfer := 26303)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge26305.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound11511.actual selector witness)
    (summaryMagnitude := LeftBound26304.actual selector witness)
    (reconstruction := LeftOperatorMerge26305.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult26299.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11512.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11511.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound11511.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge26305.operationAgreement
  · exact LeftBound26304.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge26305.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 26306 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6779⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6779⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge26305.working
    [{ coefficient := (-1), key := LeftRelationMerge26306.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge26306.frameStart
      LeftRelationMerge26306.owner (.relation 26306) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge26306.deltas
    rows := LeftRelationMerge26306.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge26305.working LeftRelationMerge26306.source
        (relationContext LeftRelationMerge26306.source
          LeftRelationMerge26306.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge26305.working, LeftRelationMerge26306.deltas,
    LeftRelationMerge26306.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 26306)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨14241⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge26305.working) (working := relationWorking0)
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
end SemanticResult26309

namespace SemanticResult26315
def owner : Owner := ⟨.program ⟨214⟩, ⟨14242⟩⟩
def rawTerms : List Term := Proof.Events102.exact26315RawTerms
def summary : Bound := (.finite 95435392)
def resultEvent : Nat := 26315
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26315.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge26313.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult26309.owner)
    (rightOwner := SemanticResult26279.owner)
    (leftResult := 26309) (rightResult := 26279)
    (leftActual := SemanticResult26309.actual selector witness)
    (rightActual := SemanticResult26279.actual selector witness)
    (leftRaw := SemanticResult26309.rawTerms)
    (rightRaw := SemanticResult26279.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 14976) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 26310) (rightBinding := 26311)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14241⟩) (rightExpression := ⟨14237⟩)
    (coefficientTransfer := 26312) (summaryTransfer := 26314)
    (base := LeftOperatorMerge26313.base)
    (reconstruction := LeftOperatorMerge26313.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult26309.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult26279.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge26313.operationAgreement
  · rfl
  · decide
end SemanticResult26315

namespace SemanticResult26325
def owner : Owner := ⟨.program ⟨214⟩, ⟨26082⟩⟩
def rawTerms : List Term := Proof.Events102.exact26325RawTerms
def summary : Bound := (.finite 350249415606272)
def resultEvent : Nat := 26325
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26325.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95435392, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge26321.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge26321.frameStart)
    (owner := owner) (leftOwner := SemanticResult26315.owner)
    (rightOwner := SemanticResult26251.owner)
    (leftResult := 26315) (rightResult := 26251)
    (leftActual := SemanticResult26315.actual selector witness)
    (rightActual := SemanticResult26251.actual selector witness)
    (leftRaw := SemanticResult26315.rawTerms)
    (rightRaw := SemanticResult26251.rawTerms)
    (working := LeftOperatorMerge26321.working)
    (leftBinding := 26316) (rightBinding := 26317)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14242⟩) (rightExpression := ⟨26081⟩)
    (coefficientTransfer := 26318) (summaryTransfer := 26320)
    (rightCoefficientProducer := 26250)
    (rightSummaryTransfer := 26319)
    (leftMaximum := ⟨95435392, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge26321.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority26250.actual selector witness)
    (summaryMagnitude := LeftBound26320.actual selector witness)
    (reconstruction := LeftOperatorMerge26321.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult26315.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult26251.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26250.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority26250.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge26321.operationAgreement
  · exact LeftBound26320.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge26321.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 26322 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23590⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23590⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge26321.working
    [{ coefficient := (-1), key := LeftRelationMerge26322.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge26322.frameStart
      LeftRelationMerge26322.owner (.relation 26322) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge26322.deltas
    rows := LeftRelationMerge26322.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge26321.working LeftRelationMerge26322.source
        (relationContext LeftRelationMerge26322.source
          LeftRelationMerge26322.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge26321.working, LeftRelationMerge26322.deltas,
    LeftRelationMerge26322.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 26322)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨26082⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge26321.working) (working := relationWorking0)
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
end SemanticResult26325

namespace SemanticResult26328
def owner : Owner := ⟨.program ⟨214⟩, ⟨19540⟩⟩
def rawTerms : List Term := Proof.Events102.exact26328RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 26328
def producerEvent : Nat := 26327
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26328.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨15⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨15⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult26328

namespace SemanticResult26332
def owner : Owner := ⟨.program ⟨214⟩, ⟨19542⟩⟩
def rawTerms : List Term := Proof.Events102.exact26332RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 26332
def producerEvent : Nat := 26331
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult26332.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 26329 .coefficient) (.value (.predecessor 1 26330 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 26329 .coefficient) (.value (.predecessor 1 26330 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult26332

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
