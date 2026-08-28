import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard305
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard014
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard101
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard102
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard265
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard303
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard304

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult41331
def owner : Owner := ⟨.program ⟨214⟩, ⟨21411⟩⟩
def rawTerms : List Term := Proof.Events161.exact41331RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 41331
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41331.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41166.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge41166.frameStart)
    (owner := owner) (leftOwner := SemanticResult36137.owner)
    (rightOwner := SemanticResult41160.owner)
    (leftResult := 36137) (rightResult := 41160)
    (leftActual := SemanticResult36137.actual selector witness)
    (rightActual := SemanticResult41160.actual selector witness)
    (leftRaw := SemanticResult36137.rawTerms)
    (rightRaw := SemanticResult41160.rawTerms)
    (working := LeftOperatorMerge41166.working)
    (leftBinding := 41161) (rightBinding := 41162)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5553⟩) (rightExpression := ⟨21410⟩)
    (coefficientTransfer := 41163) (summaryTransfer := 41165)
    (rightCoefficientProducer := 41159)
    (rightSummaryTransfer := 41164)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge41166.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound41159.actual selector witness)
    (summaryMagnitude := LeftBound41165.actual selector witness)
    (reconstruction := LeftOperatorMerge41166.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36137.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41160.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41159.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound41159.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge41166.operationAgreement
  · exact LeftBound41165.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41166.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 41326 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27892⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15948⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24168⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27892⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15948⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24168⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge41166.working
    [{ coefficient := (1), key := LeftRelationMerge41326.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge41326.frameStart
      LeftRelationMerge41326.owner (.relation 41326) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge41326.deltas
    rows := LeftRelationMerge41326.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge41166.working LeftRelationMerge41326.source
        (relationContext LeftRelationMerge41326.source
          LeftRelationMerge41326.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge41166.working, LeftRelationMerge41326.deltas,
    LeftRelationMerge41326.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 41326)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨21411⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21408⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21408⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge41166.working) (working := relationWorking0)
    (reconstruction := relationReconstruction0)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt0 selector selectorLower selectorUpper
  · rfl
  · rfl
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
end SemanticResult41331

namespace SemanticResult41338
def owner : Owner := ⟨.program ⟨214⟩, ⟨27895⟩⟩
def rawTerms : List Term := Proof.Events161.exact41338RawTerms
def summary : Bound := (.finite 1292068473939586330624)
def resultEvent : Nat := 41338
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41338.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge41335.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult41331.owner)
    (rightOwner := SemanticResult41153.owner)
    (leftResult := 41331) (rightResult := 41153)
    (leftActual := SemanticResult41331.actual selector witness)
    (rightActual := SemanticResult41153.actual selector witness)
    (leftRaw := SemanticResult41331.rawTerms)
    (rightRaw := SemanticResult41153.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292068472128282820608) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 41332) (rightBinding := 41333)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21411⟩) (rightExpression := ⟨27894⟩)
    (coefficientTransfer := 41334) (summaryTransfer := 41337)
    (base := LeftOperatorMerge41335.base)
    (reconstruction := LeftOperatorMerge41335.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41331.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41153.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge41335.operationAgreement
  · rfl
  · decide
end SemanticResult41338

namespace SemanticResult41345
def owner : Owner := ⟨.program ⟨214⟩, ⟨24105⟩⟩
def rawTerms : List Term := Proof.Events161.exact41345RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41345
def producerEvent : Nat := 41344
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41345.actual selector witness
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
end SemanticResult41345

namespace SemanticResult41348
def owner : Owner := ⟨.program ⟨214⟩, ⟨27675⟩⟩
def rawTerms : List Term := Proof.Events161.exact41348RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41348
def producerEvent : Nat := 41347
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41348.actual selector witness
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
end SemanticResult41348

namespace SemanticResult41355
def owner : Owner := ⟨.program ⟨214⟩, ⟨23546⟩⟩
def rawTerms : List Term := Proof.Events161.exact41355RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41355
def producerEvent : Nat := 41354
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41355.actual selector witness
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
end SemanticResult41355

namespace SemanticResult41358
def owner : Owner := ⟨.program ⟨214⟩, ⟨25999⟩⟩
def rawTerms : List Term := Proof.Events161.exact41358RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41358
def producerEvent : Nat := 41357
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41358.actual selector witness
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
end SemanticResult41358

namespace SemanticResult41363
def owner : Owner := ⟨.program ⟨214⟩, ⟨11394⟩⟩
def rawTerms : List Term := Proof.Events161.exact41363RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41363
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41363.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41362.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge41362.frameStart)
    (transferEvent := 41361) (owner := owner)
    (leftResult := 1843) (rightResult := 36045)
    (working := LeftOperatorMerge41362.working)
    (reconstruction := LeftOperatorMerge41362.reconstruction)
    (leftReference := .predecessor 0 41359 .coefficient) (rightReference := .predecessor 1 41360 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1843.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge41362.operationAgreement
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
end SemanticResult41363

namespace SemanticResult41368
def owner : Owner := ⟨.program ⟨214⟩, ⟨7310⟩⟩
def rawTerms : List Term := Proof.Events161.exact41368RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41368
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41368.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41367.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge41367.frameStart)
    (transferEvent := 41366) (owner := owner)
    (leftResult := 35915) (rightResult := 11983)
    (working := LeftOperatorMerge41367.working)
    (reconstruction := LeftOperatorMerge41367.reconstruction)
    (leftReference := .predecessor 0 41364 .coefficient) (rightReference := .predecessor 1 41365 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11983.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge41367.operationAgreement
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
end SemanticResult41368

namespace SemanticResult41372
def owner : Owner := ⟨.program ⟨214⟩, ⟨11395⟩⟩
def rawTerms : List Term := Proof.Events161.exact41372RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41372
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41372.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41369) (rightBinding := 41370)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7310⟩) (rightExpression := ⟨11394⟩)
    (transferEvent := 41371)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41368.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41363.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41372

namespace SemanticResult41378
def owner : Owner := ⟨.program ⟨214⟩, ⟨11396⟩⟩
def rawTerms : List Term := Proof.Events161.exact41378RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 41378
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41378.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 41375) (survivorTransfer := 41376)
    (survivorEvent := 41377) (resultEvent := resultEvent)
    (rightCoefficientProducer := 11974)
    (owner := owner) (leftOwner := SemanticResult41372.owner)
    (rightOwner := SemanticResult11975.owner)
    (leftResult := 41372) (rightResult := 11975)
    (leftBinding := 41373) (rightBinding := 41374)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11395⟩) (rightExpression := ⟨92⟩)
    (leftActual := SemanticResult41372.actual selector witness)
    (rightActual := SemanticResult11975.actual selector witness)
    (leftRaw := SemanticResult41372.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨92⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound11974.actual selector witness)
    (survivorMagnitude := LeftBound41376.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41372.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11975.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11974.derived selector witness)
  · exact LeftBound41376.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult41378

namespace SemanticResult41386
def owner : Owner := ⟨.program ⟨214⟩, ⟨14011⟩⟩
def rawTerms : List Term := Proof.Events161.exact41386RawTerms
def summary : Bound := (.finite 13312)
def resultEvent : Nat := 41386
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41386.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨16, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41384.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge41384.frameStart)
    (owner := owner) (leftOwner := SemanticResult41378.owner)
    (rightOwner := SemanticResult1846.owner)
    (leftResult := 41378) (rightResult := 1846)
    (leftActual := SemanticResult41378.actual selector witness)
    (rightActual := SemanticResult1846.actual selector witness)
    (leftRaw := SemanticResult41378.rawTerms)
    (rightRaw := SemanticResult1846.rawTerms)
    (working := LeftOperatorMerge41384.working)
    (leftBinding := 41379) (rightBinding := 41380)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11396⟩) (rightExpression := ⟨14008⟩)
    (coefficientTransfer := 41381) (summaryTransfer := 41383)
    (rightCoefficientProducer := 1845)
    (rightSummaryTransfer := 41382)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨16, by decide⟩)
    (rightRecordedMaximum := 16)
    (rightSummaryMaximum := ⟨16, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge41384.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1845.actual selector witness)
    (summaryMagnitude := LeftBound41383.actual selector witness)
    (reconstruction := LeftOperatorMerge41384.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41378.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1846.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1845.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1845.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge41384.operationAgreement
  · exact LeftBound41383.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41384.working summary) := by
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
end SemanticResult41386

namespace SemanticResult41391
def owner : Owner := ⟨.program ⟨214⟩, ⟨14012⟩⟩
def rawTerms : List Term := Proof.Events161.exact41391RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41391
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41391.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41390.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge41390.frameStart)
    (transferEvent := 41389) (owner := owner)
    (leftResult := 1846) (rightResult := 36045)
    (working := LeftOperatorMerge41390.working)
    (reconstruction := LeftOperatorMerge41390.reconstruction)
    (leftReference := .predecessor 0 41387 .coefficient) (rightReference := .predecessor 1 41388 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1846.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge41390.operationAgreement
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
end SemanticResult41391

namespace SemanticResult41396
def owner : Owner := ⟨.program ⟨214⟩, ⟨7290⟩⟩
def rawTerms : List Term := Proof.Events161.exact41396RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41396
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41396.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41395.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge41395.frameStart)
    (transferEvent := 41394) (owner := owner)
    (leftResult := 35915) (rightResult := 12024)
    (working := LeftOperatorMerge41395.working)
    (reconstruction := LeftOperatorMerge41395.reconstruction)
    (leftReference := .predecessor 0 41392 .coefficient) (rightReference := .predecessor 1 41393 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12024.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge41395.operationAgreement
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
end SemanticResult41396

namespace SemanticResult41400
def owner : Owner := ⟨.program ⟨214⟩, ⟨14013⟩⟩
def rawTerms : List Term := Proof.Events161.exact41400RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41400
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41400.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41397) (rightBinding := 41398)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7290⟩) (rightExpression := ⟨14012⟩)
    (transferEvent := 41399)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41396.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41391.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41400

namespace SemanticResult41406
def owner : Owner := ⟨.program ⟨214⟩, ⟨14014⟩⟩
def rawTerms : List Term := Proof.Events161.exact41406RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 41406
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41406.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 41403) (survivorTransfer := 41404)
    (survivorEvent := 41405) (resultEvent := resultEvent)
    (rightCoefficientProducer := 12015)
    (owner := owner) (leftOwner := SemanticResult41400.owner)
    (rightOwner := SemanticResult12016.owner)
    (leftResult := 41400) (rightResult := 12016)
    (leftBinding := 41401) (rightBinding := 41402)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14013⟩) (rightExpression := ⟨72⟩)
    (leftActual := SemanticResult41400.actual selector witness)
    (rightActual := SemanticResult12016.actual selector witness)
    (leftRaw := SemanticResult41400.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨72⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound12015.actual selector witness)
    (survivorMagnitude := LeftBound41404.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41400.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12016.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12015.derived selector witness)
  · exact LeftBound41404.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult41406

namespace SemanticResult41416
def owner : Owner := ⟨.program ⟨214⟩, ⟨14015⟩⟩
def rawTerms : List Term := Proof.Events161.exact41416RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 41416
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41416.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41412.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge41412.frameStart)
    (owner := owner) (leftOwner := SemanticResult41406.owner)
    (rightOwner := SemanticResult12013.owner)
    (leftResult := 41406) (rightResult := 12013)
    (leftActual := SemanticResult41406.actual selector witness)
    (rightActual := SemanticResult12013.actual selector witness)
    (leftRaw := SemanticResult41406.rawTerms)
    (rightRaw := SemanticResult12013.rawTerms)
    (working := LeftOperatorMerge41412.working)
    (leftBinding := 41407) (rightBinding := 41408)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14014⟩) (rightExpression := ⟨7850⟩)
    (coefficientTransfer := 41409) (summaryTransfer := 41411)
    (rightCoefficientProducer := 12012)
    (rightSummaryTransfer := 41410)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge41412.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound12012.actual selector witness)
    (summaryMagnitude := LeftBound41411.actual selector witness)
    (reconstruction := LeftOperatorMerge41412.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41406.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12013.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12012.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound12012.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge41412.operationAgreement
  · exact LeftBound41411.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41412.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 41413 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6758⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6778⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge41412.working
    [{ coefficient := (-1), key := LeftRelationMerge41413.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge41413.frameStart
      LeftRelationMerge41413.owner (.relation 41413) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge41413.deltas
    rows := LeftRelationMerge41413.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge41412.working LeftRelationMerge41413.source
        (relationContext LeftRelationMerge41413.source
          LeftRelationMerge41413.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge41412.working, LeftRelationMerge41413.deltas,
    LeftRelationMerge41413.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 41413)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨14015⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14008⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge41412.working) (working := relationWorking0)
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
end SemanticResult41416

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
