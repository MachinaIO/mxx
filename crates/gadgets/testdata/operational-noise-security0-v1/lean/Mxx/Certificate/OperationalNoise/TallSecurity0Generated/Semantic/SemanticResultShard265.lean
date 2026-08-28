import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard265
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard013
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard057
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard264

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult36078
def owner : Owner := ⟨.program ⟨214⟩, ⟨10356⟩⟩
def rawTerms : List Term := Proof.Events140.exact36078RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36078
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36078.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36077.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge36077.frameStart)
    (transferEvent := 36076) (owner := owner)
    (leftResult := 1593) (rightResult := 36045)
    (working := LeftOperatorMerge36077.working)
    (reconstruction := LeftOperatorMerge36077.reconstruction)
    (leftReference := .predecessor 0 36074 .coefficient) (rightReference := .predecessor 1 36075 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1593.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge36077.operationAgreement
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
end SemanticResult36078

namespace SemanticResult36083
def owner : Owner := ⟨.program ⟨214⟩, ⟨7302⟩⟩
def rawTerms : List Term := Proof.Events140.exact36083RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36083
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36083.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36082.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge36082.frameStart)
    (transferEvent := 36081) (owner := owner)
    (leftResult := 35915) (rightResult := 6498)
    (working := LeftOperatorMerge36082.working)
    (reconstruction := LeftOperatorMerge36082.reconstruction)
    (leftReference := .predecessor 0 36079 .coefficient) (rightReference := .predecessor 1 36080 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6498.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge36082.operationAgreement
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
end SemanticResult36083

namespace SemanticResult36087
def owner : Owner := ⟨.program ⟨214⟩, ⟨10357⟩⟩
def rawTerms : List Term := Proof.Events140.exact36087RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36087
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36087.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 36084) (rightBinding := 36085)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7302⟩) (rightExpression := ⟨10356⟩)
    (transferEvent := 36086)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36083.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36078.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult36087

namespace SemanticResult36093
def owner : Owner := ⟨.program ⟨214⟩, ⟨10358⟩⟩
def rawTerms : List Term := Proof.Events140.exact36093RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 36093
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36093.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 36090) (survivorTransfer := 36091)
    (survivorEvent := 36092) (resultEvent := resultEvent)
    (rightCoefficientProducer := 6489)
    (owner := owner) (leftOwner := SemanticResult36087.owner)
    (rightOwner := SemanticResult6490.owner)
    (leftResult := 36087) (rightResult := 6490)
    (leftBinding := 36088) (rightBinding := 36089)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10357⟩) (rightExpression := ⟨84⟩)
    (leftActual := SemanticResult36087.actual selector witness)
    (rightActual := SemanticResult6490.actual selector witness)
    (leftRaw := SemanticResult36087.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨84⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound6489.actual selector witness)
    (survivorMagnitude := LeftBound36091.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36087.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6490.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6489.derived selector witness)
  · exact LeftBound36091.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult36093

namespace SemanticResult36103
def owner : Owner := ⟨.program ⟨214⟩, ⟨10359⟩⟩
def rawTerms : List Term := Proof.Events141.exact36103RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 36103
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36103.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36099.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge36099.frameStart)
    (owner := owner) (leftOwner := SemanticResult36093.owner)
    (rightOwner := SemanticResult6487.owner)
    (leftResult := 36093) (rightResult := 6487)
    (leftActual := SemanticResult36093.actual selector witness)
    (rightActual := SemanticResult6487.actual selector witness)
    (leftRaw := SemanticResult36093.rawTerms)
    (rightRaw := SemanticResult6487.rawTerms)
    (working := LeftOperatorMerge36099.working)
    (leftBinding := 36094) (rightBinding := 36095)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10358⟩) (rightExpression := ⟨7883⟩)
    (coefficientTransfer := 36096) (summaryTransfer := 36098)
    (rightCoefficientProducer := 6486)
    (rightSummaryTransfer := 36097)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge36099.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound6486.actual selector witness)
    (summaryMagnitude := LeftBound36098.actual selector witness)
    (reconstruction := LeftOperatorMerge36099.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36093.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6487.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6486.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound6486.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge36099.operationAgreement
  · exact LeftBound36098.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36099.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 36100 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10355⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge36099.working
    [{ coefficient := (-1), key := LeftRelationMerge36100.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge36100.frameStart
      LeftRelationMerge36100.owner (.relation 36100) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge36100.deltas
    rows := LeftRelationMerge36100.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge36099.working LeftRelationMerge36100.source
        (relationContext LeftRelationMerge36100.source
          LeftRelationMerge36100.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge36099.working, LeftRelationMerge36100.deltas,
    LeftRelationMerge36100.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 36100)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10359⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10355⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge36099.working) (working := relationWorking0)
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
end SemanticResult36103

namespace SemanticResult36109
def owner : Owner := ⟨.program ⟨214⟩, ⟨13373⟩⟩
def rawTerms : List Term := Proof.Events141.exact36109RawTerms
def summary : Bound := (.finite 95470336)
def resultEvent : Nat := 36109
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36109.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge36107.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult36103.owner)
    (rightOwner := SemanticResult36073.owner)
    (leftResult := 36103) (rightResult := 36073)
    (leftActual := SemanticResult36103.actual selector witness)
    (rightActual := SemanticResult36073.actual selector witness)
    (leftRaw := SemanticResult36103.rawTerms)
    (rightRaw := SemanticResult36073.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 49920) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 36104) (rightBinding := 36105)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10359⟩) (rightExpression := ⟨13372⟩)
    (coefficientTransfer := 36106) (summaryTransfer := 36108)
    (base := LeftOperatorMerge36107.base)
    (reconstruction := LeftOperatorMerge36107.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36103.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36073.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge36107.operationAgreement
  · rfl
  · decide
end SemanticResult36109

namespace SemanticResult36119
def owner : Owner := ⟨.program ⟨214⟩, ⟨25769⟩⟩
def rawTerms : List Term := Proof.Events141.exact36119RawTerms
def summary : Bound := (.finite 350377660645376)
def resultEvent : Nat := 36119
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36119.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95470336, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36115.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge36115.frameStart)
    (owner := owner) (leftOwner := SemanticResult36109.owner)
    (rightOwner := SemanticResult36040.owner)
    (leftResult := 36109) (rightResult := 36040)
    (leftActual := SemanticResult36109.actual selector witness)
    (rightActual := SemanticResult36040.actual selector witness)
    (leftRaw := SemanticResult36109.rawTerms)
    (rightRaw := SemanticResult36040.rawTerms)
    (working := LeftOperatorMerge36115.working)
    (leftBinding := 36110) (rightBinding := 36111)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13373⟩) (rightExpression := ⟨25768⟩)
    (coefficientTransfer := 36112) (summaryTransfer := 36114)
    (rightCoefficientProducer := 36039)
    (rightSummaryTransfer := 36113)
    (leftMaximum := ⟨95470336, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge36115.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority36039.actual selector witness)
    (summaryMagnitude := LeftBound36114.actual selector witness)
    (reconstruction := LeftOperatorMerge36115.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36109.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36040.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36039.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority36039.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge36115.operationAgreement
  · exact LeftBound36114.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36115.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 36116 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25768⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23420⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23420⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge36115.working
    [{ coefficient := (-1), key := LeftRelationMerge36116.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge36116.frameStart
      LeftRelationMerge36116.owner (.relation 36116) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge36116.deltas
    rows := LeftRelationMerge36116.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge36115.working LeftRelationMerge36116.source
        (relationContext LeftRelationMerge36116.source
          LeftRelationMerge36116.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge36115.working, LeftRelationMerge36116.deltas,
    LeftRelationMerge36116.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 36116)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25769⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25768⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25768⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge36115.working) (working := relationWorking0)
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
end SemanticResult36119

namespace SemanticResult36122
def owner : Owner := ⟨.program ⟨214⟩, ⟨20256⟩⟩
def rawTerms : List Term := Proof.Events141.exact36122RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36122
def producerEvent : Nat := 36121
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36122.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨26⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨26⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult36122

namespace SemanticResult36126
def owner : Owner := ⟨.program ⟨214⟩, ⟨20258⟩⟩
def rawTerms : List Term := Proof.Events141.exact36126RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36126
def producerEvent : Nat := 36125
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36126.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 36123 .coefficient) (.value (.predecessor 1 36124 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 36123 .coefficient) (.value (.predecessor 1 36124 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult36126

namespace SemanticResult36131
def owner : Owner := ⟨.program ⟨214⟩, ⟨5552⟩⟩
def rawTerms : List Term := Proof.Events141.exact36131RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36131
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36131.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36130.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge36130.frameStart)
    (transferEvent := 36129) (owner := owner)
    (leftResult := 35915) (rightResult := 6550)
    (working := LeftOperatorMerge36130.working)
    (reconstruction := LeftOperatorMerge36130.reconstruction)
    (leftReference := .predecessor 0 36127 .coefficient) (rightReference := .predecessor 1 36128 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6550.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge36130.operationAgreement
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
end SemanticResult36131

namespace SemanticResult36137
def owner : Owner := ⟨.program ⟨214⟩, ⟨5553⟩⟩
def rawTerms : List Term := Proof.Events141.exact36137RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 36137
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36137.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 16)
    (coefficientTransfer := 36134) (survivorTransfer := 36135)
    (survivorEvent := 36136) (resultEvent := resultEvent)
    (rightCoefficientProducer := 6547)
    (owner := owner) (leftOwner := SemanticResult36131.owner)
    (rightOwner := SemanticResult6548.owner)
    (leftResult := 36131) (rightResult := 6548)
    (leftBinding := 36132) (rightBinding := 36133)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5552⟩) (rightExpression := ⟨22⟩)
    (leftActual := SemanticResult36131.actual selector witness)
    (rightActual := SemanticResult6548.actual selector witness)
    (leftRaw := SemanticResult36131.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨22⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftAuthority6547.actual selector witness)
    (survivorMagnitude := LeftBound36135.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36131.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6548.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6547.derived selector witness)
  · exact LeftBound36135.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult36137

namespace SemanticResult36215
def owner : Owner := ⟨.program ⟨214⟩, ⟨13366⟩⟩
def rawTerms : List Term := Proof.Events141.exact36215RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36215
def producerEvent : Nat := 36214
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36215.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 36192, .finite 60, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult36215

namespace SemanticResult36218
def owner : Owner := ⟨.program ⟨214⟩, ⟨10355⟩⟩
def rawTerms : List Term := Proof.Events141.exact36218RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36218
def producerEvent : Nat := 36217
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36218.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 36192, .finite 60, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult36218

namespace SemanticResult36223
def owner : Owner := ⟨.program ⟨214⟩, ⟨13367⟩⟩
def rawTerms : List Term := Proof.Events141.exact36223RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36223
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36223.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36222.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge36222.frameStart)
    (transferEvent := 36221) (owner := owner)
    (leftResult := 36218) (rightResult := 36215)
    (working := LeftOperatorMerge36222.working)
    (reconstruction := LeftOperatorMerge36222.reconstruction)
    (leftReference := .predecessor 0 36219 .coefficient) (rightReference := .predecessor 1 36220 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult36218.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36215.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge36222.operationAgreement
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
end SemanticResult36223

namespace SemanticResult36234
def owner : Owner := ⟨.program ⟨214⟩, ⟨23420⟩⟩
def rawTerms : List Term := Proof.Events141.exact36234RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36234
def producerEvent : Nat := 36233
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36234.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 36192, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult36234

namespace SemanticResult36237
def owner : Owner := ⟨.program ⟨214⟩, ⟨25768⟩⟩
def rawTerms : List Term := Proof.Events141.exact36237RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36237
def producerEvent : Nat := 36236
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36237.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 36192, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult36237

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
