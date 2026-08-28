import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard183
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard008
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard077
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard182

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult23874
def owner : Owner := ⟨.program ⟨214⟩, ⟨9836⟩⟩
def rawTerms : List Term := Proof.Events093.exact23874RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23874
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23874.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23873.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge23873.frameStart)
    (transferEvent := 23872) (owner := owner)
    (leftResult := 960) (rightResult := 21420)
    (working := LeftOperatorMerge23873.working)
    (reconstruction := LeftOperatorMerge23873.reconstruction)
    (leftReference := .predecessor 0 23870 .coefficient) (rightReference := .predecessor 1 23871 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult960.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge23873.operationAgreement
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
end SemanticResult23874

namespace SemanticResult23879
def owner : Owner := ⟨.program ⟨214⟩, ⟨7335⟩⟩
def rawTerms : List Term := Proof.Events093.exact23879RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23879
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23879.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23878.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge23878.frameStart)
    (transferEvent := 23877) (owner := owner)
    (leftResult := 21290) (rightResult := 9018)
    (working := LeftOperatorMerge23878.working)
    (reconstruction := LeftOperatorMerge23878.reconstruction)
    (leftReference := .predecessor 0 23875 .coefficient) (rightReference := .predecessor 1 23876 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9018.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge23878.operationAgreement
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
end SemanticResult23879

namespace SemanticResult23883
def owner : Owner := ⟨.program ⟨214⟩, ⟨9837⟩⟩
def rawTerms : List Term := Proof.Events093.exact23883RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23883
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23883.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 23880) (rightBinding := 23881)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7335⟩) (rightExpression := ⟨9836⟩)
    (transferEvent := 23882)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23879.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23874.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult23883

namespace SemanticResult23889
def owner : Owner := ⟨.program ⟨214⟩, ⟨9838⟩⟩
def rawTerms : List Term := Proof.Events093.exact23889RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 23889
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23889.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 23886) (survivorTransfer := 23887)
    (survivorEvent := 23888) (resultEvent := resultEvent)
    (rightCoefficientProducer := 9009)
    (owner := owner) (leftOwner := SemanticResult23883.owner)
    (rightOwner := SemanticResult9010.owner)
    (leftResult := 23883) (rightResult := 9010)
    (leftBinding := 23884) (rightBinding := 23885)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9837⟩) (rightExpression := ⟨79⟩)
    (leftActual := SemanticResult23883.actual selector witness)
    (rightActual := SemanticResult9010.actual selector witness)
    (leftRaw := SemanticResult23883.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨79⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound9009.actual selector witness)
    (survivorMagnitude := LeftBound23887.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23883.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9010.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9009.derived selector witness)
  · exact LeftBound23887.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult23889

namespace SemanticResult23899
def owner : Owner := ⟨.program ⟨214⟩, ⟨9839⟩⟩
def rawTerms : List Term := Proof.Events093.exact23899RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 23899
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23899.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23895.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge23895.frameStart)
    (owner := owner) (leftOwner := SemanticResult23889.owner)
    (rightOwner := SemanticResult9007.owner)
    (leftResult := 23889) (rightResult := 9007)
    (leftActual := SemanticResult23889.actual selector witness)
    (rightActual := SemanticResult9007.actual selector witness)
    (leftRaw := SemanticResult23889.rawTerms)
    (rightRaw := SemanticResult9007.rawTerms)
    (working := LeftOperatorMerge23895.working)
    (leftBinding := 23890) (rightBinding := 23891)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9838⟩) (rightExpression := ⟨7868⟩)
    (coefficientTransfer := 23892) (summaryTransfer := 23894)
    (rightCoefficientProducer := 9006)
    (rightSummaryTransfer := 23893)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge23895.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound9006.actual selector witness)
    (summaryMagnitude := LeftBound23894.actual selector witness)
    (reconstruction := LeftOperatorMerge23895.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23889.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9007.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9006.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound9006.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge23895.operationAgreement
  · exact LeftBound23894.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23895.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 23896 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6785⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge23895.working
    [{ coefficient := (-1), key := LeftRelationMerge23896.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge23896.frameStart
      LeftRelationMerge23896.owner (.relation 23896) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge23896.deltas
    rows := LeftRelationMerge23896.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge23895.working LeftRelationMerge23896.source
        (relationContext LeftRelationMerge23896.source
          LeftRelationMerge23896.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge23895.working, LeftRelationMerge23896.deltas,
    LeftRelationMerge23896.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 23896)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9839⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge23895.working) (working := relationWorking0)
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
end SemanticResult23899

namespace SemanticResult23905
def owner : Owner := ⟨.program ⟨214⟩, ⟨12401⟩⟩
def rawTerms : List Term := Proof.Events093.exact23905RawTerms
def summary : Bound := (.finite 95453696)
def resultEvent : Nat := 23905
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23905.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge23903.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult23899.owner)
    (rightOwner := SemanticResult23869.owner)
    (leftResult := 23899) (rightResult := 23869)
    (leftActual := SemanticResult23899.actual selector witness)
    (rightActual := SemanticResult23869.actual selector witness)
    (leftRaw := SemanticResult23899.rawTerms)
    (rightRaw := SemanticResult23869.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 33280) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 23900) (rightBinding := 23901)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9839⟩) (rightExpression := ⟨12400⟩)
    (coefficientTransfer := 23902) (summaryTransfer := 23904)
    (base := LeftOperatorMerge23903.base)
    (reconstruction := LeftOperatorMerge23903.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23899.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23869.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge23903.operationAgreement
  · rfl
  · decide
end SemanticResult23905

namespace SemanticResult23915
def owner : Owner := ⟨.program ⟨214⟩, ⟨25389⟩⟩
def rawTerms : List Term := Proof.Events093.exact23915RawTerms
def summary : Bound := (.finite 350316591579136)
def resultEvent : Nat := 23915
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23915.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95453696, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23911.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge23911.frameStart)
    (owner := owner) (leftOwner := SemanticResult23905.owner)
    (rightOwner := SemanticResult23841.owner)
    (leftResult := 23905) (rightResult := 23841)
    (leftActual := SemanticResult23905.actual selector witness)
    (rightActual := SemanticResult23841.actual selector witness)
    (leftRaw := SemanticResult23905.rawTerms)
    (rightRaw := SemanticResult23841.rawTerms)
    (working := LeftOperatorMerge23911.working)
    (leftBinding := 23906) (rightBinding := 23907)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12401⟩) (rightExpression := ⟨25388⟩)
    (coefficientTransfer := 23908) (summaryTransfer := 23910)
    (rightCoefficientProducer := 23840)
    (rightSummaryTransfer := 23909)
    (leftMaximum := ⟨95453696, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge23911.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority23840.actual selector witness)
    (summaryMagnitude := LeftBound23910.actual selector witness)
    (reconstruction := LeftOperatorMerge23911.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23905.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23841.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23840.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority23840.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge23911.operationAgreement
  · exact LeftBound23910.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23911.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 23912 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23212⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23212⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge23911.working
    [{ coefficient := (-1), key := LeftRelationMerge23912.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge23912.frameStart
      LeftRelationMerge23912.owner (.relation 23912) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge23912.deltas
    rows := LeftRelationMerge23912.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge23911.working LeftRelationMerge23912.source
        (relationContext LeftRelationMerge23912.source
          LeftRelationMerge23912.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge23911.working, LeftRelationMerge23912.deltas,
    LeftRelationMerge23912.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 23912)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25389⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9835⟩⟩, ⟨.program ⟨214⟩, ⟨12394⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25388⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge23911.working) (working := relationWorking0)
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
end SemanticResult23915

namespace SemanticResult23918
def owner : Owner := ⟨.program ⟨214⟩, ⟨19900⟩⟩
def rawTerms : List Term := Proof.Events093.exact23918RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23918
def producerEvent : Nat := 23917
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23918.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨20⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨20⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult23918

namespace SemanticResult23922
def owner : Owner := ⟨.program ⟨214⟩, ⟨19902⟩⟩
def rawTerms : List Term := Proof.Events093.exact23922RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23922
def producerEvent : Nat := 23921
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23922.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 23919 .coefficient) (.value (.predecessor 1 23920 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 23919 .coefficient) (.value (.predecessor 1 23920 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult23922

namespace SemanticResult24000
def owner : Owner := ⟨.program ⟨214⟩, ⟨12394⟩⟩
def rawTerms : List Term := Proof.Events093.exact24000RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24000
def producerEvent : Nat := 23999
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24000.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 23977, .finite 40, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult24000

namespace SemanticResult24003
def owner : Owner := ⟨.program ⟨214⟩, ⟨9835⟩⟩
def rawTerms : List Term := Proof.Events093.exact24003RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24003
def producerEvent : Nat := 24002
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24003.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 23977, .finite 40, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult24003

namespace SemanticResult24008
def owner : Owner := ⟨.program ⟨214⟩, ⟨12395⟩⟩
def rawTerms : List Term := Proof.Events093.exact24008RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24008
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24008.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge24007.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge24007.frameStart)
    (transferEvent := 24006) (owner := owner)
    (leftResult := 24003) (rightResult := 24000)
    (working := LeftOperatorMerge24007.working)
    (reconstruction := LeftOperatorMerge24007.reconstruction)
    (leftReference := .predecessor 0 24004 .coefficient) (rightReference := .predecessor 1 24005 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult24003.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult24000.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge24007.operationAgreement
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
end SemanticResult24008

namespace SemanticResult24019
def owner : Owner := ⟨.program ⟨214⟩, ⟨23212⟩⟩
def rawTerms : List Term := Proof.Events093.exact24019RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24019
def producerEvent : Nat := 24018
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24019.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 23977, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult24019

namespace SemanticResult24022
def owner : Owner := ⟨.program ⟨214⟩, ⟨25388⟩⟩
def rawTerms : List Term := Proof.Events093.exact24022RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24022
def producerEvent : Nat := 24021
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24022.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 23977, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult24022

namespace SemanticResult24031
def owner : Owner := ⟨.program ⟨214⟩, ⟨12479⟩⟩
def rawTerms : List Term := Proof.Events093.exact24031RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24031
def producerEvent : Nat := 24030
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24031.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 24029 .coefficient), 23977, .finite 1600, .identity (.predecessor 0 24029 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult24031

namespace SemanticResult24033
def owner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rawTerms : List Term := Proof.Events093.exact24033RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 24033
def producerEvent : Nat := 24032
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult24033.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 23977, .large, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult24033

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
