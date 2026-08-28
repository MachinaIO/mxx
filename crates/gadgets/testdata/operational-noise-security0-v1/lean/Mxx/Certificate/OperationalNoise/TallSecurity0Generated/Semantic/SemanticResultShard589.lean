import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard589
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard081
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard588

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult82849
def owner : Owner := ⟨.program ⟨214⟩, ⟨7220⟩⟩
def rawTerms : List Term := Proof.Events323.exact82849RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82849
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82849.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82848.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge82848.frameStart)
    (transferEvent := 82847) (owner := owner)
    (leftResult := 79790) (rightResult := 9519)
    (working := LeftOperatorMerge82848.working)
    (reconstruction := LeftOperatorMerge82848.reconstruction)
    (leftReference := .predecessor 0 82845 .coefficient) (rightReference := .predecessor 1 82846 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9519.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge82848.operationAgreement
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
end SemanticResult82849

namespace SemanticResult82853
def owner : Owner := ⟨.program ⟨214⟩, ⟨9717⟩⟩
def rawTerms : List Term := Proof.Events323.exact82853RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82853
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82853.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 82850) (rightBinding := 82851)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7220⟩) (rightExpression := ⟨9716⟩)
    (transferEvent := 82852)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult82849.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult82844.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult82853

namespace SemanticResult82859
def owner : Owner := ⟨.program ⟨214⟩, ⟨9718⟩⟩
def rawTerms : List Term := Proof.Events323.exact82859RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 82859
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82859.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 82856) (survivorTransfer := 82857)
    (survivorEvent := 82858) (resultEvent := resultEvent)
    (rightCoefficientProducer := 9510)
    (owner := owner) (leftOwner := SemanticResult82853.owner)
    (rightOwner := SemanticResult9511.owner)
    (leftResult := 82853) (rightResult := 9511)
    (leftBinding := 82854) (rightBinding := 82855)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9717⟩) (rightExpression := ⟨78⟩)
    (leftActual := SemanticResult82853.actual selector witness)
    (rightActual := SemanticResult9511.actual selector witness)
    (leftRaw := SemanticResult82853.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨78⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound9510.actual selector witness)
    (survivorMagnitude := LeftBound82857.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult82853.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9511.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9510.derived selector witness)
  · exact LeftBound82857.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult82859

namespace SemanticResult82869
def owner : Owner := ⟨.program ⟨214⟩, ⟨9719⟩⟩
def rawTerms : List Term := Proof.Events323.exact82869RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 82869
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82869.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82865.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge82865.frameStart)
    (owner := owner) (leftOwner := SemanticResult82859.owner)
    (rightOwner := SemanticResult9508.owner)
    (leftResult := 82859) (rightResult := 9508)
    (leftActual := SemanticResult82859.actual selector witness)
    (rightActual := SemanticResult9508.actual selector witness)
    (leftRaw := SemanticResult82859.rawTerms)
    (rightRaw := SemanticResult9508.rawTerms)
    (working := LeftOperatorMerge82865.working)
    (leftBinding := 82860) (rightBinding := 82861)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9718⟩) (rightExpression := ⟨7865⟩)
    (coefficientTransfer := 82862) (summaryTransfer := 82864)
    (rightCoefficientProducer := 9507)
    (rightSummaryTransfer := 82863)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge82865.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound9507.actual selector witness)
    (summaryMagnitude := LeftBound82864.actual selector witness)
    (reconstruction := LeftOperatorMerge82865.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult82859.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9508.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9507.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound9507.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge82865.operationAgreement
  · exact LeftBound82864.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82865.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 82866 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6784⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge82865.working
    [{ coefficient := (-1), key := LeftRelationMerge82866.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge82866.frameStart
      LeftRelationMerge82866.owner (.relation 82866) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge82866.deltas
    rows := LeftRelationMerge82866.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge82865.working LeftRelationMerge82866.source
        (relationContext LeftRelationMerge82866.source
          LeftRelationMerge82866.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge82865.working, LeftRelationMerge82866.deltas,
    LeftRelationMerge82866.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 82866)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9719⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge82865.working) (working := relationWorking0)
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
end SemanticResult82869

namespace SemanticResult82875
def owner : Owner := ⟨.program ⟨214⟩, ⟨11964⟩⟩
def rawTerms : List Term := Proof.Events323.exact82875RawTerms
def summary : Bound := (.finite 95450368)
def resultEvent : Nat := 82875
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82875.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge82873.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult82869.owner)
    (rightOwner := SemanticResult82839.owner)
    (leftResult := 82869) (rightResult := 82839)
    (leftActual := SemanticResult82869.actual selector witness)
    (rightActual := SemanticResult82839.actual selector witness)
    (leftRaw := SemanticResult82869.rawTerms)
    (rightRaw := SemanticResult82839.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 29952) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 82870) (rightBinding := 82871)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9719⟩) (rightExpression := ⟨11963⟩)
    (coefficientTransfer := 82872) (summaryTransfer := 82874)
    (base := LeftOperatorMerge82873.base)
    (reconstruction := LeftOperatorMerge82873.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult82869.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult82839.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge82873.operationAgreement
  · rfl
  · decide
end SemanticResult82875

namespace SemanticResult82885
def owner : Owner := ⟨.program ⟨214⟩, ⟨25220⟩⟩
def rawTerms : List Term := Proof.Events323.exact82885RawTerms
def summary : Bound := (.finite 350304377765888)
def resultEvent : Nat := 82885
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82885.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95450368, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82881.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge82881.frameStart)
    (owner := owner) (leftOwner := SemanticResult82875.owner)
    (rightOwner := SemanticResult82811.owner)
    (leftResult := 82875) (rightResult := 82811)
    (leftActual := SemanticResult82875.actual selector witness)
    (rightActual := SemanticResult82811.actual selector witness)
    (leftRaw := SemanticResult82875.rawTerms)
    (rightRaw := SemanticResult82811.rawTerms)
    (working := LeftOperatorMerge82881.working)
    (leftBinding := 82876) (rightBinding := 82877)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11964⟩) (rightExpression := ⟨25219⟩)
    (coefficientTransfer := 82878) (summaryTransfer := 82880)
    (rightCoefficientProducer := 82810)
    (rightSummaryTransfer := 82879)
    (leftMaximum := ⟨95450368, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge82881.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority82810.actual selector witness)
    (summaryMagnitude := LeftBound82880.actual selector witness)
    (reconstruction := LeftOperatorMerge82881.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult82875.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult82811.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82810.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority82810.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge82881.operationAgreement
  · exact LeftBound82880.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82881.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 82882 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6764⟩⟩, ⟨.program ⟨214⟩, ⟨7864⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23122⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23122⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge82881.working
    [{ coefficient := (-1), key := LeftRelationMerge82882.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge82882.frameStart
      LeftRelationMerge82882.owner (.relation 82882) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge82882.deltas
    rows := LeftRelationMerge82882.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge82881.working LeftRelationMerge82882.source
        (relationContext LeftRelationMerge82882.source
          LeftRelationMerge82882.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge82881.working, LeftRelationMerge82882.deltas,
    LeftRelationMerge82882.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 82882)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25220⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9715⟩⟩, ⟨.program ⟨214⟩, ⟨11957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge82881.working) (working := relationWorking0)
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
end SemanticResult82885

namespace SemanticResult82888
def owner : Owner := ⟨.program ⟨214⟩, ⟨19816⟩⟩
def rawTerms : List Term := Proof.Events323.exact82888RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82888
def producerEvent : Nat := 82887
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82888.actual selector witness
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
end SemanticResult82888

namespace SemanticResult82892
def owner : Owner := ⟨.program ⟨214⟩, ⟨19818⟩⟩
def rawTerms : List Term := Proof.Events323.exact82892RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82892
def producerEvent : Nat := 82891
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82892.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 82889 .coefficient) (.value (.predecessor 1 82890 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 82889 .coefficient) (.value (.predecessor 1 82890 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult82892

namespace SemanticResult82970
def owner : Owner := ⟨.program ⟨214⟩, ⟨11957⟩⟩
def rawTerms : List Term := Proof.Events324.exact82970RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82970
def producerEvent : Nat := 82969
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82970.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 82947, .finite 36, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult82970

namespace SemanticResult82973
def owner : Owner := ⟨.program ⟨214⟩, ⟨9715⟩⟩
def rawTerms : List Term := Proof.Events324.exact82973RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82973
def producerEvent : Nat := 82972
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82973.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 82947, .finite 36, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult82973

namespace SemanticResult82978
def owner : Owner := ⟨.program ⟨214⟩, ⟨11958⟩⟩
def rawTerms : List Term := Proof.Events324.exact82978RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82978
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82978.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge82977.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge82977.frameStart)
    (transferEvent := 82976) (owner := owner)
    (leftResult := 82973) (rightResult := 82970)
    (working := LeftOperatorMerge82977.working)
    (reconstruction := LeftOperatorMerge82977.reconstruction)
    (leftReference := .predecessor 0 82974 .coefficient) (rightReference := .predecessor 1 82975 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult82973.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult82970.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge82977.operationAgreement
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
end SemanticResult82978

namespace SemanticResult82989
def owner : Owner := ⟨.program ⟨214⟩, ⟨23122⟩⟩
def rawTerms : List Term := Proof.Events324.exact82989RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82989
def producerEvent : Nat := 82988
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82989.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 82947, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult82989

namespace SemanticResult82992
def owner : Owner := ⟨.program ⟨214⟩, ⟨25219⟩⟩
def rawTerms : List Term := Proof.Events324.exact82992RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 82992
def producerEvent : Nat := 82991
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult82992.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 82947, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult82992

namespace SemanticResult83001
def owner : Owner := ⟨.program ⟨214⟩, ⟨12054⟩⟩
def rawTerms : List Term := Proof.Events324.exact83001RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83001
def producerEvent : Nat := 83000
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83001.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 82999 .coefficient), 82947, .finite 1296, .identity (.predecessor 0 82999 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult83001

namespace SemanticResult83003
def owner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rawTerms : List Term := Proof.Events324.exact83003RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83003
def producerEvent : Nat := 83002
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83003.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 82947, .large, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult83003

namespace SemanticResult83008
def owner : Owner := ⟨.program ⟨214⟩, ⟨12055⟩⟩
def rawTerms : List Term := Proof.Events324.exact83008RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83008
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83008.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83007.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge83007.frameStart)
    (transferEvent := 83006) (owner := owner)
    (leftResult := 83003) (rightResult := 83001)
    (working := LeftOperatorMerge83007.working)
    (reconstruction := LeftOperatorMerge83007.reconstruction)
    (leftReference := .predecessor 0 83004 .coefficient) (rightReference := .predecessor 1 83005 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult83003.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult83001.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge83007.operationAgreement
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
end SemanticResult83008

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
