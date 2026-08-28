import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard362
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard049
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard050
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard161
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard265
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard353
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard355
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard356
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard358
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard359
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard360
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard361

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult50132
def owner : Owner := ⟨.program ⟨214⟩, ⟨26381⟩⟩
def rawTerms : List Term := Proof.Events195.exact50132RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50132
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50132.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 50129) (rightBinding := 50130)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14900⟩) (rightExpression := ⟨26376⟩)
    (transferEvent := 50131)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50128.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50113.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50132

namespace SemanticResult50141
def owner : Owner := ⟨.program ⟨214⟩, ⟨20331⟩⟩
def rawTerms : List Term := Proof.Events195.exact50141RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 50141
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50141.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge49976.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge49976.frameStart)
    (owner := owner) (leftOwner := SemanticResult36137.owner)
    (rightOwner := SemanticResult49970.owner)
    (leftResult := 36137) (rightResult := 49970)
    (leftActual := SemanticResult36137.actual selector witness)
    (rightActual := SemanticResult49970.actual selector witness)
    (leftRaw := SemanticResult36137.rawTerms)
    (rightRaw := SemanticResult49970.rawTerms)
    (working := LeftOperatorMerge49976.working)
    (leftBinding := 49971) (rightBinding := 49972)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5553⟩) (rightExpression := ⟨20330⟩)
    (coefficientTransfer := 49973) (summaryTransfer := 49975)
    (rightCoefficientProducer := 49969)
    (rightSummaryTransfer := 49974)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge49976.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound49969.actual selector witness)
    (summaryMagnitude := LeftBound49975.actual selector witness)
    (reconstruction := LeftOperatorMerge49976.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36137.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult49970.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49969.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound49969.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge49976.operationAgreement
  · exact LeftBound49975.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge49976.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 50136 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26375⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23726⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26375⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23726⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge49976.working
    [{ coefficient := (1), key := LeftRelationMerge50136.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge50136.frameStart
      LeftRelationMerge50136.owner (.relation 50136) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge50136.deltas
    rows := LeftRelationMerge50136.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge49976.working LeftRelationMerge50136.source
        (relationContext LeftRelationMerge50136.source
          LeftRelationMerge50136.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge49976.working, LeftRelationMerge50136.deltas,
    LeftRelationMerge50136.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 50136)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨20331⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20328⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20328⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge49976.working) (working := relationWorking0)
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
end SemanticResult50141

namespace SemanticResult50148
def owner : Owner := ⟨.program ⟨214⟩, ⟨26378⟩⟩
def rawTerms : List Term := Proof.Events195.exact50148RawTerms
def summary : Bound := (.finite 1291889174379421642752)
def resultEvent : Nat := 50148
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50148.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge50145.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50141.owner)
    (rightOwner := SemanticResult49963.owner)
    (leftResult := 50141) (rightResult := 49963)
    (leftActual := SemanticResult50141.actual selector witness)
    (rightActual := SemanticResult49963.actual selector witness)
    (leftRaw := SemanticResult50141.rawTerms)
    (rightRaw := SemanticResult49963.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1291889172568118132736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50142) (rightBinding := 50143)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20331⟩) (rightExpression := ⟨26377⟩)
    (coefficientTransfer := 50144) (summaryTransfer := 50147)
    (base := LeftOperatorMerge50145.base)
    (reconstruction := LeftOperatorMerge50145.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50141.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult49963.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge50145.operationAgreement
  · rfl
  · decide
end SemanticResult50148

namespace SemanticResult50158
def owner : Owner := ⟨.program ⟨214⟩, ⟨26379⟩⟩
def rawTerms : List Term := Proof.Events195.exact50158RawTerms
def summary : Bound := (.finite 4741253940199267499646124032)
def resultEvent : Nat := 50158
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50158.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨1291889174379421642752, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50154.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge50154.frameStart)
    (owner := owner) (leftOwner := SemanticResult50148.owner)
    (rightOwner := SemanticResult5859.owner)
    (leftResult := 50148) (rightResult := 5859)
    (leftActual := SemanticResult50148.actual selector witness)
    (rightActual := SemanticResult5859.actual selector witness)
    (leftRaw := SemanticResult50148.rawTerms)
    (rightRaw := SemanticResult5859.rawTerms)
    (working := LeftOperatorMerge50154.working)
    (leftBinding := 50149) (rightBinding := 50150)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26378⟩) (rightExpression := ⟨6680⟩)
    (coefficientTransfer := 50151) (summaryTransfer := 50153)
    (rightCoefficientProducer := 5858)
    (rightSummaryTransfer := 50152)
    (leftMaximum := ⟨1291889174379421642752, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge50154.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound5858.actual selector witness)
    (summaryMagnitude := LeftBound50153.actual selector witness)
    (reconstruction := LeftOperatorMerge50154.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50148.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5859.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5858.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound5858.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge50154.operationAgreement
  · exact LeftBound50153.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50154.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 50156 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge50154.working
    [{ coefficient := (-1), key := LeftRelationMerge50156.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge50156.frameStart
      LeftRelationMerge50156.owner (.relation 50156) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge50156.deltas
    rows := LeftRelationMerge50156.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge50154.working LeftRelationMerge50156.source
        (relationContext LeftRelationMerge50156.source
          LeftRelationMerge50156.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge50154.working, LeftRelationMerge50156.deltas,
    LeftRelationMerge50156.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 50156)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨26379⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14896⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge50154.working) (working := relationWorking0)
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
end SemanticResult50158

namespace SemanticResult50163
def owner : Owner := ⟨.program ⟨214⟩, ⟨6628⟩⟩
def rawTerms : List Term := Proof.Events195.exact50163RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50163
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50163.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50162.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge50162.frameStart)
    (transferEvent := 50161) (owner := owner)
    (leftResult := 723) (rightResult := 36045)
    (working := LeftOperatorMerge50162.working)
    (reconstruction := LeftOperatorMerge50162.reconstruction)
    (leftReference := .predecessor 0 50159 .coefficient) (rightReference := .predecessor 1 50160 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult723.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge50162.operationAgreement
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
end SemanticResult50163

namespace SemanticResult50168
def owner : Owner := ⟨.program ⟨214⟩, ⟨7292⟩⟩
def rawTerms : List Term := Proof.Events195.exact50168RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50168
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50168.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50167.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge50167.frameStart)
    (transferEvent := 50166) (owner := owner)
    (leftResult := 35915) (rightResult := 5873)
    (working := LeftOperatorMerge50167.working)
    (reconstruction := LeftOperatorMerge50167.reconstruction)
    (leftReference := .predecessor 0 50164 .coefficient) (rightReference := .predecessor 1 50165 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5873.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge50167.operationAgreement
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
end SemanticResult50168

namespace SemanticResult50172
def owner : Owner := ⟨.program ⟨214⟩, ⟨7761⟩⟩
def rawTerms : List Term := Proof.Events195.exact50172RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50172
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50172.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 50169) (rightBinding := 50170)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7292⟩) (rightExpression := ⟨6628⟩)
    (transferEvent := 50171)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50168.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50163.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50172

namespace SemanticResult50178
def owner : Owner := ⟨.program ⟨214⟩, ⟨7762⟩⟩
def rawTerms : List Term := Proof.Events196.exact50178RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 50178
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50178.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 50175) (survivorTransfer := 50176)
    (survivorEvent := 50177) (resultEvent := resultEvent)
    (rightCoefficientProducer := 20907)
    (owner := owner) (leftOwner := SemanticResult50172.owner)
    (rightOwner := SemanticResult20908.owner)
    (leftResult := 50172) (rightResult := 20908)
    (leftBinding := 50173) (rightBinding := 50174)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7761⟩) (rightExpression := ⟨74⟩)
    (leftActual := SemanticResult50172.actual selector witness)
    (rightActual := SemanticResult20908.actual selector witness)
    (leftRaw := SemanticResult50172.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨74⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound20907.actual selector witness)
    (survivorMagnitude := LeftBound50176.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50172.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20908.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)
  · exact LeftBound50176.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult50178

namespace SemanticResult50185
def owner : Owner := ⟨.program ⟨214⟩, ⟨7810⟩⟩
def rawTerms : List Term := Proof.Events196.exact50185RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 50185
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50185.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge50182.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50178.owner)
    (rightOwner := SemanticResult50178.owner)
    (leftResult := 50178) (rightResult := 50178)
    (leftActual := SemanticResult50178.actual selector witness)
    (rightActual := SemanticResult50178.actual selector witness)
    (leftRaw := SemanticResult50178.rawTerms)
    (rightRaw := SemanticResult50178.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50179) (rightBinding := 50180)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7762⟩) (rightExpression := ⟨7762⟩)
    (coefficientTransfer := 50181) (summaryTransfer := 50184)
    (base := LeftOperatorMerge50182.base)
    (reconstruction := LeftOperatorMerge50182.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50178.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50178.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge50182.operationAgreement
  · rfl
  · decide
end SemanticResult50185

namespace SemanticResult50190
def owner : Owner := ⟨.program ⟨214⟩, ⟨26380⟩⟩
def rawTerms : List Term := Proof.Events196.exact50190RawTerms
def summary : Bound := (.finite 4741253940199267499646124084)
def resultEvent : Nat := 50190
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50190.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50185.owner)
    (rightOwner := SemanticResult50158.owner)
    (leftResult := 50185) (rightResult := 50158)
    (leftActual := SemanticResult50185.actual selector witness)
    (rightActual := SemanticResult50158.actual selector witness)
    (leftRaw := SemanticResult50185.rawTerms)
    (rightRaw := SemanticResult50158.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 4741253940199267499646124032) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50186) (rightBinding := 50187)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7810⟩) (rightExpression := ⟨26379⟩)
    (transferEvent := 50188) (summaryTransferEvent := 50189)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50185.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50158.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50190

namespace SemanticResult50195
def owner : Owner := ⟨.program ⟨214⟩, ⟨26588⟩⟩
def rawTerms : List Term := Proof.Events196.exact50195RawTerms
def summary : Bound := (.finite 9482549007414447334737575988)
def resultEvent : Nat := 50195
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50195.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50190.owner)
    (rightOwner := SemanticResult49946.owner)
    (leftResult := 50190) (rightResult := 49946)
    (leftActual := SemanticResult50190.actual selector witness)
    (rightActual := SemanticResult49946.actual selector witness)
    (leftRaw := SemanticResult50190.rawTerms)
    (rightRaw := SemanticResult49946.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4741253940199267499646124084)
    (rightMaximum := 4741295067215179835091451904) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50191) (rightBinding := 50192)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26380⟩) (rightExpression := ⟨26587⟩)
    (transferEvent := 50193) (summaryTransferEvent := 50194)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50190.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult49946.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50195

namespace SemanticResult50200
def owner : Owner := ⟨.program ⟨214⟩, ⟨26805⟩⟩
def rawTerms : List Term := Proof.Events196.exact50200RawTerms
def summary : Bound := (.finite 14223885201645539505274355764)
def resultEvent : Nat := 50200
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50200.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50195.owner)
    (rightOwner := SemanticResult49734.owner)
    (leftResult := 50195) (rightResult := 49734)
    (leftActual := SemanticResult50195.actual selector witness)
    (rightActual := SemanticResult49734.actual selector witness)
    (leftRaw := SemanticResult50195.rawTerms)
    (rightRaw := SemanticResult49734.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 9482549007414447334737575988)
    (rightMaximum := 4741336194231092170536779776) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50196) (rightBinding := 50197)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26588⟩) (rightExpression := ⟨26804⟩)
    (transferEvent := 50198) (summaryTransferEvent := 50199)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50195.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult49734.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50200

namespace SemanticResult50205
def owner : Owner := ⟨.program ⟨214⟩, ⟨27022⟩⟩
def rawTerms : List Term := Proof.Events196.exact50205RawTerms
def summary : Bound := (.finite 18965303649908456346701791284)
def resultEvent : Nat := 50205
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50205.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50200.owner)
    (rightOwner := SemanticResult49522.owner)
    (leftResult := 50200) (rightResult := 49522)
    (leftActual := SemanticResult50200.actual selector witness)
    (rightActual := SemanticResult49522.actual selector witness)
    (leftRaw := SemanticResult50200.rawTerms)
    (rightRaw := SemanticResult49522.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 14223885201645539505274355764)
    (rightMaximum := 4741418448262916841427435520) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50201) (rightBinding := 50202)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26805⟩) (rightExpression := ⟨27021⟩)
    (transferEvent := 50203) (summaryTransferEvent := 50204)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50200.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult49522.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50205

namespace SemanticResult50210
def owner : Owner := ⟨.program ⟨214⟩, ⟨27239⟩⟩
def rawTerms : List Term := Proof.Events196.exact50210RawTerms
def summary : Bound := (.finite 23706886606235022529910538292)
def resultEvent : Nat := 50210
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50210.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50205.owner)
    (rightOwner := SemanticResult49310.owner)
    (leftResult := 50205) (rightResult := 49310)
    (leftActual := SemanticResult50205.actual selector witness)
    (rightActual := SemanticResult49310.actual selector witness)
    (leftRaw := SemanticResult50205.rawTerms)
    (rightRaw := SemanticResult49310.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 18965303649908456346701791284)
    (rightMaximum := 4741582956326566183208747008) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50206) (rightBinding := 50207)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27022⟩) (rightExpression := ⟨27238⟩)
    (transferEvent := 50208) (summaryTransferEvent := 50209)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50205.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult49310.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50210

namespace SemanticResult50215
def owner : Owner := ⟨.program ⟨214⟩, ⟨27456⟩⟩
def rawTerms : List Term := Proof.Events196.exact50215RawTerms
def summary : Bound := (.finite 28448551816593413384009941044)
def resultEvent : Nat := 50215
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50215.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50210.owner)
    (rightOwner := SemanticResult49098.owner)
    (leftResult := 50210) (rightResult := 49098)
    (leftActual := SemanticResult50210.actual selector witness)
    (rightActual := SemanticResult49098.actual selector witness)
    (leftRaw := SemanticResult50210.rawTerms)
    (rightRaw := SemanticResult49098.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 23706886606235022529910538292)
    (rightMaximum := 4741665210358390854099402752) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50211) (rightBinding := 50212)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27239⟩) (rightExpression := ⟨27455⟩)
    (transferEvent := 50213) (summaryTransferEvent := 50214)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50210.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult49098.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50215

namespace SemanticResult50220
def owner : Owner := ⟨.program ⟨214⟩, ⟨27673⟩⟩
def rawTerms : List Term := Proof.Events196.exact50220RawTerms
def summary : Bound := (.finite 33190381535015453579890655284)
def resultEvent : Nat := 50220
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50220.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50215.owner)
    (rightOwner := SemanticResult48886.owner)
    (leftResult := 50215) (rightResult := 48886)
    (leftActual := SemanticResult50215.actual selector witness)
    (rightActual := SemanticResult48886.actual selector witness)
    (leftRaw := SemanticResult50215.rawTerms)
    (rightRaw := SemanticResult48886.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 28448551816593413384009941044)
    (rightMaximum := 4741829718422040195880714240) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50216) (rightBinding := 50217)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27456⟩) (rightExpression := ⟨27672⟩)
    (transferEvent := 50218) (summaryTransferEvent := 50219)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50215.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult48886.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50220

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
