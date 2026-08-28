import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard514
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard027
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard109
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard110
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard513

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult71586
def owner : Owner := ⟨.program ⟨214⟩, ⟨11215⟩⟩
def rawTerms : List Term := Proof.Events279.exact71586RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71586
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71586.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71583) (rightBinding := 71584)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7194⟩) (rightExpression := ⟨11214⟩)
    (transferEvent := 71585)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71582.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult71577.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71586

namespace SemanticResult71592
def owner : Owner := ⟨.program ⟨214⟩, ⟨11216⟩⟩
def rawTerms : List Term := Proof.Events279.exact71592RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 71592
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71592.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 71589) (survivorTransfer := 71590)
    (survivorEvent := 71591) (resultEvent := resultEvent)
    (rightCoefficientProducer := 12976)
    (owner := owner) (leftOwner := SemanticResult71586.owner)
    (rightOwner := SemanticResult12977.owner)
    (leftResult := 71586) (rightResult := 12977)
    (leftBinding := 71587) (rightBinding := 71588)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11215⟩) (rightExpression := ⟨90⟩)
    (leftActual := SemanticResult71586.actual selector witness)
    (rightActual := SemanticResult12977.actual selector witness)
    (leftRaw := SemanticResult71586.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨90⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound12976.actual selector witness)
    (survivorMagnitude := LeftBound71590.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71586.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12977.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12976.derived selector witness)
  · exact LeftBound71590.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult71592

namespace SemanticResult71600
def owner : Owner := ⟨.program ⟨214⟩, ⟨13550⟩⟩
def rawTerms : List Term := Proof.Events279.exact71600RawTerms
def summary : Bound := (.finite 8320)
def resultEvent : Nat := 71600
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71600.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨10, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge71598.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge71598.frameStart)
    (owner := owner) (leftOwner := SemanticResult71592.owner)
    (rightOwner := SemanticResult3388.owner)
    (leftResult := 71592) (rightResult := 3388)
    (leftActual := SemanticResult71592.actual selector witness)
    (rightActual := SemanticResult3388.actual selector witness)
    (leftRaw := SemanticResult71592.rawTerms)
    (rightRaw := SemanticResult3388.rawTerms)
    (working := LeftOperatorMerge71598.working)
    (leftBinding := 71593) (rightBinding := 71594)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11216⟩) (rightExpression := ⟨13547⟩)
    (coefficientTransfer := 71595) (summaryTransfer := 71597)
    (rightCoefficientProducer := 3387)
    (rightSummaryTransfer := 71596)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨10, by decide⟩)
    (rightRecordedMaximum := 10)
    (rightSummaryMaximum := ⟨10, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge71598.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority3387.actual selector witness)
    (summaryMagnitude := LeftBound71597.actual selector witness)
    (reconstruction := LeftOperatorMerge71598.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71592.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3388.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3387.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority3387.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge71598.operationAgreement
  · exact LeftBound71597.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge71598.working summary) := by
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
end SemanticResult71600

namespace SemanticResult71605
def owner : Owner := ⟨.program ⟨214⟩, ⟨13551⟩⟩
def rawTerms : List Term := Proof.Events279.exact71605RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71605
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71605.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge71604.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge71604.frameStart)
    (transferEvent := 71603) (owner := owner)
    (leftResult := 3388) (rightResult := 65295)
    (working := LeftOperatorMerge71604.working)
    (reconstruction := LeftOperatorMerge71604.reconstruction)
    (leftReference := .predecessor 0 71601 .coefficient) (rightReference := .predecessor 1 71602 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3388.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge71604.operationAgreement
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
end SemanticResult71605

namespace SemanticResult71610
def owner : Owner := ⟨.program ⟨214⟩, ⟨7211⟩⟩
def rawTerms : List Term := Proof.Events279.exact71610RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71610
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71610.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge71609.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge71609.frameStart)
    (transferEvent := 71608) (owner := owner)
    (leftResult := 65165) (rightResult := 13026)
    (working := LeftOperatorMerge71609.working)
    (reconstruction := LeftOperatorMerge71609.reconstruction)
    (leftReference := .predecessor 0 71606 .coefficient) (rightReference := .predecessor 1 71607 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13026.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge71609.operationAgreement
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
end SemanticResult71610

namespace SemanticResult71614
def owner : Owner := ⟨.program ⟨214⟩, ⟨13552⟩⟩
def rawTerms : List Term := Proof.Events279.exact71614RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71614
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71614.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 71611) (rightBinding := 71612)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7211⟩) (rightExpression := ⟨13551⟩)
    (transferEvent := 71613)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71610.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult71605.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult71614

namespace SemanticResult71620
def owner : Owner := ⟨.program ⟨214⟩, ⟨13553⟩⟩
def rawTerms : List Term := Proof.Events279.exact71620RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 71620
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71620.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 71617) (survivorTransfer := 71618)
    (survivorEvent := 71619) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13017)
    (owner := owner) (leftOwner := SemanticResult71614.owner)
    (rightOwner := SemanticResult13018.owner)
    (leftResult := 71614) (rightResult := 13018)
    (leftBinding := 71615) (rightBinding := 71616)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13552⟩) (rightExpression := ⟨107⟩)
    (leftActual := SemanticResult71614.actual selector witness)
    (rightActual := SemanticResult13018.actual selector witness)
    (leftRaw := SemanticResult71614.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨107⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13017.actual selector witness)
    (survivorMagnitude := LeftBound71618.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71614.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13018.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13017.derived selector witness)
  · exact LeftBound71618.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult71620

namespace SemanticResult71630
def owner : Owner := ⟨.program ⟨214⟩, ⟨13554⟩⟩
def rawTerms : List Term := Proof.Events279.exact71630RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 71630
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71630.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge71626.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge71626.frameStart)
    (owner := owner) (leftOwner := SemanticResult71620.owner)
    (rightOwner := SemanticResult13015.owner)
    (leftResult := 71620) (rightResult := 13015)
    (leftActual := SemanticResult71620.actual selector witness)
    (rightActual := SemanticResult13015.actual selector witness)
    (leftRaw := SemanticResult71620.rawTerms)
    (rightRaw := SemanticResult13015.rawTerms)
    (working := LeftOperatorMerge71626.working)
    (leftBinding := 71621) (rightBinding := 71622)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13553⟩) (rightExpression := ⟨7844⟩)
    (coefficientTransfer := 71623) (summaryTransfer := 71625)
    (rightCoefficientProducer := 13014)
    (rightSummaryTransfer := 71624)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge71626.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound13014.actual selector witness)
    (summaryMagnitude := LeftBound71625.actual selector witness)
    (reconstruction := LeftOperatorMerge71626.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71620.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13015.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13014.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound13014.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge71626.operationAgreement
  · exact LeftBound71625.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge71626.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 71627 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge71626.working
    [{ coefficient := (-1), key := LeftRelationMerge71627.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge71627.frameStart
      LeftRelationMerge71627.owner (.relation 71627) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge71627.deltas
    rows := LeftRelationMerge71627.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge71626.working LeftRelationMerge71627.source
        (relationContext LeftRelationMerge71627.source
          LeftRelationMerge71627.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge71626.working, LeftRelationMerge71627.deltas,
    LeftRelationMerge71627.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 71627)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨13554⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge71626.working) (working := relationWorking0)
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
end SemanticResult71630

namespace SemanticResult71636
def owner : Owner := ⟨.program ⟨214⟩, ⟨13555⟩⟩
def rawTerms : List Term := Proof.Events279.exact71636RawTerms
def summary : Bound := (.finite 95428736)
def resultEvent : Nat := 71636
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71636.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge71634.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult71630.owner)
    (rightOwner := SemanticResult71600.owner)
    (leftResult := 71630) (rightResult := 71600)
    (leftActual := SemanticResult71630.actual selector witness)
    (rightActual := SemanticResult71600.actual selector witness)
    (leftRaw := SemanticResult71630.rawTerms)
    (rightRaw := SemanticResult71600.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 8320) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 71631) (rightBinding := 71632)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13554⟩) (rightExpression := ⟨13550⟩)
    (coefficientTransfer := 71633) (summaryTransfer := 71635)
    (base := LeftOperatorMerge71634.base)
    (reconstruction := LeftOperatorMerge71634.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71630.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult71600.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge71634.operationAgreement
  · rfl
  · decide
end SemanticResult71636

namespace SemanticResult71646
def owner : Owner := ⟨.program ⟨214⟩, ⟨25831⟩⟩
def rawTerms : List Term := Proof.Events279.exact71646RawTerms
def summary : Bound := (.finite 350224987979776)
def resultEvent : Nat := 71646
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71646.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95428736, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge71642.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge71642.frameStart)
    (owner := owner) (leftOwner := SemanticResult71636.owner)
    (rightOwner := SemanticResult71572.owner)
    (leftResult := 71636) (rightResult := 71572)
    (leftActual := SemanticResult71636.actual selector witness)
    (rightActual := SemanticResult71572.actual selector witness)
    (leftRaw := SemanticResult71636.rawTerms)
    (rightRaw := SemanticResult71572.rawTerms)
    (working := LeftOperatorMerge71642.working)
    (leftBinding := 71637) (rightBinding := 71638)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13555⟩) (rightExpression := ⟨25830⟩)
    (coefficientTransfer := 71639) (summaryTransfer := 71641)
    (rightCoefficientProducer := 71571)
    (rightSummaryTransfer := 71640)
    (leftMaximum := ⟨95428736, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge71642.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority71571.actual selector witness)
    (summaryMagnitude := LeftBound71641.actual selector witness)
    (reconstruction := LeftOperatorMerge71642.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult71636.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult71572.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71571.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority71571.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge71642.operationAgreement
  · exact LeftBound71641.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge71642.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 71643 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23456⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23456⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge71642.working
    [{ coefficient := (-1), key := LeftRelationMerge71643.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge71643.frameStart
      LeftRelationMerge71643.owner (.relation 71643) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge71643.deltas
    rows := LeftRelationMerge71643.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge71642.working LeftRelationMerge71643.source
        (relationContext LeftRelationMerge71643.source
          LeftRelationMerge71643.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge71642.working, LeftRelationMerge71643.deltas,
    LeftRelationMerge71643.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 71643)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25831⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11213⟩⟩, ⟨.program ⟨214⟩, ⟨13547⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25830⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge71642.working) (working := relationWorking0)
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
end SemanticResult71646

namespace SemanticResult71649
def owner : Owner := ⟨.program ⟨214⟩, ⟨19308⟩⟩
def rawTerms : List Term := Proof.Events279.exact71649RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71649
def producerEvent : Nat := 71648
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71649.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨12⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨12⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult71649

namespace SemanticResult71653
def owner : Owner := ⟨.program ⟨214⟩, ⟨19310⟩⟩
def rawTerms : List Term := Proof.Events279.exact71653RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71653
def producerEvent : Nat := 71652
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71653.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 71650 .coefficient) (.value (.predecessor 1 71651 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 71650 .coefficient) (.value (.predecessor 1 71651 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult71653

namespace SemanticResult71731
def owner : Owner := ⟨.program ⟨214⟩, ⟨11213⟩⟩
def rawTerms : List Term := Proof.Events280.exact71731RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71731
def producerEvent : Nat := 71730
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71731.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 71708, .finite 10, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult71731

namespace SemanticResult71734
def owner : Owner := ⟨.program ⟨214⟩, ⟨13547⟩⟩
def rawTerms : List Term := Proof.Events280.exact71734RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71734
def producerEvent : Nat := 71733
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71734.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 71708, .finite 10, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult71734

namespace SemanticResult71739
def owner : Owner := ⟨.program ⟨214⟩, ⟨13548⟩⟩
def rawTerms : List Term := Proof.Events280.exact71739RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71739
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71739.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge71738.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge71738.frameStart)
    (transferEvent := 71737) (owner := owner)
    (leftResult := 71734) (rightResult := 71731)
    (working := LeftOperatorMerge71738.working)
    (reconstruction := LeftOperatorMerge71738.reconstruction)
    (leftReference := .predecessor 0 71735 .coefficient) (rightReference := .predecessor 1 71736 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult71734.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult71731.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge71738.operationAgreement
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
end SemanticResult71739

namespace SemanticResult71750
def owner : Owner := ⟨.program ⟨214⟩, ⟨23456⟩⟩
def rawTerms : List Term := Proof.Events280.exact71750RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 71750
def producerEvent : Nat := 71749
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult71750.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 71708, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult71750

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
