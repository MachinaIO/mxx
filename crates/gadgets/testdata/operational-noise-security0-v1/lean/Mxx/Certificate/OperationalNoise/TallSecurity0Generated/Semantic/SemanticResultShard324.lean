import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard324
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard015
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard121
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard122
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard323

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult43788
def owner : Owner := ⟨.program ⟨214⟩, ⟨10697⟩⟩
def rawTerms : List Term := Proof.Events171.exact43788RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 43788
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43788.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 43785) (survivorTransfer := 43786)
    (survivorEvent := 43787) (resultEvent := resultEvent)
    (rightCoefficientProducer := 14479)
    (owner := owner) (leftOwner := SemanticResult43782.owner)
    (rightOwner := SemanticResult14480.owner)
    (leftResult := 43782) (rightResult := 14480)
    (leftBinding := 43783) (rightBinding := 43784)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10696⟩) (rightExpression := ⟨87⟩)
    (leftActual := SemanticResult43782.actual selector witness)
    (rightActual := SemanticResult14480.actual selector witness)
    (leftRaw := SemanticResult43782.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨87⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound14479.actual selector witness)
    (survivorMagnitude := LeftBound43786.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult43782.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14480.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14479.derived selector witness)
  · exact LeftBound43786.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult43788

namespace SemanticResult43796
def owner : Owner := ⟨.program ⟨214⟩, ⟨10698⟩⟩
def rawTerms : List Term := Proof.Events171.exact43796RawTerms
def summary : Bound := (.finite 2496)
def resultEvent : Nat := 43796
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43796.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨3, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43794.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge43794.frameStart)
    (owner := owner) (leftOwner := SemanticResult43788.owner)
    (rightOwner := SemanticResult1961.owner)
    (leftResult := 43788) (rightResult := 1961)
    (leftActual := SemanticResult43788.actual selector witness)
    (rightActual := SemanticResult1961.actual selector witness)
    (leftRaw := SemanticResult43788.rawTerms)
    (rightRaw := SemanticResult1961.rawTerms)
    (working := LeftOperatorMerge43794.working)
    (leftBinding := 43789) (rightBinding := 43790)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10697⟩) (rightExpression := ⟨9515⟩)
    (coefficientTransfer := 43791) (summaryTransfer := 43793)
    (rightCoefficientProducer := 1960)
    (rightSummaryTransfer := 43792)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨3, by decide⟩)
    (rightRecordedMaximum := 3)
    (rightSummaryMaximum := ⟨3, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge43794.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1960.actual selector witness)
    (summaryMagnitude := LeftBound43793.actual selector witness)
    (reconstruction := LeftOperatorMerge43794.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult43788.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1961.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1960.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1960.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge43794.operationAgreement
  · exact LeftBound43793.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43794.working summary) := by
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
end SemanticResult43796

namespace SemanticResult43801
def owner : Owner := ⟨.program ⟨214⟩, ⟨9516⟩⟩
def rawTerms : List Term := Proof.Events171.exact43801RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43801
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43801.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43800.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge43800.frameStart)
    (transferEvent := 43799) (owner := owner)
    (leftResult := 1961) (rightResult := 36045)
    (working := LeftOperatorMerge43800.working)
    (reconstruction := LeftOperatorMerge43800.reconstruction)
    (leftReference := .predecessor 0 43797 .coefficient) (rightReference := .predecessor 1 43798 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1961.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge43800.operationAgreement
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
end SemanticResult43801

namespace SemanticResult43806
def owner : Owner := ⟨.program ⟨214⟩, ⟨7314⟩⟩
def rawTerms : List Term := Proof.Events171.exact43806RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43806
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43806.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43805.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge43805.frameStart)
    (transferEvent := 43804) (owner := owner)
    (leftResult := 35915) (rightResult := 14529)
    (working := LeftOperatorMerge43805.working)
    (reconstruction := LeftOperatorMerge43805.reconstruction)
    (leftReference := .predecessor 0 43802 .coefficient) (rightReference := .predecessor 1 43803 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14529.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge43805.operationAgreement
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
end SemanticResult43806

namespace SemanticResult43810
def owner : Owner := ⟨.program ⟨214⟩, ⟨9517⟩⟩
def rawTerms : List Term := Proof.Events171.exact43810RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43810
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43810.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 43807) (rightBinding := 43808)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7314⟩) (rightExpression := ⟨9516⟩)
    (transferEvent := 43809)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult43806.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult43801.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult43810

namespace SemanticResult43816
def owner : Owner := ⟨.program ⟨214⟩, ⟨9518⟩⟩
def rawTerms : List Term := Proof.Events171.exact43816RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 43816
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43816.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 43813) (survivorTransfer := 43814)
    (survivorEvent := 43815) (resultEvent := resultEvent)
    (rightCoefficientProducer := 14520)
    (owner := owner) (leftOwner := SemanticResult43810.owner)
    (rightOwner := SemanticResult14521.owner)
    (leftResult := 43810) (rightResult := 14521)
    (leftBinding := 43811) (rightBinding := 43812)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9517⟩) (rightExpression := ⟨96⟩)
    (leftActual := SemanticResult43810.actual selector witness)
    (rightActual := SemanticResult14521.actual selector witness)
    (leftRaw := SemanticResult43810.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨96⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound14520.actual selector witness)
    (survivorMagnitude := LeftBound43814.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult43810.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14521.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14520.derived selector witness)
  · exact LeftBound43814.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult43816

namespace SemanticResult43826
def owner : Owner := ⟨.program ⟨214⟩, ⟨9519⟩⟩
def rawTerms : List Term := Proof.Events171.exact43826RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 43826
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43826.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43822.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge43822.frameStart)
    (owner := owner) (leftOwner := SemanticResult43816.owner)
    (rightOwner := SemanticResult14518.owner)
    (leftResult := 43816) (rightResult := 14518)
    (leftActual := SemanticResult43816.actual selector witness)
    (rightActual := SemanticResult14518.actual selector witness)
    (leftRaw := SemanticResult43816.rawTerms)
    (rightRaw := SemanticResult14518.rawTerms)
    (working := LeftOperatorMerge43822.working)
    (leftBinding := 43817) (rightBinding := 43818)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9518⟩) (rightExpression := ⟨7835⟩)
    (coefficientTransfer := 43819) (summaryTransfer := 43821)
    (rightCoefficientProducer := 14517)
    (rightSummaryTransfer := 43820)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge43822.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound14517.actual selector witness)
    (summaryMagnitude := LeftBound43821.actual selector witness)
    (reconstruction := LeftOperatorMerge43822.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult43816.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14518.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14517.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound14517.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge43822.operationAgreement
  · exact LeftBound43821.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43822.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 43823 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge43822.working
    [{ coefficient := (-1), key := LeftRelationMerge43823.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge43823.frameStart
      LeftRelationMerge43823.owner (.relation 43823) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge43823.deltas
    rows := LeftRelationMerge43823.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge43822.working LeftRelationMerge43823.source
        (relationContext LeftRelationMerge43823.source
          LeftRelationMerge43823.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge43822.working, LeftRelationMerge43823.deltas,
    LeftRelationMerge43823.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 43823)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9519⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9515⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge43822.working) (working := relationWorking0)
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
end SemanticResult43826

namespace SemanticResult43832
def owner : Owner := ⟨.program ⟨214⟩, ⟨10699⟩⟩
def rawTerms : List Term := Proof.Events171.exact43832RawTerms
def summary : Bound := (.finite 95422912)
def resultEvent : Nat := 43832
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43832.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge43830.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult43826.owner)
    (rightOwner := SemanticResult43796.owner)
    (leftResult := 43826) (rightResult := 43796)
    (leftActual := SemanticResult43826.actual selector witness)
    (rightActual := SemanticResult43796.actual selector witness)
    (leftRaw := SemanticResult43826.rawTerms)
    (rightRaw := SemanticResult43796.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 2496) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 43827) (rightBinding := 43828)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9519⟩) (rightExpression := ⟨10698⟩)
    (coefficientTransfer := 43829) (summaryTransfer := 43831)
    (base := LeftOperatorMerge43830.base)
    (reconstruction := LeftOperatorMerge43830.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult43826.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult43796.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge43830.operationAgreement
  · rfl
  · decide
end SemanticResult43832

namespace SemanticResult43842
def owner : Owner := ⟨.program ⟨214⟩, ⟨24999⟩⟩
def rawTerms : List Term := Proof.Events171.exact43842RawTerms
def summary : Bound := (.finite 350203613806592)
def resultEvent : Nat := 43842
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43842.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95422912, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43838.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge43838.frameStart)
    (owner := owner) (leftOwner := SemanticResult43832.owner)
    (rightOwner := SemanticResult43768.owner)
    (leftResult := 43832) (rightResult := 43768)
    (leftActual := SemanticResult43832.actual selector witness)
    (rightActual := SemanticResult43768.actual selector witness)
    (leftRaw := SemanticResult43832.rawTerms)
    (rightRaw := SemanticResult43768.rawTerms)
    (working := LeftOperatorMerge43838.working)
    (leftBinding := 43833) (rightBinding := 43834)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10699⟩) (rightExpression := ⟨24998⟩)
    (coefficientTransfer := 43835) (summaryTransfer := 43837)
    (rightCoefficientProducer := 43767)
    (rightSummaryTransfer := 43836)
    (leftMaximum := ⟨95422912, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge43838.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority43767.actual selector witness)
    (summaryMagnitude := LeftBound43837.actual selector witness)
    (reconstruction := LeftOperatorMerge43838.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult43832.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult43768.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43767.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority43767.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge43838.operationAgreement
  · exact LeftBound43837.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43838.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 43839 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23000⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23000⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge43838.working
    [{ coefficient := (-1), key := LeftRelationMerge43839.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge43839.frameStart
      LeftRelationMerge43839.owner (.relation 43839) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge43839.deltas
    rows := LeftRelationMerge43839.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge43838.working LeftRelationMerge43839.source
        (relationContext LeftRelationMerge43839.source
          LeftRelationMerge43839.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge43838.working, LeftRelationMerge43839.deltas,
    LeftRelationMerge43839.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 43839)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨24999⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9515⟩⟩, ⟨.program ⟨214⟩, ⟨10692⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24998⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge43838.working) (working := relationWorking0)
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
end SemanticResult43842

namespace SemanticResult43845
def owner : Owner := ⟨.program ⟨214⟩, ⟨19104⟩⟩
def rawTerms : List Term := Proof.Events171.exact43845RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43845
def producerEvent : Nat := 43844
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43845.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨8⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨8⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult43845

namespace SemanticResult43849
def owner : Owner := ⟨.program ⟨214⟩, ⟨19106⟩⟩
def rawTerms : List Term := Proof.Events171.exact43849RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43849
def producerEvent : Nat := 43848
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43849.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 43846 .coefficient) (.value (.predecessor 1 43847 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 43846 .coefficient) (.value (.predecessor 1 43847 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult43849

namespace SemanticResult43927
def owner : Owner := ⟨.program ⟨214⟩, ⟨10692⟩⟩
def rawTerms : List Term := Proof.Events171.exact43927RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43927
def producerEvent : Nat := 43926
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43927.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 43904, .finite 3, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult43927

namespace SemanticResult43930
def owner : Owner := ⟨.program ⟨214⟩, ⟨9515⟩⟩
def rawTerms : List Term := Proof.Events171.exact43930RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43930
def producerEvent : Nat := 43929
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43930.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 43904, .finite 3, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult43930

namespace SemanticResult43935
def owner : Owner := ⟨.program ⟨214⟩, ⟨10693⟩⟩
def rawTerms : List Term := Proof.Events171.exact43935RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43935
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43935.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43934.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge43934.frameStart)
    (transferEvent := 43933) (owner := owner)
    (leftResult := 43930) (rightResult := 43927)
    (working := LeftOperatorMerge43934.working)
    (reconstruction := LeftOperatorMerge43934.reconstruction)
    (leftReference := .predecessor 0 43931 .coefficient) (rightReference := .predecessor 1 43932 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult43930.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult43927.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge43934.operationAgreement
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
end SemanticResult43935

namespace SemanticResult43946
def owner : Owner := ⟨.program ⟨214⟩, ⟨23000⟩⟩
def rawTerms : List Term := Proof.Events171.exact43946RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43946
def producerEvent : Nat := 43945
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43946.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 43904, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult43946

namespace SemanticResult43949
def owner : Owner := ⟨.program ⟨214⟩, ⟨24998⟩⟩
def rawTerms : List Term := Proof.Events171.exact43949RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43949
def producerEvent : Nat := 43948
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43949.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 43904, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult43949

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
