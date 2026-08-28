import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard611
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard032
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard106
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard566
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard610

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult85719
def owner : Owner := ⟨.program ⟨214⟩, ⟨13776⟩⟩
def rawTerms : List Term := Proof.Events334.exact85719RawTerms
def summary : Bound := (.finite 9984)
def resultEvent : Nat := 85719
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85719.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨12, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge85717.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge85717.frameStart)
    (owner := owner) (leftOwner := SemanticResult85711.owner)
    (rightOwner := SemanticResult4107.owner)
    (leftResult := 85711) (rightResult := 4107)
    (leftActual := SemanticResult85711.actual selector witness)
    (rightActual := SemanticResult4107.actual selector witness)
    (leftRaw := SemanticResult85711.rawTerms)
    (rightRaw := SemanticResult4107.rawTerms)
    (working := LeftOperatorMerge85717.working)
    (leftBinding := 85712) (rightBinding := 85713)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11304⟩) (rightExpression := ⟨13773⟩)
    (coefficientTransfer := 85714) (summaryTransfer := 85716)
    (rightCoefficientProducer := 4106)
    (rightSummaryTransfer := 85715)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨12, by decide⟩)
    (rightRecordedMaximum := 12)
    (rightSummaryMaximum := ⟨12, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge85717.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority4106.actual selector witness)
    (summaryMagnitude := LeftBound85716.actual selector witness)
    (reconstruction := LeftOperatorMerge85717.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult85711.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4107.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4106.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority4106.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge85717.operationAgreement
  · exact LeftBound85716.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge85717.working summary) := by
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
end SemanticResult85719

namespace SemanticResult85724
def owner : Owner := ⟨.program ⟨214⟩, ⟨13777⟩⟩
def rawTerms : List Term := Proof.Events334.exact85724RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85724
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85724.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge85723.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge85723.frameStart)
    (transferEvent := 85722) (owner := owner)
    (leftResult := 4107) (rightResult := 79920)
    (working := LeftOperatorMerge85723.working)
    (reconstruction := LeftOperatorMerge85723.reconstruction)
    (leftReference := .predecessor 0 85720 .coefficient) (rightReference := .predecessor 1 85721 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4107.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79920.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge85723.operationAgreement
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
end SemanticResult85724

namespace SemanticResult85729
def owner : Owner := ⟨.program ⟨214⟩, ⟨7250⟩⟩
def rawTerms : List Term := Proof.Events334.exact85729RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85729
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85729.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge85728.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge85728.frameStart)
    (transferEvent := 85727) (owner := owner)
    (leftResult := 79790) (rightResult := 12525)
    (working := LeftOperatorMerge85728.working)
    (reconstruction := LeftOperatorMerge85728.reconstruction)
    (leftReference := .predecessor 0 85725 .coefficient) (rightReference := .predecessor 1 85726 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12525.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge85728.operationAgreement
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
end SemanticResult85729

namespace SemanticResult85733
def owner : Owner := ⟨.program ⟨214⟩, ⟨13778⟩⟩
def rawTerms : List Term := Proof.Events334.exact85733RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85733
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85733.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 85730) (rightBinding := 85731)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7250⟩) (rightExpression := ⟨13777⟩)
    (transferEvent := 85732)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult85729.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult85724.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult85733

namespace SemanticResult85739
def owner : Owner := ⟨.program ⟨214⟩, ⟨13779⟩⟩
def rawTerms : List Term := Proof.Events334.exact85739RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 85739
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85739.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 85736) (survivorTransfer := 85737)
    (survivorEvent := 85738) (resultEvent := resultEvent)
    (rightCoefficientProducer := 12516)
    (owner := owner) (leftOwner := SemanticResult85733.owner)
    (rightOwner := SemanticResult12517.owner)
    (leftResult := 85733) (rightResult := 12517)
    (leftBinding := 85734) (rightBinding := 85735)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13778⟩) (rightExpression := ⟨108⟩)
    (leftActual := SemanticResult85733.actual selector witness)
    (rightActual := SemanticResult12517.actual selector witness)
    (leftRaw := SemanticResult85733.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨108⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound12516.actual selector witness)
    (survivorMagnitude := LeftBound85737.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult85733.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12517.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12516.derived selector witness)
  · exact LeftBound85737.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult85739

namespace SemanticResult85749
def owner : Owner := ⟨.program ⟨214⟩, ⟨13780⟩⟩
def rawTerms : List Term := Proof.Events334.exact85749RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 85749
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85749.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge85745.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge85745.frameStart)
    (owner := owner) (leftOwner := SemanticResult85739.owner)
    (rightOwner := SemanticResult12514.owner)
    (leftResult := 85739) (rightResult := 12514)
    (leftActual := SemanticResult85739.actual selector witness)
    (rightActual := SemanticResult12514.actual selector witness)
    (leftRaw := SemanticResult85739.rawTerms)
    (rightRaw := SemanticResult12514.rawTerms)
    (working := LeftOperatorMerge85745.working)
    (leftBinding := 85740) (rightBinding := 85741)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13779⟩) (rightExpression := ⟨7847⟩)
    (coefficientTransfer := 85742) (summaryTransfer := 85744)
    (rightCoefficientProducer := 12513)
    (rightSummaryTransfer := 85743)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge85745.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound12513.actual selector witness)
    (summaryMagnitude := LeftBound85744.actual selector witness)
    (reconstruction := LeftOperatorMerge85745.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult85739.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12514.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12513.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound12513.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge85745.operationAgreement
  · exact LeftBound85744.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge85745.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 85746 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge85745.working
    [{ coefficient := (-1), key := LeftRelationMerge85746.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge85746.frameStart
      LeftRelationMerge85746.owner (.relation 85746) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge85746.deltas
    rows := LeftRelationMerge85746.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge85745.working LeftRelationMerge85746.source
        (relationContext LeftRelationMerge85746.source
          LeftRelationMerge85746.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge85745.working, LeftRelationMerge85746.deltas,
    LeftRelationMerge85746.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 85746)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨13780⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge85745.working) (working := relationWorking0)
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
end SemanticResult85749

namespace SemanticResult85755
def owner : Owner := ⟨.program ⟨214⟩, ⟨13781⟩⟩
def rawTerms : List Term := Proof.Events334.exact85755RawTerms
def summary : Bound := (.finite 95430400)
def resultEvent : Nat := 85755
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85755.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge85753.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult85749.owner)
    (rightOwner := SemanticResult85719.owner)
    (leftResult := 85749) (rightResult := 85719)
    (leftActual := SemanticResult85749.actual selector witness)
    (rightActual := SemanticResult85719.actual selector witness)
    (leftRaw := SemanticResult85749.rawTerms)
    (rightRaw := SemanticResult85719.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 9984) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 85750) (rightBinding := 85751)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13780⟩) (rightExpression := ⟨13776⟩)
    (coefficientTransfer := 85752) (summaryTransfer := 85754)
    (base := LeftOperatorMerge85753.base)
    (reconstruction := LeftOperatorMerge85753.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult85749.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult85719.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge85753.operationAgreement
  · rfl
  · decide
end SemanticResult85755

namespace SemanticResult85765
def owner : Owner := ⟨.program ⟨214⟩, ⟨25913⟩⟩
def rawTerms : List Term := Proof.Events335.exact85765RawTerms
def summary : Bound := (.finite 350231094886400)
def resultEvent : Nat := 85765
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85765.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95430400, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge85761.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge85761.frameStart)
    (owner := owner) (leftOwner := SemanticResult85755.owner)
    (rightOwner := SemanticResult85691.owner)
    (leftResult := 85755) (rightResult := 85691)
    (leftActual := SemanticResult85755.actual selector witness)
    (rightActual := SemanticResult85691.actual selector witness)
    (leftRaw := SemanticResult85755.rawTerms)
    (rightRaw := SemanticResult85691.rawTerms)
    (working := LeftOperatorMerge85761.working)
    (leftBinding := 85756) (rightBinding := 85757)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13781⟩) (rightExpression := ⟨25912⟩)
    (coefficientTransfer := 85758) (summaryTransfer := 85760)
    (rightCoefficientProducer := 85690)
    (rightSummaryTransfer := 85759)
    (leftMaximum := ⟨95430400, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge85761.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority85690.actual selector witness)
    (summaryMagnitude := LeftBound85760.actual selector witness)
    (reconstruction := LeftOperatorMerge85761.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult85755.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult85691.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85690.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority85690.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge85761.operationAgreement
  · exact LeftBound85760.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge85761.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 85762 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23500⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23500⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge85761.working
    [{ coefficient := (-1), key := LeftRelationMerge85762.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge85762.frameStart
      LeftRelationMerge85762.owner (.relation 85762) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge85762.deltas
    rows := LeftRelationMerge85762.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge85761.working LeftRelationMerge85762.source
        (relationContext LeftRelationMerge85762.source
          LeftRelationMerge85762.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge85761.working, LeftRelationMerge85762.deltas,
    LeftRelationMerge85762.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 85762)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25913⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge85761.working) (working := relationWorking0)
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
end SemanticResult85765

namespace SemanticResult85768
def owner : Owner := ⟨.program ⟨214⟩, ⟨19384⟩⟩
def rawTerms : List Term := Proof.Events335.exact85768RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85768
def producerEvent : Nat := 85767
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85768.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨13⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨13⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult85768

namespace SemanticResult85772
def owner : Owner := ⟨.program ⟨214⟩, ⟨19386⟩⟩
def rawTerms : List Term := Proof.Events335.exact85772RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85772
def producerEvent : Nat := 85771
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85772.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 85769 .coefficient) (.value (.predecessor 1 85770 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 85769 .coefficient) (.value (.predecessor 1 85770 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult85772

namespace SemanticResult85850
def owner : Owner := ⟨.program ⟨214⟩, ⟨11301⟩⟩
def rawTerms : List Term := Proof.Events335.exact85850RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85850
def producerEvent : Nat := 85849
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85850.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 85827, .finite 12, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult85850

namespace SemanticResult85853
def owner : Owner := ⟨.program ⟨214⟩, ⟨13773⟩⟩
def rawTerms : List Term := Proof.Events335.exact85853RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85853
def producerEvent : Nat := 85852
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85853.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 85827, .finite 12, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult85853

namespace SemanticResult85858
def owner : Owner := ⟨.program ⟨214⟩, ⟨13774⟩⟩
def rawTerms : List Term := Proof.Events335.exact85858RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85858
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85858.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge85857.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge85857.frameStart)
    (transferEvent := 85856) (owner := owner)
    (leftResult := 85853) (rightResult := 85850)
    (working := LeftOperatorMerge85857.working)
    (reconstruction := LeftOperatorMerge85857.reconstruction)
    (leftReference := .predecessor 0 85854 .coefficient) (rightReference := .predecessor 1 85855 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult85853.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult85850.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge85857.operationAgreement
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
end SemanticResult85858

namespace SemanticResult85869
def owner : Owner := ⟨.program ⟨214⟩, ⟨23500⟩⟩
def rawTerms : List Term := Proof.Events335.exact85869RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85869
def producerEvent : Nat := 85868
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85869.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 85827, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult85869

namespace SemanticResult85872
def owner : Owner := ⟨.program ⟨214⟩, ⟨25912⟩⟩
def rawTerms : List Term := Proof.Events335.exact85872RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85872
def producerEvent : Nat := 85871
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85872.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 85827, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult85872

namespace SemanticResult85881
def owner : Owner := ⟨.program ⟨214⟩, ⟨13881⟩⟩
def rawTerms : List Term := Proof.Events335.exact85881RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85881
def producerEvent : Nat := 85880
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85881.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 85879 .coefficient), 85827, .finite 144, .identity (.predecessor 0 85879 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult85881

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
