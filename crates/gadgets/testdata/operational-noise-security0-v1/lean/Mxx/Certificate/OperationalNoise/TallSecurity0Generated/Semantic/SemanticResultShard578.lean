import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard578
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard069
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard577

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult81413
def owner : Owner := ⟨.program ⟨214⟩, ⟨10032⟩⟩
def rawTerms : List Term := Proof.Events318.exact81413RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81413
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81413.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 81410) (rightBinding := 81411)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7223⟩) (rightExpression := ⟨10031⟩)
    (transferEvent := 81412)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult81409.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult81404.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult81413

namespace SemanticResult81419
def owner : Owner := ⟨.program ⟨214⟩, ⟨10033⟩⟩
def rawTerms : List Term := Proof.Events318.exact81419RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 81419
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81419.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 81416) (survivorTransfer := 81417)
    (survivorEvent := 81418) (resultEvent := resultEvent)
    (rightCoefficientProducer := 8007)
    (owner := owner) (leftOwner := SemanticResult81413.owner)
    (rightOwner := SemanticResult8008.owner)
    (leftResult := 81413) (rightResult := 8008)
    (leftBinding := 81414) (rightBinding := 81415)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10032⟩) (rightExpression := ⟨81⟩)
    (leftActual := SemanticResult81413.actual selector witness)
    (rightActual := SemanticResult8008.actual selector witness)
    (leftRaw := SemanticResult81413.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨81⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound8007.actual selector witness)
    (survivorMagnitude := LeftBound81417.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult81413.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8008.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8007.derived selector witness)
  · exact LeftBound81417.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult81419

namespace SemanticResult81429
def owner : Owner := ⟨.program ⟨214⟩, ⟨10034⟩⟩
def rawTerms : List Term := Proof.Events318.exact81429RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 81429
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81429.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81425.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge81425.frameStart)
    (owner := owner) (leftOwner := SemanticResult81419.owner)
    (rightOwner := SemanticResult8005.owner)
    (leftResult := 81419) (rightResult := 8005)
    (leftActual := SemanticResult81419.actual selector witness)
    (rightActual := SemanticResult8005.actual selector witness)
    (leftRaw := SemanticResult81419.rawTerms)
    (rightRaw := SemanticResult8005.rawTerms)
    (working := LeftOperatorMerge81425.working)
    (leftBinding := 81420) (rightBinding := 81421)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10033⟩) (rightExpression := ⟨7874⟩)
    (coefficientTransfer := 81422) (summaryTransfer := 81424)
    (rightCoefficientProducer := 8004)
    (rightSummaryTransfer := 81423)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge81425.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound8004.actual selector witness)
    (summaryMagnitude := LeftBound81424.actual selector witness)
    (reconstruction := LeftOperatorMerge81425.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult81419.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8005.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8004.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound8004.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge81425.operationAgreement
  · exact LeftBound81424.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81425.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 81426 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge81425.working
    [{ coefficient := (-1), key := LeftRelationMerge81426.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge81426.frameStart
      LeftRelationMerge81426.owner (.relation 81426) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge81426.deltas
    rows := LeftRelationMerge81426.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge81425.working LeftRelationMerge81426.source
        (relationContext LeftRelationMerge81426.source
          LeftRelationMerge81426.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge81425.working, LeftRelationMerge81426.deltas,
    LeftRelationMerge81426.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 81426)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10034⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge81425.working) (working := relationWorking0)
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
end SemanticResult81429

namespace SemanticResult81435
def owner : Owner := ⟨.program ⟨214⟩, ⟨12769⟩⟩
def rawTerms : List Term := Proof.Events318.exact81435RawTerms
def summary : Bound := (.finite 95458688)
def resultEvent : Nat := 81435
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81435.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge81433.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult81429.owner)
    (rightOwner := SemanticResult81399.owner)
    (leftResult := 81429) (rightResult := 81399)
    (leftActual := SemanticResult81429.actual selector witness)
    (rightActual := SemanticResult81399.actual selector witness)
    (leftRaw := SemanticResult81429.rawTerms)
    (rightRaw := SemanticResult81399.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 38272) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 81430) (rightBinding := 81431)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10034⟩) (rightExpression := ⟨12768⟩)
    (coefficientTransfer := 81432) (summaryTransfer := 81434)
    (base := LeftOperatorMerge81433.base)
    (reconstruction := LeftOperatorMerge81433.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult81429.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult81399.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge81433.operationAgreement
  · rfl
  · decide
end SemanticResult81435

namespace SemanticResult81445
def owner : Owner := ⟨.program ⟨214⟩, ⟨25528⟩⟩
def rawTerms : List Term := Proof.Events318.exact81445RawTerms
def summary : Bound := (.finite 350334912299008)
def resultEvent : Nat := 81445
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81445.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95458688, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81441.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge81441.frameStart)
    (owner := owner) (leftOwner := SemanticResult81435.owner)
    (rightOwner := SemanticResult81371.owner)
    (leftResult := 81435) (rightResult := 81371)
    (leftActual := SemanticResult81435.actual selector witness)
    (rightActual := SemanticResult81371.actual selector witness)
    (leftRaw := SemanticResult81435.rawTerms)
    (rightRaw := SemanticResult81371.rawTerms)
    (working := LeftOperatorMerge81441.working)
    (leftBinding := 81436) (rightBinding := 81437)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12769⟩) (rightExpression := ⟨25527⟩)
    (coefficientTransfer := 81438) (summaryTransfer := 81440)
    (rightCoefficientProducer := 81370)
    (rightSummaryTransfer := 81439)
    (leftMaximum := ⟨95458688, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge81441.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority81370.actual selector witness)
    (summaryMagnitude := LeftBound81440.actual selector witness)
    (reconstruction := LeftOperatorMerge81441.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult81435.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult81371.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81370.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority81370.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge81441.operationAgreement
  · exact LeftBound81440.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81441.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 81442 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23290⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23290⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge81441.working
    [{ coefficient := (-1), key := LeftRelationMerge81442.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge81442.frameStart
      LeftRelationMerge81442.owner (.relation 81442) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge81442.deltas
    rows := LeftRelationMerge81442.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge81441.working LeftRelationMerge81442.source
        (relationContext LeftRelationMerge81442.source
          LeftRelationMerge81442.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge81441.working, LeftRelationMerge81442.deltas,
    LeftRelationMerge81442.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 81442)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25528⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨10030⟩⟩, ⟨.program ⟨214⟩, ⟨12762⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25527⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge81441.working) (working := relationWorking0)
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
end SemanticResult81445

namespace SemanticResult81448
def owner : Owner := ⟨.program ⟨214⟩, ⟨20032⟩⟩
def rawTerms : List Term := Proof.Events318.exact81448RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81448
def producerEvent : Nat := 81447
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81448.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨23⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨23⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult81448

namespace SemanticResult81452
def owner : Owner := ⟨.program ⟨214⟩, ⟨20034⟩⟩
def rawTerms : List Term := Proof.Events318.exact81452RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81452
def producerEvent : Nat := 81451
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81452.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 81449 .coefficient) (.value (.predecessor 1 81450 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 81449 .coefficient) (.value (.predecessor 1 81450 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult81452

namespace SemanticResult81530
def owner : Owner := ⟨.program ⟨214⟩, ⟨12762⟩⟩
def rawTerms : List Term := Proof.Events318.exact81530RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81530
def producerEvent : Nat := 81529
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81530.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 81507, .finite 46, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult81530

namespace SemanticResult81533
def owner : Owner := ⟨.program ⟨214⟩, ⟨10030⟩⟩
def rawTerms : List Term := Proof.Events318.exact81533RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81533
def producerEvent : Nat := 81532
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81533.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 81507, .finite 46, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult81533

namespace SemanticResult81538
def owner : Owner := ⟨.program ⟨214⟩, ⟨12763⟩⟩
def rawTerms : List Term := Proof.Events318.exact81538RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81538
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81538.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81537.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge81537.frameStart)
    (transferEvent := 81536) (owner := owner)
    (leftResult := 81533) (rightResult := 81530)
    (working := LeftOperatorMerge81537.working)
    (reconstruction := LeftOperatorMerge81537.reconstruction)
    (leftReference := .predecessor 0 81534 .coefficient) (rightReference := .predecessor 1 81535 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult81533.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult81530.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge81537.operationAgreement
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
end SemanticResult81538

namespace SemanticResult81549
def owner : Owner := ⟨.program ⟨214⟩, ⟨23290⟩⟩
def rawTerms : List Term := Proof.Events318.exact81549RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81549
def producerEvent : Nat := 81548
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81549.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 81507, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult81549

namespace SemanticResult81552
def owner : Owner := ⟨.program ⟨214⟩, ⟨25527⟩⟩
def rawTerms : List Term := Proof.Events318.exact81552RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81552
def producerEvent : Nat := 81551
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81552.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 81507, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult81552

namespace SemanticResult81561
def owner : Owner := ⟨.program ⟨214⟩, ⟨12859⟩⟩
def rawTerms : List Term := Proof.Events318.exact81561RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81561
def producerEvent : Nat := 81560
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81561.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 81559 .coefficient), 81507, .finite 2116, .identity (.predecessor 0 81559 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult81561

namespace SemanticResult81563
def owner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rawTerms : List Term := Proof.Events318.exact81563RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81563
def producerEvent : Nat := 81562
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81563.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 81507, .large, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult81563

namespace SemanticResult81568
def owner : Owner := ⟨.program ⟨214⟩, ⟨12860⟩⟩
def rawTerms : List Term := Proof.Events318.exact81568RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81568
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81568.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge81567.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge81567.frameStart)
    (transferEvent := 81566) (owner := owner)
    (leftResult := 81563) (rightResult := 81561)
    (working := LeftOperatorMerge81567.working)
    (reconstruction := LeftOperatorMerge81567.reconstruction)
    (leftReference := .predecessor 0 81564 .coefficient) (rightReference := .predecessor 1 81565 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult81563.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult81561.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge81567.operationAgreement
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
end SemanticResult81568

namespace SemanticResult81571
def owner : Owner := ⟨.program ⟨214⟩, ⟨6757⟩⟩
def rawTerms : List Term := Proof.Events318.exact81571RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 81571
def producerEvent : Nat := 81570
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult81571.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 81507, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult81571

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
