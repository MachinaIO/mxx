import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard098
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard001
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard055
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard097

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult11512
def owner : Owner := ⟨.program ⟨214⟩, ⟨7853⟩⟩
def rawTerms : List Term := Proof.Events044.exact11512RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11512
def producerEvent : Nat := 11511
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11512.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 11509 .coefficient) (.value (.predecessor 1 11510 .coefficient)), 0, .finite 8192, .scale (.predecessor 0 11509 .coefficient) (.value (.predecessor 1 11510 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult11512

namespace SemanticResult11515
def owner : Owner := ⟨.program ⟨214⟩, ⟨73⟩⟩
def rawTerms : List Term := Proof.Events044.exact11515RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11515
def producerEvent : Nat := 11514
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11515.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 11513 .coefficient), 0, .finite 26, .identity (.predecessor 0 11513 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult11515

namespace SemanticResult11520
def owner : Owner := ⟨.program ⟨214⟩, ⟨14247⟩⟩
def rawTerms : List Term := Proof.Events045.exact11520RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11520
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11520.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge11519.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge11519.frameStart)
    (transferEvent := 11518) (owner := owner)
    (leftResult := 284) (rightResult := 6449)
    (working := LeftOperatorMerge11519.working)
    (reconstruction := LeftOperatorMerge11519.reconstruction)
    (leftReference := .predecessor 0 11516 .coefficient) (rightReference := .predecessor 1 11517 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult284.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6449.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge11519.operationAgreement
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
end SemanticResult11520

namespace SemanticResult11523
def owner : Owner := ⟨.program ⟨214⟩, ⟨6759⟩⟩
def rawTerms : List Term := Proof.Events045.exact11523RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11523
def producerEvent : Nat := 11522
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11523.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 11521 .coefficient), 0, .large, .identity (.predecessor 0 11521 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult11523

namespace SemanticResult11528
def owner : Owner := ⟨.program ⟨214⟩, ⟨7367⟩⟩
def rawTerms : List Term := Proof.Events045.exact11528RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11528
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11528.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge11527.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge11527.frameStart)
    (transferEvent := 11526) (owner := owner)
    (leftResult := 6314) (rightResult := 11523)
    (working := LeftOperatorMerge11527.working)
    (reconstruction := LeftOperatorMerge11527.reconstruction)
    (leftReference := .predecessor 0 11524 .coefficient) (rightReference := .predecessor 1 11525 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult6314.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11523.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge11527.operationAgreement
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
end SemanticResult11528

namespace SemanticResult11532
def owner : Owner := ⟨.program ⟨214⟩, ⟨14248⟩⟩
def rawTerms : List Term := Proof.Events045.exact11532RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11532
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11532.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 11529) (rightBinding := 11530)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7367⟩) (rightExpression := ⟨14247⟩)
    (transferEvent := 11531)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult11528.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11520.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult11532

namespace SemanticResult11538
def owner : Owner := ⟨.program ⟨214⟩, ⟨14249⟩⟩
def rawTerms : List Term := Proof.Events045.exact11538RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 11538
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11538.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 11535) (survivorTransfer := 11536)
    (survivorEvent := 11537) (resultEvent := resultEvent)
    (rightCoefficientProducer := 11514)
    (owner := owner) (leftOwner := SemanticResult11532.owner)
    (rightOwner := SemanticResult11515.owner)
    (leftResult := 11532) (rightResult := 11515)
    (leftBinding := 11533) (rightBinding := 11534)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14248⟩) (rightExpression := ⟨73⟩)
    (leftActual := SemanticResult11532.actual selector witness)
    (rightActual := SemanticResult11515.actual selector witness)
    (leftRaw := SemanticResult11532.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨73⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound11514.actual selector witness)
    (survivorMagnitude := LeftBound11536.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult11532.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11515.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11514.derived selector witness)
  · exact LeftBound11536.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult11538

namespace SemanticResult11548
def owner : Owner := ⟨.program ⟨214⟩, ⟨14250⟩⟩
def rawTerms : List Term := Proof.Events045.exact11548RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 11548
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11548.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge11544.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge11544.frameStart)
    (owner := owner) (leftOwner := SemanticResult11538.owner)
    (rightOwner := SemanticResult11512.owner)
    (leftResult := 11538) (rightResult := 11512)
    (leftActual := SemanticResult11538.actual selector witness)
    (rightActual := SemanticResult11512.actual selector witness)
    (leftRaw := SemanticResult11538.rawTerms)
    (rightRaw := SemanticResult11512.rawTerms)
    (working := LeftOperatorMerge11544.working)
    (leftBinding := 11539) (rightBinding := 11540)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14249⟩) (rightExpression := ⟨7853⟩)
    (coefficientTransfer := 11541) (summaryTransfer := 11543)
    (rightCoefficientProducer := 11511)
    (rightSummaryTransfer := 11542)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge11544.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound11511.actual selector witness)
    (summaryMagnitude := LeftBound11543.actual selector witness)
    (reconstruction := LeftOperatorMerge11544.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult11538.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11512.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11511.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound11511.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge11544.operationAgreement
  · exact LeftBound11543.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge11544.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 11545 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14243⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6779⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6779⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge11544.working
    [{ coefficient := (-1), key := LeftRelationMerge11545.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge11545.frameStart
      LeftRelationMerge11545.owner (.relation 11545) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge11545.deltas
    rows := LeftRelationMerge11545.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge11544.working LeftRelationMerge11545.source
        (relationContext LeftRelationMerge11545.source
          LeftRelationMerge11545.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge11544.working, LeftRelationMerge11545.deltas,
    LeftRelationMerge11545.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 11545)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨14250⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14243⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge11544.working) (working := relationWorking0)
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
end SemanticResult11548

namespace SemanticResult11554
def owner : Owner := ⟨.program ⟨214⟩, ⟨14251⟩⟩
def rawTerms : List Term := Proof.Events045.exact11554RawTerms
def summary : Bound := (.finite 95435392)
def resultEvent : Nat := 11554
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11554.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge11552.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult11548.owner)
    (rightOwner := SemanticResult11505.owner)
    (leftResult := 11548) (rightResult := 11505)
    (leftActual := SemanticResult11548.actual selector witness)
    (rightActual := SemanticResult11505.actual selector witness)
    (leftRaw := SemanticResult11548.rawTerms)
    (rightRaw := SemanticResult11505.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 14976) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 11549) (rightBinding := 11550)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14250⟩) (rightExpression := ⟨14246⟩)
    (coefficientTransfer := 11551) (summaryTransfer := 11553)
    (base := LeftOperatorMerge11552.base)
    (reconstruction := LeftOperatorMerge11552.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult11548.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11505.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge11552.operationAgreement
  · rfl
  · decide
end SemanticResult11554

namespace SemanticResult11564
def owner : Owner := ⟨.program ⟨214⟩, ⟨26087⟩⟩
def rawTerms : List Term := Proof.Events045.exact11564RawTerms
def summary : Bound := (.finite 350249415606272)
def resultEvent : Nat := 11564
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11564.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95435392, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge11560.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge11560.frameStart)
    (owner := owner) (leftOwner := SemanticResult11554.owner)
    (rightOwner := SemanticResult11471.owner)
    (leftResult := 11554) (rightResult := 11471)
    (leftActual := SemanticResult11554.actual selector witness)
    (rightActual := SemanticResult11471.actual selector witness)
    (leftRaw := SemanticResult11554.rawTerms)
    (rightRaw := SemanticResult11471.rawTerms)
    (working := LeftOperatorMerge11560.working)
    (leftBinding := 11555) (rightBinding := 11556)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14251⟩) (rightExpression := ⟨26086⟩)
    (coefficientTransfer := 11557) (summaryTransfer := 11559)
    (rightCoefficientProducer := 11470)
    (rightSummaryTransfer := 11558)
    (leftMaximum := ⟨95435392, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge11560.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority11470.actual selector witness)
    (summaryMagnitude := LeftBound11559.actual selector witness)
    (reconstruction := LeftOperatorMerge11560.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult11554.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11471.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11470.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority11470.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge11560.operationAgreement
  · exact LeftBound11559.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge11560.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 11561 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26086⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11485⟩⟩, ⟨.program ⟨214⟩, ⟨14243⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23592⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23592⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge11560.working
    [{ coefficient := (-1), key := LeftRelationMerge11561.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge11561.frameStart
      LeftRelationMerge11561.owner (.relation 11561) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge11561.deltas
    rows := LeftRelationMerge11561.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge11560.working LeftRelationMerge11561.source
        (relationContext LeftRelationMerge11561.source
          LeftRelationMerge11561.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge11560.working, LeftRelationMerge11561.deltas,
    LeftRelationMerge11561.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 11561)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨26087⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11485⟩⟩, ⟨.program ⟨214⟩, ⟨14243⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26086⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26086⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge11560.working) (working := relationWorking0)
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
end SemanticResult11564

namespace SemanticResult11567
def owner : Owner := ⟨.program ⟨214⟩, ⟨19544⟩⟩
def rawTerms : List Term := Proof.Events045.exact11567RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11567
def producerEvent : Nat := 11566
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11567.actual selector witness
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
end SemanticResult11567

namespace SemanticResult11571
def owner : Owner := ⟨.program ⟨214⟩, ⟨19546⟩⟩
def rawTerms : List Term := Proof.Events045.exact11571RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11571
def producerEvent : Nat := 11570
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11571.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 11568 .coefficient) (.value (.predecessor 1 11569 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 11568 .coefficient) (.value (.predecessor 1 11569 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult11571

namespace SemanticResult11649
def owner : Owner := ⟨.program ⟨214⟩, ⟨11485⟩⟩
def rawTerms : List Term := Proof.Events045.exact11649RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11649
def producerEvent : Nat := 11648
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11649.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 11626, .finite 18, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult11649

namespace SemanticResult11652
def owner : Owner := ⟨.program ⟨214⟩, ⟨14243⟩⟩
def rawTerms : List Term := Proof.Events045.exact11652RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11652
def producerEvent : Nat := 11651
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11652.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 11626, .finite 18, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult11652

namespace SemanticResult11657
def owner : Owner := ⟨.program ⟨214⟩, ⟨14244⟩⟩
def rawTerms : List Term := Proof.Events045.exact11657RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11657
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11657.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge11656.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge11656.frameStart)
    (transferEvent := 11655) (owner := owner)
    (leftResult := 11652) (rightResult := 11649)
    (working := LeftOperatorMerge11656.working)
    (reconstruction := LeftOperatorMerge11656.reconstruction)
    (leftReference := .predecessor 0 11653 .coefficient) (rightReference := .predecessor 1 11654 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult11652.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11649.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge11656.operationAgreement
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
end SemanticResult11657

namespace SemanticResult11668
def owner : Owner := ⟨.program ⟨214⟩, ⟨23592⟩⟩
def rawTerms : List Term := Proof.Events045.exact11668RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 11668
def producerEvent : Nat := 11667
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult11668.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 11626, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult11668

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
