import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard710
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard038
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard105
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard667
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard709

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult99534
def owner : Owner := ⟨.program ⟨214⟩, ⟨27615⟩⟩
def rawTerms : List Term := Proof.Events388.exact99534RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99534
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99534.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99530.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge99530.frameStart)
    (transferEvent := 99529) (owner := owner)
    (leftResult := 99526) (rightResult := 99503)
    (working := LeftOperatorMerge99530.working)
    (reconstruction := LeftOperatorMerge99530.reconstruction)
    (leftReference := .predecessor 0 99527 .coefficient) (rightReference := .predecessor 1 99528 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult99526.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99503.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge99530.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 99532 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27614⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15811⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24090⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24090⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge99530.working
    [{ coefficient := (-1), key := LeftRelationMerge99532.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge99532.frameStart
      LeftRelationMerge99532.owner (.relation 99532) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge99532.deltas
    rows := LeftRelationMerge99532.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge99530.working LeftRelationMerge99532.source
        (relationContext LeftRelationMerge99532.source
          LeftRelationMerge99532.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge99530.working, LeftRelationMerge99532.deltas,
    LeftRelationMerge99532.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 99532)
    (frameStart := 99464) (owner := ⟨.program ⟨214⟩, ⟨27615⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨15811⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27614⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27614⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge99530.working) (working := relationWorking0)
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
end SemanticResult99534

namespace SemanticResult99537
def owner : Owner := ⟨.program ⟨214⟩, ⟨15860⟩⟩
def rawTerms : List Term := Proof.Events388.exact99537RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99537
def producerEvent : Nat := 99536
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99537.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 99464, .finite 60, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult99537

namespace SemanticResult99542
def owner : Owner := ⟨.program ⟨214⟩, ⟨15861⟩⟩
def rawTerms : List Term := Proof.Events388.exact99542RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99542
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99542.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99541.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge99541.frameStart)
    (transferEvent := 99540) (owner := owner)
    (leftResult := 99514) (rightResult := 99537)
    (working := LeftOperatorMerge99541.working)
    (reconstruction := LeftOperatorMerge99541.reconstruction)
    (leftReference := .predecessor 0 99538 .coefficient) (rightReference := .predecessor 1 99539 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult99514.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99537.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge99541.operationAgreement
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
end SemanticResult99542

namespace SemanticResult99545
def owner : Owner := ⟨.program ⟨214⟩, ⟨6721⟩⟩
def rawTerms : List Term := Proof.Events388.exact99545RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99545
def producerEvent : Nat := 99544
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99545.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 99464, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult99545

namespace SemanticResult99549
def owner : Owner := ⟨.program ⟨214⟩, ⟨15862⟩⟩
def rawTerms : List Term := Proof.Events388.exact99549RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99549
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99549.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 99546) (rightBinding := 99547)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6721⟩) (rightExpression := ⟨15861⟩)
    (transferEvent := 99548)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99545.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99542.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99549

namespace SemanticResult99553
def owner : Owner := ⟨.program ⟨214⟩, ⟨27619⟩⟩
def rawTerms : List Term := Proof.Events388.exact99553RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99553
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99553.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 99550) (rightBinding := 99551)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15862⟩) (rightExpression := ⟨27615⟩)
    (transferEvent := 99552)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99549.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99534.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99553

namespace SemanticResult99562
def owner : Owner := ⟨.program ⟨214⟩, ⟨21248⟩⟩
def rawTerms : List Term := Proof.Events388.exact99562RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 99562
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99562.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99421.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge99421.frameStart)
    (owner := owner) (leftOwner := SemanticResult94462.owner)
    (rightOwner := SemanticResult99415.owner)
    (leftResult := 94462) (rightResult := 99415)
    (leftActual := SemanticResult94462.actual selector witness)
    (rightActual := SemanticResult99415.actual selector witness)
    (leftRaw := SemanticResult94462.rawTerms)
    (rightRaw := SemanticResult99415.rawTerms)
    (working := LeftOperatorMerge99421.working)
    (leftBinding := 99416) (rightBinding := 99417)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5509⟩) (rightExpression := ⟨21247⟩)
    (coefficientTransfer := 99418) (summaryTransfer := 99420)
    (rightCoefficientProducer := 99414)
    (rightSummaryTransfer := 99419)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge99421.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound99414.actual selector witness)
    (summaryMagnitude := LeftBound99420.actual selector witness)
    (reconstruction := LeftOperatorMerge99421.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94462.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99415.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99414.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound99414.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge99421.operationAgreement
  · exact LeftBound99420.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99421.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 99557 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27614⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15811⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24090⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27614⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15811⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24090⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge99421.working
    [{ coefficient := (1), key := LeftRelationMerge99557.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge99557.frameStart
      LeftRelationMerge99557.owner (.relation 99557) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge99557.deltas
    rows := LeftRelationMerge99557.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge99421.working LeftRelationMerge99557.source
        (relationContext LeftRelationMerge99557.source
          LeftRelationMerge99557.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge99421.working, LeftRelationMerge99557.deltas,
    LeftRelationMerge99557.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 99557)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨21248⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21245⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21245⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge99421.working) (working := relationWorking0)
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
end SemanticResult99562

namespace SemanticResult99569
def owner : Owner := ⟨.program ⟨214⟩, ⟨27617⟩⟩
def rawTerms : List Term := Proof.Events388.exact99569RawTerms
def summary : Bound := (.finite 1292046061494565744640)
def resultEvent : Nat := 99569
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99569.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge99566.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult99562.owner)
    (rightOwner := SemanticResult99408.owner)
    (leftResult := 99562) (rightResult := 99408)
    (leftActual := SemanticResult99562.actual selector witness)
    (rightActual := SemanticResult99408.actual selector witness)
    (leftRaw := SemanticResult99562.rawTerms)
    (rightRaw := SemanticResult99408.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292046059683262234624) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 99563) (rightBinding := 99564)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21248⟩) (rightExpression := ⟨27616⟩)
    (coefficientTransfer := 99565) (summaryTransfer := 99568)
    (base := LeftOperatorMerge99566.base)
    (reconstruction := LeftOperatorMerge99566.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99562.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99408.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge99566.operationAgreement
  · rfl
  · decide
end SemanticResult99569

namespace SemanticResult99576
def owner : Owner := ⟨.program ⟨214⟩, ⟨24027⟩⟩
def rawTerms : List Term := Proof.Events388.exact99576RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99576
def producerEvent : Nat := 99575
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99576.actual selector witness
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
end SemanticResult99576

namespace SemanticResult99579
def owner : Owner := ⟨.program ⟨214⟩, ⟨27397⟩⟩
def rawTerms : List Term := Proof.Events388.exact99579RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99579
def producerEvent : Nat := 99578
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99579.actual selector witness
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
end SemanticResult99579

namespace SemanticResult99586
def owner : Owner := ⟨.program ⟨214⟩, ⟨23494⟩⟩
def rawTerms : List Term := Proof.Events389.exact99586RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99586
def producerEvent : Nat := 99585
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99586.actual selector witness
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
end SemanticResult99586

namespace SemanticResult99589
def owner : Owner := ⟨.program ⟨214⟩, ⟨25899⟩⟩
def rawTerms : List Term := Proof.Events389.exact99589RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99589
def producerEvent : Nat := 99588
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99589.actual selector witness
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
end SemanticResult99589

namespace SemanticResult99594
def owner : Owner := ⟨.program ⟨214⟩, ⟨11290⟩⟩
def rawTerms : List Term := Proof.Events389.exact99594RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99594
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99594.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99593.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge99593.frameStart)
    (transferEvent := 99592) (owner := owner)
    (leftResult := 4842) (rightResult := 32)
    (working := LeftOperatorMerge99593.working)
    (reconstruction := LeftOperatorMerge99593.reconstruction)
    (leftReference := .predecessor 0 99590 .coefficient) (rightReference := .predecessor 1 99591 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4842.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge99593.operationAgreement
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
end SemanticResult99594

namespace SemanticResult99599
def owner : Owner := ⟨.program ⟨214⟩, ⟨7114⟩⟩
def rawTerms : List Term := Proof.Events389.exact99599RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99599
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99599.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99598.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge99598.frameStart)
    (transferEvent := 99597) (owner := owner)
    (leftResult := 27) (rightResult := 12484)
    (working := LeftOperatorMerge99598.working)
    (reconstruction := LeftOperatorMerge99598.reconstruction)
    (leftReference := .predecessor 0 99595 .coefficient) (rightReference := .predecessor 1 99596 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12484.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge99598.operationAgreement
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
end SemanticResult99599

namespace SemanticResult99603
def owner : Owner := ⟨.program ⟨214⟩, ⟨11291⟩⟩
def rawTerms : List Term := Proof.Events389.exact99603RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99603
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99603.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 99600) (rightBinding := 99601)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7114⟩) (rightExpression := ⟨11290⟩)
    (transferEvent := 99602)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99599.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99594.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99603

namespace SemanticResult99609
def owner : Owner := ⟨.program ⟨214⟩, ⟨11292⟩⟩
def rawTerms : List Term := Proof.Events389.exact99609RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 99609
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99609.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 99606) (survivorTransfer := 99607)
    (survivorEvent := 99608) (resultEvent := resultEvent)
    (rightCoefficientProducer := 12475)
    (owner := owner) (leftOwner := SemanticResult99603.owner)
    (rightOwner := SemanticResult12476.owner)
    (leftResult := 99603) (rightResult := 12476)
    (leftBinding := 99604) (rightBinding := 99605)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11291⟩) (rightExpression := ⟨91⟩)
    (leftActual := SemanticResult99603.actual selector witness)
    (rightActual := SemanticResult12476.actual selector witness)
    (leftRaw := SemanticResult99603.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨91⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound12475.actual selector witness)
    (survivorMagnitude := LeftBound99607.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99603.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12476.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12475.derived selector witness)
  · exact LeftBound99607.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult99609

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
