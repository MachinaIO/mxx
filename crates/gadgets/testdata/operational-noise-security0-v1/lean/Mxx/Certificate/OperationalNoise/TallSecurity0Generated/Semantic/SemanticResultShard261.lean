import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard261
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard049
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard050
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard161
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard165
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard260

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult35476
def owner : Owner := ⟨.program ⟨214⟩, ⟨6690⟩⟩
def rawTerms : List Term := Proof.Events138.exact35476RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 35476
def producerEvent : Nat := 35475
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35476.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 35406, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult35476

namespace SemanticResult35480
def owner : Owner := ⟨.program ⟨214⟩, ⟨14847⟩⟩
def rawTerms : List Term := Proof.Events138.exact35480RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 35480
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35480.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 35477) (rightBinding := 35478)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6690⟩) (rightExpression := ⟨14846⟩)
    (transferEvent := 35479)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35476.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35473.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35480

namespace SemanticResult35488
def owner : Owner := ⟨.program ⟨214⟩, ⟨26388⟩⟩
def rawTerms : List Term := Proof.Events138.exact35488RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 35488
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35488.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge35484.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge35484.frameStart)
    (transferEvent := 35483) (owner := owner)
    (leftResult := 35480) (rightResult := 35457)
    (working := LeftOperatorMerge35484.working)
    (reconstruction := LeftOperatorMerge35484.reconstruction)
    (leftReference := .predecessor 0 35481 .coefficient) (rightReference := .predecessor 1 35482 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35480.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35457.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge35484.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 35486 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26387⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14804⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23729⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23729⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge35484.working
    [{ coefficient := (-1), key := LeftRelationMerge35486.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge35486.frameStart
      LeftRelationMerge35486.owner (.relation 35486) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge35486.deltas
    rows := LeftRelationMerge35486.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge35484.working LeftRelationMerge35486.source
        (relationContext LeftRelationMerge35486.source
          LeftRelationMerge35486.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge35484.working, LeftRelationMerge35486.deltas,
    LeftRelationMerge35486.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 35486)
    (frameStart := 35406) (owner := ⟨.program ⟨214⟩, ⟨26388⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨14804⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26387⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26387⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge35484.working) (working := relationWorking0)
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
end SemanticResult35488

namespace SemanticResult35491
def owner : Owner := ⟨.program ⟨214⟩, ⟨14901⟩⟩
def rawTerms : List Term := Proof.Events138.exact35491RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 35491
def producerEvent : Nat := 35490
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35491.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 35406, .finite 2, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult35491

namespace SemanticResult35496
def owner : Owner := ⟨.program ⟨214⟩, ⟨14904⟩⟩
def rawTerms : List Term := Proof.Events138.exact35496RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 35496
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35496.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge35495.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge35495.frameStart)
    (transferEvent := 35494) (owner := owner)
    (leftResult := 35468) (rightResult := 35491)
    (working := LeftOperatorMerge35495.working)
    (reconstruction := LeftOperatorMerge35495.reconstruction)
    (leftReference := .predecessor 0 35492 .coefficient) (rightReference := .predecessor 1 35493 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35468.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35491.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge35495.operationAgreement
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
end SemanticResult35496

namespace SemanticResult35499
def owner : Owner := ⟨.program ⟨214⟩, ⟨6708⟩⟩
def rawTerms : List Term := Proof.Events138.exact35499RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 35499
def producerEvent : Nat := 35498
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35499.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 35406, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult35499

namespace SemanticResult35503
def owner : Owner := ⟨.program ⟨214⟩, ⟨14905⟩⟩
def rawTerms : List Term := Proof.Events138.exact35503RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 35503
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35503.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 35500) (rightBinding := 35501)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6708⟩) (rightExpression := ⟨14904⟩)
    (transferEvent := 35502)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35499.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35496.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35503

namespace SemanticResult35507
def owner : Owner := ⟨.program ⟨214⟩, ⟨26393⟩⟩
def rawTerms : List Term := Proof.Events138.exact35507RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 35507
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35507.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 35504) (rightBinding := 35505)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14905⟩) (rightExpression := ⟨26388⟩)
    (transferEvent := 35506)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35503.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35488.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35507

namespace SemanticResult35516
def owner : Owner := ⟨.program ⟨214⟩, ⟨20335⟩⟩
def rawTerms : List Term := Proof.Events138.exact35516RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 35516
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35516.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge35351.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge35351.frameStart)
    (owner := owner) (leftOwner := SemanticResult21512.owner)
    (rightOwner := SemanticResult35345.owner)
    (leftResult := 21512) (rightResult := 35345)
    (leftActual := SemanticResult21512.actual selector witness)
    (rightActual := SemanticResult35345.actual selector witness)
    (leftRaw := SemanticResult21512.rawTerms)
    (rightRaw := SemanticResult35345.rawTerms)
    (working := LeftOperatorMerge35351.working)
    (leftBinding := 35346) (rightBinding := 35347)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5559⟩) (rightExpression := ⟨20334⟩)
    (coefficientTransfer := 35348) (summaryTransfer := 35350)
    (rightCoefficientProducer := 35344)
    (rightSummaryTransfer := 35349)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge35351.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound35344.actual selector witness)
    (summaryMagnitude := LeftBound35350.actual selector witness)
    (reconstruction := LeftOperatorMerge35351.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21512.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35345.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35344.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound35344.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge35351.operationAgreement
  · exact LeftBound35350.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge35351.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 35511 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26387⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14804⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23729⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14901⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26387⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14804⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23729⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14901⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge35351.working
    [{ coefficient := (1), key := LeftRelationMerge35511.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge35511.frameStart
      LeftRelationMerge35511.owner (.relation 35511) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge35511.deltas
    rows := LeftRelationMerge35511.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge35351.working LeftRelationMerge35511.source
        (relationContext LeftRelationMerge35511.source
          LeftRelationMerge35511.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge35351.working, LeftRelationMerge35511.deltas,
    LeftRelationMerge35511.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 35511)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨20335⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20332⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20332⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge35351.working) (working := relationWorking0)
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
end SemanticResult35516

namespace SemanticResult35523
def owner : Owner := ⟨.program ⟨214⟩, ⟨26390⟩⟩
def rawTerms : List Term := Proof.Events138.exact35523RawTerms
def summary : Bound := (.finite 1291889174379421642752)
def resultEvent : Nat := 35523
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35523.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge35520.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35516.owner)
    (rightOwner := SemanticResult35338.owner)
    (leftResult := 35516) (rightResult := 35338)
    (leftActual := SemanticResult35516.actual selector witness)
    (rightActual := SemanticResult35338.actual selector witness)
    (leftRaw := SemanticResult35516.rawTerms)
    (rightRaw := SemanticResult35338.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1291889172568118132736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35517) (rightBinding := 35518)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20335⟩) (rightExpression := ⟨26389⟩)
    (coefficientTransfer := 35519) (summaryTransfer := 35522)
    (base := LeftOperatorMerge35520.base)
    (reconstruction := LeftOperatorMerge35520.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35516.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35338.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge35520.operationAgreement
  · rfl
  · decide
end SemanticResult35523

namespace SemanticResult35533
def owner : Owner := ⟨.program ⟨214⟩, ⟨26391⟩⟩
def rawTerms : List Term := Proof.Events138.exact35533RawTerms
def summary : Bound := (.finite 4741253940199267499646124032)
def resultEvent : Nat := 35533
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35533.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨1291889174379421642752, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge35529.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge35529.frameStart)
    (owner := owner) (leftOwner := SemanticResult35523.owner)
    (rightOwner := SemanticResult5859.owner)
    (leftResult := 35523) (rightResult := 5859)
    (leftActual := SemanticResult35523.actual selector witness)
    (rightActual := SemanticResult5859.actual selector witness)
    (leftRaw := SemanticResult35523.rawTerms)
    (rightRaw := SemanticResult5859.rawTerms)
    (working := LeftOperatorMerge35529.working)
    (leftBinding := 35524) (rightBinding := 35525)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26390⟩) (rightExpression := ⟨6680⟩)
    (coefficientTransfer := 35526) (summaryTransfer := 35528)
    (rightCoefficientProducer := 5858)
    (rightSummaryTransfer := 35527)
    (leftMaximum := ⟨1291889174379421642752, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge35529.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound5858.actual selector witness)
    (summaryMagnitude := LeftBound35528.actual selector witness)
    (reconstruction := LeftOperatorMerge35529.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35523.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5859.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5858.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound5858.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge35529.operationAgreement
  · exact LeftBound35528.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge35529.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 35531 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14901⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge35529.working
    [{ coefficient := (-1), key := LeftRelationMerge35531.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge35531.frameStart
      LeftRelationMerge35531.owner (.relation 35531) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge35531.deltas
    rows := LeftRelationMerge35531.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge35529.working LeftRelationMerge35531.source
        (relationContext LeftRelationMerge35531.source
          LeftRelationMerge35531.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge35529.working, LeftRelationMerge35531.deltas,
    LeftRelationMerge35531.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 35531)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨26391⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14901⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge35529.working) (working := relationWorking0)
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
end SemanticResult35533

namespace SemanticResult35538
def owner : Owner := ⟨.program ⟨214⟩, ⟨6629⟩⟩
def rawTerms : List Term := Proof.Events138.exact35538RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 35538
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35538.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge35537.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge35537.frameStart)
    (transferEvent := 35536) (owner := owner)
    (leftResult := 723) (rightResult := 21420)
    (working := LeftOperatorMerge35537.working)
    (reconstruction := LeftOperatorMerge35537.reconstruction)
    (leftReference := .predecessor 0 35534 .coefficient) (rightReference := .predecessor 1 35535 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult723.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge35537.operationAgreement
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
end SemanticResult35538

namespace SemanticResult35543
def owner : Owner := ⟨.program ⟨214⟩, ⟨7330⟩⟩
def rawTerms : List Term := Proof.Events138.exact35543RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 35543
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35543.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge35542.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge35542.frameStart)
    (transferEvent := 35541) (owner := owner)
    (leftResult := 21290) (rightResult := 5873)
    (working := LeftOperatorMerge35542.working)
    (reconstruction := LeftOperatorMerge35542.reconstruction)
    (leftReference := .predecessor 0 35539 .coefficient) (rightReference := .predecessor 1 35540 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5873.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge35542.operationAgreement
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
end SemanticResult35543

namespace SemanticResult35547
def owner : Owner := ⟨.program ⟨214⟩, ⟨7765⟩⟩
def rawTerms : List Term := Proof.Events138.exact35547RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 35547
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35547.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 35544) (rightBinding := 35545)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7330⟩) (rightExpression := ⟨6629⟩)
    (transferEvent := 35546)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35543.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35538.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult35547

namespace SemanticResult35553
def owner : Owner := ⟨.program ⟨214⟩, ⟨7766⟩⟩
def rawTerms : List Term := Proof.Events138.exact35553RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 35553
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35553.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 35550) (survivorTransfer := 35551)
    (survivorEvent := 35552) (resultEvent := resultEvent)
    (rightCoefficientProducer := 20907)
    (owner := owner) (leftOwner := SemanticResult35547.owner)
    (rightOwner := SemanticResult20908.owner)
    (leftResult := 35547) (rightResult := 20908)
    (leftBinding := 35548) (rightBinding := 35549)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7765⟩) (rightExpression := ⟨74⟩)
    (leftActual := SemanticResult35547.actual selector witness)
    (rightActual := SemanticResult20908.actual selector witness)
    (leftRaw := SemanticResult35547.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨74⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound20907.actual selector witness)
    (survivorMagnitude := LeftBound35551.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35547.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20908.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)
  · exact LeftBound35551.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult35553

namespace SemanticResult35560
def owner : Owner := ⟨.program ⟨214⟩, ⟨7811⟩⟩
def rawTerms : List Term := Proof.Events138.exact35560RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 35560
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult35560.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge35557.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult35553.owner)
    (rightOwner := SemanticResult35553.owner)
    (leftResult := 35553) (rightResult := 35553)
    (leftActual := SemanticResult35553.actual selector witness)
    (rightActual := SemanticResult35553.actual selector witness)
    (leftRaw := SemanticResult35553.rawTerms)
    (rightRaw := SemanticResult35553.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 35554) (rightBinding := 35555)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7766⟩) (rightExpression := ⟨7766⟩)
    (coefficientTransfer := 35556) (summaryTransfer := 35559)
    (base := LeftOperatorMerge35557.base)
    (reconstruction := LeftOperatorMerge35557.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult35553.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult35553.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge35557.operationAgreement
  · rfl
  · decide
end SemanticResult35560

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
