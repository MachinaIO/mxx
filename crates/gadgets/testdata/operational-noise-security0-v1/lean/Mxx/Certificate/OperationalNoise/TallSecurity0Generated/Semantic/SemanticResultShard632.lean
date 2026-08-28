import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard632
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard567
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard610
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard614
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard617
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard621
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard625
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard628
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard631

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult88504
def owner : Owner := ⟨.program ⟨214⟩, ⟨6690⟩⟩
def rawTerms : List Term := Proof.Events345.exact88504RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 88504
def producerEvent : Nat := 88503
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88504.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 88434, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult88504

namespace SemanticResult88508
def owner : Owner := ⟨.program ⟨214⟩, ⟨14835⟩⟩
def rawTerms : List Term := Proof.Events345.exact88508RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 88508
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88508.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 88505) (rightBinding := 88506)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6690⟩) (rightExpression := ⟨14834⟩)
    (transferEvent := 88507)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult88504.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult88501.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult88508

namespace SemanticResult88516
def owner : Owner := ⟨.program ⟨214⟩, ⟨26359⟩⟩
def rawTerms : List Term := Proof.Events345.exact88516RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 88516
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88516.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge88512.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge88512.frameStart)
    (transferEvent := 88511) (owner := owner)
    (leftResult := 88508) (rightResult := 88485)
    (working := LeftOperatorMerge88512.working)
    (reconstruction := LeftOperatorMerge88512.reconstruction)
    (leftReference := .predecessor 0 88509 .coefficient) (rightReference := .predecessor 1 88510 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult88508.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult88485.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge88512.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 88514 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14792⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23721⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23721⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge88512.working
    [{ coefficient := (-1), key := LeftRelationMerge88514.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge88514.frameStart
      LeftRelationMerge88514.owner (.relation 88514) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge88514.deltas
    rows := LeftRelationMerge88514.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge88512.working LeftRelationMerge88514.source
        (relationContext LeftRelationMerge88514.source
          LeftRelationMerge88514.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge88512.working, LeftRelationMerge88514.deltas,
    LeftRelationMerge88514.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 88514)
    (frameStart := 88434) (owner := ⟨.program ⟨214⟩, ⟨26359⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨14792⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge88512.working) (working := relationWorking0)
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
end SemanticResult88516

namespace SemanticResult88519
def owner : Owner := ⟨.program ⟨214⟩, ⟨15265⟩⟩
def rawTerms : List Term := Proof.Events345.exact88519RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 88519
def producerEvent : Nat := 88518
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88519.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 88434, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult88519

namespace SemanticResult88524
def owner : Owner := ⟨.program ⟨214⟩, ⟨15266⟩⟩
def rawTerms : List Term := Proof.Events345.exact88524RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 88524
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88524.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge88523.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge88523.frameStart)
    (transferEvent := 88522) (owner := owner)
    (leftResult := 88496) (rightResult := 88519)
    (working := LeftOperatorMerge88523.working)
    (reconstruction := LeftOperatorMerge88523.reconstruction)
    (leftReference := .predecessor 0 88520 .coefficient) (rightReference := .predecessor 1 88521 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult88496.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult88519.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge88523.operationAgreement
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
end SemanticResult88524

namespace SemanticResult88527
def owner : Owner := ⟨.program ⟨214⟩, ⟨6709⟩⟩
def rawTerms : List Term := Proof.Events345.exact88527RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 88527
def producerEvent : Nat := 88526
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88527.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 88434, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult88527

namespace SemanticResult88531
def owner : Owner := ⟨.program ⟨214⟩, ⟨15267⟩⟩
def rawTerms : List Term := Proof.Events345.exact88531RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 88531
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88531.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 88528) (rightBinding := 88529)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6709⟩) (rightExpression := ⟨15266⟩)
    (transferEvent := 88530)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult88527.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult88524.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult88531

namespace SemanticResult88535
def owner : Owner := ⟨.program ⟨214⟩, ⟨26362⟩⟩
def rawTerms : List Term := Proof.Events345.exact88535RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 88535
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88535.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 88532) (rightBinding := 88533)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15267⟩) (rightExpression := ⟨26359⟩)
    (transferEvent := 88534)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult88531.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult88516.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult88535

namespace SemanticResult88544
def owner : Owner := ⟨.program ⟨214⟩, ⟨20395⟩⟩
def rawTerms : List Term := Proof.Events345.exact88544RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 88544
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88544.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge88379.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge88379.frameStart)
    (owner := owner) (leftOwner := SemanticResult80012.owner)
    (rightOwner := SemanticResult88373.owner)
    (leftResult := 80012) (rightResult := 88373)
    (leftActual := SemanticResult80012.actual selector witness)
    (rightActual := SemanticResult88373.actual selector witness)
    (leftRaw := SemanticResult80012.rawTerms)
    (rightRaw := SemanticResult88373.rawTerms)
    (working := LeftOperatorMerge88379.working)
    (leftBinding := 88374) (rightBinding := 88375)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5541⟩) (rightExpression := ⟨20394⟩)
    (coefficientTransfer := 88376) (summaryTransfer := 88378)
    (rightCoefficientProducer := 88372)
    (rightSummaryTransfer := 88377)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge88379.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound88372.actual selector witness)
    (summaryMagnitude := LeftBound88378.actual selector witness)
    (reconstruction := LeftOperatorMerge88379.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult80012.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult88373.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88372.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound88372.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge88379.operationAgreement
  · exact LeftBound88378.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge88379.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 88539 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨14792⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23721⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26358⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14792⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23721⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15265⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge88379.working
    [{ coefficient := (1), key := LeftRelationMerge88539.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge88539.frameStart
      LeftRelationMerge88539.owner (.relation 88539) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge88539.deltas
    rows := LeftRelationMerge88539.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge88379.working LeftRelationMerge88539.source
        (relationContext LeftRelationMerge88539.source
          LeftRelationMerge88539.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge88379.working, LeftRelationMerge88539.deltas,
    LeftRelationMerge88539.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 88539)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨20395⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20392⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20392⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge88379.working) (working := relationWorking0)
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
end SemanticResult88544

namespace SemanticResult88551
def owner : Owner := ⟨.program ⟨214⟩, ⟨26361⟩⟩
def rawTerms : List Term := Proof.Events345.exact88551RawTerms
def summary : Bound := (.finite 1291889174379421642752)
def resultEvent : Nat := 88551
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88551.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge88548.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult88544.owner)
    (rightOwner := SemanticResult88366.owner)
    (leftResult := 88544) (rightResult := 88366)
    (leftActual := SemanticResult88544.actual selector witness)
    (rightActual := SemanticResult88366.actual selector witness)
    (leftRaw := SemanticResult88544.rawTerms)
    (rightRaw := SemanticResult88366.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1291889172568118132736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 88545) (rightBinding := 88546)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20395⟩) (rightExpression := ⟨26360⟩)
    (coefficientTransfer := 88547) (summaryTransfer := 88550)
    (base := LeftOperatorMerge88548.base)
    (reconstruction := LeftOperatorMerge88548.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult88544.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult88366.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge88548.operationAgreement
  · rfl
  · decide
end SemanticResult88551

namespace SemanticResult88556
def owner : Owner := ⟨.program ⟨214⟩, ⟨26568⟩⟩
def rawTerms : List Term := Proof.Events345.exact88556RawTerms
def summary : Bound := (.finite 2583789554981353578496)
def resultEvent : Nat := 88556
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88556.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult88551.owner)
    (rightOwner := SemanticResult88071.owner)
    (leftResult := 88551) (rightResult := 88071)
    (leftActual := SemanticResult88551.actual selector witness)
    (rightActual := SemanticResult88071.actual selector witness)
    (leftRaw := SemanticResult88551.rawTerms)
    (rightRaw := SemanticResult88071.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1291889174379421642752)
    (rightMaximum := 1291900380601931935744) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 88552) (rightBinding := 88553)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26361⟩) (rightExpression := ⟨26567⟩)
    (transferEvent := 88554) (summaryTransferEvent := 88555)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult88551.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult88071.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult88556

namespace SemanticResult88561
def owner : Owner := ⟨.program ⟨214⟩, ⟨26785⟩⟩
def rawTerms : List Term := Proof.Events345.exact88561RawTerms
def summary : Bound := (.finite 3875701141805795807232)
def resultEvent : Nat := 88561
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88561.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult88556.owner)
    (rightOwner := SemanticResult87591.owner)
    (leftResult := 88556) (rightResult := 87591)
    (leftActual := SemanticResult88556.actual selector witness)
    (rightActual := SemanticResult87591.actual selector witness)
    (leftRaw := SemanticResult88556.rawTerms)
    (rightRaw := SemanticResult87591.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2583789554981353578496)
    (rightMaximum := 1291911586824442228736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 88557) (rightBinding := 88558)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26568⟩) (rightExpression := ⟨26784⟩)
    (transferEvent := 88559) (summaryTransferEvent := 88560)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult88556.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult87591.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult88561

namespace SemanticResult88566
def owner : Owner := ⟨.program ⟨214⟩, ⟨27002⟩⟩
def rawTerms : List Term := Proof.Events345.exact88566RawTerms
def summary : Bound := (.finite 5167635141075258621952)
def resultEvent : Nat := 88566
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88566.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult88561.owner)
    (rightOwner := SemanticResult87111.owner)
    (leftResult := 88561) (rightResult := 87111)
    (leftActual := SemanticResult88561.actual selector witness)
    (rightActual := SemanticResult87111.actual selector witness)
    (leftRaw := SemanticResult88561.rawTerms)
    (rightRaw := SemanticResult87111.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3875701141805795807232)
    (rightMaximum := 1291933999269462814720) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 88562) (rightBinding := 88563)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26785⟩) (rightExpression := ⟨27001⟩)
    (transferEvent := 88564) (summaryTransferEvent := 88565)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult88561.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult87111.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult88566

namespace SemanticResult88571
def owner : Owner := ⟨.program ⟨214⟩, ⟨27219⟩⟩
def rawTerms : List Term := Proof.Events345.exact88571RawTerms
def summary : Bound := (.finite 6459613965234762608640)
def resultEvent : Nat := 88571
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88571.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult88566.owner)
    (rightOwner := SemanticResult86631.owner)
    (leftResult := 88566) (rightResult := 86631)
    (leftActual := SemanticResult88566.actual selector witness)
    (rightActual := SemanticResult86631.actual selector witness)
    (leftRaw := SemanticResult88566.rawTerms)
    (rightRaw := SemanticResult86631.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5167635141075258621952)
    (rightMaximum := 1291978824159503986688) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 88567) (rightBinding := 88568)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27002⟩) (rightExpression := ⟨27218⟩)
    (transferEvent := 88569) (summaryTransferEvent := 88570)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult88566.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult86631.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult88571

namespace SemanticResult88576
def owner : Owner := ⟨.program ⟨214⟩, ⟨27436⟩⟩
def rawTerms : List Term := Proof.Events346.exact88576RawTerms
def summary : Bound := (.finite 7751615201839287181312)
def resultEvent : Nat := 88576
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88576.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult88571.owner)
    (rightOwner := SemanticResult86151.owner)
    (leftResult := 88571) (rightResult := 86151)
    (leftActual := SemanticResult88571.actual selector witness)
    (rightActual := SemanticResult86151.actual selector witness)
    (leftRaw := SemanticResult88571.rawTerms)
    (rightRaw := SemanticResult86151.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 6459613965234762608640)
    (rightMaximum := 1292001236604524572672) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 88572) (rightBinding := 88573)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27219⟩) (rightExpression := ⟨27435⟩)
    (transferEvent := 88574) (summaryTransferEvent := 88575)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult88571.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult86151.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult88576

namespace SemanticResult88581
def owner : Owner := ⟨.program ⟨214⟩, ⟨27653⟩⟩
def rawTerms : List Term := Proof.Events346.exact88581RawTerms
def summary : Bound := (.finite 9043661263333852925952)
def resultEvent : Nat := 88581
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult88581.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult88576.owner)
    (rightOwner := SemanticResult85671.owner)
    (leftResult := 88576) (rightResult := 85671)
    (leftActual := SemanticResult88576.actual selector witness)
    (rightActual := SemanticResult85671.actual selector witness)
    (leftRaw := SemanticResult88576.rawTerms)
    (rightRaw := SemanticResult85671.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 7751615201839287181312)
    (rightMaximum := 1292046061494565744640) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 88577) (rightBinding := 88578)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27436⟩) (rightExpression := ⟨27652⟩)
    (transferEvent := 88579) (summaryTransferEvent := 88580)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult88576.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult85671.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult88581

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
