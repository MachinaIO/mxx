import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard610
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard032
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard105
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard566
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard567
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard609

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult85636
def owner : Owner := ⟨.program ⟨214⟩, ⟨27650⟩⟩
def rawTerms : List Term := Proof.Events334.exact85636RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85636
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85636.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge85632.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge85632.frameStart)
    (transferEvent := 85631) (owner := owner)
    (leftResult := 85628) (rightResult := 85605)
    (working := LeftOperatorMerge85632.working)
    (reconstruction := LeftOperatorMerge85632.reconstruction)
    (leftReference := .predecessor 0 85629 .coefficient) (rightReference := .predecessor 1 85630 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult85628.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult85605.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge85632.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 85634 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24099⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24099⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge85632.working
    [{ coefficient := (-1), key := LeftRelationMerge85634.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge85634.frameStart
      LeftRelationMerge85634.owner (.relation 85634) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge85634.deltas
    rows := LeftRelationMerge85634.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge85632.working LeftRelationMerge85634.source
        (relationContext LeftRelationMerge85634.source
          LeftRelationMerge85634.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge85632.working, LeftRelationMerge85634.deltas,
    LeftRelationMerge85634.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 85634)
    (frameStart := 85554) (owner := ⟨.program ⟨214⟩, ⟨27650⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨15821⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge85632.working) (working := relationWorking0)
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
end SemanticResult85636

namespace SemanticResult85639
def owner : Owner := ⟨.program ⟨214⟩, ⟨15867⟩⟩
def rawTerms : List Term := Proof.Events334.exact85639RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85639
def producerEvent : Nat := 85638
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85639.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 85554, .finite 60, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult85639

namespace SemanticResult85644
def owner : Owner := ⟨.program ⟨214⟩, ⟨15868⟩⟩
def rawTerms : List Term := Proof.Events334.exact85644RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85644
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85644.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge85643.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge85643.frameStart)
    (transferEvent := 85642) (owner := owner)
    (leftResult := 85616) (rightResult := 85639)
    (working := LeftOperatorMerge85643.working)
    (reconstruction := LeftOperatorMerge85643.reconstruction)
    (leftReference := .predecessor 0 85640 .coefficient) (rightReference := .predecessor 1 85641 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult85616.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult85639.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge85643.operationAgreement
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
end SemanticResult85644

namespace SemanticResult85647
def owner : Owner := ⟨.program ⟨214⟩, ⟨6721⟩⟩
def rawTerms : List Term := Proof.Events334.exact85647RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85647
def producerEvent : Nat := 85646
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85647.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 85554, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult85647

namespace SemanticResult85651
def owner : Owner := ⟨.program ⟨214⟩, ⟨15869⟩⟩
def rawTerms : List Term := Proof.Events334.exact85651RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85651
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85651.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 85648) (rightBinding := 85649)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6721⟩) (rightExpression := ⟨15868⟩)
    (transferEvent := 85650)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult85647.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult85644.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult85651

namespace SemanticResult85655
def owner : Owner := ⟨.program ⟨214⟩, ⟨27654⟩⟩
def rawTerms : List Term := Proof.Events334.exact85655RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85655
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85655.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 85652) (rightBinding := 85653)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15869⟩) (rightExpression := ⟨27650⟩)
    (transferEvent := 85654)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult85651.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult85636.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult85655

namespace SemanticResult85664
def owner : Owner := ⟨.program ⟨214⟩, ⟨21259⟩⟩
def rawTerms : List Term := Proof.Events334.exact85664RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 85664
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85664.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge85499.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge85499.frameStart)
    (owner := owner) (leftOwner := SemanticResult80012.owner)
    (rightOwner := SemanticResult85493.owner)
    (leftResult := 80012) (rightResult := 85493)
    (leftActual := SemanticResult80012.actual selector witness)
    (rightActual := SemanticResult85493.actual selector witness)
    (leftRaw := SemanticResult80012.rawTerms)
    (rightRaw := SemanticResult85493.rawTerms)
    (working := LeftOperatorMerge85499.working)
    (leftBinding := 85494) (rightBinding := 85495)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5541⟩) (rightExpression := ⟨21258⟩)
    (coefficientTransfer := 85496) (summaryTransfer := 85498)
    (rightCoefficientProducer := 85492)
    (rightSummaryTransfer := 85497)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge85499.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound85492.actual selector witness)
    (summaryMagnitude := LeftBound85498.actual selector witness)
    (reconstruction := LeftOperatorMerge85499.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult80012.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult85493.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85492.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound85492.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge85499.operationAgreement
  · exact LeftBound85498.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge85499.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 85659 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24099⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6696⟩⟩, ⟨.program ⟨214⟩, ⟨27649⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15821⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24099⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge85499.working
    [{ coefficient := (1), key := LeftRelationMerge85659.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge85659.frameStart
      LeftRelationMerge85659.owner (.relation 85659) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge85659.deltas
    rows := LeftRelationMerge85659.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge85499.working LeftRelationMerge85659.source
        (relationContext LeftRelationMerge85659.source
          LeftRelationMerge85659.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge85499.working, LeftRelationMerge85659.deltas,
    LeftRelationMerge85659.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 85659)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨21259⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21256⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21256⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge85499.working) (working := relationWorking0)
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
end SemanticResult85664

namespace SemanticResult85671
def owner : Owner := ⟨.program ⟨214⟩, ⟨27652⟩⟩
def rawTerms : List Term := Proof.Events334.exact85671RawTerms
def summary : Bound := (.finite 1292046061494565744640)
def resultEvent : Nat := 85671
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85671.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge85668.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult85664.owner)
    (rightOwner := SemanticResult85486.owner)
    (leftResult := 85664) (rightResult := 85486)
    (leftActual := SemanticResult85664.actual selector witness)
    (rightActual := SemanticResult85486.actual selector witness)
    (leftRaw := SemanticResult85664.rawTerms)
    (rightRaw := SemanticResult85486.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292046059683262234624) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 85665) (rightBinding := 85666)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21259⟩) (rightExpression := ⟨27651⟩)
    (coefficientTransfer := 85667) (summaryTransfer := 85670)
    (base := LeftOperatorMerge85668.base)
    (reconstruction := LeftOperatorMerge85668.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult85664.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult85486.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge85668.operationAgreement
  · rfl
  · decide
end SemanticResult85671

namespace SemanticResult85678
def owner : Owner := ⟨.program ⟨214⟩, ⟨24036⟩⟩
def rawTerms : List Term := Proof.Events334.exact85678RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85678
def producerEvent : Nat := 85677
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85678.actual selector witness
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
end SemanticResult85678

namespace SemanticResult85681
def owner : Owner := ⟨.program ⟨214⟩, ⟨27432⟩⟩
def rawTerms : List Term := Proof.Events334.exact85681RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85681
def producerEvent : Nat := 85680
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85681.actual selector witness
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
end SemanticResult85681

namespace SemanticResult85688
def owner : Owner := ⟨.program ⟨214⟩, ⟨23500⟩⟩
def rawTerms : List Term := Proof.Events334.exact85688RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85688
def producerEvent : Nat := 85687
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85688.actual selector witness
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
end SemanticResult85688

namespace SemanticResult85691
def owner : Owner := ⟨.program ⟨214⟩, ⟨25912⟩⟩
def rawTerms : List Term := Proof.Events334.exact85691RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85691
def producerEvent : Nat := 85690
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85691.actual selector witness
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
end SemanticResult85691

namespace SemanticResult85696
def owner : Owner := ⟨.program ⟨214⟩, ⟨11302⟩⟩
def rawTerms : List Term := Proof.Events334.exact85696RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85696
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85696.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge85695.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge85695.frameStart)
    (transferEvent := 85694) (owner := owner)
    (leftResult := 4104) (rightResult := 79920)
    (working := LeftOperatorMerge85695.working)
    (reconstruction := LeftOperatorMerge85695.reconstruction)
    (leftReference := .predecessor 0 85692 .coefficient) (rightReference := .predecessor 1 85693 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4104.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79920.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge85695.operationAgreement
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
end SemanticResult85696

namespace SemanticResult85701
def owner : Owner := ⟨.program ⟨214⟩, ⟨7233⟩⟩
def rawTerms : List Term := Proof.Events334.exact85701RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85701
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85701.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge85700.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge85700.frameStart)
    (transferEvent := 85699) (owner := owner)
    (leftResult := 79790) (rightResult := 12484)
    (working := LeftOperatorMerge85700.working)
    (reconstruction := LeftOperatorMerge85700.reconstruction)
    (leftReference := .predecessor 0 85697 .coefficient) (rightReference := .predecessor 1 85698 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12484.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge85700.operationAgreement
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
end SemanticResult85701

namespace SemanticResult85705
def owner : Owner := ⟨.program ⟨214⟩, ⟨11303⟩⟩
def rawTerms : List Term := Proof.Events334.exact85705RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 85705
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85705.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 85702) (rightBinding := 85703)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7233⟩) (rightExpression := ⟨11302⟩)
    (transferEvent := 85704)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult85701.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult85696.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult85705

namespace SemanticResult85711
def owner : Owner := ⟨.program ⟨214⟩, ⟨11304⟩⟩
def rawTerms : List Term := Proof.Events334.exact85711RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 85711
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult85711.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 85708) (survivorTransfer := 85709)
    (survivorEvent := 85710) (resultEvent := resultEvent)
    (rightCoefficientProducer := 12475)
    (owner := owner) (leftOwner := SemanticResult85705.owner)
    (rightOwner := SemanticResult12476.owner)
    (leftResult := 85705) (rightResult := 12476)
    (leftBinding := 85706) (rightBinding := 85707)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11303⟩) (rightExpression := ⟨91⟩)
    (leftActual := SemanticResult85705.actual selector witness)
    (rightActual := SemanticResult12476.actual selector witness)
    (leftRaw := SemanticResult85705.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨91⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound12475.actual selector witness)
    (survivorMagnitude := LeftBound85709.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult85705.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12476.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12475.derived selector witness)
  · exact LeftBound85709.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult85711

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
