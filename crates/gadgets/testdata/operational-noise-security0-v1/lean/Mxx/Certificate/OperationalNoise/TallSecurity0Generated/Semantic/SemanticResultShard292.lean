import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard292
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard265
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard290
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard291

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult39632
def owner : Owner := ⟨.program ⟨214⟩, ⟨6757⟩⟩
def rawTerms : List Term := Proof.Events154.exact39632RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39632
def producerEvent : Nat := 39631
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39632.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 39566, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult39632

namespace SemanticResult39638
def owner : Owner := ⟨.program ⟨214⟩, ⟨7861⟩⟩
def rawTerms : List Term := Proof.Events154.exact39638RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39638
def producerEvent : Nat := 39637
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39638.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 39566, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult39638

namespace SemanticResult39642
def owner : Owner := ⟨.program ⟨214⟩, ⟨7862⟩⟩
def rawTerms : List Term := Proof.Events154.exact39642RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39642
def producerEvent : Nat := 39641
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39642.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 39639 .coefficient) (.value (.predecessor 1 39640 .coefficient)), 39566, .finite 8192, .scale (.predecessor 0 39639 .coefficient) (.value (.predecessor 1 39640 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult39642

namespace SemanticResult39645
def owner : Owner := ⟨.program ⟨214⟩, ⟨6763⟩⟩
def rawTerms : List Term := Proof.Events154.exact39645RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39645
def producerEvent : Nat := 39644
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39645.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 39643 .coefficient), 39566, .large, .identity (.predecessor 0 39643 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult39645

namespace SemanticResult39650
def owner : Owner := ⟨.program ⟨214⟩, ⟨7863⟩⟩
def rawTerms : List Term := Proof.Events154.exact39650RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39650
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39650.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39649.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge39649.frameStart)
    (transferEvent := 39648) (owner := owner)
    (leftResult := 39645) (rightResult := 39642)
    (working := LeftOperatorMerge39649.working)
    (reconstruction := LeftOperatorMerge39649.reconstruction)
    (leftReference := .predecessor 0 39646 .coefficient) (rightReference := .predecessor 1 39647 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult39645.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39642.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge39649.operationAgreement
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
end SemanticResult39650

namespace SemanticResult39654
def owner : Owner := ⟨.program ⟨214⟩, ⟨11868⟩⟩
def rawTerms : List Term := Proof.Events154.exact39654RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39654
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39654.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 39651) (rightBinding := 39652)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7863⟩) (rightExpression := ⟨11867⟩)
    (transferEvent := 39653)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult39650.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39627.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult39654

namespace SemanticResult39662
def owner : Owner := ⟨.program ⟨214⟩, ⟨25155⟩⟩
def rawTerms : List Term := Proof.Events154.exact39662RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39662
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39662.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39658.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge39658.frameStart)
    (transferEvent := 39657) (owner := owner)
    (leftResult := 39654) (rightResult := 39611)
    (working := LeftOperatorMerge39658.working)
    (reconstruction := LeftOperatorMerge39658.reconstruction)
    (leftReference := .predecessor 0 39655 .coefficient) (rightReference := .predecessor 1 39656 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult39654.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39611.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge39658.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 39660 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25152⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9620⟩⟩, ⟨.program ⟨214⟩, ⟨11777⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23084⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23084⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge39658.working
    [{ coefficient := (-1), key := LeftRelationMerge39660.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge39660.frameStart
      LeftRelationMerge39660.owner (.relation 39660) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge39660.deltas
    rows := LeftRelationMerge39660.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge39658.working LeftRelationMerge39660.source
        (relationContext LeftRelationMerge39660.source
          LeftRelationMerge39660.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge39658.working, LeftRelationMerge39660.deltas,
    LeftRelationMerge39660.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 39660)
    (frameStart := 39566) (owner := ⟨.program ⟨214⟩, ⟨25155⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨9620⟩⟩, ⟨.program ⟨214⟩, ⟨11777⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25152⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25152⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge39658.working) (working := relationWorking0)
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
end SemanticResult39662

namespace SemanticResult39665
def owner : Owner := ⟨.program ⟨214⟩, ⟨16270⟩⟩
def rawTerms : List Term := Proof.Events154.exact39665RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39665
def producerEvent : Nat := 39664
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39665.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 39566, .finite 30, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult39665

namespace SemanticResult39670
def owner : Owner := ⟨.program ⟨214⟩, ⟨16272⟩⟩
def rawTerms : List Term := Proof.Events154.exact39670RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39670
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39670.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39669.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge39669.frameStart)
    (transferEvent := 39668) (owner := owner)
    (leftResult := 39622) (rightResult := 39665)
    (working := LeftOperatorMerge39669.working)
    (reconstruction := LeftOperatorMerge39669.reconstruction)
    (leftReference := .predecessor 0 39666 .coefficient) (rightReference := .predecessor 1 39667 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult39622.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39665.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge39669.operationAgreement
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
end SemanticResult39670

namespace SemanticResult39673
def owner : Owner := ⟨.program ⟨214⟩, ⟨6700⟩⟩
def rawTerms : List Term := Proof.Events154.exact39673RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39673
def producerEvent : Nat := 39672
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39673.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 39566, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult39673

namespace SemanticResult39677
def owner : Owner := ⟨.program ⟨214⟩, ⟨16273⟩⟩
def rawTerms : List Term := Proof.Events154.exact39677RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39677
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39677.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 39674) (rightBinding := 39675)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6700⟩) (rightExpression := ⟨16272⟩)
    (transferEvent := 39676)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult39673.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39670.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult39677

namespace SemanticResult39681
def owner : Owner := ⟨.program ⟨214⟩, ⟨25156⟩⟩
def rawTerms : List Term := Proof.Events155.exact39681RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39681
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39681.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 39678) (rightBinding := 39679)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16273⟩) (rightExpression := ⟨25155⟩)
    (transferEvent := 39680)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult39677.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39662.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult39681

namespace SemanticResult39690
def owner : Owner := ⟨.program ⟨214⟩, ⟨19755⟩⟩
def rawTerms : List Term := Proof.Events155.exact39690RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 39690
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39690.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39517.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge39517.frameStart)
    (owner := owner) (leftOwner := SemanticResult36137.owner)
    (rightOwner := SemanticResult39511.owner)
    (leftResult := 36137) (rightResult := 39511)
    (leftActual := SemanticResult36137.actual selector witness)
    (rightActual := SemanticResult39511.actual selector witness)
    (leftRaw := SemanticResult36137.rawTerms)
    (rightRaw := SemanticResult39511.rawTerms)
    (working := LeftOperatorMerge39517.working)
    (leftBinding := 39512) (rightBinding := 39513)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5553⟩) (rightExpression := ⟨19754⟩)
    (coefficientTransfer := 39514) (summaryTransfer := 39516)
    (rightCoefficientProducer := 39510)
    (rightSummaryTransfer := 39515)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge39517.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound39510.actual selector witness)
    (summaryMagnitude := LeftBound39516.actual selector witness)
    (reconstruction := LeftOperatorMerge39517.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36137.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39511.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39510.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound39510.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge39517.operationAgreement
  · exact LeftBound39516.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39517.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 39685 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25152⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9620⟩⟩, ⟨.program ⟨214⟩, ⟨11777⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23084⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25152⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9620⟩⟩, ⟨.program ⟨214⟩, ⟨11777⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23084⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge39517.working
    [{ coefficient := (1), key := LeftRelationMerge39685.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge39685.frameStart
      LeftRelationMerge39685.owner (.relation 39685) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge39685.deltas
    rows := LeftRelationMerge39685.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge39517.working LeftRelationMerge39685.source
        (relationContext LeftRelationMerge39685.source
          LeftRelationMerge39685.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge39517.working, LeftRelationMerge39685.deltas,
    LeftRelationMerge39685.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 39685)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨19755⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19752⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19752⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge39517.working) (working := relationWorking0)
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
end SemanticResult39690

namespace SemanticResult39697
def owner : Owner := ⟨.program ⟨214⟩, ⟨25154⟩⟩
def rawTerms : List Term := Proof.Events155.exact39697RawTerms
def summary : Bound := (.finite 352097360556032)
def resultEvent : Nat := 39697
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39697.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge39694.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult39690.owner)
    (rightOwner := SemanticResult39504.owner)
    (leftResult := 39690) (rightResult := 39504)
    (leftActual := SemanticResult39690.actual selector witness)
    (rightActual := SemanticResult39504.actual selector witness)
    (leftRaw := SemanticResult39690.rawTerms)
    (rightRaw := SemanticResult39504.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 350286057046016) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 39691) (rightBinding := 39692)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨19755⟩) (rightExpression := ⟨25153⟩)
    (coefficientTransfer := 39693) (summaryTransfer := 39696)
    (base := LeftOperatorMerge39694.base)
    (reconstruction := LeftOperatorMerge39694.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult39690.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39504.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge39694.operationAgreement
  · rfl
  · decide
end SemanticResult39697

namespace SemanticResult39707
def owner : Owner := ⟨.program ⟨214⟩, ⟨28545⟩⟩
def rawTerms : List Term := Proof.Events155.exact39707RawTerms
def summary : Bound := (.finite 1292202946798406336512)
def resultEvent : Nat := 39707
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39707.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨352097360556032, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39703.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge39703.frameStart)
    (owner := owner) (leftOwner := SemanticResult39697.owner)
    (rightOwner := SemanticResult39420.owner)
    (leftResult := 39697) (rightResult := 39420)
    (leftActual := SemanticResult39697.actual selector witness)
    (rightActual := SemanticResult39420.actual selector witness)
    (leftRaw := SemanticResult39697.rawTerms)
    (rightRaw := SemanticResult39420.rawTerms)
    (working := LeftOperatorMerge39703.working)
    (leftBinding := 39698) (rightBinding := 39699)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨25154⟩) (rightExpression := ⟨28543⟩)
    (coefficientTransfer := 39700) (summaryTransfer := 39702)
    (rightCoefficientProducer := 39419)
    (rightSummaryTransfer := 39701)
    (leftMaximum := ⟨352097360556032, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge39703.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority39419.actual selector witness)
    (summaryMagnitude := LeftBound39702.actual selector witness)
    (reconstruction := LeftOperatorMerge39703.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult39697.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39420.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39419.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority39419.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge39703.operationAgreement
  · exact LeftBound39702.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39703.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 39705 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24357⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24357⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge39703.working
    [{ coefficient := (-1), key := LeftRelationMerge39705.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge39705.frameStart
      LeftRelationMerge39705.owner (.relation 39705) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge39705.deltas
    rows := LeftRelationMerge39705.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge39703.working LeftRelationMerge39705.source
        (relationContext LeftRelationMerge39705.source
          LeftRelationMerge39705.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge39703.working, LeftRelationMerge39705.deltas,
    LeftRelationMerge39705.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 39705)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨28545⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28543⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge39703.working) (working := relationWorking0)
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
end SemanticResult39707

namespace SemanticResult39710
def owner : Owner := ⟨.program ⟨214⟩, ⟨21840⟩⟩
def rawTerms : List Term := Proof.Events155.exact39710RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39710
def producerEvent : Nat := 39709
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39710.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨50⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨50⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult39710

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
