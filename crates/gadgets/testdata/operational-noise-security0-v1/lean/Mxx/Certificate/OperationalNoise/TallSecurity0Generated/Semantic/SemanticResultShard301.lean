import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard301
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard014
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard097
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard265
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard300

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult40824
def owner : Owner := ⟨.program ⟨214⟩, ⟨16111⟩⟩
def rawTerms : List Term := Proof.Events159.exact40824RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40824
def producerEvent : Nat := 40823
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40824.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 40739, .finite 61, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult40824

namespace SemanticResult40829
def owner : Owner := ⟨.program ⟨214⟩, ⟨16112⟩⟩
def rawTerms : List Term := Proof.Events159.exact40829RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40829
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40829.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40828.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge40828.frameStart)
    (transferEvent := 40827) (owner := owner)
    (leftResult := 40801) (rightResult := 40824)
    (working := LeftOperatorMerge40828.working)
    (reconstruction := LeftOperatorMerge40828.reconstruction)
    (leftReference := .predecessor 0 40825 .coefficient) (rightReference := .predecessor 1 40826 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult40801.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40824.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge40828.operationAgreement
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
end SemanticResult40829

namespace SemanticResult40832
def owner : Owner := ⟨.program ⟨214⟩, ⟨6725⟩⟩
def rawTerms : List Term := Proof.Events159.exact40832RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40832
def producerEvent : Nat := 40831
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40832.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 40739, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult40832

namespace SemanticResult40836
def owner : Owner := ⟨.program ⟨214⟩, ⟨16113⟩⟩
def rawTerms : List Term := Proof.Events159.exact40836RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40836
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40836.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 40833) (rightBinding := 40834)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6725⟩) (rightExpression := ⟨16112⟩)
    (transferEvent := 40835)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40832.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40829.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40836

namespace SemanticResult40840
def owner : Owner := ⟨.program ⟨214⟩, ⟨28114⟩⟩
def rawTerms : List Term := Proof.Events159.exact40840RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40840
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40840.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 40837) (rightBinding := 40838)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16113⟩) (rightExpression := ⟨28110⟩)
    (transferEvent := 40839)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40836.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40821.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40840

namespace SemanticResult40849
def owner : Owner := ⟨.program ⟨214⟩, ⟨21555⟩⟩
def rawTerms : List Term := Proof.Events159.exact40849RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 40849
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40849.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40684.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge40684.frameStart)
    (owner := owner) (leftOwner := SemanticResult36137.owner)
    (rightOwner := SemanticResult40678.owner)
    (leftResult := 36137) (rightResult := 40678)
    (leftActual := SemanticResult36137.actual selector witness)
    (rightActual := SemanticResult40678.actual selector witness)
    (leftRaw := SemanticResult36137.rawTerms)
    (rightRaw := SemanticResult40678.rawTerms)
    (working := LeftOperatorMerge40684.working)
    (leftBinding := 40679) (rightBinding := 40680)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5553⟩) (rightExpression := ⟨21554⟩)
    (coefficientTransfer := 40681) (summaryTransfer := 40683)
    (rightCoefficientProducer := 40677)
    (rightSummaryTransfer := 40682)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge40684.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound40677.actual selector witness)
    (summaryMagnitude := LeftBound40683.actual selector witness)
    (reconstruction := LeftOperatorMerge40684.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36137.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40678.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40677.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound40677.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge40684.operationAgreement
  · exact LeftBound40683.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40684.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 40844 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28109⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16067⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24231⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6698⟩⟩, ⟨.program ⟨214⟩, ⟨28109⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16067⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24231⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge40684.working
    [{ coefficient := (1), key := LeftRelationMerge40844.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge40844.frameStart
      LeftRelationMerge40844.owner (.relation 40844) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge40844.deltas
    rows := LeftRelationMerge40844.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge40684.working LeftRelationMerge40844.source
        (relationContext LeftRelationMerge40844.source
          LeftRelationMerge40844.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge40684.working, LeftRelationMerge40844.deltas,
    LeftRelationMerge40844.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 40844)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨21555⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21552⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21552⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge40684.working) (working := relationWorking0)
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
end SemanticResult40849

namespace SemanticResult40856
def owner : Owner := ⟨.program ⟨214⟩, ⟨28112⟩⟩
def rawTerms : List Term := Proof.Events159.exact40856RawTerms
def summary : Bound := (.finite 1292113298829627502592)
def resultEvent : Nat := 40856
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40856.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge40853.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult40849.owner)
    (rightOwner := SemanticResult40671.owner)
    (leftResult := 40849) (rightResult := 40671)
    (leftActual := SemanticResult40849.actual selector witness)
    (rightActual := SemanticResult40671.actual selector witness)
    (leftRaw := SemanticResult40849.rawTerms)
    (rightRaw := SemanticResult40671.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292113297018323992576) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 40850) (rightBinding := 40851)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21555⟩) (rightExpression := ⟨28111⟩)
    (coefficientTransfer := 40852) (summaryTransfer := 40855)
    (base := LeftOperatorMerge40853.base)
    (reconstruction := LeftOperatorMerge40853.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40849.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40671.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge40853.operationAgreement
  · rfl
  · decide
end SemanticResult40856

namespace SemanticResult40863
def owner : Owner := ⟨.program ⟨214⟩, ⟨24168⟩⟩
def rawTerms : List Term := Proof.Events159.exact40863RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40863
def producerEvent : Nat := 40862
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40863.actual selector witness
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
end SemanticResult40863

namespace SemanticResult40866
def owner : Owner := ⟨.program ⟨214⟩, ⟨27892⟩⟩
def rawTerms : List Term := Proof.Events159.exact40866RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40866
def producerEvent : Nat := 40865
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40866.actual selector witness
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
end SemanticResult40866

namespace SemanticResult40873
def owner : Owner := ⟨.program ⟨214⟩, ⟨23588⟩⟩
def rawTerms : List Term := Proof.Events159.exact40873RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40873
def producerEvent : Nat := 40872
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40873.actual selector witness
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
end SemanticResult40873

namespace SemanticResult40876
def owner : Owner := ⟨.program ⟨214⟩, ⟨26076⟩⟩
def rawTerms : List Term := Proof.Events159.exact40876RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40876
def producerEvent : Nat := 40875
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40876.actual selector witness
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
end SemanticResult40876

namespace SemanticResult40881
def owner : Owner := ⟨.program ⟨214⟩, ⟨11478⟩⟩
def rawTerms : List Term := Proof.Events159.exact40881RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40881
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40881.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40880.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge40880.frameStart)
    (transferEvent := 40879) (owner := owner)
    (leftResult := 1820) (rightResult := 36045)
    (working := LeftOperatorMerge40880.working)
    (reconstruction := LeftOperatorMerge40880.reconstruction)
    (leftReference := .predecessor 0 40877 .coefficient) (rightReference := .predecessor 1 40878 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1820.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge40880.operationAgreement
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
end SemanticResult40881

namespace SemanticResult40886
def owner : Owner := ⟨.program ⟨214⟩, ⟨7311⟩⟩
def rawTerms : List Term := Proof.Events159.exact40886RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40886
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40886.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40885.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge40885.frameStart)
    (transferEvent := 40884) (owner := owner)
    (leftResult := 35915) (rightResult := 11482)
    (working := LeftOperatorMerge40885.working)
    (reconstruction := LeftOperatorMerge40885.reconstruction)
    (leftReference := .predecessor 0 40882 .coefficient) (rightReference := .predecessor 1 40883 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11482.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge40885.operationAgreement
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
end SemanticResult40886

namespace SemanticResult40890
def owner : Owner := ⟨.program ⟨214⟩, ⟨11479⟩⟩
def rawTerms : List Term := Proof.Events159.exact40890RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40890
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40890.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 40887) (rightBinding := 40888)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7311⟩) (rightExpression := ⟨11478⟩)
    (transferEvent := 40889)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40886.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40881.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40890

namespace SemanticResult40896
def owner : Owner := ⟨.program ⟨214⟩, ⟨11480⟩⟩
def rawTerms : List Term := Proof.Events159.exact40896RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 40896
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40896.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 40893) (survivorTransfer := 40894)
    (survivorEvent := 40895) (resultEvent := resultEvent)
    (rightCoefficientProducer := 11473)
    (owner := owner) (leftOwner := SemanticResult40890.owner)
    (rightOwner := SemanticResult11474.owner)
    (leftResult := 40890) (rightResult := 11474)
    (leftBinding := 40891) (rightBinding := 40892)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11479⟩) (rightExpression := ⟨93⟩)
    (leftActual := SemanticResult40890.actual selector witness)
    (rightActual := SemanticResult11474.actual selector witness)
    (leftRaw := SemanticResult40890.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨93⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound11473.actual selector witness)
    (survivorMagnitude := LeftBound40894.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40890.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11474.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)
  · exact LeftBound40894.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult40896

namespace SemanticResult40904
def owner : Owner := ⟨.program ⟨214⟩, ⟨14228⟩⟩
def rawTerms : List Term := Proof.Events159.exact40904RawTerms
def summary : Bound := (.finite 14976)
def resultEvent : Nat := 40904
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40904.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨18, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40902.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge40902.frameStart)
    (owner := owner) (leftOwner := SemanticResult40896.owner)
    (rightOwner := SemanticResult1823.owner)
    (leftResult := 40896) (rightResult := 1823)
    (leftActual := SemanticResult40896.actual selector witness)
    (rightActual := SemanticResult1823.actual selector witness)
    (leftRaw := SemanticResult40896.rawTerms)
    (rightRaw := SemanticResult1823.rawTerms)
    (working := LeftOperatorMerge40902.working)
    (leftBinding := 40897) (rightBinding := 40898)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11480⟩) (rightExpression := ⟨14225⟩)
    (coefficientTransfer := 40899) (summaryTransfer := 40901)
    (rightCoefficientProducer := 1822)
    (rightSummaryTransfer := 40900)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨18, by decide⟩)
    (rightRecordedMaximum := 18)
    (rightSummaryMaximum := ⟨18, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge40902.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1822.actual selector witness)
    (summaryMagnitude := LeftBound40901.actual selector witness)
    (reconstruction := LeftOperatorMerge40902.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40896.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1823.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1822.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1822.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge40902.operationAgreement
  · exact LeftBound40901.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40902.working summary) := by
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
end SemanticResult40904

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
