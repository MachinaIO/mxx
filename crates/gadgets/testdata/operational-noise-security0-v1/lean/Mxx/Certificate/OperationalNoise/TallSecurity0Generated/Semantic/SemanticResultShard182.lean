import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard182
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard008
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard077
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard165
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard181

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult23789
def owner : Owner := ⟨.program ⟨214⟩, ⟨18214⟩⟩
def rawTerms : List Term := Proof.Events092.exact23789RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23789
def producerEvent : Nat := 23788
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23789.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 23704, .finite 63, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult23789

namespace SemanticResult23794
def owner : Owner := ⟨.program ⟨214⟩, ⟨18215⟩⟩
def rawTerms : List Term := Proof.Events092.exact23794RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23794
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23794.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23793.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge23793.frameStart)
    (transferEvent := 23792) (owner := owner)
    (leftResult := 23766) (rightResult := 23789)
    (working := LeftOperatorMerge23793.working)
    (reconstruction := LeftOperatorMerge23793.reconstruction)
    (leftReference := .predecessor 0 23790 .coefficient) (rightReference := .predecessor 1 23791 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult23766.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23789.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge23793.operationAgreement
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
end SemanticResult23794

namespace SemanticResult23797
def owner : Owner := ⟨.program ⟨214⟩, ⟨6735⟩⟩
def rawTerms : List Term := Proof.Events092.exact23797RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23797
def producerEvent : Nat := 23796
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23797.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 23704, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult23797

namespace SemanticResult23801
def owner : Owner := ⟨.program ⟨214⟩, ⟨18216⟩⟩
def rawTerms : List Term := Proof.Events092.exact23801RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23801
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23801.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 23798) (rightBinding := 23799)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6735⟩) (rightExpression := ⟨18215⟩)
    (transferEvent := 23800)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23797.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23794.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult23801

namespace SemanticResult23805
def owner : Owner := ⟨.program ⟨214⟩, ⟨29212⟩⟩
def rawTerms : List Term := Proof.Events092.exact23805RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23805
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23805.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 23802) (rightBinding := 23803)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18216⟩) (rightExpression := ⟨29208⟩)
    (transferEvent := 23804)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23801.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23786.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult23805

namespace SemanticResult23814
def owner : Owner := ⟨.program ⟨214⟩, ⟨22279⟩⟩
def rawTerms : List Term := Proof.Events093.exact23814RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 23814
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23814.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23649.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge23649.frameStart)
    (owner := owner) (leftOwner := SemanticResult21512.owner)
    (rightOwner := SemanticResult23643.owner)
    (leftResult := 21512) (rightResult := 23643)
    (leftActual := SemanticResult21512.actual selector witness)
    (rightActual := SemanticResult23643.actual selector witness)
    (leftRaw := SemanticResult21512.rawTerms)
    (rightRaw := SemanticResult23643.rawTerms)
    (working := LeftOperatorMerge23649.working)
    (leftBinding := 23644) (rightBinding := 23645)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5559⟩) (rightExpression := ⟨22278⟩)
    (coefficientTransfer := 23646) (summaryTransfer := 23648)
    (rightCoefficientProducer := 23642)
    (rightSummaryTransfer := 23647)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge23649.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound23642.actual selector witness)
    (summaryMagnitude := LeftBound23648.actual selector witness)
    (reconstruction := LeftOperatorMerge23649.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21512.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23643.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23642.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound23642.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge23649.operationAgreement
  · exact LeftBound23648.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23649.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 23809 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29207⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16561⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24549⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18214⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29207⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16561⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24549⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18214⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge23649.working
    [{ coefficient := (1), key := LeftRelationMerge23809.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge23809.frameStart
      LeftRelationMerge23809.owner (.relation 23809) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge23809.deltas
    rows := LeftRelationMerge23809.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge23649.working LeftRelationMerge23809.source
        (relationContext LeftRelationMerge23809.source
          LeftRelationMerge23809.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge23649.working, LeftRelationMerge23809.deltas,
    LeftRelationMerge23809.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 23809)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨22279⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22276⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22276⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge23649.working) (working := relationWorking0)
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
end SemanticResult23814

namespace SemanticResult23821
def owner : Owner := ⟨.program ⟨214⟩, ⟨29210⟩⟩
def rawTerms : List Term := Proof.Events093.exact23821RawTerms
def summary : Bound := (.finite 1292337423279833362432)
def resultEvent : Nat := 23821
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23821.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge23818.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult23814.owner)
    (rightOwner := SemanticResult23636.owner)
    (leftResult := 23814) (rightResult := 23636)
    (leftActual := SemanticResult23814.actual selector witness)
    (rightActual := SemanticResult23636.actual selector witness)
    (leftRaw := SemanticResult23814.rawTerms)
    (rightRaw := SemanticResult23636.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292337421468529852416) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 23815) (rightBinding := 23816)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22279⟩) (rightExpression := ⟨29209⟩)
    (coefficientTransfer := 23817) (summaryTransfer := 23820)
    (base := LeftOperatorMerge23818.base)
    (reconstruction := LeftOperatorMerge23818.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23814.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23636.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge23818.operationAgreement
  · rfl
  · decide
end SemanticResult23821

namespace SemanticResult23828
def owner : Owner := ⟨.program ⟨214⟩, ⟨24486⟩⟩
def rawTerms : List Term := Proof.Events093.exact23828RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23828
def producerEvent : Nat := 23827
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23828.actual selector witness
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
end SemanticResult23828

namespace SemanticResult23831
def owner : Owner := ⟨.program ⟨214⟩, ⟨28990⟩⟩
def rawTerms : List Term := Proof.Events093.exact23831RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23831
def producerEvent : Nat := 23830
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23831.actual selector witness
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
end SemanticResult23831

namespace SemanticResult23838
def owner : Owner := ⟨.program ⟨214⟩, ⟨23212⟩⟩
def rawTerms : List Term := Proof.Events093.exact23838RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23838
def producerEvent : Nat := 23837
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23838.actual selector witness
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
end SemanticResult23838

namespace SemanticResult23841
def owner : Owner := ⟨.program ⟨214⟩, ⟨25388⟩⟩
def rawTerms : List Term := Proof.Events093.exact23841RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23841
def producerEvent : Nat := 23840
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23841.actual selector witness
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
end SemanticResult23841

namespace SemanticResult23846
def owner : Owner := ⟨.program ⟨214⟩, ⟨12397⟩⟩
def rawTerms : List Term := Proof.Events093.exact23846RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23846
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23846.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23845.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge23845.frameStart)
    (transferEvent := 23844) (owner := owner)
    (leftResult := 957) (rightResult := 21420)
    (working := LeftOperatorMerge23845.working)
    (reconstruction := LeftOperatorMerge23845.reconstruction)
    (leftReference := .predecessor 0 23842 .coefficient) (rightReference := .predecessor 1 23843 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult957.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge23845.operationAgreement
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
end SemanticResult23846

namespace SemanticResult23851
def owner : Owner := ⟨.program ⟨214⟩, ⟨7355⟩⟩
def rawTerms : List Term := Proof.Events093.exact23851RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23851
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23851.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23850.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge23850.frameStart)
    (transferEvent := 23849) (owner := owner)
    (leftResult := 21290) (rightResult := 8977)
    (working := LeftOperatorMerge23850.working)
    (reconstruction := LeftOperatorMerge23850.reconstruction)
    (leftReference := .predecessor 0 23847 .coefficient) (rightReference := .predecessor 1 23848 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8977.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge23850.operationAgreement
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
end SemanticResult23851

namespace SemanticResult23855
def owner : Owner := ⟨.program ⟨214⟩, ⟨12398⟩⟩
def rawTerms : List Term := Proof.Events093.exact23855RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23855
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23855.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 23852) (rightBinding := 23853)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7355⟩) (rightExpression := ⟨12397⟩)
    (transferEvent := 23854)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23851.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23846.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult23855

namespace SemanticResult23861
def owner : Owner := ⟨.program ⟨214⟩, ⟨12399⟩⟩
def rawTerms : List Term := Proof.Events093.exact23861RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 23861
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23861.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 23858) (survivorTransfer := 23859)
    (survivorEvent := 23860) (resultEvent := resultEvent)
    (rightCoefficientProducer := 8968)
    (owner := owner) (leftOwner := SemanticResult23855.owner)
    (rightOwner := SemanticResult8969.owner)
    (leftResult := 23855) (rightResult := 8969)
    (leftBinding := 23856) (rightBinding := 23857)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12398⟩) (rightExpression := ⟨99⟩)
    (leftActual := SemanticResult23855.actual selector witness)
    (rightActual := SemanticResult8969.actual selector witness)
    (leftRaw := SemanticResult23855.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨99⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound8968.actual selector witness)
    (survivorMagnitude := LeftBound23859.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23855.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8969.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8968.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8968.derived selector witness)
  · exact LeftBound23859.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult23861

namespace SemanticResult23869
def owner : Owner := ⟨.program ⟨214⟩, ⟨12400⟩⟩
def rawTerms : List Term := Proof.Events093.exact23869RawTerms
def summary : Bound := (.finite 33280)
def resultEvent : Nat := 23869
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23869.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨40, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23867.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge23867.frameStart)
    (owner := owner) (leftOwner := SemanticResult23861.owner)
    (rightOwner := SemanticResult960.owner)
    (leftResult := 23861) (rightResult := 960)
    (leftActual := SemanticResult23861.actual selector witness)
    (rightActual := SemanticResult960.actual selector witness)
    (leftRaw := SemanticResult23861.rawTerms)
    (rightRaw := SemanticResult960.rawTerms)
    (working := LeftOperatorMerge23867.working)
    (leftBinding := 23862) (rightBinding := 23863)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12399⟩) (rightExpression := ⟨9835⟩)
    (coefficientTransfer := 23864) (summaryTransfer := 23866)
    (rightCoefficientProducer := 959)
    (rightSummaryTransfer := 23865)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨40, by decide⟩)
    (rightRecordedMaximum := 40)
    (rightSummaryMaximum := ⟨40, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge23867.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority959.actual selector witness)
    (summaryMagnitude := LeftBound23866.actual selector witness)
    (reconstruction := LeftOperatorMerge23867.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23861.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult960.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority959.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority959.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge23867.operationAgreement
  · exact LeftBound23866.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23867.working summary) := by
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
end SemanticResult23869

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
