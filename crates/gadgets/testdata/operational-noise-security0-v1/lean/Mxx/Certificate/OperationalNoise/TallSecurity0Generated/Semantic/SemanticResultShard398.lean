import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard398
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard093
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard094
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard397

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult54975
def owner : Owner := ⟨.program ⟨214⟩, ⟨6727⟩⟩
def rawTerms : List Term := Proof.Events214.exact54975RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54975
def producerEvent : Nat := 54974
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54975.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 54882, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult54975

namespace SemanticResult54979
def owner : Owner := ⟨.program ⟨214⟩, ⟨18365⟩⟩
def rawTerms : List Term := Proof.Events214.exact54979RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54979
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54979.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 54976) (rightBinding := 54977)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6727⟩) (rightExpression := ⟨18364⟩)
    (transferEvent := 54978)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54975.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult54972.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult54979

namespace SemanticResult54983
def owner : Owner := ⟨.program ⟨214⟩, ⟨28318⟩⟩
def rawTerms : List Term := Proof.Events214.exact54983RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 54983
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54983.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 54980) (rightBinding := 54981)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18365⟩) (rightExpression := ⟨28314⟩)
    (transferEvent := 54982)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54979.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult54964.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult54983

namespace SemanticResult54992
def owner : Owner := ⟨.program ⟨214⟩, ⟨21695⟩⟩
def rawTerms : List Term := Proof.Events214.exact54992RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 54992
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54992.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54827.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge54827.frameStart)
    (owner := owner) (leftOwner := SemanticResult50762.owner)
    (rightOwner := SemanticResult54821.owner)
    (leftResult := 50762) (rightResult := 54821)
    (leftActual := SemanticResult50762.actual selector witness)
    (rightActual := SemanticResult54821.actual selector witness)
    (leftRaw := SemanticResult50762.rawTerms)
    (rightRaw := SemanticResult54821.rawTerms)
    (working := LeftOperatorMerge54827.working)
    (leftBinding := 54822) (rightBinding := 54823)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5547⟩) (rightExpression := ⟨21694⟩)
    (coefficientTransfer := 54824) (summaryTransfer := 54826)
    (rightCoefficientProducer := 54820)
    (rightSummaryTransfer := 54825)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge54827.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound54820.actual selector witness)
    (summaryMagnitude := LeftBound54826.actual selector witness)
    (reconstruction := LeftOperatorMerge54827.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50762.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult54821.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54820.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound54820.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge54827.operationAgreement
  · exact LeftBound54826.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge54827.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 54987 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28313⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16182⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24291⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge54827.working
    [{ coefficient := (1), key := LeftRelationMerge54987.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge54987.frameStart
      LeftRelationMerge54987.owner (.relation 54987) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge54987.deltas
    rows := LeftRelationMerge54987.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge54827.working LeftRelationMerge54987.source
        (relationContext LeftRelationMerge54987.source
          LeftRelationMerge54987.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge54827.working, LeftRelationMerge54987.deltas,
    LeftRelationMerge54987.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 54987)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨21695⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21692⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge54827.working) (working := relationWorking0)
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
end SemanticResult54992

namespace SemanticResult54999
def owner : Owner := ⟨.program ⟨214⟩, ⟨28316⟩⟩
def rawTerms : List Term := Proof.Events214.exact54999RawTerms
def summary : Bound := (.finite 1292180536164689260544)
def resultEvent : Nat := 54999
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult54999.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge54996.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult54992.owner)
    (rightOwner := SemanticResult54814.owner)
    (leftResult := 54992) (rightResult := 54814)
    (leftActual := SemanticResult54992.actual selector witness)
    (rightActual := SemanticResult54814.actual selector witness)
    (leftRaw := SemanticResult54992.rawTerms)
    (rightRaw := SemanticResult54814.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292180534353385750528) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 54993) (rightBinding := 54994)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21695⟩) (rightExpression := ⟨28315⟩)
    (coefficientTransfer := 54995) (summaryTransfer := 54998)
    (base := LeftOperatorMerge54996.base)
    (reconstruction := LeftOperatorMerge54996.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult54992.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult54814.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge54996.operationAgreement
  · rfl
  · decide
end SemanticResult54999

namespace SemanticResult55006
def owner : Owner := ⟨.program ⟨214⟩, ⟨24228⟩⟩
def rawTerms : List Term := Proof.Events214.exact55006RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55006
def producerEvent : Nat := 55005
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55006.actual selector witness
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
end SemanticResult55006

namespace SemanticResult55009
def owner : Owner := ⟨.program ⟨214⟩, ⟨28096⟩⟩
def rawTerms : List Term := Proof.Events214.exact55009RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55009
def producerEvent : Nat := 55008
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55009.actual selector witness
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
end SemanticResult55009

namespace SemanticResult55016
def owner : Owner := ⟨.program ⟨214⟩, ⟨23628⟩⟩
def rawTerms : List Term := Proof.Events214.exact55016RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55016
def producerEvent : Nat := 55015
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55016.actual selector witness
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
end SemanticResult55016

namespace SemanticResult55019
def owner : Owner := ⟨.program ⟨214⟩, ⟨26148⟩⟩
def rawTerms : List Term := Proof.Events214.exact55019RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55019
def producerEvent : Nat := 55018
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55019.actual selector witness
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
end SemanticResult55019

namespace SemanticResult55024
def owner : Owner := ⟨.program ⟨214⟩, ⟨11558⟩⟩
def rawTerms : List Term := Proof.Events214.exact55024RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55024
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55024.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55023.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge55023.frameStart)
    (transferEvent := 55022) (owner := owner)
    (leftResult := 2545) (rightResult := 50670)
    (working := LeftOperatorMerge55023.working)
    (reconstruction := LeftOperatorMerge55023.reconstruction)
    (leftReference := .predecessor 0 55020 .coefficient) (rightReference := .predecessor 1 55021 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2545.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge55023.operationAgreement
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
end SemanticResult55024

namespace SemanticResult55029
def owner : Owner := ⟨.program ⟨214⟩, ⟨7274⟩⟩
def rawTerms : List Term := Proof.Events214.exact55029RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55029
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55029.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55028.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge55028.frameStart)
    (transferEvent := 55027) (owner := owner)
    (leftResult := 50540) (rightResult := 10981)
    (working := LeftOperatorMerge55028.working)
    (reconstruction := LeftOperatorMerge55028.reconstruction)
    (leftReference := .predecessor 0 55025 .coefficient) (rightReference := .predecessor 1 55026 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10981.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge55028.operationAgreement
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
end SemanticResult55029

namespace SemanticResult55033
def owner : Owner := ⟨.program ⟨214⟩, ⟨11559⟩⟩
def rawTerms : List Term := Proof.Events214.exact55033RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55033
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55033.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 55030) (rightBinding := 55031)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7274⟩) (rightExpression := ⟨11558⟩)
    (transferEvent := 55032)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult55029.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult55024.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult55033

namespace SemanticResult55039
def owner : Owner := ⟨.program ⟨214⟩, ⟨11560⟩⟩
def rawTerms : List Term := Proof.Events214.exact55039RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 55039
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55039.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 55036) (survivorTransfer := 55037)
    (survivorEvent := 55038) (resultEvent := resultEvent)
    (rightCoefficientProducer := 10972)
    (owner := owner) (leftOwner := SemanticResult55033.owner)
    (rightOwner := SemanticResult10973.owner)
    (leftResult := 55033) (rightResult := 10973)
    (leftBinding := 55034) (rightBinding := 55035)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11559⟩) (rightExpression := ⟨94⟩)
    (leftActual := SemanticResult55033.actual selector witness)
    (rightActual := SemanticResult10973.actual selector witness)
    (leftRaw := SemanticResult55033.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨94⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound10972.actual selector witness)
    (survivorMagnitude := LeftBound55037.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult55033.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10973.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10972.derived selector witness)
  · exact LeftBound55037.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult55039

namespace SemanticResult55047
def owner : Owner := ⟨.program ⟨214⟩, ⟨14436⟩⟩
def rawTerms : List Term := Proof.Events215.exact55047RawTerms
def summary : Bound := (.finite 18304)
def resultEvent : Nat := 55047
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55047.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨22, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55045.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge55045.frameStart)
    (owner := owner) (leftOwner := SemanticResult55039.owner)
    (rightOwner := SemanticResult2548.owner)
    (leftResult := 55039) (rightResult := 2548)
    (leftActual := SemanticResult55039.actual selector witness)
    (rightActual := SemanticResult2548.actual selector witness)
    (leftRaw := SemanticResult55039.rawTerms)
    (rightRaw := SemanticResult2548.rawTerms)
    (working := LeftOperatorMerge55045.working)
    (leftBinding := 55040) (rightBinding := 55041)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11560⟩) (rightExpression := ⟨14433⟩)
    (coefficientTransfer := 55042) (summaryTransfer := 55044)
    (rightCoefficientProducer := 2547)
    (rightSummaryTransfer := 55043)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨22, by decide⟩)
    (rightRecordedMaximum := 22)
    (rightSummaryMaximum := ⟨22, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge55045.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority2547.actual selector witness)
    (summaryMagnitude := LeftBound55044.actual selector witness)
    (reconstruction := LeftOperatorMerge55045.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult55039.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2548.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2547.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority2547.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge55045.operationAgreement
  · exact LeftBound55044.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55045.working summary) := by
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
end SemanticResult55047

namespace SemanticResult55052
def owner : Owner := ⟨.program ⟨214⟩, ⟨14437⟩⟩
def rawTerms : List Term := Proof.Events215.exact55052RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55052
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55052.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55051.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge55051.frameStart)
    (transferEvent := 55050) (owner := owner)
    (leftResult := 2548) (rightResult := 50670)
    (working := LeftOperatorMerge55051.working)
    (reconstruction := LeftOperatorMerge55051.reconstruction)
    (leftReference := .predecessor 0 55048 .coefficient) (rightReference := .predecessor 1 55049 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2548.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge55051.operationAgreement
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
end SemanticResult55052

namespace SemanticResult55057
def owner : Owner := ⟨.program ⟨214⟩, ⟨7255⟩⟩
def rawTerms : List Term := Proof.Events215.exact55057RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 55057
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult55057.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge55056.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge55056.frameStart)
    (transferEvent := 55055) (owner := owner)
    (leftResult := 50540) (rightResult := 11022)
    (working := LeftOperatorMerge55056.working)
    (reconstruction := LeftOperatorMerge55056.reconstruction)
    (leftReference := .predecessor 0 55053 .coefficient) (rightReference := .predecessor 1 55054 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11022.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge55056.operationAgreement
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
end SemanticResult55057

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
