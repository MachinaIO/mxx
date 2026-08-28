import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard387
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard081
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard386

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult53533
def owner : Owner := ⟨.program ⟨214⟩, ⟨17909⟩⟩
def rawTerms : List Term := Proof.Events209.exact53533RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53533
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53533.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 53530) (rightBinding := 53531)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6733⟩) (rightExpression := ⟨17908⟩)
    (transferEvent := 53532)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult53529.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult53526.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult53533

namespace SemanticResult53537
def owner : Owner := ⟨.program ⟨214⟩, ⟨28969⟩⟩
def rawTerms : List Term := Proof.Events209.exact53537RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53537
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53537.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 53534) (rightBinding := 53535)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17909⟩) (rightExpression := ⟨28965⟩)
    (transferEvent := 53536)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult53533.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult53518.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult53537

namespace SemanticResult53546
def owner : Owner := ⟨.program ⟨214⟩, ⟨22127⟩⟩
def rawTerms : List Term := Proof.Events209.exact53546RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 53546
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53546.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge53381.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge53381.frameStart)
    (owner := owner) (leftOwner := SemanticResult50762.owner)
    (rightOwner := SemanticResult53375.owner)
    (leftResult := 50762) (rightResult := 53375)
    (leftActual := SemanticResult50762.actual selector witness)
    (rightActual := SemanticResult53375.actual selector witness)
    (leftRaw := SemanticResult50762.rawTerms)
    (rightRaw := SemanticResult53375.rawTerms)
    (working := LeftOperatorMerge53381.working)
    (leftBinding := 53376) (rightBinding := 53377)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5547⟩) (rightExpression := ⟨22126⟩)
    (coefficientTransfer := 53378) (summaryTransfer := 53380)
    (rightCoefficientProducer := 53374)
    (rightSummaryTransfer := 53379)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge53381.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound53374.actual selector witness)
    (summaryMagnitude := LeftBound53380.actual selector witness)
    (reconstruction := LeftOperatorMerge53381.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50762.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult53375.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53374.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound53374.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge53381.operationAgreement
  · exact LeftBound53380.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge53381.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 53541 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28964⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16469⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24480⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17907⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28964⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16469⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24480⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17907⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge53381.working
    [{ coefficient := (1), key := LeftRelationMerge53541.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge53541.frameStart
      LeftRelationMerge53541.owner (.relation 53541) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge53541.deltas
    rows := LeftRelationMerge53541.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge53381.working LeftRelationMerge53541.source
        (relationContext LeftRelationMerge53541.source
          LeftRelationMerge53541.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge53381.working, LeftRelationMerge53541.deltas,
    LeftRelationMerge53541.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 53541)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨22127⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22124⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22124⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge53381.working) (working := relationWorking0)
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
end SemanticResult53546

namespace SemanticResult53553
def owner : Owner := ⟨.program ⟨214⟩, ⟨28967⟩⟩
def rawTerms : List Term := Proof.Events209.exact53553RawTerms
def summary : Bound := (.finite 1292315010834812776448)
def resultEvent : Nat := 53553
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53553.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge53550.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult53546.owner)
    (rightOwner := SemanticResult53368.owner)
    (leftResult := 53546) (rightResult := 53368)
    (leftActual := SemanticResult53546.actual selector witness)
    (rightActual := SemanticResult53368.actual selector witness)
    (leftRaw := SemanticResult53546.rawTerms)
    (rightRaw := SemanticResult53368.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292315009023509266432) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 53547) (rightBinding := 53548)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22127⟩) (rightExpression := ⟨28966⟩)
    (coefficientTransfer := 53549) (summaryTransfer := 53552)
    (base := LeftOperatorMerge53550.base)
    (reconstruction := LeftOperatorMerge53550.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult53546.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult53368.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge53550.operationAgreement
  · rfl
  · decide
end SemanticResult53553

namespace SemanticResult53560
def owner : Owner := ⟨.program ⟨214⟩, ⟨24417⟩⟩
def rawTerms : List Term := Proof.Events209.exact53560RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53560
def producerEvent : Nat := 53559
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53560.actual selector witness
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
end SemanticResult53560

namespace SemanticResult53563
def owner : Owner := ⟨.program ⟨214⟩, ⟨28747⟩⟩
def rawTerms : List Term := Proof.Events209.exact53563RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53563
def producerEvent : Nat := 53562
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53563.actual selector witness
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
end SemanticResult53563

namespace SemanticResult53570
def owner : Owner := ⟨.program ⟨214⟩, ⟨23124⟩⟩
def rawTerms : List Term := Proof.Events209.exact53570RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53570
def producerEvent : Nat := 53569
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53570.actual selector witness
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
end SemanticResult53570

namespace SemanticResult53573
def owner : Owner := ⟨.program ⟨214⟩, ⟨25224⟩⟩
def rawTerms : List Term := Proof.Events209.exact53573RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53573
def producerEvent : Nat := 53572
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53573.actual selector witness
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
end SemanticResult53573

namespace SemanticResult53578
def owner : Owner := ⟨.program ⟨214⟩, ⟨11968⟩⟩
def rawTerms : List Term := Proof.Events209.exact53578RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53578
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53578.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge53577.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge53577.frameStart)
    (transferEvent := 53576) (owner := owner)
    (leftResult := 2476) (rightResult := 50670)
    (working := LeftOperatorMerge53577.working)
    (reconstruction := LeftOperatorMerge53577.reconstruction)
    (leftReference := .predecessor 0 53574 .coefficient) (rightReference := .predecessor 1 53575 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2476.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge53577.operationAgreement
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
end SemanticResult53578

namespace SemanticResult53583
def owner : Owner := ⟨.program ⟨214⟩, ⟨7278⟩⟩
def rawTerms : List Term := Proof.Events209.exact53583RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53583
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53583.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge53582.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge53582.frameStart)
    (transferEvent := 53581) (owner := owner)
    (leftResult := 50540) (rightResult := 9478)
    (working := LeftOperatorMerge53582.working)
    (reconstruction := LeftOperatorMerge53582.reconstruction)
    (leftReference := .predecessor 0 53579 .coefficient) (rightReference := .predecessor 1 53580 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9478.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge53582.operationAgreement
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
end SemanticResult53583

namespace SemanticResult53587
def owner : Owner := ⟨.program ⟨214⟩, ⟨11969⟩⟩
def rawTerms : List Term := Proof.Events209.exact53587RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53587
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53587.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 53584) (rightBinding := 53585)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7278⟩) (rightExpression := ⟨11968⟩)
    (transferEvent := 53586)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult53583.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult53578.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult53587

namespace SemanticResult53593
def owner : Owner := ⟨.program ⟨214⟩, ⟨11970⟩⟩
def rawTerms : List Term := Proof.Events209.exact53593RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 53593
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53593.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 53590) (survivorTransfer := 53591)
    (survivorEvent := 53592) (resultEvent := resultEvent)
    (rightCoefficientProducer := 9469)
    (owner := owner) (leftOwner := SemanticResult53587.owner)
    (rightOwner := SemanticResult9470.owner)
    (leftResult := 53587) (rightResult := 9470)
    (leftBinding := 53588) (rightBinding := 53589)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11969⟩) (rightExpression := ⟨98⟩)
    (leftActual := SemanticResult53587.actual selector witness)
    (rightActual := SemanticResult9470.actual selector witness)
    (leftRaw := SemanticResult53587.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨98⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound9469.actual selector witness)
    (survivorMagnitude := LeftBound53591.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult53587.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9470.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9469.derived selector witness)
  · exact LeftBound53591.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult53593

namespace SemanticResult53601
def owner : Owner := ⟨.program ⟨214⟩, ⟨11971⟩⟩
def rawTerms : List Term := Proof.Events209.exact53601RawTerms
def summary : Bound := (.finite 29952)
def resultEvent : Nat := 53601
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53601.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨36, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge53599.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge53599.frameStart)
    (owner := owner) (leftOwner := SemanticResult53593.owner)
    (rightOwner := SemanticResult2479.owner)
    (leftResult := 53593) (rightResult := 2479)
    (leftActual := SemanticResult53593.actual selector witness)
    (rightActual := SemanticResult2479.actual selector witness)
    (leftRaw := SemanticResult53593.rawTerms)
    (rightRaw := SemanticResult2479.rawTerms)
    (working := LeftOperatorMerge53599.working)
    (leftBinding := 53594) (rightBinding := 53595)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11970⟩) (rightExpression := ⟨9720⟩)
    (coefficientTransfer := 53596) (summaryTransfer := 53598)
    (rightCoefficientProducer := 2478)
    (rightSummaryTransfer := 53597)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨36, by decide⟩)
    (rightRecordedMaximum := 36)
    (rightSummaryMaximum := ⟨36, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge53599.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority2478.actual selector witness)
    (summaryMagnitude := LeftBound53598.actual selector witness)
    (reconstruction := LeftOperatorMerge53599.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult53593.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2479.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2478.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority2478.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge53599.operationAgreement
  · exact LeftBound53598.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge53599.working summary) := by
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
end SemanticResult53601

namespace SemanticResult53606
def owner : Owner := ⟨.program ⟨214⟩, ⟨9721⟩⟩
def rawTerms : List Term := Proof.Events209.exact53606RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53606
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53606.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge53605.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge53605.frameStart)
    (transferEvent := 53604) (owner := owner)
    (leftResult := 2479) (rightResult := 50670)
    (working := LeftOperatorMerge53605.working)
    (reconstruction := LeftOperatorMerge53605.reconstruction)
    (leftReference := .predecessor 0 53602 .coefficient) (rightReference := .predecessor 1 53603 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2479.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge53605.operationAgreement
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
end SemanticResult53606

namespace SemanticResult53611
def owner : Owner := ⟨.program ⟨214⟩, ⟨7258⟩⟩
def rawTerms : List Term := Proof.Events209.exact53611RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53611
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53611.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge53610.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge53610.frameStart)
    (transferEvent := 53609) (owner := owner)
    (leftResult := 50540) (rightResult := 9519)
    (working := LeftOperatorMerge53610.working)
    (reconstruction := LeftOperatorMerge53610.reconstruction)
    (leftReference := .predecessor 0 53607 .coefficient) (rightReference := .predecessor 1 53608 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9519.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge53610.operationAgreement
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
end SemanticResult53611

namespace SemanticResult53615
def owner : Owner := ⟨.program ⟨214⟩, ⟨9722⟩⟩
def rawTerms : List Term := Proof.Events209.exact53615RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53615
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53615.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 53612) (rightBinding := 53613)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7258⟩) (rightExpression := ⟨9721⟩)
    (transferEvent := 53614)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult53611.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult53606.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult53615

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
