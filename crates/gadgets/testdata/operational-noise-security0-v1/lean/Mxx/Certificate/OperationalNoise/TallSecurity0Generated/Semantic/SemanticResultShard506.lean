import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard506
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard026
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard101
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard102
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard505

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult70568
def owner : Owner := ⟨.program ⟨214⟩, ⟨15985⟩⟩
def rawTerms : List Term := Proof.Events275.exact70568RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70568
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70568.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 70565) (rightBinding := 70566)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6723⟩) (rightExpression := ⟨15984⟩)
    (transferEvent := 70567)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult70564.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70561.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult70568

namespace SemanticResult70572
def owner : Owner := ⟨.program ⟨214⟩, ⟨27858⟩⟩
def rawTerms : List Term := Proof.Events275.exact70572RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70572
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70572.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 70569) (rightBinding := 70570)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15985⟩) (rightExpression := ⟨27854⟩)
    (transferEvent := 70571)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult70568.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70553.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult70572

namespace SemanticResult70581
def owner : Owner := ⟨.program ⟨214⟩, ⟨21399⟩⟩
def rawTerms : List Term := Proof.Events275.exact70581RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 70581
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70581.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70416.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge70416.frameStart)
    (owner := owner) (leftOwner := SemanticResult65387.owner)
    (rightOwner := SemanticResult70410.owner)
    (leftResult := 65387) (rightResult := 70410)
    (leftActual := SemanticResult65387.actual selector witness)
    (rightActual := SemanticResult70410.actual selector witness)
    (leftRaw := SemanticResult65387.rawTerms)
    (rightRaw := SemanticResult70410.rawTerms)
    (working := LeftOperatorMerge70416.working)
    (leftBinding := 70411) (rightBinding := 70412)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5535⟩) (rightExpression := ⟨21398⟩)
    (coefficientTransfer := 70413) (summaryTransfer := 70415)
    (rightCoefficientProducer := 70409)
    (rightSummaryTransfer := 70414)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge70416.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound70409.actual selector witness)
    (summaryMagnitude := LeftBound70415.actual selector witness)
    (reconstruction := LeftOperatorMerge70416.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult65387.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70410.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70409.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound70409.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge70416.operationAgreement
  · exact LeftBound70415.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70416.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 70576 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15936⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24159⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩, ⟨.program ⟨214⟩, ⟨27853⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15936⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24159⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge70416.working
    [{ coefficient := (1), key := LeftRelationMerge70576.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge70576.frameStart
      LeftRelationMerge70576.owner (.relation 70576) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge70576.deltas
    rows := LeftRelationMerge70576.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge70416.working LeftRelationMerge70576.source
        (relationContext LeftRelationMerge70576.source
          LeftRelationMerge70576.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge70416.working, LeftRelationMerge70576.deltas,
    LeftRelationMerge70576.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 70576)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨21399⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21396⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21396⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge70416.working) (working := relationWorking0)
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
end SemanticResult70581

namespace SemanticResult70588
def owner : Owner := ⟨.program ⟨214⟩, ⟨27856⟩⟩
def rawTerms : List Term := Proof.Events275.exact70588RawTerms
def summary : Bound := (.finite 1292068473939586330624)
def resultEvent : Nat := 70588
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70588.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge70585.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult70581.owner)
    (rightOwner := SemanticResult70403.owner)
    (leftResult := 70581) (rightResult := 70403)
    (leftActual := SemanticResult70581.actual selector witness)
    (rightActual := SemanticResult70403.actual selector witness)
    (leftRaw := SemanticResult70581.rawTerms)
    (rightRaw := SemanticResult70403.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292068472128282820608) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 70582) (rightBinding := 70583)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21399⟩) (rightExpression := ⟨27855⟩)
    (coefficientTransfer := 70584) (summaryTransfer := 70587)
    (base := LeftOperatorMerge70585.base)
    (reconstruction := LeftOperatorMerge70585.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult70581.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70403.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge70585.operationAgreement
  · rfl
  · decide
end SemanticResult70588

namespace SemanticResult70595
def owner : Owner := ⟨.program ⟨214⟩, ⟨24096⟩⟩
def rawTerms : List Term := Proof.Events275.exact70595RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70595
def producerEvent : Nat := 70594
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70595.actual selector witness
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
end SemanticResult70595

namespace SemanticResult70598
def owner : Owner := ⟨.program ⟨214⟩, ⟨27636⟩⟩
def rawTerms : List Term := Proof.Events275.exact70598RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70598
def producerEvent : Nat := 70597
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70598.actual selector witness
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
end SemanticResult70598

namespace SemanticResult70605
def owner : Owner := ⟨.program ⟨214⟩, ⟨23540⟩⟩
def rawTerms : List Term := Proof.Events275.exact70605RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70605
def producerEvent : Nat := 70604
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70605.actual selector witness
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
end SemanticResult70605

namespace SemanticResult70608
def owner : Owner := ⟨.program ⟨214⟩, ⟨25984⟩⟩
def rawTerms : List Term := Proof.Events275.exact70608RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70608
def producerEvent : Nat := 70607
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70608.actual selector witness
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
end SemanticResult70608

namespace SemanticResult70613
def owner : Owner := ⟨.program ⟨214⟩, ⟨11382⟩⟩
def rawTerms : List Term := Proof.Events275.exact70613RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70613
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70613.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70612.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge70612.frameStart)
    (transferEvent := 70611) (owner := owner)
    (leftResult := 3339) (rightResult := 65295)
    (working := LeftOperatorMerge70612.working)
    (reconstruction := LeftOperatorMerge70612.reconstruction)
    (leftReference := .predecessor 0 70609 .coefficient) (rightReference := .predecessor 1 70610 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3339.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge70612.operationAgreement
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
end SemanticResult70613

namespace SemanticResult70618
def owner : Owner := ⟨.program ⟨214⟩, ⟨7196⟩⟩
def rawTerms : List Term := Proof.Events275.exact70618RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70618
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70618.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70617.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge70617.frameStart)
    (transferEvent := 70616) (owner := owner)
    (leftResult := 65165) (rightResult := 11983)
    (working := LeftOperatorMerge70617.working)
    (reconstruction := LeftOperatorMerge70617.reconstruction)
    (leftReference := .predecessor 0 70614 .coefficient) (rightReference := .predecessor 1 70615 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11983.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge70617.operationAgreement
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
end SemanticResult70618

namespace SemanticResult70622
def owner : Owner := ⟨.program ⟨214⟩, ⟨11383⟩⟩
def rawTerms : List Term := Proof.Events275.exact70622RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70622
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70622.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 70619) (rightBinding := 70620)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7196⟩) (rightExpression := ⟨11382⟩)
    (transferEvent := 70621)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult70618.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70613.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult70622

namespace SemanticResult70628
def owner : Owner := ⟨.program ⟨214⟩, ⟨11384⟩⟩
def rawTerms : List Term := Proof.Events275.exact70628RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 70628
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70628.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 70625) (survivorTransfer := 70626)
    (survivorEvent := 70627) (resultEvent := resultEvent)
    (rightCoefficientProducer := 11974)
    (owner := owner) (leftOwner := SemanticResult70622.owner)
    (rightOwner := SemanticResult11975.owner)
    (leftResult := 70622) (rightResult := 11975)
    (leftBinding := 70623) (rightBinding := 70624)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11383⟩) (rightExpression := ⟨92⟩)
    (leftActual := SemanticResult70622.actual selector witness)
    (rightActual := SemanticResult11975.actual selector witness)
    (leftRaw := SemanticResult70622.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨92⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound11974.actual selector witness)
    (survivorMagnitude := LeftBound70626.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult70622.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11975.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11974.derived selector witness)
  · exact LeftBound70626.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult70628

namespace SemanticResult70636
def owner : Owner := ⟨.program ⟨214⟩, ⟨13984⟩⟩
def rawTerms : List Term := Proof.Events275.exact70636RawTerms
def summary : Bound := (.finite 13312)
def resultEvent : Nat := 70636
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70636.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨16, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70634.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge70634.frameStart)
    (owner := owner) (leftOwner := SemanticResult70628.owner)
    (rightOwner := SemanticResult3342.owner)
    (leftResult := 70628) (rightResult := 3342)
    (leftActual := SemanticResult70628.actual selector witness)
    (rightActual := SemanticResult3342.actual selector witness)
    (leftRaw := SemanticResult70628.rawTerms)
    (rightRaw := SemanticResult3342.rawTerms)
    (working := LeftOperatorMerge70634.working)
    (leftBinding := 70629) (rightBinding := 70630)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11384⟩) (rightExpression := ⟨13981⟩)
    (coefficientTransfer := 70631) (summaryTransfer := 70633)
    (rightCoefficientProducer := 3341)
    (rightSummaryTransfer := 70632)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨16, by decide⟩)
    (rightRecordedMaximum := 16)
    (rightSummaryMaximum := ⟨16, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge70634.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority3341.actual selector witness)
    (summaryMagnitude := LeftBound70633.actual selector witness)
    (reconstruction := LeftOperatorMerge70634.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult70628.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3342.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3341.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority3341.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge70634.operationAgreement
  · exact LeftBound70633.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70634.working summary) := by
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
end SemanticResult70636

namespace SemanticResult70641
def owner : Owner := ⟨.program ⟨214⟩, ⟨13985⟩⟩
def rawTerms : List Term := Proof.Events275.exact70641RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70641
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70641.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70640.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge70640.frameStart)
    (transferEvent := 70639) (owner := owner)
    (leftResult := 3342) (rightResult := 65295)
    (working := LeftOperatorMerge70640.working)
    (reconstruction := LeftOperatorMerge70640.reconstruction)
    (leftReference := .predecessor 0 70637 .coefficient) (rightReference := .predecessor 1 70638 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3342.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge70640.operationAgreement
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
end SemanticResult70641

namespace SemanticResult70646
def owner : Owner := ⟨.program ⟨214⟩, ⟨7176⟩⟩
def rawTerms : List Term := Proof.Events275.exact70646RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70646
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70646.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge70645.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge70645.frameStart)
    (transferEvent := 70644) (owner := owner)
    (leftResult := 65165) (rightResult := 12024)
    (working := LeftOperatorMerge70645.working)
    (reconstruction := LeftOperatorMerge70645.reconstruction)
    (leftReference := .predecessor 0 70642 .coefficient) (rightReference := .predecessor 1 70643 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12024.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge70645.operationAgreement
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
end SemanticResult70646

namespace SemanticResult70650
def owner : Owner := ⟨.program ⟨214⟩, ⟨13986⟩⟩
def rawTerms : List Term := Proof.Events275.exact70650RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 70650
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult70650.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 70647) (rightBinding := 70648)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7176⟩) (rightExpression := ⟨13985⟩)
    (transferEvent := 70649)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult70646.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult70641.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult70650

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
