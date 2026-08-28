import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard528
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard027
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard125
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard126
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard527

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult73453
def owner : Owner := ⟨.program ⟨214⟩, ⟨15308⟩⟩
def rawTerms : List Term := Proof.Events286.exact73453RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73453
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73453.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73452.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge73452.frameStart)
    (transferEvent := 73451) (owner := owner)
    (leftResult := 73425) (rightResult := 73448)
    (working := LeftOperatorMerge73452.working)
    (reconstruction := LeftOperatorMerge73452.reconstruction)
    (leftReference := .predecessor 0 73449 .coefficient) (rightReference := .predecessor 1 73450 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult73425.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73448.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge73452.operationAgreement
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
end SemanticResult73453

namespace SemanticResult73456
def owner : Owner := ⟨.program ⟨214⟩, ⟨6711⟩⟩
def rawTerms : List Term := Proof.Events286.exact73456RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73456
def producerEvent : Nat := 73455
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73456.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 73363, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult73456

namespace SemanticResult73460
def owner : Owner := ⟨.program ⟨214⟩, ⟨15309⟩⟩
def rawTerms : List Term := Proof.Events286.exact73460RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73460
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73460.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 73457) (rightBinding := 73458)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6711⟩) (rightExpression := ⟨15308⟩)
    (transferEvent := 73459)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73456.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73453.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult73460

namespace SemanticResult73464
def owner : Owner := ⟨.program ⟨214⟩, ⟨26556⟩⟩
def rawTerms : List Term := Proof.Events286.exact73464RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73464
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73464.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 73461) (rightBinding := 73462)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15309⟩) (rightExpression := ⟨26552⟩)
    (transferEvent := 73463)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73460.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73445.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult73464

namespace SemanticResult73473
def owner : Owner := ⟨.program ⟨214⟩, ⟨20535⟩⟩
def rawTerms : List Term := Proof.Events287.exact73473RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 73473
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73473.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73308.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge73308.frameStart)
    (owner := owner) (leftOwner := SemanticResult65387.owner)
    (rightOwner := SemanticResult73302.owner)
    (leftResult := 65387) (rightResult := 73302)
    (leftActual := SemanticResult65387.actual selector witness)
    (rightActual := SemanticResult73302.actual selector witness)
    (leftRaw := SemanticResult65387.rawTerms)
    (rightRaw := SemanticResult73302.rawTerms)
    (working := LeftOperatorMerge73308.working)
    (leftBinding := 73303) (rightBinding := 73304)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5535⟩) (rightExpression := ⟨20534⟩)
    (coefficientTransfer := 73305) (summaryTransfer := 73307)
    (rightCoefficientProducer := 73301)
    (rightSummaryTransfer := 73306)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge73308.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound73301.actual selector witness)
    (summaryMagnitude := LeftBound73307.actual selector witness)
    (reconstruction := LeftOperatorMerge73308.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult65387.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73302.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73301.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound73301.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge73308.operationAgreement
  · exact LeftBound73307.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73308.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 73468 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23781⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26551⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14949⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23781⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge73308.working
    [{ coefficient := (1), key := LeftRelationMerge73468.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge73468.frameStart
      LeftRelationMerge73468.owner (.relation 73468) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge73468.deltas
    rows := LeftRelationMerge73468.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge73308.working LeftRelationMerge73468.source
        (relationContext LeftRelationMerge73468.source
          LeftRelationMerge73468.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge73308.working, LeftRelationMerge73468.deltas,
    LeftRelationMerge73468.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 73468)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨20535⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20532⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20532⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge73308.working) (working := relationWorking0)
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
end SemanticResult73473

namespace SemanticResult73480
def owner : Owner := ⟨.program ⟨214⟩, ⟨26554⟩⟩
def rawTerms : List Term := Proof.Events287.exact73480RawTerms
def summary : Bound := (.finite 1291900380601931935744)
def resultEvent : Nat := 73480
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73480.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge73477.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult73473.owner)
    (rightOwner := SemanticResult73295.owner)
    (leftResult := 73473) (rightResult := 73295)
    (leftActual := SemanticResult73473.actual selector witness)
    (rightActual := SemanticResult73295.actual selector witness)
    (leftRaw := SemanticResult73473.rawTerms)
    (rightRaw := SemanticResult73295.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1291900378790628425728) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 73474) (rightBinding := 73475)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20535⟩) (rightExpression := ⟨26553⟩)
    (coefficientTransfer := 73476) (summaryTransfer := 73479)
    (base := LeftOperatorMerge73477.base)
    (reconstruction := LeftOperatorMerge73477.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73473.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge73477.operationAgreement
  · rfl
  · decide
end SemanticResult73480

namespace SemanticResult73487
def owner : Owner := ⟨.program ⟨214⟩, ⟨23718⟩⟩
def rawTerms : List Term := Proof.Events287.exact73487RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73487
def producerEvent : Nat := 73486
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73487.actual selector witness
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
end SemanticResult73487

namespace SemanticResult73490
def owner : Owner := ⟨.program ⟨214⟩, ⟨26346⟩⟩
def rawTerms : List Term := Proof.Events287.exact73490RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73490
def producerEvent : Nat := 73489
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73490.actual selector witness
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
end SemanticResult73490

namespace SemanticResult73497
def owner : Owner := ⟨.program ⟨214⟩, ⟨22952⟩⟩
def rawTerms : List Term := Proof.Events287.exact73497RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73497
def producerEvent : Nat := 73496
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73497.actual selector witness
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
end SemanticResult73497

namespace SemanticResult73500
def owner : Owner := ⟨.program ⟨214⟩, ⟨24906⟩⟩
def rawTerms : List Term := Proof.Events287.exact73500RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73500
def producerEvent : Nat := 73499
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73500.actual selector witness
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
end SemanticResult73500

namespace SemanticResult73505
def owner : Owner := ⟨.program ⟨214⟩, ⟨10475⟩⟩
def rawTerms : List Term := Proof.Events287.exact73505RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73505
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73505.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73504.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge73504.frameStart)
    (transferEvent := 73503) (owner := owner)
    (leftResult := 3477) (rightResult := 65295)
    (working := LeftOperatorMerge73504.working)
    (reconstruction := LeftOperatorMerge73504.reconstruction)
    (leftReference := .predecessor 0 73501 .coefficient) (rightReference := .predecessor 1 73502 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3477.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge73504.operationAgreement
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
end SemanticResult73505

namespace SemanticResult73510
def owner : Owner := ⟨.program ⟨214⟩, ⟨7190⟩⟩
def rawTerms : List Term := Proof.Events287.exact73510RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73510
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73510.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73509.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge73509.frameStart)
    (transferEvent := 73508) (owner := owner)
    (leftResult := 65165) (rightResult := 14989)
    (working := LeftOperatorMerge73509.working)
    (reconstruction := LeftOperatorMerge73509.reconstruction)
    (leftReference := .predecessor 0 73506 .coefficient) (rightReference := .predecessor 1 73507 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14989.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge73509.operationAgreement
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
end SemanticResult73510

namespace SemanticResult73514
def owner : Owner := ⟨.program ⟨214⟩, ⟨10476⟩⟩
def rawTerms : List Term := Proof.Events287.exact73514RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73514
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73514.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 73511) (rightBinding := 73512)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7190⟩) (rightExpression := ⟨10475⟩)
    (transferEvent := 73513)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73510.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult73505.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult73514

namespace SemanticResult73520
def owner : Owner := ⟨.program ⟨214⟩, ⟨10477⟩⟩
def rawTerms : List Term := Proof.Events287.exact73520RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 73520
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73520.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 73517) (survivorTransfer := 73518)
    (survivorEvent := 73519) (resultEvent := resultEvent)
    (rightCoefficientProducer := 14980)
    (owner := owner) (leftOwner := SemanticResult73514.owner)
    (rightOwner := SemanticResult14981.owner)
    (leftResult := 73514) (rightResult := 14981)
    (leftBinding := 73515) (rightBinding := 73516)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10476⟩) (rightExpression := ⟨86⟩)
    (leftActual := SemanticResult73514.actual selector witness)
    (rightActual := SemanticResult14981.actual selector witness)
    (leftRaw := SemanticResult73514.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨86⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound14980.actual selector witness)
    (survivorMagnitude := LeftBound73518.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73514.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14981.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14980.derived selector witness)
  · exact LeftBound73518.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult73520

namespace SemanticResult73528
def owner : Owner := ⟨.program ⟨214⟩, ⟨10478⟩⟩
def rawTerms : List Term := Proof.Events287.exact73528RawTerms
def summary : Bound := (.finite 1664)
def resultEvent : Nat := 73528
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73528.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨2, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73526.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge73526.frameStart)
    (owner := owner) (leftOwner := SemanticResult73520.owner)
    (rightOwner := SemanticResult3480.owner)
    (leftResult := 73520) (rightResult := 3480)
    (leftActual := SemanticResult73520.actual selector witness)
    (rightActual := SemanticResult3480.actual selector witness)
    (leftRaw := SemanticResult73520.rawTerms)
    (rightRaw := SemanticResult3480.rawTerms)
    (working := LeftOperatorMerge73526.working)
    (leftBinding := 73521) (rightBinding := 73522)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10477⟩) (rightExpression := ⟨9395⟩)
    (coefficientTransfer := 73523) (summaryTransfer := 73525)
    (rightCoefficientProducer := 3479)
    (rightSummaryTransfer := 73524)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨2, by decide⟩)
    (rightRecordedMaximum := 2)
    (rightSummaryMaximum := ⟨2, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge73526.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority3479.actual selector witness)
    (summaryMagnitude := LeftBound73525.actual selector witness)
    (reconstruction := LeftOperatorMerge73526.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult73520.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3480.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3479.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority3479.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge73526.operationAgreement
  · exact LeftBound73525.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73526.working summary) := by
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
end SemanticResult73528

namespace SemanticResult73533
def owner : Owner := ⟨.program ⟨214⟩, ⟨9396⟩⟩
def rawTerms : List Term := Proof.Events287.exact73533RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 73533
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult73533.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge73532.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge73532.frameStart)
    (transferEvent := 73531) (owner := owner)
    (leftResult := 3480) (rightResult := 65295)
    (working := LeftOperatorMerge73532.working)
    (reconstruction := LeftOperatorMerge73532.reconstruction)
    (leftReference := .predecessor 0 73529 .coefficient) (rightReference := .predecessor 1 73530 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3480.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge73532.operationAgreement
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
end SemanticResult73533

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
