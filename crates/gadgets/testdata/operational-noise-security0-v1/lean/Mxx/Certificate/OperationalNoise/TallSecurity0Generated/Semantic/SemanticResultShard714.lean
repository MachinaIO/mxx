import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard714
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard038
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard109
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard110
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard667
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard713

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult99987
def owner : Owner := ⟨.program ⟨214⟩, ⟨27402⟩⟩
def rawTerms : List Term := Proof.Events390.exact99987RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99987
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99987.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 99984) (rightBinding := 99985)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15743⟩) (rightExpression := ⟨27398⟩)
    (transferEvent := 99986)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99983.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99968.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99987

namespace SemanticResult99996
def owner : Owner := ⟨.program ⟨214⟩, ⟨21104⟩⟩
def rawTerms : List Term := Proof.Events390.exact99996RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 99996
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult99996.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99855.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge99855.frameStart)
    (owner := owner) (leftOwner := SemanticResult94462.owner)
    (rightOwner := SemanticResult99849.owner)
    (leftResult := 94462) (rightResult := 99849)
    (leftActual := SemanticResult94462.actual selector witness)
    (rightActual := SemanticResult99849.actual selector witness)
    (leftRaw := SemanticResult94462.rawTerms)
    (rightRaw := SemanticResult99849.rawTerms)
    (working := LeftOperatorMerge99855.working)
    (leftBinding := 99850) (rightBinding := 99851)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5509⟩) (rightExpression := ⟨21103⟩)
    (coefficientTransfer := 99852) (summaryTransfer := 99854)
    (rightCoefficientProducer := 99848)
    (rightSummaryTransfer := 99853)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge99855.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound99848.actual selector witness)
    (summaryMagnitude := LeftBound99854.actual selector witness)
    (reconstruction := LeftOperatorMerge99855.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94462.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99849.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99848.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound99848.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge99855.operationAgreement
  · exact LeftBound99854.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99855.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 99991 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27397⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15692⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24027⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27397⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15692⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24027⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15741⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge99855.working
    [{ coefficient := (1), key := LeftRelationMerge99991.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge99991.frameStart
      LeftRelationMerge99991.owner (.relation 99991) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge99991.deltas
    rows := LeftRelationMerge99991.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge99855.working LeftRelationMerge99991.source
        (relationContext LeftRelationMerge99991.source
          LeftRelationMerge99991.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge99855.working, LeftRelationMerge99991.deltas,
    LeftRelationMerge99991.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 99991)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨21104⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21101⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21101⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge99855.working) (working := relationWorking0)
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
end SemanticResult99996

namespace SemanticResult100003
def owner : Owner := ⟨.program ⟨214⟩, ⟨27400⟩⟩
def rawTerms : List Term := Proof.Events390.exact100003RawTerms
def summary : Bound := (.finite 1292001236604524572672)
def resultEvent : Nat := 100003
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100003.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge100000.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult99996.owner)
    (rightOwner := SemanticResult99842.owner)
    (leftResult := 99996) (rightResult := 99842)
    (leftActual := SemanticResult99996.actual selector witness)
    (rightActual := SemanticResult99842.actual selector witness)
    (leftRaw := SemanticResult99996.rawTerms)
    (rightRaw := SemanticResult99842.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292001234793221062656) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 99997) (rightBinding := 99998)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21104⟩) (rightExpression := ⟨27399⟩)
    (coefficientTransfer := 99999) (summaryTransfer := 100002)
    (base := LeftOperatorMerge100000.base)
    (reconstruction := LeftOperatorMerge100000.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99996.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99842.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge100000.operationAgreement
  · rfl
  · decide
end SemanticResult100003

namespace SemanticResult100010
def owner : Owner := ⟨.program ⟨214⟩, ⟨23964⟩⟩
def rawTerms : List Term := Proof.Events390.exact100010RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100010
def producerEvent : Nat := 100009
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100010.actual selector witness
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
end SemanticResult100010

namespace SemanticResult100013
def owner : Owner := ⟨.program ⟨214⟩, ⟨27180⟩⟩
def rawTerms : List Term := Proof.Events390.exact100013RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100013
def producerEvent : Nat := 100012
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100013.actual selector witness
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
end SemanticResult100013

namespace SemanticResult100020
def owner : Owner := ⟨.program ⟨214⟩, ⟨23452⟩⟩
def rawTerms : List Term := Proof.Events390.exact100020RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100020
def producerEvent : Nat := 100019
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100020.actual selector witness
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
end SemanticResult100020

namespace SemanticResult100023
def owner : Owner := ⟨.program ⟨214⟩, ⟨25822⟩⟩
def rawTerms : List Term := Proof.Events390.exact100023RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100023
def producerEvent : Nat := 100022
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100023.actual selector witness
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
end SemanticResult100023

namespace SemanticResult100028
def owner : Owner := ⟨.program ⟨214⟩, ⟨11206⟩⟩
def rawTerms : List Term := Proof.Events390.exact100028RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100028
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100028.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100027.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge100027.frameStart)
    (transferEvent := 100026) (owner := owner)
    (leftResult := 4865) (rightResult := 32)
    (working := LeftOperatorMerge100027.working)
    (reconstruction := LeftOperatorMerge100027.reconstruction)
    (leftReference := .predecessor 0 100024 .coefficient) (rightReference := .predecessor 1 100025 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4865.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge100027.operationAgreement
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
end SemanticResult100028

namespace SemanticResult100033
def owner : Owner := ⟨.program ⟨214⟩, ⟨7113⟩⟩
def rawTerms : List Term := Proof.Events390.exact100033RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100033
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100033.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100032.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge100032.frameStart)
    (transferEvent := 100031) (owner := owner)
    (leftResult := 27) (rightResult := 12985)
    (working := LeftOperatorMerge100032.working)
    (reconstruction := LeftOperatorMerge100032.reconstruction)
    (leftReference := .predecessor 0 100029 .coefficient) (rightReference := .predecessor 1 100030 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12985.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge100032.operationAgreement
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
end SemanticResult100033

namespace SemanticResult100037
def owner : Owner := ⟨.program ⟨214⟩, ⟨11207⟩⟩
def rawTerms : List Term := Proof.Events390.exact100037RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100037
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100037.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100034) (rightBinding := 100035)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7113⟩) (rightExpression := ⟨11206⟩)
    (transferEvent := 100036)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100033.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100028.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100037

namespace SemanticResult100043
def owner : Owner := ⟨.program ⟨214⟩, ⟨11208⟩⟩
def rawTerms : List Term := Proof.Events390.exact100043RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 100043
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100043.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 100040) (survivorTransfer := 100041)
    (survivorEvent := 100042) (resultEvent := resultEvent)
    (rightCoefficientProducer := 12976)
    (owner := owner) (leftOwner := SemanticResult100037.owner)
    (rightOwner := SemanticResult12977.owner)
    (leftResult := 100037) (rightResult := 12977)
    (leftBinding := 100038) (rightBinding := 100039)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11207⟩) (rightExpression := ⟨90⟩)
    (leftActual := SemanticResult100037.actual selector witness)
    (rightActual := SemanticResult12977.actual selector witness)
    (leftRaw := SemanticResult100037.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨90⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound12976.actual selector witness)
    (survivorMagnitude := LeftBound100041.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100037.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12977.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12976.derived selector witness)
  · exact LeftBound100041.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult100043

namespace SemanticResult100051
def owner : Owner := ⟨.program ⟨214⟩, ⟨13532⟩⟩
def rawTerms : List Term := Proof.Events390.exact100051RawTerms
def summary : Bound := (.finite 8320)
def resultEvent : Nat := 100051
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100051.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨10, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100049.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge100049.frameStart)
    (owner := owner) (leftOwner := SemanticResult100043.owner)
    (rightOwner := SemanticResult4868.owner)
    (leftResult := 100043) (rightResult := 4868)
    (leftActual := SemanticResult100043.actual selector witness)
    (rightActual := SemanticResult4868.actual selector witness)
    (leftRaw := SemanticResult100043.rawTerms)
    (rightRaw := SemanticResult4868.rawTerms)
    (working := LeftOperatorMerge100049.working)
    (leftBinding := 100044) (rightBinding := 100045)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11208⟩) (rightExpression := ⟨13529⟩)
    (coefficientTransfer := 100046) (summaryTransfer := 100048)
    (rightCoefficientProducer := 4867)
    (rightSummaryTransfer := 100047)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨10, by decide⟩)
    (rightRecordedMaximum := 10)
    (rightSummaryMaximum := ⟨10, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge100049.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority4867.actual selector witness)
    (summaryMagnitude := LeftBound100048.actual selector witness)
    (reconstruction := LeftOperatorMerge100049.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100043.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4868.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4867.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority4867.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge100049.operationAgreement
  · exact LeftBound100048.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100049.working summary) := by
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
end SemanticResult100051

namespace SemanticResult100056
def owner : Owner := ⟨.program ⟨214⟩, ⟨13533⟩⟩
def rawTerms : List Term := Proof.Events390.exact100056RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100056
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100056.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100055.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge100055.frameStart)
    (transferEvent := 100054) (owner := owner)
    (leftResult := 4868) (rightResult := 32)
    (working := LeftOperatorMerge100055.working)
    (reconstruction := LeftOperatorMerge100055.reconstruction)
    (leftReference := .predecessor 0 100052 .coefficient) (rightReference := .predecessor 1 100053 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4868.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge100055.operationAgreement
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
end SemanticResult100056

namespace SemanticResult100061
def owner : Owner := ⟨.program ⟨214⟩, ⟨7130⟩⟩
def rawTerms : List Term := Proof.Events390.exact100061RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100061
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100061.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge100060.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge100060.frameStart)
    (transferEvent := 100059) (owner := owner)
    (leftResult := 27) (rightResult := 13026)
    (working := LeftOperatorMerge100060.working)
    (reconstruction := LeftOperatorMerge100060.reconstruction)
    (leftReference := .predecessor 0 100057 .coefficient) (rightReference := .predecessor 1 100058 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13026.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge100060.operationAgreement
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
end SemanticResult100061

namespace SemanticResult100065
def owner : Owner := ⟨.program ⟨214⟩, ⟨13534⟩⟩
def rawTerms : List Term := Proof.Events390.exact100065RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 100065
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100065.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 100062) (rightBinding := 100063)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7130⟩) (rightExpression := ⟨13533⟩)
    (transferEvent := 100064)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100061.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100056.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult100065

namespace SemanticResult100071
def owner : Owner := ⟨.program ⟨214⟩, ⟨13535⟩⟩
def rawTerms : List Term := Proof.Events390.exact100071RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 100071
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult100071.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 100068) (survivorTransfer := 100069)
    (survivorEvent := 100070) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13017)
    (owner := owner) (leftOwner := SemanticResult100065.owner)
    (rightOwner := SemanticResult13018.owner)
    (leftResult := 100065) (rightResult := 13018)
    (leftBinding := 100066) (rightBinding := 100067)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13534⟩) (rightExpression := ⟨107⟩)
    (leftActual := SemanticResult100065.actual selector witness)
    (rightActual := SemanticResult13018.actual selector witness)
    (leftRaw := SemanticResult100065.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨107⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13017.actual selector witness)
    (survivorMagnitude := LeftBound100069.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult100065.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13018.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13017.derived selector witness)
  · exact LeftBound100069.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult100071

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
