import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard230
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard165
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard193
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard197
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard200
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard204
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard208
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard211
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard215
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard219
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard222
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard226
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard229

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult30060
def owner : Owner := ⟨.program ⟨214⟩, ⟨15275⟩⟩
def rawTerms : List Term := Proof.Events117.exact30060RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 30060
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30060.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge30059.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge30059.frameStart)
    (transferEvent := 30058) (owner := owner)
    (leftResult := 30032) (rightResult := 30055)
    (working := LeftOperatorMerge30059.working)
    (reconstruction := LeftOperatorMerge30059.reconstruction)
    (leftReference := .predecessor 0 30056 .coefficient) (rightReference := .predecessor 1 30057 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult30032.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult30055.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge30059.operationAgreement
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
end SemanticResult30060

namespace SemanticResult30063
def owner : Owner := ⟨.program ⟨214⟩, ⟨6709⟩⟩
def rawTerms : List Term := Proof.Events117.exact30063RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 30063
def producerEvent : Nat := 30062
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30063.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 29970, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult30063

namespace SemanticResult30067
def owner : Owner := ⟨.program ⟨214⟩, ⟨15276⟩⟩
def rawTerms : List Term := Proof.Events117.exact30067RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 30067
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30067.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 30064) (rightBinding := 30065)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6709⟩) (rightExpression := ⟨15275⟩)
    (transferEvent := 30066)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult30063.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult30060.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult30067

namespace SemanticResult30071
def owner : Owner := ⟨.program ⟨214⟩, ⟨26398⟩⟩
def rawTerms : List Term := Proof.Events117.exact30071RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 30071
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30071.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 30068) (rightBinding := 30069)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15276⟩) (rightExpression := ⟨26395⟩)
    (transferEvent := 30070)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult30067.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult30052.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult30071

namespace SemanticResult30080
def owner : Owner := ⟨.program ⟨214⟩, ⟨20407⟩⟩
def rawTerms : List Term := Proof.Events117.exact30080RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 30080
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30080.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge29915.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge29915.frameStart)
    (owner := owner) (leftOwner := SemanticResult21512.owner)
    (rightOwner := SemanticResult29909.owner)
    (leftResult := 21512) (rightResult := 29909)
    (leftActual := SemanticResult21512.actual selector witness)
    (rightActual := SemanticResult29909.actual selector witness)
    (leftRaw := SemanticResult21512.rawTerms)
    (rightRaw := SemanticResult29909.rawTerms)
    (working := LeftOperatorMerge29915.working)
    (leftBinding := 29910) (rightBinding := 29911)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5559⟩) (rightExpression := ⟨20406⟩)
    (coefficientTransfer := 29912) (summaryTransfer := 29914)
    (rightCoefficientProducer := 29908)
    (rightSummaryTransfer := 29913)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge29915.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound29908.actual selector witness)
    (summaryMagnitude := LeftBound29914.actual selector witness)
    (reconstruction := LeftOperatorMerge29915.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21512.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult29909.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29908.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound29908.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge29915.operationAgreement
  · exact LeftBound29914.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge29915.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 30075 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26394⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14804⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23730⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26394⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14804⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23730⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge29915.working
    [{ coefficient := (1), key := LeftRelationMerge30075.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge30075.frameStart
      LeftRelationMerge30075.owner (.relation 30075) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge30075.deltas
    rows := LeftRelationMerge30075.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge29915.working LeftRelationMerge30075.source
        (relationContext LeftRelationMerge30075.source
          LeftRelationMerge30075.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge29915.working, LeftRelationMerge30075.deltas,
    LeftRelationMerge30075.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 30075)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨20407⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20404⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20404⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge29915.working) (working := relationWorking0)
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
end SemanticResult30080

namespace SemanticResult30087
def owner : Owner := ⟨.program ⟨214⟩, ⟨26397⟩⟩
def rawTerms : List Term := Proof.Events117.exact30087RawTerms
def summary : Bound := (.finite 1291889174379421642752)
def resultEvent : Nat := 30087
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30087.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge30084.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult30080.owner)
    (rightOwner := SemanticResult29902.owner)
    (leftResult := 30080) (rightResult := 29902)
    (leftActual := SemanticResult30080.actual selector witness)
    (rightActual := SemanticResult29902.actual selector witness)
    (leftRaw := SemanticResult30080.rawTerms)
    (rightRaw := SemanticResult29902.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1291889172568118132736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 30081) (rightBinding := 30082)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20407⟩) (rightExpression := ⟨26396⟩)
    (coefficientTransfer := 30083) (summaryTransfer := 30086)
    (base := LeftOperatorMerge30084.base)
    (reconstruction := LeftOperatorMerge30084.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult30080.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult29902.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge30084.operationAgreement
  · rfl
  · decide
end SemanticResult30087

namespace SemanticResult30092
def owner : Owner := ⟨.program ⟨214⟩, ⟨26607⟩⟩
def rawTerms : List Term := Proof.Events117.exact30092RawTerms
def summary : Bound := (.finite 2583789554981353578496)
def resultEvent : Nat := 30092
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30092.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult30087.owner)
    (rightOwner := SemanticResult29605.owner)
    (leftResult := 30087) (rightResult := 29605)
    (leftActual := SemanticResult30087.actual selector witness)
    (rightActual := SemanticResult29605.actual selector witness)
    (leftRaw := SemanticResult30087.rawTerms)
    (rightRaw := SemanticResult29605.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1291889174379421642752)
    (rightMaximum := 1291900380601931935744) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 30088) (rightBinding := 30089)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26397⟩) (rightExpression := ⟨26606⟩)
    (transferEvent := 30090) (summaryTransferEvent := 30091)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult30087.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult29605.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult30092

namespace SemanticResult30097
def owner : Owner := ⟨.program ⟨214⟩, ⟨26824⟩⟩
def rawTerms : List Term := Proof.Events117.exact30097RawTerms
def summary : Bound := (.finite 3875701141805795807232)
def resultEvent : Nat := 30097
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30097.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult30092.owner)
    (rightOwner := SemanticResult29123.owner)
    (leftResult := 30092) (rightResult := 29123)
    (leftActual := SemanticResult30092.actual selector witness)
    (rightActual := SemanticResult29123.actual selector witness)
    (leftRaw := SemanticResult30092.rawTerms)
    (rightRaw := SemanticResult29123.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2583789554981353578496)
    (rightMaximum := 1291911586824442228736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 30093) (rightBinding := 30094)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26607⟩) (rightExpression := ⟨26823⟩)
    (transferEvent := 30095) (summaryTransferEvent := 30096)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult30092.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult29123.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult30097

namespace SemanticResult30102
def owner : Owner := ⟨.program ⟨214⟩, ⟨27041⟩⟩
def rawTerms : List Term := Proof.Events117.exact30102RawTerms
def summary : Bound := (.finite 5167635141075258621952)
def resultEvent : Nat := 30102
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30102.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult30097.owner)
    (rightOwner := SemanticResult28641.owner)
    (leftResult := 30097) (rightResult := 28641)
    (leftActual := SemanticResult30097.actual selector witness)
    (rightActual := SemanticResult28641.actual selector witness)
    (leftRaw := SemanticResult30097.rawTerms)
    (rightRaw := SemanticResult28641.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3875701141805795807232)
    (rightMaximum := 1291933999269462814720) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 30098) (rightBinding := 30099)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26824⟩) (rightExpression := ⟨27040⟩)
    (transferEvent := 30100) (summaryTransferEvent := 30101)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult30097.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult28641.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult30102

namespace SemanticResult30107
def owner : Owner := ⟨.program ⟨214⟩, ⟨27258⟩⟩
def rawTerms : List Term := Proof.Events117.exact30107RawTerms
def summary : Bound := (.finite 6459613965234762608640)
def resultEvent : Nat := 30107
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30107.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult30102.owner)
    (rightOwner := SemanticResult28159.owner)
    (leftResult := 30102) (rightResult := 28159)
    (leftActual := SemanticResult30102.actual selector witness)
    (rightActual := SemanticResult28159.actual selector witness)
    (leftRaw := SemanticResult30102.rawTerms)
    (rightRaw := SemanticResult28159.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5167635141075258621952)
    (rightMaximum := 1291978824159503986688) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 30103) (rightBinding := 30104)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27041⟩) (rightExpression := ⟨27257⟩)
    (transferEvent := 30105) (summaryTransferEvent := 30106)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult30102.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult28159.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult30107

namespace SemanticResult30112
def owner : Owner := ⟨.program ⟨214⟩, ⟨27475⟩⟩
def rawTerms : List Term := Proof.Events117.exact30112RawTerms
def summary : Bound := (.finite 7751615201839287181312)
def resultEvent : Nat := 30112
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30112.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult30107.owner)
    (rightOwner := SemanticResult27677.owner)
    (leftResult := 30107) (rightResult := 27677)
    (leftActual := SemanticResult30107.actual selector witness)
    (rightActual := SemanticResult27677.actual selector witness)
    (leftRaw := SemanticResult30107.rawTerms)
    (rightRaw := SemanticResult27677.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 6459613965234762608640)
    (rightMaximum := 1292001236604524572672) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 30108) (rightBinding := 30109)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27258⟩) (rightExpression := ⟨27474⟩)
    (transferEvent := 30110) (summaryTransferEvent := 30111)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult30107.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27677.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult30112

namespace SemanticResult30117
def owner : Owner := ⟨.program ⟨214⟩, ⟨27692⟩⟩
def rawTerms : List Term := Proof.Events117.exact30117RawTerms
def summary : Bound := (.finite 9043661263333852925952)
def resultEvent : Nat := 30117
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30117.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult30112.owner)
    (rightOwner := SemanticResult27195.owner)
    (leftResult := 30112) (rightResult := 27195)
    (leftActual := SemanticResult30112.actual selector witness)
    (rightActual := SemanticResult27195.actual selector witness)
    (leftRaw := SemanticResult30112.rawTerms)
    (rightRaw := SemanticResult27195.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 7751615201839287181312)
    (rightMaximum := 1292046061494565744640) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 30113) (rightBinding := 30114)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27475⟩) (rightExpression := ⟨27691⟩)
    (transferEvent := 30115) (summaryTransferEvent := 30116)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult30112.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult27195.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult30117

namespace SemanticResult30122
def owner : Owner := ⟨.program ⟨214⟩, ⟨27909⟩⟩
def rawTerms : List Term := Proof.Events117.exact30122RawTerms
def summary : Bound := (.finite 10335729737273439256576)
def resultEvent : Nat := 30122
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30122.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult30117.owner)
    (rightOwner := SemanticResult26713.owner)
    (leftResult := 30117) (rightResult := 26713)
    (leftActual := SemanticResult30117.actual selector witness)
    (rightActual := SemanticResult26713.actual selector witness)
    (leftRaw := SemanticResult30117.rawTerms)
    (rightRaw := SemanticResult26713.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 9043661263333852925952)
    (rightMaximum := 1292068473939586330624) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 30118) (rightBinding := 30119)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27692⟩) (rightExpression := ⟨27908⟩)
    (transferEvent := 30120) (summaryTransferEvent := 30121)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult30117.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult26713.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult30122

namespace SemanticResult30127
def owner : Owner := ⟨.program ⟨214⟩, ⟨28126⟩⟩
def rawTerms : List Term := Proof.Events117.exact30127RawTerms
def summary : Bound := (.finite 11627843036103066759168)
def resultEvent : Nat := 30127
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30127.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult30122.owner)
    (rightOwner := SemanticResult26231.owner)
    (leftResult := 30122) (rightResult := 26231)
    (leftActual := SemanticResult30122.actual selector witness)
    (rightActual := SemanticResult26231.actual selector witness)
    (leftRaw := SemanticResult30122.rawTerms)
    (rightRaw := SemanticResult26231.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 10335729737273439256576)
    (rightMaximum := 1292113298829627502592) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 30123) (rightBinding := 30124)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27909⟩) (rightExpression := ⟨28125⟩)
    (transferEvent := 30125) (summaryTransferEvent := 30126)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult30122.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult26231.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult30127

namespace SemanticResult30132
def owner : Owner := ⟨.program ⟨214⟩, ⟨28343⟩⟩
def rawTerms : List Term := Proof.Events117.exact30132RawTerms
def summary : Bound := (.finite 12920023572267756019712)
def resultEvent : Nat := 30132
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30132.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult30127.owner)
    (rightOwner := SemanticResult25749.owner)
    (leftResult := 30127) (rightResult := 25749)
    (leftActual := SemanticResult30127.actual selector witness)
    (rightActual := SemanticResult25749.actual selector witness)
    (leftRaw := SemanticResult30127.rawTerms)
    (rightRaw := SemanticResult25749.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 11627843036103066759168)
    (rightMaximum := 1292180536164689260544) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 30128) (rightBinding := 30129)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28126⟩) (rightExpression := ⟨28342⟩)
    (transferEvent := 30130) (summaryTransferEvent := 30131)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult30127.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult25749.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult30132

namespace SemanticResult30137
def owner : Owner := ⟨.program ⟨214⟩, ⟨28560⟩⟩
def rawTerms : List Term := Proof.Events117.exact30137RawTerms
def summary : Bound := (.finite 14212226520877465866240)
def resultEvent : Nat := 30137
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult30137.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult30132.owner)
    (rightOwner := SemanticResult25267.owner)
    (leftResult := 30132) (rightResult := 25267)
    (leftActual := SemanticResult30132.actual selector witness)
    (rightActual := SemanticResult25267.actual selector witness)
    (leftRaw := SemanticResult30132.rawTerms)
    (rightRaw := SemanticResult25267.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 12920023572267756019712)
    (rightMaximum := 1292202948609709846528) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 30133) (rightBinding := 30134)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨28343⟩) (rightExpression := ⟨28559⟩)
    (transferEvent := 30135) (summaryTransferEvent := 30136)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult30132.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult25267.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult30137

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
