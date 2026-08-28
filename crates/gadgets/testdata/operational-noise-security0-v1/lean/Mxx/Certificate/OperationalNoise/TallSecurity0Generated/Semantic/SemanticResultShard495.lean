import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard495
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard026
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard089
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard090
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard494

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult69126
def owner : Owner := ⟨.program ⟨214⟩, ⟨28509⟩⟩
def rawTerms : List Term := Proof.Events270.exact69126RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69126
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69126.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 69123) (rightBinding := 69124)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16307⟩) (rightExpression := ⟨28505⟩)
    (transferEvent := 69125)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69122.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult69107.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult69126

namespace SemanticResult69135
def owner : Owner := ⟨.program ⟨214⟩, ⟨21831⟩⟩
def rawTerms : List Term := Proof.Events270.exact69135RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 69135
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69135.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge68970.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge68970.frameStart)
    (owner := owner) (leftOwner := SemanticResult65387.owner)
    (rightOwner := SemanticResult68964.owner)
    (leftResult := 65387) (rightResult := 68964)
    (leftActual := SemanticResult65387.actual selector witness)
    (rightActual := SemanticResult68964.actual selector witness)
    (leftRaw := SemanticResult65387.rawTerms)
    (rightRaw := SemanticResult68964.rawTerms)
    (working := LeftOperatorMerge68970.working)
    (leftBinding := 68965) (rightBinding := 68966)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5535⟩) (rightExpression := ⟨21830⟩)
    (coefficientTransfer := 68967) (summaryTransfer := 68969)
    (rightCoefficientProducer := 68963)
    (rightSummaryTransfer := 68968)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge68970.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound68963.actual selector witness)
    (summaryMagnitude := LeftBound68969.actual selector witness)
    (reconstruction := LeftOperatorMerge68970.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult65387.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult68964.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68963.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound68963.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge68970.operationAgreement
  · exact LeftBound68969.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge68970.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 69130 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24348⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28504⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24348⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge68970.working
    [{ coefficient := (1), key := LeftRelationMerge69130.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge69130.frameStart
      LeftRelationMerge69130.owner (.relation 69130) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge69130.deltas
    rows := LeftRelationMerge69130.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge68970.working LeftRelationMerge69130.source
        (relationContext LeftRelationMerge69130.source
          LeftRelationMerge69130.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge68970.working, LeftRelationMerge69130.deltas,
    LeftRelationMerge69130.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 69130)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨21831⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21828⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge68970.working) (working := relationWorking0)
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
end SemanticResult69135

namespace SemanticResult69142
def owner : Owner := ⟨.program ⟨214⟩, ⟨28507⟩⟩
def rawTerms : List Term := Proof.Events270.exact69142RawTerms
def summary : Bound := (.finite 1292202948609709846528)
def resultEvent : Nat := 69142
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69142.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge69139.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult69135.owner)
    (rightOwner := SemanticResult68957.owner)
    (leftResult := 69135) (rightResult := 68957)
    (leftActual := SemanticResult69135.actual selector witness)
    (rightActual := SemanticResult68957.actual selector witness)
    (leftRaw := SemanticResult69135.rawTerms)
    (rightRaw := SemanticResult68957.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292202946798406336512) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 69136) (rightBinding := 69137)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21831⟩) (rightExpression := ⟨28506⟩)
    (coefficientTransfer := 69138) (summaryTransfer := 69141)
    (base := LeftOperatorMerge69139.base)
    (reconstruction := LeftOperatorMerge69139.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69135.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult68957.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge69139.operationAgreement
  · rfl
  · decide
end SemanticResult69142

namespace SemanticResult69149
def owner : Owner := ⟨.program ⟨214⟩, ⟨24285⟩⟩
def rawTerms : List Term := Proof.Events270.exact69149RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69149
def producerEvent : Nat := 69148
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69149.actual selector witness
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
end SemanticResult69149

namespace SemanticResult69152
def owner : Owner := ⟨.program ⟨214⟩, ⟨28287⟩⟩
def rawTerms : List Term := Proof.Events270.exact69152RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69152
def producerEvent : Nat := 69151
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69152.actual selector witness
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
end SemanticResult69152

namespace SemanticResult69159
def owner : Owner := ⟨.program ⟨214⟩, ⟨23666⟩⟩
def rawTerms : List Term := Proof.Events270.exact69159RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69159
def producerEvent : Nat := 69158
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69159.actual selector witness
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
end SemanticResult69159

namespace SemanticResult69162
def owner : Owner := ⟨.program ⟨214⟩, ⟨26215⟩⟩
def rawTerms : List Term := Proof.Events270.exact69162RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69162
def producerEvent : Nat := 69161
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69162.actual selector witness
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
end SemanticResult69162

namespace SemanticResult69167
def owner : Owner := ⟨.program ⟨214⟩, ⟨11634⟩⟩
def rawTerms : List Term := Proof.Events270.exact69167RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69167
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69167.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge69166.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge69166.frameStart)
    (transferEvent := 69165) (owner := owner)
    (leftResult := 3270) (rightResult := 65295)
    (working := LeftOperatorMerge69166.working)
    (reconstruction := LeftOperatorMerge69166.reconstruction)
    (leftReference := .predecessor 0 69163 .coefficient) (rightReference := .predecessor 1 69164 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3270.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge69166.operationAgreement
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
end SemanticResult69167

namespace SemanticResult69172
def owner : Owner := ⟨.program ⟨214⟩, ⟨7199⟩⟩
def rawTerms : List Term := Proof.Events270.exact69172RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69172
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69172.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge69171.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge69171.frameStart)
    (transferEvent := 69170) (owner := owner)
    (leftResult := 65165) (rightResult := 10480)
    (working := LeftOperatorMerge69171.working)
    (reconstruction := LeftOperatorMerge69171.reconstruction)
    (leftReference := .predecessor 0 69168 .coefficient) (rightReference := .predecessor 1 69169 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10480.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge69171.operationAgreement
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
end SemanticResult69172

namespace SemanticResult69176
def owner : Owner := ⟨.program ⟨214⟩, ⟨11635⟩⟩
def rawTerms : List Term := Proof.Events270.exact69176RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69176
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69176.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 69173) (rightBinding := 69174)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7199⟩) (rightExpression := ⟨11634⟩)
    (transferEvent := 69175)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69172.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult69167.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult69176

namespace SemanticResult69182
def owner : Owner := ⟨.program ⟨214⟩, ⟨11636⟩⟩
def rawTerms : List Term := Proof.Events270.exact69182RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 69182
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69182.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 69179) (survivorTransfer := 69180)
    (survivorEvent := 69181) (resultEvent := resultEvent)
    (rightCoefficientProducer := 10471)
    (owner := owner) (leftOwner := SemanticResult69176.owner)
    (rightOwner := SemanticResult10472.owner)
    (leftResult := 69176) (rightResult := 10472)
    (leftBinding := 69177) (rightBinding := 69178)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11635⟩) (rightExpression := ⟨95⟩)
    (leftActual := SemanticResult69176.actual selector witness)
    (rightActual := SemanticResult10472.actual selector witness)
    (leftRaw := SemanticResult69176.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨95⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound10471.actual selector witness)
    (survivorMagnitude := LeftBound69180.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69176.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10472.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10471.derived selector witness)
  · exact LeftBound69180.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult69182

namespace SemanticResult69190
def owner : Owner := ⟨.program ⟨214⟩, ⟨14635⟩⟩
def rawTerms : List Term := Proof.Events270.exact69190RawTerms
def summary : Bound := (.finite 23296)
def resultEvent : Nat := 69190
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69190.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨28, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge69188.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge69188.frameStart)
    (owner := owner) (leftOwner := SemanticResult69182.owner)
    (rightOwner := SemanticResult3273.owner)
    (leftResult := 69182) (rightResult := 3273)
    (leftActual := SemanticResult69182.actual selector witness)
    (rightActual := SemanticResult3273.actual selector witness)
    (leftRaw := SemanticResult69182.rawTerms)
    (rightRaw := SemanticResult3273.rawTerms)
    (working := LeftOperatorMerge69188.working)
    (leftBinding := 69183) (rightBinding := 69184)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11636⟩) (rightExpression := ⟨14632⟩)
    (coefficientTransfer := 69185) (summaryTransfer := 69187)
    (rightCoefficientProducer := 3272)
    (rightSummaryTransfer := 69186)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨28, by decide⟩)
    (rightRecordedMaximum := 28)
    (rightSummaryMaximum := ⟨28, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge69188.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority3272.actual selector witness)
    (summaryMagnitude := LeftBound69187.actual selector witness)
    (reconstruction := LeftOperatorMerge69188.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69182.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3273.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3272.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority3272.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge69188.operationAgreement
  · exact LeftBound69187.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge69188.working summary) := by
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
end SemanticResult69190

namespace SemanticResult69195
def owner : Owner := ⟨.program ⟨214⟩, ⟨14636⟩⟩
def rawTerms : List Term := Proof.Events270.exact69195RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69195
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69195.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge69194.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge69194.frameStart)
    (transferEvent := 69193) (owner := owner)
    (leftResult := 3273) (rightResult := 65295)
    (working := LeftOperatorMerge69194.working)
    (reconstruction := LeftOperatorMerge69194.reconstruction)
    (leftReference := .predecessor 0 69191 .coefficient) (rightReference := .predecessor 1 69192 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3273.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge69194.operationAgreement
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
end SemanticResult69195

namespace SemanticResult69200
def owner : Owner := ⟨.program ⟨214⟩, ⟨7180⟩⟩
def rawTerms : List Term := Proof.Events270.exact69200RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69200
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69200.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge69199.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge69199.frameStart)
    (transferEvent := 69198) (owner := owner)
    (leftResult := 65165) (rightResult := 10521)
    (working := LeftOperatorMerge69199.working)
    (reconstruction := LeftOperatorMerge69199.reconstruction)
    (leftReference := .predecessor 0 69196 .coefficient) (rightReference := .predecessor 1 69197 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10521.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge69199.operationAgreement
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
end SemanticResult69200

namespace SemanticResult69204
def owner : Owner := ⟨.program ⟨214⟩, ⟨14637⟩⟩
def rawTerms : List Term := Proof.Events270.exact69204RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 69204
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69204.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 69201) (rightBinding := 69202)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7180⟩) (rightExpression := ⟨14636⟩)
    (transferEvent := 69203)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69200.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult69195.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult69204

namespace SemanticResult69210
def owner : Owner := ⟨.program ⟨214⟩, ⟨14638⟩⟩
def rawTerms : List Term := Proof.Events270.exact69210RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 69210
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult69210.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 69207) (survivorTransfer := 69208)
    (survivorEvent := 69209) (resultEvent := resultEvent)
    (rightCoefficientProducer := 10512)
    (owner := owner) (leftOwner := SemanticResult69204.owner)
    (rightOwner := SemanticResult10513.owner)
    (leftResult := 69204) (rightResult := 10513)
    (leftBinding := 69205) (rightBinding := 69206)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14637⟩) (rightExpression := ⟨76⟩)
    (leftActual := SemanticResult69204.actual selector witness)
    (rightActual := SemanticResult10513.actual selector witness)
    (leftRaw := SemanticResult69204.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨76⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound10512.actual selector witness)
    (survivorMagnitude := LeftBound69208.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult69204.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10513.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10512.derived selector witness)
  · exact LeftBound69208.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult69210

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
