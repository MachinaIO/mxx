import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard129
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard058
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard113
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard117
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard121
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard125
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard128

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult15404
def owner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rawTerms : List Term := Proof.Events060.exact15404RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15404
def producerEvent : Nat := 15403
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15404.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 15342, .large, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult15404

namespace SemanticResult15409
def owner : Owner := ⟨.program ⟨214⟩, ⟨14850⟩⟩
def rawTerms : List Term := Proof.Events060.exact15409RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15409
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15409.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge15408.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge15408.frameStart)
    (transferEvent := 15407) (owner := owner)
    (leftResult := 15404) (rightResult := 15402)
    (working := LeftOperatorMerge15408.working)
    (reconstruction := LeftOperatorMerge15408.reconstruction)
    (leftReference := .predecessor 0 15405 .coefficient) (rightReference := .predecessor 1 15406 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult15404.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15402.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge15408.operationAgreement
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
end SemanticResult15409

namespace SemanticResult15412
def owner : Owner := ⟨.program ⟨214⟩, ⟨6690⟩⟩
def rawTerms : List Term := Proof.Events060.exact15412RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15412
def producerEvent : Nat := 15411
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15412.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 15342, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult15412

namespace SemanticResult15416
def owner : Owner := ⟨.program ⟨214⟩, ⟨14851⟩⟩
def rawTerms : List Term := Proof.Events060.exact15416RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15416
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15416.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15413) (rightBinding := 15414)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6690⟩) (rightExpression := ⟨14850⟩)
    (transferEvent := 15415)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15412.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15409.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15416

namespace SemanticResult15424
def owner : Owner := ⟨.program ⟨214⟩, ⟨26407⟩⟩
def rawTerms : List Term := Proof.Events060.exact15424RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15424
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15424.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge15420.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge15420.frameStart)
    (transferEvent := 15419) (owner := owner)
    (leftResult := 15416) (rightResult := 15393)
    (working := LeftOperatorMerge15420.working)
    (reconstruction := LeftOperatorMerge15420.reconstruction)
    (leftReference := .predecessor 0 15417 .coefficient) (rightReference := .predecessor 1 15418 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult15416.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15393.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge15420.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 15421 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26406⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14808⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23733⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23733⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge15420.working
    [{ coefficient := (-1), key := LeftRelationMerge15421.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge15421.frameStart
      LeftRelationMerge15421.owner (.relation 15421) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge15421.deltas
    rows := LeftRelationMerge15421.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge15420.working LeftRelationMerge15421.source
        (relationContext LeftRelationMerge15421.source
          LeftRelationMerge15421.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge15420.working, LeftRelationMerge15421.deltas,
    LeftRelationMerge15421.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 15421)
    (frameStart := 15342) (owner := ⟨.program ⟨214⟩, ⟨26407⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨14808⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26406⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26406⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge15420.working) (working := relationWorking0)
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
end SemanticResult15424

namespace SemanticResult15427
def owner : Owner := ⟨.program ⟨214⟩, ⟨15277⟩⟩
def rawTerms : List Term := Proof.Events060.exact15427RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15427
def producerEvent : Nat := 15426
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15427.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 15342, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult15427

namespace SemanticResult15432
def owner : Owner := ⟨.program ⟨214⟩, ⟨15278⟩⟩
def rawTerms : List Term := Proof.Events060.exact15432RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15432
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15432.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge15431.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge15431.frameStart)
    (transferEvent := 15430) (owner := owner)
    (leftResult := 15404) (rightResult := 15427)
    (working := LeftOperatorMerge15431.working)
    (reconstruction := LeftOperatorMerge15431.reconstruction)
    (leftReference := .predecessor 0 15428 .coefficient) (rightReference := .predecessor 1 15429 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult15404.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15427.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge15431.operationAgreement
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
end SemanticResult15432

namespace SemanticResult15435
def owner : Owner := ⟨.program ⟨214⟩, ⟨6709⟩⟩
def rawTerms : List Term := Proof.Events060.exact15435RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15435
def producerEvent : Nat := 15434
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15435.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 15342, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult15435

namespace SemanticResult15439
def owner : Owner := ⟨.program ⟨214⟩, ⟨15279⟩⟩
def rawTerms : List Term := Proof.Events060.exact15439RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15439
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15439.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15436) (rightBinding := 15437)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6709⟩) (rightExpression := ⟨15278⟩)
    (transferEvent := 15438)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15435.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15432.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15439

namespace SemanticResult15443
def owner : Owner := ⟨.program ⟨214⟩, ⟨26410⟩⟩
def rawTerms : List Term := Proof.Events060.exact15443RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 15443
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15443.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 15440) (rightBinding := 15441)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15279⟩) (rightExpression := ⟨26407⟩)
    (transferEvent := 15442)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15439.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15424.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15443

namespace SemanticResult15452
def owner : Owner := ⟨.program ⟨214⟩, ⟨20411⟩⟩
def rawTerms : List Term := Proof.Events060.exact15452RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 15452
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15452.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge15287.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge15287.frameStart)
    (owner := owner) (leftOwner := SemanticResult6561.owner)
    (rightOwner := SemanticResult15281.owner)
    (leftResult := 6561) (rightResult := 15281)
    (leftActual := SemanticResult6561.actual selector witness)
    (rightActual := SemanticResult15281.actual selector witness)
    (leftRaw := SemanticResult6561.rawTerms)
    (rightRaw := SemanticResult15281.rawTerms)
    (working := LeftOperatorMerge15287.working)
    (leftBinding := 15282) (rightBinding := 15283)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5565⟩) (rightExpression := ⟨20410⟩)
    (coefficientTransfer := 15284) (summaryTransfer := 15286)
    (rightCoefficientProducer := 15280)
    (rightSummaryTransfer := 15285)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge15287.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound15280.actual selector witness)
    (summaryMagnitude := LeftBound15286.actual selector witness)
    (reconstruction := LeftOperatorMerge15287.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult6561.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15281.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15280.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound15280.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge15287.operationAgreement
  · exact LeftBound15286.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge15287.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 15447 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26406⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14808⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23733⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26406⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14808⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23733⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge15287.working
    [{ coefficient := (1), key := LeftRelationMerge15447.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge15447.frameStart
      LeftRelationMerge15447.owner (.relation 15447) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge15447.deltas
    rows := LeftRelationMerge15447.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge15287.working LeftRelationMerge15447.source
        (relationContext LeftRelationMerge15447.source
          LeftRelationMerge15447.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge15287.working, LeftRelationMerge15447.deltas,
    LeftRelationMerge15447.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 15447)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨20411⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20408⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20408⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge15287.working) (working := relationWorking0)
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
end SemanticResult15452

namespace SemanticResult15459
def owner : Owner := ⟨.program ⟨214⟩, ⟨26409⟩⟩
def rawTerms : List Term := Proof.Events060.exact15459RawTerms
def summary : Bound := (.finite 1291889174379421642752)
def resultEvent : Nat := 15459
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15459.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge15456.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult15452.owner)
    (rightOwner := SemanticResult15274.owner)
    (leftResult := 15452) (rightResult := 15274)
    (leftActual := SemanticResult15452.actual selector witness)
    (rightActual := SemanticResult15274.actual selector witness)
    (leftRaw := SemanticResult15452.rawTerms)
    (rightRaw := SemanticResult15274.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1291889172568118132736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 15453) (rightBinding := 15454)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20411⟩) (rightExpression := ⟨26408⟩)
    (coefficientTransfer := 15455) (summaryTransfer := 15458)
    (base := LeftOperatorMerge15456.base)
    (reconstruction := LeftOperatorMerge15456.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15452.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15274.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge15456.operationAgreement
  · rfl
  · decide
end SemanticResult15459

namespace SemanticResult15464
def owner : Owner := ⟨.program ⟨214⟩, ⟨26620⟩⟩
def rawTerms : List Term := Proof.Events060.exact15464RawTerms
def summary : Bound := (.finite 2583789554981353578496)
def resultEvent : Nat := 15464
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15464.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult15459.owner)
    (rightOwner := SemanticResult14958.owner)
    (leftResult := 15459) (rightResult := 14958)
    (leftActual := SemanticResult15459.actual selector witness)
    (rightActual := SemanticResult14958.actual selector witness)
    (leftRaw := SemanticResult15459.rawTerms)
    (rightRaw := SemanticResult14958.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1291889174379421642752)
    (rightMaximum := 1291900380601931935744) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 15460) (rightBinding := 15461)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26409⟩) (rightExpression := ⟨26619⟩)
    (transferEvent := 15462) (summaryTransferEvent := 15463)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15459.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14958.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15464

namespace SemanticResult15469
def owner : Owner := ⟨.program ⟨214⟩, ⟨26837⟩⟩
def rawTerms : List Term := Proof.Events060.exact15469RawTerms
def summary : Bound := (.finite 3875701141805795807232)
def resultEvent : Nat := 15469
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15469.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult15464.owner)
    (rightOwner := SemanticResult14457.owner)
    (leftResult := 15464) (rightResult := 14457)
    (leftActual := SemanticResult15464.actual selector witness)
    (rightActual := SemanticResult14457.actual selector witness)
    (leftRaw := SemanticResult15464.rawTerms)
    (rightRaw := SemanticResult14457.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2583789554981353578496)
    (rightMaximum := 1291911586824442228736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 15465) (rightBinding := 15466)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26620⟩) (rightExpression := ⟨26836⟩)
    (transferEvent := 15467) (summaryTransferEvent := 15468)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15464.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14457.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15469

namespace SemanticResult15474
def owner : Owner := ⟨.program ⟨214⟩, ⟨27054⟩⟩
def rawTerms : List Term := Proof.Events060.exact15474RawTerms
def summary : Bound := (.finite 5167635141075258621952)
def resultEvent : Nat := 15474
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15474.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult15469.owner)
    (rightOwner := SemanticResult13956.owner)
    (leftResult := 15469) (rightResult := 13956)
    (leftActual := SemanticResult15469.actual selector witness)
    (rightActual := SemanticResult13956.actual selector witness)
    (leftRaw := SemanticResult15469.rawTerms)
    (rightRaw := SemanticResult13956.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3875701141805795807232)
    (rightMaximum := 1291933999269462814720) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 15470) (rightBinding := 15471)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26837⟩) (rightExpression := ⟨27053⟩)
    (transferEvent := 15472) (summaryTransferEvent := 15473)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15469.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13956.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15474

namespace SemanticResult15479
def owner : Owner := ⟨.program ⟨214⟩, ⟨27271⟩⟩
def rawTerms : List Term := Proof.Events060.exact15479RawTerms
def summary : Bound := (.finite 6459613965234762608640)
def resultEvent : Nat := 15479
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult15479.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult15474.owner)
    (rightOwner := SemanticResult13455.owner)
    (leftResult := 15474) (rightResult := 13455)
    (leftActual := SemanticResult15474.actual selector witness)
    (rightActual := SemanticResult13455.actual selector witness)
    (leftRaw := SemanticResult15474.rawTerms)
    (rightRaw := SemanticResult13455.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5167635141075258621952)
    (rightMaximum := 1291978824159503986688) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 15475) (rightBinding := 15476)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27054⟩) (rightExpression := ⟨27270⟩)
    (transferEvent := 15477) (summaryTransferEvent := 15478)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult15474.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13455.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult15479

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
