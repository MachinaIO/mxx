import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard197
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard008
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard093
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard094
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard165
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard196

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult25733
def owner : Owner := ⟨.program ⟨214⟩, ⟨28344⟩⟩
def rawTerms : List Term := Proof.Events100.exact25733RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 25733
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25733.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 25730) (rightBinding := 25731)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18391⟩) (rightExpression := ⟨28340⟩)
    (transferEvent := 25732)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult25729.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult25714.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult25733

namespace SemanticResult25742
def owner : Owner := ⟨.program ⟨214⟩, ⟨21703⟩⟩
def rawTerms : List Term := Proof.Events100.exact25742RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 25742
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25742.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge25577.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge25577.frameStart)
    (owner := owner) (leftOwner := SemanticResult21512.owner)
    (rightOwner := SemanticResult25571.owner)
    (leftResult := 21512) (rightResult := 25571)
    (leftActual := SemanticResult21512.actual selector witness)
    (rightActual := SemanticResult25571.actual selector witness)
    (leftRaw := SemanticResult21512.rawTerms)
    (rightRaw := SemanticResult25571.rawTerms)
    (working := LeftOperatorMerge25577.working)
    (leftBinding := 25572) (rightBinding := 25573)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5559⟩) (rightExpression := ⟨21702⟩)
    (coefficientTransfer := 25574) (summaryTransfer := 25576)
    (rightCoefficientProducer := 25570)
    (rightSummaryTransfer := 25575)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge25577.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound25570.actual selector witness)
    (summaryMagnitude := LeftBound25576.actual selector witness)
    (reconstruction := LeftOperatorMerge25577.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult21512.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult25571.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25570.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound25570.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge25577.operationAgreement
  · exact LeftBound25576.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge25577.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 25737 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24297⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18379⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24297⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18379⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge25577.working
    [{ coefficient := (1), key := LeftRelationMerge25737.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge25737.frameStart
      LeftRelationMerge25737.owner (.relation 25737) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge25737.deltas
    rows := LeftRelationMerge25737.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge25577.working LeftRelationMerge25737.source
        (relationContext LeftRelationMerge25737.source
          LeftRelationMerge25737.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge25577.working, LeftRelationMerge25737.deltas,
    LeftRelationMerge25737.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 25737)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨21703⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21700⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21700⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge25577.working) (working := relationWorking0)
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
end SemanticResult25742

namespace SemanticResult25749
def owner : Owner := ⟨.program ⟨214⟩, ⟨28342⟩⟩
def rawTerms : List Term := Proof.Events100.exact25749RawTerms
def summary : Bound := (.finite 1292180536164689260544)
def resultEvent : Nat := 25749
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25749.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge25746.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult25742.owner)
    (rightOwner := SemanticResult25564.owner)
    (leftResult := 25742) (rightResult := 25564)
    (leftActual := SemanticResult25742.actual selector witness)
    (rightActual := SemanticResult25564.actual selector witness)
    (leftRaw := SemanticResult25742.rawTerms)
    (rightRaw := SemanticResult25564.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292180534353385750528) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 25743) (rightBinding := 25744)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21703⟩) (rightExpression := ⟨28341⟩)
    (coefficientTransfer := 25745) (summaryTransfer := 25748)
    (base := LeftOperatorMerge25746.base)
    (reconstruction := LeftOperatorMerge25746.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult25742.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult25564.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge25746.operationAgreement
  · rfl
  · decide
end SemanticResult25749

namespace SemanticResult25756
def owner : Owner := ⟨.program ⟨214⟩, ⟨24234⟩⟩
def rawTerms : List Term := Proof.Events100.exact25756RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 25756
def producerEvent : Nat := 25755
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25756.actual selector witness
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
end SemanticResult25756

namespace SemanticResult25759
def owner : Owner := ⟨.program ⟨214⟩, ⟨28122⟩⟩
def rawTerms : List Term := Proof.Events100.exact25759RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 25759
def producerEvent : Nat := 25758
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25759.actual selector witness
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
end SemanticResult25759

namespace SemanticResult25766
def owner : Owner := ⟨.program ⟨214⟩, ⟨23632⟩⟩
def rawTerms : List Term := Proof.Events100.exact25766RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 25766
def producerEvent : Nat := 25765
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25766.actual selector witness
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
end SemanticResult25766

namespace SemanticResult25769
def owner : Owner := ⟨.program ⟨214⟩, ⟨26158⟩⟩
def rawTerms : List Term := Proof.Events100.exact25769RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 25769
def producerEvent : Nat := 25768
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25769.actual selector witness
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
end SemanticResult25769

namespace SemanticResult25774
def owner : Owner := ⟨.program ⟨214⟩, ⟨11566⟩⟩
def rawTerms : List Term := Proof.Events100.exact25774RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 25774
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25774.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge25773.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge25773.frameStart)
    (transferEvent := 25772) (owner := owner)
    (leftResult := 1049) (rightResult := 21420)
    (working := LeftOperatorMerge25773.working)
    (reconstruction := LeftOperatorMerge25773.reconstruction)
    (leftReference := .predecessor 0 25770 .coefficient) (rightReference := .predecessor 1 25771 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1049.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge25773.operationAgreement
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
end SemanticResult25774

namespace SemanticResult25779
def owner : Owner := ⟨.program ⟨214⟩, ⟨7350⟩⟩
def rawTerms : List Term := Proof.Events100.exact25779RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 25779
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25779.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge25778.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge25778.frameStart)
    (transferEvent := 25777) (owner := owner)
    (leftResult := 21290) (rightResult := 10981)
    (working := LeftOperatorMerge25778.working)
    (reconstruction := LeftOperatorMerge25778.reconstruction)
    (leftReference := .predecessor 0 25775 .coefficient) (rightReference := .predecessor 1 25776 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10981.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge25778.operationAgreement
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
end SemanticResult25779

namespace SemanticResult25783
def owner : Owner := ⟨.program ⟨214⟩, ⟨11567⟩⟩
def rawTerms : List Term := Proof.Events100.exact25783RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 25783
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25783.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 25780) (rightBinding := 25781)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7350⟩) (rightExpression := ⟨11566⟩)
    (transferEvent := 25782)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult25779.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult25774.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult25783

namespace SemanticResult25789
def owner : Owner := ⟨.program ⟨214⟩, ⟨11568⟩⟩
def rawTerms : List Term := Proof.Events100.exact25789RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 25789
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25789.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 25786) (survivorTransfer := 25787)
    (survivorEvent := 25788) (resultEvent := resultEvent)
    (rightCoefficientProducer := 10972)
    (owner := owner) (leftOwner := SemanticResult25783.owner)
    (rightOwner := SemanticResult10973.owner)
    (leftResult := 25783) (rightResult := 10973)
    (leftBinding := 25784) (rightBinding := 25785)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11567⟩) (rightExpression := ⟨94⟩)
    (leftActual := SemanticResult25783.actual selector witness)
    (rightActual := SemanticResult10973.actual selector witness)
    (leftRaw := SemanticResult25783.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨94⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound10972.actual selector witness)
    (survivorMagnitude := LeftBound25787.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult25783.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10973.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10972.derived selector witness)
  · exact LeftBound25787.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult25789

namespace SemanticResult25797
def owner : Owner := ⟨.program ⟨214⟩, ⟨14454⟩⟩
def rawTerms : List Term := Proof.Events100.exact25797RawTerms
def summary : Bound := (.finite 18304)
def resultEvent : Nat := 25797
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25797.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨22, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge25795.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge25795.frameStart)
    (owner := owner) (leftOwner := SemanticResult25789.owner)
    (rightOwner := SemanticResult1052.owner)
    (leftResult := 25789) (rightResult := 1052)
    (leftActual := SemanticResult25789.actual selector witness)
    (rightActual := SemanticResult1052.actual selector witness)
    (leftRaw := SemanticResult25789.rawTerms)
    (rightRaw := SemanticResult1052.rawTerms)
    (working := LeftOperatorMerge25795.working)
    (leftBinding := 25790) (rightBinding := 25791)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11568⟩) (rightExpression := ⟨14451⟩)
    (coefficientTransfer := 25792) (summaryTransfer := 25794)
    (rightCoefficientProducer := 1051)
    (rightSummaryTransfer := 25793)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨22, by decide⟩)
    (rightRecordedMaximum := 22)
    (rightSummaryMaximum := ⟨22, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge25795.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1051.actual selector witness)
    (summaryMagnitude := LeftBound25794.actual selector witness)
    (reconstruction := LeftOperatorMerge25795.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult25789.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1052.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1051.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1051.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge25795.operationAgreement
  · exact LeftBound25794.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge25795.working summary) := by
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
end SemanticResult25797

namespace SemanticResult25802
def owner : Owner := ⟨.program ⟨214⟩, ⟨14455⟩⟩
def rawTerms : List Term := Proof.Events100.exact25802RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 25802
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25802.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge25801.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge25801.frameStart)
    (transferEvent := 25800) (owner := owner)
    (leftResult := 1052) (rightResult := 21420)
    (working := LeftOperatorMerge25801.working)
    (reconstruction := LeftOperatorMerge25801.reconstruction)
    (leftReference := .predecessor 0 25798 .coefficient) (rightReference := .predecessor 1 25799 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1052.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge25801.operationAgreement
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
end SemanticResult25802

namespace SemanticResult25807
def owner : Owner := ⟨.program ⟨214⟩, ⟨7331⟩⟩
def rawTerms : List Term := Proof.Events100.exact25807RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 25807
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25807.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge25806.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge25806.frameStart)
    (transferEvent := 25805) (owner := owner)
    (leftResult := 21290) (rightResult := 11022)
    (working := LeftOperatorMerge25806.working)
    (reconstruction := LeftOperatorMerge25806.reconstruction)
    (leftReference := .predecessor 0 25803 .coefficient) (rightReference := .predecessor 1 25804 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11022.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge25806.operationAgreement
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
end SemanticResult25807

namespace SemanticResult25811
def owner : Owner := ⟨.program ⟨214⟩, ⟨14456⟩⟩
def rawTerms : List Term := Proof.Events100.exact25811RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 25811
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25811.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 25808) (rightBinding := 25809)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7331⟩) (rightExpression := ⟨14455⟩)
    (transferEvent := 25810)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult25807.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult25802.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult25811

namespace SemanticResult25817
def owner : Owner := ⟨.program ⟨214⟩, ⟨14457⟩⟩
def rawTerms : List Term := Proof.Events100.exact25817RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 25817
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult25817.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 25814) (survivorTransfer := 25815)
    (survivorEvent := 25816) (resultEvent := resultEvent)
    (rightCoefficientProducer := 11013)
    (owner := owner) (leftOwner := SemanticResult25811.owner)
    (rightOwner := SemanticResult11014.owner)
    (leftResult := 25811) (rightResult := 11014)
    (leftBinding := 25812) (rightBinding := 25813)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14456⟩) (rightExpression := ⟨75⟩)
    (leftActual := SemanticResult25811.actual selector witness)
    (rightActual := SemanticResult11014.actual selector witness)
    (leftRaw := SemanticResult25811.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨75⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound11013.actual selector witness)
    (survivorMagnitude := LeftBound25815.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult25811.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11014.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11013.derived selector witness)
  · exact LeftBound25815.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult25817

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
