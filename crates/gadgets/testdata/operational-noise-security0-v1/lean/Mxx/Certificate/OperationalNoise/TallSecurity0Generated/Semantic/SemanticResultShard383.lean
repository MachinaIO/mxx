import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard383
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard077
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard382

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult53028
def owner : Owner := ⟨.program ⟨214⟩, ⟨16596⟩⟩
def rawTerms : List Term := Proof.Events207.exact53028RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53028
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53028.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 53025) (rightBinding := 53026)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6703⟩) (rightExpression := ⟨16595⟩)
    (transferEvent := 53027)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult53024.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult53021.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult53028

namespace SemanticResult53036
def owner : Owner := ⟨.program ⟨214⟩, ⟨29182⟩⟩
def rawTerms : List Term := Proof.Events207.exact53036RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53036
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53036.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge53032.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge53032.frameStart)
    (transferEvent := 53031) (owner := owner)
    (leftResult := 53028) (rightResult := 53005)
    (working := LeftOperatorMerge53032.working)
    (reconstruction := LeftOperatorMerge53032.reconstruction)
    (leftReference := .predecessor 0 53029 .coefficient) (rightReference := .predecessor 1 53030 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult53028.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult53005.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge53032.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 53034 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29181⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16553⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24543⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24543⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge53032.working
    [{ coefficient := (-1), key := LeftRelationMerge53034.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge53034.frameStart
      LeftRelationMerge53034.owner (.relation 53034) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge53034.deltas
    rows := LeftRelationMerge53034.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge53032.working LeftRelationMerge53034.source
        (relationContext LeftRelationMerge53034.source
          LeftRelationMerge53034.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge53032.working, LeftRelationMerge53034.deltas,
    LeftRelationMerge53034.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 53034)
    (frameStart := 52954) (owner := ⟨.program ⟨214⟩, ⟨29182⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨16553⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29181⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29181⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge53032.working) (working := relationWorking0)
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
end SemanticResult53036

namespace SemanticResult53039
def owner : Owner := ⟨.program ⟨214⟩, ⟨18208⟩⟩
def rawTerms : List Term := Proof.Events207.exact53039RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53039
def producerEvent : Nat := 53038
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53039.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 52954, .finite 63, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult53039

namespace SemanticResult53044
def owner : Owner := ⟨.program ⟨214⟩, ⟨18209⟩⟩
def rawTerms : List Term := Proof.Events207.exact53044RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53044
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53044.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge53043.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge53043.frameStart)
    (transferEvent := 53042) (owner := owner)
    (leftResult := 53016) (rightResult := 53039)
    (working := LeftOperatorMerge53043.working)
    (reconstruction := LeftOperatorMerge53043.reconstruction)
    (leftReference := .predecessor 0 53040 .coefficient) (rightReference := .predecessor 1 53041 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult53016.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult53039.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge53043.operationAgreement
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
end SemanticResult53044

namespace SemanticResult53047
def owner : Owner := ⟨.program ⟨214⟩, ⟨6735⟩⟩
def rawTerms : List Term := Proof.Events207.exact53047RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53047
def producerEvent : Nat := 53046
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53047.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 52954, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult53047

namespace SemanticResult53051
def owner : Owner := ⟨.program ⟨214⟩, ⟨18210⟩⟩
def rawTerms : List Term := Proof.Events207.exact53051RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53051
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53051.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 53048) (rightBinding := 53049)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6735⟩) (rightExpression := ⟨18209⟩)
    (transferEvent := 53050)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult53047.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult53044.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult53051

namespace SemanticResult53055
def owner : Owner := ⟨.program ⟨214⟩, ⟨29186⟩⟩
def rawTerms : List Term := Proof.Events207.exact53055RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53055
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53055.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 53052) (rightBinding := 53053)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18210⟩) (rightExpression := ⟨29182⟩)
    (transferEvent := 53054)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult53051.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult53036.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult53055

namespace SemanticResult53064
def owner : Owner := ⟨.program ⟨214⟩, ⟨22271⟩⟩
def rawTerms : List Term := Proof.Events207.exact53064RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 53064
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53064.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge52899.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge52899.frameStart)
    (owner := owner) (leftOwner := SemanticResult50762.owner)
    (rightOwner := SemanticResult52893.owner)
    (leftResult := 50762) (rightResult := 52893)
    (leftActual := SemanticResult50762.actual selector witness)
    (rightActual := SemanticResult52893.actual selector witness)
    (leftRaw := SemanticResult50762.rawTerms)
    (rightRaw := SemanticResult52893.rawTerms)
    (working := LeftOperatorMerge52899.working)
    (leftBinding := 52894) (rightBinding := 52895)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5547⟩) (rightExpression := ⟨22270⟩)
    (coefficientTransfer := 52896) (summaryTransfer := 52898)
    (rightCoefficientProducer := 52892)
    (rightSummaryTransfer := 52897)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge52899.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound52892.actual selector witness)
    (summaryMagnitude := LeftBound52898.actual selector witness)
    (reconstruction := LeftOperatorMerge52899.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50762.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult52893.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52892.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound52892.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge52899.operationAgreement
  · exact LeftBound52898.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge52899.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 53059 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29181⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16553⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24543⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18208⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29181⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16553⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24543⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18208⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge52899.working
    [{ coefficient := (1), key := LeftRelationMerge53059.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge53059.frameStart
      LeftRelationMerge53059.owner (.relation 53059) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge53059.deltas
    rows := LeftRelationMerge53059.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge52899.working LeftRelationMerge53059.source
        (relationContext LeftRelationMerge53059.source
          LeftRelationMerge53059.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge52899.working, LeftRelationMerge53059.deltas,
    LeftRelationMerge53059.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 53059)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨22271⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22268⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22268⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge52899.working) (working := relationWorking0)
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
end SemanticResult53064

namespace SemanticResult53071
def owner : Owner := ⟨.program ⟨214⟩, ⟨29184⟩⟩
def rawTerms : List Term := Proof.Events207.exact53071RawTerms
def summary : Bound := (.finite 1292337423279833362432)
def resultEvent : Nat := 53071
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53071.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge53068.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult53064.owner)
    (rightOwner := SemanticResult52886.owner)
    (leftResult := 53064) (rightResult := 52886)
    (leftActual := SemanticResult53064.actual selector witness)
    (rightActual := SemanticResult52886.actual selector witness)
    (leftRaw := SemanticResult53064.rawTerms)
    (rightRaw := SemanticResult52886.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292337421468529852416) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 53065) (rightBinding := 53066)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22271⟩) (rightExpression := ⟨29183⟩)
    (coefficientTransfer := 53067) (summaryTransfer := 53070)
    (base := LeftOperatorMerge53068.base)
    (reconstruction := LeftOperatorMerge53068.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult53064.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult52886.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge53068.operationAgreement
  · rfl
  · decide
end SemanticResult53071

namespace SemanticResult53078
def owner : Owner := ⟨.program ⟨214⟩, ⟨24480⟩⟩
def rawTerms : List Term := Proof.Events207.exact53078RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53078
def producerEvent : Nat := 53077
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53078.actual selector witness
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
end SemanticResult53078

namespace SemanticResult53081
def owner : Owner := ⟨.program ⟨214⟩, ⟨28964⟩⟩
def rawTerms : List Term := Proof.Events207.exact53081RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53081
def producerEvent : Nat := 53080
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53081.actual selector witness
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
end SemanticResult53081

namespace SemanticResult53088
def owner : Owner := ⟨.program ⟨214⟩, ⟨23208⟩⟩
def rawTerms : List Term := Proof.Events207.exact53088RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53088
def producerEvent : Nat := 53087
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53088.actual selector witness
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
end SemanticResult53088

namespace SemanticResult53091
def owner : Owner := ⟨.program ⟨214⟩, ⟨25378⟩⟩
def rawTerms : List Term := Proof.Events207.exact53091RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53091
def producerEvent : Nat := 53090
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53091.actual selector witness
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
end SemanticResult53091

namespace SemanticResult53096
def owner : Owner := ⟨.program ⟨214⟩, ⟨12381⟩⟩
def rawTerms : List Term := Proof.Events207.exact53096RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53096
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53096.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge53095.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge53095.frameStart)
    (transferEvent := 53094) (owner := owner)
    (leftResult := 2453) (rightResult := 50670)
    (working := LeftOperatorMerge53095.working)
    (reconstruction := LeftOperatorMerge53095.reconstruction)
    (leftReference := .predecessor 0 53092 .coefficient) (rightReference := .predecessor 1 53093 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2453.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge53095.operationAgreement
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
end SemanticResult53096

namespace SemanticResult53101
def owner : Owner := ⟨.program ⟨214⟩, ⟨7279⟩⟩
def rawTerms : List Term := Proof.Events207.exact53101RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53101
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53101.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge53100.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge53100.frameStart)
    (transferEvent := 53099) (owner := owner)
    (leftResult := 50540) (rightResult := 8977)
    (working := LeftOperatorMerge53100.working)
    (reconstruction := LeftOperatorMerge53100.reconstruction)
    (leftReference := .predecessor 0 53097 .coefficient) (rightReference := .predecessor 1 53098 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8977.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge53100.operationAgreement
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
end SemanticResult53101

namespace SemanticResult53105
def owner : Owner := ⟨.program ⟨214⟩, ⟨12382⟩⟩
def rawTerms : List Term := Proof.Events207.exact53105RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 53105
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult53105.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 53102) (rightBinding := 53103)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7279⟩) (rightExpression := ⟨12381⟩)
    (transferEvent := 53104)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult53101.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult53096.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult53105

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
