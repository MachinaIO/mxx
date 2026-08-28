import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard367
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard366

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult50901
def owner : Owner := ⟨.program ⟨214⟩, ⟨7884⟩⟩
def rawTerms : List Term := Proof.Events198.exact50901RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50901
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50901.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50900.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge50900.frameStart)
    (transferEvent := 50899) (owner := owner)
    (leftResult := 50896) (rightResult := 50893)
    (working := LeftOperatorMerge50900.working)
    (reconstruction := LeftOperatorMerge50900.reconstruction)
    (leftReference := .predecessor 0 50897 .coefficient) (rightReference := .predecessor 1 50898 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50896.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50893.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge50900.operationAgreement
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
end SemanticResult50901

namespace SemanticResult50905
def owner : Owner := ⟨.program ⟨214⟩, ⟨13453⟩⟩
def rawTerms : List Term := Proof.Events198.exact50905RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50905
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50905.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 50902) (rightBinding := 50903)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7884⟩) (rightExpression := ⟨13452⟩)
    (transferEvent := 50904)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50901.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50878.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50905

namespace SemanticResult50913
def owner : Owner := ⟨.program ⟨214⟩, ⟨25766⟩⟩
def rawTerms : List Term := Proof.Events198.exact50913RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50913
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50913.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50909.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge50909.frameStart)
    (transferEvent := 50908) (owner := owner)
    (leftResult := 50905) (rightResult := 50862)
    (working := LeftOperatorMerge50909.working)
    (reconstruction := LeftOperatorMerge50909.reconstruction)
    (leftReference := .predecessor 0 50906 .coefficient) (rightReference := .predecessor 1 50907 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50905.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50862.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge50909.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 50911 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23418⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23418⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge50909.working
    [{ coefficient := (-1), key := LeftRelationMerge50911.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge50911.frameStart
      LeftRelationMerge50911.owner (.relation 50911) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge50911.deltas
    rows := LeftRelationMerge50911.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge50909.working LeftRelationMerge50911.source
        (relationContext LeftRelationMerge50911.source
          LeftRelationMerge50911.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge50909.working, LeftRelationMerge50911.deltas,
    LeftRelationMerge50911.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 50911)
    (frameStart := 50817) (owner := ⟨.program ⟨214⟩, ⟨25766⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge50909.working) (working := relationWorking0)
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
end SemanticResult50913

namespace SemanticResult50916
def owner : Owner := ⟨.program ⟨214⟩, ⟨17015⟩⟩
def rawTerms : List Term := Proof.Events198.exact50916RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50916
def producerEvent : Nat := 50915
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50916.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 50817, .finite 60, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult50916

namespace SemanticResult50921
def owner : Owner := ⟨.program ⟨214⟩, ⟨17017⟩⟩
def rawTerms : List Term := Proof.Events198.exact50921RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50921
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50921.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50920.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge50920.frameStart)
    (transferEvent := 50919) (owner := owner)
    (leftResult := 50873) (rightResult := 50916)
    (working := LeftOperatorMerge50920.working)
    (reconstruction := LeftOperatorMerge50920.reconstruction)
    (leftReference := .predecessor 0 50917 .coefficient) (rightReference := .predecessor 1 50918 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50873.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50916.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge50920.operationAgreement
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
end SemanticResult50921

namespace SemanticResult50924
def owner : Owner := ⟨.program ⟨214⟩, ⟨6707⟩⟩
def rawTerms : List Term := Proof.Events198.exact50924RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50924
def producerEvent : Nat := 50923
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50924.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 50817, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult50924

namespace SemanticResult50928
def owner : Owner := ⟨.program ⟨214⟩, ⟨17018⟩⟩
def rawTerms : List Term := Proof.Events198.exact50928RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50928
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50928.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 50925) (rightBinding := 50926)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6707⟩) (rightExpression := ⟨17017⟩)
    (transferEvent := 50927)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50924.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50921.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50928

namespace SemanticResult50932
def owner : Owner := ⟨.program ⟨214⟩, ⟨25767⟩⟩
def rawTerms : List Term := Proof.Events198.exact50932RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50932
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50932.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 50929) (rightBinding := 50930)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17018⟩) (rightExpression := ⟨25766⟩)
    (transferEvent := 50931)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50928.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50913.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult50932

namespace SemanticResult50941
def owner : Owner := ⟨.program ⟨214⟩, ⟨20255⟩⟩
def rawTerms : List Term := Proof.Events198.exact50941RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 50941
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50941.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50768.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge50768.frameStart)
    (owner := owner) (leftOwner := SemanticResult50762.owner)
    (rightOwner := SemanticResult50751.owner)
    (leftResult := 50762) (rightResult := 50751)
    (leftActual := SemanticResult50762.actual selector witness)
    (rightActual := SemanticResult50751.actual selector witness)
    (leftRaw := SemanticResult50762.rawTerms)
    (rightRaw := SemanticResult50751.rawTerms)
    (working := LeftOperatorMerge50768.working)
    (leftBinding := 50763) (rightBinding := 50764)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5547⟩) (rightExpression := ⟨20254⟩)
    (coefficientTransfer := 50765) (summaryTransfer := 50767)
    (rightCoefficientProducer := 50750)
    (rightSummaryTransfer := 50766)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge50768.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound50750.actual selector witness)
    (summaryMagnitude := LeftBound50767.actual selector witness)
    (reconstruction := LeftOperatorMerge50768.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50762.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50751.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50750.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound50750.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge50768.operationAgreement
  · exact LeftBound50767.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50768.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 50936 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23418⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25763⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10350⟩⟩, ⟨.program ⟨214⟩, ⟨13358⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23418⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge50768.working
    [{ coefficient := (1), key := LeftRelationMerge50936.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge50936.frameStart
      LeftRelationMerge50936.owner (.relation 50936) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge50936.deltas
    rows := LeftRelationMerge50936.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge50768.working LeftRelationMerge50936.source
        (relationContext LeftRelationMerge50936.source
          LeftRelationMerge50936.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge50768.working, LeftRelationMerge50936.deltas,
    LeftRelationMerge50936.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 50936)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨20255⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20252⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20252⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge50768.working) (working := relationWorking0)
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
end SemanticResult50941

namespace SemanticResult50948
def owner : Owner := ⟨.program ⟨214⟩, ⟨25765⟩⟩
def rawTerms : List Term := Proof.Events199.exact50948RawTerms
def summary : Bound := (.finite 352188964155392)
def resultEvent : Nat := 50948
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50948.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge50945.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult50941.owner)
    (rightOwner := SemanticResult50744.owner)
    (leftResult := 50941) (rightResult := 50744)
    (leftActual := SemanticResult50941.actual selector witness)
    (rightActual := SemanticResult50744.actual selector witness)
    (leftRaw := SemanticResult50941.rawTerms)
    (rightRaw := SemanticResult50744.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 350377660645376) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 50942) (rightBinding := 50943)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20255⟩) (rightExpression := ⟨25764⟩)
    (coefficientTransfer := 50944) (summaryTransfer := 50947)
    (base := LeftOperatorMerge50945.base)
    (reconstruction := LeftOperatorMerge50945.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50941.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50744.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge50945.operationAgreement
  · rfl
  · decide
end SemanticResult50948

namespace SemanticResult50958
def owner : Owner := ⟨.program ⟨214⟩, ⟨30141⟩⟩
def rawTerms : List Term := Proof.Events199.exact50958RawTerms
def summary : Bound := (.finite 1292539133473715126272)
def resultEvent : Nat := 50958
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50958.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨352188964155392, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50954.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge50954.frameStart)
    (owner := owner) (leftOwner := SemanticResult50948.owner)
    (rightOwner := SemanticResult50655.owner)
    (leftResult := 50948) (rightResult := 50655)
    (leftActual := SemanticResult50948.actual selector witness)
    (rightActual := SemanticResult50655.actual selector witness)
    (leftRaw := SemanticResult50948.rawTerms)
    (rightRaw := SemanticResult50655.rawTerms)
    (working := LeftOperatorMerge50954.working)
    (leftBinding := 50949) (rightBinding := 50950)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨25765⟩) (rightExpression := ⟨30139⟩)
    (coefficientTransfer := 50951) (summaryTransfer := 50953)
    (rightCoefficientProducer := 50654)
    (rightSummaryTransfer := 50952)
    (leftMaximum := ⟨352188964155392, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge50954.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority50654.actual selector witness)
    (summaryMagnitude := LeftBound50953.actual selector witness)
    (reconstruction := LeftOperatorMerge50954.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50948.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50655.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50654.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority50654.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge50954.operationAgreement
  · exact LeftBound50953.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50954.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 50956 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24795⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24795⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge50954.working
    [{ coefficient := (-1), key := LeftRelationMerge50956.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge50956.frameStart
      LeftRelationMerge50956.owner (.relation 50956) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge50956.deltas
    rows := LeftRelationMerge50956.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge50954.working LeftRelationMerge50956.source
        (relationContext LeftRelationMerge50956.source
          LeftRelationMerge50956.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge50954.working, LeftRelationMerge50956.deltas,
    LeftRelationMerge50956.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 50956)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨30141⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge50954.working) (working := relationWorking0)
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
end SemanticResult50958

namespace SemanticResult50961
def owner : Owner := ⟨.program ⟨214⟩, ⟨22844⟩⟩
def rawTerms : List Term := Proof.Events199.exact50961RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50961
def producerEvent : Nat := 50960
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50961.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨65⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨65⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult50961

namespace SemanticResult50965
def owner : Owner := ⟨.program ⟨214⟩, ⟨22846⟩⟩
def rawTerms : List Term := Proof.Events199.exact50965RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 50965
def producerEvent : Nat := 50964
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult50965.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 50962 .coefficient) (.value (.predecessor 1 50963 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 50962 .coefficient) (.value (.predecessor 1 50963 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult50965

namespace SemanticResult51063
def owner : Owner := ⟨.program ⟨214⟩, ⟨17015⟩⟩
def rawTerms : List Term := Proof.Events199.exact51063RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51063
def producerEvent : Nat := 51062
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51063.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 51026, .finite 60, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult51063

namespace SemanticResult51074
def owner : Owner := ⟨.program ⟨214⟩, ⟨24795⟩⟩
def rawTerms : List Term := Proof.Events199.exact51074RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51074
def producerEvent : Nat := 51073
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51074.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 51026, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult51074

namespace SemanticResult51077
def owner : Owner := ⟨.program ⟨214⟩, ⟨30139⟩⟩
def rawTerms : List Term := Proof.Events199.exact51077RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51077
def producerEvent : Nat := 51076
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51077.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 51026, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult51077

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
