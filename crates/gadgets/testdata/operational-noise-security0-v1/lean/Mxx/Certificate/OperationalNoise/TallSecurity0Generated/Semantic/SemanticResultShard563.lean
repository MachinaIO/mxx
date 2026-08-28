import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard563
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard049
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard050
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard161
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard557
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard559
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard560
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard561
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard562

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult79374
def owner : Owner := ⟨.program ⟨214⟩, ⟨6708⟩⟩
def rawTerms : List Term := Proof.Events310.exact79374RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 79374
def producerEvent : Nat := 79373
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79374.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 79281, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult79374

namespace SemanticResult79378
def owner : Owner := ⟨.program ⟨214⟩, ⟨14885⟩⟩
def rawTerms : List Term := Proof.Events310.exact79378RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 79378
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79378.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 79375) (rightBinding := 79376)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6708⟩) (rightExpression := ⟨14884⟩)
    (transferEvent := 79377)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79374.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79371.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79378

namespace SemanticResult79382
def owner : Owner := ⟨.program ⟨214⟩, ⟨26345⟩⟩
def rawTerms : List Term := Proof.Events310.exact79382RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 79382
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79382.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 79379) (rightBinding := 79380)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14885⟩) (rightExpression := ⟨26340⟩)
    (transferEvent := 79381)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79378.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79363.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79382

namespace SemanticResult79391
def owner : Owner := ⟨.program ⟨214⟩, ⟨20319⟩⟩
def rawTerms : List Term := Proof.Events310.exact79391RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 79391
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79391.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge79226.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge79226.frameStart)
    (owner := owner) (leftOwner := SemanticResult65387.owner)
    (rightOwner := SemanticResult79220.owner)
    (leftResult := 65387) (rightResult := 79220)
    (leftActual := SemanticResult65387.actual selector witness)
    (rightActual := SemanticResult79220.actual selector witness)
    (leftRaw := SemanticResult65387.rawTerms)
    (rightRaw := SemanticResult79220.rawTerms)
    (working := LeftOperatorMerge79226.working)
    (leftBinding := 79221) (rightBinding := 79222)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5535⟩) (rightExpression := ⟨20318⟩)
    (coefficientTransfer := 79223) (summaryTransfer := 79225)
    (rightCoefficientProducer := 79219)
    (rightSummaryTransfer := 79224)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge79226.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound79219.actual selector witness)
    (summaryMagnitude := LeftBound79225.actual selector witness)
    (reconstruction := LeftOperatorMerge79226.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult65387.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79220.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79219.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound79219.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge79226.operationAgreement
  · exact LeftBound79225.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge79226.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 79386 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26339⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23717⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26339⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14788⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23717⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge79226.working
    [{ coefficient := (1), key := LeftRelationMerge79386.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge79386.frameStart
      LeftRelationMerge79386.owner (.relation 79386) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge79386.deltas
    rows := LeftRelationMerge79386.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge79226.working LeftRelationMerge79386.source
        (relationContext LeftRelationMerge79386.source
          LeftRelationMerge79386.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge79226.working, LeftRelationMerge79386.deltas,
    LeftRelationMerge79386.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 79386)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨20319⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20316⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20316⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge79226.working) (working := relationWorking0)
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
end SemanticResult79391

namespace SemanticResult79398
def owner : Owner := ⟨.program ⟨214⟩, ⟨26342⟩⟩
def rawTerms : List Term := Proof.Events310.exact79398RawTerms
def summary : Bound := (.finite 1291889174379421642752)
def resultEvent : Nat := 79398
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79398.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge79395.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79391.owner)
    (rightOwner := SemanticResult79213.owner)
    (leftResult := 79391) (rightResult := 79213)
    (leftActual := SemanticResult79391.actual selector witness)
    (rightActual := SemanticResult79213.actual selector witness)
    (leftRaw := SemanticResult79391.rawTerms)
    (rightRaw := SemanticResult79213.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1291889172568118132736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79392) (rightBinding := 79393)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20319⟩) (rightExpression := ⟨26341⟩)
    (coefficientTransfer := 79394) (summaryTransfer := 79397)
    (base := LeftOperatorMerge79395.base)
    (reconstruction := LeftOperatorMerge79395.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79391.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79213.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge79395.operationAgreement
  · rfl
  · decide
end SemanticResult79398

namespace SemanticResult79408
def owner : Owner := ⟨.program ⟨214⟩, ⟨26343⟩⟩
def rawTerms : List Term := Proof.Events310.exact79408RawTerms
def summary : Bound := (.finite 4741253940199267499646124032)
def resultEvent : Nat := 79408
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79408.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨1291889174379421642752, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge79404.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge79404.frameStart)
    (owner := owner) (leftOwner := SemanticResult79398.owner)
    (rightOwner := SemanticResult5859.owner)
    (leftResult := 79398) (rightResult := 5859)
    (leftActual := SemanticResult79398.actual selector witness)
    (rightActual := SemanticResult5859.actual selector witness)
    (leftRaw := SemanticResult79398.rawTerms)
    (rightRaw := SemanticResult5859.rawTerms)
    (working := LeftOperatorMerge79404.working)
    (leftBinding := 79399) (rightBinding := 79400)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26342⟩) (rightExpression := ⟨6680⟩)
    (coefficientTransfer := 79401) (summaryTransfer := 79403)
    (rightCoefficientProducer := 5858)
    (rightSummaryTransfer := 79402)
    (leftMaximum := ⟨1291889174379421642752, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge79404.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound5858.actual selector witness)
    (summaryMagnitude := LeftBound79403.actual selector witness)
    (reconstruction := LeftOperatorMerge79404.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79398.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5859.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5858.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound5858.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge79404.operationAgreement
  · exact LeftBound79403.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge79404.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 79406 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge79404.working
    [{ coefficient := (-1), key := LeftRelationMerge79406.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge79406.frameStart
      LeftRelationMerge79406.owner (.relation 79406) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge79406.deltas
    rows := LeftRelationMerge79406.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge79404.working LeftRelationMerge79406.source
        (relationContext LeftRelationMerge79406.source
          LeftRelationMerge79406.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge79404.working, LeftRelationMerge79406.deltas,
    LeftRelationMerge79406.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 79406)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨26343⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨14881⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge79404.working) (working := relationWorking0)
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
end SemanticResult79408

namespace SemanticResult79413
def owner : Owner := ⟨.program ⟨214⟩, ⟨6625⟩⟩
def rawTerms : List Term := Proof.Events310.exact79413RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 79413
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79413.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge79412.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge79412.frameStart)
    (transferEvent := 79411) (owner := owner)
    (leftResult := 723) (rightResult := 65295)
    (working := LeftOperatorMerge79412.working)
    (reconstruction := LeftOperatorMerge79412.reconstruction)
    (leftReference := .predecessor 0 79409 .coefficient) (rightReference := .predecessor 1 79410 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult723.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge79412.operationAgreement
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
end SemanticResult79413

namespace SemanticResult79418
def owner : Owner := ⟨.program ⟨214⟩, ⟨7178⟩⟩
def rawTerms : List Term := Proof.Events310.exact79418RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 79418
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79418.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge79417.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge79417.frameStart)
    (transferEvent := 79416) (owner := owner)
    (leftResult := 65165) (rightResult := 5873)
    (working := LeftOperatorMerge79417.working)
    (reconstruction := LeftOperatorMerge79417.reconstruction)
    (leftReference := .predecessor 0 79414 .coefficient) (rightReference := .predecessor 1 79415 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5873.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge79417.operationAgreement
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
end SemanticResult79418

namespace SemanticResult79422
def owner : Owner := ⟨.program ⟨214⟩, ⟨7749⟩⟩
def rawTerms : List Term := Proof.Events310.exact79422RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 79422
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79422.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 79419) (rightBinding := 79420)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7178⟩) (rightExpression := ⟨6625⟩)
    (transferEvent := 79421)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79418.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79413.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79422

namespace SemanticResult79428
def owner : Owner := ⟨.program ⟨214⟩, ⟨7750⟩⟩
def rawTerms : List Term := Proof.Events310.exact79428RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 79428
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79428.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 79425) (survivorTransfer := 79426)
    (survivorEvent := 79427) (resultEvent := resultEvent)
    (rightCoefficientProducer := 20907)
    (owner := owner) (leftOwner := SemanticResult79422.owner)
    (rightOwner := SemanticResult20908.owner)
    (leftResult := 79422) (rightResult := 20908)
    (leftBinding := 79423) (rightBinding := 79424)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7749⟩) (rightExpression := ⟨74⟩)
    (leftActual := SemanticResult79422.actual selector witness)
    (rightActual := SemanticResult20908.actual selector witness)
    (leftRaw := SemanticResult79422.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨74⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound20907.actual selector witness)
    (survivorMagnitude := LeftBound79426.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79422.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20908.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)
  · exact LeftBound79426.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult79428

namespace SemanticResult79435
def owner : Owner := ⟨.program ⟨214⟩, ⟨7807⟩⟩
def rawTerms : List Term := Proof.Events310.exact79435RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 79435
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79435.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge79432.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79428.owner)
    (rightOwner := SemanticResult79428.owner)
    (leftResult := 79428) (rightResult := 79428)
    (leftActual := SemanticResult79428.actual selector witness)
    (rightActual := SemanticResult79428.actual selector witness)
    (leftRaw := SemanticResult79428.rawTerms)
    (rightRaw := SemanticResult79428.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79429) (rightBinding := 79430)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7750⟩) (rightExpression := ⟨7750⟩)
    (coefficientTransfer := 79431) (summaryTransfer := 79434)
    (base := LeftOperatorMerge79432.base)
    (reconstruction := LeftOperatorMerge79432.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79428.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79428.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge79432.operationAgreement
  · rfl
  · decide
end SemanticResult79435

namespace SemanticResult79440
def owner : Owner := ⟨.program ⟨214⟩, ⟨26344⟩⟩
def rawTerms : List Term := Proof.Events310.exact79440RawTerms
def summary : Bound := (.finite 4741253940199267499646124084)
def resultEvent : Nat := 79440
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79440.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79435.owner)
    (rightOwner := SemanticResult79408.owner)
    (leftResult := 79435) (rightResult := 79408)
    (leftActual := SemanticResult79435.actual selector witness)
    (rightActual := SemanticResult79408.actual selector witness)
    (leftRaw := SemanticResult79435.rawTerms)
    (rightRaw := SemanticResult79408.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 4741253940199267499646124032) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79436) (rightBinding := 79437)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7807⟩) (rightExpression := ⟨26343⟩)
    (transferEvent := 79438) (summaryTransferEvent := 79439)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79435.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79408.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79440

namespace SemanticResult79445
def owner : Owner := ⟨.program ⟨214⟩, ⟨26549⟩⟩
def rawTerms : List Term := Proof.Events310.exact79445RawTerms
def summary : Bound := (.finite 9482549007414447334737575988)
def resultEvent : Nat := 79445
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79445.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79440.owner)
    (rightOwner := SemanticResult79196.owner)
    (leftResult := 79440) (rightResult := 79196)
    (leftActual := SemanticResult79440.actual selector witness)
    (rightActual := SemanticResult79196.actual selector witness)
    (leftRaw := SemanticResult79440.rawTerms)
    (rightRaw := SemanticResult79196.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4741253940199267499646124084)
    (rightMaximum := 4741295067215179835091451904) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79441) (rightBinding := 79442)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26344⟩) (rightExpression := ⟨26548⟩)
    (transferEvent := 79443) (summaryTransferEvent := 79444)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79440.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79196.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79445

namespace SemanticResult79450
def owner : Owner := ⟨.program ⟨214⟩, ⟨26766⟩⟩
def rawTerms : List Term := Proof.Events310.exact79450RawTerms
def summary : Bound := (.finite 14223885201645539505274355764)
def resultEvent : Nat := 79450
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79450.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79445.owner)
    (rightOwner := SemanticResult78984.owner)
    (leftResult := 79445) (rightResult := 78984)
    (leftActual := SemanticResult79445.actual selector witness)
    (rightActual := SemanticResult78984.actual selector witness)
    (leftRaw := SemanticResult79445.rawTerms)
    (rightRaw := SemanticResult78984.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 9482549007414447334737575988)
    (rightMaximum := 4741336194231092170536779776) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79446) (rightBinding := 79447)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26549⟩) (rightExpression := ⟨26765⟩)
    (transferEvent := 79448) (summaryTransferEvent := 79449)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79445.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult78984.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79450

namespace SemanticResult79455
def owner : Owner := ⟨.program ⟨214⟩, ⟨26983⟩⟩
def rawTerms : List Term := Proof.Events310.exact79455RawTerms
def summary : Bound := (.finite 18965303649908456346701791284)
def resultEvent : Nat := 79455
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79455.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79450.owner)
    (rightOwner := SemanticResult78772.owner)
    (leftResult := 79450) (rightResult := 78772)
    (leftActual := SemanticResult79450.actual selector witness)
    (rightActual := SemanticResult78772.actual selector witness)
    (leftRaw := SemanticResult79450.rawTerms)
    (rightRaw := SemanticResult78772.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 14223885201645539505274355764)
    (rightMaximum := 4741418448262916841427435520) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79451) (rightBinding := 79452)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26766⟩) (rightExpression := ⟨26982⟩)
    (transferEvent := 79453) (summaryTransferEvent := 79454)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79450.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult78772.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79455

namespace SemanticResult79460
def owner : Owner := ⟨.program ⟨214⟩, ⟨27200⟩⟩
def rawTerms : List Term := Proof.Events310.exact79460RawTerms
def summary : Bound := (.finite 23706886606235022529910538292)
def resultEvent : Nat := 79460
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult79460.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult79455.owner)
    (rightOwner := SemanticResult78560.owner)
    (leftResult := 79455) (rightResult := 78560)
    (leftActual := SemanticResult79455.actual selector witness)
    (rightActual := SemanticResult78560.actual selector witness)
    (leftRaw := SemanticResult79455.rawTerms)
    (rightRaw := SemanticResult78560.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 18965303649908456346701791284)
    (rightMaximum := 4741582956326566183208747008) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 79456) (rightBinding := 79457)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26983⟩) (rightExpression := ⟨27199⟩)
    (transferEvent := 79458) (summaryTransferEvent := 79459)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult79455.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult78560.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult79460

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
