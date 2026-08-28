import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard161
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard049
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard050
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard055
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard150
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard151
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard152
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard154
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard155
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard156
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard158
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard159
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard160

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult20905
def owner : Owner := ⟨.program ⟨214⟩, ⟨26403⟩⟩
def rawTerms : List Term := Proof.Events081.exact20905RawTerms
def summary : Bound := (.finite 4741253940199267499646124032)
def resultEvent : Nat := 20905
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20905.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨1291889174379421642752, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge20901.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge20901.frameStart)
    (owner := owner) (leftOwner := SemanticResult20895.owner)
    (rightOwner := SemanticResult5859.owner)
    (leftResult := 20895) (rightResult := 5859)
    (leftActual := SemanticResult20895.actual selector witness)
    (rightActual := SemanticResult5859.actual selector witness)
    (leftRaw := SemanticResult20895.rawTerms)
    (rightRaw := SemanticResult5859.rawTerms)
    (working := LeftOperatorMerge20901.working)
    (leftBinding := 20896) (rightBinding := 20897)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26402⟩) (rightExpression := ⟨6680⟩)
    (coefficientTransfer := 20898) (summaryTransfer := 20900)
    (rightCoefficientProducer := 5858)
    (rightSummaryTransfer := 20899)
    (leftMaximum := ⟨1291889174379421642752, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge20901.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound5858.actual selector witness)
    (summaryMagnitude := LeftBound20900.actual selector witness)
    (reconstruction := LeftOperatorMerge20901.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20895.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5859.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5858.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound5858.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge20901.operationAgreement
  · exact LeftBound20900.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge20901.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 20903 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨6495⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6495⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge20901.working
    [{ coefficient := (-1), key := LeftRelationMerge20903.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge20903.frameStart
      LeftRelationMerge20903.owner (.relation 20903) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge20903.deltas
    rows := LeftRelationMerge20903.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge20901.working LeftRelationMerge20903.source
        (relationContext LeftRelationMerge20903.source
          LeftRelationMerge20903.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge20901.working, LeftRelationMerge20903.deltas,
    LeftRelationMerge20903.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 20903)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨26403⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14906⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge20901.working) (working := relationWorking0)
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
end SemanticResult20905

namespace SemanticResult20908
def owner : Owner := ⟨.program ⟨214⟩, ⟨74⟩⟩
def rawTerms : List Term := Proof.Events081.exact20908RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 20908
def producerEvent : Nat := 20907
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20908.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 20906 .coefficient), 0, .finite 26, .identity (.predecessor 0 20906 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult20908

namespace SemanticResult20913
def owner : Owner := ⟨.program ⟨214⟩, ⟨6630⟩⟩
def rawTerms : List Term := Proof.Events081.exact20913RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 20913
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20913.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge20912.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge20912.frameStart)
    (transferEvent := 20911) (owner := owner)
    (leftResult := 723) (rightResult := 6449)
    (working := LeftOperatorMerge20912.working)
    (reconstruction := LeftOperatorMerge20912.reconstruction)
    (leftReference := .predecessor 0 20909 .coefficient) (rightReference := .predecessor 1 20910 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult723.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6449.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge20912.operationAgreement
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
end SemanticResult20913

namespace SemanticResult20918
def owner : Owner := ⟨.program ⟨214⟩, ⟨7368⟩⟩
def rawTerms : List Term := Proof.Events081.exact20918RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 20918
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20918.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge20917.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge20917.frameStart)
    (transferEvent := 20916) (owner := owner)
    (leftResult := 6314) (rightResult := 5873)
    (working := LeftOperatorMerge20917.working)
    (reconstruction := LeftOperatorMerge20917.reconstruction)
    (leftReference := .predecessor 0 20914 .coefficient) (rightReference := .predecessor 1 20915 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult6314.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult5873.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge20917.operationAgreement
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
end SemanticResult20918

namespace SemanticResult20922
def owner : Owner := ⟨.program ⟨214⟩, ⟨7769⟩⟩
def rawTerms : List Term := Proof.Events081.exact20922RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 20922
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20922.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 20919) (rightBinding := 20920)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7368⟩) (rightExpression := ⟨6630⟩)
    (transferEvent := 20921)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20918.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20913.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult20922

namespace SemanticResult20928
def owner : Owner := ⟨.program ⟨214⟩, ⟨7770⟩⟩
def rawTerms : List Term := Proof.Events081.exact20928RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 20928
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20928.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 20925) (survivorTransfer := 20926)
    (survivorEvent := 20927) (resultEvent := resultEvent)
    (rightCoefficientProducer := 20907)
    (owner := owner) (leftOwner := SemanticResult20922.owner)
    (rightOwner := SemanticResult20908.owner)
    (leftResult := 20922) (rightResult := 20908)
    (leftBinding := 20923) (rightBinding := 20924)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7769⟩) (rightExpression := ⟨74⟩)
    (leftActual := SemanticResult20922.actual selector witness)
    (rightActual := SemanticResult20908.actual selector witness)
    (leftRaw := SemanticResult20922.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨74⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound20907.actual selector witness)
    (survivorMagnitude := LeftBound20926.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20922.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20908.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)
  · exact LeftBound20926.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult20928

namespace SemanticResult20935
def owner : Owner := ⟨.program ⟨214⟩, ⟨7812⟩⟩
def rawTerms : List Term := Proof.Events081.exact20935RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 20935
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20935.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge20932.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult20928.owner)
    (rightOwner := SemanticResult20928.owner)
    (leftResult := 20928) (rightResult := 20928)
    (leftActual := SemanticResult20928.actual selector witness)
    (rightActual := SemanticResult20928.actual selector witness)
    (leftRaw := SemanticResult20928.rawTerms)
    (rightRaw := SemanticResult20928.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 20929) (rightBinding := 20930)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7770⟩) (rightExpression := ⟨7770⟩)
    (coefficientTransfer := 20931) (summaryTransfer := 20934)
    (base := LeftOperatorMerge20932.base)
    (reconstruction := LeftOperatorMerge20932.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20928.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20928.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge20932.operationAgreement
  · rfl
  · decide
end SemanticResult20935

namespace SemanticResult20940
def owner : Owner := ⟨.program ⟨214⟩, ⟨26404⟩⟩
def rawTerms : List Term := Proof.Events081.exact20940RawTerms
def summary : Bound := (.finite 4741253940199267499646124084)
def resultEvent : Nat := 20940
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20940.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult20935.owner)
    (rightOwner := SemanticResult20905.owner)
    (leftResult := 20935) (rightResult := 20905)
    (leftActual := SemanticResult20935.actual selector witness)
    (rightActual := SemanticResult20905.actual selector witness)
    (leftRaw := SemanticResult20935.rawTerms)
    (rightRaw := SemanticResult20905.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 4741253940199267499646124032) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 20936) (rightBinding := 20937)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7812⟩) (rightExpression := ⟨26403⟩)
    (transferEvent := 20938) (summaryTransferEvent := 20939)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20935.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20905.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult20940

namespace SemanticResult20945
def owner : Owner := ⟨.program ⟨214⟩, ⟨26614⟩⟩
def rawTerms : List Term := Proof.Events081.exact20945RawTerms
def summary : Bound := (.finite 9482549007414447334737575988)
def resultEvent : Nat := 20945
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20945.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult20940.owner)
    (rightOwner := SemanticResult20693.owner)
    (leftResult := 20940) (rightResult := 20693)
    (leftActual := SemanticResult20940.actual selector witness)
    (rightActual := SemanticResult20693.actual selector witness)
    (leftRaw := SemanticResult20940.rawTerms)
    (rightRaw := SemanticResult20693.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 4741253940199267499646124084)
    (rightMaximum := 4741295067215179835091451904) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 20941) (rightBinding := 20942)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26404⟩) (rightExpression := ⟨26613⟩)
    (transferEvent := 20943) (summaryTransferEvent := 20944)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20940.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20693.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult20945

namespace SemanticResult20950
def owner : Owner := ⟨.program ⟨214⟩, ⟨26831⟩⟩
def rawTerms : List Term := Proof.Events081.exact20950RawTerms
def summary : Bound := (.finite 14223885201645539505274355764)
def resultEvent : Nat := 20950
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20950.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult20945.owner)
    (rightOwner := SemanticResult20481.owner)
    (leftResult := 20945) (rightResult := 20481)
    (leftActual := SemanticResult20945.actual selector witness)
    (rightActual := SemanticResult20481.actual selector witness)
    (leftRaw := SemanticResult20945.rawTerms)
    (rightRaw := SemanticResult20481.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 9482549007414447334737575988)
    (rightMaximum := 4741336194231092170536779776) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 20946) (rightBinding := 20947)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26614⟩) (rightExpression := ⟨26830⟩)
    (transferEvent := 20948) (summaryTransferEvent := 20949)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20945.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20481.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult20950

namespace SemanticResult20955
def owner : Owner := ⟨.program ⟨214⟩, ⟨27048⟩⟩
def rawTerms : List Term := Proof.Events081.exact20955RawTerms
def summary : Bound := (.finite 18965303649908456346701791284)
def resultEvent : Nat := 20955
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20955.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult20950.owner)
    (rightOwner := SemanticResult20269.owner)
    (leftResult := 20950) (rightResult := 20269)
    (leftActual := SemanticResult20950.actual selector witness)
    (rightActual := SemanticResult20269.actual selector witness)
    (leftRaw := SemanticResult20950.rawTerms)
    (rightRaw := SemanticResult20269.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 14223885201645539505274355764)
    (rightMaximum := 4741418448262916841427435520) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 20951) (rightBinding := 20952)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26831⟩) (rightExpression := ⟨27047⟩)
    (transferEvent := 20953) (summaryTransferEvent := 20954)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20950.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20269.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult20955

namespace SemanticResult20960
def owner : Owner := ⟨.program ⟨214⟩, ⟨27265⟩⟩
def rawTerms : List Term := Proof.Events081.exact20960RawTerms
def summary : Bound := (.finite 23706886606235022529910538292)
def resultEvent : Nat := 20960
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20960.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult20955.owner)
    (rightOwner := SemanticResult20057.owner)
    (leftResult := 20955) (rightResult := 20057)
    (leftActual := SemanticResult20955.actual selector witness)
    (rightActual := SemanticResult20057.actual selector witness)
    (leftRaw := SemanticResult20955.rawTerms)
    (rightRaw := SemanticResult20057.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 18965303649908456346701791284)
    (rightMaximum := 4741582956326566183208747008) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 20956) (rightBinding := 20957)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27048⟩) (rightExpression := ⟨27264⟩)
    (transferEvent := 20958) (summaryTransferEvent := 20959)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20955.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult20057.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult20960

namespace SemanticResult20965
def owner : Owner := ⟨.program ⟨214⟩, ⟨27482⟩⟩
def rawTerms : List Term := Proof.Events081.exact20965RawTerms
def summary : Bound := (.finite 28448551816593413384009941044)
def resultEvent : Nat := 20965
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20965.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult20960.owner)
    (rightOwner := SemanticResult19845.owner)
    (leftResult := 20960) (rightResult := 19845)
    (leftActual := SemanticResult20960.actual selector witness)
    (rightActual := SemanticResult19845.actual selector witness)
    (leftRaw := SemanticResult20960.rawTerms)
    (rightRaw := SemanticResult19845.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 23706886606235022529910538292)
    (rightMaximum := 4741665210358390854099402752) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 20961) (rightBinding := 20962)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27265⟩) (rightExpression := ⟨27481⟩)
    (transferEvent := 20963) (summaryTransferEvent := 20964)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20960.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult19845.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult20965

namespace SemanticResult20970
def owner : Owner := ⟨.program ⟨214⟩, ⟨27699⟩⟩
def rawTerms : List Term := Proof.Events081.exact20970RawTerms
def summary : Bound := (.finite 33190381535015453579890655284)
def resultEvent : Nat := 20970
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20970.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult20965.owner)
    (rightOwner := SemanticResult19633.owner)
    (leftResult := 20965) (rightResult := 19633)
    (leftActual := SemanticResult20965.actual selector witness)
    (rightActual := SemanticResult19633.actual selector witness)
    (leftRaw := SemanticResult20965.rawTerms)
    (rightRaw := SemanticResult19633.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 28448551816593413384009941044)
    (rightMaximum := 4741829718422040195880714240) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 20966) (rightBinding := 20967)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27482⟩) (rightExpression := ⟨27698⟩)
    (transferEvent := 20968) (summaryTransferEvent := 20969)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20965.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult19633.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult20970

namespace SemanticResult20975
def owner : Owner := ⟨.program ⟨214⟩, ⟨27916⟩⟩
def rawTerms : List Term := Proof.Events081.exact20975RawTerms
def summary : Bound := (.finite 37932293507469318446662025268)
def resultEvent : Nat := 20975
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20975.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult20970.owner)
    (rightOwner := SemanticResult19421.owner)
    (leftResult := 20970) (rightResult := 19421)
    (leftActual := SemanticResult20970.actual selector witness)
    (rightActual := SemanticResult19421.actual selector witness)
    (leftRaw := SemanticResult20970.rawTerms)
    (rightRaw := SemanticResult19421.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 33190381535015453579890655284)
    (rightMaximum := 4741911972453864866771369984) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 20971) (rightBinding := 20972)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27699⟩) (rightExpression := ⟨27915⟩)
    (transferEvent := 20973) (summaryTransferEvent := 20974)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20970.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult19421.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult20975

namespace SemanticResult20980
def owner : Owner := ⟨.program ⟨214⟩, ⟨28133⟩⟩
def rawTerms : List Term := Proof.Events081.exact20980RawTerms
def summary : Bound := (.finite 42674369987986832655214706740)
def resultEvent : Nat := 20980
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult20980.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult20975.owner)
    (rightOwner := SemanticResult19209.owner)
    (leftResult := 20975) (rightResult := 19209)
    (leftActual := SemanticResult20975.actual selector witness)
    (rightActual := SemanticResult19209.actual selector witness)
    (leftRaw := SemanticResult20975.rawTerms)
    (rightRaw := SemanticResult19209.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 37932293507469318446662025268)
    (rightMaximum := 4742076480517514208552681472) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 20976) (rightBinding := 20977)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27916⟩) (rightExpression := ⟨28132⟩)
    (transferEvent := 20978) (summaryTransferEvent := 20979)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult20975.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult19209.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult20980

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
