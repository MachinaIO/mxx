import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard309
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard015
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard105
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard106
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard264

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult41840
def owner : Owner := ⟨.program ⟨214⟩, ⟨25922⟩⟩
def rawTerms : List Term := Proof.Events163.exact41840RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41840
def producerEvent : Nat := 41839
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41840.actual selector witness
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
end SemanticResult41840

namespace SemanticResult41845
def owner : Owner := ⟨.program ⟨214⟩, ⟨11310⟩⟩
def rawTerms : List Term := Proof.Events163.exact41845RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41845
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41845.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41844.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge41844.frameStart)
    (transferEvent := 41843) (owner := owner)
    (leftResult := 1866) (rightResult := 36045)
    (working := LeftOperatorMerge41844.working)
    (reconstruction := LeftOperatorMerge41844.reconstruction)
    (leftReference := .predecessor 0 41841 .coefficient) (rightReference := .predecessor 1 41842 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1866.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge41844.operationAgreement
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
end SemanticResult41845

namespace SemanticResult41850
def owner : Owner := ⟨.program ⟨214⟩, ⟨7309⟩⟩
def rawTerms : List Term := Proof.Events163.exact41850RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41850
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41850.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41849.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge41849.frameStart)
    (transferEvent := 41848) (owner := owner)
    (leftResult := 35915) (rightResult := 12484)
    (working := LeftOperatorMerge41849.working)
    (reconstruction := LeftOperatorMerge41849.reconstruction)
    (leftReference := .predecessor 0 41846 .coefficient) (rightReference := .predecessor 1 41847 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12484.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge41849.operationAgreement
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
end SemanticResult41850

namespace SemanticResult41854
def owner : Owner := ⟨.program ⟨214⟩, ⟨11311⟩⟩
def rawTerms : List Term := Proof.Events163.exact41854RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41854
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41854.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41851) (rightBinding := 41852)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7309⟩) (rightExpression := ⟨11310⟩)
    (transferEvent := 41853)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41850.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41845.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41854

namespace SemanticResult41860
def owner : Owner := ⟨.program ⟨214⟩, ⟨11312⟩⟩
def rawTerms : List Term := Proof.Events163.exact41860RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 41860
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41860.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 41857) (survivorTransfer := 41858)
    (survivorEvent := 41859) (resultEvent := resultEvent)
    (rightCoefficientProducer := 12475)
    (owner := owner) (leftOwner := SemanticResult41854.owner)
    (rightOwner := SemanticResult12476.owner)
    (leftResult := 41854) (rightResult := 12476)
    (leftBinding := 41855) (rightBinding := 41856)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11311⟩) (rightExpression := ⟨91⟩)
    (leftActual := SemanticResult41854.actual selector witness)
    (rightActual := SemanticResult12476.actual selector witness)
    (leftRaw := SemanticResult41854.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨91⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound12475.actual selector witness)
    (survivorMagnitude := LeftBound41858.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41854.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12476.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12475.derived selector witness)
  · exact LeftBound41858.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult41860

namespace SemanticResult41868
def owner : Owner := ⟨.program ⟨214⟩, ⟨13794⟩⟩
def rawTerms : List Term := Proof.Events163.exact41868RawTerms
def summary : Bound := (.finite 9984)
def resultEvent : Nat := 41868
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41868.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨12, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41866.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge41866.frameStart)
    (owner := owner) (leftOwner := SemanticResult41860.owner)
    (rightOwner := SemanticResult1869.owner)
    (leftResult := 41860) (rightResult := 1869)
    (leftActual := SemanticResult41860.actual selector witness)
    (rightActual := SemanticResult1869.actual selector witness)
    (leftRaw := SemanticResult41860.rawTerms)
    (rightRaw := SemanticResult1869.rawTerms)
    (working := LeftOperatorMerge41866.working)
    (leftBinding := 41861) (rightBinding := 41862)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11312⟩) (rightExpression := ⟨13791⟩)
    (coefficientTransfer := 41863) (summaryTransfer := 41865)
    (rightCoefficientProducer := 1868)
    (rightSummaryTransfer := 41864)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨12, by decide⟩)
    (rightRecordedMaximum := 12)
    (rightSummaryMaximum := ⟨12, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge41866.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1868.actual selector witness)
    (summaryMagnitude := LeftBound41865.actual selector witness)
    (reconstruction := LeftOperatorMerge41866.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41860.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1869.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1868.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1868.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge41866.operationAgreement
  · exact LeftBound41865.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41866.working summary) := by
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
end SemanticResult41868

namespace SemanticResult41873
def owner : Owner := ⟨.program ⟨214⟩, ⟨13795⟩⟩
def rawTerms : List Term := Proof.Events163.exact41873RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41873
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41873.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41872.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge41872.frameStart)
    (transferEvent := 41871) (owner := owner)
    (leftResult := 1869) (rightResult := 36045)
    (working := LeftOperatorMerge41872.working)
    (reconstruction := LeftOperatorMerge41872.reconstruction)
    (leftReference := .predecessor 0 41869 .coefficient) (rightReference := .predecessor 1 41870 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1869.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge41872.operationAgreement
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
end SemanticResult41873

namespace SemanticResult41878
def owner : Owner := ⟨.program ⟨214⟩, ⟨7326⟩⟩
def rawTerms : List Term := Proof.Events163.exact41878RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41878
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41878.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41877.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge41877.frameStart)
    (transferEvent := 41876) (owner := owner)
    (leftResult := 35915) (rightResult := 12525)
    (working := LeftOperatorMerge41877.working)
    (reconstruction := LeftOperatorMerge41877.reconstruction)
    (leftReference := .predecessor 0 41874 .coefficient) (rightReference := .predecessor 1 41875 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12525.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge41877.operationAgreement
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
end SemanticResult41878

namespace SemanticResult41882
def owner : Owner := ⟨.program ⟨214⟩, ⟨13796⟩⟩
def rawTerms : List Term := Proof.Events163.exact41882RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41882
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41882.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 41879) (rightBinding := 41880)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7326⟩) (rightExpression := ⟨13795⟩)
    (transferEvent := 41881)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41878.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41873.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult41882

namespace SemanticResult41888
def owner : Owner := ⟨.program ⟨214⟩, ⟨13797⟩⟩
def rawTerms : List Term := Proof.Events163.exact41888RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 41888
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41888.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 41885) (survivorTransfer := 41886)
    (survivorEvent := 41887) (resultEvent := resultEvent)
    (rightCoefficientProducer := 12516)
    (owner := owner) (leftOwner := SemanticResult41882.owner)
    (rightOwner := SemanticResult12517.owner)
    (leftResult := 41882) (rightResult := 12517)
    (leftBinding := 41883) (rightBinding := 41884)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13796⟩) (rightExpression := ⟨108⟩)
    (leftActual := SemanticResult41882.actual selector witness)
    (rightActual := SemanticResult12517.actual selector witness)
    (leftRaw := SemanticResult41882.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨108⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound12516.actual selector witness)
    (survivorMagnitude := LeftBound41886.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41882.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12517.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12516.derived selector witness)
  · exact LeftBound41886.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult41888

namespace SemanticResult41898
def owner : Owner := ⟨.program ⟨214⟩, ⟨13798⟩⟩
def rawTerms : List Term := Proof.Events163.exact41898RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 41898
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41898.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41894.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge41894.frameStart)
    (owner := owner) (leftOwner := SemanticResult41888.owner)
    (rightOwner := SemanticResult12514.owner)
    (leftResult := 41888) (rightResult := 12514)
    (leftActual := SemanticResult41888.actual selector witness)
    (rightActual := SemanticResult12514.actual selector witness)
    (leftRaw := SemanticResult41888.rawTerms)
    (rightRaw := SemanticResult12514.rawTerms)
    (working := LeftOperatorMerge41894.working)
    (leftBinding := 41889) (rightBinding := 41890)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13797⟩) (rightExpression := ⟨7847⟩)
    (coefficientTransfer := 41891) (summaryTransfer := 41893)
    (rightCoefficientProducer := 12513)
    (rightSummaryTransfer := 41892)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge41894.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound12513.actual selector witness)
    (summaryMagnitude := LeftBound41893.actual selector witness)
    (reconstruction := LeftOperatorMerge41894.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41888.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult12514.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12513.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound12513.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge41894.operationAgreement
  · exact LeftBound41893.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41894.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 41895 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6777⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge41894.working
    [{ coefficient := (-1), key := LeftRelationMerge41895.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge41895.frameStart
      LeftRelationMerge41895.owner (.relation 41895) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge41895.deltas
    rows := LeftRelationMerge41895.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge41894.working LeftRelationMerge41895.source
        (relationContext LeftRelationMerge41895.source
          LeftRelationMerge41895.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge41894.working, LeftRelationMerge41895.deltas,
    LeftRelationMerge41895.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 41895)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨13798⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge41894.working) (working := relationWorking0)
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
end SemanticResult41898

namespace SemanticResult41904
def owner : Owner := ⟨.program ⟨214⟩, ⟨13799⟩⟩
def rawTerms : List Term := Proof.Events163.exact41904RawTerms
def summary : Bound := (.finite 95430400)
def resultEvent : Nat := 41904
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41904.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge41902.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult41898.owner)
    (rightOwner := SemanticResult41868.owner)
    (leftResult := 41898) (rightResult := 41868)
    (leftActual := SemanticResult41898.actual selector witness)
    (rightActual := SemanticResult41868.actual selector witness)
    (leftRaw := SemanticResult41898.rawTerms)
    (rightRaw := SemanticResult41868.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 9984) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 41899) (rightBinding := 41900)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13798⟩) (rightExpression := ⟨13794⟩)
    (coefficientTransfer := 41901) (summaryTransfer := 41903)
    (base := LeftOperatorMerge41902.base)
    (reconstruction := LeftOperatorMerge41902.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41898.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41868.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge41902.operationAgreement
  · rfl
  · decide
end SemanticResult41904

namespace SemanticResult41914
def owner : Owner := ⟨.program ⟨214⟩, ⟨25923⟩⟩
def rawTerms : List Term := Proof.Events163.exact41914RawTerms
def summary : Bound := (.finite 350231094886400)
def resultEvent : Nat := 41914
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41914.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95430400, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41910.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge41910.frameStart)
    (owner := owner) (leftOwner := SemanticResult41904.owner)
    (rightOwner := SemanticResult41840.owner)
    (leftResult := 41904) (rightResult := 41840)
    (leftActual := SemanticResult41904.actual selector witness)
    (rightActual := SemanticResult41840.actual selector witness)
    (leftRaw := SemanticResult41904.rawTerms)
    (rightRaw := SemanticResult41840.rawTerms)
    (working := LeftOperatorMerge41910.working)
    (leftBinding := 41905) (rightBinding := 41906)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13799⟩) (rightExpression := ⟨25922⟩)
    (coefficientTransfer := 41907) (summaryTransfer := 41909)
    (rightCoefficientProducer := 41839)
    (rightSummaryTransfer := 41908)
    (leftMaximum := ⟨95430400, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge41910.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority41839.actual selector witness)
    (summaryMagnitude := LeftBound41909.actual selector witness)
    (reconstruction := LeftOperatorMerge41910.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult41904.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult41840.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41839.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority41839.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge41910.operationAgreement
  · exact LeftBound41909.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge41910.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 41911 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23504⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23504⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge41910.working
    [{ coefficient := (-1), key := LeftRelationMerge41911.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge41911.frameStart
      LeftRelationMerge41911.owner (.relation 41911) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge41911.deltas
    rows := LeftRelationMerge41911.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge41910.working LeftRelationMerge41911.source
        (relationContext LeftRelationMerge41911.source
          LeftRelationMerge41911.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge41910.working, LeftRelationMerge41911.deltas,
    LeftRelationMerge41911.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 41911)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25923⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11309⟩⟩, ⟨.program ⟨214⟩, ⟨13791⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25922⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge41910.working) (working := relationWorking0)
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
end SemanticResult41914

namespace SemanticResult41917
def owner : Owner := ⟨.program ⟨214⟩, ⟨19392⟩⟩
def rawTerms : List Term := Proof.Events163.exact41917RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41917
def producerEvent : Nat := 41916
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41917.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨13⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨13⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult41917

namespace SemanticResult41921
def owner : Owner := ⟨.program ⟨214⟩, ⟨19394⟩⟩
def rawTerms : List Term := Proof.Events163.exact41921RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41921
def producerEvent : Nat := 41920
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41921.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 41918 .coefficient) (.value (.predecessor 1 41919 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 41918 .coefficient) (.value (.predecessor 1 41919 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult41921

namespace SemanticResult41999
def owner : Owner := ⟨.program ⟨214⟩, ⟨11309⟩⟩
def rawTerms : List Term := Proof.Events164.exact41999RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 41999
def producerEvent : Nat := 41998
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult41999.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 41976, .finite 12, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult41999

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
