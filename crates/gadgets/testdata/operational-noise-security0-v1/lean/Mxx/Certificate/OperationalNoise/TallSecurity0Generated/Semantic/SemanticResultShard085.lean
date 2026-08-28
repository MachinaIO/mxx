import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard085
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard001
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard055
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard083
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard084

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult9948
def owner : Owner := ⟨.program ⟨214⟩, ⟨28789⟩⟩
def rawTerms : List Term := Proof.Events038.exact9948RawTerms
def summary : Bound := (.finite 1292270185944771604480)
def resultEvent : Nat := 9948
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult9948.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge9945.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult9941.owner)
    (rightOwner := SemanticResult9763.owner)
    (leftResult := 9941) (rightResult := 9763)
    (leftActual := SemanticResult9941.actual selector witness)
    (rightActual := SemanticResult9763.actual selector witness)
    (leftRaw := SemanticResult9941.rawTerms)
    (rightRaw := SemanticResult9763.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292270184133468094464) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 9942) (rightBinding := 9943)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21995⟩) (rightExpression := ⟨28788⟩)
    (coefficientTransfer := 9944) (summaryTransfer := 9947)
    (base := LeftOperatorMerge9945.base)
    (reconstruction := LeftOperatorMerge9945.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9941.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9763.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge9945.operationAgreement
  · rfl
  · decide
end SemanticResult9948

namespace SemanticResult9955
def owner : Owner := ⟨.program ⟨214⟩, ⟨24363⟩⟩
def rawTerms : List Term := Proof.Events038.exact9955RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9955
def producerEvent : Nat := 9954
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult9955.actual selector witness
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
end SemanticResult9955

namespace SemanticResult9958
def owner : Owner := ⟨.program ⟨214⟩, ⟨28569⟩⟩
def rawTerms : List Term := Proof.Events038.exact9958RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9958
def producerEvent : Nat := 9957
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult9958.actual selector witness
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
end SemanticResult9958

namespace SemanticResult9965
def owner : Owner := ⟨.program ⟨214⟩, ⟨23088⟩⟩
def rawTerms : List Term := Proof.Events038.exact9965RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9965
def producerEvent : Nat := 9964
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult9965.actual selector witness
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
end SemanticResult9965

namespace SemanticResult9968
def owner : Owner := ⟨.program ⟨214⟩, ⟨25162⟩⟩
def rawTerms : List Term := Proof.Events038.exact9968RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9968
def producerEvent : Nat := 9967
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult9968.actual selector witness
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
end SemanticResult9968

namespace SemanticResult9971
def owner : Owner := ⟨.program ⟨214⟩, ⟨97⟩⟩
def rawTerms : List Term := Proof.Events038.exact9971RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9971
def producerEvent : Nat := 9970
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult9971.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 9969 .coefficient), 0, .finite 26, .identity (.predecessor 0 9969 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult9971

namespace SemanticResult9976
def owner : Owner := ⟨.program ⟨214⟩, ⟨11796⟩⟩
def rawTerms : List Term := Proof.Events038.exact9976RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9976
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult9976.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge9975.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge9975.frameStart)
    (transferEvent := 9974) (owner := owner)
    (leftResult := 212) (rightResult := 6449)
    (working := LeftOperatorMerge9975.working)
    (reconstruction := LeftOperatorMerge9975.reconstruction)
    (leftReference := .predecessor 0 9972 .coefficient) (rightReference := .predecessor 1 9973 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult212.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6449.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge9975.operationAgreement
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
end SemanticResult9976

namespace SemanticResult9979
def owner : Owner := ⟨.program ⟨214⟩, ⟨6783⟩⟩
def rawTerms : List Term := Proof.Events038.exact9979RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9979
def producerEvent : Nat := 9978
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult9979.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 9977 .coefficient), 0, .large, .identity (.predecessor 0 9977 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult9979

namespace SemanticResult9984
def owner : Owner := ⟨.program ⟨214⟩, ⟨7391⟩⟩
def rawTerms : List Term := Proof.Events039.exact9984RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9984
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult9984.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge9983.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge9983.frameStart)
    (transferEvent := 9982) (owner := owner)
    (leftResult := 6314) (rightResult := 9979)
    (working := LeftOperatorMerge9983.working)
    (reconstruction := LeftOperatorMerge9983.reconstruction)
    (leftReference := .predecessor 0 9980 .coefficient) (rightReference := .predecessor 1 9981 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult6314.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9979.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge9983.operationAgreement
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
end SemanticResult9984

namespace SemanticResult9988
def owner : Owner := ⟨.program ⟨214⟩, ⟨11797⟩⟩
def rawTerms : List Term := Proof.Events039.exact9988RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 9988
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult9988.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 9985) (rightBinding := 9986)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7391⟩) (rightExpression := ⟨11796⟩)
    (transferEvent := 9987)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9984.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9976.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult9988

namespace SemanticResult9994
def owner : Owner := ⟨.program ⟨214⟩, ⟨11798⟩⟩
def rawTerms : List Term := Proof.Events039.exact9994RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 9994
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult9994.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 9991) (survivorTransfer := 9992)
    (survivorEvent := 9993) (resultEvent := resultEvent)
    (rightCoefficientProducer := 9970)
    (owner := owner) (leftOwner := SemanticResult9988.owner)
    (rightOwner := SemanticResult9971.owner)
    (leftResult := 9988) (rightResult := 9971)
    (leftBinding := 9989) (rightBinding := 9990)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11797⟩) (rightExpression := ⟨97⟩)
    (leftActual := SemanticResult9988.actual selector witness)
    (rightActual := SemanticResult9971.actual selector witness)
    (leftRaw := SemanticResult9988.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨97⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound9970.actual selector witness)
    (survivorMagnitude := LeftBound9992.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9988.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9971.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9970.derived selector witness)
  · exact LeftBound9992.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult9994

namespace SemanticResult10002
def owner : Owner := ⟨.program ⟨214⟩, ⟨11799⟩⟩
def rawTerms : List Term := Proof.Events039.exact10002RawTerms
def summary : Bound := (.finite 24960)
def resultEvent : Nat := 10002
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult10002.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨30, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge10000.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge10000.frameStart)
    (owner := owner) (leftOwner := SemanticResult9994.owner)
    (rightOwner := SemanticResult215.owner)
    (leftResult := 9994) (rightResult := 215)
    (leftActual := SemanticResult9994.actual selector witness)
    (rightActual := SemanticResult215.actual selector witness)
    (leftRaw := SemanticResult9994.rawTerms)
    (rightRaw := SemanticResult215.rawTerms)
    (working := LeftOperatorMerge10000.working)
    (leftBinding := 9995) (rightBinding := 9996)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11798⟩) (rightExpression := ⟨9630⟩)
    (coefficientTransfer := 9997) (summaryTransfer := 9999)
    (rightCoefficientProducer := 214)
    (rightSummaryTransfer := 9998)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨30, by decide⟩)
    (rightRecordedMaximum := 30)
    (rightSummaryMaximum := ⟨30, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge10000.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority214.actual selector witness)
    (summaryMagnitude := LeftBound9999.actual selector witness)
    (reconstruction := LeftOperatorMerge10000.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult9994.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult215.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority214.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority214.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge10000.operationAgreement
  · exact LeftBound9999.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge10000.working summary) := by
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
end SemanticResult10002

namespace SemanticResult10005
def owner : Owner := ⟨.program ⟨214⟩, ⟨7861⟩⟩
def rawTerms : List Term := Proof.Events039.exact10005RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10005
def producerEvent : Nat := 10004
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult10005.actual selector witness
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
end SemanticResult10005

namespace SemanticResult10009
def owner : Owner := ⟨.program ⟨214⟩, ⟨7862⟩⟩
def rawTerms : List Term := Proof.Events039.exact10009RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10009
def producerEvent : Nat := 10008
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult10009.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 10006 .coefficient) (.value (.predecessor 1 10007 .coefficient)), 0, .finite 8192, .scale (.predecessor 0 10006 .coefficient) (.value (.predecessor 1 10007 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult10009

namespace SemanticResult10012
def owner : Owner := ⟨.program ⟨214⟩, ⟨77⟩⟩
def rawTerms : List Term := Proof.Events039.exact10012RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10012
def producerEvent : Nat := 10011
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult10012.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 10010 .coefficient), 0, .finite 26, .identity (.predecessor 0 10010 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult10012

namespace SemanticResult10017
def owner : Owner := ⟨.program ⟨214⟩, ⟨9631⟩⟩
def rawTerms : List Term := Proof.Events039.exact10017RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 10017
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult10017.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge10016.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge10016.frameStart)
    (transferEvent := 10015) (owner := owner)
    (leftResult := 215) (rightResult := 6449)
    (working := LeftOperatorMerge10016.working)
    (reconstruction := LeftOperatorMerge10016.reconstruction)
    (leftReference := .predecessor 0 10013 .coefficient) (rightReference := .predecessor 1 10014 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult215.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6449.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge10016.operationAgreement
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
end SemanticResult10017

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
