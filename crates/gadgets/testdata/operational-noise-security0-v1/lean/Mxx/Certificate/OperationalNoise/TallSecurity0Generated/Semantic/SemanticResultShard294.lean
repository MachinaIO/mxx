import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard294
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard014
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard089
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard090
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard292
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard293

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult39892
def owner : Owner := ⟨.program ⟨214⟩, ⟨28546⟩⟩
def rawTerms : List Term := Proof.Events155.exact39892RawTerms
def summary : Bound := (.finite 1292202948609709846528)
def resultEvent : Nat := 39892
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39892.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge39889.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult39885.owner)
    (rightOwner := SemanticResult39707.owner)
    (leftResult := 39885) (rightResult := 39707)
    (leftActual := SemanticResult39885.actual selector witness)
    (rightActual := SemanticResult39707.actual selector witness)
    (leftRaw := SemanticResult39885.rawTerms)
    (rightRaw := SemanticResult39707.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292202946798406336512) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 39886) (rightBinding := 39887)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21843⟩) (rightExpression := ⟨28545⟩)
    (coefficientTransfer := 39888) (summaryTransfer := 39891)
    (base := LeftOperatorMerge39889.base)
    (reconstruction := LeftOperatorMerge39889.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult39885.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39707.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge39889.operationAgreement
  · rfl
  · decide
end SemanticResult39892

namespace SemanticResult39899
def owner : Owner := ⟨.program ⟨214⟩, ⟨24294⟩⟩
def rawTerms : List Term := Proof.Events155.exact39899RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39899
def producerEvent : Nat := 39898
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39899.actual selector witness
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
end SemanticResult39899

namespace SemanticResult39902
def owner : Owner := ⟨.program ⟨214⟩, ⟨28326⟩⟩
def rawTerms : List Term := Proof.Events155.exact39902RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39902
def producerEvent : Nat := 39901
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39902.actual selector witness
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
end SemanticResult39902

namespace SemanticResult39909
def owner : Owner := ⟨.program ⟨214⟩, ⟨23672⟩⟩
def rawTerms : List Term := Proof.Events155.exact39909RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39909
def producerEvent : Nat := 39908
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39909.actual selector witness
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
end SemanticResult39909

namespace SemanticResult39912
def owner : Owner := ⟨.program ⟨214⟩, ⟨26230⟩⟩
def rawTerms : List Term := Proof.Events155.exact39912RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39912
def producerEvent : Nat := 39911
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39912.actual selector witness
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
end SemanticResult39912

namespace SemanticResult39917
def owner : Owner := ⟨.program ⟨214⟩, ⟨11646⟩⟩
def rawTerms : List Term := Proof.Events155.exact39917RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39917
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39917.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39916.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge39916.frameStart)
    (transferEvent := 39915) (owner := owner)
    (leftResult := 1774) (rightResult := 36045)
    (working := LeftOperatorMerge39916.working)
    (reconstruction := LeftOperatorMerge39916.reconstruction)
    (leftReference := .predecessor 0 39913 .coefficient) (rightReference := .predecessor 1 39914 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1774.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge39916.operationAgreement
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
end SemanticResult39917

namespace SemanticResult39922
def owner : Owner := ⟨.program ⟨214⟩, ⟨7313⟩⟩
def rawTerms : List Term := Proof.Events155.exact39922RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39922
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39922.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39921.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge39921.frameStart)
    (transferEvent := 39920) (owner := owner)
    (leftResult := 35915) (rightResult := 10480)
    (working := LeftOperatorMerge39921.working)
    (reconstruction := LeftOperatorMerge39921.reconstruction)
    (leftReference := .predecessor 0 39918 .coefficient) (rightReference := .predecessor 1 39919 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10480.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge39921.operationAgreement
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
end SemanticResult39922

namespace SemanticResult39926
def owner : Owner := ⟨.program ⟨214⟩, ⟨11647⟩⟩
def rawTerms : List Term := Proof.Events155.exact39926RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39926
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39926.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 39923) (rightBinding := 39924)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7313⟩) (rightExpression := ⟨11646⟩)
    (transferEvent := 39925)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult39922.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39917.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult39926

namespace SemanticResult39932
def owner : Owner := ⟨.program ⟨214⟩, ⟨11648⟩⟩
def rawTerms : List Term := Proof.Events155.exact39932RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 39932
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39932.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 39929) (survivorTransfer := 39930)
    (survivorEvent := 39931) (resultEvent := resultEvent)
    (rightCoefficientProducer := 10471)
    (owner := owner) (leftOwner := SemanticResult39926.owner)
    (rightOwner := SemanticResult10472.owner)
    (leftResult := 39926) (rightResult := 10472)
    (leftBinding := 39927) (rightBinding := 39928)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11647⟩) (rightExpression := ⟨95⟩)
    (leftActual := SemanticResult39926.actual selector witness)
    (rightActual := SemanticResult10472.actual selector witness)
    (leftRaw := SemanticResult39926.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨95⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound10471.actual selector witness)
    (survivorMagnitude := LeftBound39930.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult39926.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10472.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10471.derived selector witness)
  · exact LeftBound39930.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult39932

namespace SemanticResult39940
def owner : Owner := ⟨.program ⟨214⟩, ⟨14662⟩⟩
def rawTerms : List Term := Proof.Events156.exact39940RawTerms
def summary : Bound := (.finite 23296)
def resultEvent : Nat := 39940
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39940.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨28, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39938.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge39938.frameStart)
    (owner := owner) (leftOwner := SemanticResult39932.owner)
    (rightOwner := SemanticResult1777.owner)
    (leftResult := 39932) (rightResult := 1777)
    (leftActual := SemanticResult39932.actual selector witness)
    (rightActual := SemanticResult1777.actual selector witness)
    (leftRaw := SemanticResult39932.rawTerms)
    (rightRaw := SemanticResult1777.rawTerms)
    (working := LeftOperatorMerge39938.working)
    (leftBinding := 39933) (rightBinding := 39934)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11648⟩) (rightExpression := ⟨14659⟩)
    (coefficientTransfer := 39935) (summaryTransfer := 39937)
    (rightCoefficientProducer := 1776)
    (rightSummaryTransfer := 39936)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨28, by decide⟩)
    (rightRecordedMaximum := 28)
    (rightSummaryMaximum := ⟨28, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge39938.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1776.actual selector witness)
    (summaryMagnitude := LeftBound39937.actual selector witness)
    (reconstruction := LeftOperatorMerge39938.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult39932.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1777.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1776.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1776.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge39938.operationAgreement
  · exact LeftBound39937.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39938.working summary) := by
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
end SemanticResult39940

namespace SemanticResult39945
def owner : Owner := ⟨.program ⟨214⟩, ⟨14663⟩⟩
def rawTerms : List Term := Proof.Events156.exact39945RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39945
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39945.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39944.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge39944.frameStart)
    (transferEvent := 39943) (owner := owner)
    (leftResult := 1777) (rightResult := 36045)
    (working := LeftOperatorMerge39944.working)
    (reconstruction := LeftOperatorMerge39944.reconstruction)
    (leftReference := .predecessor 0 39941 .coefficient) (rightReference := .predecessor 1 39942 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1777.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge39944.operationAgreement
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
end SemanticResult39945

namespace SemanticResult39950
def owner : Owner := ⟨.program ⟨214⟩, ⟨7294⟩⟩
def rawTerms : List Term := Proof.Events156.exact39950RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39950
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39950.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39949.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge39949.frameStart)
    (transferEvent := 39948) (owner := owner)
    (leftResult := 35915) (rightResult := 10521)
    (working := LeftOperatorMerge39949.working)
    (reconstruction := LeftOperatorMerge39949.reconstruction)
    (leftReference := .predecessor 0 39946 .coefficient) (rightReference := .predecessor 1 39947 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10521.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge39949.operationAgreement
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
end SemanticResult39950

namespace SemanticResult39954
def owner : Owner := ⟨.program ⟨214⟩, ⟨14664⟩⟩
def rawTerms : List Term := Proof.Events156.exact39954RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 39954
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39954.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 39951) (rightBinding := 39952)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7294⟩) (rightExpression := ⟨14663⟩)
    (transferEvent := 39953)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult39950.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39945.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult39954

namespace SemanticResult39960
def owner : Owner := ⟨.program ⟨214⟩, ⟨14665⟩⟩
def rawTerms : List Term := Proof.Events156.exact39960RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 39960
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39960.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 39957) (survivorTransfer := 39958)
    (survivorEvent := 39959) (resultEvent := resultEvent)
    (rightCoefficientProducer := 10512)
    (owner := owner) (leftOwner := SemanticResult39954.owner)
    (rightOwner := SemanticResult10513.owner)
    (leftResult := 39954) (rightResult := 10513)
    (leftBinding := 39955) (rightBinding := 39956)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14664⟩) (rightExpression := ⟨76⟩)
    (leftActual := SemanticResult39954.actual selector witness)
    (rightActual := SemanticResult10513.actual selector witness)
    (leftRaw := SemanticResult39954.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨76⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound10512.actual selector witness)
    (survivorMagnitude := LeftBound39958.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult39954.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10513.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10512.derived selector witness)
  · exact LeftBound39958.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult39960

namespace SemanticResult39970
def owner : Owner := ⟨.program ⟨214⟩, ⟨14666⟩⟩
def rawTerms : List Term := Proof.Events156.exact39970RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 39970
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39970.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39966.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge39966.frameStart)
    (owner := owner) (leftOwner := SemanticResult39960.owner)
    (rightOwner := SemanticResult10510.owner)
    (leftResult := 39960) (rightResult := 10510)
    (leftActual := SemanticResult39960.actual selector witness)
    (rightActual := SemanticResult10510.actual selector witness)
    (leftRaw := SemanticResult39960.rawTerms)
    (rightRaw := SemanticResult10510.rawTerms)
    (working := LeftOperatorMerge39966.working)
    (leftBinding := 39961) (rightBinding := 39962)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14665⟩) (rightExpression := ⟨7859⟩)
    (coefficientTransfer := 39963) (summaryTransfer := 39965)
    (rightCoefficientProducer := 10509)
    (rightSummaryTransfer := 39964)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge39966.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound10509.actual selector witness)
    (summaryMagnitude := LeftBound39965.actual selector witness)
    (reconstruction := LeftOperatorMerge39966.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult39960.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10510.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10509.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound10509.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge39966.operationAgreement
  · exact LeftBound39965.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge39966.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 39967 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6781⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge39966.working
    [{ coefficient := (-1), key := LeftRelationMerge39967.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge39967.frameStart
      LeftRelationMerge39967.owner (.relation 39967) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge39967.deltas
    rows := LeftRelationMerge39967.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge39966.working LeftRelationMerge39967.source
        (relationContext LeftRelationMerge39967.source
          LeftRelationMerge39967.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge39966.working, LeftRelationMerge39967.deltas,
    LeftRelationMerge39967.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 39967)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨14666⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge39966.working) (working := relationWorking0)
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
end SemanticResult39970

namespace SemanticResult39976
def owner : Owner := ⟨.program ⟨214⟩, ⟨14667⟩⟩
def rawTerms : List Term := Proof.Events156.exact39976RawTerms
def summary : Bound := (.finite 95443712)
def resultEvent : Nat := 39976
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult39976.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge39974.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult39970.owner)
    (rightOwner := SemanticResult39940.owner)
    (leftResult := 39970) (rightResult := 39940)
    (leftActual := SemanticResult39970.actual selector witness)
    (rightActual := SemanticResult39940.actual selector witness)
    (leftRaw := SemanticResult39970.rawTerms)
    (rightRaw := SemanticResult39940.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 23296) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 39971) (rightBinding := 39972)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14666⟩) (rightExpression := ⟨14662⟩)
    (coefficientTransfer := 39973) (summaryTransfer := 39975)
    (base := LeftOperatorMerge39974.base)
    (reconstruction := LeftOperatorMerge39974.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult39970.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult39940.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge39974.operationAgreement
  · rfl
  · decide
end SemanticResult39976

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
