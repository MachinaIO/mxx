import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard681
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard037
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard073

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult96104
def owner : Owner := ⟨.program ⟨214⟩, ⟨24531⟩⟩
def rawTerms : List Term := Proof.Events375.exact96104RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96104
def producerEvent : Nat := 96103
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96104.actual selector witness
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
end SemanticResult96104

namespace SemanticResult96107
def owner : Owner := ⟨.program ⟨214⟩, ⟨29133⟩⟩
def rawTerms : List Term := Proof.Events375.exact96107RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96107
def producerEvent : Nat := 96106
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96107.actual selector witness
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
end SemanticResult96107

namespace SemanticResult96114
def owner : Owner := ⟨.program ⟨214⟩, ⟨23242⟩⟩
def rawTerms : List Term := Proof.Events375.exact96114RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96114
def producerEvent : Nat := 96113
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96114.actual selector witness
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
end SemanticResult96114

namespace SemanticResult96117
def owner : Owner := ⟨.program ⟨214⟩, ⟨25437⟩⟩
def rawTerms : List Term := Proof.Events375.exact96117RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96117
def producerEvent : Nat := 96116
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96117.actual selector witness
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
end SemanticResult96117

namespace SemanticResult96122
def owner : Owner := ⟨.program ⟨214⟩, ⟨12545⟩⟩
def rawTerms : List Term := Proof.Events375.exact96122RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96122
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96122.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96121.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge96121.frameStart)
    (transferEvent := 96120) (owner := owner)
    (leftResult := 4658) (rightResult := 32)
    (working := LeftOperatorMerge96121.working)
    (reconstruction := LeftOperatorMerge96121.reconstruction)
    (leftReference := .predecessor 0 96118 .coefficient) (rightReference := .predecessor 1 96119 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4658.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge96121.operationAgreement
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
end SemanticResult96122

namespace SemanticResult96127
def owner : Owner := ⟨.program ⟨214⟩, ⟨7123⟩⟩
def rawTerms : List Term := Proof.Events375.exact96127RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96127
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96127.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96126.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge96126.frameStart)
    (transferEvent := 96125) (owner := owner)
    (leftResult := 27) (rightResult := 8476)
    (working := LeftOperatorMerge96126.working)
    (reconstruction := LeftOperatorMerge96126.reconstruction)
    (leftReference := .predecessor 0 96123 .coefficient) (rightReference := .predecessor 1 96124 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8476.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge96126.operationAgreement
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
end SemanticResult96127

namespace SemanticResult96131
def owner : Owner := ⟨.program ⟨214⟩, ⟨12546⟩⟩
def rawTerms : List Term := Proof.Events375.exact96131RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96131
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96131.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 96128) (rightBinding := 96129)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7123⟩) (rightExpression := ⟨12545⟩)
    (transferEvent := 96130)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96127.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult96122.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult96131

namespace SemanticResult96137
def owner : Owner := ⟨.program ⟨214⟩, ⟨12547⟩⟩
def rawTerms : List Term := Proof.Events375.exact96137RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 96137
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96137.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 96134) (survivorTransfer := 96135)
    (survivorEvent := 96136) (resultEvent := resultEvent)
    (rightCoefficientProducer := 8467)
    (owner := owner) (leftOwner := SemanticResult96131.owner)
    (rightOwner := SemanticResult8468.owner)
    (leftResult := 96131) (rightResult := 8468)
    (leftBinding := 96132) (rightBinding := 96133)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12546⟩) (rightExpression := ⟨100⟩)
    (leftActual := SemanticResult96131.actual selector witness)
    (rightActual := SemanticResult8468.actual selector witness)
    (leftRaw := SemanticResult96131.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨100⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound8467.actual selector witness)
    (survivorMagnitude := LeftBound96135.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96131.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8468.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8467.derived selector witness)
  · exact LeftBound96135.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult96137

namespace SemanticResult96145
def owner : Owner := ⟨.program ⟨214⟩, ⟨12548⟩⟩
def rawTerms : List Term := Proof.Events375.exact96145RawTerms
def summary : Bound := (.finite 34944)
def resultEvent : Nat := 96145
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96145.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨42, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96143.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge96143.frameStart)
    (owner := owner) (leftOwner := SemanticResult96137.owner)
    (rightOwner := SemanticResult4661.owner)
    (leftResult := 96137) (rightResult := 4661)
    (leftActual := SemanticResult96137.actual selector witness)
    (rightActual := SemanticResult4661.actual selector witness)
    (leftRaw := SemanticResult96137.rawTerms)
    (rightRaw := SemanticResult4661.rawTerms)
    (working := LeftOperatorMerge96143.working)
    (leftBinding := 96138) (rightBinding := 96139)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12547⟩) (rightExpression := ⟨9910⟩)
    (coefficientTransfer := 96140) (summaryTransfer := 96142)
    (rightCoefficientProducer := 4660)
    (rightSummaryTransfer := 96141)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨42, by decide⟩)
    (rightRecordedMaximum := 42)
    (rightSummaryMaximum := ⟨42, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge96143.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority4660.actual selector witness)
    (summaryMagnitude := LeftBound96142.actual selector witness)
    (reconstruction := LeftOperatorMerge96143.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96137.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4661.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4660.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority4660.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge96143.operationAgreement
  · exact LeftBound96142.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96143.working summary) := by
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
end SemanticResult96145

namespace SemanticResult96150
def owner : Owner := ⟨.program ⟨214⟩, ⟨9911⟩⟩
def rawTerms : List Term := Proof.Events375.exact96150RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96150
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96150.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96149.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge96149.frameStart)
    (transferEvent := 96148) (owner := owner)
    (leftResult := 4661) (rightResult := 32)
    (working := LeftOperatorMerge96149.working)
    (reconstruction := LeftOperatorMerge96149.reconstruction)
    (leftReference := .predecessor 0 96146 .coefficient) (rightReference := .predecessor 1 96147 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4661.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge96149.operationAgreement
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
end SemanticResult96150

namespace SemanticResult96155
def owner : Owner := ⟨.program ⟨214⟩, ⟨7103⟩⟩
def rawTerms : List Term := Proof.Events375.exact96155RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96155
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96155.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96154.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge96154.frameStart)
    (transferEvent := 96153) (owner := owner)
    (leftResult := 27) (rightResult := 8517)
    (working := LeftOperatorMerge96154.working)
    (reconstruction := LeftOperatorMerge96154.reconstruction)
    (leftReference := .predecessor 0 96151 .coefficient) (rightReference := .predecessor 1 96152 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8517.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge96154.operationAgreement
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
end SemanticResult96155

namespace SemanticResult96159
def owner : Owner := ⟨.program ⟨214⟩, ⟨9912⟩⟩
def rawTerms : List Term := Proof.Events375.exact96159RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 96159
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96159.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 96156) (rightBinding := 96157)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7103⟩) (rightExpression := ⟨9911⟩)
    (transferEvent := 96158)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96155.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult96150.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult96159

namespace SemanticResult96165
def owner : Owner := ⟨.program ⟨214⟩, ⟨9913⟩⟩
def rawTerms : List Term := Proof.Events375.exact96165RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 96165
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96165.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 96162) (survivorTransfer := 96163)
    (survivorEvent := 96164) (resultEvent := resultEvent)
    (rightCoefficientProducer := 8508)
    (owner := owner) (leftOwner := SemanticResult96159.owner)
    (rightOwner := SemanticResult8509.owner)
    (leftResult := 96159) (rightResult := 8509)
    (leftBinding := 96160) (rightBinding := 96161)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9912⟩) (rightExpression := ⟨80⟩)
    (leftActual := SemanticResult96159.actual selector witness)
    (rightActual := SemanticResult8509.actual selector witness)
    (leftRaw := SemanticResult96159.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨80⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound8508.actual selector witness)
    (survivorMagnitude := LeftBound96163.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96159.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8509.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8508.derived selector witness)
  · exact LeftBound96163.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult96165

namespace SemanticResult96175
def owner : Owner := ⟨.program ⟨214⟩, ⟨9914⟩⟩
def rawTerms : List Term := Proof.Events375.exact96175RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 96175
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96175.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96171.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge96171.frameStart)
    (owner := owner) (leftOwner := SemanticResult96165.owner)
    (rightOwner := SemanticResult8506.owner)
    (leftResult := 96165) (rightResult := 8506)
    (leftActual := SemanticResult96165.actual selector witness)
    (rightActual := SemanticResult8506.actual selector witness)
    (leftRaw := SemanticResult96165.rawTerms)
    (rightRaw := SemanticResult8506.rawTerms)
    (working := LeftOperatorMerge96171.working)
    (leftBinding := 96166) (rightBinding := 96167)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9913⟩) (rightExpression := ⟨7871⟩)
    (coefficientTransfer := 96168) (summaryTransfer := 96170)
    (rightCoefficientProducer := 8505)
    (rightSummaryTransfer := 96169)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge96171.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound8505.actual selector witness)
    (summaryMagnitude := LeftBound96170.actual selector witness)
    (reconstruction := LeftOperatorMerge96171.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8506.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8505.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound8505.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge96171.operationAgreement
  · exact LeftBound96170.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96171.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 96172 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9910⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge96171.working
    [{ coefficient := (-1), key := LeftRelationMerge96172.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge96172.frameStart
      LeftRelationMerge96172.owner (.relation 96172) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge96172.deltas
    rows := LeftRelationMerge96172.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge96171.working LeftRelationMerge96172.source
        (relationContext LeftRelationMerge96172.source
          LeftRelationMerge96172.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge96171.working, LeftRelationMerge96172.deltas,
    LeftRelationMerge96172.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 96172)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9914⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9910⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge96171.working) (working := relationWorking0)
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
end SemanticResult96175

namespace SemanticResult96181
def owner : Owner := ⟨.program ⟨214⟩, ⟨12549⟩⟩
def rawTerms : List Term := Proof.Events375.exact96181RawTerms
def summary : Bound := (.finite 95455360)
def resultEvent : Nat := 96181
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96181.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge96179.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult96175.owner)
    (rightOwner := SemanticResult96145.owner)
    (leftResult := 96175) (rightResult := 96145)
    (leftActual := SemanticResult96175.actual selector witness)
    (rightActual := SemanticResult96145.actual selector witness)
    (leftRaw := SemanticResult96175.rawTerms)
    (rightRaw := SemanticResult96145.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 34944) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 96176) (rightBinding := 96177)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9914⟩) (rightExpression := ⟨12548⟩)
    (coefficientTransfer := 96178) (summaryTransfer := 96180)
    (base := LeftOperatorMerge96179.base)
    (reconstruction := LeftOperatorMerge96179.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96175.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult96145.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge96179.operationAgreement
  · rfl
  · decide
end SemanticResult96181

namespace SemanticResult96191
def owner : Owner := ⟨.program ⟨214⟩, ⟨25438⟩⟩
def rawTerms : List Term := Proof.Events375.exact96191RawTerms
def summary : Bound := (.finite 350322698485760)
def resultEvent : Nat := 96191
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult96191.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95455360, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96187.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge96187.frameStart)
    (owner := owner) (leftOwner := SemanticResult96181.owner)
    (rightOwner := SemanticResult96117.owner)
    (leftResult := 96181) (rightResult := 96117)
    (leftActual := SemanticResult96181.actual selector witness)
    (rightActual := SemanticResult96117.actual selector witness)
    (leftRaw := SemanticResult96181.rawTerms)
    (rightRaw := SemanticResult96117.rawTerms)
    (working := LeftOperatorMerge96187.working)
    (leftBinding := 96182) (rightBinding := 96183)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12549⟩) (rightExpression := ⟨25437⟩)
    (coefficientTransfer := 96184) (summaryTransfer := 96186)
    (rightCoefficientProducer := 96116)
    (rightSummaryTransfer := 96185)
    (leftMaximum := ⟨95455360, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge96187.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority96116.actual selector witness)
    (summaryMagnitude := LeftBound96186.actual selector witness)
    (reconstruction := LeftOperatorMerge96187.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult96181.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult96117.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96116.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority96116.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge96187.operationAgreement
  · exact LeftBound96186.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge96187.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 96188 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25437⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23242⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23242⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge96187.working
    [{ coefficient := (-1), key := LeftRelationMerge96188.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge96188.frameStart
      LeftRelationMerge96188.owner (.relation 96188) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge96188.deltas
    rows := LeftRelationMerge96188.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge96187.working LeftRelationMerge96188.source
        (relationContext LeftRelationMerge96188.source
          LeftRelationMerge96188.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge96187.working, LeftRelationMerge96188.deltas,
    LeftRelationMerge96188.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 96188)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25438⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9910⟩⟩, ⟨.program ⟨214⟩, ⟨12542⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25437⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25437⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge96187.working) (working := relationWorking0)
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
end SemanticResult96191

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
