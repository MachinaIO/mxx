import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard223
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard009
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard121
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard122
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard164

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult29130
def owner : Owner := ⟨.program ⟨214⟩, ⟨23793⟩⟩
def rawTerms : List Term := Proof.Events113.exact29130RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 29130
def producerEvent : Nat := 29129
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29130.actual selector witness
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
end SemanticResult29130

namespace SemanticResult29133
def owner : Owner := ⟨.program ⟨214⟩, ⟨26603⟩⟩
def rawTerms : List Term := Proof.Events113.exact29133RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 29133
def producerEvent : Nat := 29132
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29133.actual selector witness
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
end SemanticResult29133

namespace SemanticResult29140
def owner : Owner := ⟨.program ⟨214⟩, ⟨23002⟩⟩
def rawTerms : List Term := Proof.Events113.exact29140RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 29140
def producerEvent : Nat := 29139
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29140.actual selector witness
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
end SemanticResult29140

namespace SemanticResult29143
def owner : Owner := ⟨.program ⟨214⟩, ⟨25003⟩⟩
def rawTerms : List Term := Proof.Events113.exact29143RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 29143
def producerEvent : Nat := 29142
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29143.actual selector witness
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
end SemanticResult29143

namespace SemanticResult29148
def owner : Owner := ⟨.program ⟨214⟩, ⟨10703⟩⟩
def rawTerms : List Term := Proof.Events113.exact29148RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 29148
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29148.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge29147.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge29147.frameStart)
    (transferEvent := 29146) (owner := owner)
    (leftResult := 1210) (rightResult := 21420)
    (working := LeftOperatorMerge29147.working)
    (reconstruction := LeftOperatorMerge29147.reconstruction)
    (leftReference := .predecessor 0 29144 .coefficient) (rightReference := .predecessor 1 29145 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1210.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge29147.operationAgreement
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
end SemanticResult29148

namespace SemanticResult29153
def owner : Owner := ⟨.program ⟨214⟩, ⟨7343⟩⟩
def rawTerms : List Term := Proof.Events113.exact29153RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 29153
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29153.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge29152.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge29152.frameStart)
    (transferEvent := 29151) (owner := owner)
    (leftResult := 21290) (rightResult := 14488)
    (working := LeftOperatorMerge29152.working)
    (reconstruction := LeftOperatorMerge29152.reconstruction)
    (leftReference := .predecessor 0 29149 .coefficient) (rightReference := .predecessor 1 29150 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14488.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge29152.operationAgreement
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
end SemanticResult29153

namespace SemanticResult29157
def owner : Owner := ⟨.program ⟨214⟩, ⟨10704⟩⟩
def rawTerms : List Term := Proof.Events113.exact29157RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 29157
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29157.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 29154) (rightBinding := 29155)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7343⟩) (rightExpression := ⟨10703⟩)
    (transferEvent := 29156)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult29153.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult29148.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult29157

namespace SemanticResult29163
def owner : Owner := ⟨.program ⟨214⟩, ⟨10705⟩⟩
def rawTerms : List Term := Proof.Events113.exact29163RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 29163
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29163.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 29160) (survivorTransfer := 29161)
    (survivorEvent := 29162) (resultEvent := resultEvent)
    (rightCoefficientProducer := 14479)
    (owner := owner) (leftOwner := SemanticResult29157.owner)
    (rightOwner := SemanticResult14480.owner)
    (leftResult := 29157) (rightResult := 14480)
    (leftBinding := 29158) (rightBinding := 29159)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10704⟩) (rightExpression := ⟨87⟩)
    (leftActual := SemanticResult29157.actual selector witness)
    (rightActual := SemanticResult14480.actual selector witness)
    (leftRaw := SemanticResult29157.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨87⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound14479.actual selector witness)
    (survivorMagnitude := LeftBound29161.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult29157.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14480.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14479.derived selector witness)
  · exact LeftBound29161.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult29163

namespace SemanticResult29171
def owner : Owner := ⟨.program ⟨214⟩, ⟨10706⟩⟩
def rawTerms : List Term := Proof.Events113.exact29171RawTerms
def summary : Bound := (.finite 2496)
def resultEvent : Nat := 29171
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29171.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨3, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge29169.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge29169.frameStart)
    (owner := owner) (leftOwner := SemanticResult29163.owner)
    (rightOwner := SemanticResult1213.owner)
    (leftResult := 29163) (rightResult := 1213)
    (leftActual := SemanticResult29163.actual selector witness)
    (rightActual := SemanticResult1213.actual selector witness)
    (leftRaw := SemanticResult29163.rawTerms)
    (rightRaw := SemanticResult1213.rawTerms)
    (working := LeftOperatorMerge29169.working)
    (leftBinding := 29164) (rightBinding := 29165)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10705⟩) (rightExpression := ⟨9520⟩)
    (coefficientTransfer := 29166) (summaryTransfer := 29168)
    (rightCoefficientProducer := 1212)
    (rightSummaryTransfer := 29167)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨3, by decide⟩)
    (rightRecordedMaximum := 3)
    (rightSummaryMaximum := ⟨3, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge29169.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1212.actual selector witness)
    (summaryMagnitude := LeftBound29168.actual selector witness)
    (reconstruction := LeftOperatorMerge29169.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult29163.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1213.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1212.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1212.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge29169.operationAgreement
  · exact LeftBound29168.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge29169.working summary) := by
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
end SemanticResult29171

namespace SemanticResult29176
def owner : Owner := ⟨.program ⟨214⟩, ⟨9521⟩⟩
def rawTerms : List Term := Proof.Events113.exact29176RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 29176
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29176.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge29175.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge29175.frameStart)
    (transferEvent := 29174) (owner := owner)
    (leftResult := 1213) (rightResult := 21420)
    (working := LeftOperatorMerge29175.working)
    (reconstruction := LeftOperatorMerge29175.reconstruction)
    (leftReference := .predecessor 0 29172 .coefficient) (rightReference := .predecessor 1 29173 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1213.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge29175.operationAgreement
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
end SemanticResult29176

namespace SemanticResult29181
def owner : Owner := ⟨.program ⟨214⟩, ⟨7352⟩⟩
def rawTerms : List Term := Proof.Events113.exact29181RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 29181
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29181.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge29180.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge29180.frameStart)
    (transferEvent := 29179) (owner := owner)
    (leftResult := 21290) (rightResult := 14529)
    (working := LeftOperatorMerge29180.working)
    (reconstruction := LeftOperatorMerge29180.reconstruction)
    (leftReference := .predecessor 0 29177 .coefficient) (rightReference := .predecessor 1 29178 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14529.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge29180.operationAgreement
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
end SemanticResult29181

namespace SemanticResult29185
def owner : Owner := ⟨.program ⟨214⟩, ⟨9522⟩⟩
def rawTerms : List Term := Proof.Events114.exact29185RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 29185
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29185.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 29182) (rightBinding := 29183)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7352⟩) (rightExpression := ⟨9521⟩)
    (transferEvent := 29184)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult29181.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult29176.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult29185

namespace SemanticResult29191
def owner : Owner := ⟨.program ⟨214⟩, ⟨9523⟩⟩
def rawTerms : List Term := Proof.Events114.exact29191RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 29191
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29191.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 29188) (survivorTransfer := 29189)
    (survivorEvent := 29190) (resultEvent := resultEvent)
    (rightCoefficientProducer := 14520)
    (owner := owner) (leftOwner := SemanticResult29185.owner)
    (rightOwner := SemanticResult14521.owner)
    (leftResult := 29185) (rightResult := 14521)
    (leftBinding := 29186) (rightBinding := 29187)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9522⟩) (rightExpression := ⟨96⟩)
    (leftActual := SemanticResult29185.actual selector witness)
    (rightActual := SemanticResult14521.actual selector witness)
    (leftRaw := SemanticResult29185.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨96⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound14520.actual selector witness)
    (survivorMagnitude := LeftBound29189.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult29185.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14521.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14520.derived selector witness)
  · exact LeftBound29189.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult29191

namespace SemanticResult29201
def owner : Owner := ⟨.program ⟨214⟩, ⟨9524⟩⟩
def rawTerms : List Term := Proof.Events114.exact29201RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 29201
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29201.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge29197.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge29197.frameStart)
    (owner := owner) (leftOwner := SemanticResult29191.owner)
    (rightOwner := SemanticResult14518.owner)
    (leftResult := 29191) (rightResult := 14518)
    (leftActual := SemanticResult29191.actual selector witness)
    (rightActual := SemanticResult14518.actual selector witness)
    (leftRaw := SemanticResult29191.rawTerms)
    (rightRaw := SemanticResult14518.rawTerms)
    (working := LeftOperatorMerge29197.working)
    (leftBinding := 29192) (rightBinding := 29193)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9523⟩) (rightExpression := ⟨7835⟩)
    (coefficientTransfer := 29194) (summaryTransfer := 29196)
    (rightCoefficientProducer := 14517)
    (rightSummaryTransfer := 29195)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge29197.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound14517.actual selector witness)
    (summaryMagnitude := LeftBound29196.actual selector witness)
    (reconstruction := LeftOperatorMerge29197.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult29191.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14518.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14517.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound14517.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge29197.operationAgreement
  · exact LeftBound29196.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge29197.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 29198 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9520⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge29197.working
    [{ coefficient := (-1), key := LeftRelationMerge29198.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge29198.frameStart
      LeftRelationMerge29198.owner (.relation 29198) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge29198.deltas
    rows := LeftRelationMerge29198.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge29197.working LeftRelationMerge29198.source
        (relationContext LeftRelationMerge29198.source
          LeftRelationMerge29198.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge29197.working, LeftRelationMerge29198.deltas,
    LeftRelationMerge29198.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 29198)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9524⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9520⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge29197.working) (working := relationWorking0)
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
end SemanticResult29201

namespace SemanticResult29207
def owner : Owner := ⟨.program ⟨214⟩, ⟨10707⟩⟩
def rawTerms : List Term := Proof.Events114.exact29207RawTerms
def summary : Bound := (.finite 95422912)
def resultEvent : Nat := 29207
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29207.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge29205.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult29201.owner)
    (rightOwner := SemanticResult29171.owner)
    (leftResult := 29201) (rightResult := 29171)
    (leftActual := SemanticResult29201.actual selector witness)
    (rightActual := SemanticResult29171.actual selector witness)
    (leftRaw := SemanticResult29201.rawTerms)
    (rightRaw := SemanticResult29171.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 2496) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 29202) (rightBinding := 29203)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9524⟩) (rightExpression := ⟨10706⟩)
    (coefficientTransfer := 29204) (summaryTransfer := 29206)
    (base := LeftOperatorMerge29205.base)
    (reconstruction := LeftOperatorMerge29205.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult29201.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult29171.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge29205.operationAgreement
  · rfl
  · decide
end SemanticResult29207

namespace SemanticResult29217
def owner : Owner := ⟨.program ⟨214⟩, ⟨25004⟩⟩
def rawTerms : List Term := Proof.Events114.exact29217RawTerms
def summary : Bound := (.finite 350203613806592)
def resultEvent : Nat := 29217
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult29217.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95422912, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge29213.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge29213.frameStart)
    (owner := owner) (leftOwner := SemanticResult29207.owner)
    (rightOwner := SemanticResult29143.owner)
    (leftResult := 29207) (rightResult := 29143)
    (leftActual := SemanticResult29207.actual selector witness)
    (rightActual := SemanticResult29143.actual selector witness)
    (leftRaw := SemanticResult29207.rawTerms)
    (rightRaw := SemanticResult29143.rawTerms)
    (working := LeftOperatorMerge29213.working)
    (leftBinding := 29208) (rightBinding := 29209)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10707⟩) (rightExpression := ⟨25003⟩)
    (coefficientTransfer := 29210) (summaryTransfer := 29212)
    (rightCoefficientProducer := 29142)
    (rightSummaryTransfer := 29211)
    (leftMaximum := ⟨95422912, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge29213.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority29142.actual selector witness)
    (summaryMagnitude := LeftBound29212.actual selector witness)
    (reconstruction := LeftOperatorMerge29213.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult29207.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult29143.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29142.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority29142.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge29213.operationAgreement
  · exact LeftBound29212.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge29213.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 29214 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23002⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23002⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge29213.working
    [{ coefficient := (-1), key := LeftRelationMerge29214.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge29214.frameStart
      LeftRelationMerge29214.owner (.relation 29214) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge29214.deltas
    rows := LeftRelationMerge29214.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge29213.working LeftRelationMerge29214.source
        (relationContext LeftRelationMerge29214.source
          LeftRelationMerge29214.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge29213.working, LeftRelationMerge29214.deltas,
    LeftRelationMerge29214.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 29214)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25004⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge29213.working) (working := relationWorking0)
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
end SemanticResult29217

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
