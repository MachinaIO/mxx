import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard272
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard013
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard065
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard264

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult37010
def owner : Owner := ⟨.program ⟨214⟩, ⟨29628⟩⟩
def rawTerms : List Term := Proof.Events144.exact37010RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37010
def producerEvent : Nat := 37009
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37010.actual selector witness
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
end SemanticResult37010

namespace SemanticResult37017
def owner : Owner := ⟨.program ⟨214⟩, ⟨23336⟩⟩
def rawTerms : List Term := Proof.Events144.exact37017RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37017
def producerEvent : Nat := 37016
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37017.actual selector witness
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
end SemanticResult37017

namespace SemanticResult37020
def owner : Owner := ⟨.program ⟨214⟩, ⟨25614⟩⟩
def rawTerms : List Term := Proof.Events144.exact37020RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37020
def producerEvent : Nat := 37019
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37020.actual selector witness
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
end SemanticResult37020

namespace SemanticResult37025
def owner : Owner := ⟨.program ⟨214⟩, ⟨12977⟩⟩
def rawTerms : List Term := Proof.Events144.exact37025RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37025
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37025.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37024.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge37024.frameStart)
    (transferEvent := 37023) (owner := owner)
    (leftResult := 1636) (rightResult := 36045)
    (working := LeftOperatorMerge37024.working)
    (reconstruction := LeftOperatorMerge37024.reconstruction)
    (leftReference := .predecessor 0 37021 .coefficient) (rightReference := .predecessor 1 37022 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1636.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge37024.operationAgreement
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
end SemanticResult37025

namespace SemanticResult37030
def owner : Owner := ⟨.program ⟨214⟩, ⟨7320⟩⟩
def rawTerms : List Term := Proof.Events144.exact37030RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37030
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37030.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37029.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge37029.frameStart)
    (transferEvent := 37028) (owner := owner)
    (leftResult := 35915) (rightResult := 7474)
    (working := LeftOperatorMerge37029.working)
    (reconstruction := LeftOperatorMerge37029.reconstruction)
    (leftReference := .predecessor 0 37026 .coefficient) (rightReference := .predecessor 1 37027 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7474.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge37029.operationAgreement
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
end SemanticResult37030

namespace SemanticResult37034
def owner : Owner := ⟨.program ⟨214⟩, ⟨12978⟩⟩
def rawTerms : List Term := Proof.Events144.exact37034RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37034
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37034.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 37031) (rightBinding := 37032)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7320⟩) (rightExpression := ⟨12977⟩)
    (transferEvent := 37033)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37030.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult37025.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult37034

namespace SemanticResult37040
def owner : Owner := ⟨.program ⟨214⟩, ⟨12979⟩⟩
def rawTerms : List Term := Proof.Events144.exact37040RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 37040
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37040.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 37037) (survivorTransfer := 37038)
    (survivorEvent := 37039) (resultEvent := resultEvent)
    (rightCoefficientProducer := 7465)
    (owner := owner) (leftOwner := SemanticResult37034.owner)
    (rightOwner := SemanticResult7466.owner)
    (leftResult := 37034) (rightResult := 7466)
    (leftBinding := 37035) (rightBinding := 37036)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12978⟩) (rightExpression := ⟨102⟩)
    (leftActual := SemanticResult37034.actual selector witness)
    (rightActual := SemanticResult7466.actual selector witness)
    (leftRaw := SemanticResult37034.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨102⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound7465.actual selector witness)
    (survivorMagnitude := LeftBound37038.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37034.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7466.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7465.derived selector witness)
  · exact LeftBound37038.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult37040

namespace SemanticResult37048
def owner : Owner := ⟨.program ⟨214⟩, ⟨12980⟩⟩
def rawTerms : List Term := Proof.Events144.exact37048RawTerms
def summary : Bound := (.finite 43264)
def resultEvent : Nat := 37048
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37048.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨52, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37046.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge37046.frameStart)
    (owner := owner) (leftOwner := SemanticResult37040.owner)
    (rightOwner := SemanticResult1639.owner)
    (leftResult := 37040) (rightResult := 1639)
    (leftActual := SemanticResult37040.actual selector witness)
    (rightActual := SemanticResult1639.actual selector witness)
    (leftRaw := SemanticResult37040.rawTerms)
    (rightRaw := SemanticResult1639.rawTerms)
    (working := LeftOperatorMerge37046.working)
    (leftBinding := 37041) (rightBinding := 37042)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12979⟩) (rightExpression := ⟨10145⟩)
    (coefficientTransfer := 37043) (summaryTransfer := 37045)
    (rightCoefficientProducer := 1638)
    (rightSummaryTransfer := 37044)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨52, by decide⟩)
    (rightRecordedMaximum := 52)
    (rightSummaryMaximum := ⟨52, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge37046.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1638.actual selector witness)
    (summaryMagnitude := LeftBound37045.actual selector witness)
    (reconstruction := LeftOperatorMerge37046.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37040.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1639.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1638.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1638.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge37046.operationAgreement
  · exact LeftBound37045.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37046.working summary) := by
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
end SemanticResult37048

namespace SemanticResult37053
def owner : Owner := ⟨.program ⟨214⟩, ⟨10146⟩⟩
def rawTerms : List Term := Proof.Events144.exact37053RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37053
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37053.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37052.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge37052.frameStart)
    (transferEvent := 37051) (owner := owner)
    (leftResult := 1639) (rightResult := 36045)
    (working := LeftOperatorMerge37052.working)
    (reconstruction := LeftOperatorMerge37052.reconstruction)
    (leftReference := .predecessor 0 37049 .coefficient) (rightReference := .predecessor 1 37050 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1639.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge37052.operationAgreement
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
end SemanticResult37053

namespace SemanticResult37058
def owner : Owner := ⟨.program ⟨214⟩, ⟨7300⟩⟩
def rawTerms : List Term := Proof.Events144.exact37058RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37058
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37058.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37057.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge37057.frameStart)
    (transferEvent := 37056) (owner := owner)
    (leftResult := 35915) (rightResult := 7515)
    (working := LeftOperatorMerge37057.working)
    (reconstruction := LeftOperatorMerge37057.reconstruction)
    (leftReference := .predecessor 0 37054 .coefficient) (rightReference := .predecessor 1 37055 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7515.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge37057.operationAgreement
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
end SemanticResult37058

namespace SemanticResult37062
def owner : Owner := ⟨.program ⟨214⟩, ⟨10147⟩⟩
def rawTerms : List Term := Proof.Events144.exact37062RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37062
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37062.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 37059) (rightBinding := 37060)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7300⟩) (rightExpression := ⟨10146⟩)
    (transferEvent := 37061)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37058.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult37053.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult37062

namespace SemanticResult37068
def owner : Owner := ⟨.program ⟨214⟩, ⟨10148⟩⟩
def rawTerms : List Term := Proof.Events144.exact37068RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 37068
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37068.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 37065) (survivorTransfer := 37066)
    (survivorEvent := 37067) (resultEvent := resultEvent)
    (rightCoefficientProducer := 7506)
    (owner := owner) (leftOwner := SemanticResult37062.owner)
    (rightOwner := SemanticResult7507.owner)
    (leftResult := 37062) (rightResult := 7507)
    (leftBinding := 37063) (rightBinding := 37064)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10147⟩) (rightExpression := ⟨82⟩)
    (leftActual := SemanticResult37062.actual selector witness)
    (rightActual := SemanticResult7507.actual selector witness)
    (leftRaw := SemanticResult37062.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨82⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound7506.actual selector witness)
    (survivorMagnitude := LeftBound37066.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37062.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7507.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7506.derived selector witness)
  · exact LeftBound37066.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult37068

namespace SemanticResult37078
def owner : Owner := ⟨.program ⟨214⟩, ⟨10149⟩⟩
def rawTerms : List Term := Proof.Events144.exact37078RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 37078
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37078.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37074.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge37074.frameStart)
    (owner := owner) (leftOwner := SemanticResult37068.owner)
    (rightOwner := SemanticResult7504.owner)
    (leftResult := 37068) (rightResult := 7504)
    (leftActual := SemanticResult37068.actual selector witness)
    (rightActual := SemanticResult7504.actual selector witness)
    (leftRaw := SemanticResult37068.rawTerms)
    (rightRaw := SemanticResult7504.rawTerms)
    (working := LeftOperatorMerge37074.working)
    (leftBinding := 37069) (rightBinding := 37070)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10148⟩) (rightExpression := ⟨7877⟩)
    (coefficientTransfer := 37071) (summaryTransfer := 37073)
    (rightCoefficientProducer := 7503)
    (rightSummaryTransfer := 37072)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge37074.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound7503.actual selector witness)
    (summaryMagnitude := LeftBound37073.actual selector witness)
    (reconstruction := LeftOperatorMerge37074.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37068.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7504.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7503.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound7503.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge37074.operationAgreement
  · exact LeftBound37073.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37074.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 37075 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge37074.working
    [{ coefficient := (-1), key := LeftRelationMerge37075.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge37075.frameStart
      LeftRelationMerge37075.owner (.relation 37075) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge37075.deltas
    rows := LeftRelationMerge37075.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge37074.working LeftRelationMerge37075.source
        (relationContext LeftRelationMerge37075.source
          LeftRelationMerge37075.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge37074.working, LeftRelationMerge37075.deltas,
    LeftRelationMerge37075.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 37075)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10149⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge37074.working) (working := relationWorking0)
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
end SemanticResult37078

namespace SemanticResult37084
def owner : Owner := ⟨.program ⟨214⟩, ⟨12981⟩⟩
def rawTerms : List Term := Proof.Events144.exact37084RawTerms
def summary : Bound := (.finite 95463680)
def resultEvent : Nat := 37084
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37084.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge37082.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult37078.owner)
    (rightOwner := SemanticResult37048.owner)
    (leftResult := 37078) (rightResult := 37048)
    (leftActual := SemanticResult37078.actual selector witness)
    (rightActual := SemanticResult37048.actual selector witness)
    (leftRaw := SemanticResult37078.rawTerms)
    (rightRaw := SemanticResult37048.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 43264) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 37079) (rightBinding := 37080)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10149⟩) (rightExpression := ⟨12980⟩)
    (coefficientTransfer := 37081) (summaryTransfer := 37083)
    (base := LeftOperatorMerge37082.base)
    (reconstruction := LeftOperatorMerge37082.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37078.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult37048.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge37082.operationAgreement
  · rfl
  · decide
end SemanticResult37084

namespace SemanticResult37094
def owner : Owner := ⟨.program ⟨214⟩, ⟨25615⟩⟩
def rawTerms : List Term := Proof.Events144.exact37094RawTerms
def summary : Bound := (.finite 350353233018880)
def resultEvent : Nat := 37094
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37094.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95463680, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37090.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge37090.frameStart)
    (owner := owner) (leftOwner := SemanticResult37084.owner)
    (rightOwner := SemanticResult37020.owner)
    (leftResult := 37084) (rightResult := 37020)
    (leftActual := SemanticResult37084.actual selector witness)
    (rightActual := SemanticResult37020.actual selector witness)
    (leftRaw := SemanticResult37084.rawTerms)
    (rightRaw := SemanticResult37020.rawTerms)
    (working := LeftOperatorMerge37090.working)
    (leftBinding := 37085) (rightBinding := 37086)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12981⟩) (rightExpression := ⟨25614⟩)
    (coefficientTransfer := 37087) (summaryTransfer := 37089)
    (rightCoefficientProducer := 37019)
    (rightSummaryTransfer := 37088)
    (leftMaximum := ⟨95463680, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge37090.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority37019.actual selector witness)
    (summaryMagnitude := LeftBound37089.actual selector witness)
    (reconstruction := LeftOperatorMerge37090.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37084.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult37020.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37019.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority37019.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge37090.operationAgreement
  · exact LeftBound37089.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37090.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 37091 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25614⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩, ⟨.program ⟨214⟩, ⟨12974⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23336⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23336⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge37090.working
    [{ coefficient := (-1), key := LeftRelationMerge37091.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge37091.frameStart
      LeftRelationMerge37091.owner (.relation 37091) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge37091.deltas
    rows := LeftRelationMerge37091.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge37090.working LeftRelationMerge37091.source
        (relationContext LeftRelationMerge37091.source
          LeftRelationMerge37091.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge37090.working, LeftRelationMerge37091.deltas,
    LeftRelationMerge37091.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 37091)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25615⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10145⟩⟩, ⟨.program ⟨214⟩, ⟨12974⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25614⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25614⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge37090.working) (working := relationWorking0)
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
end SemanticResult37094

namespace SemanticResult37097
def owner : Owner := ⟨.program ⟨214⟩, ⟨20112⟩⟩
def rawTerms : List Term := Proof.Events144.exact37097RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37097
def producerEvent : Nat := 37096
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37097.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨24⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨24⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult37097

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
