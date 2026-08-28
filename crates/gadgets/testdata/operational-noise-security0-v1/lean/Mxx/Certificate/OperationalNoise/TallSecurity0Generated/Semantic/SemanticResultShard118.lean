import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard118
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard002
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard055
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard117

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult13992
def owner : Owner := ⟨.program ⟨214⟩, ⟨7382⟩⟩
def rawTerms : List Term := Proof.Events054.exact13992RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13992
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13992.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge13991.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge13991.frameStart)
    (transferEvent := 13990) (owner := owner)
    (leftResult := 6314) (rightResult := 13987)
    (working := LeftOperatorMerge13991.working)
    (reconstruction := LeftOperatorMerge13991.reconstruction)
    (leftReference := .predecessor 0 13988 .coefficient) (rightReference := .predecessor 1 13989 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult6314.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13987.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge13991.operationAgreement
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
end SemanticResult13992

namespace SemanticResult13996
def owner : Owner := ⟨.program ⟨214⟩, ⟨11013⟩⟩
def rawTerms : List Term := Proof.Events054.exact13996RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 13996
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult13996.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 13993) (rightBinding := 13994)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7382⟩) (rightExpression := ⟨11012⟩)
    (transferEvent := 13995)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13992.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13984.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult13996

namespace SemanticResult14002
def owner : Owner := ⟨.program ⟨214⟩, ⟨11014⟩⟩
def rawTerms : List Term := Proof.Events054.exact14002RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 14002
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14002.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 13999) (survivorTransfer := 14000)
    (survivorEvent := 14001) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13978)
    (owner := owner) (leftOwner := SemanticResult13996.owner)
    (rightOwner := SemanticResult13979.owner)
    (leftResult := 13996) (rightResult := 13979)
    (leftBinding := 13997) (rightBinding := 13998)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11013⟩) (rightExpression := ⟨88⟩)
    (leftActual := SemanticResult13996.actual selector witness)
    (rightActual := SemanticResult13979.actual selector witness)
    (leftRaw := SemanticResult13996.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨88⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13978.actual selector witness)
    (survivorMagnitude := LeftBound14000.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult13996.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13979.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13978.derived selector witness)
  · exact LeftBound14000.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult14002

namespace SemanticResult14010
def owner : Owner := ⟨.program ⟨214⟩, ⟨11015⟩⟩
def rawTerms : List Term := Proof.Events054.exact14010RawTerms
def summary : Bound := (.finite 3328)
def resultEvent : Nat := 14010
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14010.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨4, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge14008.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge14008.frameStart)
    (owner := owner) (leftOwner := SemanticResult14002.owner)
    (rightOwner := SemanticResult399.owner)
    (leftResult := 14002) (rightResult := 399)
    (leftActual := SemanticResult14002.actual selector witness)
    (rightActual := SemanticResult399.actual selector witness)
    (leftRaw := SemanticResult14002.rawTerms)
    (rightRaw := SemanticResult399.rawTerms)
    (working := LeftOperatorMerge14008.working)
    (leftBinding := 14003) (rightBinding := 14004)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11014⟩) (rightExpression := ⟨10862⟩)
    (coefficientTransfer := 14005) (summaryTransfer := 14007)
    (rightCoefficientProducer := 398)
    (rightSummaryTransfer := 14006)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨4, by decide⟩)
    (rightRecordedMaximum := 4)
    (rightSummaryMaximum := ⟨4, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge14008.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority398.actual selector witness)
    (summaryMagnitude := LeftBound14007.actual selector witness)
    (reconstruction := LeftOperatorMerge14008.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14002.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult399.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority398.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority398.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge14008.operationAgreement
  · exact LeftBound14007.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge14008.working summary) := by
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
end SemanticResult14010

namespace SemanticResult14013
def owner : Owner := ⟨.program ⟨214⟩, ⟨7837⟩⟩
def rawTerms : List Term := Proof.Events054.exact14013RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14013
def producerEvent : Nat := 14012
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14013.actual selector witness
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
end SemanticResult14013

namespace SemanticResult14017
def owner : Owner := ⟨.program ⟨214⟩, ⟨7838⟩⟩
def rawTerms : List Term := Proof.Events054.exact14017RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14017
def producerEvent : Nat := 14016
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14017.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 14014 .coefficient) (.value (.predecessor 1 14015 .coefficient)), 0, .finite 8192, .scale (.predecessor 0 14014 .coefficient) (.value (.predecessor 1 14015 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult14017

namespace SemanticResult14020
def owner : Owner := ⟨.program ⟨214⟩, ⟨105⟩⟩
def rawTerms : List Term := Proof.Events054.exact14020RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14020
def producerEvent : Nat := 14019
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14020.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 14018 .coefficient), 0, .finite 26, .identity (.predecessor 0 14018 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult14020

namespace SemanticResult14025
def owner : Owner := ⟨.program ⟨214⟩, ⟨10863⟩⟩
def rawTerms : List Term := Proof.Events054.exact14025RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14025
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14025.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge14024.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge14024.frameStart)
    (transferEvent := 14023) (owner := owner)
    (leftResult := 399) (rightResult := 6449)
    (working := LeftOperatorMerge14024.working)
    (reconstruction := LeftOperatorMerge14024.reconstruction)
    (leftReference := .predecessor 0 14021 .coefficient) (rightReference := .predecessor 1 14022 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult399.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6449.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge14024.operationAgreement
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
end SemanticResult14025

namespace SemanticResult14028
def owner : Owner := ⟨.program ⟨214⟩, ⟨6791⟩⟩
def rawTerms : List Term := Proof.Events054.exact14028RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14028
def producerEvent : Nat := 14027
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14028.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 14026 .coefficient), 0, .large, .identity (.predecessor 0 14026 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult14028

namespace SemanticResult14033
def owner : Owner := ⟨.program ⟨214⟩, ⟨7399⟩⟩
def rawTerms : List Term := Proof.Events054.exact14033RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14033
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14033.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge14032.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge14032.frameStart)
    (transferEvent := 14031) (owner := owner)
    (leftResult := 6314) (rightResult := 14028)
    (working := LeftOperatorMerge14032.working)
    (reconstruction := LeftOperatorMerge14032.reconstruction)
    (leftReference := .predecessor 0 14029 .coefficient) (rightReference := .predecessor 1 14030 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult6314.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14028.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge14032.operationAgreement
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
end SemanticResult14033

namespace SemanticResult14037
def owner : Owner := ⟨.program ⟨214⟩, ⟨10864⟩⟩
def rawTerms : List Term := Proof.Events054.exact14037RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14037
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14037.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 14034) (rightBinding := 14035)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7399⟩) (rightExpression := ⟨10863⟩)
    (transferEvent := 14036)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14033.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14025.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult14037

namespace SemanticResult14043
def owner : Owner := ⟨.program ⟨214⟩, ⟨10865⟩⟩
def rawTerms : List Term := Proof.Events054.exact14043RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 14043
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14043.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 14040) (survivorTransfer := 14041)
    (survivorEvent := 14042) (resultEvent := resultEvent)
    (rightCoefficientProducer := 14019)
    (owner := owner) (leftOwner := SemanticResult14037.owner)
    (rightOwner := SemanticResult14020.owner)
    (leftResult := 14037) (rightResult := 14020)
    (leftBinding := 14038) (rightBinding := 14039)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10864⟩) (rightExpression := ⟨105⟩)
    (leftActual := SemanticResult14037.actual selector witness)
    (rightActual := SemanticResult14020.actual selector witness)
    (leftRaw := SemanticResult14037.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨105⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound14019.actual selector witness)
    (survivorMagnitude := LeftBound14041.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14037.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14020.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14019.derived selector witness)
  · exact LeftBound14041.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult14043

namespace SemanticResult14053
def owner : Owner := ⟨.program ⟨214⟩, ⟨10866⟩⟩
def rawTerms : List Term := Proof.Events054.exact14053RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 14053
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14053.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge14049.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge14049.frameStart)
    (owner := owner) (leftOwner := SemanticResult14043.owner)
    (rightOwner := SemanticResult14017.owner)
    (leftResult := 14043) (rightResult := 14017)
    (leftActual := SemanticResult14043.actual selector witness)
    (rightActual := SemanticResult14017.actual selector witness)
    (leftRaw := SemanticResult14043.rawTerms)
    (rightRaw := SemanticResult14017.rawTerms)
    (working := LeftOperatorMerge14049.working)
    (leftBinding := 14044) (rightBinding := 14045)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10865⟩) (rightExpression := ⟨7838⟩)
    (coefficientTransfer := 14046) (summaryTransfer := 14048)
    (rightCoefficientProducer := 14016)
    (rightSummaryTransfer := 14047)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge14049.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound14016.actual selector witness)
    (summaryMagnitude := LeftBound14048.actual selector witness)
    (reconstruction := LeftOperatorMerge14049.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14043.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14017.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14016.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound14016.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge14049.operationAgreement
  · exact LeftBound14048.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge14049.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 14050 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10862⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge14049.working
    [{ coefficient := (-1), key := LeftRelationMerge14050.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge14050.frameStart
      LeftRelationMerge14050.owner (.relation 14050) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge14050.deltas
    rows := LeftRelationMerge14050.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge14049.working LeftRelationMerge14050.source
        (relationContext LeftRelationMerge14050.source
          LeftRelationMerge14050.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge14049.working, LeftRelationMerge14050.deltas,
    LeftRelationMerge14050.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 14050)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10866⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10862⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge14049.working) (working := relationWorking0)
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
end SemanticResult14053

namespace SemanticResult14059
def owner : Owner := ⟨.program ⟨214⟩, ⟨11016⟩⟩
def rawTerms : List Term := Proof.Events054.exact14059RawTerms
def summary : Bound := (.finite 95423744)
def resultEvent : Nat := 14059
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14059.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge14057.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult14053.owner)
    (rightOwner := SemanticResult14010.owner)
    (leftResult := 14053) (rightResult := 14010)
    (leftActual := SemanticResult14053.actual selector witness)
    (rightActual := SemanticResult14010.actual selector witness)
    (leftRaw := SemanticResult14053.rawTerms)
    (rightRaw := SemanticResult14010.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 3328) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 14054) (rightBinding := 14055)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10866⟩) (rightExpression := ⟨11015⟩)
    (coefficientTransfer := 14056) (summaryTransfer := 14058)
    (base := LeftOperatorMerge14057.base)
    (reconstruction := LeftOperatorMerge14057.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14053.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14010.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge14057.operationAgreement
  · rfl
  · decide
end SemanticResult14059

namespace SemanticResult14069
def owner : Owner := ⟨.program ⟨214⟩, ⟨25086⟩⟩
def rawTerms : List Term := Proof.Events054.exact14069RawTerms
def summary : Bound := (.finite 350206667259904)
def resultEvent : Nat := 14069
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14069.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95423744, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge14065.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge14065.frameStart)
    (owner := owner) (leftOwner := SemanticResult14059.owner)
    (rightOwner := SemanticResult13976.owner)
    (leftResult := 14059) (rightResult := 13976)
    (leftActual := SemanticResult14059.actual selector witness)
    (rightActual := SemanticResult13976.actual selector witness)
    (leftRaw := SemanticResult14059.rawTerms)
    (rightRaw := SemanticResult13976.rawTerms)
    (working := LeftOperatorMerge14065.working)
    (leftBinding := 14060) (rightBinding := 14061)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11016⟩) (rightExpression := ⟨25085⟩)
    (coefficientTransfer := 14062) (summaryTransfer := 14064)
    (rightCoefficientProducer := 13975)
    (rightSummaryTransfer := 14063)
    (leftMaximum := ⟨95423744, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge14065.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority13975.actual selector witness)
    (summaryMagnitude := LeftBound14064.actual selector witness)
    (reconstruction := LeftOperatorMerge14065.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult14059.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13976.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13975.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority13975.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge14065.operationAgreement
  · exact LeftBound14064.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge14065.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 14066 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25085⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10862⟩⟩, ⟨.program ⟨214⟩, ⟨11009⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23046⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23046⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge14065.working
    [{ coefficient := (-1), key := LeftRelationMerge14066.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge14066.frameStart
      LeftRelationMerge14066.owner (.relation 14066) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge14066.deltas
    rows := LeftRelationMerge14066.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge14065.working LeftRelationMerge14066.source
        (relationContext LeftRelationMerge14066.source
          LeftRelationMerge14066.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge14065.working, LeftRelationMerge14066.deltas,
    LeftRelationMerge14066.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 14066)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25086⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10862⟩⟩, ⟨.program ⟨214⟩, ⟨11009⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25085⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25085⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge14065.working) (working := relationWorking0)
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
end SemanticResult14069

namespace SemanticResult14072
def owner : Owner := ⟨.program ⟨214⟩, ⟨19184⟩⟩
def rawTerms : List Term := Proof.Events054.exact14072RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 14072
def producerEvent : Nat := 14071
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult14072.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨9⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨9⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult14072

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
