import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard179
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard008
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard073
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard163
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard164
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard178

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult23364
def owner : Owner := ⟨.program ⟨214⟩, ⟨12593⟩⟩
def rawTerms : List Term := Proof.Events091.exact23364RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23364
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23364.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23363.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge23363.frameStart)
    (transferEvent := 23362) (owner := owner)
    (leftResult := 934) (rightResult := 21420)
    (working := LeftOperatorMerge23363.working)
    (reconstruction := LeftOperatorMerge23363.reconstruction)
    (leftReference := .predecessor 0 23360 .coefficient) (rightReference := .predecessor 1 23361 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult934.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge23363.operationAgreement
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
end SemanticResult23364

namespace SemanticResult23369
def owner : Owner := ⟨.program ⟨214⟩, ⟨7356⟩⟩
def rawTerms : List Term := Proof.Events091.exact23369RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23369
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23369.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23368.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge23368.frameStart)
    (transferEvent := 23367) (owner := owner)
    (leftResult := 21290) (rightResult := 8476)
    (working := LeftOperatorMerge23368.working)
    (reconstruction := LeftOperatorMerge23368.reconstruction)
    (leftReference := .predecessor 0 23365 .coefficient) (rightReference := .predecessor 1 23366 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8476.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge23368.operationAgreement
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
end SemanticResult23369

namespace SemanticResult23373
def owner : Owner := ⟨.program ⟨214⟩, ⟨12594⟩⟩
def rawTerms : List Term := Proof.Events091.exact23373RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23373
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23373.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 23370) (rightBinding := 23371)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7356⟩) (rightExpression := ⟨12593⟩)
    (transferEvent := 23372)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23369.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23364.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult23373

namespace SemanticResult23379
def owner : Owner := ⟨.program ⟨214⟩, ⟨12595⟩⟩
def rawTerms : List Term := Proof.Events091.exact23379RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 23379
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23379.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 23376) (survivorTransfer := 23377)
    (survivorEvent := 23378) (resultEvent := resultEvent)
    (rightCoefficientProducer := 8467)
    (owner := owner) (leftOwner := SemanticResult23373.owner)
    (rightOwner := SemanticResult8468.owner)
    (leftResult := 23373) (rightResult := 8468)
    (leftBinding := 23374) (rightBinding := 23375)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12594⟩) (rightExpression := ⟨100⟩)
    (leftActual := SemanticResult23373.actual selector witness)
    (rightActual := SemanticResult8468.actual selector witness)
    (leftRaw := SemanticResult23373.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨100⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound8467.actual selector witness)
    (survivorMagnitude := LeftBound23377.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23373.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8468.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8467.derived selector witness)
  · exact LeftBound23377.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult23379

namespace SemanticResult23387
def owner : Owner := ⟨.program ⟨214⟩, ⟨12596⟩⟩
def rawTerms : List Term := Proof.Events091.exact23387RawTerms
def summary : Bound := (.finite 34944)
def resultEvent : Nat := 23387
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23387.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨42, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23385.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge23385.frameStart)
    (owner := owner) (leftOwner := SemanticResult23379.owner)
    (rightOwner := SemanticResult937.owner)
    (leftResult := 23379) (rightResult := 937)
    (leftActual := SemanticResult23379.actual selector witness)
    (rightActual := SemanticResult937.actual selector witness)
    (leftRaw := SemanticResult23379.rawTerms)
    (rightRaw := SemanticResult937.rawTerms)
    (working := LeftOperatorMerge23385.working)
    (leftBinding := 23380) (rightBinding := 23381)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12595⟩) (rightExpression := ⟨9940⟩)
    (coefficientTransfer := 23382) (summaryTransfer := 23384)
    (rightCoefficientProducer := 936)
    (rightSummaryTransfer := 23383)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨42, by decide⟩)
    (rightRecordedMaximum := 42)
    (rightSummaryMaximum := ⟨42, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge23385.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority936.actual selector witness)
    (summaryMagnitude := LeftBound23384.actual selector witness)
    (reconstruction := LeftOperatorMerge23385.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23379.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult937.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority936.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority936.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge23385.operationAgreement
  · exact LeftBound23384.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23385.working summary) := by
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
end SemanticResult23387

namespace SemanticResult23392
def owner : Owner := ⟨.program ⟨214⟩, ⟨9941⟩⟩
def rawTerms : List Term := Proof.Events091.exact23392RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23392
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23392.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23391.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge23391.frameStart)
    (transferEvent := 23390) (owner := owner)
    (leftResult := 937) (rightResult := 21420)
    (working := LeftOperatorMerge23391.working)
    (reconstruction := LeftOperatorMerge23391.reconstruction)
    (leftReference := .predecessor 0 23388 .coefficient) (rightReference := .predecessor 1 23389 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult937.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult21420.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge23391.operationAgreement
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
end SemanticResult23392

namespace SemanticResult23397
def owner : Owner := ⟨.program ⟨214⟩, ⟨7336⟩⟩
def rawTerms : List Term := Proof.Events091.exact23397RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23397
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23397.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23396.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge23396.frameStart)
    (transferEvent := 23395) (owner := owner)
    (leftResult := 21290) (rightResult := 8517)
    (working := LeftOperatorMerge23396.working)
    (reconstruction := LeftOperatorMerge23396.reconstruction)
    (leftReference := .predecessor 0 23393 .coefficient) (rightReference := .predecessor 1 23394 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult21290.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8517.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge23396.operationAgreement
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
end SemanticResult23397

namespace SemanticResult23401
def owner : Owner := ⟨.program ⟨214⟩, ⟨9942⟩⟩
def rawTerms : List Term := Proof.Events091.exact23401RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23401
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23401.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 23398) (rightBinding := 23399)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7336⟩) (rightExpression := ⟨9941⟩)
    (transferEvent := 23400)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23397.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23392.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult23401

namespace SemanticResult23407
def owner : Owner := ⟨.program ⟨214⟩, ⟨9943⟩⟩
def rawTerms : List Term := Proof.Events091.exact23407RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 23407
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23407.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 23404) (survivorTransfer := 23405)
    (survivorEvent := 23406) (resultEvent := resultEvent)
    (rightCoefficientProducer := 8508)
    (owner := owner) (leftOwner := SemanticResult23401.owner)
    (rightOwner := SemanticResult8509.owner)
    (leftResult := 23401) (rightResult := 8509)
    (leftBinding := 23402) (rightBinding := 23403)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9942⟩) (rightExpression := ⟨80⟩)
    (leftActual := SemanticResult23401.actual selector witness)
    (rightActual := SemanticResult8509.actual selector witness)
    (leftRaw := SemanticResult23401.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨80⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound8508.actual selector witness)
    (survivorMagnitude := LeftBound23405.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23401.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8509.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8508.derived selector witness)
  · exact LeftBound23405.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult23407

namespace SemanticResult23417
def owner : Owner := ⟨.program ⟨214⟩, ⟨9944⟩⟩
def rawTerms : List Term := Proof.Events091.exact23417RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 23417
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23417.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23413.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge23413.frameStart)
    (owner := owner) (leftOwner := SemanticResult23407.owner)
    (rightOwner := SemanticResult8506.owner)
    (leftResult := 23407) (rightResult := 8506)
    (leftActual := SemanticResult23407.actual selector witness)
    (rightActual := SemanticResult8506.actual selector witness)
    (leftRaw := SemanticResult23407.rawTerms)
    (rightRaw := SemanticResult8506.rawTerms)
    (working := LeftOperatorMerge23413.working)
    (leftBinding := 23408) (rightBinding := 23409)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9943⟩) (rightExpression := ⟨7871⟩)
    (coefficientTransfer := 23410) (summaryTransfer := 23412)
    (rightCoefficientProducer := 8505)
    (rightSummaryTransfer := 23411)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge23413.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound8505.actual selector witness)
    (summaryMagnitude := LeftBound23412.actual selector witness)
    (reconstruction := LeftOperatorMerge23413.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23407.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8506.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8505.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound8505.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge23413.operationAgreement
  · exact LeftBound23412.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23413.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 23414 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9940⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge23413.working
    [{ coefficient := (-1), key := LeftRelationMerge23414.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge23414.frameStart
      LeftRelationMerge23414.owner (.relation 23414) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge23414.deltas
    rows := LeftRelationMerge23414.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge23413.working LeftRelationMerge23414.source
        (relationContext LeftRelationMerge23414.source
          LeftRelationMerge23414.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge23413.working, LeftRelationMerge23414.deltas,
    LeftRelationMerge23414.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 23414)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9944⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9940⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge23413.working) (working := relationWorking0)
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
end SemanticResult23417

namespace SemanticResult23423
def owner : Owner := ⟨.program ⟨214⟩, ⟨12597⟩⟩
def rawTerms : List Term := Proof.Events091.exact23423RawTerms
def summary : Bound := (.finite 95455360)
def resultEvent : Nat := 23423
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23423.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge23421.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult23417.owner)
    (rightOwner := SemanticResult23387.owner)
    (leftResult := 23417) (rightResult := 23387)
    (leftActual := SemanticResult23417.actual selector witness)
    (rightActual := SemanticResult23387.actual selector witness)
    (leftRaw := SemanticResult23417.rawTerms)
    (rightRaw := SemanticResult23387.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 34944) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 23418) (rightBinding := 23419)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9944⟩) (rightExpression := ⟨12596⟩)
    (coefficientTransfer := 23420) (summaryTransfer := 23422)
    (base := LeftOperatorMerge23421.base)
    (reconstruction := LeftOperatorMerge23421.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23417.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23387.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge23421.operationAgreement
  · rfl
  · decide
end SemanticResult23423

namespace SemanticResult23433
def owner : Owner := ⟨.program ⟨214⟩, ⟨25466⟩⟩
def rawTerms : List Term := Proof.Events091.exact23433RawTerms
def summary : Bound := (.finite 350322698485760)
def resultEvent : Nat := 23433
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23433.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95455360, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23429.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge23429.frameStart)
    (owner := owner) (leftOwner := SemanticResult23423.owner)
    (rightOwner := SemanticResult23359.owner)
    (leftResult := 23423) (rightResult := 23359)
    (leftActual := SemanticResult23423.actual selector witness)
    (rightActual := SemanticResult23359.actual selector witness)
    (leftRaw := SemanticResult23423.rawTerms)
    (rightRaw := SemanticResult23359.rawTerms)
    (working := LeftOperatorMerge23429.working)
    (leftBinding := 23424) (rightBinding := 23425)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12597⟩) (rightExpression := ⟨25465⟩)
    (coefficientTransfer := 23426) (summaryTransfer := 23428)
    (rightCoefficientProducer := 23358)
    (rightSummaryTransfer := 23427)
    (leftMaximum := ⟨95455360, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge23429.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority23358.actual selector witness)
    (summaryMagnitude := LeftBound23428.actual selector witness)
    (reconstruction := LeftOperatorMerge23429.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult23423.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult23359.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23358.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority23358.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge23429.operationAgreement
  · exact LeftBound23428.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge23429.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 23430 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25465⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9940⟩⟩, ⟨.program ⟨214⟩, ⟨12590⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23254⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23254⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge23429.working
    [{ coefficient := (-1), key := LeftRelationMerge23430.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge23430.frameStart
      LeftRelationMerge23430.owner (.relation 23430) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge23430.deltas
    rows := LeftRelationMerge23430.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge23429.working LeftRelationMerge23430.source
        (relationContext LeftRelationMerge23430.source
          LeftRelationMerge23430.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge23429.working, LeftRelationMerge23430.deltas,
    LeftRelationMerge23430.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 23430)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25466⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9940⟩⟩, ⟨.program ⟨214⟩, ⟨12590⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25465⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25465⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge23429.working) (working := relationWorking0)
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
end SemanticResult23433

namespace SemanticResult23436
def owner : Owner := ⟨.program ⟨214⟩, ⟨19972⟩⟩
def rawTerms : List Term := Proof.Events091.exact23436RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23436
def producerEvent : Nat := 23435
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23436.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨21⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨21⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult23436

namespace SemanticResult23440
def owner : Owner := ⟨.program ⟨214⟩, ⟨19974⟩⟩
def rawTerms : List Term := Proof.Events091.exact23440RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23440
def producerEvent : Nat := 23439
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23440.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 23437 .coefficient) (.value (.predecessor 1 23438 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 23437 .coefficient) (.value (.predecessor 1 23438 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult23440

namespace SemanticResult23518
def owner : Owner := ⟨.program ⟨214⟩, ⟨12590⟩⟩
def rawTerms : List Term := Proof.Events091.exact23518RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23518
def producerEvent : Nat := 23517
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23518.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 23495, .finite 42, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult23518

namespace SemanticResult23521
def owner : Owner := ⟨.program ⟨214⟩, ⟨9940⟩⟩
def rawTerms : List Term := Proof.Events091.exact23521RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 23521
def producerEvent : Nat := 23520
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult23521.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 23495, .finite 42, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult23521

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
