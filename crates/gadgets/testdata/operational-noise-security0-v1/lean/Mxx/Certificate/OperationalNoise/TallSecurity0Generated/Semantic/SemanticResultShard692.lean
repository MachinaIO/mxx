import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard692
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard038
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard085
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard086
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard690
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard691

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult97399
def owner : Owner := ⟨.program ⟨214⟩, ⟨28702⟩⟩
def rawTerms : List Term := Proof.Events380.exact97399RawTerms
def summary : Bound := (.finite 1292270185944771604480)
def resultEvent : Nat := 97399
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97399.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge97396.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult97392.owner)
    (rightOwner := SemanticResult97238.owner)
    (leftResult := 97392) (rightResult := 97238)
    (leftActual := SemanticResult97392.actual selector witness)
    (rightActual := SemanticResult97238.actual selector witness)
    (leftRaw := SemanticResult97392.rawTerms)
    (rightRaw := SemanticResult97238.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292270184133468094464) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 97393) (rightBinding := 97394)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21968⟩) (rightExpression := ⟨28701⟩)
    (coefficientTransfer := 97395) (summaryTransfer := 97398)
    (base := LeftOperatorMerge97396.base)
    (reconstruction := LeftOperatorMerge97396.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult97392.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult97238.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge97396.operationAgreement
  · rfl
  · decide
end SemanticResult97399

namespace SemanticResult97406
def owner : Owner := ⟨.program ⟨214⟩, ⟨24342⟩⟩
def rawTerms : List Term := Proof.Events380.exact97406RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 97406
def producerEvent : Nat := 97405
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97406.actual selector witness
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
end SemanticResult97406

namespace SemanticResult97409
def owner : Owner := ⟨.program ⟨214⟩, ⟨28482⟩⟩
def rawTerms : List Term := Proof.Events380.exact97409RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 97409
def producerEvent : Nat := 97408
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97409.actual selector witness
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
end SemanticResult97409

namespace SemanticResult97416
def owner : Owner := ⟨.program ⟨214⟩, ⟨23074⟩⟩
def rawTerms : List Term := Proof.Events380.exact97416RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 97416
def producerEvent : Nat := 97415
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97416.actual selector witness
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
end SemanticResult97416

namespace SemanticResult97419
def owner : Owner := ⟨.program ⟨214⟩, ⟨25129⟩⟩
def rawTerms : List Term := Proof.Events380.exact97419RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 97419
def producerEvent : Nat := 97418
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97419.actual selector witness
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
end SemanticResult97419

namespace SemanticResult97424
def owner : Owner := ⟨.program ⟨214⟩, ⟨11740⟩⟩
def rawTerms : List Term := Proof.Events380.exact97424RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 97424
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97424.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge97423.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge97423.frameStart)
    (transferEvent := 97422) (owner := owner)
    (leftResult := 4727) (rightResult := 32)
    (working := LeftOperatorMerge97423.working)
    (reconstruction := LeftOperatorMerge97423.reconstruction)
    (leftReference := .predecessor 0 97420 .coefficient) (rightReference := .predecessor 1 97421 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4727.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge97423.operationAgreement
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
end SemanticResult97424

namespace SemanticResult97429
def owner : Owner := ⟨.program ⟨214⟩, ⟨7120⟩⟩
def rawTerms : List Term := Proof.Events380.exact97429RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 97429
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97429.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge97428.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge97428.frameStart)
    (transferEvent := 97427) (owner := owner)
    (leftResult := 27) (rightResult := 9979)
    (working := LeftOperatorMerge97428.working)
    (reconstruction := LeftOperatorMerge97428.reconstruction)
    (leftReference := .predecessor 0 97425 .coefficient) (rightReference := .predecessor 1 97426 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9979.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge97428.operationAgreement
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
end SemanticResult97429

namespace SemanticResult97433
def owner : Owner := ⟨.program ⟨214⟩, ⟨11741⟩⟩
def rawTerms : List Term := Proof.Events380.exact97433RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 97433
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97433.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 97430) (rightBinding := 97431)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7120⟩) (rightExpression := ⟨11740⟩)
    (transferEvent := 97432)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult97429.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult97424.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult97433

namespace SemanticResult97439
def owner : Owner := ⟨.program ⟨214⟩, ⟨11742⟩⟩
def rawTerms : List Term := Proof.Events380.exact97439RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 97439
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97439.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 97436) (survivorTransfer := 97437)
    (survivorEvent := 97438) (resultEvent := resultEvent)
    (rightCoefficientProducer := 9970)
    (owner := owner) (leftOwner := SemanticResult97433.owner)
    (rightOwner := SemanticResult9971.owner)
    (leftResult := 97433) (rightResult := 9971)
    (leftBinding := 97434) (rightBinding := 97435)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11741⟩) (rightExpression := ⟨97⟩)
    (leftActual := SemanticResult97433.actual selector witness)
    (rightActual := SemanticResult9971.actual selector witness)
    (leftRaw := SemanticResult97433.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨97⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound9970.actual selector witness)
    (survivorMagnitude := LeftBound97437.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult97433.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9971.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9970.derived selector witness)
  · exact LeftBound97437.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult97439

namespace SemanticResult97447
def owner : Owner := ⟨.program ⟨214⟩, ⟨11743⟩⟩
def rawTerms : List Term := Proof.Events380.exact97447RawTerms
def summary : Bound := (.finite 24960)
def resultEvent : Nat := 97447
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97447.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨30, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge97445.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge97445.frameStart)
    (owner := owner) (leftOwner := SemanticResult97439.owner)
    (rightOwner := SemanticResult4730.owner)
    (leftResult := 97439) (rightResult := 4730)
    (leftActual := SemanticResult97439.actual selector witness)
    (rightActual := SemanticResult4730.actual selector witness)
    (leftRaw := SemanticResult97439.rawTerms)
    (rightRaw := SemanticResult4730.rawTerms)
    (working := LeftOperatorMerge97445.working)
    (leftBinding := 97440) (rightBinding := 97441)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11742⟩) (rightExpression := ⟨9595⟩)
    (coefficientTransfer := 97442) (summaryTransfer := 97444)
    (rightCoefficientProducer := 4729)
    (rightSummaryTransfer := 97443)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨30, by decide⟩)
    (rightRecordedMaximum := 30)
    (rightSummaryMaximum := ⟨30, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge97445.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority4729.actual selector witness)
    (summaryMagnitude := LeftBound97444.actual selector witness)
    (reconstruction := LeftOperatorMerge97445.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult97439.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4730.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4729.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority4729.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge97445.operationAgreement
  · exact LeftBound97444.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge97445.working summary) := by
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
end SemanticResult97447

namespace SemanticResult97452
def owner : Owner := ⟨.program ⟨214⟩, ⟨9596⟩⟩
def rawTerms : List Term := Proof.Events380.exact97452RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 97452
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97452.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge97451.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge97451.frameStart)
    (transferEvent := 97450) (owner := owner)
    (leftResult := 4730) (rightResult := 32)
    (working := LeftOperatorMerge97451.working)
    (reconstruction := LeftOperatorMerge97451.reconstruction)
    (leftReference := .predecessor 0 97448 .coefficient) (rightReference := .predecessor 1 97449 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4730.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge97451.operationAgreement
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
end SemanticResult97452

namespace SemanticResult97457
def owner : Owner := ⟨.program ⟨214⟩, ⟨7100⟩⟩
def rawTerms : List Term := Proof.Events380.exact97457RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 97457
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97457.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge97456.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge97456.frameStart)
    (transferEvent := 97455) (owner := owner)
    (leftResult := 27) (rightResult := 10020)
    (working := LeftOperatorMerge97456.working)
    (reconstruction := LeftOperatorMerge97456.reconstruction)
    (leftReference := .predecessor 0 97453 .coefficient) (rightReference := .predecessor 1 97454 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10020.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge97456.operationAgreement
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
end SemanticResult97457

namespace SemanticResult97461
def owner : Owner := ⟨.program ⟨214⟩, ⟨9597⟩⟩
def rawTerms : List Term := Proof.Events380.exact97461RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 97461
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97461.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 97458) (rightBinding := 97459)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7100⟩) (rightExpression := ⟨9596⟩)
    (transferEvent := 97460)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult97457.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult97452.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult97461

namespace SemanticResult97467
def owner : Owner := ⟨.program ⟨214⟩, ⟨9598⟩⟩
def rawTerms : List Term := Proof.Events380.exact97467RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 97467
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97467.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 97464) (survivorTransfer := 97465)
    (survivorEvent := 97466) (resultEvent := resultEvent)
    (rightCoefficientProducer := 10011)
    (owner := owner) (leftOwner := SemanticResult97461.owner)
    (rightOwner := SemanticResult10012.owner)
    (leftResult := 97461) (rightResult := 10012)
    (leftBinding := 97462) (rightBinding := 97463)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9597⟩) (rightExpression := ⟨77⟩)
    (leftActual := SemanticResult97461.actual selector witness)
    (rightActual := SemanticResult10012.actual selector witness)
    (leftRaw := SemanticResult97461.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨77⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound10011.actual selector witness)
    (survivorMagnitude := LeftBound97465.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult97461.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10012.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10011.derived selector witness)
  · exact LeftBound97465.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult97467

namespace SemanticResult97477
def owner : Owner := ⟨.program ⟨214⟩, ⟨9599⟩⟩
def rawTerms : List Term := Proof.Events380.exact97477RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 97477
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97477.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge97473.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge97473.frameStart)
    (owner := owner) (leftOwner := SemanticResult97467.owner)
    (rightOwner := SemanticResult10009.owner)
    (leftResult := 97467) (rightResult := 10009)
    (leftActual := SemanticResult97467.actual selector witness)
    (rightActual := SemanticResult10009.actual selector witness)
    (leftRaw := SemanticResult97467.rawTerms)
    (rightRaw := SemanticResult10009.rawTerms)
    (working := LeftOperatorMerge97473.working)
    (leftBinding := 97468) (rightBinding := 97469)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9598⟩) (rightExpression := ⟨7862⟩)
    (coefficientTransfer := 97470) (summaryTransfer := 97472)
    (rightCoefficientProducer := 10008)
    (rightSummaryTransfer := 97471)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge97473.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound10008.actual selector witness)
    (summaryMagnitude := LeftBound97472.actual selector witness)
    (reconstruction := LeftOperatorMerge97473.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult97467.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10009.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10008.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound10008.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge97473.operationAgreement
  · exact LeftBound97472.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge97473.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 97474 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge97473.working
    [{ coefficient := (-1), key := LeftRelationMerge97474.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge97474.frameStart
      LeftRelationMerge97474.owner (.relation 97474) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge97474.deltas
    rows := LeftRelationMerge97474.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge97473.working LeftRelationMerge97474.source
        (relationContext LeftRelationMerge97474.source
          LeftRelationMerge97474.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge97473.working, LeftRelationMerge97474.deltas,
    LeftRelationMerge97474.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 97474)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9599⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9595⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge97473.working) (working := relationWorking0)
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
end SemanticResult97477

namespace SemanticResult97483
def owner : Owner := ⟨.program ⟨214⟩, ⟨11744⟩⟩
def rawTerms : List Term := Proof.Events380.exact97483RawTerms
def summary : Bound := (.finite 95445376)
def resultEvent : Nat := 97483
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult97483.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge97481.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult97477.owner)
    (rightOwner := SemanticResult97447.owner)
    (leftResult := 97477) (rightResult := 97447)
    (leftActual := SemanticResult97477.actual selector witness)
    (rightActual := SemanticResult97447.actual selector witness)
    (leftRaw := SemanticResult97477.rawTerms)
    (rightRaw := SemanticResult97447.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 24960) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 97478) (rightBinding := 97479)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9599⟩) (rightExpression := ⟨11743⟩)
    (coefficientTransfer := 97480) (summaryTransfer := 97482)
    (base := LeftOperatorMerge97481.base)
    (reconstruction := LeftOperatorMerge97481.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult97477.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult97447.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge97481.operationAgreement
  · rfl
  · decide
end SemanticResult97483

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
