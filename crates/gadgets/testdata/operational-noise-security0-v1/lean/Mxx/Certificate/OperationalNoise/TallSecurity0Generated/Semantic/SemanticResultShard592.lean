import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard592
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard032
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard085
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard086
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard565
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard566
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard590
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard591

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult83271
def owner : Owner := ⟨.program ⟨214⟩, ⟨28737⟩⟩
def rawTerms : List Term := Proof.Events325.exact83271RawTerms
def summary : Bound := (.finite 1292270185944771604480)
def resultEvent : Nat := 83271
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83271.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge83268.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult83264.owner)
    (rightOwner := SemanticResult83086.owner)
    (leftResult := 83264) (rightResult := 83086)
    (leftActual := SemanticResult83264.actual selector witness)
    (rightActual := SemanticResult83086.actual selector witness)
    (leftRaw := SemanticResult83264.rawTerms)
    (rightRaw := SemanticResult83086.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292270184133468094464) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 83265) (rightBinding := 83266)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨21979⟩) (rightExpression := ⟨28736⟩)
    (coefficientTransfer := 83267) (summaryTransfer := 83270)
    (base := LeftOperatorMerge83268.base)
    (reconstruction := LeftOperatorMerge83268.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83264.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult83086.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge83268.operationAgreement
  · rfl
  · decide
end SemanticResult83271

namespace SemanticResult83278
def owner : Owner := ⟨.program ⟨214⟩, ⟨24351⟩⟩
def rawTerms : List Term := Proof.Events325.exact83278RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83278
def producerEvent : Nat := 83277
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83278.actual selector witness
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
end SemanticResult83278

namespace SemanticResult83281
def owner : Owner := ⟨.program ⟨214⟩, ⟨28517⟩⟩
def rawTerms : List Term := Proof.Events325.exact83281RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83281
def producerEvent : Nat := 83280
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83281.actual selector witness
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
end SemanticResult83281

namespace SemanticResult83288
def owner : Owner := ⟨.program ⟨214⟩, ⟨23080⟩⟩
def rawTerms : List Term := Proof.Events325.exact83288RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83288
def producerEvent : Nat := 83287
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83288.actual selector witness
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
end SemanticResult83288

namespace SemanticResult83291
def owner : Owner := ⟨.program ⟨214⟩, ⟨25142⟩⟩
def rawTerms : List Term := Proof.Events325.exact83291RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83291
def producerEvent : Nat := 83290
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83291.actual selector witness
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
end SemanticResult83291

namespace SemanticResult83296
def owner : Owner := ⟨.program ⟨214⟩, ⟨11764⟩⟩
def rawTerms : List Term := Proof.Events325.exact83296RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83296
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83296.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83295.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge83295.frameStart)
    (transferEvent := 83294) (owner := owner)
    (leftResult := 3989) (rightResult := 79920)
    (working := LeftOperatorMerge83295.working)
    (reconstruction := LeftOperatorMerge83295.reconstruction)
    (leftReference := .predecessor 0 83292 .coefficient) (rightReference := .predecessor 1 83293 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3989.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79920.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge83295.operationAgreement
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
end SemanticResult83296

namespace SemanticResult83301
def owner : Owner := ⟨.program ⟨214⟩, ⟨7239⟩⟩
def rawTerms : List Term := Proof.Events325.exact83301RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83301
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83301.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83300.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge83300.frameStart)
    (transferEvent := 83299) (owner := owner)
    (leftResult := 79790) (rightResult := 9979)
    (working := LeftOperatorMerge83300.working)
    (reconstruction := LeftOperatorMerge83300.reconstruction)
    (leftReference := .predecessor 0 83297 .coefficient) (rightReference := .predecessor 1 83298 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9979.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge83300.operationAgreement
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
end SemanticResult83301

namespace SemanticResult83305
def owner : Owner := ⟨.program ⟨214⟩, ⟨11765⟩⟩
def rawTerms : List Term := Proof.Events325.exact83305RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83305
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83305.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 83302) (rightBinding := 83303)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7239⟩) (rightExpression := ⟨11764⟩)
    (transferEvent := 83304)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83301.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult83296.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult83305

namespace SemanticResult83311
def owner : Owner := ⟨.program ⟨214⟩, ⟨11766⟩⟩
def rawTerms : List Term := Proof.Events325.exact83311RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 83311
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83311.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 83308) (survivorTransfer := 83309)
    (survivorEvent := 83310) (resultEvent := resultEvent)
    (rightCoefficientProducer := 9970)
    (owner := owner) (leftOwner := SemanticResult83305.owner)
    (rightOwner := SemanticResult9971.owner)
    (leftResult := 83305) (rightResult := 9971)
    (leftBinding := 83306) (rightBinding := 83307)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11765⟩) (rightExpression := ⟨97⟩)
    (leftActual := SemanticResult83305.actual selector witness)
    (rightActual := SemanticResult9971.actual selector witness)
    (leftRaw := SemanticResult83305.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨97⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound9970.actual selector witness)
    (survivorMagnitude := LeftBound83309.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83305.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult9971.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9970.derived selector witness)
  · exact LeftBound83309.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult83311

namespace SemanticResult83319
def owner : Owner := ⟨.program ⟨214⟩, ⟨11767⟩⟩
def rawTerms : List Term := Proof.Events325.exact83319RawTerms
def summary : Bound := (.finite 24960)
def resultEvent : Nat := 83319
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83319.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨30, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83317.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge83317.frameStart)
    (owner := owner) (leftOwner := SemanticResult83311.owner)
    (rightOwner := SemanticResult3992.owner)
    (leftResult := 83311) (rightResult := 3992)
    (leftActual := SemanticResult83311.actual selector witness)
    (rightActual := SemanticResult3992.actual selector witness)
    (leftRaw := SemanticResult83311.rawTerms)
    (rightRaw := SemanticResult3992.rawTerms)
    (working := LeftOperatorMerge83317.working)
    (leftBinding := 83312) (rightBinding := 83313)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11766⟩) (rightExpression := ⟨9610⟩)
    (coefficientTransfer := 83314) (summaryTransfer := 83316)
    (rightCoefficientProducer := 3991)
    (rightSummaryTransfer := 83315)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨30, by decide⟩)
    (rightRecordedMaximum := 30)
    (rightSummaryMaximum := ⟨30, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge83317.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority3991.actual selector witness)
    (summaryMagnitude := LeftBound83316.actual selector witness)
    (reconstruction := LeftOperatorMerge83317.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83311.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3992.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3991.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority3991.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge83317.operationAgreement
  · exact LeftBound83316.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83317.working summary) := by
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
end SemanticResult83319

namespace SemanticResult83324
def owner : Owner := ⟨.program ⟨214⟩, ⟨9611⟩⟩
def rawTerms : List Term := Proof.Events325.exact83324RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83324
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83324.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83323.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge83323.frameStart)
    (transferEvent := 83322) (owner := owner)
    (leftResult := 3992) (rightResult := 79920)
    (working := LeftOperatorMerge83323.working)
    (reconstruction := LeftOperatorMerge83323.reconstruction)
    (leftReference := .predecessor 0 83320 .coefficient) (rightReference := .predecessor 1 83321 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3992.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult79920.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge83323.operationAgreement
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
end SemanticResult83324

namespace SemanticResult83329
def owner : Owner := ⟨.program ⟨214⟩, ⟨7219⟩⟩
def rawTerms : List Term := Proof.Events325.exact83329RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83329
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83329.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83328.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge83328.frameStart)
    (transferEvent := 83327) (owner := owner)
    (leftResult := 79790) (rightResult := 10020)
    (working := LeftOperatorMerge83328.working)
    (reconstruction := LeftOperatorMerge83328.reconstruction)
    (leftReference := .predecessor 0 83325 .coefficient) (rightReference := .predecessor 1 83326 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult79790.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10020.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge83328.operationAgreement
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
end SemanticResult83329

namespace SemanticResult83333
def owner : Owner := ⟨.program ⟨214⟩, ⟨9612⟩⟩
def rawTerms : List Term := Proof.Events325.exact83333RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 83333
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83333.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 83330) (rightBinding := 83331)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7219⟩) (rightExpression := ⟨9611⟩)
    (transferEvent := 83332)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83329.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult83324.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult83333

namespace SemanticResult83339
def owner : Owner := ⟨.program ⟨214⟩, ⟨9613⟩⟩
def rawTerms : List Term := Proof.Events325.exact83339RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 83339
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83339.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 83336) (survivorTransfer := 83337)
    (survivorEvent := 83338) (resultEvent := resultEvent)
    (rightCoefficientProducer := 10011)
    (owner := owner) (leftOwner := SemanticResult83333.owner)
    (rightOwner := SemanticResult10012.owner)
    (leftResult := 83333) (rightResult := 10012)
    (leftBinding := 83334) (rightBinding := 83335)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9612⟩) (rightExpression := ⟨77⟩)
    (leftActual := SemanticResult83333.actual selector witness)
    (rightActual := SemanticResult10012.actual selector witness)
    (leftRaw := SemanticResult83333.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨77⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound10011.actual selector witness)
    (survivorMagnitude := LeftBound83337.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83333.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10012.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10011.derived selector witness)
  · exact LeftBound83337.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult83339

namespace SemanticResult83349
def owner : Owner := ⟨.program ⟨214⟩, ⟨9614⟩⟩
def rawTerms : List Term := Proof.Events325.exact83349RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 83349
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83349.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83345.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge83345.frameStart)
    (owner := owner) (leftOwner := SemanticResult83339.owner)
    (rightOwner := SemanticResult10009.owner)
    (leftResult := 83339) (rightResult := 10009)
    (leftActual := SemanticResult83339.actual selector witness)
    (rightActual := SemanticResult10009.actual selector witness)
    (leftRaw := SemanticResult83339.rawTerms)
    (rightRaw := SemanticResult10009.rawTerms)
    (working := LeftOperatorMerge83345.working)
    (leftBinding := 83340) (rightBinding := 83341)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9613⟩) (rightExpression := ⟨7862⟩)
    (coefficientTransfer := 83342) (summaryTransfer := 83344)
    (rightCoefficientProducer := 10008)
    (rightSummaryTransfer := 83343)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge83345.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound10008.actual selector witness)
    (summaryMagnitude := LeftBound83344.actual selector witness)
    (reconstruction := LeftOperatorMerge83345.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83339.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10009.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10008.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound10008.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge83345.operationAgreement
  · exact LeftBound83344.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge83345.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 83346 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge83345.working
    [{ coefficient := (-1), key := LeftRelationMerge83346.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge83346.frameStart
      LeftRelationMerge83346.owner (.relation 83346) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge83346.deltas
    rows := LeftRelationMerge83346.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge83345.working LeftRelationMerge83346.source
        (relationContext LeftRelationMerge83346.source
          LeftRelationMerge83346.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge83345.working, LeftRelationMerge83346.deltas,
    LeftRelationMerge83346.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 83346)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨9614⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge83345.working) (working := relationWorking0)
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
end SemanticResult83349

namespace SemanticResult83355
def owner : Owner := ⟨.program ⟨214⟩, ⟨11768⟩⟩
def rawTerms : List Term := Proof.Events325.exact83355RawTerms
def summary : Bound := (.finite 95445376)
def resultEvent : Nat := 83355
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult83355.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge83353.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult83349.owner)
    (rightOwner := SemanticResult83319.owner)
    (leftResult := 83349) (rightResult := 83319)
    (leftActual := SemanticResult83349.actual selector witness)
    (rightActual := SemanticResult83319.actual selector witness)
    (leftRaw := SemanticResult83349.rawTerms)
    (rightRaw := SemanticResult83319.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 24960) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 83350) (rightBinding := 83351)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9614⟩) (rightExpression := ⟨11767⟩)
    (coefficientTransfer := 83352) (summaryTransfer := 83354)
    (base := LeftOperatorMerge83353.base)
    (reconstruction := LeftOperatorMerge83353.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult83349.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult83319.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge83353.operationAgreement
  · rfl
  · decide
end SemanticResult83355

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
