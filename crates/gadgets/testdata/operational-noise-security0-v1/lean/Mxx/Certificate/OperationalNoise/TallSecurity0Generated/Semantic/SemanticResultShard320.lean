import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard320
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard015
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard117
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard118
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard264

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult43283
def owner : Owner := ⟨.program ⟨214⟩, ⟨23042⟩⟩
def rawTerms : List Term := Proof.Events169.exact43283RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43283
def producerEvent : Nat := 43282
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43283.actual selector witness
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
end SemanticResult43283

namespace SemanticResult43286
def owner : Owner := ⟨.program ⟨214⟩, ⟨25075⟩⟩
def rawTerms : List Term := Proof.Events169.exact43286RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43286
def producerEvent : Nat := 43285
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43286.actual selector witness
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
end SemanticResult43286

namespace SemanticResult43291
def owner : Owner := ⟨.program ⟨214⟩, ⟨10996⟩⟩
def rawTerms : List Term := Proof.Events169.exact43291RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43291
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43291.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43290.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge43290.frameStart)
    (transferEvent := 43289) (owner := owner)
    (leftResult := 1935) (rightResult := 36045)
    (working := LeftOperatorMerge43290.working)
    (reconstruction := LeftOperatorMerge43290.reconstruction)
    (leftReference := .predecessor 0 43287 .coefficient) (rightReference := .predecessor 1 43288 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1935.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge43290.operationAgreement
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
end SemanticResult43291

namespace SemanticResult43296
def owner : Owner := ⟨.program ⟨214⟩, ⟨7306⟩⟩
def rawTerms : List Term := Proof.Events169.exact43296RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43296
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43296.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43295.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge43295.frameStart)
    (transferEvent := 43294) (owner := owner)
    (leftResult := 35915) (rightResult := 13987)
    (working := LeftOperatorMerge43295.working)
    (reconstruction := LeftOperatorMerge43295.reconstruction)
    (leftReference := .predecessor 0 43292 .coefficient) (rightReference := .predecessor 1 43293 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13987.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge43295.operationAgreement
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
end SemanticResult43296

namespace SemanticResult43300
def owner : Owner := ⟨.program ⟨214⟩, ⟨10997⟩⟩
def rawTerms : List Term := Proof.Events169.exact43300RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43300
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43300.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 43297) (rightBinding := 43298)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7306⟩) (rightExpression := ⟨10996⟩)
    (transferEvent := 43299)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult43296.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult43291.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult43300

namespace SemanticResult43306
def owner : Owner := ⟨.program ⟨214⟩, ⟨10998⟩⟩
def rawTerms : List Term := Proof.Events169.exact43306RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 43306
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43306.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 43303) (survivorTransfer := 43304)
    (survivorEvent := 43305) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13978)
    (owner := owner) (leftOwner := SemanticResult43300.owner)
    (rightOwner := SemanticResult13979.owner)
    (leftResult := 43300) (rightResult := 13979)
    (leftBinding := 43301) (rightBinding := 43302)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10997⟩) (rightExpression := ⟨88⟩)
    (leftActual := SemanticResult43300.actual selector witness)
    (rightActual := SemanticResult13979.actual selector witness)
    (leftRaw := SemanticResult43300.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨88⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13978.actual selector witness)
    (survivorMagnitude := LeftBound43304.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult43300.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13979.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13978.derived selector witness)
  · exact LeftBound43304.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult43306

namespace SemanticResult43314
def owner : Owner := ⟨.program ⟨214⟩, ⟨10999⟩⟩
def rawTerms : List Term := Proof.Events169.exact43314RawTerms
def summary : Bound := (.finite 3328)
def resultEvent : Nat := 43314
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43314.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨4, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43312.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge43312.frameStart)
    (owner := owner) (leftOwner := SemanticResult43306.owner)
    (rightOwner := SemanticResult1938.owner)
    (leftResult := 43306) (rightResult := 1938)
    (leftActual := SemanticResult43306.actual selector witness)
    (rightActual := SemanticResult1938.actual selector witness)
    (leftRaw := SemanticResult43306.rawTerms)
    (rightRaw := SemanticResult1938.rawTerms)
    (working := LeftOperatorMerge43312.working)
    (leftBinding := 43307) (rightBinding := 43308)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10998⟩) (rightExpression := ⟨10852⟩)
    (coefficientTransfer := 43309) (summaryTransfer := 43311)
    (rightCoefficientProducer := 1937)
    (rightSummaryTransfer := 43310)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨4, by decide⟩)
    (rightRecordedMaximum := 4)
    (rightSummaryMaximum := ⟨4, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge43312.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1937.actual selector witness)
    (summaryMagnitude := LeftBound43311.actual selector witness)
    (reconstruction := LeftOperatorMerge43312.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult43306.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1938.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1937.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1937.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge43312.operationAgreement
  · exact LeftBound43311.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43312.working summary) := by
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
end SemanticResult43314

namespace SemanticResult43319
def owner : Owner := ⟨.program ⟨214⟩, ⟨10853⟩⟩
def rawTerms : List Term := Proof.Events169.exact43319RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43319
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43319.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43318.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge43318.frameStart)
    (transferEvent := 43317) (owner := owner)
    (leftResult := 1938) (rightResult := 36045)
    (working := LeftOperatorMerge43318.working)
    (reconstruction := LeftOperatorMerge43318.reconstruction)
    (leftReference := .predecessor 0 43315 .coefficient) (rightReference := .predecessor 1 43316 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1938.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge43318.operationAgreement
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
end SemanticResult43319

namespace SemanticResult43324
def owner : Owner := ⟨.program ⟨214⟩, ⟨7323⟩⟩
def rawTerms : List Term := Proof.Events169.exact43324RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43324
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43324.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43323.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge43323.frameStart)
    (transferEvent := 43322) (owner := owner)
    (leftResult := 35915) (rightResult := 14028)
    (working := LeftOperatorMerge43323.working)
    (reconstruction := LeftOperatorMerge43323.reconstruction)
    (leftReference := .predecessor 0 43320 .coefficient) (rightReference := .predecessor 1 43321 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14028.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge43323.operationAgreement
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
end SemanticResult43324

namespace SemanticResult43328
def owner : Owner := ⟨.program ⟨214⟩, ⟨10854⟩⟩
def rawTerms : List Term := Proof.Events169.exact43328RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43328
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43328.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 43325) (rightBinding := 43326)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7323⟩) (rightExpression := ⟨10853⟩)
    (transferEvent := 43327)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult43324.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult43319.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult43328

namespace SemanticResult43334
def owner : Owner := ⟨.program ⟨214⟩, ⟨10855⟩⟩
def rawTerms : List Term := Proof.Events169.exact43334RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 43334
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43334.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 43331) (survivorTransfer := 43332)
    (survivorEvent := 43333) (resultEvent := resultEvent)
    (rightCoefficientProducer := 14019)
    (owner := owner) (leftOwner := SemanticResult43328.owner)
    (rightOwner := SemanticResult14020.owner)
    (leftResult := 43328) (rightResult := 14020)
    (leftBinding := 43329) (rightBinding := 43330)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10854⟩) (rightExpression := ⟨105⟩)
    (leftActual := SemanticResult43328.actual selector witness)
    (rightActual := SemanticResult14020.actual selector witness)
    (leftRaw := SemanticResult43328.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨105⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound14019.actual selector witness)
    (survivorMagnitude := LeftBound43332.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult43328.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14020.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14019.derived selector witness)
  · exact LeftBound43332.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult43334

namespace SemanticResult43344
def owner : Owner := ⟨.program ⟨214⟩, ⟨10856⟩⟩
def rawTerms : List Term := Proof.Events169.exact43344RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 43344
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43344.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43340.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge43340.frameStart)
    (owner := owner) (leftOwner := SemanticResult43334.owner)
    (rightOwner := SemanticResult14017.owner)
    (leftResult := 43334) (rightResult := 14017)
    (leftActual := SemanticResult43334.actual selector witness)
    (rightActual := SemanticResult14017.actual selector witness)
    (leftRaw := SemanticResult43334.rawTerms)
    (rightRaw := SemanticResult14017.rawTerms)
    (working := LeftOperatorMerge43340.working)
    (leftBinding := 43335) (rightBinding := 43336)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10855⟩) (rightExpression := ⟨7838⟩)
    (coefficientTransfer := 43337) (summaryTransfer := 43339)
    (rightCoefficientProducer := 14016)
    (rightSummaryTransfer := 43338)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge43340.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound14016.actual selector witness)
    (summaryMagnitude := LeftBound43339.actual selector witness)
    (reconstruction := LeftOperatorMerge43340.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult43334.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult14017.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14016.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound14016.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge43340.operationAgreement
  · exact LeftBound43339.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43340.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 43341 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6774⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge43340.working
    [{ coefficient := (-1), key := LeftRelationMerge43341.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge43341.frameStart
      LeftRelationMerge43341.owner (.relation 43341) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge43341.deltas
    rows := LeftRelationMerge43341.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge43340.working LeftRelationMerge43341.source
        (relationContext LeftRelationMerge43341.source
          LeftRelationMerge43341.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge43340.working, LeftRelationMerge43341.deltas,
    LeftRelationMerge43341.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 43341)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10856⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge43340.working) (working := relationWorking0)
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
end SemanticResult43344

namespace SemanticResult43350
def owner : Owner := ⟨.program ⟨214⟩, ⟨11000⟩⟩
def rawTerms : List Term := Proof.Events169.exact43350RawTerms
def summary : Bound := (.finite 95423744)
def resultEvent : Nat := 43350
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43350.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge43348.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult43344.owner)
    (rightOwner := SemanticResult43314.owner)
    (leftResult := 43344) (rightResult := 43314)
    (leftActual := SemanticResult43344.actual selector witness)
    (rightActual := SemanticResult43314.actual selector witness)
    (leftRaw := SemanticResult43344.rawTerms)
    (rightRaw := SemanticResult43314.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 3328) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 43345) (rightBinding := 43346)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10856⟩) (rightExpression := ⟨10999⟩)
    (coefficientTransfer := 43347) (summaryTransfer := 43349)
    (base := LeftOperatorMerge43348.base)
    (reconstruction := LeftOperatorMerge43348.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult43344.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult43314.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge43348.operationAgreement
  · rfl
  · decide
end SemanticResult43350

namespace SemanticResult43360
def owner : Owner := ⟨.program ⟨214⟩, ⟨25076⟩⟩
def rawTerms : List Term := Proof.Events169.exact43360RawTerms
def summary : Bound := (.finite 350206667259904)
def resultEvent : Nat := 43360
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43360.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95423744, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43356.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge43356.frameStart)
    (owner := owner) (leftOwner := SemanticResult43350.owner)
    (rightOwner := SemanticResult43286.owner)
    (leftResult := 43350) (rightResult := 43286)
    (leftActual := SemanticResult43350.actual selector witness)
    (rightActual := SemanticResult43286.actual selector witness)
    (leftRaw := SemanticResult43350.rawTerms)
    (rightRaw := SemanticResult43286.rawTerms)
    (working := LeftOperatorMerge43356.working)
    (leftBinding := 43351) (rightBinding := 43352)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11000⟩) (rightExpression := ⟨25075⟩)
    (coefficientTransfer := 43353) (summaryTransfer := 43355)
    (rightCoefficientProducer := 43285)
    (rightSummaryTransfer := 43354)
    (leftMaximum := ⟨95423744, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge43356.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority43285.actual selector witness)
    (summaryMagnitude := LeftBound43355.actual selector witness)
    (reconstruction := LeftOperatorMerge43356.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult43350.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult43286.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43285.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority43285.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge43356.operationAgreement
  · exact LeftBound43355.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge43356.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 43357 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6791⟩⟩, ⟨.program ⟨214⟩, ⟨7837⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23042⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23042⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge43356.working
    [{ coefficient := (-1), key := LeftRelationMerge43357.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge43357.frameStart
      LeftRelationMerge43357.owner (.relation 43357) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge43357.deltas
    rows := LeftRelationMerge43357.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge43356.working LeftRelationMerge43357.source
        (relationContext LeftRelationMerge43357.source
          LeftRelationMerge43357.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge43356.working, LeftRelationMerge43357.deltas,
    LeftRelationMerge43357.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 43357)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25076⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10852⟩⟩, ⟨.program ⟨214⟩, ⟨10993⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25075⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge43356.working) (working := relationWorking0)
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
end SemanticResult43360

namespace SemanticResult43363
def owner : Owner := ⟨.program ⟨214⟩, ⟨19176⟩⟩
def rawTerms : List Term := Proof.Events169.exact43363RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43363
def producerEvent : Nat := 43362
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43363.actual selector witness
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
end SemanticResult43363

namespace SemanticResult43367
def owner : Owner := ⟨.program ⟨214⟩, ⟨19178⟩⟩
def rawTerms : List Term := Proof.Events169.exact43367RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 43367
def producerEvent : Nat := 43366
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult43367.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 43364 .coefficient) (.value (.predecessor 1 43365 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 43364 .coefficient) (.value (.predecessor 1 43365 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult43367

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
