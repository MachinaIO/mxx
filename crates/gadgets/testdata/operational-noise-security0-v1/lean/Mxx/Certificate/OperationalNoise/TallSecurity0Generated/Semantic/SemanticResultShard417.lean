import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard417
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard113
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard114
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard365
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard416

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult57434
def owner : Owner := ⟨.program ⟨214⟩, ⟨11138⟩⟩
def rawTerms : List Term := Proof.Events224.exact57434RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 57434
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57434.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge57433.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge57433.frameStart)
    (transferEvent := 57432) (owner := owner)
    (leftResult := 2660) (rightResult := 50670)
    (working := LeftOperatorMerge57433.working)
    (reconstruction := LeftOperatorMerge57433.reconstruction)
    (leftReference := .predecessor 0 57430 .coefficient) (rightReference := .predecessor 1 57431 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2660.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge57433.operationAgreement
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
end SemanticResult57434

namespace SemanticResult57439
def owner : Owner := ⟨.program ⟨214⟩, ⟨7269⟩⟩
def rawTerms : List Term := Proof.Events224.exact57439RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 57439
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57439.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge57438.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge57438.frameStart)
    (transferEvent := 57437) (owner := owner)
    (leftResult := 50540) (rightResult := 13486)
    (working := LeftOperatorMerge57438.working)
    (reconstruction := LeftOperatorMerge57438.reconstruction)
    (leftReference := .predecessor 0 57435 .coefficient) (rightReference := .predecessor 1 57436 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13486.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge57438.operationAgreement
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
end SemanticResult57439

namespace SemanticResult57443
def owner : Owner := ⟨.program ⟨214⟩, ⟨11139⟩⟩
def rawTerms : List Term := Proof.Events224.exact57443RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 57443
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57443.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 57440) (rightBinding := 57441)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7269⟩) (rightExpression := ⟨11138⟩)
    (transferEvent := 57442)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult57439.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult57434.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult57443

namespace SemanticResult57449
def owner : Owner := ⟨.program ⟨214⟩, ⟨11140⟩⟩
def rawTerms : List Term := Proof.Events224.exact57449RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 57449
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57449.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 57446) (survivorTransfer := 57447)
    (survivorEvent := 57448) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13477)
    (owner := owner) (leftOwner := SemanticResult57443.owner)
    (rightOwner := SemanticResult13478.owner)
    (leftResult := 57443) (rightResult := 13478)
    (leftBinding := 57444) (rightBinding := 57445)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11139⟩) (rightExpression := ⟨89⟩)
    (leftActual := SemanticResult57443.actual selector witness)
    (rightActual := SemanticResult13478.actual selector witness)
    (leftRaw := SemanticResult57443.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨89⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13477.actual selector witness)
    (survivorMagnitude := LeftBound57447.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult57443.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13478.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13477.derived selector witness)
  · exact LeftBound57447.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult57449

namespace SemanticResult57457
def owner : Owner := ⟨.program ⟨214⟩, ⟨12175⟩⟩
def rawTerms : List Term := Proof.Events224.exact57457RawTerms
def summary : Bound := (.finite 4992)
def resultEvent : Nat := 57457
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57457.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨6, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge57455.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge57455.frameStart)
    (owner := owner) (leftOwner := SemanticResult57449.owner)
    (rightOwner := SemanticResult2663.owner)
    (leftResult := 57449) (rightResult := 2663)
    (leftActual := SemanticResult57449.actual selector witness)
    (rightActual := SemanticResult2663.actual selector witness)
    (leftRaw := SemanticResult57449.rawTerms)
    (rightRaw := SemanticResult2663.rawTerms)
    (working := LeftOperatorMerge57455.working)
    (leftBinding := 57450) (rightBinding := 57451)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11140⟩) (rightExpression := ⟨12172⟩)
    (coefficientTransfer := 57452) (summaryTransfer := 57454)
    (rightCoefficientProducer := 2662)
    (rightSummaryTransfer := 57453)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨6, by decide⟩)
    (rightRecordedMaximum := 6)
    (rightSummaryMaximum := ⟨6, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge57455.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority2662.actual selector witness)
    (summaryMagnitude := LeftBound57454.actual selector witness)
    (reconstruction := LeftOperatorMerge57455.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult57449.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult2663.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2662.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority2662.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge57455.operationAgreement
  · exact LeftBound57454.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge57455.working summary) := by
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
end SemanticResult57457

namespace SemanticResult57462
def owner : Owner := ⟨.program ⟨214⟩, ⟨12176⟩⟩
def rawTerms : List Term := Proof.Events224.exact57462RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 57462
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57462.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge57461.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge57461.frameStart)
    (transferEvent := 57460) (owner := owner)
    (leftResult := 2663) (rightResult := 50670)
    (working := LeftOperatorMerge57461.working)
    (reconstruction := LeftOperatorMerge57461.reconstruction)
    (leftReference := .predecessor 0 57458 .coefficient) (rightReference := .predecessor 1 57459 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult2663.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50670.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge57461.operationAgreement
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
end SemanticResult57462

namespace SemanticResult57467
def owner : Owner := ⟨.program ⟨214⟩, ⟨7286⟩⟩
def rawTerms : List Term := Proof.Events224.exact57467RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 57467
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57467.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge57466.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge57466.frameStart)
    (transferEvent := 57465) (owner := owner)
    (leftResult := 50540) (rightResult := 13527)
    (working := LeftOperatorMerge57466.working)
    (reconstruction := LeftOperatorMerge57466.reconstruction)
    (leftReference := .predecessor 0 57463 .coefficient) (rightReference := .predecessor 1 57464 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult50540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13527.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge57466.operationAgreement
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
end SemanticResult57467

namespace SemanticResult57471
def owner : Owner := ⟨.program ⟨214⟩, ⟨12177⟩⟩
def rawTerms : List Term := Proof.Events224.exact57471RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 57471
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57471.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 57468) (rightBinding := 57469)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7286⟩) (rightExpression := ⟨12176⟩)
    (transferEvent := 57470)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult57467.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult57462.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult57471

namespace SemanticResult57477
def owner : Owner := ⟨.program ⟨214⟩, ⟨12178⟩⟩
def rawTerms : List Term := Proof.Events224.exact57477RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 57477
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57477.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 57474) (survivorTransfer := 57475)
    (survivorEvent := 57476) (resultEvent := resultEvent)
    (rightCoefficientProducer := 13518)
    (owner := owner) (leftOwner := SemanticResult57471.owner)
    (rightOwner := SemanticResult13519.owner)
    (leftResult := 57471) (rightResult := 13519)
    (leftBinding := 57472) (rightBinding := 57473)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12177⟩) (rightExpression := ⟨106⟩)
    (leftActual := SemanticResult57471.actual selector witness)
    (rightActual := SemanticResult13519.actual selector witness)
    (leftRaw := SemanticResult57471.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨106⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound13518.actual selector witness)
    (survivorMagnitude := LeftBound57475.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult57471.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13519.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13518.derived selector witness)
  · exact LeftBound57475.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult57477

namespace SemanticResult57487
def owner : Owner := ⟨.program ⟨214⟩, ⟨12179⟩⟩
def rawTerms : List Term := Proof.Events224.exact57487RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 57487
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57487.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge57483.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge57483.frameStart)
    (owner := owner) (leftOwner := SemanticResult57477.owner)
    (rightOwner := SemanticResult13516.owner)
    (leftResult := 57477) (rightResult := 13516)
    (leftActual := SemanticResult57477.actual selector witness)
    (rightActual := SemanticResult13516.actual selector witness)
    (leftRaw := SemanticResult57477.rawTerms)
    (rightRaw := SemanticResult13516.rawTerms)
    (working := LeftOperatorMerge57483.working)
    (leftBinding := 57478) (rightBinding := 57479)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12178⟩) (rightExpression := ⟨7841⟩)
    (coefficientTransfer := 57480) (summaryTransfer := 57482)
    (rightCoefficientProducer := 13515)
    (rightSummaryTransfer := 57481)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge57483.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound13515.actual selector witness)
    (summaryMagnitude := LeftBound57482.actual selector witness)
    (reconstruction := LeftOperatorMerge57483.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult57477.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult13516.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13515.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound13515.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge57483.operationAgreement
  · exact LeftBound57482.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge57483.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 57484 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge57483.working
    [{ coefficient := (-1), key := LeftRelationMerge57484.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge57484.frameStart
      LeftRelationMerge57484.owner (.relation 57484) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge57484.deltas
    rows := LeftRelationMerge57484.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge57483.working LeftRelationMerge57484.source
        (relationContext LeftRelationMerge57484.source
          LeftRelationMerge57484.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge57483.working, LeftRelationMerge57484.deltas,
    LeftRelationMerge57484.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 57484)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨12179⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge57483.working) (working := relationWorking0)
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
end SemanticResult57487

namespace SemanticResult57493
def owner : Owner := ⟨.program ⟨214⟩, ⟨12180⟩⟩
def rawTerms : List Term := Proof.Events224.exact57493RawTerms
def summary : Bound := (.finite 95425408)
def resultEvent : Nat := 57493
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57493.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge57491.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult57487.owner)
    (rightOwner := SemanticResult57457.owner)
    (leftResult := 57487) (rightResult := 57457)
    (leftActual := SemanticResult57487.actual selector witness)
    (rightActual := SemanticResult57457.actual selector witness)
    (leftRaw := SemanticResult57487.rawTerms)
    (rightRaw := SemanticResult57457.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 4992) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 57488) (rightBinding := 57489)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12179⟩) (rightExpression := ⟨12175⟩)
    (coefficientTransfer := 57490) (summaryTransfer := 57492)
    (base := LeftOperatorMerge57491.base)
    (reconstruction := LeftOperatorMerge57491.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult57487.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult57457.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge57491.operationAgreement
  · rfl
  · decide
end SemanticResult57493

namespace SemanticResult57503
def owner : Owner := ⟨.program ⟨214⟩, ⟨25302⟩⟩
def rawTerms : List Term := Proof.Events224.exact57503RawTerms
def summary : Bound := (.finite 350212774166528)
def resultEvent : Nat := 57503
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57503.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95425408, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge57499.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge57499.frameStart)
    (owner := owner) (leftOwner := SemanticResult57493.owner)
    (rightOwner := SemanticResult57429.owner)
    (leftResult := 57493) (rightResult := 57429)
    (leftActual := SemanticResult57493.actual selector witness)
    (rightActual := SemanticResult57429.actual selector witness)
    (leftRaw := SemanticResult57493.rawTerms)
    (rightRaw := SemanticResult57429.rawTerms)
    (working := LeftOperatorMerge57499.working)
    (leftBinding := 57494) (rightBinding := 57495)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12180⟩) (rightExpression := ⟨25301⟩)
    (coefficientTransfer := 57496) (summaryTransfer := 57498)
    (rightCoefficientProducer := 57428)
    (rightSummaryTransfer := 57497)
    (leftMaximum := ⟨95425408, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge57499.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority57428.actual selector witness)
    (summaryMagnitude := LeftBound57498.actual selector witness)
    (reconstruction := LeftOperatorMerge57499.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult57493.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult57429.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57428.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority57428.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge57499.operationAgreement
  · exact LeftBound57498.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge57499.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 57500 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23166⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23166⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge57499.working
    [{ coefficient := (-1), key := LeftRelationMerge57500.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge57500.frameStart
      LeftRelationMerge57500.owner (.relation 57500) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge57500.deltas
    rows := LeftRelationMerge57500.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge57499.working LeftRelationMerge57500.source
        (relationContext LeftRelationMerge57500.source
          LeftRelationMerge57500.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge57499.working, LeftRelationMerge57500.deltas,
    LeftRelationMerge57500.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 57500)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25302⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge57499.working) (working := relationWorking0)
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
end SemanticResult57503

namespace SemanticResult57506
def owner : Owner := ⟨.program ⟨214⟩, ⟨19244⟩⟩
def rawTerms : List Term := Proof.Events224.exact57506RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 57506
def producerEvent : Nat := 57505
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57506.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨10⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨10⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult57506

namespace SemanticResult57510
def owner : Owner := ⟨.program ⟨214⟩, ⟨19246⟩⟩
def rawTerms : List Term := Proof.Events224.exact57510RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 57510
def producerEvent : Nat := 57509
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57510.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 57507 .coefficient) (.value (.predecessor 1 57508 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 57507 .coefficient) (.value (.predecessor 1 57508 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult57510

namespace SemanticResult57588
def owner : Owner := ⟨.program ⟨214⟩, ⟨11137⟩⟩
def rawTerms : List Term := Proof.Events224.exact57588RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 57588
def producerEvent : Nat := 57587
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57588.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 57565, .finite 6, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult57588

namespace SemanticResult57591
def owner : Owner := ⟨.program ⟨214⟩, ⟨12172⟩⟩
def rawTerms : List Term := Proof.Events224.exact57591RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 57591
def producerEvent : Nat := 57590
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult57591.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 57565, .finite 6, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult57591

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
