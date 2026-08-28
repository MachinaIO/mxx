import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard276
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard013
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard069
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard275

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult37516
def owner : Owner := ⟨.program ⟨214⟩, ⟨12782⟩⟩
def rawTerms : List Term := Proof.Events146.exact37516RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37516
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37516.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 37513) (rightBinding := 37514)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7319⟩) (rightExpression := ⟨12781⟩)
    (transferEvent := 37515)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37512.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult37507.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult37516

namespace SemanticResult37522
def owner : Owner := ⟨.program ⟨214⟩, ⟨12783⟩⟩
def rawTerms : List Term := Proof.Events146.exact37522RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 37522
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37522.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 37519) (survivorTransfer := 37520)
    (survivorEvent := 37521) (resultEvent := resultEvent)
    (rightCoefficientProducer := 7966)
    (owner := owner) (leftOwner := SemanticResult37516.owner)
    (rightOwner := SemanticResult7967.owner)
    (leftResult := 37516) (rightResult := 7967)
    (leftBinding := 37517) (rightBinding := 37518)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12782⟩) (rightExpression := ⟨101⟩)
    (leftActual := SemanticResult37516.actual selector witness)
    (rightActual := SemanticResult7967.actual selector witness)
    (leftRaw := SemanticResult37516.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨101⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound7966.actual selector witness)
    (survivorMagnitude := LeftBound37520.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37516.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7967.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7966.derived selector witness)
  · exact LeftBound37520.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult37522

namespace SemanticResult37530
def owner : Owner := ⟨.program ⟨214⟩, ⟨12784⟩⟩
def rawTerms : List Term := Proof.Events146.exact37530RawTerms
def summary : Bound := (.finite 38272)
def resultEvent : Nat := 37530
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37530.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨46, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37528.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge37528.frameStart)
    (owner := owner) (leftOwner := SemanticResult37522.owner)
    (rightOwner := SemanticResult1662.owner)
    (leftResult := 37522) (rightResult := 1662)
    (leftActual := SemanticResult37522.actual selector witness)
    (rightActual := SemanticResult1662.actual selector witness)
    (leftRaw := SemanticResult37522.rawTerms)
    (rightRaw := SemanticResult1662.rawTerms)
    (working := LeftOperatorMerge37528.working)
    (leftBinding := 37523) (rightBinding := 37524)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12783⟩) (rightExpression := ⟨10040⟩)
    (coefficientTransfer := 37525) (summaryTransfer := 37527)
    (rightCoefficientProducer := 1661)
    (rightSummaryTransfer := 37526)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨46, by decide⟩)
    (rightRecordedMaximum := 46)
    (rightSummaryMaximum := ⟨46, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge37528.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1661.actual selector witness)
    (summaryMagnitude := LeftBound37527.actual selector witness)
    (reconstruction := LeftOperatorMerge37528.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37522.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1662.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1661.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1661.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge37528.operationAgreement
  · exact LeftBound37527.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37528.working summary) := by
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
end SemanticResult37530

namespace SemanticResult37535
def owner : Owner := ⟨.program ⟨214⟩, ⟨10041⟩⟩
def rawTerms : List Term := Proof.Events146.exact37535RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37535
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37535.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37534.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge37534.frameStart)
    (transferEvent := 37533) (owner := owner)
    (leftResult := 1662) (rightResult := 36045)
    (working := LeftOperatorMerge37534.working)
    (reconstruction := LeftOperatorMerge37534.reconstruction)
    (leftReference := .predecessor 0 37531 .coefficient) (rightReference := .predecessor 1 37532 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1662.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge37534.operationAgreement
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
end SemanticResult37535

namespace SemanticResult37540
def owner : Owner := ⟨.program ⟨214⟩, ⟨7299⟩⟩
def rawTerms : List Term := Proof.Events146.exact37540RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37540
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37540.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37539.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge37539.frameStart)
    (transferEvent := 37538) (owner := owner)
    (leftResult := 35915) (rightResult := 8016)
    (working := LeftOperatorMerge37539.working)
    (reconstruction := LeftOperatorMerge37539.reconstruction)
    (leftReference := .predecessor 0 37536 .coefficient) (rightReference := .predecessor 1 37537 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8016.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge37539.operationAgreement
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
end SemanticResult37540

namespace SemanticResult37544
def owner : Owner := ⟨.program ⟨214⟩, ⟨10042⟩⟩
def rawTerms : List Term := Proof.Events146.exact37544RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37544
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37544.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 37541) (rightBinding := 37542)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7299⟩) (rightExpression := ⟨10041⟩)
    (transferEvent := 37543)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37540.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult37535.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult37544

namespace SemanticResult37550
def owner : Owner := ⟨.program ⟨214⟩, ⟨10043⟩⟩
def rawTerms : List Term := Proof.Events146.exact37550RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 37550
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37550.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 37547) (survivorTransfer := 37548)
    (survivorEvent := 37549) (resultEvent := resultEvent)
    (rightCoefficientProducer := 8007)
    (owner := owner) (leftOwner := SemanticResult37544.owner)
    (rightOwner := SemanticResult8008.owner)
    (leftResult := 37544) (rightResult := 8008)
    (leftBinding := 37545) (rightBinding := 37546)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10042⟩) (rightExpression := ⟨81⟩)
    (leftActual := SemanticResult37544.actual selector witness)
    (rightActual := SemanticResult8008.actual selector witness)
    (leftRaw := SemanticResult37544.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨81⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound8007.actual selector witness)
    (survivorMagnitude := LeftBound37548.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37544.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8008.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8007.derived selector witness)
  · exact LeftBound37548.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult37550

namespace SemanticResult37560
def owner : Owner := ⟨.program ⟨214⟩, ⟨10044⟩⟩
def rawTerms : List Term := Proof.Events146.exact37560RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 37560
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37560.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37556.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge37556.frameStart)
    (owner := owner) (leftOwner := SemanticResult37550.owner)
    (rightOwner := SemanticResult8005.owner)
    (leftResult := 37550) (rightResult := 8005)
    (leftActual := SemanticResult37550.actual selector witness)
    (rightActual := SemanticResult8005.actual selector witness)
    (leftRaw := SemanticResult37550.rawTerms)
    (rightRaw := SemanticResult8005.rawTerms)
    (working := LeftOperatorMerge37556.working)
    (leftBinding := 37551) (rightBinding := 37552)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10043⟩) (rightExpression := ⟨7874⟩)
    (coefficientTransfer := 37553) (summaryTransfer := 37555)
    (rightCoefficientProducer := 8004)
    (rightSummaryTransfer := 37554)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge37556.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound8004.actual selector witness)
    (summaryMagnitude := LeftBound37555.actual selector witness)
    (reconstruction := LeftOperatorMerge37556.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37550.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult8005.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8004.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound8004.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge37556.operationAgreement
  · exact LeftBound37555.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37556.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 37557 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10040⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge37556.working
    [{ coefficient := (-1), key := LeftRelationMerge37557.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge37557.frameStart
      LeftRelationMerge37557.owner (.relation 37557) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge37557.deltas
    rows := LeftRelationMerge37557.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge37556.working LeftRelationMerge37557.source
        (relationContext LeftRelationMerge37557.source
          LeftRelationMerge37557.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge37556.working, LeftRelationMerge37557.deltas,
    LeftRelationMerge37557.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 37557)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10044⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10040⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge37556.working) (working := relationWorking0)
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
end SemanticResult37560

namespace SemanticResult37566
def owner : Owner := ⟨.program ⟨214⟩, ⟨12785⟩⟩
def rawTerms : List Term := Proof.Events146.exact37566RawTerms
def summary : Bound := (.finite 95458688)
def resultEvent : Nat := 37566
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37566.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge37564.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult37560.owner)
    (rightOwner := SemanticResult37530.owner)
    (leftResult := 37560) (rightResult := 37530)
    (leftActual := SemanticResult37560.actual selector witness)
    (rightActual := SemanticResult37530.actual selector witness)
    (leftRaw := SemanticResult37560.rawTerms)
    (rightRaw := SemanticResult37530.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 38272) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 37561) (rightBinding := 37562)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10044⟩) (rightExpression := ⟨12784⟩)
    (coefficientTransfer := 37563) (summaryTransfer := 37565)
    (base := LeftOperatorMerge37564.base)
    (reconstruction := LeftOperatorMerge37564.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37560.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult37530.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge37564.operationAgreement
  · rfl
  · decide
end SemanticResult37566

namespace SemanticResult37576
def owner : Owner := ⟨.program ⟨214⟩, ⟨25538⟩⟩
def rawTerms : List Term := Proof.Events146.exact37576RawTerms
def summary : Bound := (.finite 350334912299008)
def resultEvent : Nat := 37576
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37576.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95458688, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37572.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge37572.frameStart)
    (owner := owner) (leftOwner := SemanticResult37566.owner)
    (rightOwner := SemanticResult37502.owner)
    (leftResult := 37566) (rightResult := 37502)
    (leftActual := SemanticResult37566.actual selector witness)
    (rightActual := SemanticResult37502.actual selector witness)
    (leftRaw := SemanticResult37566.rawTerms)
    (rightRaw := SemanticResult37502.rawTerms)
    (working := LeftOperatorMerge37572.working)
    (leftBinding := 37567) (rightBinding := 37568)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12785⟩) (rightExpression := ⟨25537⟩)
    (coefficientTransfer := 37569) (summaryTransfer := 37571)
    (rightCoefficientProducer := 37501)
    (rightSummaryTransfer := 37570)
    (leftMaximum := ⟨95458688, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge37572.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority37501.actual selector witness)
    (summaryMagnitude := LeftBound37571.actual selector witness)
    (reconstruction := LeftOperatorMerge37572.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult37566.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult37502.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37501.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority37501.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge37572.operationAgreement
  · exact LeftBound37571.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37572.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 37573 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩, ⟨.program ⟨214⟩, ⟨25537⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10040⟩⟩, ⟨.program ⟨214⟩, ⟨12778⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23294⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23294⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge37572.working
    [{ coefficient := (-1), key := LeftRelationMerge37573.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge37573.frameStart
      LeftRelationMerge37573.owner (.relation 37573) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge37573.deltas
    rows := LeftRelationMerge37573.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge37572.working LeftRelationMerge37573.source
        (relationContext LeftRelationMerge37573.source
          LeftRelationMerge37573.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge37572.working, LeftRelationMerge37573.deltas,
    LeftRelationMerge37573.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 37573)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25538⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10040⟩⟩, ⟨.program ⟨214⟩, ⟨12778⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25537⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25537⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge37572.working) (working := relationWorking0)
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
end SemanticResult37576

namespace SemanticResult37579
def owner : Owner := ⟨.program ⟨214⟩, ⟨20040⟩⟩
def rawTerms : List Term := Proof.Events146.exact37579RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37579
def producerEvent : Nat := 37578
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37579.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨23⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨23⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult37579

namespace SemanticResult37583
def owner : Owner := ⟨.program ⟨214⟩, ⟨20042⟩⟩
def rawTerms : List Term := Proof.Events146.exact37583RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37583
def producerEvent : Nat := 37582
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37583.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 37580 .coefficient) (.value (.predecessor 1 37581 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 37580 .coefficient) (.value (.predecessor 1 37581 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult37583

namespace SemanticResult37661
def owner : Owner := ⟨.program ⟨214⟩, ⟨12778⟩⟩
def rawTerms : List Term := Proof.Events147.exact37661RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37661
def producerEvent : Nat := 37660
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37661.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 37638, .finite 46, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult37661

namespace SemanticResult37664
def owner : Owner := ⟨.program ⟨214⟩, ⟨10040⟩⟩
def rawTerms : List Term := Proof.Events147.exact37664RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37664
def producerEvent : Nat := 37663
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37664.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 37638, .finite 46, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult37664

namespace SemanticResult37669
def owner : Owner := ⟨.program ⟨214⟩, ⟨12779⟩⟩
def rawTerms : List Term := Proof.Events147.exact37669RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37669
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37669.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge37668.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge37668.frameStart)
    (transferEvent := 37667) (owner := owner)
    (leftResult := 37664) (rightResult := 37661)
    (working := LeftOperatorMerge37668.working)
    (reconstruction := LeftOperatorMerge37668.reconstruction)
    (leftReference := .predecessor 0 37665 .coefficient) (rightReference := .predecessor 1 37666 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult37664.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult37661.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge37668.operationAgreement
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
end SemanticResult37669

namespace SemanticResult37680
def owner : Owner := ⟨.program ⟨214⟩, ⟨23294⟩⟩
def rawTerms : List Term := Proof.Events147.exact37680RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 37680
def producerEvent : Nat := 37679
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult37680.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 37638, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult37680

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
