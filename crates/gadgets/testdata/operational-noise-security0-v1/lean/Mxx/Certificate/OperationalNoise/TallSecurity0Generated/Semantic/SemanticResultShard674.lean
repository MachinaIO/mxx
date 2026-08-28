import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard674
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard037
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard065
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard673

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult95263
def owner : Owner := ⟨.program ⟨214⟩, ⟨12938⟩⟩
def rawTerms : List Term := Proof.Events372.exact95263RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95263
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95263.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 95260) (rightBinding := 95261)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7125⟩) (rightExpression := ⟨12937⟩)
    (transferEvent := 95262)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95259.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult95254.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult95263

namespace SemanticResult95269
def owner : Owner := ⟨.program ⟨214⟩, ⟨12939⟩⟩
def rawTerms : List Term := Proof.Events372.exact95269RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 95269
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95269.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 95266) (survivorTransfer := 95267)
    (survivorEvent := 95268) (resultEvent := resultEvent)
    (rightCoefficientProducer := 7465)
    (owner := owner) (leftOwner := SemanticResult95263.owner)
    (rightOwner := SemanticResult7466.owner)
    (leftResult := 95263) (rightResult := 7466)
    (leftBinding := 95264) (rightBinding := 95265)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12938⟩) (rightExpression := ⟨102⟩)
    (leftActual := SemanticResult95263.actual selector witness)
    (rightActual := SemanticResult7466.actual selector witness)
    (leftRaw := SemanticResult95263.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨102⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound7465.actual selector witness)
    (survivorMagnitude := LeftBound95267.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95263.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7466.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7465.derived selector witness)
  · exact LeftBound95267.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult95269

namespace SemanticResult95277
def owner : Owner := ⟨.program ⟨214⟩, ⟨12940⟩⟩
def rawTerms : List Term := Proof.Events372.exact95277RawTerms
def summary : Bound := (.finite 43264)
def resultEvent : Nat := 95277
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95277.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨52, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95275.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge95275.frameStart)
    (owner := owner) (leftOwner := SemanticResult95269.owner)
    (rightOwner := SemanticResult4615.owner)
    (leftResult := 95269) (rightResult := 4615)
    (leftActual := SemanticResult95269.actual selector witness)
    (rightActual := SemanticResult4615.actual selector witness)
    (leftRaw := SemanticResult95269.rawTerms)
    (rightRaw := SemanticResult4615.rawTerms)
    (working := LeftOperatorMerge95275.working)
    (leftBinding := 95270) (rightBinding := 95271)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12939⟩) (rightExpression := ⟨10120⟩)
    (coefficientTransfer := 95272) (summaryTransfer := 95274)
    (rightCoefficientProducer := 4614)
    (rightSummaryTransfer := 95273)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨52, by decide⟩)
    (rightRecordedMaximum := 52)
    (rightSummaryMaximum := ⟨52, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge95275.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority4614.actual selector witness)
    (summaryMagnitude := LeftBound95274.actual selector witness)
    (reconstruction := LeftOperatorMerge95275.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95269.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4615.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4614.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority4614.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge95275.operationAgreement
  · exact LeftBound95274.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95275.working summary) := by
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
end SemanticResult95277

namespace SemanticResult95282
def owner : Owner := ⟨.program ⟨214⟩, ⟨10121⟩⟩
def rawTerms : List Term := Proof.Events372.exact95282RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95282
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95282.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95281.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge95281.frameStart)
    (transferEvent := 95280) (owner := owner)
    (leftResult := 4615) (rightResult := 32)
    (working := LeftOperatorMerge95281.working)
    (reconstruction := LeftOperatorMerge95281.reconstruction)
    (leftReference := .predecessor 0 95278 .coefficient) (rightReference := .predecessor 1 95279 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult4615.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult32.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge95281.operationAgreement
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
end SemanticResult95282

namespace SemanticResult95287
def owner : Owner := ⟨.program ⟨214⟩, ⟨7105⟩⟩
def rawTerms : List Term := Proof.Events372.exact95287RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95287
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95287.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95286.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge95286.frameStart)
    (transferEvent := 95285) (owner := owner)
    (leftResult := 27) (rightResult := 7515)
    (working := LeftOperatorMerge95286.working)
    (reconstruction := LeftOperatorMerge95286.reconstruction)
    (leftReference := .predecessor 0 95283 .coefficient) (rightReference := .predecessor 1 95284 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult27.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7515.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge95286.operationAgreement
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
end SemanticResult95287

namespace SemanticResult95291
def owner : Owner := ⟨.program ⟨214⟩, ⟨10122⟩⟩
def rawTerms : List Term := Proof.Events372.exact95291RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95291
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95291.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 95288) (rightBinding := 95289)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7105⟩) (rightExpression := ⟨10121⟩)
    (transferEvent := 95290)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95287.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult95282.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult95291

namespace SemanticResult95297
def owner : Owner := ⟨.program ⟨214⟩, ⟨10123⟩⟩
def rawTerms : List Term := Proof.Events372.exact95297RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 95297
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95297.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 95294) (survivorTransfer := 95295)
    (survivorEvent := 95296) (resultEvent := resultEvent)
    (rightCoefficientProducer := 7506)
    (owner := owner) (leftOwner := SemanticResult95291.owner)
    (rightOwner := SemanticResult7507.owner)
    (leftResult := 95291) (rightResult := 7507)
    (leftBinding := 95292) (rightBinding := 95293)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10122⟩) (rightExpression := ⟨82⟩)
    (leftActual := SemanticResult95291.actual selector witness)
    (rightActual := SemanticResult7507.actual selector witness)
    (leftRaw := SemanticResult95291.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨82⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound7506.actual selector witness)
    (survivorMagnitude := LeftBound95295.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95291.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7507.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7506.derived selector witness)
  · exact LeftBound95295.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult95297

namespace SemanticResult95307
def owner : Owner := ⟨.program ⟨214⟩, ⟨10124⟩⟩
def rawTerms : List Term := Proof.Events372.exact95307RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 95307
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95307.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95303.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge95303.frameStart)
    (owner := owner) (leftOwner := SemanticResult95297.owner)
    (rightOwner := SemanticResult7504.owner)
    (leftResult := 95297) (rightResult := 7504)
    (leftActual := SemanticResult95297.actual selector witness)
    (rightActual := SemanticResult7504.actual selector witness)
    (leftRaw := SemanticResult95297.rawTerms)
    (rightRaw := SemanticResult7504.rawTerms)
    (working := LeftOperatorMerge95303.working)
    (leftBinding := 95298) (rightBinding := 95299)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10123⟩) (rightExpression := ⟨7877⟩)
    (coefficientTransfer := 95300) (summaryTransfer := 95302)
    (rightCoefficientProducer := 7503)
    (rightSummaryTransfer := 95301)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge95303.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound7503.actual selector witness)
    (summaryMagnitude := LeftBound95302.actual selector witness)
    (reconstruction := LeftOperatorMerge95303.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95297.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7504.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7503.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound7503.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge95303.operationAgreement
  · exact LeftBound95302.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95303.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 95304 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10120⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge95303.working
    [{ coefficient := (-1), key := LeftRelationMerge95304.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge95304.frameStart
      LeftRelationMerge95304.owner (.relation 95304) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge95304.deltas
    rows := LeftRelationMerge95304.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge95303.working LeftRelationMerge95304.source
        (relationContext LeftRelationMerge95304.source
          LeftRelationMerge95304.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge95303.working, LeftRelationMerge95304.deltas,
    LeftRelationMerge95304.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 95304)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨10124⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge95303.working) (working := relationWorking0)
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
end SemanticResult95307

namespace SemanticResult95313
def owner : Owner := ⟨.program ⟨214⟩, ⟨12941⟩⟩
def rawTerms : List Term := Proof.Events372.exact95313RawTerms
def summary : Bound := (.finite 95463680)
def resultEvent : Nat := 95313
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95313.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge95311.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult95307.owner)
    (rightOwner := SemanticResult95277.owner)
    (leftResult := 95307) (rightResult := 95277)
    (leftActual := SemanticResult95307.actual selector witness)
    (rightActual := SemanticResult95277.actual selector witness)
    (leftRaw := SemanticResult95307.rawTerms)
    (rightRaw := SemanticResult95277.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 43264) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 95308) (rightBinding := 95309)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10124⟩) (rightExpression := ⟨12940⟩)
    (coefficientTransfer := 95310) (summaryTransfer := 95312)
    (base := LeftOperatorMerge95311.base)
    (reconstruction := LeftOperatorMerge95311.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95307.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult95277.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge95311.operationAgreement
  · rfl
  · decide
end SemanticResult95313

namespace SemanticResult95323
def owner : Owner := ⟨.program ⟨214⟩, ⟨25592⟩⟩
def rawTerms : List Term := Proof.Events372.exact95323RawTerms
def summary : Bound := (.finite 350353233018880)
def resultEvent : Nat := 95323
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95323.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95463680, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95319.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge95319.frameStart)
    (owner := owner) (leftOwner := SemanticResult95313.owner)
    (rightOwner := SemanticResult95249.owner)
    (leftResult := 95313) (rightResult := 95249)
    (leftActual := SemanticResult95313.actual selector witness)
    (rightActual := SemanticResult95249.actual selector witness)
    (leftRaw := SemanticResult95313.rawTerms)
    (rightRaw := SemanticResult95249.rawTerms)
    (working := LeftOperatorMerge95319.working)
    (leftBinding := 95314) (rightBinding := 95315)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨12941⟩) (rightExpression := ⟨25591⟩)
    (coefficientTransfer := 95316) (summaryTransfer := 95318)
    (rightCoefficientProducer := 95248)
    (rightSummaryTransfer := 95317)
    (leftMaximum := ⟨95463680, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge95319.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority95248.actual selector witness)
    (summaryMagnitude := LeftBound95318.actual selector witness)
    (reconstruction := LeftOperatorMerge95319.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult95313.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult95249.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95248.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority95248.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge95319.operationAgreement
  · exact LeftBound95318.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95319.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 95320 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23326⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23326⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge95319.working
    [{ coefficient := (-1), key := LeftRelationMerge95320.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge95320.frameStart
      LeftRelationMerge95320.owner (.relation 95320) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge95320.deltas
    rows := LeftRelationMerge95320.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge95319.working LeftRelationMerge95320.source
        (relationContext LeftRelationMerge95320.source
          LeftRelationMerge95320.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge95319.working, LeftRelationMerge95320.deltas,
    LeftRelationMerge95320.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 95320)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨25592⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10120⟩⟩, ⟨.program ⟨214⟩, ⟨12934⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25591⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge95319.working) (working := relationWorking0)
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
end SemanticResult95323

namespace SemanticResult95326
def owner : Owner := ⟨.program ⟨214⟩, ⟨20093⟩⟩
def rawTerms : List Term := Proof.Events372.exact95326RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95326
def producerEvent : Nat := 95325
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95326.actual selector witness
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
end SemanticResult95326

namespace SemanticResult95330
def owner : Owner := ⟨.program ⟨214⟩, ⟨20095⟩⟩
def rawTerms : List Term := Proof.Events372.exact95330RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95330
def producerEvent : Nat := 95329
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95330.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 95327 .coefficient) (.value (.predecessor 1 95328 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 95327 .coefficient) (.value (.predecessor 1 95328 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult95330

namespace SemanticResult95384
def owner : Owner := ⟨.program ⟨214⟩, ⟨12934⟩⟩
def rawTerms : List Term := Proof.Events372.exact95384RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95384
def producerEvent : Nat := 95383
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95384.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 95373, .finite 52, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult95384

namespace SemanticResult95387
def owner : Owner := ⟨.program ⟨214⟩, ⟨10120⟩⟩
def rawTerms : List Term := Proof.Events372.exact95387RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95387
def producerEvent : Nat := 95386
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95387.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 95373, .finite 52, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult95387

namespace SemanticResult95392
def owner : Owner := ⟨.program ⟨214⟩, ⟨12935⟩⟩
def rawTerms : List Term := Proof.Events372.exact95392RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95392
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95392.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge95391.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge95391.frameStart)
    (transferEvent := 95390) (owner := owner)
    (leftResult := 95387) (rightResult := 95384)
    (working := LeftOperatorMerge95391.working)
    (reconstruction := LeftOperatorMerge95391.reconstruction)
    (leftReference := .predecessor 0 95388 .coefficient) (rightReference := .predecessor 1 95389 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult95387.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult95384.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge95391.operationAgreement
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
end SemanticResult95392

namespace SemanticResult95403
def owner : Owner := ⟨.program ⟨214⟩, ⟨23326⟩⟩
def rawTerms : List Term := Proof.Events372.exact95403RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 95403
def producerEvent : Nat := 95402
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult95403.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 95373, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult95403

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
